"""プロバイダ資格情報の暗号化保存。

Fernet (cryptography) で API キーを対称暗号化して SQLite に保存する。
鍵は settings.secret_key から HKDF で導出 (専用 KMS 不要)。

SQLite テーブル `provider_credentials`:
- provider_name: TEXT PRIMARY KEY (例: "anthropic")
- api_key_enc:   BLOB (Fernet token bytes)
- extra:         TEXT (JSON, モデル名/エンドポイント/region 等)
- updated_at:    TEXT ISO8601
"""
from __future__ import annotations

import base64
import hashlib
import json

from cryptography.fernet import Fernet

from app.config import settings
from app.database import db
from app.services.providers import get_meta, reset_instances


def _derive_key() -> bytes:
    """settings.secret_key から 32 バイト鍵を導出 (Fernet は base64url 必須)。"""
    digest = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet() -> Fernet:
    return Fernet(_derive_key())


async def save_credentials(provider_name: str, api_key: str, extra: dict | None = None) -> None:
    """資格情報を暗号化して upsert。既存インスタンスは破棄して次回再生成。"""
    # メタ存在チェック (未知のプロバイダは登録不可)
    get_meta(provider_name)

    api_key_enc = _fernet().encrypt(api_key.encode("utf-8"))
    extra_json = json.dumps(extra or {}, ensure_ascii=False)
    await db.execute(
        "INSERT INTO provider_credentials (provider_name, api_key_enc, extra, updated_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(provider_name) DO UPDATE SET "
        "api_key_enc=excluded.api_key_enc, extra=excluded.extra, updated_at=excluded.updated_at",
        (provider_name, api_key_enc, extra_json),
    )
    reset_instances()


async def load_credentials(provider_name: str) -> dict | None:
    """復号して {api_key, extra} を返す。未登録なら None。"""
    row = await db.fetchone(
        "SELECT api_key_enc, extra FROM provider_credentials WHERE provider_name=?",
        (provider_name,),
    )
    if row is None:
        return None
    api_key = _fernet().decrypt(row["api_key_enc"]).decode("utf-8")
    extra = json.loads(row["extra"]) if row["extra"] else {}
    return {"api_key": api_key, "extra": extra}


async def delete_credentials(provider_name: str) -> None:
    await db.execute(
        "DELETE FROM provider_credentials WHERE provider_name=?", (provider_name,)
    )
    reset_instances()


async def has_credentials(provider_name: str) -> bool:
    row = await db.fetchone(
        "SELECT 1 FROM provider_credentials WHERE provider_name=?", (provider_name,)
    )
    return row is not None
