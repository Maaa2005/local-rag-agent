"""SQLite (aiosqlite) のシングル接続管理。

aiosqlite の Connection は内部で 1 つのワーカスレッドを持つため、
全リクエストで 1 接続を共有しても安全。短命接続を毎回開閉する従来実装は
bcrypt 込みのログインや watcher → タスク投入のたびに無視できないオーバーヘッドが発生していた。
"""
from __future__ import annotations

import asyncio
import json
import logging
import ntpath
import posixpath
from pathlib import Path

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)


def _basename_for_migration(path: str | None) -> str:
    """フルパスからファイル名のみを取り出す (ここでのみ使う軽量版)。

    app.services 側のヘルパーと重複するが、database.py は app.services に
    依存させたくない (循環 import 回避) ためここに単独定義する。
    """
    if not path:
        return path or ""
    return posixpath.basename(ntpath.basename(path))

DB_PATH = Path(settings.data_dir) / "app.db"
MIGRATION_PATH = Path(__file__).parent.parent / "migrations" / "init.sql"


class Database:
    """単一の aiosqlite.Connection をシングルトンとして保持するラッパー。"""

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._conn is not None:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(DB_PATH)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        await conn.executescript(sql)
        await conn.commit()
        await self._migrate_legacy_columns(conn)
        self._conn = conn

    @staticmethod
    async def _migrate_legacy_columns(conn: aiosqlite.Connection) -> None:
        """`CREATE TABLE IF NOT EXISTS` は既存テーブルへカラムを追加しないため、
        init.sql に後から列を足した場合はここで ALTER TABLE を補う。
        """
        async with conn.execute("PRAGMA table_info(documents)") as cur:
            doc_columns = {row[1] async for row in cur}
        if "unclassified" not in doc_columns:
            await conn.execute(
                "ALTER TABLE documents ADD COLUMN unclassified INTEGER NOT NULL DEFAULT 0"
            )
            await conn.commit()
        if "index_version" not in doc_columns:
            await conn.execute(
                "ALTER TABLE documents ADD COLUMN index_version INTEGER NOT NULL DEFAULT 0"
            )
            await conn.commit()

        async with conn.execute("PRAGMA table_info(users)") as cur:
            user_columns = {row[1] async for row in cur}
        if "is_admin" not in user_columns:
            await conn.execute(
                "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
            )
            # 既存 DB の互換維持: これまで access_level>=3 を管理者とみなしていたため、
            # 移行時点で access_level=3 のユーザーには is_admin=1 を引き継ぐ。
            await conn.execute("UPDATE users SET is_admin=1 WHERE access_level>=3")
            await conn.commit()
        if "token_version" not in user_columns:
            await conn.execute(
                "ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0"
            )
            await conn.commit()
        if "must_change_password" not in user_columns:
            await conn.execute(
                "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
            )
            await conn.commit()
        if "password_changed_at" not in user_columns:
            await conn.execute("ALTER TABLE users ADD COLUMN password_changed_at TEXT")
            await conn.commit()

        async with conn.execute("PRAGMA table_info(audit_logs)") as cur:
            audit_columns = {row[1] async for row in cur}
        if "answer_chars" not in audit_columns:
            await conn.execute("ALTER TABLE audit_logs ADD COLUMN answer_chars INTEGER")
            # 項目高1: 過去に保存された監査ログの回答本文を一度だけスクラブする。
            # 列追加と同じタイミングでしか実行されないため、以降の起動では
            # answer は既に NULL のままで再実行の必要がない。
            await conn.execute("UPDATE audit_logs SET answer=NULL WHERE answer IS NOT NULL")
            await conn.commit()

        await Database._strip_legacy_message_source_content(conn)

    @staticmethod
    async def _strip_legacy_message_source_content(conn: aiosqlite.Connection) -> None:
        """項目高2: messages.sources の旧形式 (content キー込み) を一度だけ剥がす。

        schema_migrations テーブルに適用済みフラグを立てて冪等性を担保し、
        起動のたびに全 messages 行をスキャンしないようにする。
        """
        migration_key = "strip_legacy_message_sources_content"
        async with conn.execute(
            "SELECT 1 FROM schema_migrations WHERE key = ?", (migration_key,)
        ) as cur:
            already_applied = await cur.fetchone()
        if already_applied:
            return

        async with conn.execute("SELECT id, sources FROM messages") as cur:
            rows = await cur.fetchall()

        for row in rows:
            message_id, raw_sources = row[0], row[1]
            try:
                sources = json.loads(raw_sources)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable messages.sources during legacy strip migration "
                    "(message_id=%s)",
                    message_id,
                )
                continue
            if not isinstance(sources, list):
                continue

            changed = False
            stripped: list[dict] = []
            for item in sources:
                if not isinstance(item, dict):
                    stripped.append(item)
                    continue
                if "content" in item:
                    changed = True
                    cleaned = {"id": item.get("id"), "score": item.get("score")}
                    if item.get("document_id") is not None:
                        cleaned["document_id"] = item.get("document_id")
                    if item.get("chunk_index") is not None:
                        cleaned["chunk_index"] = item.get("chunk_index")
                    if item.get("source_file") is not None:
                        cleaned["source_file"] = _basename_for_migration(item.get("source_file"))
                    stripped.append(cleaned)
                else:
                    stripped.append(item)

            if changed:
                await conn.execute(
                    "UPDATE messages SET sources = ? WHERE id = ?",
                    (json.dumps(stripped, ensure_ascii=False), message_id),
                )

        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (key) VALUES (?)", (migration_key,)
        )
        await conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected; call connect() first.")
        return self._conn

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        """純粋な読み取り（SELECT）用。commit しない。"""
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                return await cur.fetchone()

    async def fetchone_write(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        """UPDATE/INSERT ... RETURNING 用。取得後に commit して永続化する。"""
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                row = await cur.fetchone()
            await conn.commit()
            return row

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                return await cur.fetchall()

    async def execute(self, sql: str, params: tuple = ()) -> int:
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                last = cur.lastrowid or 0
            await conn.commit()
            return last

    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        conn = self._require()
        async with self._lock:
            await conn.executemany(sql, params_list)
            await conn.commit()


db = Database()


async def init_db() -> None:
    """旧 API 互換: 接続を確立してスキーマを初期化する。"""
    await db.connect()


async def close_db() -> None:
    await db.close()
