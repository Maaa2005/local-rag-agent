"""セキュリティレビュー最優先7項目のうち、他のテストファイルでカバーされていない
部分の回帰テスト。

- 項目6: デフォルト資格情報 (admin/admin) のまま他 API を叩くと 403、
  パスワード変更後は解除される。/me と /password は変更前でも使える。
- 項目7: システム管理者権限 (is_admin) と文書アクセスレベル (access_level) の分離。
  is_admin=1 でも access_level を超える文書は検索に出てこない。
  is_admin=0 なら access_level=3 でも管理 API は使えない。
- 項目5 / 3b: 会話履歴 (messages.sources) は本文を複製保存せず、表示時に
  document_id から権限再チェックして解決する。権限降格・削除後は redacted になる。
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.api.auth as auth_module
import app.core.security as security_module
import app.database as db_module
from app.core.security import hash_password


def _freeze_datetime(monkeypatch):
    """項目高3: iat と password_changed_at は秒単位で比較される。bcrypt の
    ハッシュ計算に数百 ms かかるため、実時刻のままログイン→パスワード変更を
    行うと秒境界をまたいで稀に flaky になる (同一トークンを使い回すテストのみ)。
    その区間だけ時刻を固定し iat == password_changed_at (同一秒) を保証する。
    設計上「同一秒は許可」なので、これはテストを安定させるための時間固定であり
    セキュリティ判定のロジック自体は変更しない。"""
    frozen = datetime.now(timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(security_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(auth_module, "datetime", _FrozenDatetime)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.db._conn = None
    yield db_path
    if db_module.db._conn is not None:
        asyncio.run(db_module.db.close())


@pytest.fixture()
def client(isolated_db, monkeypatch):
    async def _noop(*args, **kwargs):
        return None

    def _noop_sync(*args, **kwargs):
        return None

    monkeypatch.setattr("app.main.ensure_collection", _noop)
    monkeypatch.setattr("app.main.start_watcher", _noop_sync)
    monkeypatch.setattr("app.main.stop_watcher", _noop_sync)

    async def _run_task_processor_stub():
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr("app.main.run_task_processor", _run_task_processor_stub)

    from app.main import app

    with TestClient(app) as c:
        yield c


def _token(client, username="admin", password="admin"):
    r = client.post("/api/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200
    return r.json()["access_token"]


# ─── 項目6: 初期 admin/admin のパスワード変更強制 ─────────────────
def test_default_admin_blocked_from_other_apis_until_password_changed(client):
    token = _token(client)

    # /me と /password は変更前でも使える (自分を変更するための抜け道)。
    r_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r_me.status_code == 200
    assert r_me.json()["must_change_password"] is True

    # それ以外 (チャット・管理 API) は 403。
    r_chat = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "テスト"},
    )
    assert r_chat.status_code == 403

    r_folders = client.get(
        "/api/admin/folders", headers={"Authorization": f"Bearer {token}"}
    )
    assert r_folders.status_code == 403

    r_docs = client.get(
        "/api/admin/documents", headers={"Authorization": f"Bearer {token}"}
    )
    assert r_docs.status_code == 403


def test_apis_unblocked_after_admin_password_changed(client, monkeypatch):
    _freeze_datetime(monkeypatch)
    token = _token(client)
    r = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "admin", "new_password": "adminpw12345"},
    )
    assert r.status_code == 200
    token = r.json()['access_token']

    r_docs = client.get(
        "/api/admin/documents", headers={"Authorization": f"Bearer {token}"}
    )
    assert r_docs.status_code == 200


# ─── 項目7: is_admin と access_level の分離 ───────────────────────
def test_is_admin_independent_of_access_level(client, isolated_db):
    """access_level=3 (最高位) でも is_admin=0 なら管理 API は使えない。
    逆に is_admin=1 でも access_level=1 なら Lv3 文書は検索フィルタに出てこない。
    """
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, access_level, is_admin, "
            "must_change_password) VALUES (?, ?, 3, 0, 0)",
            ("exec_no_admin", hash_password("execpw123")),
        )
        conn.execute(
            "INSERT INTO users (username, password_hash, access_level, is_admin, "
            "must_change_password) VALUES (?, ?, 1, 1, 0)",
            ("junior_admin", hash_password("juniorpw123")),
        )
        conn.commit()

    token_exec = _token(client, "exec_no_admin", "execpw123")
    r = client.get(
        "/api/admin/folders", headers={"Authorization": f"Bearer {token_exec}"}
    )
    assert r.status_code == 403  # access_level=3 だが is_admin=0

    token_admin = _token(client, "junior_admin", "juniorpw123")
    r2 = client.get(
        "/api/admin/folders", headers={"Authorization": f"Bearer {token_admin}"}
    )
    assert r2.status_code == 200  # is_admin=1 なので管理 API は使える

    from app.services.retriever import build_access_filter

    # junior_admin の access_level=1 相当のフィルタは Lv3 (>1) を通さない。
    cond = {c.key: c for c in build_access_filter(1).must}["access_level"]
    assert cond.range.lte == 1


# ─── 項目5 / 3b: 会話履歴の本文最小化 + 表示時の権限再チェック ─────
@pytest.fixture()
def chat_ready_token(client, isolated_db, monkeypatch):
    _freeze_datetime(monkeypatch)
    token = _token(client)
    r = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "admin", "new_password": "adminpw12345"},
    )
    assert r.status_code == 200
    return r.json()['access_token']


def test_conversation_history_does_not_persist_chunk_content(
    monkeypatch, client, isolated_db, chat_ready_token
):
    from app.api import chat as chat_module

    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
            "status, unclassified, created_at, updated_at) "
            "VALUES ('doc1', '/watched/doc1.txt', 'h', 1, '.txt', 'done', 0, 't', 't')"
        )
        conn.commit()

    # admin (access_level=3, 最高位) では access_level を上げても降格を再現できないため、
    # Lv2 の一般ユーザーで会話履歴を作る。
    r_user = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {chat_ready_token}"},
        json={"username": "lv2user", "password": "lv2userpw1", "access_level": 2},
    )
    assert r_user.status_code == 200
    token = _token(client, "lv2user", "lv2userpw1")
    # 項目中6: 新規作成ユーザーは must_change_password=1 のため、変更するまで
    # /api/chat 等は 403 になる。テストの前提として一度パスワードを変更する。
    r_pw = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "lv2userpw1", "new_password": "lv2userpwnew1"},
    )
    assert r_pw.status_code == 200
    token = _token(client, "lv2user", "lv2userpwnew1")

    async def fake_retrieve(question, access_level):
        return [
            {
                "content": "極秘の本文です",
                "source_file": "/watched/doc1.txt",
                "score": 0.9,
                "access_level": 1,
                "document_id": "doc1",
                "chunk_index": 0,
            }
        ]

    async def fake_stream_answer(question, chunks, provider_name=None):
        for tok in ["回答", "です[1]"]:
            yield tok

    async def fake_get_chunk_content(document_id, chunk_index):
        return "極秘の本文です"

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)
    monkeypatch.setattr(chat_module, "get_chunk_content", fake_get_chunk_content)

    r = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "教えて"},
    )
    assert r.status_code == 200
    body_text = r.text
    conv_line = next(l for l in body_text.splitlines() if '"type": "meta"' in l)
    conversation_id = json.loads(conv_line[len("data: "):])["conversation_id"]

    # 永続化された sources には本文 (content) が含まれない (項目5)。
    # 実行中のアプリが使う DB シングルトン接続を閉じてしまわないよう、
    # 直接 sqlite3 で読む (isolated_db は生ファイルパス)。
    with sqlite3.connect(isolated_db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sources FROM messages WHERE conversation_id=? AND role='assistant'",
            (conversation_id,),
        ).fetchone()
    raw_sources = json.loads(row["sources"])
    assert len(raw_sources) == 1
    assert "content" not in raw_sources[0]
    assert raw_sources[0]["document_id"] == "doc1"

    # 表示時 (GET messages) は権限チェックの上で本文を解決して返す。
    r2 = client.get(
        f"/api/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 200
    msgs = r2.json()
    assistant_msg = next(m for m in msgs if m["role"] == "assistant")
    assert assistant_msg["sources"][0]["content"] == "極秘の本文です"
    assert not assistant_msg["sources"][0].get("redacted")

    # 文書の access_level を上げて (降格) lv2user (access_level=2) の権限外にすると、
    # 履歴表示は redacted になる。
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("UPDATE documents SET access_level=3 WHERE id='doc1'")
        conn.commit()

    r3 = client.get(
        f"/api/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r3.status_code == 200
    msgs3 = r3.json()
    assistant_msg3 = next(m for m in msgs3 if m["role"] == "assistant")
    src = assistant_msg3["sources"][0]
    assert src.get("redacted") is True
    assert "content" not in src
    assert "source_file" not in src


def test_conversation_history_redacts_after_document_deleted(
    monkeypatch, client, isolated_db, chat_ready_token
):
    from app.api import chat as chat_module

    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
            "status, unclassified, created_at, updated_at) "
            "VALUES ('doc2', '/watched/doc2.txt', 'h', 1, '.txt', 'done', 0, 't', 't')"
        )
        conn.commit()

    async def fake_retrieve(question, access_level):
        return [
            {
                "content": "秘密",
                "source_file": "/watched/doc2.txt",
                "score": 0.5,
                "access_level": 1,
                "document_id": "doc2",
                "chunk_index": 0,
            }
        ]

    async def fake_stream_answer(question, chunks, provider_name=None):
        yield "回答[1]"

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    r = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {chat_ready_token}"},
        json={"question": "教えて"},
    )
    conv_line = next(l for l in r.text.splitlines() if '"type": "meta"' in l)
    conversation_id = json.loads(conv_line[len("data: "):])["conversation_id"]

    # 文書削除 (即時失効: 項目3)。
    with sqlite3.connect(isolated_db) as conn:
        conn.execute("UPDATE documents SET status='deleted' WHERE id='doc2'")
        conn.commit()

    r2 = client.get(
        f"/api/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {chat_ready_token}"},
    )
    assistant_msg = next(m for m in r2.json() if m["role"] == "assistant")
    assert assistant_msg["sources"][0].get("redacted") is True
