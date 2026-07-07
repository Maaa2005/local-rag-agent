"""Phase 2 (セキュリティ/運用強化) のテスト。

- watcher / access_sync: 未分類パス (fail-open) の警告ログと documents.unclassified フラグ
- chat: 監査ログ (audit_logs) への記録
- admin API: 未分類文書一覧・監査ログ一覧 (管理者限定)
- llm.build_prompt (_build_context): source_file のフルパス→ファイル名縮約
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app.database as db_module
from app.database import db
from app.services import access_sync
from app.services.watcher import _Handler


# ─── watcher: 未分類パスの警告ログ・フラグ ─────────────────────────
@pytest.fixture()
def watch_sqlite(tmp_path):
    db_path = tmp_path / "watch.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE watch_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                access_level INTEGER NOT NULL DEFAULT 1,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT
            );
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_path TEXT UNIQUE NOT NULL,
                file_hash TEXT,
                access_level INTEGER NOT NULL DEFAULT 1,
                file_type TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                chunk_count INTEGER DEFAULT 0,
                error_msg TEXT,
                unclassified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                error_msg TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.commit()
    return db_path


def test_unclassified_path_is_flagged_and_warns(tmp_path, watch_sqlite, caplog):
    """どの監視フォルダにも属さないパスは Lv1 でインデックスされ、
    documents.unclassified=1 が立ち、警告ログが出る。"""
    target = tmp_path / "a.txt"
    target.write_text("hello")

    handler = _Handler(str(watch_sqlite))
    with caplog.at_level(logging.WARNING):
        handler._handle_file(str(target))

    with sqlite3.connect(watch_sqlite) as conn:
        row = conn.execute(
            "SELECT access_level, unclassified FROM documents WHERE source_path=?",
            (str(target),),
        ).fetchone()
    assert row == (1, 1)
    assert any("Unclassified path indexed at level 1" in r.message for r in caplog.records)


def test_classified_path_is_not_flagged(tmp_path, watch_sqlite):
    watched_dir = tmp_path / "exec"
    watched_dir.mkdir()
    with sqlite3.connect(watch_sqlite) as conn:
        conn.execute(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, 2, 1)",
            (str(watched_dir),),
        )
        conn.commit()

    target = watched_dir / "b.txt"
    target.write_text("hello")

    handler = _Handler(str(watch_sqlite))
    handler._handle_file(str(target))

    with sqlite3.connect(watch_sqlite) as conn:
        row = conn.execute(
            "SELECT access_level, unclassified FROM documents WHERE source_path=?",
            (str(target),),
        ).fetchone()
    assert row == (2, 0)


def test_unclassified_flag_clears_on_update_once_folder_registered(tmp_path, watch_sqlite):
    """先に未分類として取り込まれた後、監視フォルダが登録され再取り込みされると
    unclassified が 0 に戻る (watcher の更新イベント経由)。"""
    watched_dir = tmp_path / "newfolder"
    watched_dir.mkdir()
    target = watched_dir / "c.txt"
    target.write_text("v1")

    handler = _Handler(str(watch_sqlite))
    handler._handle_file(str(target))
    with sqlite3.connect(watch_sqlite) as conn:
        row = conn.execute(
            "SELECT unclassified FROM documents WHERE source_path=?", (str(target),)
        ).fetchone()
    assert row == (1,)

    with sqlite3.connect(watch_sqlite) as conn:
        conn.execute(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, 1, 1)",
            (str(watched_dir),),
        )
        conn.commit()

    target.write_text("v2 updated")
    handler._handle_file(str(target))

    with sqlite3.connect(watch_sqlite) as conn:
        row = conn.execute(
            "SELECT unclassified FROM documents WHERE source_path=?", (str(target),)
        ).fetchone()
    assert row == (0,)


# ─── access_sync.resync_access_levels: unclassified の波及 ─────────────
@pytest.fixture()
def fake_async_db(tmp_path):
    import aiosqlite

    db_path = tmp_path / "resync.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE watch_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                access_level INTEGER NOT NULL DEFAULT 1,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT
            );
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_path TEXT UNIQUE NOT NULL,
                file_hash TEXT,
                access_level INTEGER NOT NULL DEFAULT 1,
                file_type TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                chunk_count INTEGER DEFAULT 0,
                error_msg TEXT,
                unclassified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
            "status, unclassified, created_at, updated_at) "
            "VALUES ('d1', '/watched/newteam/a.txt', 'h', 1, '.txt', 'done', 1, 't', 't')"
        )
        conn.commit()

    class _DB:
        async def fetchone(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, params) as cur:
                    return await cur.fetchone()

        async def fetchall(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, params) as cur:
                    return await cur.fetchall()

        async def execute(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                async with conn.execute(sql, params) as cur:
                    await conn.commit()
                    return cur.lastrowid or 0

    return db_path, _DB()


def test_resync_clears_unclassified_once_folder_matches(monkeypatch, fake_async_db):
    from unittest.mock import AsyncMock

    db_path, fake_db = fake_async_db
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, 2, 1)",
            ("/watched/newteam",),
        )
        conn.commit()

    monkeypatch.setattr(access_sync, "db", fake_db)
    mock_client = AsyncMock()
    monkeypatch.setattr(access_sync, "get_client", lambda: mock_client)
    monkeypatch.setattr(access_sync, "deactivate_document", AsyncMock())

    result = asyncio.run(access_sync.resync_access_levels())

    assert result == {"deleted": 0, "updated": 1}
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT access_level, unclassified FROM documents WHERE id='d1'"
        ).fetchone()
    assert row == (2, 0)


def test_resync_flags_unclassified_and_warns_when_no_folder_matches(monkeypatch, fake_async_db, caplog):
    from unittest.mock import AsyncMock

    db_path, fake_db = fake_async_db  # 監視フォルダなし → 既存の doc は未分類のまま
    monkeypatch.setattr(access_sync, "db", fake_db)
    mock_client = AsyncMock()
    monkeypatch.setattr(access_sync, "get_client", lambda: mock_client)
    monkeypatch.setattr(access_sync, "deactivate_document", AsyncMock())

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(access_sync.resync_access_levels())

    assert result == {"deleted": 0, "updated": 0}  # 既に access_level=1, unclassified=1 のまま変化なし
    assert any("Unclassified path indexed at level 1" in r.message for r in caplog.records)


# ─── /api/admin: 未分類文書一覧・監査ログ ─────────────────────────
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


def _login(client, username="admin", password="admin"):
    r = client.post("/api/auth/token", data={"username": username, "password": password})
    return r.json()["access_token"]


def test_admin_lists_unclassified_documents(client, isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
            "status, unclassified, created_at, updated_at) "
            "VALUES ('d1', '/watched/unknown/x.txt', 'h', 1, '.txt', 'done', 1, 't', 't')"
        )
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
            "status, unclassified, created_at, updated_at) "
            "VALUES ('d2', '/watched/known/y.txt', 'h', 2, '.txt', 'done', 0, 't', 't')"
        )
        conn.commit()

    token = _login(client)
    r = client.get(
        "/api/admin/documents/unclassified", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["id"] == "d1"


def test_admin_unclassified_requires_admin(client, isolated_db):
    from app.core.security import hash_password

    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
            ("bob", hash_password("bobpw"), 1),
        )
        conn.commit()

    token = _login(client, "bob", "bobpw")
    r = client.get(
        "/api/admin/documents/unclassified", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403


def test_admin_lists_audit_logs(client, isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, username, question, retrieved_chunks, answer, "
            "error, created_at) VALUES (1, 'admin', 'Q1', ?, 'A1', NULL, '2026-01-01T00:00:00')",
            (json.dumps([{"source_file": "a.txt", "score": 0.9, "access_level": 1}]),),
        )
        conn.commit()

    token = _login(client)
    r = client.get(
        "/api/admin/audit-logs?limit=10&offset=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["question"] == "Q1"
    assert body[0]["retrieved_chunks"][0]["source_file"] == "a.txt"


def test_admin_audit_logs_requires_admin(client, isolated_db):
    from app.core.security import hash_password

    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
            ("bob", hash_password("bobpw"), 1),
        )
        conn.commit()

    token = _login(client, "bob", "bobpw")
    r = client.get(
        "/api/admin/audit-logs", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403


# ─── chat: 監査ログの記録 ─────────────────────────────────────────
def test_chat_stream_records_audit_log(monkeypatch, isolated_db):
    """_event_stream を直接駆動し、ストリーム完了後に audit_logs へ1件記録されること。"""
    from app.api import chat as chat_module

    async def fake_retrieve(question, access_level):
        return [
            {"content": "本文", "source_file": "/watched/a.txt", "score": 0.8, "access_level": 1}
        ]

    async def fake_stream_answer(question, chunks, provider_name=None):
        for tok in ["こんにち", "は"]:
            yield tok

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    async def _run():
        await db.connect()
        try:
            user = {"id": 1, "username": "admin", "access_level": 3}
            events = [e async for e in chat_module._event_stream("質問です", user)]
            assert any('"type": "sources"' in e for e in events)
            row = await db.fetchone(
                "SELECT username, question, answer, error, retrieved_chunks FROM audit_logs"
            )
            return row
        finally:
            await db.close()

    row = asyncio.run(_run())
    assert row is not None
    assert row["username"] == "admin"
    assert row["question"] == "質問です"
    assert row["answer"] == "こんにちは"
    assert row["error"] is None
    chunks = json.loads(row["retrieved_chunks"])
    assert chunks == [{"source_file": "/watched/a.txt", "score": 0.8, "access_level": 1}]


def test_chat_stream_records_error_in_audit_log(monkeypatch, isolated_db):
    from app.api import chat as chat_module

    async def fake_retrieve(question, access_level):
        return [{"content": "本文", "source_file": "/watched/a.txt", "score": 0.5, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        raise RuntimeError("boom")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    async def _run():
        await db.connect()
        try:
            user = {"id": 1, "username": "admin", "access_level": 3}
            _ = [e async for e in chat_module._event_stream("質問です", user)]
            return await db.fetchone("SELECT answer, error FROM audit_logs")
        finally:
            await db.close()

    row = asyncio.run(_run())
    assert row is not None
    assert row["error"] is not None
    assert "エラー" in row["error"]
