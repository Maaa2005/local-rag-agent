"""監視フォルダの access_level 変更・無効化を既存文書へ波及させる機能のテスト。

- match_folder: 境界判定・ネスト時の最長一致・無効フォルダの扱い
- resync_access_levels: documents / Qdrant への波及、無効フォルダ配下の deactivate、
  変化なしの文書には Qdrant 呼び出しが発生しないこと
- watcher: 無効フォルダ配下の新規/更新ファイルがタスク登録されないこと
"""
from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest

from app.services import access_sync
from app.services.access import match_folder
from app.services.watcher import SyncDatabase, _Handler


# ─── match_folder ──────────────────────────────────────────────
def test_match_folder_does_not_confuse_sibling_prefix():
    folders = [{"path": "/watched/exec", "access_level": 2, "is_active": 1}]
    assert match_folder("/watched/exec/file.txt", folders) is not None
    assert match_folder("/watched/exec-public/file.txt", folders) is None
    # 完全一致もOK
    assert match_folder("/watched/exec", folders) is not None


def test_match_folder_picks_longest_nested_match():
    folders = [
        {"path": "/watched", "access_level": 1, "is_active": 1},
        {"path": "/watched/sub", "access_level": 3, "is_active": 1},
    ]
    matched = match_folder("/watched/sub/deep/file.txt", folders)
    assert matched["path"] == "/watched/sub"
    assert matched["access_level"] == 3


def test_match_folder_returns_inactive_folder_too():
    """is_active=0 のフォルダも一致対象。無効かどうかの判断は呼び出し側が行う。"""
    folders = [{"path": "/watched/exec", "access_level": 2, "is_active": 0}]
    matched = match_folder("/watched/exec/file.txt", folders)
    assert matched is not None
    assert matched["is_active"] == 0


def test_match_folder_no_match_returns_none():
    folders = [{"path": "/watched/exec", "access_level": 2, "is_active": 1}]
    assert match_folder("/other/place/file.txt", folders) is None


# ─── resync_access_levels ──────────────────────────────────────
@pytest.fixture()
def fake_async_db(tmp_path):
    """access_sync.db を差し替える簡易非同期 SQLite ラッパー。"""
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

    class _DB:
        async def fetchone(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, params) as cur:
                    row = await cur.fetchone()
                await conn.commit()
                return row

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


def _seed(db_path, folders=(), documents=()):
    with sqlite3.connect(db_path) as conn:
        for f in folders:
            conn.execute(
                "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, ?, ?)",
                (f["path"], f["access_level"], f["is_active"]),
            )
        for d in documents:
            conn.execute(
                "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
                "status, unclassified, created_at, updated_at) "
                "VALUES (?, ?, 'h', ?, '.txt', ?, ?, 't', 't')",
                (
                    d["id"], d["source_path"], d["access_level"],
                    d.get("status", "done"), d.get("unclassified", 0),
                ),
            )
        conn.commit()


def test_resync_propagates_level_change_to_documents_and_qdrant(monkeypatch, fake_async_db):
    db_path, fake_db = fake_async_db
    _seed(
        db_path,
        folders=[{"path": "/watched/exec", "access_level": 3, "is_active": 1}],
        documents=[{"id": "d1", "source_path": "/watched/exec/a.txt", "access_level": 1}],
    )
    monkeypatch.setattr(access_sync, "db", fake_db)
    mock_client = AsyncMock()
    monkeypatch.setattr(access_sync, "get_client", lambda: mock_client)
    deactivate_mock = AsyncMock()
    monkeypatch.setattr(access_sync, "deactivate_document", deactivate_mock)

    result = asyncio.run(access_sync.resync_access_levels())

    assert result == {"deleted": 0, "updated": 1}
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT access_level FROM documents WHERE id='d1'").fetchone()
    assert row[0] == 3
    mock_client.set_payload.assert_awaited_once()
    deactivate_mock.assert_not_called()


def test_resync_deactivates_documents_under_inactive_folder(monkeypatch, fake_async_db):
    db_path, fake_db = fake_async_db
    _seed(
        db_path,
        folders=[{"path": "/watched/exec", "access_level": 2, "is_active": 0}],
        documents=[{"id": "d1", "source_path": "/watched/exec/a.txt", "access_level": 2}],
    )
    monkeypatch.setattr(access_sync, "db", fake_db)
    mock_client = AsyncMock()
    monkeypatch.setattr(access_sync, "get_client", lambda: mock_client)
    deactivate_mock = AsyncMock()
    monkeypatch.setattr(access_sync, "deactivate_document", deactivate_mock)

    result = asyncio.run(access_sync.resync_access_levels())

    assert result == {"deleted": 1, "updated": 0}
    deactivate_mock.assert_awaited_once_with("d1")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status FROM documents WHERE id='d1'").fetchone()
    assert row[0] == "deleted"
    mock_client.set_payload.assert_not_called()


def test_resync_no_change_triggers_no_qdrant_calls(monkeypatch, fake_async_db):
    db_path, fake_db = fake_async_db
    _seed(
        db_path,
        folders=[{"path": "/watched/exec", "access_level": 2, "is_active": 1}],
        documents=[{"id": "d1", "source_path": "/watched/exec/a.txt", "access_level": 2}],
    )
    monkeypatch.setattr(access_sync, "db", fake_db)
    mock_client = AsyncMock()
    monkeypatch.setattr(access_sync, "get_client", lambda: mock_client)
    deactivate_mock = AsyncMock()
    monkeypatch.setattr(access_sync, "deactivate_document", deactivate_mock)

    result = asyncio.run(access_sync.resync_access_levels())

    assert result == {"deleted": 0, "updated": 0}
    mock_client.set_payload.assert_not_called()
    deactivate_mock.assert_not_called()


# ─── watcher: 無効フォルダ配下のスキップ ─────────────────────────
@pytest.fixture()
def watch_sqlite(tmp_path):
    """SyncDatabase が使う実 SQLite に watch_folders/documents/tasks を用意する。"""
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


def test_watcher_skips_new_file_under_inactive_folder(tmp_path, watch_sqlite):
    watched_dir = tmp_path / "exec"
    watched_dir.mkdir()
    with sqlite3.connect(watch_sqlite) as conn:
        conn.execute(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, 2, 0)",
            (str(watched_dir),),
        )
        conn.commit()

    target = watched_dir / "a.txt"
    target.write_text("hello")

    handler = _Handler(str(watch_sqlite))
    handler._handle_file(str(target))

    with sqlite3.connect(watch_sqlite) as conn:
        docs = conn.execute("SELECT * FROM documents").fetchall()
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
    assert docs == []
    assert tasks == []


def test_watcher_skips_update_under_inactive_folder(tmp_path, watch_sqlite):
    watched_dir = tmp_path / "exec"
    watched_dir.mkdir()
    target = watched_dir / "a.txt"
    target.write_text("hello")

    with sqlite3.connect(watch_sqlite) as conn:
        # 有効な状態で先に取り込み済みのドキュメントを用意
        conn.execute(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, 2, 1)",
            (str(watched_dir),),
        )
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
            "status, created_at, updated_at) VALUES ('d1', ?, 'oldhash', 2, '.txt', 'done', 't', 't')",
            (str(target),),
        )
        conn.commit()

    # フォルダを無効化した後にファイルが更新される (下方向リーク再現ケース)
    with sqlite3.connect(watch_sqlite) as conn:
        conn.execute("UPDATE watch_folders SET is_active=0 WHERE path=?", (str(watched_dir),))
        conn.commit()

    target.write_text("updated content")

    handler = _Handler(str(watch_sqlite))
    handler._handle_file(str(target))

    with sqlite3.connect(watch_sqlite) as conn:
        row = conn.execute("SELECT file_hash, status FROM documents WHERE id='d1'").fetchone()
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
    # file_hash は更新されず (再インデックスされていない)、タスクも積まれない
    assert row[0] == "oldhash"
    assert tasks == []
