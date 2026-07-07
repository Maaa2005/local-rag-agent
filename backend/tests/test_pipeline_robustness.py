"""インデックスパイプライン堅牢化 4 点の回帰テスト。

- 起動時の stale processing 回復 (_recover_stale_processing)
- コピー途中ファイルの settle 判定 (_process_task)
- watcher の pending タスク重複抑止
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from unittest.mock import AsyncMock

from app.services import indexer, task_processor
from app.services.watcher import _Handler


# ─── 共通: task_processor.db を差し替える簡易非同期 SQLite ラッパー ───
class _FakeAsyncDB:
    def __init__(self, db_path):
        self._path = str(db_path)

    async def fetchone(self, sql, params=()):
        import aiosqlite

        async with aiosqlite.connect(self._path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, params) as cur:
                row = await cur.fetchone()
            await conn.commit()
            return row

    async def fetchall(self, sql, params=()):
        import aiosqlite

        async with aiosqlite.connect(self._path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
            return rows

    async def execute(self, sql, params=()):
        import aiosqlite

        async with aiosqlite.connect(self._path) as conn:
            async with conn.execute(sql, params) as cur:
                await conn.commit()
                return cur.lastrowid or 0


def _make_db(tmp_path):
    db_path = tmp_path / "pipeline.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY, source_path TEXT, file_hash TEXT,
                access_level INTEGER, file_type TEXT,
                status TEXT, chunk_count INTEGER DEFAULT 0,
                error_msg TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT, status TEXT, attempts INTEGER DEFAULT 0,
                error_msg TEXT, created_at TEXT, updated_at TEXT
            );
            """
        )
    return db_path


# ─── 修正2: stale processing 回復 ──────────────────────────────
def test_recover_stale_processing_resets_processing_rows(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type,"
            " status, created_at, updated_at) VALUES ('d1','/x','h',1,'.txt','processing','t','t')"
        )
        conn.execute(
            "INSERT INTO tasks (document_id, status, created_at, updated_at)"
            " VALUES ('d1','processing','t','t')"
        )
        conn.commit()

    monkeypatch.setattr(task_processor, "db", _FakeAsyncDB(db_path))

    asyncio.run(task_processor._recover_stale_processing())

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT status FROM tasks").fetchone()[0] == "pending"
        assert conn.execute("SELECT status FROM documents").fetchone()[0] == "pending"


# ─── 修正3(b): コピー途中ファイルの settle 判定 ──────────────────
def test_process_task_defers_when_file_still_settling(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    src = tmp_path / "recent.txt"
    src.write_text("hello")
    # mtime を「今」にしておく (settle 秒未満)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type,"
            " status, created_at, updated_at) VALUES ('d1',?,'h',1,'.txt','processing','t','t')",
            (str(src),),
        )
        conn.execute(
            "INSERT INTO tasks (document_id, status, attempts, created_at, updated_at)"
            " VALUES ('d1','processing',0,'t','t')"
        )
        conn.commit()

    monkeypatch.setattr(task_processor, "db", _FakeAsyncDB(db_path))
    parse_mock = AsyncMock()
    index_mock = AsyncMock()
    monkeypatch.setattr(task_processor, "parse_file", parse_mock)
    monkeypatch.setattr(task_processor, "index_document", index_mock)

    task = {"id": 1, "document_id": "d1", "attempts": 0}
    asyncio.run(task_processor._process_task(task))

    parse_mock.assert_not_called()
    index_mock.assert_not_called()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, attempts FROM tasks WHERE id=1").fetchone()
        assert row[0] == "pending"
        assert row[1] == 0  # attempts は増やさない
        assert conn.execute("SELECT status FROM documents WHERE id='d1'").fetchone()[0] == "pending"


def test_process_task_proceeds_when_file_settled(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    src = tmp_path / "old.txt"
    src.write_text("hello")
    # mtime を十分過去に設定 (settle 秒を超える)
    old_time = os.path.getmtime(src) - 3600
    os.utime(src, (old_time, old_time))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type,"
            " status, created_at, updated_at) VALUES ('d1',?,'h',1,'.txt','processing','t','t')",
            (str(src),),
        )
        conn.execute(
            "INSERT INTO tasks (document_id, status, attempts, created_at, updated_at)"
            " VALUES ('d1','processing',0,'t','t')"
        )
        conn.commit()

    monkeypatch.setattr(task_processor, "db", _FakeAsyncDB(db_path))
    parse_mock = AsyncMock(return_value="本文")
    index_mock = AsyncMock(return_value=3)
    monkeypatch.setattr(task_processor, "parse_file", parse_mock)
    monkeypatch.setattr(task_processor, "index_document", index_mock)

    task = {"id": 1, "document_id": "d1", "attempts": 0}
    asyncio.run(task_processor._process_task(task))

    parse_mock.assert_called_once()
    index_mock.assert_called_once()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=1").fetchone()
        assert row[0] == "done"
        doc_row = conn.execute(
            "SELECT status, chunk_count FROM documents WHERE id='d1'"
        ).fetchone()
        assert doc_row[0] == "done"
        assert doc_row[1] == 3


# ─── 修正3(a): watcher の pending タスク重複抑止 ────────────────
def test_watcher_does_not_duplicate_task_when_pending_exists(tmp_path):
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
                attempts INTEGER NOT NULL DEFAULT 0,
                error_msg TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )
        conn.commit()

    src = tmp_path / "doc.txt"
    src.write_text("最初の内容")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type,"
            " status, created_at, updated_at) VALUES ('d1', ?, 'oldhash', 1, '.txt',"
            " 'pending', 't', 't')",
            (str(src),),
        )
        conn.execute(
            "INSERT INTO tasks (document_id, status, created_at, updated_at)"
            " VALUES ('d1', 'pending', 't', 't')"
        )
        conn.commit()

    # ファイル内容を変更 → hash が変わる
    src.write_text("更新された内容")

    handler = _Handler(str(db_path))
    handler._handle_file(str(src))

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 1  # 新規タスクは追加されない
        doc_hash = conn.execute("SELECT file_hash FROM documents WHERE id='d1'").fetchone()[0]
        assert doc_hash != "oldhash"  # ハッシュ自体は更新される


# ─── 修正4: 非アクティブ旧チャンクの物理削除 ──────────────────
def test_index_document_deletes_inactive_chunks_after_upsert(monkeypatch):
    mock_client = AsyncMock()
    monkeypatch.setattr(indexer, "get_client", lambda: mock_client)

    calls: list[str] = []

    async def record_upsert(*args, **kwargs):
        calls.append("upsert")
        return None

    async def record_delete(*args, **kwargs):
        calls.append("delete")
        return None

    mock_client.upsert.side_effect = record_upsert
    mock_client.delete.side_effect = record_delete

    async def fake_set_payload(*args, **kwargs):
        return None

    mock_client.set_payload.side_effect = fake_set_payload

    asyncio.run(
        indexer.index_document(
            document_id="doc-1",
            text="これはテスト文書です。" * 10,
            source_file="/x/doc.txt",
            access_level=1,
        )
    )

    assert mock_client.upsert.await_count == 1
    assert mock_client.delete.await_count == 1
    # upsert が delete より先に呼ばれていること
    assert calls.index("upsert") < calls.index("delete")

    delete_kwargs = mock_client.delete.await_args.kwargs
    filter_obj = delete_kwargs["points_selector"]
    conditions = filter_obj.must
    keys = {c.key for c in conditions}
    assert "document_id" in keys
    assert "is_active" in keys
    is_active_cond = next(c for c in conditions if c.key == "is_active")
    assert is_active_cond.match.value is False
