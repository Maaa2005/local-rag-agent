"""Critical 修正の回帰テスト。

- chat_session: 並列チャットで活性カウンタが正しく上下し、片方終了でも他方が動作中ならアクティブ。
- _claim_next_task: pending タスクを 1 件アトミックに processing へ遷移して返す。
- Watcher 単一トランザクション: documents INSERT が失敗したら tasks も登録されない。
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.services import task_processor
from app.services.watcher import SyncDatabase


# ─── chat_session ─────────────────────────────────────────────
def test_chat_session_increments_and_decrements():
    async def scenario():
        assert task_processor.is_chat_active() is False
        async with task_processor.chat_session():
            assert task_processor.is_chat_active() is True
        assert task_processor.is_chat_active() is False

    asyncio.run(scenario())


def test_chat_session_handles_concurrent_sessions():
    """2つの並列チャット中に1つが終了しても、もう片方が走っていれば active。"""

    async def scenario():
        ev_started = asyncio.Event()
        ev_release = asyncio.Event()

        async def inner():
            async with task_processor.chat_session():
                ev_started.set()
                await ev_release.wait()

        task = asyncio.create_task(inner())
        await ev_started.wait()

        async with task_processor.chat_session():
            assert task_processor.is_chat_active() is True
        # 短いほう (with-block) は抜けたが、長いほう (task) はまだ実行中
        assert task_processor.is_chat_active() is True

        ev_release.set()
        await task
        assert task_processor.is_chat_active() is False

    asyncio.run(scenario())


# ─── _claim_next_task ─────────────────────────────────────────
@pytest.fixture()
def patched_db(tmp_path, monkeypatch):
    """task_processor が触る db を一時 DB に差し替える。"""
    import aiosqlite

    db_path = tmp_path / "claim.db"
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

    class _DB:
        async def fetchone(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, params) as cur:
                    row = await cur.fetchone()
                await conn.commit()
                return row

        async def fetchone_write(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, params) as cur:
                    row = await cur.fetchone()
                await conn.commit()
                return row

        async def execute(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                async with conn.execute(sql, params) as cur:
                    await conn.commit()
                    return cur.lastrowid or 0

    monkeypatch.setattr(task_processor, "db", _DB())
    return db_path


def test_claim_next_task_returns_pending_and_marks_processing(patched_db):
    with sqlite3.connect(patched_db) as conn:
        conn.execute(
            "INSERT INTO documents (id, source_path, file_hash, access_level, file_type,"
            " status, created_at, updated_at) VALUES ('d1','/x','h',1,'.txt','pending','t','t')"
        )
        conn.execute(
            "INSERT INTO tasks (document_id, status, created_at, updated_at)"
            " VALUES ('d1','pending','2024-01-01T00:00:00','t')"
        )
        conn.commit()

    claimed = asyncio.run(task_processor._claim_next_task())
    assert claimed is not None
    assert claimed["document_id"] == "d1"

    with sqlite3.connect(patched_db) as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (claimed["id"],)).fetchone()
        assert row[0] == "processing"
        row = conn.execute("SELECT status FROM documents WHERE id='d1'").fetchone()
        assert row[0] == "processing"


def test_claim_next_task_returns_none_when_no_pending(patched_db):
    claimed = asyncio.run(task_processor._claim_next_task())
    assert claimed is None


# ─── Watcher 単一トランザクション ─────────────────────────────
def test_execute_many_atomic_rolls_back_on_failure(tmp_path):
    db_path = tmp_path / "tx.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT NOT NULL)")
        conn.commit()

    sdb = SyncDatabase(str(db_path))
    with pytest.raises(sqlite3.IntegrityError):
        sdb.execute_many_atomic([
            ("INSERT INTO t (id, v) VALUES (?, ?)", (1, "ok")),
            ("INSERT INTO t (id, v) VALUES (?, ?)", (2, None)),  # NOT NULL 違反
        ])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id FROM t").fetchall()
    assert rows == []  # 1 件目もロールバックされている
