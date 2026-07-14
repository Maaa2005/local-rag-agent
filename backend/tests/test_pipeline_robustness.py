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
                error_msg TEXT, unclassified INTEGER NOT NULL DEFAULT 0,
                index_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT, updated_at TEXT
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


# ─── 項目8: 再インデックスの世代切替 ──────────────────
def test_index_document_generation_switch_order(monkeypatch):
    """成功時: 残骸削除 → upsert(is_active=False) → activate → 旧世代削除、の順で呼ばれる。"""
    mock_client = AsyncMock()
    monkeypatch.setattr(indexer, "get_client", lambda: mock_client)

    calls: list[str] = []

    async def record_delete(*args, **kwargs):
        calls.append("delete")
        return None

    async def record_upsert(*args, **kwargs):
        calls.append("upsert")
        return None

    async def record_set_payload(*args, **kwargs):
        calls.append("set_payload")
        return None

    mock_client.delete.side_effect = record_delete
    mock_client.upsert.side_effect = record_upsert
    mock_client.set_payload.side_effect = record_set_payload

    chunk_count = asyncio.run(
        indexer.index_document(
            document_id="doc-1",
            text="これはテスト文書です。" * 10,
            source_file="/x/doc.txt",
            access_level=1,
            index_version=1,
        )
    )

    assert chunk_count > 0
    assert calls == ["delete", "upsert", "set_payload", "delete"]

    # upsert された全ポイントは is_active=False + index_version=1
    upsert_points = mock_client.upsert.await_args.kwargs["points"]
    assert all(p.payload["is_active"] is False for p in upsert_points)
    assert all(p.payload["index_version"] == 1 for p in upsert_points)

    # activate (set_payload) は is_active=True をこの document_id/index_version に適用
    set_payload_kwargs = mock_client.set_payload.await_args.kwargs
    assert set_payload_kwargs["payload"] == {"is_active": True}
    activate_conditions = set_payload_kwargs["points"].must
    activate_keys = {c.key for c in activate_conditions}
    assert activate_keys == {"document_id", "index_version"}

    # 2回目の delete (旧世代削除) は index_version 不一致を対象にする
    final_delete_kwargs = mock_client.delete.await_args.kwargs
    filter_obj = final_delete_kwargs["points_selector"]
    assert {c.key for c in filter_obj.must} == {"document_id"}
    assert {c.key for c in filter_obj.must_not} == {"index_version"}


def test_index_document_keeps_old_generation_active_when_upsert_fails(monkeypatch):
    """upsert 失敗時: 旧世代の activate/物理削除に進まない (= 旧世代は is_active=True のまま残る)。"""
    mock_client = AsyncMock()
    monkeypatch.setattr(indexer, "get_client", lambda: mock_client)

    mock_client.upsert.side_effect = RuntimeError("qdrant down")

    try:
        asyncio.run(
            indexer.index_document(
                document_id="doc-1",
                text="これはテスト文書です。" * 10,
                source_file="/x/doc.txt",
                access_level=1,
                index_version=1,
            )
        )
    except RuntimeError:
        pass

    # activate も旧世代削除も呼ばれていない = 旧世代 (is_active=True) はそのまま残る
    mock_client.set_payload.assert_not_called()
    # delete は手順1 (残骸掃除) の1回のみ呼ばれ、旧世代削除 (手順4) には到達していない
    assert mock_client.delete.await_count == 1


def test_index_document_cleans_up_stale_inactive_points_on_next_run(monkeypatch):
    """残骸 (is_active=False) ポイントは次回実行の冒頭で物理削除される。"""
    mock_client = AsyncMock()
    monkeypatch.setattr(indexer, "get_client", lambda: mock_client)

    asyncio.run(
        indexer.index_document(
            document_id="doc-1",
            text="これはテスト文書です。" * 10,
            source_file="/x/doc.txt",
            access_level=1,
            index_version=2,
        )
    )

    first_delete_kwargs = mock_client.delete.await_args_list[0].kwargs
    filter_obj = first_delete_kwargs["points_selector"]
    conditions = filter_obj.must
    keys = {c.key for c in conditions}
    assert keys == {"document_id", "is_active"}
    is_active_cond = next(c for c in conditions if c.key == "is_active")
    assert is_active_cond.match.value is False
