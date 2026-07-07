"""
バックグラウンドタスク処理ループ。
チャット実行中はカウンタで管理し、同時並行チャットでも誤って早期解除しない。
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from app.config import settings
from app.database import db
from app.services.indexer import delete_document, index_document
from app.services.parser import parse_file

logger = logging.getLogger(__name__)

# 並列チャットを安全にカウントするためのロック付きカウンタ
_chat_lock = asyncio.Lock()
_active_chats: int = 0
_POLL_INTERVAL = 5  # 秒


def is_chat_active() -> bool:
    """テストや内部ループから現在の状態を確認するための参照。"""
    return _active_chats > 0


@asynccontextmanager
async def chat_session():
    """チャット開始時に enter / 終了時に exit。複数同時呼び出しでも安全。"""
    global _active_chats
    async with _chat_lock:
        _active_chats += 1
    try:
        yield
    finally:
        async with _chat_lock:
            _active_chats = max(0, _active_chats - 1)


async def _claim_next_task() -> dict | None:
    """pending タスクを 1 件アトミックに processing へ遷移して返す。"""
    now = datetime.now(timezone.utc).isoformat()
    row = await db.fetchone_write(
        "UPDATE tasks SET status='processing', updated_at=? "
        "WHERE id = (SELECT id FROM tasks WHERE status='pending' "
        "            ORDER BY created_at LIMIT 1) "
        "RETURNING id, document_id, attempts",
        (now,),
    )
    if row is None:
        return None
    claimed = dict(row)
    await db.execute(
        "UPDATE documents SET status='processing', updated_at=? WHERE id=?",
        (now, claimed["document_id"]),
    )
    return claimed


async def _recover_stale_processing() -> None:
    """前回プロセスが processing 中に落ちた場合、tasks / documents を pending に戻す。"""
    now = datetime.now(timezone.utc).isoformat()
    stale = await db.fetchall("SELECT id FROM tasks WHERE status='processing'")
    await db.execute(
        "UPDATE tasks SET status='pending', updated_at=? WHERE status='processing'",
        (now,),
    )
    await db.execute(
        "UPDATE documents SET status='pending', updated_at=? WHERE status='processing'",
        (now,),
    )
    if len(stale) > 0:
        logger.info("Recovered %d stale processing tasks", len(stale))


async def _process_task(task: dict) -> None:
    task_id = task["id"]
    doc_id = task["document_id"]

    doc = await db.fetchone(
        "SELECT source_path, access_level FROM documents WHERE id=?", (doc_id,)
    )
    if doc is None:
        raise ValueError(f"document {doc_id} not found")

    mtime = os.path.getmtime(doc["source_path"])
    now_ts = datetime.now(timezone.utc).timestamp()
    if now_ts - mtime < settings.index_settle_seconds:
        # コピー途中など書き込み中の可能性があるため、猶予時間を空けて再試行する。
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE tasks SET status='pending', updated_at=? WHERE id=?",
            (now, task_id),
        )
        await db.execute(
            "UPDATE documents SET status='pending', updated_at=? WHERE id=?",
            (now, doc_id),
        )
        logger.info("File still being written, deferred: %s", doc["source_path"])
        return

    text = await parse_file(doc["source_path"])
    chunk_count = await index_document(
        document_id=doc_id,
        text=text,
        source_file=doc["source_path"],
        access_level=doc["access_level"],
    )

    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "UPDATE tasks SET status='done', updated_at=? WHERE id=?",
        (now, task_id),
    )
    await db.execute(
        "UPDATE documents SET status='done', chunk_count=?, updated_at=? WHERE id=?",
        (chunk_count, now, doc_id),
    )
    logger.info("Indexed %s → %d chunks", doc["source_path"], chunk_count)


async def _process_deleted() -> None:
    deleted = await db.fetchall(
        "SELECT id FROM documents WHERE status='deleted'"
    )
    for row in deleted:
        doc_id = row["id"]
        try:
            await delete_document(doc_id)
            await db.execute(
                "UPDATE documents SET status='purged', updated_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), doc_id),
            )
            logger.info("Purged document: %s", doc_id)
        except Exception as exc:
            logger.error("Failed to purge document %s: %s", doc_id, exc)


async def run_task_processor() -> None:
    logger.info("Task processor started")
    await _recover_stale_processing()
    while True:
        await asyncio.sleep(_POLL_INTERVAL)
        if is_chat_active():
            continue

        await _process_deleted()

        task = await _claim_next_task()
        if task is None:
            continue

        try:
            await _process_task(task)
        except Exception as exc:
            logger.error("Task %d failed: %s", task["id"], exc)
            now = datetime.now(timezone.utc).isoformat()
            attempts = task["attempts"] + 1
            status = "failed" if attempts >= 3 else "pending"
            await db.execute(
                "UPDATE tasks SET status=?, attempts=?, error_msg=?, updated_at=? WHERE id=?",
                (status, attempts, str(exc), now, task["id"]),
            )
            await db.execute(
                "UPDATE documents SET status=?, error_msg=?, updated_at=? WHERE id=?",
                (status, str(exc), now, task["document_id"]),
            )
