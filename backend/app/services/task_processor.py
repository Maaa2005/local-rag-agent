"""
バックグラウンドタスク処理ループ。
chat_active フラグが True の間は一時停止して VRAM をチャットに全振りする。
"""
import asyncio
import logging
from datetime import datetime, timezone

from app.database import db
from app.services.indexer import delete_document, index_document
from app.services.parser import parse_file

logger = logging.getLogger(__name__)

# チャット中は True に設定してバックグラウンド処理を一時停止
chat_active = False
_POLL_INTERVAL = 5  # 秒


async def _process_task(task: dict) -> None:
    task_id = task["id"]
    doc_id = task["document_id"]

    await db.execute(
        "UPDATE tasks SET status='processing', updated_at=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), task_id),
    )
    await db.execute(
        "UPDATE documents SET status='processing', updated_at=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), doc_id),
    )

    doc = await db.fetchone(
        "SELECT source_path, access_level FROM documents WHERE id=?", (doc_id,)
    )
    if doc is None:
        raise ValueError(f"document {doc_id} not found")

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
    while True:
        await asyncio.sleep(_POLL_INTERVAL)
        if chat_active:
            continue

        await _process_deleted()

        task = await db.fetchone(
            "SELECT id, document_id, attempts FROM tasks WHERE status='pending' "
            "ORDER BY created_at LIMIT 1"
        )
        if task is None:
            continue

        try:
            await _process_task(dict(task))
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
