"""監視フォルダの access_level 変更・無効化を既存文書へ波及させる。

フォルダ追加・access_level 変更・無効化の API 操作後に必ず呼び出し、
documents テーブルと Qdrant のペイロードを最新の監視フォルダ設定に合わせて
再同期する。判定は app.services.access.match_folder に一本化しており、
watcher (新規/更新ファイル取り込み時) と同じロジックで矛盾なく判定する。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.database import db
from app.services.access import match_folder
from app.services.indexer import deactivate_document
from app.services.qdrant import get_client

logger = logging.getLogger(__name__)


async def resync_access_levels() -> dict:
    """全 watch_folders × 全 documents を突き合わせて access_level を再同期する。

    - 一致フォルダが無効 → 文書を即時ソフト削除 (Qdrant is_active=False) し
      documents.status='deleted' にする (fail-closed)。物理削除は既存の
      task_processor による purge 経路に任せる。
    - 一致フォルダが有効 → そのレベル、一致なし → 既定 1。
      documents.access_level と異なる場合のみ更新し、Qdrant のペイロードにも
      同じレベルを反映する。

    戻り値: {"deleted": 無効化件数, "updated": レベル変更件数}
    """
    folders = [dict(f) for f in await db.fetchall(
        "SELECT path, access_level, is_active FROM watch_folders"
    )]
    documents = [dict(d) for d in await db.fetchall(
        "SELECT id, source_path, access_level, unclassified FROM documents "
        "WHERE status NOT IN ('deleted', 'purged')"
    )]

    deleted = 0
    updated = 0
    now = datetime.now(timezone.utc).isoformat()

    for doc in documents:
        doc_id = doc["id"]
        matched = match_folder(doc["source_path"], folders)

        if matched is not None and not matched["is_active"]:
            await deactivate_document(doc_id)
            await db.execute(
                "UPDATE documents SET status='deleted', updated_at=? WHERE id=?",
                (now, doc_id),
            )
            deleted += 1
            logger.info("Deactivated document (folder disabled): %s", doc["source_path"])
            continue

        desired_level = matched["access_level"] if matched is not None else 1
        desired_unclassified = 1 if matched is None else 0
        if matched is None:
            logger.warning(
                "Unclassified path indexed at level 1 (resync): %s", doc["source_path"]
            )

        unclassified_changed = desired_unclassified != doc.get("unclassified", 0)
        if desired_level != doc["access_level"] or unclassified_changed:
            await db.execute(
                "UPDATE documents SET access_level=?, unclassified=?, updated_at=? WHERE id=?",
                (desired_level, desired_unclassified, now, doc_id),
            )
            qdrant_payload: dict = {}
            if desired_level != doc["access_level"]:
                qdrant_payload["access_level"] = desired_level
            if unclassified_changed:
                # unclassified フラグも Qdrant のペイロードへ即時反映し、
                # フォルダ登録により分類が完了した際に検索対象へ戻す
                # (逆に未分類化された場合は検索対象から即時除外する、項目1)。
                qdrant_payload["unclassified"] = bool(desired_unclassified)
            if qdrant_payload:
                await get_client().set_payload(
                    collection_name=settings.qdrant_collection,
                    payload=qdrant_payload,
                    points=Filter(
                        must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
                    ),
                )
            updated += 1
            logger.info(
                "Updated access_level for %s: %s -> %s",
                doc["source_path"], doc["access_level"], desired_level,
            )

    return {"deleted": deleted, "updated": updated}
