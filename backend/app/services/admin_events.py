"""システム管理者操作の監査記録 (項目高4)。

admin による自己昇格 (access_level=3 のユーザーを自ら作成する等) の経路自体は
仕様上防いでいない。代わりに操作を記録し、事後に検知できるようにする。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.database import db

logger = logging.getLogger(__name__)


async def record_admin_event(user: dict, action: str, detail: dict) -> None:
    """管理操作を1件記録する。失敗してもAPI自体は失敗させない。"""
    try:
        await db.execute(
            "INSERT INTO admin_events (user_id, username, action, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                user.get("id"),
                user.get("username", ""),
                action,
                json.dumps(detail, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except Exception:
        logger.exception("Failed to record admin event")
