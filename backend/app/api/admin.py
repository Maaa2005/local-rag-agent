import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.security import require_admin
from app.database import db
from app.services.access_sync import resync_access_levels
from app.services.admin_events import record_admin_event

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ─── 監視フォルダ ────────────────────────────────────────────
class FolderCreate(BaseModel):
    path: str
    access_level: int = 1


class FolderUpdate(BaseModel):
    access_level: int


@router.get("/folders", dependencies=[Depends(require_admin)])
async def list_folders():
    rows = await db.fetchall("SELECT * FROM watch_folders ORDER BY created_at DESC")
    return [dict(r) for r in rows]


@router.post("/folders")
async def add_folder(body: FolderCreate, actor: dict = Depends(require_admin)):
    if body.access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level は 1, 2, 3 のいずれかです")
    existing = await db.fetchone(
        "SELECT id, is_active FROM watch_folders WHERE path=?", (body.path,)
    )
    if existing and existing["is_active"]:
        raise HTTPException(400, "そのパスは既に登録されています")

    if existing:
        # 無効化済みの同一パス → 再有効化 (新しい access_level で上書き)
        await db.execute(
            "UPDATE watch_folders SET is_active=1, access_level=? WHERE id=?",
            (body.access_level, existing["id"]),
        )
        folder_id = existing["id"]
    else:
        folder_id = await db.execute(
            "INSERT INTO watch_folders (path, access_level) VALUES (?, ?)",
            (body.path, body.access_level),
        )

    result = await resync_access_levels()
    await record_admin_event(
        actor,
        "add_folder",
        {"path": body.path, "access_level": body.access_level, "folder_id": folder_id},
    )
    return {
        "message": f"監視フォルダ {body.path} を追加しました (access_level={body.access_level})",
        **result,
    }


@router.patch("/folders/{folder_id}")
async def update_folder(folder_id: int, body: FolderUpdate, actor: dict = Depends(require_admin)):
    if body.access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level は 1, 2, 3 のいずれかです")
    row = await db.fetchone(
        "SELECT id, path FROM watch_folders WHERE id=? AND is_active=1", (folder_id,)
    )
    if not row:
        raise HTTPException(404, "フォルダが見つかりません")
    await db.execute(
        "UPDATE watch_folders SET access_level=? WHERE id=?",
        (body.access_level, folder_id),
    )
    result = await resync_access_levels()
    await record_admin_event(
        actor,
        "update_folder",
        {"path": row["path"], "access_level": body.access_level, "folder_id": folder_id},
    )
    return {
        "message": f"監視フォルダの access_level を {body.access_level} に更新しました",
        **result,
    }


@router.delete("/folders/{folder_id}")
async def remove_folder(folder_id: int, actor: dict = Depends(require_admin)):
    row = await db.fetchone("SELECT id, path, access_level FROM watch_folders WHERE id=?", (folder_id,))
    if not row:
        raise HTTPException(404, "フォルダが見つかりません")
    await db.execute("UPDATE watch_folders SET is_active=0 WHERE id=?", (folder_id,))
    result = await resync_access_levels()
    await record_admin_event(
        actor,
        "remove_folder",
        {"path": row["path"], "access_level": row["access_level"], "folder_id": folder_id},
    )
    return {"message": "監視フォルダを無効化しました", **result}


# ─── ドキュメント ─────────────────────────────────────────────
@router.get("/documents", dependencies=[Depends(require_admin)])
async def list_documents():
    """全文書のメタデータ (パス・状態等) を一覧する (管理者限定, 項目中7)。

    ここは管理者向けの運用管理画面用であり、access_level によるフィルタは
    かけない (管理者は全文書のメタデータを把握できる)。ただし本文検索・
    チャットでの引用は従来どおり retriever 側の access_level フィルタで
    制限され、この API がそれを緩めるわけではない。
    """
    rows = await db.fetchall(
        "SELECT id, source_path, file_type, status, chunk_count, access_level, created_at, updated_at "
        "FROM documents ORDER BY updated_at DESC"
    )
    return [dict(r) for r in rows]


@router.get("/tasks", dependencies=[Depends(require_admin)])
async def list_tasks():
    rows = await db.fetchall(
        "SELECT t.*, d.source_path FROM tasks t "
        "JOIN documents d ON d.id = t.document_id "
        "ORDER BY t.created_at DESC LIMIT 100"
    )
    return [dict(r) for r in rows]


@router.get("/documents/unclassified", dependencies=[Depends(require_admin)])
async def list_unclassified_documents():
    """どの監視フォルダにも一致せず fail-open で Lv1 (全員閲覧可) になっている文書。"""
    rows = await db.fetchall(
        "SELECT id, source_path, file_type, status, access_level, created_at, updated_at "
        "FROM documents WHERE unclassified=1 AND status NOT IN ('deleted', 'purged') "
        "ORDER BY updated_at DESC"
    )
    return [dict(r) for r in rows]


# ─── 監査ログ ────────────────────────────────────────────────
@router.get("/audit-logs", dependencies=[Depends(require_admin)])
async def list_audit_logs(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """回答本文 (answer) は保存していないため返さない (項目高1)。桁数のみ返す。"""
    rows = await db.fetchall(
        "SELECT id, user_id, username, question, retrieved_chunks, answer_chars, error, created_at "
        "FROM audit_logs ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["retrieved_chunks"] = json.loads(d["retrieved_chunks"])
        except (TypeError, ValueError):
            d["retrieved_chunks"] = []
        results.append(d)
    return results


# ─── 管理操作の監査ログ ──────────────────────────────────────
@router.get("/admin-events", dependencies=[Depends(require_admin)])
async def list_admin_events(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """管理者操作 (ユーザー作成・フォルダ変更・パスワード変更) の監査ログ (項目高4)。"""
    rows = await db.fetchall(
        "SELECT id, user_id, username, action, detail, created_at "
        "FROM admin_events ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["detail"] = json.loads(d["detail"]) if d["detail"] else {}
        except (TypeError, ValueError):
            d["detail"] = {}
        results.append(d)
    return results
