from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import get_current_user, require_admin
from app.database import db
from app.services.access_sync import resync_access_levels

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


@router.post("/folders", dependencies=[Depends(require_admin)])
async def add_folder(body: FolderCreate):
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
    else:
        await db.execute(
            "INSERT INTO watch_folders (path, access_level) VALUES (?, ?)",
            (body.path, body.access_level),
        )

    result = await resync_access_levels()
    return {
        "message": f"監視フォルダ {body.path} を追加しました (access_level={body.access_level})",
        **result,
    }


@router.patch("/folders/{folder_id}", dependencies=[Depends(require_admin)])
async def update_folder(folder_id: int, body: FolderUpdate):
    if body.access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level は 1, 2, 3 のいずれかです")
    row = await db.fetchone(
        "SELECT id FROM watch_folders WHERE id=? AND is_active=1", (folder_id,)
    )
    if not row:
        raise HTTPException(404, "フォルダが見つかりません")
    await db.execute(
        "UPDATE watch_folders SET access_level=? WHERE id=?",
        (body.access_level, folder_id),
    )
    result = await resync_access_levels()
    return {
        "message": f"監視フォルダの access_level を {body.access_level} に更新しました",
        **result,
    }


@router.delete("/folders/{folder_id}", dependencies=[Depends(require_admin)])
async def remove_folder(folder_id: int):
    row = await db.fetchone("SELECT id FROM watch_folders WHERE id=?", (folder_id,))
    if not row:
        raise HTTPException(404, "フォルダが見つかりません")
    await db.execute("UPDATE watch_folders SET is_active=0 WHERE id=?", (folder_id,))
    result = await resync_access_levels()
    return {"message": "監視フォルダを無効化しました", **result}


# ─── ドキュメント ─────────────────────────────────────────────
@router.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    rows = await db.fetchall(
        "SELECT id, source_path, file_type, status, chunk_count, access_level, created_at, updated_at "
        "FROM documents WHERE access_level <= ? ORDER BY updated_at DESC",
        (user["access_level"],),
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
