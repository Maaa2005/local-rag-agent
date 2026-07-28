from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from app.core.security import require_password_changed
from app.services.docx_editor import edit_docx


router = APIRouter(prefix='/api/documents', tags=['documents'])
DRAFT_TTL_SECONDS = 15 * 60
MAX_ACTIVE_DRAFTS = 50


@dataclass
class Draft:
    user_id: int
    original_filename: str
    content: bytes
    changes: list[dict]
    total_replacements: int
    expires_at: float


_drafts: dict[str, Draft] = {}


class ConfirmRequest(BaseModel):
    confirmed: bool


def _cleanup() -> None:
    now = time.monotonic()
    for draft_id in [key for key, value in _drafts.items() if value.expires_at <= now]:
        _drafts.pop(draft_id, None)


def _owned_draft(draft_id: str, user: dict) -> Draft:
    _cleanup()
    draft = _drafts.get(draft_id)
    if draft is None or draft.user_id != user['id']:
        raise HTTPException(404, '編集案が存在しないか、有効期限が切れています')
    return draft


@router.post('/docx/drafts')
async def create_docx_draft(
    file: UploadFile = File(...),
    operations_json: str = Form(...),
    user: dict = Depends(require_password_changed),
):
    _cleanup()
    filename = file.filename or ''
    if not filename.lower().endswith('.docx'):
        raise HTTPException(400, '編集できる形式は.docxのみです')
    try:
        operations = json.loads(operations_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, '編集指示のJSONが不正です') from exc
    if not isinstance(operations, list) or not all(isinstance(item, dict) for item in operations):
        raise HTTPException(400, '編集指示は配列で指定してください')
    content = await file.read()
    try:
        result = edit_docx(content, operations)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if len(_drafts) >= MAX_ACTIVE_DRAFTS:
        raise HTTPException(503, '編集中の文書が多いため、しばらくしてから再度お試しください')
    draft_id = secrets.token_urlsafe(24)
    _drafts[draft_id] = Draft(
        user_id=user['id'],
        original_filename=filename,
        content=result.content,
        changes=result.changes,
        total_replacements=result.total_replacements,
        expires_at=time.monotonic() + DRAFT_TTL_SECONDS,
    )
    return {
        'draft_id': draft_id,
        'original_filename': filename,
        'total_replacements': result.total_replacements,
        'changes': result.changes,
        'expires_in_seconds': DRAFT_TTL_SECONDS,
        'notice': 'まだ保存されていません。変更内容を確認し、承諾後に保存してください。',
    }


@router.post('/docx/drafts/{draft_id}/confirm')
async def confirm_docx_draft(
    draft_id: str,
    body: ConfirmRequest,
    user: dict = Depends(require_password_changed),
):
    draft = _owned_draft(draft_id, user)
    if body.confirmed is not True:
        raise HTTPException(400, '保存には明示的な承諾が必要です')
    _drafts.pop(draft_id, None)
    return Response(
        content=draft.content,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        headers={
            'Content-Disposition': 'attachment; filename=edited_document.docx',
            'X-Document-Status': 'user-confirmed-draft',
        },
    )


@router.delete('/docx/drafts/{draft_id}', status_code=204)
async def cancel_docx_draft(
    draft_id: str,
    user: dict = Depends(require_password_changed),
):
    _owned_draft(draft_id, user)
    _drafts.pop(draft_id, None)
