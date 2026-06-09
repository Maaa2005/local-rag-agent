import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.security import get_current_user
from app.services import credentials
from app.services.llm import stream_answer
from app.services.providers import get_meta
from app.services.retriever import retrieve
from app.services.task_processor import chat_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    provider: str | None = None  # 省略時は settings.llm_provider


async def _validate_provider(name: str, user: dict) -> None:
    """プロバイダの存在・資格情報・(外部の場合) 利用可否を検証。"""
    try:
        meta = get_meta(name)
    except KeyError:
        raise HTTPException(400, f"Unknown provider: {name}")
    if meta.requires_credentials and not await credentials.has_credentials(name):
        raise HTTPException(400, f"Provider '{name}' has no credentials configured")
    if meta.is_external:
        # 監査ログ用にユーザーとプロバイダを記録
        logger.info(
            "EXTERNAL_LLM_USAGE user=%s level=%d provider=%s",
            user["username"], user["access_level"], name,
        )


async def _event_stream(question: str, user_access_level: int, provider_name: str):
    async with chat_session():
        chunks = await retrieve(question, user_access_level)

        sources = [c["source_file"] for c in chunks]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        if not chunks:
            yield f"data: {json.dumps({'type': 'token', 'content': 'その質問に関連する資料が見つかりませんでした。'}, ensure_ascii=False)}\n\n"
        else:
            try:
                async for token in stream_answer(question, chunks, provider_name=provider_name):
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                logger.exception("LLM stream failed (provider=%s)", provider_name)
                err = f"プロバイダ {provider_name} でエラー: {exc}"
                yield f"data: {json.dumps({'type': 'error', 'content': err}, ensure_ascii=False)}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"


@router.post("")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    from app.config import settings
    provider_name = body.provider or settings.llm_provider
    await _validate_provider(provider_name, user)

    return StreamingResponse(
        _event_stream(body.question, user["access_level"], provider_name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
