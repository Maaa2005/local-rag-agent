import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.security import get_current_user
from app.services.llm import stream_answer
from app.services.retriever import retrieve
from app.services.task_processor import chat_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


async def _event_stream(question: str, user_access_level: int):
    async with chat_session():
        # このアプリはローカル LLM 固定の社内機密 RAG 専用。常に社内文書を検索する。
        chunks = await retrieve(question, user_access_level)

        sources = [c["source_file"] for c in chunks]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        if not chunks:
            yield f"data: {json.dumps({'type': 'token', 'content': 'その質問に関連する資料が見つかりませんでした。'}, ensure_ascii=False)}\n\n"
        else:
            try:
                async for token in stream_answer(question, chunks):
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                logger.exception("LLM stream failed")
                err = f"回答生成でエラーが発生しました: {exc}"
                yield f"data: {json.dumps({'type': 'error', 'content': err}, ensure_ascii=False)}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"


@router.post("")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    return StreamingResponse(
        _event_stream(body.question, user["access_level"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
