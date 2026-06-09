import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.security import get_current_user
from app.services.llm import stream_answer
from app.services.retriever import retrieve
from app.services.task_processor import chat_session

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


async def _event_stream(question: str, user_access_level: int):
    async with chat_session():
        chunks = await retrieve(question, user_access_level)

        # 参照ソースを最初に送信
        sources = [c["source_file"] for c in chunks]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        if not chunks:
            yield f"data: {json.dumps({'type': 'token', 'content': 'その質問に関連する資料が見つかりませんでした。'}, ensure_ascii=False)}\n\n"
        else:
            async for token in stream_answer(question, chunks):
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"


@router.post("")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    return StreamingResponse(
        _event_stream(body.question, user["access_level"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
