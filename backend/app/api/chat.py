import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.security import get_current_user
from app.database import db
from app.services.llm import stream_answer
from app.services.retriever import retrieve
from app.services.task_processor import chat_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


async def _record_audit_log(
    user: dict, question: str, chunks: list[dict], answer: str, error: str | None
) -> None:
    """監査ログを1件書き込む。失敗してもチャット応答自体には影響させない。"""
    retrieved = [
        {
            "source_file": c.get("source_file"),
            "score": c.get("score"),
            "access_level": c.get("access_level"),
        }
        for c in chunks
    ]
    try:
        await db.execute(
            "INSERT INTO audit_logs (user_id, username, question, retrieved_chunks, answer, "
            "error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user.get("id"),
                user.get("username", ""),
                question,
                json.dumps(retrieved, ensure_ascii=False),
                answer or None,
                error,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except Exception:
        logger.exception("Failed to write audit log")


async def _event_stream(question: str, user: dict):
    user_access_level = user["access_level"]
    chunks: list[dict] = []
    answer_parts: list[str] = []
    error_msg: str | None = None

    async with chat_session():
        try:
            # このアプリはローカル LLM 固定の社内機密 RAG 専用。常に社内文書を検索する。
            try:
                chunks = await retrieve(question, user_access_level)
            except Exception:
                logger.exception("Retrieval failed")
                error_msg = "検索処理でエラーが発生しました。しばらくしてから再度お試しください。"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                return

            sources = [c["source_file"] for c in chunks]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            if not chunks:
                no_hit_msg = "その質問に関連する資料が見つかりませんでした。"
                answer_parts.append(no_hit_msg)
                yield f"data: {json.dumps({'type': 'token', 'content': no_hit_msg}, ensure_ascii=False)}\n\n"
            else:
                try:
                    async for token in stream_answer(question, chunks):
                        answer_parts.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                except Exception:
                    logger.exception("LLM stream failed")
                    error_msg = "回答生成でエラーが発生しました。しばらくしてから再度お試しください。"
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"

            yield "data: {\"type\": \"done\"}\n\n"
        finally:
            # ストリーミング自体は遅延させず、完了後 (エラー時含む) にまとめて1回書き込む。
            await _record_audit_log(
                user, question, chunks, "".join(answer_parts), error_msg
            )


@router.post("")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    return StreamingResponse(
        _event_stream(body.question, user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
