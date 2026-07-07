import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.security import get_current_user
from app.database import db
from app.services.llm import stream_answer
from app.services.retriever import retrieve
from app.services.task_processor import chat_session

logger = logging.getLogger(__name__)

# prefix を使わず各エンドポイントでフルパスを指定する。
# /api/chat (POST) と /api/conversations 配下 (GET/DELETE) を同一ルーターに同居させるため。
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


# ─── 会話 (conversations / messages) の永続化ヘルパ ────────────────────
async def _get_owned_conversation(conversation_id: int, user_id: int) -> dict | None:
    row = await db.fetchone(
        "SELECT id, user_id, title FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    return dict(row) if row else None


async def _create_conversation(user_id: int, question: str) -> int:
    title = question.strip()[:30] or "新しい会話"
    now = datetime.now(timezone.utc).isoformat()
    return await db.execute(
        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user_id, title, now, now),
    )


async def _save_message(
    conversation_id: int, role: str, content: str, sources: list[dict]
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, json.dumps(sources, ensure_ascii=False), now),
    )
    await db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
    )


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


async def _event_stream(question: str, user: dict, conversation_id: int | None):
    user_access_level = user["access_level"]
    chunks: list[dict] = []
    sources: list[dict] = []
    answer_parts: list[str] = []
    error_msg: str | None = None
    conv_id = conversation_id

    async with chat_session():
        try:
            if conv_id is None:
                conv_id = await _create_conversation(user["id"], question)

            yield (
                "data: "
                + json.dumps({"type": "meta", "conversation_id": conv_id}, ensure_ascii=False)
                + "\n\n"
            )

            # このアプリはローカル LLM 固定の社内機密 RAG 専用。常に社内文書を検索する。
            try:
                chunks = await retrieve(question, user_access_level)
            except Exception:
                logger.exception("Retrieval failed")
                error_msg = "検索処理でエラーが発生しました。しばらくしてから再度お試しください。"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                return

            sources = [
                {
                    "id": i,
                    "source_file": c.get("source_file"),
                    "content": c.get("content", ""),
                    "score": c.get("score"),
                }
                for i, c in enumerate(chunks, 1)
            ]
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
            if conv_id is not None:
                try:
                    await _save_message(conv_id, "user", question, [])
                    await _save_message(
                        conv_id, "assistant", "".join(answer_parts), sources
                    )
                except Exception:
                    logger.exception("Failed to persist conversation messages")


@router.post("/api/chat")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    conversation_id = body.conversation_id
    if conversation_id is not None:
        conv = await _get_owned_conversation(conversation_id, user["id"])
        if conv is None:
            raise HTTPException(status_code=404, detail="会話が見つかりません")

    return StreamingResponse(
        _event_stream(body.question, user, conversation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── 会話一覧・履歴 API ────────────────────────────────────────────
@router.get("/api/conversations")
async def list_conversations(user: dict = Depends(get_current_user)):
    rows = await db.fetchall(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user["id"],),
    )
    return [dict(r) for r in rows]


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int, user: dict = Depends(get_current_user)
):
    conv = await _get_owned_conversation(conversation_id, user["id"])
    if conv is None:
        raise HTTPException(status_code=404, detail="会話が見つかりません")

    rows = await db.fetchall(
        "SELECT id, role, content, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["sources"] = json.loads(d["sources"])
        except (TypeError, ValueError):
            d["sources"] = []
        results.append(d)
    return results


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, user: dict = Depends(get_current_user)):
    conv = await _get_owned_conversation(conversation_id, user["id"])
    if conv is None:
        raise HTTPException(status_code=404, detail="会話が見つかりません")

    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return {"message": "会話を削除しました"}
