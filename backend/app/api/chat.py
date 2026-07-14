import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.security import require_password_changed
from app.database import db
from app.services.llm import stream_answer
from app.services.retriever import basename_source, get_chunk_content, retrieve
from app.services.task_processor import chat_session

logger = logging.getLogger(__name__)

# 会話履歴 (messages.content) の設計判断について (項目高2):
# 回答本文そのものは「本人がその時点で権限を持っていた会話記録」として扱い、
# 後から文書の access_level が変わっても失効させない。一方、引用元チャンクの
# 本文 (sources[].content) は複製保存せず、表示のたびに document_id から
# 権限を再チェックして解決する (redacted/legacy になりうる)。

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
    """監査ログを1件書き込む。失敗してもチャット応答自体には影響させない。

    項目高1: 回答本文 (answer) はもう保存しない。不正利用調査に必要な文字数
    (answer_chars) のみ記録する。question は不正利用調査に必要なため保存を
    継続するが、利用者がここに機密情報を書き込みうる残余リスクがある
    (設計判断として確定)。
    """
    retrieved = [
        {
            "document_id": c.get("document_id"),
            "chunk_index": c.get("chunk_index"),
            "source_file": basename_source(c.get("source_file")),
            "score": c.get("score"),
            "access_level": c.get("access_level"),
        }
        for c in chunks
    ]
    try:
        await db.execute(
            "INSERT INTO audit_logs (user_id, username, question, retrieved_chunks, "
            "answer_chars, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user.get("id"),
                user.get("username", ""),
                question,
                json.dumps(retrieved, ensure_ascii=False),
                len(answer) if answer else 0,
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

            # SSE でクライアントへ返す引用元 (この応答内でのみ使う、本文込み)。
            # source_file はフルパス開示を避けるため basename 化する (項目中7)。
            live_sources = [
                {
                    "id": i,
                    "source_file": basename_source(c.get("source_file")),
                    "content": c.get("content", ""),
                    "score": c.get("score"),
                }
                for i, c in enumerate(chunks, 1)
            ]
            # 会話履歴 (messages.sources) へ永続化する引用元は ID 参照のみとし、
            # 本文を複製保存しない (項目5)。表示時に document_id から権限再チェック
            # 付きで解決する (項目3b, get_conversation_messages 参照)。
            sources = [
                {
                    "id": i,
                    "document_id": c.get("document_id"),
                    "chunk_index": c.get("chunk_index"),
                    "source_file": basename_source(c.get("source_file")),
                    "score": c.get("score"),
                }
                for i, c in enumerate(chunks, 1)
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': live_sources}, ensure_ascii=False)}\n\n"

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
async def chat(body: ChatRequest, user: dict = Depends(require_password_changed)):
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
async def list_conversations(user: dict = Depends(require_password_changed)):
    rows = await db.fetchall(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user["id"],),
    )
    return [dict(r) for r in rows]


async def _resolve_source_for_display(
    source: dict, user_access_level: int, doc_map: dict[str, dict]
) -> dict:
    """会話履歴の引用元を表示直前に権限再チェックしてから解決する (項目3b, 5, 高2)。

    document_id が無い古い形式のレコード（本文込みで永続化されていたもの、
    または高2マイグレーションで content を剥がされた旧形式）は、本文なし・
    クリック不可の legacy な形で返す。document_id がある新形式は、
    doc_map (呼び出し元が事前に一括取得した documents の状態) を見て権限が
    なければ本文・ファイル名を伏せた redacted な形で返し、権限があれば
    Qdrant から本文をその場で解決する (会話履歴に本文を複製保存しないため)。
    """
    document_id = source.get("document_id")
    if not document_id:
        # 旧形式 (本文込み、または高2マイグレーションで content を除去済み)。
        # 本文は提供できないため legacy として扱い、クリック不可にする。
        return {
            "id": source.get("id"),
            "source_file": basename_source(source.get("source_file")),
            "score": source.get("score"),
            "content": None,
            "legacy": True,
        }

    doc = doc_map.get(document_id)
    authorized = (
        doc is not None
        and doc["status"] not in ("deleted", "purged")
        and doc["access_level"] <= user_access_level
        and not doc["unclassified"]
    )
    if not authorized:
        return {
            "id": source.get("id"),
            "redacted": True,
            "score": source.get("score"),
        }

    content = await get_chunk_content(document_id, source.get("chunk_index"))
    return {
        "id": source.get("id"),
        "source_file": basename_source(source.get("source_file")),
        "score": source.get("score"),
        "content": content if content is not None else "(本文は現在参照できません)",
    }


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int, user: dict = Depends(require_password_changed)
):
    conv = await _get_owned_conversation(conversation_id, user["id"])
    if conv is None:
        raise HTTPException(status_code=404, detail="会話が見つかりません")

    rows = await db.fetchall(
        "SELECT id, role, content, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )

    # 項目低 (N+1緩和): メッセージ横断で document_id を集約し、documents を
    # 1クエリでバッチ取得してから各 source の権限判定に使い回す。
    parsed_rows: list[tuple[dict, list[dict]]] = []
    all_doc_ids: set[str] = set()
    for r in rows:
        d = dict(r)
        try:
            raw_sources = json.loads(d["sources"])
        except (TypeError, ValueError):
            raw_sources = []
        for s in raw_sources:
            doc_id = s.get("document_id")
            if doc_id:
                all_doc_ids.add(doc_id)
        parsed_rows.append((d, raw_sources))

    doc_map: dict[str, dict] = {}
    if all_doc_ids:
        placeholders = ",".join("?" * len(all_doc_ids))
        doc_rows = await db.fetchall(
            f"SELECT id, access_level, status, unclassified FROM documents "
            f"WHERE id IN ({placeholders})",
            tuple(all_doc_ids),
        )
        doc_map = {row["id"]: dict(row) for row in doc_rows}

    results = []
    for d, raw_sources in parsed_rows:
        d["sources"] = [
            await _resolve_source_for_display(s, user["access_level"], doc_map)
            for s in raw_sources
        ]
        results.append(d)
    return results


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, user: dict = Depends(require_password_changed)):
    conv = await _get_owned_conversation(conversation_id, user["id"])
    if conv is None:
        raise HTTPException(status_code=404, detail="会話が見つかりません")

    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return {"message": "会話を削除しました"}
