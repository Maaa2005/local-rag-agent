"""Phase 3 (チャットUX強化) のテスト。

- 引用アノテーション: sources イベントがオブジェクト配列 ({id, source_file, content, score})
- llm.SYSTEM_PROMPT: 引用指示文言
- 会話履歴の永続化: meta イベント・conversations/messages テーブルへの保存
- GET /api/conversations, GET /api/conversations/{id}/messages, DELETE /api/conversations/{id}
- 所有者チェック (他ユーザーの会話は 404)
"""
from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app.database as db_module
from app.database import db
from app.core.security import hash_password


# ─── llm: 引用指示文言 ────────────────────────────────────────────
def test_system_prompt_instructs_citation():
    from app.services.llm import SYSTEM_PROMPT

    assert "[1]" in SYSTEM_PROMPT
    assert "引用" in SYSTEM_PROMPT


# ─── _event_stream: sources オブジェクト形式・meta イベント ────────────
@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.db._conn = None
    yield db_path
    if db_module.db._conn is not None:
        asyncio.run(db_module.db.close())


def _sse_events(raw_lines: list[str]) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in raw_lines if line.startswith("data: ")]


def _parse_stream(raw_events: list[str]) -> list[dict]:
    """_event_stream が yield する生の 'data: {...}\\n\\n' 文字列を dict へ変換する。"""
    return [json.loads(e[len("data: "):].strip()) for e in raw_events]


def test_event_stream_emits_meta_then_object_sources(monkeypatch, isolated_db):
    from app.api import chat as chat_module

    async def fake_retrieve(question, access_level):
        return [
            {"content": "本文A", "source_file": "/watched/a.txt", "score": 0.9, "access_level": 1},
            {"content": "本文B", "source_file": "/watched/b.txt", "score": 0.7, "access_level": 1},
        ]

    async def fake_stream_answer(question, chunks, provider_name=None):
        for tok in ["回答", "[1]", "です"]:
            yield tok

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    async def _run():
        await db.connect()
        try:
            user = {"id": 1, "username": "admin", "access_level": 3}
            events = [e async for e in chat_module._event_stream("質問です", user, None)]
            return events
        finally:
            await db.close()

    events = _parse_stream(asyncio.run(_run()))

    assert events[0]["type"] == "meta"
    conv_id = events[0]["conversation_id"]
    assert isinstance(conv_id, int)

    sources_ev = next(e for e in events if e["type"] == "sources")
    # source_file はフルパス開示を避けるため basename 化される (項目中7)。
    assert sources_ev["sources"] == [
        {"id": 1, "source_file": "a.txt", "content": "本文A", "score": 0.9},
        {"id": 2, "source_file": "b.txt", "content": "本文B", "score": 0.7},
    ]


def test_event_stream_persists_messages(monkeypatch, isolated_db):
    from app.api import chat as chat_module

    async def fake_retrieve(question, access_level):
        return [{"content": "本文A", "source_file": "/watched/a.txt", "score": 0.9, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        for tok in ["回答です"]:
            yield tok

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    async def _run():
        await db.connect()
        try:
            user = {"id": 1, "username": "admin", "access_level": 3}
            events = _parse_stream([e async for e in chat_module._event_stream("質問です", user, None)])
            conv_id = events[0]["conversation_id"]
            rows = await db.fetchall(
                "SELECT role, content, sources FROM messages WHERE conversation_id=? ORDER BY id",
                (conv_id,),
            )
            conv_row = await db.fetchone(
                "SELECT title FROM conversations WHERE id=?", (conv_id,)
            )
            return rows, conv_row
        finally:
            await db.close()

    rows, conv_row = asyncio.run(_run())
    assert len(rows) == 2
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "質問です"
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content"] == "回答です"
    assert json.loads(rows[1]["sources"])[0]["source_file"] == "a.txt"
    assert conv_row["title"] == "質問です"


def test_event_stream_reuses_existing_conversation(monkeypatch, isolated_db):
    from app.api import chat as chat_module

    async def fake_retrieve(question, access_level):
        return [{"content": "本文A", "source_file": "/watched/a.txt", "score": 0.9, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        yield "ok"

    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    async def _run():
        await db.connect()
        try:
            user = {"id": 1, "username": "admin", "access_level": 3}
            conv_id = await chat_module._create_conversation(1, "既存の会話")
            events = _parse_stream(
                [e async for e in chat_module._event_stream("2件目の質問", user, conv_id)]
            )
            count_row = await db.fetchone("SELECT COUNT(*) AS c FROM conversations")
            return events, count_row["c"]
        finally:
            await db.close()

    events, conv_count = asyncio.run(_run())
    assert events[0]["conversation_id"] == 1
    assert conv_count == 1  # 新規作成されていない


# ─── /api/chat, /api/conversations の HTTP レベルテスト ────────────
@pytest.fixture()
def client(isolated_db, monkeypatch):
    async def _noop(*args, **kwargs):
        return None

    def _noop_sync(*args, **kwargs):
        return None

    monkeypatch.setattr("app.main.ensure_collection", _noop)
    monkeypatch.setattr("app.main.start_watcher", _noop_sync)
    monkeypatch.setattr("app.main.stop_watcher", _noop_sync)

    async def _run_task_processor_stub():
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr("app.main.run_task_processor", _run_task_processor_stub)

    from app.main import app

    with TestClient(app) as c:
        yield c


def _login(client, username="admin", password="admin"):
    r = client.post("/api/auth/token", data={"username": username, "password": password})
    token = r.json()["access_token"]
    if username == "admin" and password == "admin":
        # デフォルト資格情報 (admin/admin) のままだと must_change_password=1 のため
        # /api/chat 等が 403 になる (項目6)。テストの前提として一度だけ変更する。
        client.post(
            "/api/auth/password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "admin", "new_password": "adminpw12345"},
        )
        # 項目高3: パスワード変更で変更前に発行したトークンの iat が失効しうるため、
        # 変更後の状態で確実に使えるトークンを取り直す。
        r2 = client.post(
            "/api/auth/token", data={"username": username, "password": "adminpw12345"}
        )
        token = r2.json()["access_token"]
    return token


def _create_second_user(isolated_db, username="bob", password="bobpw"):
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
            (username, hash_password(password), 1),
        )
        conn.commit()


def test_chat_post_creates_conversation_and_returns_meta(client, monkeypatch):
    token = _login(client)

    async def fake_retrieve(question, level):
        return [{"content": "本文", "source_file": "a.txt", "score": 0.5, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        yield "回答"

    monkeypatch.setattr("app.api.chat.retrieve", fake_retrieve)
    monkeypatch.setattr("app.api.chat.stream_answer", fake_stream_answer)

    r = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "テスト質問"},
    )
    assert r.status_code == 200
    events = _sse_events(r.text.splitlines())
    assert events[0]["type"] == "meta"
    conv_id = events[0]["conversation_id"]

    r2 = client.get(
        f"/api/conversations/{conv_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 200
    msgs = r2.json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_chat_post_with_unknown_conversation_id_returns_404(client):
    token = _login(client)
    r = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "テスト質問", "conversation_id": 9999},
    )
    assert r.status_code == 404


def test_list_conversations_returns_only_own_newest_first(client, isolated_db, monkeypatch):
    token = _login(client)
    _create_second_user(isolated_db)
    token_bob = _login(client, "bob", "bobpw")

    async def fake_retrieve(question, level):
        return [{"content": "本文", "source_file": "a.txt", "score": 0.5, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        yield "ok"

    monkeypatch.setattr("app.api.chat.retrieve", fake_retrieve)
    monkeypatch.setattr("app.api.chat.stream_answer", fake_stream_answer)

    client.post("/api/chat", headers={"Authorization": f"Bearer {token}"}, json={"question": "admin質問1"})
    client.post("/api/chat", headers={"Authorization": f"Bearer {token}"}, json={"question": "admin質問2"})
    client.post("/api/chat", headers={"Authorization": f"Bearer {token_bob}"}, json={"question": "bobの質問"})

    r = client.get("/api/conversations", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["title"] == "admin質問2"  # 新しい順
    assert all("bob" not in c["title"] for c in body)


def test_conversation_messages_owner_check_returns_404_for_other_user(client, isolated_db, monkeypatch):
    token = _login(client)
    _create_second_user(isolated_db)
    token_bob = _login(client, "bob", "bobpw")

    async def fake_retrieve(question, level):
        return [{"content": "本文", "source_file": "a.txt", "score": 0.5, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        yield "ok"

    monkeypatch.setattr("app.api.chat.retrieve", fake_retrieve)
    monkeypatch.setattr("app.api.chat.stream_answer", fake_stream_answer)

    r = client.post("/api/chat", headers={"Authorization": f"Bearer {token}"}, json={"question": "秘密の質問"})
    conv_id = _sse_events(r.text.splitlines())[0]["conversation_id"]

    r2 = client.get(
        f"/api/conversations/{conv_id}/messages",
        headers={"Authorization": f"Bearer {token_bob}"},
    )
    assert r2.status_code == 404


def test_delete_conversation_owner_check_and_success(client, isolated_db, monkeypatch):
    token = _login(client)
    _create_second_user(isolated_db)
    token_bob = _login(client, "bob", "bobpw")

    async def fake_retrieve(question, level):
        return [{"content": "本文", "source_file": "a.txt", "score": 0.5, "access_level": 1}]

    async def fake_stream_answer(question, chunks, provider_name=None):
        yield "ok"

    monkeypatch.setattr("app.api.chat.retrieve", fake_retrieve)
    monkeypatch.setattr("app.api.chat.stream_answer", fake_stream_answer)

    r = client.post("/api/chat", headers={"Authorization": f"Bearer {token}"}, json={"question": "削除対象"})
    conv_id = _sse_events(r.text.splitlines())[0]["conversation_id"]

    # 他人は削除できない (404)
    r_forbidden = client.delete(
        f"/api/conversations/{conv_id}", headers={"Authorization": f"Bearer {token_bob}"}
    )
    assert r_forbidden.status_code == 404

    # 本人は削除できる
    r_ok = client.delete(
        f"/api/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert r_ok.status_code == 200

    r_list = client.get("/api/conversations", headers={"Authorization": f"Bearer {token}"})
    assert all(c["id"] != conv_id for c in r_list.json())


def test_delete_unknown_conversation_returns_404(client):
    token = _login(client)
    r = client.delete("/api/conversations/9999", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404


def test_resolve_source_for_display_legacy_without_document_id():
    """項目高2: document_id を持たない旧形式ソース (高2マイグレーションで
    content を剥がされたもの、または移行漏れのもの) は本文を返さず、
    legacy=True かつ content=None のクリック不可な形で返す。"""
    from app.api.chat import _resolve_source_for_display

    source = {
        "id": "abc123",
        "source_file": "/watched/legacy/old.txt",
        "score": 0.42,
    }

    result = asyncio.run(_resolve_source_for_display(source, user_access_level=3, doc_map={}))

    assert result == {
        "id": "abc123",
        "source_file": "old.txt",
        "score": 0.42,
        "content": None,
        "legacy": True,
    }
