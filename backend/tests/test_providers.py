"""ローカル LLM プロバイダ層のテスト (社内機密 RAG 専用・ローカル固定)。

外部送信プロバイダ (Anthropic / OpenAI / Codex 等) は別アプリ frontier-llm-agent に
分離済み。本アプリには vLLM のみが登録され、外部経路・資格情報管理は存在しないことを検証する。
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.database as db_module
from app.services import llm as llm_module
from app.services.providers import base as providers_base
from app.services.providers import list_providers, get_meta


# ─── registry: ローカル固定 ───────────────────────────────────
def test_only_local_provider_registered():
    names = {p.name for p in list_providers()}
    assert names == {"vllm"}


def test_vllm_is_local_and_needs_no_credentials():
    meta = get_meta("vllm")
    assert meta.is_external is False
    assert meta.requires_credentials is False
    assert meta.kind == "api"


def test_no_external_providers_registered():
    """外部プロバイダ名は一切登録されていない (流出経路なし)。"""
    for ext in ("anthropic", "openai", "gemini", "azure_openai", "bedrock", "claude_code", "codex"):
        with pytest.raises(KeyError):
            get_meta(ext)


def test_get_provider_raises_for_unknown():
    from app.services.providers import get_provider
    with pytest.raises(KeyError, match="Unknown provider"):
        asyncio.run(get_provider("anthropic"))


# ─── isolated DB ─────────────────────────────────────────────
@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.db._conn = None
    sql = (Path(__file__).resolve().parent.parent / "migrations" / "init.sql").read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
    asyncio.run(db_module.db.connect())
    yield db_path
    asyncio.run(db_module.db.close())


# ─── stream_answer ───────────────────────────────────────────
class _FakeProvider:
    def __init__(self, tokens):
        self.tokens = tokens
        self.received = None

    async def stream(self, messages):
        self.received = messages
        for t in self.tokens:
            yield t


def test_stream_answer_uses_local_provider_and_injects_question():
    fake = _FakeProvider(["a", "b"])
    providers_base._instances["vllm"] = fake
    try:
        async def collect():
            return [
                t async for t in llm_module.stream_answer(
                    "Q", [{"content": "社内文書C", "source_file": "f"}]
                )
            ]

        assert asyncio.run(collect()) == ["a", "b"]
        assert "Q" in fake.received[1]["content"]
        assert "社内文書C" in fake.received[1]["content"]
    finally:
        providers_base.reset_instances()


# ─── chat / providers API ────────────────────────────────────
@pytest.fixture()
def client(isolated_db, monkeypatch):
    async def _noop(*a, **k): return None
    def _noop_sync(*a, **k): return None
    monkeypatch.setattr("app.main.ensure_collection", _noop)
    monkeypatch.setattr("app.main.start_watcher", _noop_sync)
    monkeypatch.setattr("app.main.stop_watcher", _noop_sync)

    async def _stub_task_proc():
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise
    monkeypatch.setattr("app.main.run_task_processor", _stub_task_proc)

    from app.main import app
    with TestClient(app) as c:
        yield c


def _login(client, user="admin", pw="admin"):
    r = client.post("/api/auth/token", data={"username": user, "password": pw})
    return r.json()["access_token"]


def _sse_events(raw: str) -> list[dict]:
    out = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


def test_providers_api_is_removed(client):
    """資格情報・プロバイダ選択の管理 API はローカル固定アプリには存在しない。"""
    token = _login(client)
    r = client.get("/api/providers", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404


def test_chat_uses_local_rag(client, monkeypatch):
    token = _login(client)

    async def _fake_retrieve(question, level):
        return [{"content": "就業規則の本文", "source_file": "rules.pdf"}]

    monkeypatch.setattr("app.api.chat.retrieve", _fake_retrieve)
    providers_base._instances["vllm"] = _FakeProvider(["回答です"])
    try:
        r = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "有給は何日?"},
        )
        assert r.status_code == 200
        events = _sse_events(r.text)
        sources_ev = next(e for e in events if e["type"] == "sources")
        assert sources_ev["sources"] == ["rules.pdf"]
        assert any(e.get("content") == "回答です" for e in events)
    finally:
        providers_base.reset_instances()


def test_chat_ignores_unknown_provider_field(client, monkeypatch):
    """provider フィールドを送っても無視され、常にローカル RAG で処理される。"""
    token = _login(client)

    async def _fake_retrieve(question, level):
        return [{"content": "x", "source_file": "a.txt"}]

    monkeypatch.setattr("app.api.chat.retrieve", _fake_retrieve)
    providers_base._instances["vllm"] = _FakeProvider(["ok"])
    try:
        r = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "Q", "provider": "anthropic"},
        )
        assert r.status_code == 200
        assert any(e.get("content") == "ok" for e in _sse_events(r.text))
    finally:
        providers_base.reset_instances()
