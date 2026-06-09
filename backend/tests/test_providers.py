"""LLM プロバイダ層とクラウド統合機能のテスト。

- registry: 期待される 6 プロバイダがすべて登録されている
- credentials: Fernet ラウンドトリップ
- chat API: provider 指定が伝搬する、未設定プロバイダは 400
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.database as db_module
from app.services import credentials as creds_module
from app.services import llm as llm_module
from app.services.providers import base as providers_base
from app.services.providers import list_providers, get_meta


# ─── registry ─────────────────────────────────────────────────
def test_all_expected_providers_registered():
    names = {p.name for p in list_providers()}
    assert {"vllm", "anthropic", "openai", "gemini", "azure_openai", "bedrock"} <= names


def test_external_flags_are_correct():
    assert get_meta("vllm").is_external is False
    for ext in ("anthropic", "openai", "gemini", "azure_openai", "bedrock"):
        assert get_meta(ext).is_external is True
        assert get_meta(ext).requires_credentials is True


# ─── credentials encrypt/decrypt ─────────────────────────────
@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "creds.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.db._conn = None
    sql = (Path(__file__).resolve().parent.parent / "migrations" / "init.sql").read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
    asyncio.run(db_module.db.connect())
    yield db_path
    asyncio.run(db_module.db.close())


def test_credentials_roundtrip(isolated_db):
    asyncio.run(creds_module.save_credentials("anthropic", "sk-test-abc", {"model": "claude-x"}))
    loaded = asyncio.run(creds_module.load_credentials("anthropic"))
    assert loaded == {"api_key": "sk-test-abc", "extra": {"model": "claude-x"}}


def test_credentials_overwrite(isolated_db):
    asyncio.run(creds_module.save_credentials("openai", "k1", {"model": "gpt-5"}))
    asyncio.run(creds_module.save_credentials("openai", "k2", {"model": "gpt-5-mini"}))
    loaded = asyncio.run(creds_module.load_credentials("openai"))
    assert loaded["api_key"] == "k2"
    assert loaded["extra"]["model"] == "gpt-5-mini"


def test_credentials_delete(isolated_db):
    asyncio.run(creds_module.save_credentials("gemini", "k", {}))
    assert asyncio.run(creds_module.has_credentials("gemini")) is True
    asyncio.run(creds_module.delete_credentials("gemini"))
    assert asyncio.run(creds_module.has_credentials("gemini")) is False
    assert asyncio.run(creds_module.load_credentials("gemini")) is None


def test_credentials_rejects_unknown_provider(isolated_db):
    with pytest.raises(KeyError):
        asyncio.run(creds_module.save_credentials("not-a-provider", "k", {}))


# ─── stream_answer + provider injection ─────────────────────
class _FakeProvider:
    def __init__(self, tokens):
        self.tokens = tokens
        self.received = None

    async def stream(self, messages):
        self.received = messages
        for t in self.tokens:
            yield t


def test_stream_answer_uses_registered_provider(isolated_db):
    fake = _FakeProvider(["a", "b"])
    providers_base._instances["vllm"] = fake
    try:
        async def collect():
            out = []
            async for tok in llm_module.stream_answer(
                "Q", [{"content": "C", "source_file": "f"}], provider_name="vllm"
            ):
                out.append(tok)
            return out

        tokens = asyncio.run(collect())
        assert tokens == ["a", "b"]
        assert "Q" in fake.received[1]["content"]
    finally:
        providers_base.reset_instances()


def test_stream_answer_raises_when_external_credentials_missing(isolated_db):
    async def run():
        gen = llm_module.stream_answer(
            "Q", [{"content": "C", "source_file": "f"}], provider_name="anthropic"
        )
        return [tok async for tok in gen]

    with pytest.raises(ValueError, match="no credentials"):
        asyncio.run(run())


# ─── chat API: provider param ────────────────────────────────
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


def test_chat_rejects_unknown_provider(client):
    token = _login(client)
    r = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "テスト", "provider": "totally-fake"},
    )
    assert r.status_code == 400


def test_chat_rejects_external_provider_without_credentials(client):
    token = _login(client)
    r = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "テスト", "provider": "anthropic"},
    )
    assert r.status_code == 400
    assert "credentials" in r.json()["detail"].lower()


# ─── providers API ───────────────────────────────────────────
def test_list_providers_endpoint(client):
    token = _login(client)
    r = client.get("/api/providers", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    names = {p["name"] for p in body}
    assert {"vllm", "anthropic", "openai", "gemini", "azure_openai", "bedrock"} <= names
    vllm = next(p for p in body if p["name"] == "vllm")
    assert vllm["available"] is True
    assert vllm["is_external"] is False


def test_admin_can_upsert_credentials(client):
    token = _login(client)
    r = client.put(
        "/api/providers/anthropic/credentials",
        headers={"Authorization": f"Bearer {token}"},
        json={"api_key": "sk-test", "extra": {"model": "claude-sonnet-4-6"}},
    )
    assert r.status_code == 200

    r2 = client.get("/api/providers", headers={"Authorization": f"Bearer {token}"})
    anthropic = next(p for p in r2.json() if p["name"] == "anthropic")
    assert anthropic["has_credentials"] is True
    assert anthropic["available"] is True


def test_non_admin_cannot_upsert(client):
    from app.core.security import hash_password
    import app.database as db_mod
    asyncio.run(db_mod.db.execute(
        "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
        ("bob", hash_password("pw"), 1),
    ))
    token = _login(client, "bob", "pw")
    r = client.put(
        "/api/providers/anthropic/credentials",
        headers={"Authorization": f"Bearer {token}"},
        json={"api_key": "x"},
    )
    assert r.status_code == 403
