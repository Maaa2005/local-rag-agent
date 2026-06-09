"""High 優先度修正の回帰テスト。

- LLM プロバイダ抽象化: get_provider / set_provider が差し替え可能で stream_answer を貫通する
- _chunk_text のバリデーション: 不正な overlap/size を弾く
"""
from __future__ import annotations

import asyncio

import pytest

from app.services import llm as llm_module
from app.services.indexer import _chunk_text


# ─── LLM プロバイダ抽象化 ─────────────────────────────────────
class _FakeProvider:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.received_messages: list[dict] | None = None

    async def stream(self, messages):
        self.received_messages = messages
        for t in self.tokens:
            yield t


@pytest.fixture(autouse=True)
def _reset_provider():
    llm_module.set_provider(None)
    yield
    llm_module.set_provider(None)


def test_get_provider_returns_singleton(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "llm_provider", "vllm")
    p1 = llm_module.get_provider()
    p2 = llm_module.get_provider()
    assert p1 is p2


def test_get_provider_raises_for_unknown(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "llm_provider", "totally-fake")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm_module.get_provider()


def test_stream_answer_uses_injected_provider():
    fake = _FakeProvider(["こんに", "ちは", "世界"])
    llm_module.set_provider(fake)

    async def collect():
        out = []
        async for tok in llm_module.stream_answer(
            "何ですか", [{"content": "本文", "source_file": "f.txt"}]
        ):
            out.append(tok)
        return out

    tokens = asyncio.run(collect())
    assert tokens == ["こんに", "ちは", "世界"]
    # system + user の 2 メッセージで、user に【参考情報】が含まれる
    assert fake.received_messages is not None
    assert fake.received_messages[0]["role"] == "system"
    assert "本文" in fake.received_messages[1]["content"]
    assert "何ですか" in fake.received_messages[1]["content"]


# ─── _chunk_text バリデーション ────────────────────────────────
def test_chunk_text_rejects_zero_size():
    with pytest.raises(ValueError, match="size"):
        _chunk_text("a" * 100, size=0, overlap=0)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError, match="overlap"):
        _chunk_text("a" * 100, size=10, overlap=-1)


def test_chunk_text_rejects_overlap_ge_size():
    """overlap >= size は無限ループ確定なので即エラー。"""
    with pytest.raises(ValueError, match="overlap"):
        _chunk_text("a" * 100, size=10, overlap=10)
    with pytest.raises(ValueError, match="overlap"):
        _chunk_text("a" * 100, size=10, overlap=50)
