"""LLM プロバイダ抽象化レイヤ。

要件定義§2.3「Gemma 4 / Qwen / Llama を即時切替可能にする共通推論インターフェース」を担保する。
公開 API である `stream_answer()` は内部で settings.llm_provider に応じたプロバイダを呼び出す。
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol

from openai import AsyncOpenAI

from app.config import settings

SYSTEM_PROMPT = """あなたは社内情報専門のアシスタントです。
以下の【参考情報】のみを根拠として質問に回答してください。
参考情報に含まれていない情報については「その情報は現在の資料には含まれていません」と回答してください。
回答は必ず日本語で行ってください。"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source_file", "不明")
        parts.append(f"[{i}] {source}\n{c['content']}")
    return "\n\n".join(parts)


def _build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = _build_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考情報】\n{context}\n\n【質問】\n{question}"},
    ]


class LLMProvider(Protocol):
    """全 LLM プロバイダが満たすべきストリーミング推論インターフェース。"""

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        ...


class VLLMProvider:
    """OpenAI 互換 API (vLLM, TGI など) 向け実装。"""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key="NONE")
        self._model = model

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """settings.llm_provider に応じたシングルトンプロバイダを返す。"""
    global _provider
    if _provider is not None:
        return _provider

    name = settings.llm_provider.lower()
    if name in ("vllm", "openai"):
        _provider = VLLMProvider(settings.vllm_base_url, settings.llm_model)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
    return _provider


def set_provider(provider: LLMProvider | None) -> None:
    """テスト用: プロバイダを差し替え (None でリセット)。"""
    global _provider
    _provider = provider


async def stream_answer(question: str, chunks: list[dict]) -> AsyncIterator[str]:
    messages = _build_messages(question, chunks)
    async for token in get_provider().stream(messages):
        yield token
