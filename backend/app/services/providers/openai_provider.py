"""OpenAI 公式 API プロバイダ (GPT/o-series)。"""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.services.providers.base import LLMProvider, ProviderMeta, register

_DEFAULT_MODEL = "gpt-5"


class OpenAIProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
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


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("OpenAI provider requires api_key in credentials")
    model = (credentials.get("extra") or {}).get("model") or _DEFAULT_MODEL
    return OpenAIProvider(credentials["api_key"], model)


register(
    ProviderMeta(
        name="openai",
        display_name="OpenAI (Codex / GPT)",
        kind="api",
        is_external=True,
        requires_credentials=True,
        default_model=_DEFAULT_MODEL,
        extra_fields=("model",),
    ),
    _factory,
)
