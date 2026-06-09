"""ローカル vLLM (OpenAI 互換 API) プロバイダ。"""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.services.providers.base import LLMProvider, ProviderMeta, register


class VLLMProvider:
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


def _factory(_credentials: dict | None) -> LLMProvider:
    return VLLMProvider(settings.vllm_base_url, settings.llm_model)


register(
    ProviderMeta(
        name="vllm",
        display_name="ローカル LLM (vLLM)",
        kind="api",
        is_external=False,
        requires_credentials=False,
        default_model=settings.llm_model,
    ),
    _factory,
)
