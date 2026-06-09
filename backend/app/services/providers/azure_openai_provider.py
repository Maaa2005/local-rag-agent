"""Azure OpenAI Service プロバイダ。

extra フィールド:
- endpoint: https://<resource>.openai.azure.com
- deployment: Azure 上のデプロイ名 (モデル名ではない)
- api_version: 例 "2024-12-01-preview"
"""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncAzureOpenAI

from app.config import settings
from app.services.providers.base import LLMProvider, ProviderMeta, register


class AzureOpenAIProvider:
    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("Azure OpenAI provider requires api_key in credentials")
    extra = credentials.get("extra") or {}
    endpoint = extra.get("endpoint")
    deployment = extra.get("deployment")
    api_version = extra.get("api_version", "2024-12-01-preview")
    if not endpoint or not deployment:
        raise ValueError("Azure OpenAI requires extra.endpoint and extra.deployment")
    return AzureOpenAIProvider(credentials["api_key"], endpoint, deployment, api_version)


register(
    ProviderMeta(
        name="azure_openai",
        display_name="Azure OpenAI",
        kind="api",
        is_external=True,
        requires_credentials=True,
        default_model="",
        extra_fields=("endpoint", "deployment", "api_version"),
    ),
    _factory,
)
