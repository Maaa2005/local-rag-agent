"""Anthropic Claude API プロバイダ。"""
from __future__ import annotations

from typing import AsyncIterator

from app.config import settings
from app.services.providers.base import LLMProvider, ProviderMeta, register

_DEFAULT_MODEL = "claude-sonnet-4-6"


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic は system を独立引数で渡す。"""
    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            rest.append({"role": m["role"], "content": m.get("content", "")})
    return "\n\n".join(system_parts), rest


class AnthropicProvider:
    def __init__(self, api_key: str, model: str) -> None:
        # 遅延 import: anthropic 未インストール環境では他プロバイダだけ使える
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        system, rest = _split_system(messages)
        async with self._client.messages.stream(
            model=self._model,
            system=system or None,
            messages=rest,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("Anthropic provider requires api_key in credentials")
    model = (credentials.get("extra") or {}).get("model") or _DEFAULT_MODEL
    return AnthropicProvider(credentials["api_key"], model)


register(
    ProviderMeta(
        name="anthropic",
        display_name="Anthropic Claude",
        kind="api",
        is_external=True,
        requires_credentials=True,
        default_model=_DEFAULT_MODEL,
        extra_fields=("model",),
    ),
    _factory,
)
