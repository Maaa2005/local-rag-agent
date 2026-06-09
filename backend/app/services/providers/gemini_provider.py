"""Google Gemini API プロバイダ (google-genai SDK)。"""
from __future__ import annotations

from typing import AsyncIterator

from app.config import settings
from app.services.providers.base import LLMProvider, ProviderMeta, register

_DEFAULT_MODEL = "gemini-2.5-pro"


def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    """system は system_instruction に分離。残りを role/parts 形式へ。"""
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = m.get("content", "")
        if role == "system":
            system_parts.append(text)
            continue
        # Gemini: user/model のみ受け付ける
        gemini_role = "user" if role == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    return "\n\n".join(system_parts), contents


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        from google.genai import types

        system, contents = _to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            if getattr(chunk, "text", None):
                yield chunk.text


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("Gemini provider requires api_key in credentials")
    model = (credentials.get("extra") or {}).get("model") or _DEFAULT_MODEL
    return GeminiProvider(credentials["api_key"], model)


register(
    ProviderMeta(
        name="gemini",
        display_name="Google Gemini",
        kind="api",
        is_external=True,
        requires_credentials=True,
        default_model=_DEFAULT_MODEL,
        extra_fields=("model",),
    ),
    _factory,
)
