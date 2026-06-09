"""AWS Bedrock プロバイダ (Converse Stream API)。

api_key フィールドは access_key_id として扱い、extra に以下を入れる:
- secret_access_key: AWS シークレット
- region: 例 "us-east-1"
- model_id: 例 "anthropic.claude-sonnet-4-20250514-v1:0"
"""
from __future__ import annotations

from typing import AsyncIterator

from app.config import settings
from app.services.providers.base import LLMProvider, ProviderMeta, register

_DEFAULT_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"


def _to_bedrock_messages(messages: list[dict]) -> tuple[list[dict] | None, list[dict]]:
    """system は別配列、残りは role/content[].text 形式へ。"""
    system: list[dict] = []
    rest: list[dict] = []
    for m in messages:
        text = m.get("content", "")
        if m.get("role") == "system":
            system.append({"text": text})
        else:
            rest.append({"role": m["role"], "content": [{"text": text}]})
    return (system or None), rest


class BedrockProvider:
    def __init__(self, access_key: str, secret_key: str, region: str, model_id: str) -> None:
        import boto3
        # bedrock-runtime は同期 API。asyncio.to_thread で逃がす。
        self._client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._model_id = model_id

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        import asyncio

        system, rest = _to_bedrock_messages(messages)

        def _invoke():
            return self._client.converse_stream(
                modelId=self._model_id,
                messages=rest,
                system=system,
                inferenceConfig={
                    "maxTokens": settings.llm_max_tokens,
                    "temperature": settings.llm_temperature,
                },
            )

        response = await asyncio.to_thread(_invoke)
        for event in response["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if delta:
                yield delta


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("Bedrock provider requires api_key (AWS access key id)")
    extra = credentials.get("extra") or {}
    secret = extra.get("secret_access_key")
    region = extra.get("region", "us-east-1")
    model_id = extra.get("model_id") or _DEFAULT_MODEL
    if not secret:
        raise ValueError("Bedrock requires extra.secret_access_key")
    return BedrockProvider(credentials["api_key"], secret, region, model_id)


register(
    ProviderMeta(
        name="bedrock",
        display_name="AWS Bedrock",
        kind="api",
        is_external=True,
        requires_credentials=True,
        default_model=_DEFAULT_MODEL,
        extra_fields=("secret_access_key", "region", "model_id"),
    ),
    _factory,
)
