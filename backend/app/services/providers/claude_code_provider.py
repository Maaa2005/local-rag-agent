"""Claude Code エージェントプロバイダ (claude-agent-sdk 経由)。

Claude Code 同等のエージェントループ (read/write/edit/bash/web 全許可) を
RAG バックエンド内で動かし、検索済みコンテキストを system_prompt として渡す。

注意:
- backend コンテナに `claude` CLI が同梱されている必要がある (SDK が subprocess 起動)
- 全ツール許可なのでサーバー側でファイル編集・シェル実行が走る
  運用ではコンテナを隔離環境に置くこと
"""
from __future__ import annotations

import os
from typing import AsyncIterator

from app.services.providers.base import LLMProvider, ProviderMeta, register


class ClaudeCodeProvider:
    def __init__(self, api_key: str, model: str | None) -> None:
        self._api_key = api_key
        self._model = model

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        # 遅延 import: claude-agent-sdk 未インストール環境でも他プロバイダは動く
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        # API キーを subprocess 環境変数で渡す
        os.environ["ANTHROPIC_API_KEY"] = self._api_key

        system_parts: list[str] = []
        user_parts: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                user_parts.append(m.get("content", ""))
        system_prompt = "\n\n".join(system_parts) or None
        user_prompt = "\n\n".join(user_parts)

        options_kwargs: dict = {
            "system_prompt": system_prompt,
            "permission_mode": "bypassPermissions",  # ツール全許可 (ユーザー選択)
        }
        if self._model:
            options_kwargs["model"] = self._model
        options = ClaudeAgentOptions(**options_kwargs)

        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("Claude Code provider requires api_key in credentials")
    model = (credentials.get("extra") or {}).get("model") or None
    return ClaudeCodeProvider(credentials["api_key"], model)


register(
    ProviderMeta(
        name="claude_code",
        display_name="Claude Code エージェント",
        kind="agent",
        is_external=True,
        requires_credentials=True,
        default_model="",
        extra_fields=("model",),
    ),
    _factory,
)
