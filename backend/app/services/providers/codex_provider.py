"""OpenAI Codex エージェントプロバイダ (Codex CLI を MCP サーバーとして起動)。

`codex mcp` で stdio MCP サーバーを起動し、`codex` ツールに prompt を渡して
最終応答を一括取得する。MCP の tool 結果はストリーミングではないため、
ここでは取得した text を 1 チャンクで yield する。

extra:
- command: Codex CLI のパス (デフォルト "codex")
- args:    起動時の追加引数 (デフォルト ["mcp"])
- approval_mode: "auto" | "ask" など Codex 側の承認モード (デフォルト "auto")
"""
from __future__ import annotations

import os
from typing import AsyncIterator

from app.services.providers.base import LLMProvider, ProviderMeta, register


def _build_prompt(messages: list[dict]) -> str:
    """system + user を 1 つの prompt に直列化 (Codex は messages 形式を取らない)。"""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"[{role}]\n{m.get('content', '')}")
    return "\n\n".join(parts)


class CodexAgentProvider:
    def __init__(self, api_key: str, command: str, args: list[str], approval_mode: str) -> None:
        self._api_key = api_key
        self._command = command
        self._args = args
        self._approval_mode = approval_mode

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        # 遅延 import: mcp 未インストール環境を許容
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = self._api_key

        prompt = _build_prompt(messages)
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "codex",
                    {
                        "prompt": prompt,
                        "approval_mode": self._approval_mode,
                    },
                )
                for block in (result.content or []):
                    text = getattr(block, "text", None)
                    if text:
                        yield text


def _factory(credentials: dict | None) -> LLMProvider:
    if not credentials or not credentials.get("api_key"):
        raise ValueError("Codex provider requires api_key (OPENAI_API_KEY) in credentials")
    extra = credentials.get("extra") or {}
    command = extra.get("command") or "codex"
    args_str = extra.get("args") or "mcp"
    args = args_str.split() if isinstance(args_str, str) else list(args_str)
    approval = extra.get("approval_mode") or "auto"
    return CodexAgentProvider(credentials["api_key"], command, args, approval)


register(
    ProviderMeta(
        name="codex",
        display_name="OpenAI Codex エージェント",
        kind="agent",
        is_external=True,
        requires_credentials=True,
        default_model="",
        extra_fields=("command", "args", "approval_mode"),
    ),
    _factory,
)
