"""LLM 推論の公開エントリポイント。

プロバイダ自体は app.services.providers パッケージに分離。ここでは
- プロンプト整形
- DB から資格情報をロードしてプロバイダを取得
- ストリーミング応答を yield
を担う。
"""
from __future__ import annotations

from typing import AsyncIterator

from app.config import settings
from app.services.providers import LLMProvider, get_meta, get_provider, reset_instances

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


async def _resolve_provider(name: str) -> LLMProvider:
    # ローカル LLM 固定アプリのため資格情報は不要 (vLLM は requires_credentials=False)。
    meta = get_meta(name)
    if meta.requires_credentials:
        raise ValueError(f"Provider '{name}' は資格情報が必要ですが本アプリは未対応です")
    return await get_provider(name)


async def stream_answer(
    question: str, chunks: list[dict], provider_name: str | None = None
) -> AsyncIterator[str]:
    name = provider_name or settings.llm_provider
    provider = await _resolve_provider(name)
    messages = _build_messages(question, chunks)
    async for token in provider.stream(messages):
        yield token


# 後方互換: テストで provider を直差し替えするためのフック
def set_provider(provider: LLMProvider | None) -> None:
    """グローバル既定プロバイダを差し替える (None でリセット)。"""
    if provider is None:
        reset_instances()
        return
    # 既存 vLLM スロットを上書きしてしまうのが最も簡単
    from app.services.providers import base as _base
    _base._instances[settings.llm_provider] = provider


def get_provider_sync(name: str | None = None) -> LLMProvider:
    """同期版 (テスト用)。インスタンスが未生成ならエラー。"""
    from app.services.providers import base as _base
    key = name or settings.llm_provider
    if key not in _base._instances:
        raise RuntimeError(f"Provider '{key}' not initialised")
    return _base._instances[key]
