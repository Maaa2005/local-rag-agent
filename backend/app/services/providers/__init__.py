"""LLM プロバイダパッケージ。

統一インターフェース `LLMProvider` を満たす実装を registry に登録し、
名前で取得できるようにする。
"""
from app.services.providers.base import (
    LLMProvider,
    ProviderMeta,
    list_providers,
    get_meta,
    get_provider,
    register,
    reset_instances,
)

# 実装をインポートして registry に登録する
from app.services.providers import (  # noqa: F401
    anthropic_provider,
    azure_openai_provider,
    bedrock_provider,
    claude_code_provider,
    codex_provider,
    gemini_provider,
    openai_provider,
    vllm_provider,
)

__all__ = [
    "LLMProvider",
    "ProviderMeta",
    "list_providers",
    "get_meta",
    "get_provider",
    "register",
    "reset_instances",
]
