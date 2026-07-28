"""LLM プロバイダパッケージ (ローカル専用)。

このアプリは社内機密 RAG 専用でローカル LLM (vLLM) に**固定**されている。
外部送信を伴うフロンティアプロバイダ (Anthropic / OpenAI / Codex 等) は社内文書の
流出経路となるため本アプリには含めない。それらは別アプリ frontier-llm-agent に分離済み。
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

# ローカル LLM のみ registry に登録する
from app.services.providers import vllm_provider  # noqa: F401

__all__ = [
    "LLMProvider",
    "ProviderMeta",
    "list_providers",
    "get_meta",
    "get_provider",
    "register",
    "reset_instances",
]
