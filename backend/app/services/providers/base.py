"""プロバイダ基底: Protocol / ProviderMeta / registry。

各プロバイダ実装はモジュールトップレベルで `register(meta, factory)` を呼んで
名前で取り出せるようにする。インスタンスは初回 `get_provider(name)` で生成され
プロセス内で再利用される。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Protocol


class LLMProvider(Protocol):
    """全 LLM プロバイダが満たすべきストリーミング推論インターフェース。"""

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        ...


@dataclass(frozen=True)
class ProviderMeta:
    """プロバイダの静的メタ情報。UI セレクタ・権限判定で参照する。"""

    name: str                       # API キー: "vllm" / "anthropic" / ...
    display_name: str               # UI 表示名: "vLLM (ローカル)" / "Claude"
    kind: str                       # "api" | "agent"
    is_external: bool               # True=外部送信を伴う（警告対象）
    requires_credentials: bool      # API キー必須か
    default_model: str = ""         # 既定モデル ID。空なら settings.llm_model
    extra_fields: tuple[str, ...] = field(default_factory=tuple)  # azure 等の追加設定キー


# Factory が受け取る credentials/extra_config はプロバイダ側で解釈する
ProviderFactory = Callable[[dict | None], "LLMProvider | Awaitable[LLMProvider]"]


_registry: dict[str, tuple[ProviderMeta, ProviderFactory]] = {}
_instances: dict[str, LLMProvider] = {}


def register(meta: ProviderMeta, factory: ProviderFactory) -> None:
    if meta.name in _registry:
        raise ValueError(f"Provider already registered: {meta.name}")
    _registry[meta.name] = (meta, factory)


def list_providers() -> list[ProviderMeta]:
    return [m for m, _ in _registry.values()]


def get_meta(name: str) -> ProviderMeta:
    if name not in _registry:
        raise KeyError(f"Unknown provider: {name}")
    return _registry[name][0]


async def get_provider(name: str, credentials: dict | None = None) -> LLMProvider:
    if name in _instances:
        return _instances[name]
    if name not in _registry:
        raise KeyError(f"Unknown provider: {name}")
    _, factory = _registry[name]
    result = factory(credentials)
    instance: LLMProvider = await result if inspect.isawaitable(result) else result  # type: ignore[assignment]
    _instances[name] = instance
    return instance


def reset_instances() -> None:
    """テストや資格情報更新時に既存インスタンスを破棄する。"""
    _instances.clear()
