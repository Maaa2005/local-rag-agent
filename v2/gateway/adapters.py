"""外部送信アダプタ。

アダプタは保存済み SendPayload をそのまま送る。例外は 3 種に分類する:
- NotSentError: 送っていないと確定できる（キー未設定・SDK 未導入など）
- RejectedError: 相手が明示的に拒否した（4xx）
- それ以外の例外: 結果不明（タイムアウト・接続断・5xx など）

例外メッセージに API キーや本文を入れない。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Protocol

from common.schemas import SendPayload
from gateway.contracts import lookup


class AdapterError(Exception):
    pass


class NotSentError(AdapterError):
    """送信していないことが確定している。"""


class RejectedError(AdapterError):
    """相手が要求を拒否した。"""


class UnknownOutcomeError(AdapterError):
    """送信結果が不明。"""


class Adapter(Protocol):
    def send(self, payload: SendPayload) -> str: ...


def read_secret(path: str | Path) -> str:
    try:
        key = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        raise NotSentError("api key file unreadable") from None
    if not key:
        raise NotSentError("api key file empty")
    return key


class FakeAdapter:
    """テスト用受信先。受け取った payload を記録する。

    mode: "ok" / "not_sent" / "rejected" / "unknown" / "timeout"
    """

    def __init__(
        self,
        reply: str = "fake-output",
        mode: str = "ok",
        delay: float = 0.0,
        on_send: Callable[[SendPayload], None] | None = None,
    ) -> None:
        self.reply = reply
        self.mode = mode
        self.delay = delay
        self.on_send = on_send
        self.received: list[SendPayload] = []
        self._lock = threading.Lock()

    def send(self, payload: SendPayload) -> str:
        with self._lock:
            self.received.append(payload)
        if self.on_send:
            self.on_send(payload)
        if self.delay:
            import time

            time.sleep(self.delay)
        if self.mode == "not_sent":
            raise NotSentError("fake not sent")
        if self.mode == "rejected":
            raise RejectedError("fake rejected")
        if self.mode == "unknown":
            raise UnknownOutcomeError("fake unknown")
        if self.mode == "timeout":
            raise TimeoutError("fake timeout")
        return self.reply


def _split_messages(payload: SendPayload) -> tuple[str | None, list[dict]]:
    """先頭の system（最大 1 件）とそれ以降を分ける。連結・並び替えはしない。

    構造は service._validate で検査済みだが、ここでも崩れていれば送らない。
    """
    msgs = list(payload.messages)
    system = None
    if msgs and msgs[0].role == "system":
        system = msgs[0].content
        msgs = msgs[1:]
    if not msgs or any(m.role != "user" for m in msgs):
        raise NotSentError("invalid message structure")
    return system, [{"role": m.role, "content": m.content} for m in msgs]


def _max_output_tokens(payload: SendPayload) -> int:
    """契約の出力上限（文字数）から出力トークン上限を導出する。"""
    spec = lookup(payload.contract)
    if spec is None:
        raise NotSentError("unknown contract")
    return spec.max_output_tokens


class ClaudeAdapter:
    """Anthropic Messages API。自動再試行なし。SDK は send 時に遅延 import。"""

    def __init__(self, model: str, key_path: str | Path = "/run/secrets/anthropic_api_key",
                 timeout: float = 60.0) -> None:
        self.model = model
        self.key_path = Path(key_path)
        self.timeout = timeout

    def send(self, payload: SendPayload) -> str:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            raise NotSentError("anthropic sdk not installed") from None
        system, msgs = _split_messages(payload)
        max_tokens = _max_output_tokens(payload)
        key = read_secret(self.key_path)
        try:
            client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=self.timeout)
        except Exception:
            raise NotSentError("client init failed") from None
        kwargs: dict = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
        if system is not None:
            kwargs["system"] = system
        try:
            resp = client.messages.create(**kwargs)
        except Exception as e:  # SDK 例外の文面にキーが入る可能性を避け、型名だけ使う
            raise _classify(e, anthropic) from None
        return "".join(getattr(b, "text", "") for b in resp.content)


class CodexAdapter:
    """OpenAI Chat Completions API。自動再試行なし。SDK は send 時に遅延 import。"""

    def __init__(self, model: str, key_path: str | Path = "/run/secrets/openai_api_key",
                 timeout: float = 60.0) -> None:
        self.model = model
        self.key_path = Path(key_path)
        self.timeout = timeout

    def send(self, payload: SendPayload) -> str:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            raise NotSentError("openai sdk not installed") from None
        system, msgs = _split_messages(payload)
        if system is not None:
            msgs = [{"role": "system", "content": system}, *msgs]
        max_tokens = _max_output_tokens(payload)
        key = read_secret(self.key_path)
        try:
            client = openai.OpenAI(api_key=key, max_retries=0, timeout=self.timeout)
        except Exception:
            raise NotSentError("client init failed") from None
        try:
            resp = client.chat.completions.create(model=self.model, messages=msgs,
                                                  max_completion_tokens=max_tokens)
        except Exception as e:
            raise _classify(e, openai) from None
        return resp.choices[0].message.content or ""


def _classify(exc: Exception, sdk) -> AdapterError:
    """SDK 例外を分類。メッセージには型名だけを入れる。"""
    name = type(exc).__name__
    status_err = getattr(sdk, "APIStatusError", None)
    timeout_err = getattr(sdk, "APITimeoutError", None)
    if timeout_err is not None and isinstance(exc, timeout_err):
        return UnknownOutcomeError(name)
    if status_err is not None and isinstance(exc, status_err):
        code = getattr(exc, "status_code", 0) or 0
        if 400 <= code < 500 and code not in (408, 409):
            return RejectedError(f"{name} status={code}")
        return UnknownOutcomeError(f"{name} status={code}")
    return UnknownOutcomeError(name)
