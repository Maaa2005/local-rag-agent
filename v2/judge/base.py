"""判断モデル層（C1 振り分け・C2 送信ゲート）の共通 I/F と安全側ラッパ。

設計書 N4「判定の失敗・タイムアウト・形式不正・入力切り捨ては送らない側に倒す」を
ここで一か所に集める。実装（rules / http）は Protocol を満たすだけでよく、
失敗時の扱いは guarded_c1 / guarded_c2 が必ず安全側に変換する。
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from common.schemas import C1Decision, C2Verdict, SendPayload

GUARD_MODEL = "guard"
GUARD_REVISION = "v1"


@runtime_checkable
class C1Router(Protocol):
    def route(self, user: str, request: str, input: dict[str, Any] | None) -> C1Decision: ...


@runtime_checkable
class C2Gate(Protocol):
    def check(self, payload: SendPayload) -> C2Verdict: ...


def payload_chars(payload: SendPayload) -> int:
    """C2 が検査すべき全テキスト（全メッセージ本文）の文字数。"""
    return sum(len(m.content) for m in payload.messages)


def _hold(reason: str, *, truncated: bool = False, model: str = GUARD_MODEL, revision: str = GUARD_REVISION) -> C2Verdict:
    return C2Verdict(decision="hold", reason=reason, model=model, revision=revision, truncated=truncated)


def _run_with_timeout(fn, timeout_s: float):
    # 遅延した判定スレッドは待たずに切り捨てる（結果は使わない）
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn)
        return fut.result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def guarded_c2(gate: C2Gate, payload: SendPayload, timeout_s: float, max_input_chars: int) -> C2Verdict:
    """C2 を呼び、失敗はすべて hold に倒す。allow を返すのは正常完了かつ全文検査済みのときだけ。"""
    n = payload_chars(payload)
    if n > max_input_chars:
        # 全文を検査できないので判定モデルに渡す前に保留する（設計書: 入力切り捨て→保留）
        return _hold(f"input too long: {n} > {max_input_chars} chars", truncated=True)
    try:
        raw = _run_with_timeout(lambda: gate.check(payload), timeout_s)
    except cf.TimeoutError:
        return _hold(f"C2 timeout after {timeout_s}s")
    except Exception as e:  # noqa: BLE001  判定の失敗はすべて hold
        return _hold(f"C2 error: {type(e).__name__}: {e}")
    try:
        verdict = raw if isinstance(raw, C2Verdict) else C2Verdict.model_validate(raw)
    except (ValidationError, TypeError) as e:
        return _hold(f"C2 malformed verdict: {type(e).__name__}")
    if verdict.truncated and verdict.decision == "allow":
        return _hold("C2 reported truncated input", truncated=True, model=verdict.model, revision=verdict.revision)
    return verdict


def _human(reason: str) -> C1Decision:
    return C1Decision(route="human", destination=None, reason=reason, model=GUARD_MODEL, revision=GUARD_REVISION)


def guarded_c1(
    router: C1Router, user: str, request: str, input: dict[str, Any] | None, timeout_s: float, max_input_chars: int
) -> C1Decision:
    """C1 を呼び、失敗は route='human'（人に回す・外部へ送らない）に倒す。"""
    n = len(request) + (len(str(input)) if input else 0)
    if n > max_input_chars:
        return _human(f"input too long: {n} > {max_input_chars} chars")
    try:
        raw = _run_with_timeout(lambda: router.route(user, request, input), timeout_s)
    except cf.TimeoutError:
        return _human(f"C1 timeout after {timeout_s}s")
    except Exception as e:  # noqa: BLE001
        return _human(f"C1 error: {type(e).__name__}: {e}")
    try:
        dec = raw if isinstance(raw, C1Decision) else C1Decision.model_validate(raw)
    except (ValidationError, TypeError) as e:
        return _human(f"C1 malformed decision: {type(e).__name__}")
    if dec.route in ("A", "B") and dec.destination is None:
        # 外部経路なのに宛先が無いのは形式不正扱い
        return _human("C1 external route without destination")
    return dec
