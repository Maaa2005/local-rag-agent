"""契約 A の通し実行: C1 → 内容固定 → C2 → 確認 → prepare/commit。

C1 は必須で、必ず judge.base.guarded_c1 を通す。C1 未設定・失敗・タイムアウト・不正応答・
切り捨て・宛先欠落・契約との不整合はすべて保留（held, stopped_at="C1"）にし、Gateway には何も渡さない。
送信候補の宛先は C1 の結果（契約の唯一の許可宛先）から作る。呼び出し側の destination で
補完・上書きはしない。destination を渡した場合は C1 の結果と照合し、不一致なら停止する。
destination_hint は利用者の希望という参考情報で、C1 の入力に載せるだけで宛先を変えない。
送信手順そのものは orchestrator.pipeline.run_contract に任せる。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol

from common.schemas import C1Decision
from judge.base import guarded_c1, is_guard_stop
from orchestrator.audit_store import AuditSink, persist_run
from orchestrator.pipeline import (
    AuditEvent,
    C2Checker,
    Confirmer,
    Gateway,
    RunResult,
    c1_detail,
    run_contract,
    stopped_at_c1,
)
from orchestrator.policy import JST, Policy

CONTRACT_A = "A-faq-format@1"
C1_TIMEOUT_S = 30.0
C1_MAX_INPUT_CHARS = 4000


class C1Router(Protocol):
    def route(self, user: str, request: str, input: dict[str, Any] | None) -> C1Decision: ...


@dataclass
class FlowResult:
    outcome: str  # "sent" / "held" / "rejected" / "not_routed"
    reason: str
    c1: C1Decision | None
    run: RunResult  # C1 で止まったら stopped_at="C1"（Gateway に何も渡していない）


@dataclass
class C1Gate:
    """C1 段の結果。stop が None なら続行し、destination（C1 が決めた宛先）で候補を作る。"""

    decision: C1Decision | None
    stop: FlowResult | None
    destination: str | None
    event: AuditEvent | None


def route_with_c1(
    c1: C1Router | None,
    user: str,
    request: str,
    inp: Mapping[str, Any],
    expected_route: str,
    destination: str | None,
    destination_hint: str | None,
    *,
    timeout_s: float,
    max_input_chars: int,
    stop_run: Callable[[RunResult, C1Decision | None], RunResult],
) -> C1Gate:
    """通常入口の C1 段。guarded_c1 を通し、宛先は C1 結果から決める。"""
    if c1 is None:
        reason = "C1 が設定されていない。C1 は必須のため送らずに保留する"
        return C1Gate(None, FlowResult("held", reason, None, stop_run(stopped_at_c1(None, reason, error=True), None)), None, None)
    c1_input = dict(inp)
    if destination_hint is not None:
        c1_input["destination_hint"] = destination_hint
    decision = guarded_c1(c1, user, request, c1_input, timeout_s, max_input_chars)
    if is_guard_stop(decision):
        # C1 の失敗・不正応答・切り捨て・契約と合わない判定。C1 失敗として監査し保留する
        reason = f"C1 を安全側で保留: {decision.reason}"
        run = stopped_at_c1(decision, reason, error=True)
        return C1Gate(decision, FlowResult("held", reason, decision, stop_run(run, decision)), None, None)
    if decision.route != expected_route:
        reason = f"C1 が経路 {decision.route} を選んだ: {decision.reason}"
        return C1Gate(decision, FlowResult("not_routed", reason, decision, stop_run(stopped_at_c1(decision, reason), decision)), None, None)
    dest = decision.destination
    if destination is not None and destination != dest:
        # 呼び出し側の宛先は確認内容の前提。黙って C1 側に差し替えず、hint 扱いにもせず停止する
        reason = f"C1 の宛先 {dest} と呼び出し側の宛先 {destination} が一致しない。送らずに人の判断へ回す"
        run = stopped_at_c1(decision, reason)
        run.audit.insert(1, AuditEvent("C1", "destination_mismatch", reason))
        return C1Gate(decision, FlowResult("held", reason, decision, stop_run(run, decision)), None, None)
    hint = f" | hint={destination_hint}（参考情報・宛先は不変）" if destination_hint is not None else ""
    return C1Gate(decision, None, dest, AuditEvent("C1", decision.route, f"{c1_detail(decision)} | 宛先は C1 結果{hint}"))


def default_c1() -> C1Router | None:
    try:
        from judge.rules import RuleC1Router
    except ImportError:
        return None
    return RuleC1Router()


def default_c2() -> C2Checker | None:
    try:
        from judge.rules import RuleC2Gate
    except ImportError:
        return None
    return RuleC2Gate()


def contract_a_input(explanation_ref: str, options: Mapping[str, str]) -> dict[str, Any]:
    return {"contract": CONTRACT_A, "explanation": explanation_ref, "options": dict(options)}


def run_contract_a(
    user: str,
    explanation_ref: str,
    options: Mapping[str, str],
    destination: str | None = None,
    *,
    c2: C2Checker,
    confirmer: Confirmer,
    gateway: Gateway,
    c1: C1Router | None = None,
    request_text: str = "",
    destination_hint: str | None = None,
    now: datetime | None = None,
    policy: Policy | None = None,
    request_id: str | None = None,
    c1_timeout_s: float = C1_TIMEOUT_S,
    c1_max_input_chars: int = C1_MAX_INPUT_CHARS,
    c2_timeout_s: float | None = 30.0,
    audit_sink: AuditSink | None = None,
) -> FlowResult:
    """destination は省略可。渡した場合は C1 の宛先と一致しなければ停止する（補完には使わない）。"""
    when = now or datetime.now(JST)
    inp = contract_a_input(explanation_ref, options)

    def c1_stop(run: RunResult, decision: C1Decision | None) -> RunResult:
        return persist_run(audit_sink, run, user=user, contract=CONTRACT_A,
                           destination=(decision.destination if decision else None) or destination,
                           run_at=when, bodies=[request_text], c1=decision)

    gate = route_with_c1(c1, user, request_text, inp, "A", destination, destination_hint,
                         timeout_s=c1_timeout_s, max_input_chars=c1_max_input_chars, stop_run=c1_stop)
    if gate.stop is not None:
        return gate.stop
    assert gate.destination is not None and gate.event is not None
    run = run_contract(
        user,
        inp,
        gate.destination,
        when,
        c2,
        confirmer,
        gateway,
        policy=policy,
        c2_timeout_s=c2_timeout_s,
        request_id=request_id,
        audit_sink=audit_sink,
        audit_prefix=[gate.event],
        c1_decision=gate.decision,
        audit_bodies=[request_text],
    )
    return FlowResult(run.outcome, run.reason, gate.decision, run)
