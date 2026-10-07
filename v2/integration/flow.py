"""契約 A の通し実行: C1 → 内容固定 → C2 → 確認 → prepare/commit。

C1 が A 以外を選んだら Gateway には何も渡さない。送信手順そのものは
orchestrator.pipeline.run_contract に任せる。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

from common.schemas import C1Decision
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


class C1Router(Protocol):
    def route(self, user: str, request: str, input: dict[str, Any] | None) -> C1Decision: ...


@dataclass
class FlowResult:
    outcome: str  # "sent" / "held" / "rejected" / "not_routed"
    reason: str
    c1: C1Decision | None
    run: RunResult  # C1 で止まったら stopped_at="C1"（Gateway に何も渡していない）


def check_c1_destination(decision: C1Decision, destination: str) -> tuple[RunResult | None, AuditEvent]:
    """C1 の宛先と呼び出し側の宛先を照合する。

    設計書「決定事項 5」: どちら（Claude / Codex）に頼むかは C1 の振り分けで決める → 宛先の正は C1。
    ただし利用者が確認するのは呼び出し側の宛先で組み立てた候補なので、黙って C1 の宛先に
    差し替えると確認内容と送信先がずれる。不一致時の扱いは設計書に規定がないため、
    送らずに人の判断へ回す（held, stopped_at="C1"）安全側で仮実装する。
    C1 が宛先を返さない（None）場合の扱いも設計書に規定がないため、呼び出し側の指定を
    採用し、その旨を監査に残す（宛先の可否は送信資格で契約・承認済みテキストと照合される）。

    戻り値: (止めるなら RunResult / 続行なら None, 監査イベント)
    """
    if decision.destination is None:
        ev = AuditEvent("C1", decision.route, f"{c1_detail(decision)} | destination=None のため呼び出し側指定 {destination} を採用")
        return None, ev
    if decision.destination != destination:
        reason = f"C1 の宛先 {decision.destination} と呼び出し側の宛先 {destination} が一致しない。送らずに人の判断へ回す"
        run = stopped_at_c1(decision, reason)
        run.audit.insert(1, AuditEvent("C1", "destination_mismatch", reason))
        return run, run.audit[0]
    return None, AuditEvent("C1", decision.route, f"{c1_detail(decision)} | 宛先一致")


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
    destination: str,
    *,
    c2: C2Checker,
    confirmer: Confirmer,
    gateway: Gateway,
    c1: C1Router | None = None,
    request_text: str = "",
    now: datetime | None = None,
    policy: Policy | None = None,
    request_id: str | None = None,
    c2_timeout_s: float | None = 30.0,
    audit_sink: AuditSink | None = None,
) -> FlowResult:
    when = now or datetime.now(JST)

    def c1_stop(run: RunResult) -> RunResult:
        return persist_run(audit_sink, run, user=user, contract=CONTRACT_A, destination=destination,
                           run_at=when, bodies=[request_text], c1=decision)

    inp = contract_a_input(explanation_ref, options)
    decision: C1Decision | None = None
    c1_event: AuditEvent | None = None
    if c1 is not None:
        try:
            decision = c1.route(user, request_text, {**inp, "destination_hint": destination})
        except Exception as e:  # noqa: BLE001  C1 失敗は送らない側へ倒す
            reason = f"C1 失敗: {type(e).__name__}: {e}"
            return FlowResult("held", reason, None, c1_stop(stopped_at_c1(None, reason, error=True)))
        if decision.route != "A":
            reason = f"C1 が経路 {decision.route} を選んだ: {decision.reason}"
            return FlowResult("not_routed", reason, decision, c1_stop(stopped_at_c1(decision, reason)))
        held, c1_event = check_c1_destination(decision, destination)
        if held is not None:
            return FlowResult("held", held.reason, decision, c1_stop(held))
    run = run_contract(
        user,
        inp,
        destination,
        when,
        c2,
        confirmer,
        gateway,
        policy=policy,
        c2_timeout_s=c2_timeout_s,
        request_id=request_id,
        audit_sink=audit_sink,
        audit_prefix=[c1_event] if c1_event is not None else (),
        c1_decision=decision,
        audit_bodies=[request_text],
    )
    return FlowResult(run.outcome, run.reason, decision, run)
