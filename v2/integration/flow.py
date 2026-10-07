"""契約 A の通し実行: C1 → 内容固定 → C2 → 確認 → prepare/commit。

C1 が A 以外を選んだら Gateway には何も渡さない。送信手順そのものは
orchestrator.pipeline.run_contract に任せる。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

from common.schemas import C1Decision
from orchestrator.pipeline import C2Checker, Confirmer, Gateway, RunResult, run_contract, stopped_at_c1
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
) -> FlowResult:
    inp = contract_a_input(explanation_ref, options)
    decision: C1Decision | None = None
    if c1 is not None:
        try:
            decision = c1.route(user, request_text, {**inp, "destination_hint": destination})
        except Exception as e:  # noqa: BLE001  C1 失敗は送らない側へ倒す
            reason = f"C1 失敗: {type(e).__name__}: {e}"
            return FlowResult("held", reason, None, stopped_at_c1(None, reason, error=True))
        if decision.route != "A":
            reason = f"C1 が経路 {decision.route} を選んだ: {decision.reason}"
            return FlowResult("not_routed", reason, decision, stopped_at_c1(decision, reason))
    run = run_contract(
        user,
        inp,
        destination,
        now or datetime.now(JST),
        c2,
        confirmer,
        gateway,
        policy=policy,
        c2_timeout_s=c2_timeout_s,
        request_id=request_id,
    )
    return FlowResult(run.outcome, run.reason, decision, run)
