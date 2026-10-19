"""自由質問の入口: C1 → local ならローカル回答（権限付き RAG）。

送信契約の入力を持たない質問を扱う。C1 は必ず judge.base.guarded_c1 を通す。
- local: orchestrator.local_answer.answer_locally（Gateway を持たないので外部へは送れない）
- reject: rejected
- それ以外（human・code・A・B・C1 失敗）: held。A/B は契約の入口（run_contract_a / run_contract_b）からしか
  送らない。code（コードで処理）は数値ガード（F7）が未実装のため保留する
"""
from __future__ import annotations

from datetime import datetime

from judge.base import guarded_c1, is_guard_stop
from integration.flow import C1_MAX_INPUT_CHARS, C1_TIMEOUT_S, C1Router
from orchestrator.audit_store import AuditSink, persist_run
from orchestrator.local_answer import Generator, LocalResult, Retriever, answer_locally
from orchestrator.pipeline import AuditEvent, c1_detail, stopped_at_c1
from orchestrator.policy import JST, Policy


def ask(
    c1: C1Router | None,
    user: str,
    question: str,
    pol: Policy,
    retriever: Retriever,
    generator: Generator,
    *,
    audit_sink: AuditSink | None = None,
    now: datetime | None = None,
    timeout_s: float = C1_TIMEOUT_S,
    max_input_chars: int = C1_MAX_INPUT_CHARS,
) -> LocalResult:
    run_at = now or datetime.now(JST)
    decision = None
    if c1 is None:
        res = LocalResult(stopped_at_c1(None, "C1 が設定されていない。C1 は必須のため保留する", error=True),
                          bodies=(question,))
    else:
        decision = guarded_c1(c1, user, question, None, timeout_s, max_input_chars)
        if is_guard_stop(decision):
            res = LocalResult(stopped_at_c1(None, f"C1 failed: {decision.reason}", error=True), bodies=(question,))
        elif decision.route == "local":
            res = answer_locally(user, question, pol, retriever, generator,
                                 audit=[AuditEvent("C1", "local", c1_detail(decision))])
        elif decision.route == "code":
            res = LocalResult(stopped_at_c1(decision, "コードで処理する経路（F7 数値ガード）は未実装のため保留"),
                              bodies=(question,))
        elif decision.route in ("A", "B"):
            res = LocalResult(stopped_at_c1(decision, "外部委託は契約の入口からのみ。自由質問からは送らない"),
                              bodies=(question,))
        else:
            res = LocalResult(stopped_at_c1(decision, decision.reason), bodies=(question,))
    persist_run(audit_sink, res.run, user=user, contract=None, destination=None, run_at=run_at,
                bodies=res.bodies, c1=decision)
    return res
