"""自由質問の入口: C1 → local ならローカル回答（権限付き RAG）。

送信契約の入力を持たない質問を扱う。C1 は必ず judge.base.guarded_c1 を通す。
- local: orchestrator.local_answer.answer_locally（Gateway を持たないので外部へは送れない）。
  decider があれば local でも記録の検索を通す（全質問が記録を通る）。no_record ならこれまでどおりモデルが答える
- on_final（任意）: ask() の最後（生成の後・監査の保存の後）に 1 回だけ、最終の答えの記録
  （判断の記録 + final_answer・outcome・stopped_at・route_final）を渡す。モデルだけの回答も生成の失敗も残る。
  監査（本文を残さない）とは別。呼び出し側が架空データの評価でだけ使う
- reject: rejected
- それ以外（human・code・A・B・C1 失敗）: held。A/B は契約の入口（run_contract_a / run_contract_b）からしか
  送らない。code（コードで処理）は数値ガード（F7）が未実装のため保留する
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Callable

from judge.base import guarded_c1, is_guard_stop
from integration.flow import C1_MAX_INPUT_CHARS, C1_TIMEOUT_S, C1Router
from orchestrator.audit_store import AuditSink, persist_run
from orchestrator.local_answer import Decider, Generator, LocalResult, Retriever, answer_locally
from orchestrator.pipeline import AuditEvent, c1_detail, stopped_at_c1
from orchestrator.policy import JST, Policy


def route_final_of(res: LocalResult, decider_given: bool) -> str:
    """最終の答えがどの道で出来たか（plan 7.4）。code / held / model_only / model_failed / no_decider / other。"""
    kind, outcome = res.decider_kind, res.run.outcome
    if kind == "decided" and outcome == "answered":
        return "code"
    if kind == "held":
        return "held"
    if kind == "no_record":
        return "model_only" if outcome == "answered" else "model_failed"
    if not decider_given and outcome == "answered":
        return "no_decider"
    return "other"


def final_record_of(res: LocalResult, decider_given: bool) -> dict:
    """判断の記録（無ければ {}）に、最終の答え・結果・止まった段・経路を足した写し。"""
    out = copy.deepcopy(res.decision_record) if isinstance(res.decision_record, dict) else {}
    out.update(final_answer=res.run.output_text or None, outcome=res.run.outcome, stopped_at=res.run.stopped_at,
               route_final=route_final_of(res, decider_given))
    return out


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
    decider: Decider | None = None,
    on_final: Callable[[dict], None] | None = None,
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
                                 audit=[AuditEvent("C1", "local", c1_detail(decision))], decider=decider)
        elif decision.route == "code" and decider is not None:
            res = answer_locally(user, question, pol, retriever, generator,
                                 audit=[AuditEvent("C1", "code", c1_detail(decision))], decider=decider)
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
    # 判断の記録は最終の答えが出来た時に 1 回だけ渡す（モデルだけの回答・生成の失敗もここで残る）。
    # 渡す先の失敗は握りつぶさない（記録が残らなかった事を黙って流さない）
    if on_final is not None:
        on_final(final_record_of(res, decider is not None))
    return res
