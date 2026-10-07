"""契約 B の通し実行: C1 → 内容固定 → C2 → 確認 → prepare/commit。

契約 B = 承認済み仕様（spec-*）＋本人が書いた依頼文 1 枠＋選択値（言語）→ コード案。
依頼文の字数上限・空欄は orchestrator.candidate、依頼文の外部利用承認権限は
orchestrator.policy、送信手順は orchestrator.pipeline.run_contract に任せる。
ここで持つのは契約 B の入力の組み立てと C1 での経路確認だけ。

依頼文は呼び出し側が本人の入力をそのまま渡す。RAG 結果・会話履歴・社内コードを
ここで足すことはしない（設計書 送信契約: 契約 B の可変入力は本人が書いた汎用依頼文に限る）。
応答コードは表示のみで実行しない（RunResult.output_text は信頼しないテキスト）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from integration.flow import C1Router, FlowResult, check_c1_destination
from orchestrator.audit_store import AuditSink, persist_run
from orchestrator.pipeline import C2Checker, Confirmer, Gateway, RunResult, run_contract, stopped_at_c1
from orchestrator.policy import JST, Policy

CONTRACT_B = "B-csv-codegen@1"


def contract_b_input(spec_ref: str, free_text: str, options: Mapping[str, str]) -> dict[str, Any]:
    return {"contract": CONTRACT_B, "spec": spec_ref, "free_text": free_text, "options": dict(options)}


def run_contract_b(
    user: str,
    spec_ref: str,
    free_text: str,
    options: Mapping[str, str],
    destination: str,
    *,
    c2: C2Checker,
    confirmer: Confirmer,
    gateway: Gateway,
    c1: C1Router | None = None,
    now: datetime | None = None,
    policy: Policy | None = None,
    request_id: str | None = None,
    c2_timeout_s: float | None = 30.0,
    audit_sink: AuditSink | None = None,
) -> FlowResult:
    when = now or datetime.now(JST)

    def c1_stop(run: RunResult) -> RunResult:
        return persist_run(audit_sink, run, user=user, contract=CONTRACT_B, destination=destination,
                           run_at=when, bodies=[free_text], c1=decision)

    inp = contract_b_input(spec_ref, free_text, options)
    decision = None
    c1_event = None
    if c1 is not None:
        try:
            # C1 は社内ゾーンなので依頼文をそのまま見せてよい。経路候補を選ぶだけで権限は与えない
            decision = c1.route(user, free_text, {**inp, "destination_hint": destination})
        except Exception as e:  # noqa: BLE001  C1 失敗は送らない側へ倒す
            reason = f"C1 失敗: {type(e).__name__}: {e}"
            return FlowResult("held", reason, None, c1_stop(stopped_at_c1(None, reason, error=True)))
        if decision.route != "B":
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
        audit_bodies=[free_text],
    )
    return FlowResult(run.outcome, run.reason, decision, run)
