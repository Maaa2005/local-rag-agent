"""契約 B の通し実行: C1 → 内容固定 → C2 → 確認 → prepare/commit。

契約 B = 承認済み仕様（spec-*）＋本人が書いた依頼文 1 枠＋選択値（言語）→ コード案。
依頼文の字数上限・空欄は orchestrator.candidate、依頼文の外部利用承認権限は
orchestrator.policy、送信手順は orchestrator.pipeline.run_contract に任せる。
ここで持つのは契約 B の入力の組み立てと C1 での経路確認だけ。
C1 は必須で guarded_c1 を通し、宛先は C1 結果（契約の唯一の許可宛先）から作る（integration.flow.route_with_c1）。

依頼文は呼び出し側が本人の入力をそのまま渡す。RAG 結果・会話履歴・社内コードを
ここで足すことはしない（設計書 送信契約: 契約 B の可変入力は本人が書いた汎用依頼文に限る）。
応答コードは表示のみで実行しない（RunResult.output_text は信頼しないテキスト）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from common.schemas import C1Decision
from integration.flow import C1_MAX_INPUT_CHARS, C1_TIMEOUT_S, C1Router, FlowResult, route_with_c1
from orchestrator.audit_store import AuditSink, persist_run
from orchestrator.pipeline import C2Checker, Confirmer, Gateway, RunResult, run_contract
from orchestrator.policy import JST, Policy

CONTRACT_B = "B-csv-codegen@1"


def contract_b_input(spec_ref: str, free_text: str, options: Mapping[str, str]) -> dict[str, Any]:
    return {"contract": CONTRACT_B, "spec": spec_ref, "free_text": free_text, "options": dict(options)}


def run_contract_b(
    user: str,
    spec_ref: str,
    free_text: str,
    options: Mapping[str, str],
    destination: str | None = None,
    *,
    c2: C2Checker,
    confirmer: Confirmer,
    gateway: Gateway,
    c1: C1Router | None = None,
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
    inp = contract_b_input(spec_ref, free_text, options)

    def c1_stop(run: RunResult, decision: C1Decision | None) -> RunResult:
        return persist_run(audit_sink, run, user=user, contract=CONTRACT_B,
                           destination=(decision.destination if decision else None) or destination,
                           run_at=when, bodies=[free_text], c1=decision)

    # C1 は社内ゾーンなので依頼文をそのまま見せてよい。経路と契約の宛先を返すだけで権限は与えない
    gate = route_with_c1(c1, user, free_text, inp, "B", destination, destination_hint,
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
        audit_bodies=[free_text],
    )
    return FlowResult(run.outcome, run.reason, gate.decision, run)
