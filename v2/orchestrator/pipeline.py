"""契約 A / B の送信手順。

資格 → 候補（変更不能）→ C2 → 利用者確認 → 権限再確認 → prepare → digest 照合 → commit。
拒否・失敗・切り捨て・確認未完了なら Gateway に本文を渡さない。
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Mapping, Protocol, Sequence

from common.schemas import (
    C1Decision,
    C2Verdict,
    CommitRequest,
    PrepareRequest,
    PrepareResponse,
    SendPayload,
    SendState,
    StatusResponse,
)
from orchestrator.audit_store import AuditSink, persist_run
from orchestrator.candidate import ContractViolation, SendCandidate, build_candidate
from orchestrator.policy import ApprovedText, Contract, Policy, check_eligibility, load_policy

Outcome = Literal["sent", "held", "rejected"]
StoppedAt = Literal["C1", "eligibility", "contract", "C2", "confirm", "digest", "none"]

PREPARE_TTL = timedelta(minutes=10)


class GatewayNotReached(Exception):
    """要求（本文）が Gateway に届いていないと確定できる失敗（接続不能など）。

    Gateway クライアントはこれを継承した例外を上げる。タイムアウトなど
    到達したか不明な失敗には使わない（届いたかもしれない側に倒す）。
    """


class C2Checker(Protocol):
    def check(self, payload: SendPayload) -> C2Verdict: ...


class Confirmer(Protocol):
    def confirm(self, candidate: SendCandidate) -> str | None:
        """利用者が確認した候補の digest を返す。未確認なら None。"""
        ...


class Gateway(Protocol):
    def prepare(self, req: PrepareRequest) -> PrepareResponse: ...
    def commit(self, req: CommitRequest) -> StatusResponse: ...
    def status(self, request_id: str) -> StatusResponse: ...


@dataclass(frozen=True)
class AuditEvent:
    stage: str
    result: str
    detail: str = ""


@dataclass
class RunResult:
    outcome: Outcome
    stopped_at: StoppedAt
    reason: str
    request_id: str | None = None
    digest: str | None = None
    gateway_state: SendState | None = None
    gateway_received: bool = False  # prepare に本文を渡したか
    attempted: bool = False  # commit を呼んだか
    c2_verdict: C2Verdict | None = None
    output_text: str | None = None  # 信頼しないテキスト。表示のみ
    audit: list[AuditEvent] = field(default_factory=list)
    # 応答の閲覧制約（policies/contracts.json の response_visibility）。資格検査を通った後に設定
    response_visibility: str | None = None
    visible_to: str | None = None  # 応答を取得できるのは依頼者本人だけ
    min_view_level: int | None = None  # 元資料の閲覧レベル（source_level_* のとき）
    no_exec: bool = True  # 応答は表示のみ。コードを含んでも実行しない（F6）
    # 監査の永続化（orchestrator.audit_store）。None=sink 未指定 / False=書き込み失敗（送信判断には影響しない）
    audit_persisted: bool | None = None
    audit_error: str | None = None
    audit_run_id: str | None = None

    def can_view(self, user_id: str, level: int | None = None) -> bool:
        """user_id（閲覧レベル level）がこの応答を取得してよいか。制約が不明なら見せない。"""
        if self.response_visibility is None or self.visible_to is None or user_id != self.visible_to:
            return False
        if self.min_view_level is not None and (level is None or level < self.min_view_level):
            return False
        return True


def visibility_fields(user: str, contract: Contract, approved: ApprovedText) -> dict[str, Any]:
    vis = contract.response_visibility
    return dict(
        response_visibility=vis,
        visible_to=user,
        min_view_level=approved.source_level if vis.startswith("source_level") else None,
        no_exec=True,
    )


def c1_detail(decision: C1Decision) -> str:
    return f"route={decision.route} destination={decision.destination} model={decision.model}@{decision.revision}: {decision.reason}"


def stopped_at_c1(decision: C1Decision | None, reason: str, *, error: bool = False) -> RunResult:
    """C1 で止めた結果。Gateway には何も渡していない。

    C1 が reject を選んだら rejected、それ以外の経路・C1 失敗は held。
    """
    if error or decision is None:
        audit = [AuditEvent("C1", "error", reason)]
        outcome: Outcome = "held"
    else:
        audit = [AuditEvent("C1", decision.route, c1_detail(decision))]
        outcome = "rejected" if decision.route == "reject" else "held"
    audit.append(AuditEvent("result", outcome, f"stopped_at=C1: {reason}"))
    return RunResult(outcome=outcome, stopped_at="C1", reason=reason, audit=audit)


def _call_c2(c2: C2Checker, payload: SendPayload, timeout_s: float | None) -> C2Verdict:
    if timeout_s is None:
        return c2.check(payload)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(c2.check, payload).result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as e:
        raise TimeoutError(f"C2 が {timeout_s} 秒以内に応答しない") from e
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def run_contract(
    user: str,
    input: Mapping[str, Any],
    destination: str,
    now: datetime,
    c2: C2Checker,
    confirmer: Confirmer,
    gateway: Gateway,
    *,
    policy: Policy | None = None,
    c2_timeout_s: float | None = 30.0,
    request_id: str | None = None,
    audit_sink: AuditSink | None = None,
    audit_prefix: Sequence[AuditEvent] = (),
    c1_decision: C1Decision | None = None,
    audit_bodies: Sequence[str] = (),
) -> RunResult:
    pol = policy or load_policy()
    audit: list[AuditEvent] = list(audit_prefix)
    vis: dict[str, Any] = {}
    # 監査に残さない本文（依頼文・承認済みテキスト・送信本文）。理由文に混ざっていれば除去する
    bodies: list[str] = [t for t in (input.get("free_text"), *audit_bodies) if isinstance(t, str)]

    def stop(outcome: Outcome, at: StoppedAt, reason: str, **kw: Any) -> RunResult:
        audit.append(AuditEvent("result", outcome, f"stopped_at={at}: {reason}"))
        res = RunResult(outcome=outcome, stopped_at=at, reason=reason, audit=audit, **vis, **kw)
        contract = input.get("contract")
        return persist_run(audit_sink, res, user=user, contract=contract if isinstance(contract, str) else None,
                           destination=destination, run_at=now, bodies=bodies, c1=c1_decision)

    # 1. 送信資格
    elig = check_eligibility(pol, user, input, destination, now)
    audit.append(AuditEvent("eligibility", "ok" if elig.ok else "deny", elig.reason))
    if not elig.ok:
        return stop("rejected", "eligibility", elig.reason)
    assert elig.contract is not None and elig.approved is not None
    bodies.append(elig.approved.body)
    vis.update(visibility_fields(user, elig.contract, elig.approved))

    # 2. 変更不能な送信候補
    try:
        cand = build_candidate(
            user_id=user,
            contract=elig.contract,
            approved=elig.approved,
            inp=input,
            destination=destination,
            now=now,
            request_id=request_id,
        )
    except ContractViolation as e:
        audit.append(AuditEvent("candidate", "violation", str(e)))
        return stop("rejected", "contract", str(e))
    audit.append(AuditEvent("candidate", "created", f"request_id={cand.request_id} digest={cand.digest}"))
    ids = dict(request_id=cand.request_id, digest=cand.digest)
    bodies.extend(m.content for m in cand.payload.messages)

    # 3. C2（失敗・タイムアウト・切り捨て・保留・拒否は Gateway に渡さない）
    try:
        verdict = _call_c2(c2, cand.payload, c2_timeout_s)
    except Exception as e:  # noqa: BLE001  失敗は送らない側へ倒す
        reason = f"C2 失敗: {type(e).__name__}: {e}"
        audit.append(AuditEvent("C2", "error", reason))
        return stop("held", "C2", reason, **ids)
    audit.append(AuditEvent("C2", verdict.decision, f"{verdict.reason} truncated={verdict.truncated} model={verdict.model}@{verdict.revision}"))
    if verdict.truncated:
        return stop("held", "C2", "C2 の入力が切り捨てられ全文を検査できていない", c2_verdict=verdict, **ids)
    if verdict.decision != "allow":
        return stop("held", "C2", f"C2 が {verdict.decision}: {verdict.reason}", c2_verdict=verdict, **ids)

    # 4. 利用者確認（同じ候補の digest に結び付ける）
    try:
        confirmed = confirmer.confirm(cand)
    except Exception as e:  # noqa: BLE001
        confirmed = None
        audit.append(AuditEvent("confirm", "error", f"{type(e).__name__}: {e}"))
    if not confirmed:
        audit.append(AuditEvent("confirm", "missing", "利用者の確認がない"))
        return stop("held", "confirm", "利用者が確定本文を確認・承認していない", c2_verdict=verdict, **ids)
    if confirmed != cand.digest:
        audit.append(AuditEvent("confirm", "mismatch", f"confirmed={confirmed}"))
        return stop("rejected", "digest", "確認済みの内容と送信候補が一致しない。新しい候補として C2 と確認をやり直す", c2_verdict=verdict, **ids)
    audit.append(AuditEvent("confirm", "ok", confirmed))

    # 5. 権限再確認
    elig2 = check_eligibility(pol, user, input, destination, now)
    if not elig2.ok or elig2.approved is None or elig2.approved.sha256 != cand.approved_sha256:
        reason = elig2.reason if not elig2.ok else "承認済みテキストが候補作成後に変わった"
        audit.append(AuditEvent("recheck", "deny", reason))
        return stop("rejected", "eligibility", reason, c2_verdict=verdict, **ids)
    audit.append(AuditEvent("recheck", "ok"))

    # 6. prepare → Gateway が計算した digest と確認済み digest を照合
    expires_at = (now + PREPARE_TTL).timestamp()
    try:
        prep = gateway.prepare(PrepareRequest(request_id=cand.request_id, payload=cand.payload, expires_at=expires_at))
    except Exception as e:  # noqa: BLE001
        # 未到達と確定できる例外だけ False。タイムアウト等は届いたかもしれないので True
        reached = not isinstance(e, GatewayNotReached)
        reason = f"prepare 失敗: {type(e).__name__}: {e}"
        audit.append(AuditEvent("gateway_received", "error", f"{reason} reached={'unknown' if reached else 'no'}"))
        return stop("held", "none", reason, gateway_received=reached, c2_verdict=verdict, **ids)
    audit.append(AuditEvent("gateway_received", prep.state.value, f"digest={prep.digest}"))
    if prep.request_id != cand.request_id or prep.digest != confirmed:
        reason = "Gateway が保存した内容の digest が確認済みのものと一致しない"
        return stop("rejected", "digest", reason, gateway_received=True, gateway_state=prep.state, c2_verdict=verdict, **ids)

    # 7. commit（本文は再送しない）
    try:
        st = gateway.commit(CommitRequest(request_id=cand.request_id, expected_digest=confirmed))
    except Exception as e:  # noqa: BLE001  応答を受け取れなければ状態照会
        audit.append(AuditEvent("attempted", "error", f"{type(e).__name__}: {e}"))
        try:
            st = gateway.status(cand.request_id)
        except Exception as e2:  # noqa: BLE001
            reason = f"commit の結果不明（状態照会も失敗: {type(e2).__name__}）。自動で再送しない"
            return stop("held", "none", reason, gateway_received=True, attempted=True, c2_verdict=verdict, **ids)
    audit.append(AuditEvent("attempted", st.state.value, f"failure={st.failure.value if st.failure else None}"))
    common = dict(gateway_received=True, attempted=True, gateway_state=st.state, c2_verdict=verdict, **ids)
    if st.state == SendState.SUCCEEDED:
        return stop("sent", "none", "送信完了", output_text=st.output_text, **common)
    if st.state == SendState.FAILED:
        kind = st.failure.value if st.failure else "unknown"
        outcome: Outcome = "rejected" if kind == "rejected" else "held"
        return stop(outcome, "none", f"Gateway 送信失敗: {kind}", **common)
    return stop("held", "none", f"送信結果が未確定: {st.state.value}。自動で再送しない", **common)
