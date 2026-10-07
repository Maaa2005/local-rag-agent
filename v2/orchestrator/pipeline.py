"""契約 A / B の送信手順。

資格 → 候補（変更不能）→ C2 → 利用者確認 → 権限再確認 → prepare → digest 照合 → commit。
拒否・失敗・切り捨て・確認未完了なら Gateway に本文を渡さない。
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Mapping, Protocol

from common.schemas import (
    C2Verdict,
    CommitRequest,
    PrepareRequest,
    PrepareResponse,
    SendPayload,
    SendState,
    StatusResponse,
)
from orchestrator.candidate import ContractViolation, SendCandidate, build_candidate
from orchestrator.policy import Policy, check_eligibility, load_policy

Outcome = Literal["sent", "held", "rejected"]
StoppedAt = Literal["eligibility", "contract", "C2", "confirm", "digest", "none"]

PREPARE_TTL = timedelta(minutes=10)


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
) -> RunResult:
    pol = policy or load_policy()
    audit: list[AuditEvent] = []

    def stop(outcome: Outcome, at: StoppedAt, reason: str, **kw: Any) -> RunResult:
        audit.append(AuditEvent("result", outcome, f"stopped_at={at}: {reason}"))
        return RunResult(outcome=outcome, stopped_at=at, reason=reason, audit=audit, **kw)

    # 1. 送信資格
    elig = check_eligibility(pol, user, input, destination, now)
    audit.append(AuditEvent("eligibility", "ok" if elig.ok else "deny", elig.reason))
    if not elig.ok:
        return stop("rejected", "eligibility", elig.reason)
    assert elig.contract is not None and elig.approved is not None

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
        reason = f"prepare 失敗: {type(e).__name__}: {e}"
        audit.append(AuditEvent("gateway_received", "error", reason))
        return stop("held", "none", reason, gateway_received=True, c2_verdict=verdict, **ids)
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
