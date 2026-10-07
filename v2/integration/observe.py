"""送信観測（evaluation/v2/README.md の観測項目）を分けて読む。

- gateway_received: Gateway が本文を受け取ったか（監査 stage=gateway_received が成功）
- attempting_recorded: 外部呼び出しの直前に ATTEMPTING が記録されていたか
- destination_received: 受信先（アダプタ）が受け取ったか
- result_saved: 結果（SUCCEEDED/FAILED）が Gateway の記録に保存されたか
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.schemas import SendPayload, SendState
from gateway.store import SendStore
from orchestrator.pipeline import RunResult


@dataclass
class AttemptingProbe:
    """アダプタの on_send に差し込み、送信直前の記録状態を写し取る。"""

    store: SendStore
    seen: list[tuple[str, SendState]] = field(default_factory=list)

    def __call__(self, payload: SendPayload) -> None:  # noqa: ARG002
        # 呼び出し中の request_id はアダプタに渡らないため、ATTEMPTING の行を拾う
        conn = self.store._connect()  # noqa: SLF001  読み取り専用の観測
        try:
            for rid, st in conn.execute("SELECT request_id, state FROM sends WHERE state=?", (SendState.ATTEMPTING.value,)):
                self.seen.append((rid, SendState(st)))
        finally:
            conn.close()


@dataclass(frozen=True)
class SendObservation:
    gateway_received: bool
    attempting_recorded: bool
    destination_received: bool
    result_saved: bool


def _audit_ok(run: RunResult, stage: str) -> bool:
    return any(ev.stage == stage and ev.result != "error" for ev in run.audit)


def observe(
    run: RunResult,
    *,
    store: SendStore | None = None,
    probe: AttemptingProbe | None = None,
    received_before: int = 0,
    received_after: int = 0,
) -> SendObservation:
    rid = run.request_id
    rec = store.get(rid) if (store is not None and rid) else None
    return SendObservation(
        gateway_received=_audit_ok(run, "gateway_received") and (store is None or rec is not None),
        attempting_recorded=bool(probe and rid and any(r == rid for r, _ in probe.seen)),
        destination_received=received_after > received_before,
        result_saved=rec is not None and rec.state in (SendState.SUCCEEDED, SendState.FAILED),
    )
