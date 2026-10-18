"""判定仕様 F / R / T で C2 を聞く送信ゲート（docs/v2-c2-eval-plan.md 2 章・6 章）。

仕様（prompt・context・閾値）は judge.c2_specs.SPECS だけから作る。ゲートは閾値を引数で受けない
（受けると仕様の版と実際の判定がずれるため）。

判定は送信内容（SendPayload）と出所区分つきの部分（SendCandidate.rendered）の組で行う。
- 区分から復元した内容が送信内容と一致しない → モデルを呼ばず hold（provenance mismatch）
- 判定対象（依頼文）がない → モデルを呼ばず hold（no target）。評価では has_target で先に分け、
  「対象なし」として合格にも分母にも数えない
- 部分の数のずれ・基準のない契約 → モデルを呼ばず hold
- 通信・形式不正・タイムアウトの例外はそのまま投げ、judge.base.call_c2 / guarded_c2 が hold に倒す

C2Gate の I/F（check(payload)）に合わせるには Bound(gate, rendered) を使う。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from common.provenance import RenderedMessage, mismatch
from common.schemas import SendPayload
from judge.base import C2Verdict, _hold
from judge.c2_specs import SPECS, NoTarget, SpecError, build_request, has_target, spec_version
from judge.http_adapter import _SystemOneClient

__all__ = ["SpecC2Gate", "Bound", "has_target"]


class SpecC2Gate:
    def __init__(self, base_url: str, model: str, method: str, **kw: Any):
        if method not in SPECS:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.label = SPECS[method]["label"]
        self.question_version = spec_version(method)
        th = SPECS[method]["thresholds"]
        self.block_at, self.allow_at = th["block_at"], th["allow_at"]
        self._c = _SystemOneClient(base_url, model, **kw)

    def _hold(self, why: str) -> C2Verdict:
        return _hold(f"{self.label} {why}", question_version=self.question_version)

    def check_rendered(self, payload: SendPayload, rendered: Sequence[RenderedMessage]) -> C2Verdict:
        why = mismatch(rendered, payload.messages)
        if why is not None:
            return self._hold(f"provenance mismatch: {why}")
        try:
            req = build_request(self.method, payload.contract, rendered)
        except NoTarget:
            return self._hold("no target")
        except SpecError as e:
            return self._hold(f"spec: {e}")
        body, parsed = self._c.ask(req.context, [dict(req.question)])
        p = parsed[req.question["id"]][1]["yes"]
        truncated = bool(body.get("truncated", False))
        if truncated:
            decision = "hold"
        elif p >= self.block_at:
            decision = "block"
        elif p <= self.allow_at:
            decision = "allow"
        else:
            decision = "hold"
        return C2Verdict(decision=decision, reason=f"{self.label} p_yes={p:.3f}", model=self._c.model,
                         revision=body["revision"], truncated=truncated, prob_block=p,
                         question_version=self.question_version, context_sha256=req.context_sha256)


class RenderedGate(Protocol):
    def check_rendered(self, payload: SendPayload, rendered: Sequence[RenderedMessage]) -> C2Verdict: ...


@dataclass(frozen=True)
class Bound:
    """出所区分つきの部分を束ねて C2Gate（check(payload)）として渡すための包み。"""

    gate: RenderedGate
    rendered: tuple[RenderedMessage, ...]

    def check(self, payload: SendPayload) -> C2Verdict:
        return self.gate.check_rendered(payload, self.rendered)
