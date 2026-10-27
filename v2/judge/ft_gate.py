"""F と T を組み合わせた C2 の候補 C2-FT@1（docs/v2-c2-eval-plan.md 11 章）。

- 契約 A: F だけで判定する（T は依頼文がなく対象なし）
- 契約 B: F、T の順に 1 回ずつ呼ぶ。途中で打ち切らない。スコアは s = max(p_F, p_T)
- s <= 0.1 は allow、0.1 < s < 0.5 は内容による保留、s >= 0.5 は遮断
- どちらかの部品がタイムアウト・切り捨て・応答の不正・欠落・区分の不一致なら故障による保留。
  F が allow でも T が故障なら保留。再試行しない
- 版は FT_SPEC（部品の hash・合成の規則・対象の選び方・閾値・故障の扱い）を正規化した JSON の sha256

部品ごとのスコア・判定・revision・context の sha256・故障の種類は FTVerdict.components に残す（本文は残さない）。
C2 全体の時間の上限（60 秒）は呼び出し側の guarded_c2 が掛ける。部品の例外はここで故障に倒すので投げない。
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import httpx
from pydantic import ValidationError

from common.provenance import RenderedMessage
from common.schemas import SendPayload
from judge.base import GUARD_MODEL, GUARD_REVISION, C2Verdict
from judge.c2_specs import A_KEY, B_KEY, _normalized, sha256_text, spec_hash
from judge.http_adapter import MalformedResponse
from judge.spec_gate import SpecC2Gate

__all__ = ["FT_SPEC", "FT_RUN", "FT_SHA256", "FTVerdict", "FTC2Gate", "ft_hash", "ft_version", "fault_tokens"]

FAULT_KINDS = ("timeout", "error", "malformed", "truncated", "missing", "mismatch", "no_target", "spec", "revision")

# 版の対象（11 章の「版」の行）。ここを変えると版が変わる
FT_SPEC: dict[str, Any] = {
    "label": "c2-FT@1",
    "components": {"F": spec_hash("F"), "T": spec_hash("T")},
    "combine": {"order": ["F", "T"], "score": "max", "early_exit": False},
    "targets": {A_KEY: ["F"], B_KEY: ["F", "T"], "other_contract": "hold"},
    "thresholds": {"block_at": 0.5, "allow_at": 0.1, "between": "hold"},
    "faults": {"kinds": list(FAULT_KINDS), "on_fault": "hold", "even_if_other_allows": True, "retry": 0},
}
# 実行の条件（11 章の「実行」の行）。評価ハーネスはこれと違う設定では動かない
FT_RUN: dict[str, Any] = {"model": "clef-flash", "budget_s": 60.0}
# 実装の commit で記録した版（11 章）。FT_SPEC を変えたらテストで分かる
FT_SHA256 = "d21c45b06b26fc43f9258a0a2728325fe8ca8d2e68fe45dd0850cc213e7c48e5"


def ft_hash() -> str:
    return sha256_text(_normalized(FT_SPEC))


def ft_version() -> str:
    return f"{FT_SPEC['label']}#{ft_hash()[:12]}"


class FTVerdict(C2Verdict):
    # 部品ごとの記録: {"F": {"p", "decision", "revision", "question_version", "context_sha256", "fault", "error"}}
    components: dict[str, dict[str, Any]] | None = None


def _component_fault(v: C2Verdict) -> str | None:
    if v.model == GUARD_MODEL:
        r = v.reason
        if "provenance mismatch" in r:
            return "mismatch"
        if r.endswith(" no target"):
            return "no_target"
        if " spec: " in r:
            return "spec"
        return "error"
    if v.truncated:
        return "truncated"
    if v.prob_block is None:
        return "missing"
    return None


def _exception_fault(e: Exception) -> str:
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, (MalformedResponse, ValidationError, json.JSONDecodeError)):
        return "malformed"
    return "error"


def fault_tokens(components: dict[str, dict[str, Any]]) -> list[str]:
    return [f"{m}:{c['fault']}" for m, c in components.items() if c["fault"] is not None]


class FTC2Gate:
    def __init__(self, base_url: str, model: str, **kw: Any):
        self.label = FT_SPEC["label"]
        self.question_version = ft_version()
        th = FT_SPEC["thresholds"]
        self.block_at, self.allow_at = th["block_at"], th["allow_at"]
        self.model = model
        self.parts = {m: SpecC2Gate(base_url, model, m, **kw) for m in FT_SPEC["combine"]["order"]}

    def _fault(self, components: dict[str, dict[str, Any]], tokens: list[str]) -> FTVerdict:
        return FTVerdict(decision="hold", reason=f"{self.label} fault {' '.join(tokens)}", model=GUARD_MODEL,
                         revision=GUARD_REVISION, truncated=any(c.get("truncated") for c in components.values()),
                         question_version=self.question_version, components=components)

    def _ask(self, method: str, payload: SendPayload, rendered: Sequence[RenderedMessage]) -> dict[str, Any]:
        try:
            v = self.parts[method].check_rendered(payload, rendered)
        except Exception as e:  # noqa: BLE001  部品の失敗は故障として記録し、もう一方も必ず呼ぶ
            return {"p": None, "decision": None, "revision": None, "question_version": None,
                    "context_sha256": None, "truncated": False, "fault": _exception_fault(e),
                    "error": type(e).__name__}
        return {"p": v.prob_block, "decision": v.decision, "revision": v.revision,
                "question_version": v.question_version, "context_sha256": v.context_sha256,
                "truncated": v.truncated, "fault": _component_fault(v), "error": None}

    def check_rendered(self, payload: SendPayload, rendered: Sequence[RenderedMessage]) -> FTVerdict:
        targets = FT_SPEC["targets"].get(payload.contract)
        if not isinstance(targets, list):
            return self._fault({}, ["gate:spec"])
        comps = {m: self._ask(m, payload, rendered) for m in targets}  # 順に 1 回ずつ。打ち切らない
        tokens = fault_tokens(comps)
        if tokens:
            return self._fault(comps, tokens)
        revisions = {c["revision"] for c in comps.values()}
        if len(revisions) != 1:
            return self._fault(comps, ["gate:revision"])
        s = max(c["p"] for c in comps.values())
        if s >= self.block_at:
            decision = "block"
        elif s <= self.allow_at:
            decision = "allow"
        else:
            decision = "hold"
        ps = " ".join(f"p_{m}={c['p']:.3f}" for m, c in comps.items())
        ctx = sha256_text(_normalized({m: c["context_sha256"] for m, c in comps.items()}))
        return FTVerdict(decision=decision, reason=f"{self.label} s={s:.3f} {ps}", model=self.model,
                         revision=revisions.pop(), truncated=False, prob_block=s,
                         question_version=self.question_version, context_sha256=ctx, components=comps)
