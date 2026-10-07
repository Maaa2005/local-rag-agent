"""v2 のゾーン間で共有する固定スキーマ。

Orchestrator（社内）と Gateway（外部接続）はこのモジュールの型だけで受け渡す。
本文の組み立ては Orchestrator が行い、Gateway は保存した内容をそのまま送る。
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Destination = Literal["claude", "codex"]
Route = Literal["code", "local", "A", "B", "human", "reject"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    role: Literal["system", "user"]
    content: str


class SendPayload(BaseModel):
    """モデルに見せる全テキスト・宛先・契約版。変更不能。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    contract: str  # 例 "A-faq-format@1"
    destination: Destination
    messages: tuple[Message, ...]
    max_output_chars: int = Field(gt=0)


def payload_digest(payload: SendPayload) -> str:
    """正規化 JSON の sha256。Orchestrator と Gateway が同じ関数で計算する。"""
    canonical = json.dumps(
        payload.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PrepareRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(min_length=8, max_length=64)
    payload: SendPayload
    expires_at: float  # UNIX 秒


class PrepareResponse(BaseModel):
    request_id: str
    digest: str
    state: "SendState"


class CommitRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str
    expected_digest: str


class SendState(str, Enum):
    PREPARED = "PREPARED"
    ATTEMPTING = "ATTEMPTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class FailureKind(str, Enum):
    not_sent = "not_sent"
    rejected = "rejected"
    unknown = "unknown"


class StatusResponse(BaseModel):
    request_id: str
    digest: str
    state: SendState
    failure: FailureKind | None = None
    output_text: str | None = None  # 信頼しないテキスト。表示のみ


PrepareResponse.model_rebuild()


# ---- 判断モデル（C1 / C2）の共通 I/F ----

class C1Decision(BaseModel):
    route: Route
    destination: Destination | None = None
    reason: str
    model: str
    revision: str
    probs: dict[str, float] = {}


class C2Verdict(BaseModel):
    decision: Literal["allow", "block", "hold"]
    reason: str
    model: str
    revision: str
    truncated: bool = False
    prob_block: float | None = None
