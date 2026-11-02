"""送信本文の出所区分（docs/v2-c2-eval-plan.md 2 章「出所区分（F）」）。

送信候補を組み立てるとき（orchestrator.candidate._render）に、各メッセージを出所ごとの部分に分けて
作り、部分をつないだものを SendPayload.messages にする。区分は組み立て時の位置だけで決まり、
本文中の「仕様:」「依頼:」などの見出しは解析しない（依頼文に見出しを書けば偽装できるため）。

Gateway はこれを使わない（Gateway が受け取るのは messages と digest だけ）。C2 の判定仕様が
区分を判定材料にするときは、区分から復元した内容が確定した送信内容と一致することを先に確かめる。
"""
from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.schemas import MAX_MESSAGES, Message

# system: 契約の固定文 / template: テンプレートの固定文 / option: 選択値 /
# approved: 承認済み本文 / free_text: 依頼文（利用者の自由入力）
Source = Literal["system", "template", "option", "approved", "free_text"]


class Part(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: Source
    text: str
    # 承認済み本文の ID と版（例 "spec-csv-001@1"）。組み立て側が承認済み本文の部分にだけ付ける
    ref: str | None = None

    @model_validator(mode="after")
    def _ref_only_on_approved(self) -> "Part":
        if self.ref is not None and self.source != "approved":
            raise ValueError("ref is only for approved parts")
        return self


class RenderedMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    role: Literal["system", "user"]
    parts: tuple[Part, ...] = Field(min_length=1)

    def join(self) -> Message:
        return Message(role=self.role, content="".join(p.text for p in self.parts))


Rendered = tuple[RenderedMessage, ...]


def reconstruct(rendered: Sequence[RenderedMessage]) -> tuple[Message, ...]:
    if len(rendered) > MAX_MESSAGES:
        raise ValueError(f"too many messages: {len(rendered)}")
    return tuple(r.join() for r in rendered)


def mismatch(rendered: Sequence[RenderedMessage], messages: Sequence[Message]) -> str | None:
    """区分から復元した内容と確定した送信内容のずれ。件数・順序・role・内容がすべて一致すれば None。"""
    if len(rendered) != len(messages):
        return f"message count {len(rendered)} != {len(messages)}"
    for i, (r, m) in enumerate(zip(rendered, messages)):
        if r.role != m.role:
            return f"message {i} role {r.role} != {m.role}"
        if "".join(p.text for p in r.parts) != m.content:
            return f"message {i} content differs"
    return None


def attachments(rendered: Sequence[RenderedMessage]) -> tuple[tuple[str | None, str], ...]:
    """送信物の添付（承認済み本文）の集合: 出現順の (ID と版, 本文の sha256)。

    組み立て時の区分だけから取り出す。依頼文の中身はここに入らない（Issue #17 の不変条件）。
    """
    return tuple((p.ref, hashlib.sha256(p.text.encode("utf-8")).hexdigest())
                 for r in rendered for p in r.parts if p.source == "approved")
