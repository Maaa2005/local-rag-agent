"""送信候補の組み立て。候補は変更不能で、編集は新しい候補として作り直す。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from common.provenance import Part, Rendered, RenderedMessage, reconstruct
from common.schemas import SendPayload, payload_digest
from orchestrator.policy import ApprovedText, Contract

A_SYSTEM = (
    "あなたは社内向け FAQ の編集者です。与えられた説明文だけを使い、FAQ 形式に整えてください。\n"
    "説明文にない情報は追加しないでください。数値・期限・条件は原文どおりに残してください。"
)
B_SYSTEM = (
    "あなたは Python のコードを書くアシスタントです。与えられた仕様と依頼だけに基づいてコード案を示してください。\n"
    "仕様にない列やファイルを仮定しないでください。"
)


class ContractViolation(ValueError):
    """契約の形式・上限に合わない入力。stopped_at=contract で止める。"""


class SendCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str
    user: str
    contract: str
    approved_ref: str
    approved_sha256: str
    payload: SendPayload
    digest: str
    # payload.messages を出所区分つきの部分に分けたもの（C2 の判定仕様 F が使う）。Gateway には渡さない
    rendered: tuple[RenderedMessage, ...]
    created_at: datetime


def _check_options(contract: Contract, options: Any) -> dict[str, str]:
    if not isinstance(options, Mapping):
        raise ContractViolation("選択値がない")
    keys = set(options)
    allowed = set(contract.options)
    if keys - allowed:
        raise ContractViolation(f"契約にない選択項目: {sorted(keys - allowed)}")
    if allowed - keys:
        raise ContractViolation(f"選択値が足りない: {sorted(allowed - keys)}")
    for k, v in options.items():
        if not isinstance(v, str) or v not in contract.options[k]:
            raise ContractViolation(f"選択値 {k}={v!r} は許可リストにない")
    return dict(options)


def _render(contract: Contract, approved: ApprovedText, opts: Mapping[str, str], free_text: str | None) -> Rendered:
    """送信本文を出所区分つきの部分として組み立てる。部分をつないだものが SendPayload.messages になる。"""
    def t(text: str) -> Part:
        return Part(source="template", text=text)

    if contract.template == "A-faq-format/1":
        user = (
            t("文体: "), Part(source="option", text=opts["文体"]),
            t("\n長さ: "), Part(source="option", text=opts["長さ"]),
            t("\n形式: 質問と回答の組を3〜5個\n\n説明文:\n"),
            Part(source="approved", text=approved.body),
        )
        return (RenderedMessage(role="system", parts=(Part(source="system", text=A_SYSTEM),)),
                RenderedMessage(role="user", parts=user))
    if contract.template == "B-csv-codegen/1":
        if free_text is None:
            raise ContractViolation(f"契約 {contract.key} のテンプレートは依頼文が要る")
        user = (
            t("言語: "), Part(source="option", text=opts["言語"]),
            t("\n仕様:\n"), Part(source="approved", text=approved.body),
            t("\n依頼:\n"), Part(source="free_text", text=free_text),
        )
        return (RenderedMessage(role="system", parts=(Part(source="system", text=B_SYSTEM),)),
                RenderedMessage(role="user", parts=user))
    raise ContractViolation(f"未知のテンプレート: {contract.template}")


def build_candidate(
    *,
    user_id: str,
    contract: Contract,
    approved: ApprovedText,
    inp: Mapping[str, Any],
    destination: str,
    now: datetime,
    request_id: str | None = None,
) -> SendCandidate:
    opts = _check_options(contract, inp.get("options"))
    free_text = inp.get("free_text")
    if contract.free_text_max is None:
        if free_text not in (None, ""):
            raise ContractViolation(f"契約 {contract.key} に自由入力欄はない")
        free_text = None
    else:
        if not isinstance(free_text, str) or not free_text.strip():
            raise ContractViolation("依頼文が空")
        if len(free_text) > contract.free_text_max:
            raise ContractViolation(f"依頼文が {len(free_text)} 字で上限 {contract.free_text_max} 字を超える")
    rendered = _render(contract, approved, opts, free_text)
    messages = reconstruct(rendered)
    total = sum(len(m.content) for m in messages)
    if total > contract.payload_chars:
        raise ContractViolation(f"送信本文が {total} 字で上限 {contract.payload_chars} 字を超える")
    payload = SendPayload(
        contract=contract.key,
        destination=destination,  # type: ignore[arg-type]
        messages=messages,
        max_output_chars=contract.output_chars,
    )
    return SendCandidate(
        request_id=request_id or uuid.uuid4().hex,
        user=user_id,
        contract=contract.key,
        approved_ref=approved.key,
        approved_sha256=approved.sha256,
        payload=payload,
        digest=payload_digest(payload),
        rendered=rendered,
        created_at=now,
    )
