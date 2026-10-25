"""説明表示（F8）: 1 回の実行について、経路・送信先・確定本文・承認根拠・止めた理由・送信の状態を並べる。

画面や HTTP の層は持たない。既にある RunResult と、確認画面に出した内容（pipeline.confirmation_view）だけから組む。
- digest は計算し直さない。確認画面の digest と実行記録の digest が文字列で一致する時だけ確定本文を出す
- 確定本文と応答は RunResult.can_view を通った閲覧者にだけ出す
- 送信の状態は outcome ではなく Gateway の記録（attempted・gateway_state・gateway_failure）から決める。
  outcome="rejected" は権限での拒否と送信先の拒否の両方を含むため
- 応答（output_text）は信頼しないテキスト。HTML では必ずエスケープし、文字表示では制御文字を除く
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Literal

from common.schemas import C1Decision, SendState
from orchestrator.pipeline import RunResult

SendStatus = Literal["sent", "not_sent", "refused", "unknown"]

OUTCOME_LABELS: dict[str, str] = {
    "sent": "送信した",
    "held": "保留",
    "rejected": "拒否",
    "answered": "社内で回答した",
}

STOPPED_AT_LABELS: dict[str, str] = {
    "C1": "最初の振り分け（C1）",
    "eligibility": "権限・事前承認の確認",
    "contract": "契約の条件",
    "C2": "送る前の内容検査（C2）",
    "confirm": "利用者の確認",
    "digest": "確認した本文との照合",
    "retrieve": "社内文書の検索",
    "generate": "社内での回答作成",
    "none": "途中で止めていない",
}

ROUTE_LABELS: dict[str, str] = {
    "local": "社内だけで回答",
    "code": "コードで処理（未実装のため保留）",
    "A": "契約 A（説明文の言い換えを外部に依頼）",
    "B": "契約 B（コード作成を外部に依頼）",
    "human": "人の判断へ回す",
    "reject": "断る",
    "error": "振り分けに失敗",
    "destination_mismatch": "宛先が振り分けの結果と違う",
}

SEND_STATUS_LABELS: dict[SendStatus, str] = {
    "sent": "外部へ送信した",
    "not_sent": "外部へは送っていない（確定）",
    "refused": "送信先が受け付けなかった",
    "unknown": "送信先に届いたか分からない",
}

FREE_TEXT_AUTHORITY_LABELS: dict[bool | None, str] = {
    True: "あり",
    False: "なし",
    None: "依頼文を取らない契約（対象外）",
}


@dataclass(frozen=True)
class Explanation:
    outcome: str
    outcome_label: str
    stopped_at: str
    stopped_at_label: str
    reason: str
    route: str | None
    route_label: str
    destination: str | None
    contract: str | None
    send_status: SendStatus
    send_status_label: str
    send_detail: str
    request_id: str | None
    digest: str | None
    approval: tuple[tuple[str, str], ...]  # 承認根拠（項目, 値）
    body: tuple[tuple[str, str], ...] | None  # 確定本文（役割, 内容）。出さない時は None
    body_note: str
    output_text: str | None  # 信頼しないテキスト。出さない時は None
    output_note: str


def _route(run: RunResult, c1: C1Decision | None) -> str | None:
    if c1 is not None:
        return c1.route
    for ev in run.audit:
        if ev.stage == "C1":
            return ev.result
    return None


def send_status(run: RunResult) -> tuple[SendStatus, str]:
    """送信の状態と、その説明。commit を呼んでいなければ外部には出ていない。"""
    if run.outcome == "answered":
        return "not_sent", "社内のモデルだけで回答した。外部へ送る経路は通っていない"
    if not run.attempted:
        if run.gateway_received and run.gateway_state is None:
            return "not_sent", "Gateway に本文が届いたかは分からないが、送信の指示は出していない"
        if run.gateway_received:
            return "not_sent", "Gateway に本文を渡したが、送信の指示は出していない"
        return "not_sent", "Gateway に本文を渡していない"
    if run.gateway_state == SendState.SUCCEEDED:
        return "sent", "Gateway が送信を完了した"
    if run.gateway_state == SendState.FAILED and run.gateway_failure == "not_sent":
        return "not_sent", "Gateway が、送信していないことを確かめた"
    if run.gateway_state == SendState.FAILED and run.gateway_failure == "rejected":
        return "refused", "要求は送信先に届き、送信先が受け付けなかった。自動では送り直さない"
    rid = run.request_id or "（記録なし）"
    return "unknown", f"自動では送り直さない。要求 ID {rid} で Gateway に状態を照会して確かめる"


def _approval(run: RunResult) -> tuple[tuple[str, str], ...]:
    rows = [
        ("事前承認済みテキスト", run.approved_ref or "なし"),
        ("その承認者", run.approved_by or "なし"),
        ("確認した人", run.approver or "確認の前に止まった"),
        ("依頼文を外部に出す承認の権限", FREE_TEXT_AUTHORITY_LABELS[run.approver_authority]),
    ]
    if run.c2_verdict is not None:
        v = run.c2_verdict
        rows.append(("送る前の内容検査（C2）", f"{v.decision}（{v.model}@{v.revision}）: {v.reason}"))
    return tuple(rows)


def explain(
    run: RunResult,
    *,
    viewer_id: str,
    viewer_level: int | None = None,
    view: dict[str, Any] | None = None,
    c1: C1Decision | None = None,
) -> Explanation:
    """run を viewer_id（閲覧レベル viewer_level）に見せる説明。view は確認画面に出した内容。"""
    route = _route(run, c1)
    status, detail = send_status(run)
    allowed = run.can_view(viewer_id, viewer_level)
    view_matches = view is not None and run.digest is not None and view.get("digest") == run.digest

    if view is None:
        body, body_note = None, "確認画面に出す前に止まったか、確認画面の内容が渡されていない"
    elif not view_matches:
        body, body_note = None, "確認画面の digest と実行記録の digest が一致しないため出さない"
    elif not allowed:
        body, body_note = None, "閲覧の権限が無いため出さない"
    else:
        body = tuple((str(m["role"]), str(m["content"])) for m in view["messages"])
        body_note = "利用者が確認した本文（digest 一致）"

    if run.output_text is None:
        output, output_note = None, "応答なし"
    elif not allowed:
        output, output_note = None, "閲覧の権限が無いため出さない"
    else:
        output = run.output_text
        src = "社内のモデルの回答" if run.outcome == "answered" else "外部の応答"
        output_note = f"{src}。信頼しないテキストとして表示だけする。コードを含んでも実行しない"

    destination = view["destination"] if view_matches else (c1.destination if c1 is not None else None)
    contract = view["contract"] if view_matches else None
    return Explanation(
        outcome=run.outcome,
        outcome_label=OUTCOME_LABELS[run.outcome],
        stopped_at=run.stopped_at,
        stopped_at_label=STOPPED_AT_LABELS[run.stopped_at],
        reason=run.reason,
        route=route,
        route_label=ROUTE_LABELS.get(route, f"不明（{route}）") if route is not None else "記録なし",
        destination=destination,
        contract=contract,
        send_status=status,
        send_status_label=SEND_STATUS_LABELS[status],
        send_detail=detail,
        request_id=run.request_id,
        digest=run.digest,
        approval=_approval(run),
        body=body,
        body_note=body_note,
        output_text=output,
        output_note=output_note,
    )


def _rows(e: Explanation) -> list[tuple[str, str]]:
    return [
        ("結果", f"{e.outcome_label}（止めた所: {e.stopped_at_label}）"),
        ("理由", e.reason),
        ("経路", e.route_label),
        ("送信先", e.destination or "なし"),
        ("契約", e.contract or "なし"),
        ("送信の状態", f"{e.send_status_label}。{e.send_detail}"),
        ("要求 ID", e.request_id or "なし"),
        ("確認した本文の digest", e.digest or "なし"),
        *e.approval,
    ]


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _plain(s: str) -> str:
    return _CONTROL.sub("", s)


def to_text(e: Explanation) -> str:
    lines = [f"{k}: {_plain(v)}" for k, v in _rows(e)]
    lines.append(f"確定本文: {e.body_note}")
    for role, content in e.body or ():
        lines.append(f"  [{role}] {_plain(content)}")
    lines.append(f"応答: {e.output_note}")
    if e.output_text is not None:
        lines.append(_plain(e.output_text))
    return "\n".join(lines)


def to_html(e: Explanation) -> str:
    esc = html.escape
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in _rows(e))
    body = "".join(f"<li><b>{esc(role)}</b><pre>{esc(content)}</pre></li>" for role, content in e.body or ())
    out = f"<pre>{esc(e.output_text)}</pre>" if e.output_text is not None else ""
    return (
        f'<section class="explain" data-send-status="{esc(e.send_status)}">'
        f"<dl>{dl}</dl>"
        f"<h3>確定本文</h3><p>{esc(e.body_note)}</p><ol>{body}</ol>"
        f"<h3>応答</h3><p>{esc(e.output_note)}</p>{out}"
        "</section>"
    )
