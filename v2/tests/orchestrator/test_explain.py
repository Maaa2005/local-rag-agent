"""説明表示（F8）: 経路・送信先・確定本文・承認根拠・止めた理由・未送信/結果不明。"""
from __future__ import annotations

from typing import get_args

import pytest

from common.schemas import Route, SendState
from gateway.adapters import FakeAdapter
from gateway.service import GatewayService
from gateway.store import SendStore
from integration.flow import run_contract_a
from judge.base import C1Decision, C2Verdict
from orchestrator.explain import (
    OUTCOME_LABELS,
    ROUTE_LABELS,
    STOPPED_AT_LABELS,
    explain,
    send_status,
    to_html,
    to_text,
)
from orchestrator.pipeline import AuditEvent, Outcome, RunResult, StoppedAt, confirmation_view

USER = "u_general"
REF = "expl-keihi-001@1"
OPTS = {"文体": "です・ます調", "長さ": "400字以内"}


def test_every_outcome_stop_and_route_has_a_label():
    assert set(get_args(Outcome)) <= OUTCOME_LABELS.keys()
    assert set(get_args(StoppedAt)) <= STOPPED_AT_LABELS.keys()
    assert set(get_args(Route)) <= ROUTE_LABELS.keys()


# ---- 送信の状態: outcome ではなく Gateway の記録から決める（全部の出口） ----

def _run(**kw) -> RunResult:
    base = dict(outcome="held", stopped_at="none", reason="r", request_id="req-1", digest="d" * 64)
    return RunResult(**{**base, **kw})


EXITS = [
    # (id, RunResult の値, 期待する状態)
    ("before_gateway", dict(outcome="rejected", stopped_at="eligibility", request_id=None), "not_sent"),
    ("c2_hold", dict(stopped_at="C2"), "not_sent"),
    ("prepare_not_reached", dict(gateway_received=False), "not_sent"),
    ("prepare_maybe_reached", dict(gateway_received=True), "not_sent"),
    ("digest_mismatch", dict(outcome="rejected", stopped_at="digest", gateway_received=True,
                             gateway_state=SendState.PREPARED), "not_sent"),
    ("succeeded", dict(outcome="sent", gateway_received=True, attempted=True, gateway_state=SendState.SUCCEEDED),
     "sent"),
    ("failed_not_sent", dict(gateway_received=True, attempted=True, gateway_state=SendState.FAILED,
                             gateway_failure="not_sent"), "not_sent"),
    ("failed_rejected", dict(outcome="rejected", gateway_received=True, attempted=True,
                             gateway_state=SendState.FAILED, gateway_failure="rejected"), "refused"),
    ("failed_unknown", dict(gateway_received=True, attempted=True, gateway_state=SendState.FAILED,
                            gateway_failure="unknown"), "unknown"),
    ("failed_without_kind", dict(gateway_received=True, attempted=True, gateway_state=SendState.FAILED), "unknown"),
    ("status_attempting", dict(gateway_received=True, attempted=True, gateway_state=SendState.ATTEMPTING), "unknown"),
    ("status_prepared", dict(gateway_received=True, attempted=True, gateway_state=SendState.PREPARED), "unknown"),
    ("status_query_failed", dict(gateway_received=True, attempted=True, gateway_state=None), "unknown"),
    ("answered_locally", dict(outcome="answered", request_id=None), "not_sent"),
]


@pytest.mark.parametrize("fields,expected", [(f, e) for _, f, e in EXITS], ids=[i for i, _, _ in EXITS])
def test_send_status_for_every_exit(fields, expected):
    run = _run(**fields)
    status, _ = send_status(run)
    assert status == expected
    text = to_text(explain(run, viewer_id=USER))
    if run.attempted and run.gateway_failure != "not_sent":
        # commit を呼んだ後は、送信していないと確かめた時以外「送っていない」と言わない
        assert "送っていない" not in text and "未送信" not in text
    if expected == "unknown":
        assert "req-1" in text and "送り直さない" in text


# ---- 実際の流れ（偽の送信先）から組む ----

class _C1:
    def route(self, user, request, input):  # noqa: A002
        return C1Decision(route="A", destination="claude", reason="stub", model="stub", revision="0",
                          question_version="stub@1")


class _C2:
    def check(self, payload):
        return C2Verdict(decision="allow", reason="stub", model="stub", revision="0", question_version="stub@1")


class _Confirmer:
    def __init__(self) -> None:
        self.seen: list = []

    def confirm(self, candidate):
        self.seen.append(candidate)
        return candidate.digest


def _flow(tmp_path, mode: str = "ok", reply: str = "Q. 期限は？ A. 翌月10日まで"):
    fake = FakeAdapter(reply=reply, mode=mode)
    svc = GatewayService(SendStore(tmp_path / "gateway.db"), {"claude": fake, "codex": FakeAdapter()})
    conf = _Confirmer()
    res = run_contract_a(USER, REF, OPTS, "claude", c1=_C1(), c2=_C2(), confirmer=conf, gateway=svc,
                         request_text="経費精算の説明を FAQ にして")
    view = confirmation_view(conf.seen[-1]) if conf.seen else None
    return res, view, fake


def test_sent_shows_route_destination_body_approval_and_output(tmp_path):
    res, view, fake = _flow(tmp_path)
    e = explain(res.run, viewer_id=USER, viewer_level=1, view=view, c1=res.c1)
    assert (e.outcome, e.send_status, e.route, e.destination, e.contract) == ("sent", "sent", "A", "claude", "A-faq-format@1")
    # 確定本文は確認画面に出した本文そのもの
    assert e.body == tuple((m["role"], m["content"]) for m in view["messages"])
    assert [m.content for m in fake.received[-1].messages] == [c for _, c in e.body]
    approval = dict(e.approval)
    assert approval["事前承認済みテキスト"] == REF and approval["確認した人"] == USER
    assert approval["その承認者"] == res.run.approved_by and res.run.approved_by
    assert approval["依頼文を外部に出す承認の権限"] == "依頼文を取らない契約（対象外）"
    assert e.output_text == "Q. 期限は？ A. 翌月10日まで"
    assert e.digest == view["digest"]


@pytest.mark.parametrize("mode,status", [("not_sent", "not_sent"), ("rejected", "refused"), ("unknown", "unknown"),
                                         ("timeout", "unknown")])
def test_gateway_failures_from_real_flow(tmp_path, mode, status):
    res, view, _ = _flow(tmp_path, mode=mode)
    assert res.run.gateway_failure == ("unknown" if mode == "timeout" else mode)
    e = explain(res.run, viewer_id=USER, viewer_level=1, view=view)
    assert e.send_status == status
    if status == "unknown":
        assert res.run.request_id in e.send_detail


def test_reason_and_stop_shown_when_rejected_before_gateway(tmp_path):
    fake = FakeAdapter()
    svc = GatewayService(SendStore(tmp_path / "gateway.db"), {"claude": fake, "codex": fake})
    res = run_contract_a("u_nobody", REF, OPTS, "claude", c1=_C1(), c2=_C2(), confirmer=_Confirmer(), gateway=svc,
                         request_text="x")
    e = explain(res.run, viewer_id="u_nobody")
    assert e.outcome == "rejected" and e.stopped_at_label == "権限・事前承認の確認"
    assert e.reason == res.run.reason and e.reason
    assert e.send_status == "not_sent" and e.body is None and dict(e.approval)["確認した人"] == "確認の前に止まった"


# ---- 閲覧の権限と、確認画面との一致 ----

def test_other_user_or_low_level_sees_no_body_or_output(tmp_path):
    res, view, _ = _flow(tmp_path)
    level = res.run.min_view_level
    for viewer, lv in [("u_manager", 3), (USER, None if level is None else level - 1)]:
        if viewer == USER and level is None:
            continue
        e = explain(res.run, viewer_id=viewer, viewer_level=lv, view=view)
        assert e.body is None and e.output_text is None
        assert "Q. 期限は？" not in to_text(e) and "Q. 期限は？" not in to_html(e)
        assert e.send_status == "sent"  # 状態と理由は見せる


def test_view_with_other_digest_is_not_shown(tmp_path):
    res, view, _ = _flow(tmp_path)
    e = explain(res.run, viewer_id=USER, viewer_level=1, view={**view, "digest": "0" * 64})
    assert e.body is None and "一致しない" in e.body_note
    assert e.destination is None and e.contract is None


# ---- 信頼しないテキスト ----

EVIL = '<script>alert("x")</script>\x1b[31m赤</img onerror=1>'


def test_untrusted_output_and_reason_are_escaped_in_html(tmp_path):
    res, view, _ = _flow(tmp_path, reply=EVIL)
    e = explain(res.run, viewer_id=USER, viewer_level=1, view=view)
    page = to_html(e)
    assert "<script>" not in page and "<img" not in page.replace("&lt;img", "")
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
    run = _run(reason="<b>理由</b>", audit=[AuditEvent("C1", "<i>")])
    page2 = to_html(explain(run, viewer_id=USER))
    assert "<b>理由" not in page2 and "&lt;b&gt;理由" in page2 and "<i>" not in page2


def test_control_characters_removed_in_text(tmp_path):
    res, view, _ = _flow(tmp_path, reply=EVIL)
    text = to_text(explain(res.run, viewer_id=USER, viewer_level=1, view=view))
    assert "\x1b" not in text and "赤" in text
    assert "信頼しない" in text and "実行しない" in text
