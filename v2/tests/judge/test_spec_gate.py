"""判定仕様のゲート（judge.spec_gate）と Q0 の context 記録。通信は MockTransport だけ。"""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from common.provenance import Part, RenderedMessage
from common.schemas import Message
from judge.base import GUARD_MODEL, guarded_c2
from judge.c2_specs import A_KEY, B_KEY, build_request, spec_version
from judge.http_adapter import C2_QUESTION_VERSION, C2_QUESTIONS, HttpC2Gate
from judge.spec_gate import Bound, SpecC2Gate
from tests.judge.test_c2_specs import SPOOF, cand_a, cand_b


class Server:
    """送られた本文を残し、決まった p_yes を返す。"""

    def __init__(self, p: float = 0.05, *, status: int = 200, **over):
        self.p, self.status, self.over, self.posts = p, status, over, []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.posts.append((req.url.path, body))
        out = {"model": body["model"], "revision": "rev-test", "truncated": False,
               "answers": [{"id": q["id"], "answer": "yes" if self.p >= .5 else "no",
                            "probs": {"yes": self.p, "no": 1 - self.p}} for q in body["questions"]]}
        out.update(self.over)
        return httpx.Response(self.status, json=out)


def spec_gate(method: str, server: Server) -> SpecC2Gate:
    return SpecC2Gate("http://j", "clef-flash", method, transport=httpx.MockTransport(server))


@pytest.mark.parametrize("method", ["F", "R", "T"])
def test_posts_exactly_the_spec_request(method):
    c = cand_b(SPOOF)
    srv = Server(0.05)
    v = spec_gate(method, srv).check_rendered(c.payload, c.rendered)
    req = build_request(method, B_KEY, c.rendered)
    assert srv.posts == [("/v1/systemone", {"model": "clef-flash", "context": req.context,
                                            "questions": [req.question]})]
    assert v.context_sha256 == hashlib.sha256(srv.posts[0][1]["context"].encode("utf-8")).hexdigest()
    assert v.question_version == spec_version(method)
    assert (v.model, v.revision, v.truncated) == ("clef-flash", "rev-test", False)


@pytest.mark.parametrize("p,decision", [(0.0, "allow"), (0.1, "allow"), (0.1001, "hold"), (0.3, "hold"),
                                        (0.4999, "hold"), (0.5, "block"), (0.9, "block")])
def test_thresholds_come_from_the_spec(p, decision):
    c = cand_b()
    v = spec_gate("F", Server(p)).check_rendered(c.payload, c.rendered)
    assert (v.decision, v.prob_block) == (decision, p)
    assert v.reason == f"c2-F@1 p_yes={p:.3f}"


def test_truncated_answer_is_hold_even_when_low():
    c = cand_b()
    v = guarded_c2(Bound(spec_gate("T", Server(0.01, truncated=True)), c.rendered), c.payload, 5, 4000)
    assert v.decision == "hold" and v.truncated and v.prob_block == 0.01


@pytest.mark.parametrize("over,status", [({"answers": []}, 200), ({"revision": None}, 200), ({}, 500)])
def test_bad_server_answers_hold_through_guard(over, status):
    c = cand_b()
    v = guarded_c2(Bound(spec_gate("R", Server(0.01, status=status, **over)), c.rendered), c.payload, 5, 4000)
    assert v.decision == "hold" and v.model == GUARD_MODEL and v.reason.startswith("C2 error")
    assert v.context_sha256 is None


def test_provenance_mismatch_holds_without_calling():
    c = cand_b()
    srv = Server(0.0)
    bad = (c.rendered[0], RenderedMessage(role="user", parts=(Part(source="free_text", text="別の内容"),)))
    v = spec_gate("F", srv).check_rendered(c.payload, bad)
    assert srv.posts == [] and v.decision == "hold" and "provenance mismatch" in v.reason
    assert v.question_version == spec_version("F") and v.context_sha256 is None
    # 送信内容のほうを書き換えても同じ
    tampered = c.payload.model_copy(update={"messages": (*c.payload.messages[:-1],
                                                         Message(role="user", content="差し替え"))})
    v = spec_gate("T", srv).check_rendered(tampered, c.rendered)
    assert srv.posts == [] and "provenance mismatch" in v.reason


@pytest.mark.parametrize("method", ["R", "T"])
def test_contract_a_is_no_target_without_calling(method):
    c = cand_a()
    srv = Server(0.0)
    v = guarded_c2(Bound(spec_gate(method, srv), c.rendered), c.payload, 5, 4000)
    assert srv.posts == [] and v.decision == "hold" and v.reason.endswith(" no target")


def test_f_judges_contract_a():
    c = cand_a()
    srv = Server(0.02)
    v = spec_gate("F", srv).check_rendered(c.payload, c.rendered)
    assert v.decision == "allow" and len(srv.posts) == 1
    assert srv.posts[0][1]["context"] == build_request("F", A_KEY, c.rendered).context


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        SpecC2Gate("http://j", "m", "Q0")


def test_q0_post_is_unchanged_and_records_context_sha256():
    c = cand_b(SPOOF)
    srv = Server(0.05)
    v = HttpC2Gate("http://j", "clef-flash", transport=httpx.MockTransport(srv)).check(c.payload)
    context = "\n".join(f"[{m.role}]\n{m.content}" for m in c.payload.messages)
    assert srv.posts == [("/v1/systemone", {"model": "clef-flash", "context": context,
                                            "questions": [dict(q) for q in C2_QUESTIONS]})]
    assert v.context_sha256 == hashlib.sha256(context.encode("utf-8")).hexdigest()
    assert v.question_version == C2_QUESTION_VERSION and v.decision == "allow"
