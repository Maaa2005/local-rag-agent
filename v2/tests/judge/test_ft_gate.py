"""F と T の合成 C2-FT@1（judge.ft_gate、docs/v2-c2-eval-plan.md 11 章）。通信は MockTransport だけ。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from common.provenance import Part, RenderedMessage
from judge.base import GUARD_MODEL, guarded_c2
from judge.c2_specs import A_KEY, B_KEY, _normalized, build_request, sha256_text, spec_hash, spec_version
from judge.ft_gate import FT_RUN, FT_SHA256, FT_SPEC, FTC2Gate, FTVerdict, ft_hash, ft_version
from judge.spec_gate import Bound
from tests.judge.test_c2_specs import cand_a, cand_b

DOCS = Path(__file__).resolve().parents[3] / "docs" / "v2-c2-eval-plan.md"


class Server:
    """呼ばれた順に reply(i, body) を返す。None は 500、例外はそのまま投げる。"""

    def __init__(self, *ps, rev=None, **over):
        self.ps, self.rev, self.over, self.posts = ps, rev, over, []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.posts.append(body)
        i = len(self.posts) - 1
        p = self.ps[i]
        if isinstance(p, Exception):
            raise p
        if p is None:
            return httpx.Response(500, json={})
        out = {"model": body["model"], "revision": (self.rev or (lambda _i: "rev-test"))(i), "truncated": False,
               "answers": [{"id": q["id"], "answer": "yes" if p >= .5 else "no", "probs": {"yes": p, "no": 1 - p}}
                           for q in body["questions"]]}
        out.update(self.over)
        return httpx.Response(200, json=out)


def ft(srv) -> FTC2Gate:
    return FTC2Gate("http://j", "clef-flash", transport=httpx.MockTransport(srv))


def test_version_is_pinned_to_the_components_and_the_docs():
    assert FT_SPEC["components"] == {"F": spec_hash("F"), "T": spec_hash("T")}
    assert spec_version("F") == "c2-F@1#f2c62f305000" and spec_version("T") == "c2-T@1#418116237b14"
    assert ft_hash() == FT_SHA256 == sha256_text(_normalized(FT_SPEC))
    assert ft_version() == "c2-FT@1#d21c45b06b26"
    assert FT_RUN == {"model": "clef-flash", "budget_s": 60.0}
    ch11 = DOCS.read_text(encoding="utf-8").split("## 11.", 1)[1]
    assert ft_version() in ch11 and FT_SHA256 in ch11


def test_contract_a_asks_only_f():
    c = cand_a()
    srv = Server(0.02)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    req = build_request("F", A_KEY, c.rendered)
    assert srv.posts == [{"model": "clef-flash", "context": req.context, "questions": [req.question]}]
    assert isinstance(v, FTVerdict) and v.decision == "allow" and v.prob_block == 0.02
    assert list(v.components) == ["F"] and v.components["F"]["fault"] is None
    assert v.question_version == ft_version() and v.reason == "c2-FT@1 s=0.020 p_F=0.020"
    assert v.context_sha256 == sha256_text(_normalized({"F": req.context_sha256}))


def test_contract_b_asks_f_then_t_once_each():
    c = cand_b()
    srv = Server(0.03, 0.04)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    f, t = (build_request(m, B_KEY, c.rendered) for m in ("F", "T"))
    assert [b["context"] for b in srv.posts] == [f.context, t.context]
    assert [b["questions"] for b in srv.posts] == [[f.question], [t.question]]
    assert json.loads(srv.posts[1]["context"]) == {"依頼文": "date と amount の列を集計する関数を書いてください"}
    assert v.components["F"]["p"] == 0.03 and v.components["T"]["p"] == 0.04
    assert v.components["F"]["question_version"] == spec_version("F")
    assert v.components["T"]["question_version"] == spec_version("T")
    assert v.context_sha256 == sha256_text(_normalized({"F": f.context_sha256, "T": t.context_sha256}))
    assert (v.model, v.revision, v.truncated) == ("clef-flash", "rev-test", False)


@pytest.mark.parametrize("pf,pt,decision", [(0.05, 0.05, "allow"), (0.1, 0.1, "allow"), (0.05, 0.1001, "hold"),
                                            (0.3, 0.05, "hold"), (0.4999, 0.2, "hold"), (0.5, 0.0, "block"),
                                            (0.0, 0.9, "block")])
def test_score_is_max_and_thresholds(pf, pt, decision):
    c = cand_b()
    srv = Server(pf, pt)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    assert (v.decision, v.prob_block) == (decision, max(pf, pt)) and len(srv.posts) == 2


def test_high_f_does_not_skip_t():
    c = cand_b()
    srv = Server(0.99, 0.0)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    assert v.decision == "block" and len(srv.posts) == 2 and v.components["T"]["p"] == 0.0


def test_t_fault_holds_even_when_f_allows():
    c = cand_b()
    v = ft(Server(0.01, None)).check_rendered(c.payload, c.rendered)
    assert v.decision == "hold" and v.model == GUARD_MODEL and v.prob_block is None
    assert v.reason == "c2-FT@1 fault T:error"
    assert v.components["F"]["p"] == 0.01 and v.components["F"]["fault"] is None
    assert v.components["T"]["fault"] == "error" and v.components["T"]["error"] == "HTTPStatusError"


@pytest.mark.parametrize("bad,fault", [(None, "error"), (httpx.ReadTimeout("slow"), "timeout"),
                                       (httpx.ConnectError("down"), "error")])
def test_f_fault_still_asks_t_and_records_both(bad, fault):
    c = cand_b()
    srv = Server(bad, 0.02)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    assert len(srv.posts) == 2 and v.decision == "hold" and v.reason == f"c2-FT@1 fault F:{fault}"
    assert v.components["F"]["fault"] == fault and v.components["T"]["p"] == 0.02


@pytest.mark.parametrize("over", [{"answers": []}, {"revision": None}, {"model": "other"}])
def test_malformed_answer_is_a_fault(over):
    c = cand_b()
    v = ft(Server(0.01, 0.01, **over)).check_rendered(c.payload, c.rendered)
    assert v.decision == "hold" and v.reason == "c2-FT@1 fault F:malformed T:malformed"


def test_truncated_component_is_a_fault_even_when_low():
    c = cand_b()
    v = ft(Server(0.01, 0.01, truncated=True)).check_rendered(c.payload, c.rendered)
    assert v.decision == "hold" and v.truncated and v.reason == "c2-FT@1 fault F:truncated T:truncated"


def test_different_revisions_are_a_fault():
    c = cand_b()
    v = ft(Server(0.01, 0.01, rev=lambda i: f"rev-{i}")).check_rendered(c.payload, c.rendered)
    assert v.decision == "hold" and v.reason == "c2-FT@1 fault gate:revision"


def test_provenance_mismatch_holds_without_calling():
    c = cand_b()
    srv = Server()
    bad = (c.rendered[0], RenderedMessage(role="user", parts=(Part(source="free_text", text="別の内容"),)))
    v = ft(srv).check_rendered(c.payload, bad)
    assert srv.posts == [] and v.decision == "hold" and v.reason == "c2-FT@1 fault F:mismatch T:mismatch"


def test_other_contract_holds_without_calling():
    c = cand_b()
    srv = Server()
    v = ft(srv).check_rendered(c.payload.model_copy(update={"contract": "X-other@1"}), c.rendered)
    assert srv.posts == [] and v.decision == "hold" and v.reason == "c2-FT@1 fault gate:spec"


def test_components_survive_guarded_c2():
    c = cand_b()
    v = guarded_c2(Bound(ft(Server(0.02, 0.6)), c.rendered), c.payload, 60.0, 4000)
    assert isinstance(v, FTVerdict) and v.decision == "block"
    assert v.components["F"]["p"] == 0.02 and v.components["T"]["p"] == 0.6
    v = guarded_c2(Bound(ft(Server(0.02, None)), c.rendered), c.payload, 60.0, 4000)
    assert isinstance(v, FTVerdict) and v.components["T"]["fault"] == "error"


def test_budget_covers_both_calls():
    c = cand_b()

    def slow(req):
        time.sleep(0.6)
        return Server(0.01)(req)

    g = FTC2Gate("http://j", "clef-flash", transport=httpx.MockTransport(slow))
    t0 = time.perf_counter()
    v = guarded_c2(Bound(g, c.rendered), c.payload, 0.3, 4000)
    elapsed = time.perf_counter() - t0
    assert v.decision == "hold" and v.reason.startswith("C2 timeout") and elapsed < 0.3 + 0.2
