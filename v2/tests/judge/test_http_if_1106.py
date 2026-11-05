"""PR #26 のレビューで足した決まりの試験（docs/redesign-v2.md「判断モデルの HTTP I/F」検証規則 6〜8）。

- 止めた判定にも、送った番号を残す（判定の結果と監査 DB）。時間切れの後は次の要求を送らない
- FT の部品 F・T の番号を components と結果に残す
- 評価ハーネスが expected_revision を渡し、ready() を呼び、記録に番号を入れる
- 圧縮された応答は展開しながら上限で止める。ready() は全体の期限を守る
- 上限・番号の形を緩める改変を試験で検出する
"""
from __future__ import annotations

import gzip
import io
import json
import threading
import time
import tracemalloc
import zlib

import httpx
import pytest
from fastapi.testclient import TestClient

from common.provenance import Part, RenderedMessage
from common.schemas import Message, SendPayload
from evalharness import run_c2_compare as cmp_h
from evalharness import run_v2_eval as eval_h
from judge.base import GUARD_MODEL, C1Decision, C2Verdict, call_c2, guarded_c1, guarded_c2, is_guard_stop
from judge.c2_specs import B_KEY
from judge.ft_gate import FTC2Gate, FTVerdict
from judge.http_adapter import (
    MAX_RESPONSE_BYTES,
    REQUEST_ID_HEADER,
    REQUEST_ID_RE,
    ROUTES,
    CascadeC1Router,
    CascadeC2Gate,
    HttpC1Router,
    HttpC2Gate,
    MalformedResponse,
)
from judge.spec_gate import Bound, SpecC2Gate
from orchestrator.policy import load_policy
from tests.judge.test_c2_specs import cand_b

ID = "a" * 32
B_INPUT = {"contract": B_KEY, "spec": "spec-csv-001@1", "free_text": "x", "options": {"言語": "Python 3.11"}}


def payload() -> SendPayload:
    return SendPayload(contract=B_KEY, destination="codex", messages=(Message(role="user", content="ok"),),
                       max_output_chars=3000)


def c1_body(top: str = "local", p: float = 0.9, revision: str = "r1", **over) -> dict:
    rest = (1 - p) / (len(ROUTES) - 1)
    return {"model": "m", "revision": revision, **over,
            "answers": [{"id": "route", "answer": top, "probs": {r: (p if r == top else rest) for r in ROUTES}}]}


def c2_body(p: float = 0.01, revision: str = "r1", model: str = "m", **over) -> dict:
    return {"model": model, "revision": revision, **over,
            "answers": [{"id": "forbidden", "answer": "yes" if p >= .5 else "no", "probs": {"yes": p, "no": 1 - p}}]}


class Judge:
    """送られた番号を覚えて、指定の応答を返す偽の判断サーバ。"""

    def __init__(self, body=None, *, status=200, sleep=0.0, chunks=None, headers=None, echo=None, raises=None):
        self.body, self.status, self.sleep, self.chunks = body, status, sleep, chunks
        self.headers, self.echo, self.raises = headers or {}, echo, raises
        self.seen: list[str] = []
        self.gets: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            self.gets.append(req.url.path)
            return httpx.Response(200, json={"model": "m", "revision": "r1"})
        self.seen.append(req.headers[REQUEST_ID_HEADER])
        if self.sleep:
            time.sleep(self.sleep)
        if self.raises:
            raise self.raises
        headers = dict(self.headers)
        if self.echo:
            headers[REQUEST_ID_HEADER] = self.echo
        if self.chunks is not None:
            return httpx.Response(self.status, headers=headers, content=iter(self.chunks))
        return httpx.Response(self.status, headers=headers, json=self.body if self.body is not None else {})


def router(j: Judge, **kw) -> HttpC1Router:
    return HttpC1Router("http://j", "m", transport=httpx.MockTransport(j), **kw)


def gate(j: Judge, **kw) -> HttpC2Gate:
    return HttpC2Gate("http://j", "m", transport=httpx.MockTransport(j), **kw)


TOO_LARGE = [b"x" * (MAX_RESPONSE_BYTES + 1)]

# (要求は送るが判定は止まる経路, 偽の判断サーバ, アダプタの引数, guard の時間切れ)
C1_SENT_STOPS = {
    "revision": (lambda: Judge(c1_body(revision="r2")), {"expected_revision": "r1"}, 5.0),
    "too_large": (lambda: Judge(chunks=TOO_LARGE), {}, 5.0),
    "other_id": (lambda: Judge(c1_body(), echo="f" * 32), {}, 5.0),
    "http_503": (lambda: Judge({}, status=503), {}, 5.0),
    "not_json": (lambda: Judge(chunks=[b"not json"]), {}, 5.0),
    "connect": (lambda: Judge(raises=httpx.ConnectError("down")), {}, 5.0),
    "timeout": (lambda: Judge(c1_body(), sleep=0.6), {}, 0.2),
    "truncated": (lambda: Judge(c1_body(truncated=True)), {}, 5.0),
    "route_not_in_contract": (lambda: Judge(c1_body(top="A")), {}, 5.0),
}


@pytest.mark.parametrize("name", list(C1_SENT_STOPS))
def test_guarded_c1_keeps_the_sent_id_when_it_stops(name):
    make, kw, timeout = C1_SENT_STOPS[name]
    j = make()
    d = guarded_c1(router(j, **kw), "u", "r", dict(B_INPUT), timeout, 10_000)
    assert d.route == "human" and is_guard_stop(d)
    assert len(j.seen) == 1 and REQUEST_ID_RE.fullmatch(j.seen[0]) and d.judge_request_id == j.seen[0]


def test_c1_unusable_contract_input_keeps_the_sent_id():
    j = Judge(c1_body(top="B"))
    d = guarded_c1(router(j), "u", "r", {**B_INPUT, "source": "元資料"}, 5.0, 10_000)
    assert d.route == "human" and "unusable contract input" in d.reason and d.judge_request_id == j.seen[0]


class _StubC1:
    def __init__(self, **kw):
        self.kw = kw

    def route(self, user, request, input):  # noqa: A002
        base = dict(route="B", destination="codex", reason="stub", model="m", revision="r1", question_version="q@1",
                    judge_request_id=ID)
        return C1Decision(**{**base, **self.kw})


@pytest.mark.parametrize("kw, inp, why", [
    ({"truncated": True}, B_INPUT, "truncated"),
    ({"destination": None}, B_INPUT, "without destination"),
    ({"destination": "claude"}, B_INPUT, "not the sole destination"),
    ({"route": "A", "destination": "claude"}, B_INPUT, "not backed by contract"),
    ({}, {**B_INPUT, "contract": "X-unknown@1"}, "not backed by contract"),
    ({}, {**B_INPUT, "source": "x"}, "unusable contract input"),
])
def test_every_c1_stop_after_an_answer_keeps_the_id(kw, inp, why):
    d = guarded_c1(_StubC1(**kw), "u", "r", inp, 5.0, 10_000)
    assert d.route == "human" and why in d.reason and d.judge_request_id == ID


def test_nothing_sent_means_no_id():
    j = Judge(c1_body())
    d = guarded_c1(router(j), "u", "r" * 100, None, 5.0, 10)
    v = guarded_c2(gate(j), payload(), 5.0, 1)
    assert j.seen == [] and d.route == "human" and d.judge_request_id is None
    assert v.decision == "hold" and v.judge_request_id is None


def test_malformed_c1_answer_keeps_the_id_the_adapter_sent():
    class Bad:
        def __init__(self, inner):
            self.inner = inner

        def route(self, user, request, input):  # noqa: A002
            self.inner.route(user, request, input)
            return "not a decision"

    j = Judge(c1_body())
    d = guarded_c1(Bad(router(j)), "u", "r", None, 5.0, 10_000)
    assert "malformed" in d.reason and d.judge_request_id == j.seen[0]


C2_SENT_STOPS = {
    "revision": (lambda: Judge(c2_body(revision="r2")), {"expected_revision": "r1"}, 5.0),
    "too_large": (lambda: Judge(chunks=TOO_LARGE), {}, 5.0),
    "other_id": (lambda: Judge(c2_body(), echo="f" * 32), {}, 5.0),
    "http_503": (lambda: Judge({}, status=503), {}, 5.0),
    "timeout": (lambda: Judge(c2_body(), sleep=0.6), {}, 0.2),
}


@pytest.mark.parametrize("name", list(C2_SENT_STOPS))
@pytest.mark.parametrize("call", ["guarded_c2", "call_c2"])
def test_c2_keeps_the_sent_id_when_it_holds(name, call):
    make, kw, timeout = C2_SENT_STOPS[name]
    j = make()
    g = gate(j, **kw)
    v = guarded_c2(g, payload(), timeout, 10_000) if call == "guarded_c2" else call_c2(g, payload(), timeout)
    assert v.decision == "hold" and v.model == GUARD_MODEL
    assert len(j.seen) == 1 and v.judge_request_id == j.seen[0]


def test_c2_without_timeout_also_keeps_the_sent_id():
    j = Judge(c2_body(revision="r2"))
    v = call_c2(gate(j, expected_revision="r1"), payload(), None)
    assert v.decision == "hold" and v.judge_request_id == j.seen[0]


def test_c2_truncated_allow_keeps_the_id():
    class Gate:
        def check(self, payload):
            return C2Verdict(decision="allow", reason="stub", model="m", revision="r", truncated=True,
                             question_version="q@1", judge_request_id=ID)

    v = guarded_c2(Gate(), payload(), 5.0, 10_000)
    assert v.decision == "hold" and v.truncated and v.judge_request_id == ID


def test_c1_guard_stops_the_other_id_and_the_too_large_answer():
    """C1 の番号違いと大きすぎる応答は、guard が人へ回す。"""
    d = guarded_c1(router(Judge(c1_body(), echo="f" * 32)), "u", "r", None, 5.0, 10_000)
    assert d.route == "human" and "request id mismatch" in d.reason
    d = guarded_c1(router(Judge(chunks=TOO_LARGE)), "u", "r", None, 5.0, 10_000)
    assert d.route == "human" and "too large" in d.reason
    d = guarded_c1(router(Judge(c1_body(), headers={"content-length": str(MAX_RESPONSE_BYTES + 1)})),
                   "u", "r", None, 5.0, 10_000)
    assert d.route == "human" and "too large" in d.reason


def test_cascade_stop_keeps_both_ids_like_rule_6():
    j1, j2 = Judge(c2_body(0.3)), Judge({}, status=503)
    v = guarded_c2(CascadeC2Gate(gate(j1), gate(j2)), payload(), 5.0, 10_000)
    assert v.decision == "hold" and v.judge_request_id == j2.seen[0] and v.reason.endswith(f"; first req={j1.seen[0]}")

    j1, j2 = Judge(c1_body("local", 0.4)), Judge(c1_body("code", 0.9, truncated=True))
    d = guarded_c1(CascadeC1Router(router(j1), router(j2), escalate_below=0.6), "u", "r", None, 5.0, 10_000)
    assert d.route == "human" and d.judge_request_id == j2.seen[0] and d.reason.endswith(f"; first req={j1.seen[0]}")


# ---- 入口 run_contract_b から監査 DB まで ----

def _audit_row(tmp_path, *, c1, c2, **kw):
    from integration.flow_b import run_contract_b
    from orchestrator.audit_store import AuditStore
    from tests.orchestrator.test_pipeline import B_TEXT, NOW, FakeConfirmer, FakeGateway

    store = AuditStore(tmp_path / "audit.db")
    gw = FakeGateway()
    try:
        res = run_contract_b("u_manager", "spec-csv-001@1", B_TEXT, {"言語": "Python 3.11"}, "codex", c2=c2,
                             confirmer=FakeConfirmer(), gateway=gw, now=NOW, audit_sink=store, c1=c1, **kw)
        (row,) = store.runs()
    finally:
        store.close()
    assert gw.calls == 0 and res.outcome == "held"
    return row


@pytest.mark.parametrize("name", list(C1_SENT_STOPS))
def test_c1_stop_reaches_the_audit_db_with_the_sent_id(tmp_path, name):
    make, kw, timeout = C1_SENT_STOPS[name]
    j = make()
    row = _audit_row(tmp_path, c1=router(j, **kw), c2=gate(Judge()), c1_timeout_s=timeout)
    assert row["stopped_at"] == "C1" and row["c1_route"] == "human" and row["c1_model"] == GUARD_MODEL
    assert row["c1_judge_request_id"] == j.seen[0] and row["c2_judge_request_id"] is None


@pytest.mark.parametrize("name", list(C2_SENT_STOPS))
def test_c2_stop_reaches_the_audit_db_with_the_sent_id(tmp_path, name):
    make, kw, timeout = C2_SENT_STOPS[name]
    j = make()
    row = _audit_row(tmp_path, c1=_StubC1(), c2=gate(j, **kw), c2_timeout_s=timeout)
    assert row["stopped_at"] == "C2" and row["c2_decision"] == "hold" and row["c2_model"] == GUARD_MODEL
    assert row["c1_judge_request_id"] == ID and row["c2_judge_request_id"] == j.seen[0]


# ---- FT の部品の番号 ----

class FTServer:
    """呼ばれた順に p を答える。None は 500、例外は投げる。"""

    def __init__(self, *ps):
        self.ps, self.seen = ps, []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.seen.append(req.headers[REQUEST_ID_HEADER])
        p = self.ps[len(self.seen) - 1]
        if isinstance(p, Exception):
            raise p
        if p is None:
            return httpx.Response(500, json={})
        body = json.loads(req.content)
        return httpx.Response(200, json={"model": body["model"], "revision": "r", "answers": [
            {"id": q["id"], "answer": "yes" if p >= .5 else "no", "probs": {"yes": p, "no": 1 - p}}
            for q in body["questions"]]})


def ft(srv) -> FTC2Gate:
    return FTC2Gate("http://j", "clef-flash", transport=httpx.MockTransport(srv))


def test_ft_keeps_the_component_ids_in_components_and_in_the_result():
    c = cand_b()
    srv = FTServer(0.02, 0.03)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    assert v.decision == "allow" and len(set(srv.seen)) == 2
    assert v.components["F"]["judge_request_id"] == srv.seen[0]
    assert v.components["T"]["judge_request_id"] == srv.seen[1]
    assert v.judge_request_id == f"{srv.seen[0]},{srv.seen[1]}"


@pytest.mark.parametrize("ps, fault", [((0.01, None), "T:error"), ((httpx.ReadTimeout("slow"), 0.01), "F:timeout"),
                                       ((0.01, httpx.ConnectError("down")), "T:error")])
def test_ft_fault_keeps_the_ids_of_what_it_sent(ps, fault):
    c = cand_b()
    srv = FTServer(*ps)
    v = ft(srv).check_rendered(c.payload, c.rendered)
    assert v.decision == "hold" and v.reason == f"c2-FT@1 fault {fault}"
    assert [v.components[m]["judge_request_id"] for m in ("F", "T")] == srv.seen
    assert v.judge_request_id == ",".join(srv.seen)


def test_ft_that_sent_nothing_has_no_id():
    c = cand_b()
    srv = FTServer()
    bad = (c.rendered[0], RenderedMessage(role="user", parts=(Part(source="free_text", text="別の内容"),)))
    v = ft(srv).check_rendered(c.payload, bad)
    assert srv.seen == [] and v.judge_request_id is None
    assert v.components["F"]["judge_request_id"] is None and v.components["T"]["judge_request_id"] is None


def test_ft_through_guard_and_budget_timeout_keeps_the_ids():
    c = cand_b()
    srv = FTServer(0.02, 0.6)
    v = guarded_c2(Bound(ft(srv), c.rendered), c.payload, 5.0, 4000)
    assert isinstance(v, FTVerdict) and v.judge_request_id == ",".join(srv.seen)

    seen = []

    def slow(req):
        seen.append(req.headers[REQUEST_ID_HEADER])
        if len(seen) == 2:
            time.sleep(1.0)
        return FTServer(0.01)(req)

    v = guarded_c2(Bound(ft(slow), c.rendered), c.payload, 0.4, 4000)
    assert v.decision == "hold" and v.reason.startswith("C2 timeout")
    assert v.judge_request_id == seen[1] and v.reason.endswith(f"; first req={seen[0]}")


# ---- guard の時間切れの後に送られる要求（前段で期限を超え、見捨てたスレッドが後段を送る） ----

class Finished:
    """包んだ判定のスレッドが終わるのを待てるようにする（見捨てられたスレッドの動きを最後まで見届ける）。"""

    def __init__(self, inner):
        self.inner, self.done = inner, threading.Event()

    def check(self, payload):
        try:
            return self.inner.check(payload)
        finally:
            self.done.set()

    def route(self, user, request, input):  # noqa: A002
        try:
            return self.inner.route(user, request, input)
        finally:
            self.done.set()


def _held(handler, release: threading.Event):
    """番号を受けたあと、release まで応答を返さない（guard が先に時間切れになる）。"""

    def h(req):
        out = handler(req)
        assert release.wait(10)
        return out

    return h


def _see_it_through(wrapped: Finished, release: threading.Event) -> None:
    release.set()
    assert wrapped.done.wait(10)


def test_ft_sends_no_later_part_after_the_guard_timed_out():
    c = cand_b()
    srv, release = FTServer(0.02, 0.03), threading.Event()
    wrapped = Finished(Bound(ft(lambda req: _held(srv, release)(req) if not srv.seen else srv(req)), c.rendered))
    v = guarded_c2(wrapped, c.payload, 0.1, 4000)
    assert v.decision == "hold" and v.reason.startswith("C2 timeout") and v.judge_request_id == srv.seen[0]
    _see_it_through(wrapped, release)
    # 結果を返した後に T を送ると、その番号はどこにも残らない
    assert len(srv.seen) == 1


def test_c2_cascade_sends_no_second_stage_after_the_guard_timed_out():
    j1, j2, release = Judge(c2_body(0.3)), Judge(c2_body(0.01)), threading.Event()
    wrapped = Finished(CascadeC2Gate(gate(_held(j1, release)), gate(j2)))
    v = guarded_c2(wrapped, payload(), 0.1, 10_000)
    assert v.decision == "hold" and v.judge_request_id == j1.seen[0]
    _see_it_through(wrapped, release)
    assert j2.seen == []


def test_call_c2_sends_no_second_stage_after_the_guard_timed_out():
    j1, j2, release = Judge(c2_body(0.3)), Judge(c2_body(0.01)), threading.Event()
    wrapped = Finished(CascadeC2Gate(gate(_held(j1, release)), gate(j2)))
    v = call_c2(wrapped, payload(), 0.1)
    assert v.decision == "hold" and v.judge_request_id == j1.seen[0]
    _see_it_through(wrapped, release)
    assert j2.seen == []


def test_c1_cascade_sends_no_second_stage_after_the_guard_timed_out():
    j1, j2, release = Judge(c1_body("local", 0.4)), Judge(c1_body("code", 0.9)), threading.Event()
    wrapped = Finished(CascadeC1Router(router(_held(j1, release)), router(j2), escalate_below=0.6))
    d = guarded_c1(wrapped, "u", "r", None, 0.1, 10_000)
    assert d.route == "human" and d.judge_request_id == j1.seen[0]
    _see_it_through(wrapped, release)
    assert j2.seen == []


def test_audit_db_has_every_id_the_server_saw_after_a_c2_timeout(tmp_path):
    j1, j2, release = Judge(c2_body(0.3)), Judge(c2_body(0.01)), threading.Event()
    wrapped = Finished(CascadeC2Gate(gate(_held(j1, release)), gate(j2)))
    row = _audit_row(tmp_path, c1=_StubC1(), c2=wrapped, c2_timeout_s=0.1)
    _see_it_through(wrapped, release)
    assert row["stopped_at"] == "C2" and row["c2_judge_request_id"] == j1.seen[0] and j2.seen == []


def test_audit_db_has_every_id_the_server_saw_after_a_c1_timeout(tmp_path):
    j1, j2, release = Judge(c1_body("local", 0.4)), Judge(c1_body("code", 0.9)), threading.Event()
    wrapped = Finished(CascadeC1Router(router(_held(j1, release)), router(j2), escalate_below=0.6))
    row = _audit_row(tmp_path, c1=wrapped, c2=gate(Judge()), c1_timeout_s=0.1)
    _see_it_through(wrapped, release)
    assert row["stopped_at"] == "C1" and row["c1_judge_request_id"] == j1.seen[0] and j2.seen == []


def test_every_id_sent_while_racing_the_timeout_is_recorded():
    # 時間切れと次の要求が競っても、サーバが受けた番号はすべて結果に載る
    c = cand_b()
    timed_out = 0
    for timeout in (0.001, 0.003, 0.005, 0.007, 0.01) * 4:
        srv = FTServer(0.02, 0.03)

        def slow(req, srv=srv):
            out = srv(req)
            time.sleep(0.005)  # 時間切れが F の途中にも、T を送る前後にも当たるように
            return out

        wrapped = Finished(Bound(ft(slow), c.rendered))
        v = guarded_c2(wrapped, c.payload, timeout, 4000)
        assert wrapped.done.wait(10)
        if v.reason.startswith("C2 timeout"):
            timed_out += 1
            assert srv.seen and all(s in f"{v.judge_request_id} {v.reason}" for s in srv.seen)
    assert timed_out  # 1 回も時間切れにならなければ、この試験は何も確かめていない


def test_a_timeout_does_not_stop_requests_of_a_later_judgement():
    j1, release = Judge(c2_body(0.01)), threading.Event()
    g = gate(_held(j1, release))
    wrapped = Finished(g)
    assert guarded_c2(wrapped, payload(), 0.1, 10_000).decision == "hold"
    _see_it_through(wrapped, release)
    # 同じゲートを次の判定で使える（止めるのは見捨てた判定の要求だけ）
    assert guarded_c2(g, payload(), 5.0, 10_000).decision == "allow" and len(j1.seen) == 2


def test_spec_gate_keeps_the_id():
    c = cand_b()
    srv = FTServer(0.02)
    v = SpecC2Gate("http://j", "m", "F", transport=httpx.MockTransport(srv)).check_rendered(c.payload, c.rendered)
    assert v.judge_request_id == srv.seen[0] and REQUEST_ID_RE.fullmatch(v.judge_request_id)


# ---- 評価ハーネスの配線（規則 7） ----

class EvalServer:
    def __init__(self, revision="r1", healthz_revision=None):
        self.revision, self.healthz_revision = revision, healthz_revision or revision
        self.calls: list[str] = []
        self.ids: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req.url.path)
        if req.method == "GET":
            return httpx.Response(200, json={"model": "m", "revision": self.healthz_revision})
        self.ids.append(req.headers[REQUEST_ID_HEADER])
        body = json.loads(req.content)
        q = body["questions"][0]
        if q["id"] == "route":
            return httpx.Response(200, json=c1_body("B", revision=self.revision))
        return httpx.Response(200, json={
            "model": "m", "revision": self.revision,
            "answers": [{"id": q["id"], "answer": "no", "probs": {"yes": 0.01, "no": 0.99}}]})


def _one_case(tmp_path):
    cases = [c for c in eval_h.load_cases() if c["id"] == "V011"]
    assert cases
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(cases[0], ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _eval_main(tmp_path, monkeypatch, srv, *argv):
    make = eval_h.make_impls
    monkeypatch.setattr(eval_h, "make_impls", lambda *a, **k: make(*a, **k, transport=httpx.MockTransport(srv)))
    out = tmp_path / "out"
    code = eval_h.main(["--c1", "http", "--c2", "http", "--base-url", "http://j", "--c1-model", "m", "--c2-model", "m",
                        "--cases", str(_one_case(tmp_path)), "--out", str(out), *argv])
    return code, json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_run_v2_eval_passes_the_revision_calls_ready_and_records_ids(tmp_path, monkeypatch):
    srv = EvalServer()
    code, res = _eval_main(tmp_path, monkeypatch, srv, "--c1-expected-revision", "r1", "--c2-expected-revision", "r1")
    assert code == 0 and srv.calls[:2] == ["/healthz", "/healthz"] and srv.calls[2:] == ["/v1/systemone"] * 2
    assert res["expected_revision"] == {"c1": "r1", "c2": "r1"}
    assert res["ready"] == {"c1": {"model": "m", "revision": "r1"}, "c2": {"model": "m", "revision": "r1"}}
    row = res["cases"][0]
    assert [row["c1_judge_request_id"], row["c2_judge_request_id"]] == srv.ids


def test_run_v2_eval_stops_before_judging_when_ready_disagrees(tmp_path, monkeypatch):
    srv = EvalServer(healthz_revision="r2")
    with pytest.raises(SystemExit, match="revision mismatch"):
        _eval_main(tmp_path, monkeypatch, srv, "--c1-expected-revision", "r1")
    assert srv.ids == []


def test_run_v2_eval_response_of_another_revision_stops(tmp_path, monkeypatch):
    srv = EvalServer(revision="r2", healthz_revision="r1")
    code, res = _eval_main(tmp_path, monkeypatch, srv, "--c1-expected-revision", "r1")
    row = res["cases"][0]
    assert row["route"] == "human" and "revision mismatch" in row["c1_reason"]
    assert row["c1_judge_request_id"] == srv.ids[0] and row["c2"] is None


def test_run_v2_eval_default_call_does_not_ask_healthz(tmp_path, monkeypatch, capsys):
    srv = EvalServer()
    code, res = _eval_main(tmp_path, monkeypatch, srv)
    assert code == 0 and "/healthz" not in srv.calls and res["ready"] == {"c1": None, "c2": None}
    assert res["expected_revision"] == {"c1": None, "c2": None}
    assert res["cases"][0]["c1_judge_request_id"] == srv.ids[0]
    assert "--c1-expected-revision is not set" in capsys.readouterr().err


def test_make_impls_old_positional_call_and_passing_the_revision():
    r, g = eval_h.make_impls("rules", "rules", None, "a", "b")
    assert type(r).__name__ == "RuleC1Router" and type(g).__name__ == "RuleC2Gate"
    r, g = eval_h.make_impls("http", "http", "http://j", "a", "b", 3.0)
    assert r._c.expected_revision is None and g._c.expected_revision is None and g._c.timeout_s == 3.0
    r, g = eval_h.make_impls("http", "http", "http://j", "a", "b", c1_expected_revision="x", c2_expected_revision="y")
    assert (r._c.expected_revision, g._c.expected_revision) == ("x", "y")


def test_expected_revision_is_refused_for_rules(tmp_path):
    with pytest.raises(SystemExit):
        eval_h.main(["--c2-expected-revision", "r1", "--out", str(tmp_path)])


def _compare(tmp_path, srv, method, *extra, timeout="5"):
    args = cmp_h.parse_args(["--method", method, "--set", "B15", "--base-url", "http://j", "--timeout", timeout,
                             "--out-dir", str(tmp_path / "out"), *extra])
    path, summary = cmp_h.run(args, git=lambda: ("0" * 40, []), gate_kw={"transport": httpx.MockTransport(srv)},
                              policy=load_policy())
    return json.loads(path.read_text(encoding="utf-8"))


class CompareServer:
    def __init__(self, revision="r1", healthz_revision=None):
        self.revision, self.healthz_revision = revision, healthz_revision or revision
        self.calls: list[str] = []
        self.ids: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req.url.path)
        body = json.loads(req.content) if req.content else {}
        if req.method == "GET":
            return httpx.Response(200, json={"model": "clef-flash", "revision": self.healthz_revision})
        self.ids.append(req.headers[REQUEST_ID_HEADER])
        return httpx.Response(200, json={"model": body["model"], "revision": self.revision, "answers": [
            {"id": q["id"], "answer": "no", "probs": {"yes": 0.01, "no": 0.99}} for q in body["questions"]]})


@pytest.mark.parametrize("method", ["Q0", "F"])
def test_run_c2_compare_passes_the_revision_calls_ready_and_records_ids(tmp_path, method):
    srv = CompareServer()
    doc = _compare(tmp_path, srv, method, "--expected-revision", "r1")
    assert srv.calls[0] == "/healthz" and srv.calls[1:] == ["/v1/systemone"] * (len(srv.calls) - 1)
    assert doc["header"]["expected_revision"] == "r1"
    assert doc["header"]["ready"] == {"model": "clef-flash", "revision": "r1"}
    assert [r["judge_request_id"] for r in doc["cases"]] == srv.ids


def test_run_c2_compare_ft_records_the_component_ids(tmp_path):
    srv = CompareServer()
    doc = _compare(tmp_path, srv, "FT", "--expected-revision", "r1", timeout="60")
    assert doc["header"]["ready"]["revision"] == "r1" and srv.calls.count("/healthz") == 1
    first = doc["cases"][0]
    assert first["judge_request_id"] == ",".join(srv.ids[:2])
    assert [first["components"][m]["judge_request_id"] for m in ("F", "T")] == srv.ids[:2]


def test_run_c2_compare_stops_before_judging_when_ready_disagrees(tmp_path):
    srv = CompareServer(healthz_revision="r2")
    with pytest.raises(SystemExit, match="revision mismatch"):
        _compare(tmp_path, srv, "F", "--expected-revision", "r1")
    assert srv.ids == [] and not (tmp_path / "out").exists()


def test_run_c2_compare_response_of_another_revision_is_failure_hold(tmp_path):
    srv = CompareServer(revision="r2", healthz_revision="r1")
    doc = _compare(tmp_path, srv, "F", "--expected-revision", "r1")
    assert {r["bucket"] for r in doc["cases"]} == {"failure_hold"}
    assert [r["judge_request_id"] for r in doc["cases"]] == srv.ids


def test_run_c2_compare_default_call_does_not_ask_healthz(tmp_path, capsys):
    srv = CompareServer()
    doc = _compare(tmp_path, srv, "F")
    assert "/healthz" not in srv.calls and doc["header"]["expected_revision"] is None and doc["header"]["ready"] is None
    assert "--expected-revision is not set" in capsys.readouterr().err


# ---- 応答の上限（規則 8） ----

def test_limit_is_one_mib():
    assert MAX_RESPONSE_BYTES == 1024 * 1024


def _padded(size: int) -> bytes:
    raw = json.dumps(c2_body()).encode()
    return raw + b" " * (size - len(raw))


def test_default_limit_is_applied_exactly():
    v = gate(Judge(chunks=[_padded(MAX_RESPONSE_BYTES)])).check(payload())
    assert v.decision == "allow"
    with pytest.raises(MalformedResponse, match="too large"):
        gate(Judge(chunks=[_padded(MAX_RESPONSE_BYTES + 1)])).check(payload())


class CountingChunks:
    """どこまで読まれたかを覚える本文。"""

    def __init__(self, n: int, chunk: bytes = b"x" * 1024):
        self.n, self.chunk, self.read = n, chunk, 0

    def __iter__(self):
        for _ in range(self.n):
            self.read += 1
            yield self.chunk


def test_reading_stops_at_the_limit():
    body = CountingChunks(1000)
    with pytest.raises(MalformedResponse, match="too large"):
        gate(Judge(chunks=body), max_response_bytes=10_000).check(payload())
    assert body.read <= 11


def _health_body(body) -> httpx.MockTransport:
    return httpx.MockTransport(lambda req: httpx.Response(200, content=iter(body)))


def test_ready_has_the_limit_too():
    body = CountingChunks(1000)
    with pytest.raises(MalformedResponse, match="too large"):
        HttpC2Gate("http://j", "m", transport=_health_body(body), max_response_bytes=10_000).ready()
    assert body.read <= 11
    with pytest.raises(MalformedResponse, match="too large"):
        HttpC2Gate("http://j", "m", transport=_health_body([_padded(MAX_RESPONSE_BYTES + 1)])).ready()


def test_ready_declared_too_large_is_refused():
    def h(req):
        return httpx.Response(200, headers={"content-length": str(MAX_RESPONSE_BYTES + 1)}, content=iter([b"{}"]))

    with pytest.raises(MalformedResponse, match="too large"):
        HttpC2Gate("http://j", "m", transport=httpx.MockTransport(h)).ready()


# ---- 圧縮された応答 ----

def _gz(data: bytes) -> bytes:
    return gzip.compress(data, 9)


def test_no_compression_is_requested():
    seen = []

    def h(req):
        seen.append(req.headers["accept-encoding"])
        if req.method == "GET":
            return httpx.Response(200, json={"model": "m", "revision": "r1"})
        return httpx.Response(200, json=c2_body())

    g = HttpC2Gate("http://j", "m", transport=httpx.MockTransport(h))
    g.check(payload())
    g.ready()
    assert seen == ["identity", "identity"]


@pytest.mark.parametrize("encoding, compress", [("gzip", _gz), ("deflate", zlib.compress), ("GZIP", _gz)])
def test_small_compressed_response_is_read(encoding, compress):
    raw = compress(json.dumps(c2_body()).encode())
    j = Judge(chunks=[raw[:5], raw[5:]], headers={"content-encoding": encoding})
    assert gate(j).check(payload()).decision == "allow"


def test_compressed_limit_counts_the_expanded_size():
    raw = _padded(5000)
    hdr = {"content-encoding": "gzip"}
    assert gate(Judge(chunks=[_gz(raw)], headers=hdr), max_response_bytes=5000).check(payload()).decision == "allow"
    with pytest.raises(MalformedResponse, match="too large"):
        gate(Judge(chunks=[_gz(raw)], headers=hdr), max_response_bytes=4999).check(payload())


def test_compressed_raw_bytes_count_too():
    # 展開しても小さいが、生のまま大きい本文（空の圧縮の繰り返しなど）も読み続けない
    raw = _gz(b"{}")
    with pytest.raises(MalformedResponse, match="too large"):
        gate(Judge(chunks=[raw + b"\0" * 5000], headers={"content-encoding": "gzip"}),
             max_response_bytes=1000).check(payload())


@pytest.mark.parametrize("use_ready", [False, True])
def test_gzip_bomb_stops_while_expanding(use_ready):
    bomb = _gz(b" " * (64 * 1024 * 1024))
    assert len(bomb) < MAX_RESPONSE_BYTES
    hdr = {"content-encoding": "gzip"}
    g = HttpC2Gate("http://j", "m", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, headers=hdr, content=iter([bomb]))))
    tracemalloc.start()
    try:
        with pytest.raises(MalformedResponse, match="too large"):
            g.ready() if use_ready else g.check(payload())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 * 1024 * 1024  # 上限 1 MiB と圧縮本文の分だけ。丸ごと展開すれば 64 MiB を超える


@pytest.mark.parametrize("headers, chunks, why", [
    ({"content-encoding": "br"}, [b"x"], "unsupported"),
    ({"content-encoding": "gzip, gzip"}, [b"x"], "unsupported"),
    ({"content-encoding": "gzip"}, [b"not gzip at all"], "broken"),
    ({"content-encoding": "gzip"}, [_gz(b"{}")[:6]], "ended early"),
    ({"content-encoding": "gzip"}, [_gz(b"{}") + b"junk"], "after the end"),
    ({"content-encoding": "gzip"}, [_gz(b"{}"), b"junk"], "after the end"),
])
def test_bad_compression_is_malformed_and_guard_holds(headers, chunks, why):
    with pytest.raises(MalformedResponse, match=why):
        gate(Judge(chunks=chunks, headers=headers)).check(payload())
    j = Judge(chunks=chunks, headers=headers)
    v = guarded_c2(gate(j), payload(), 5.0, 10_000)
    assert v.decision == "hold" and why in v.reason and v.judge_request_id == j.seen[0]


def test_gzip_bomb_through_c1_guard_goes_human_with_the_id():
    j = Judge(chunks=[_gz(b" " * (8 * 1024 * 1024))], headers={"content-encoding": "gzip"})
    d = guarded_c1(router(j), "u", "r", None, 5.0, 10_000)
    assert d.route == "human" and "too large" in d.reason and d.judge_request_id == j.seen[0]


# ---- ready の全体の期限 ----

class Drip:
    """少しずつ、いつまでも返す相手。読まれた数を覚える。"""

    def __init__(self, step_s: float = 0.02, steps: int = 500):
        self.step_s, self.steps, self.read = step_s, steps, 0

    def __iter__(self):
        for _ in range(self.steps):
            time.sleep(self.step_s)
            self.read += 1
            yield b" "


def test_ready_keeps_the_overall_deadline_against_a_drip():
    body = Drip()
    g = HttpC2Gate("http://j", "m", timeout_s=0.2, transport=_health_body(body))
    t0 = time.perf_counter()
    with pytest.raises(httpx.TimeoutException):
        g.ready()
    assert time.perf_counter() - t0 < 1.0
    # 期限で見捨てたスレッドも読み続けない
    time.sleep(0.3)
    read = body.read
    time.sleep(0.2)
    assert body.read == read < body.steps


def test_ready_keeps_the_overall_deadline_against_a_silent_server():
    def h(req):
        time.sleep(1.0)
        return httpx.Response(200, json={"model": "m", "revision": "r"})

    t0 = time.perf_counter()
    with pytest.raises(httpx.TimeoutException):
        HttpC2Gate("http://j", "m", timeout_s=0.1, transport=httpx.MockTransport(h)).ready()
    assert time.perf_counter() - t0 < 0.6


def test_ready_deadline_does_not_touch_a_normal_answer():
    assert HttpC2Gate("http://j", "m", timeout_s=2.0, transport=_health_body(
        [json.dumps({"model": "m", "revision": "r"}).encode()])).ready() == {"model": "m", "revision": "r"}


@pytest.mark.parametrize("gate_factory", [
    lambda t: SpecC2Gate("http://j", "clef-flash", "F", transport=t),
    lambda t: FTC2Gate("http://j", "clef-flash", transport=t),
])
def test_spec_and_ft_gates_have_ready(gate_factory):
    t = httpx.MockTransport(lambda req: httpx.Response(200, json={"model": "clef-flash", "revision": "r"}))
    assert gate_factory(t).ready() == {"model": "clef-flash", "revision": "r"}


# ---- 番号の形（shim は形の違う値を返さない・残さない） ----

BAD_IDS = ["a" * 32 + "\n", "a" * 31, "a" * 33, "A" * 32, "bad id\nINJECT"]
Q = {"id": "q", "type": "noul", "prompt": "?"}


def _laya_client():
    from judge.laya_server import create_app
    from tests.judge.test_laya_server import REV, FakeLaya

    log = io.StringIO()
    return TestClient(create_app(FakeLaya(), model_name="m", revision=REV, log=log)), log


def _clef_client():
    from judge.clef_server import create_app
    from tests.judge.test_clef_server import REV, UP, FakeLlama

    log = io.StringIO()
    upstream = httpx.Client(base_url="http://llama", transport=httpx.MockTransport(FakeLlama().handler))
    return TestClient(create_app(upstream, UP, model_name="m", revision=REV, log=log)), log


@pytest.mark.parametrize("client", [_laya_client, _clef_client], ids=["laya", "clef"])
@pytest.mark.parametrize("bad", BAD_IDS)
def test_shims_do_not_echo_or_log_a_bad_id(client, bad):
    tc, log = client()
    r = tc.post("/v1/systemone", headers={"X-Request-Id": bad},
                json={"model": "m", "context": "x", "questions": [Q]})
    assert r.status_code == 200 and "x-request-id" not in r.headers
    assert json.loads(log.getvalue().splitlines()[-1])["req"] is None


@pytest.mark.parametrize("client", [_laya_client, _clef_client], ids=["laya", "clef"])
def test_shims_echo_a_good_id(client):
    tc, log = client()
    r = tc.post("/v1/systemone", headers={"X-Request-Id": ID}, json={"model": "m", "context": "x", "questions": [Q]})
    assert r.headers["x-request-id"] == ID and json.loads(log.getvalue().splitlines()[-1])["req"] == ID


def test_adapter_pattern_needs_the_whole_value():
    assert REQUEST_ID_RE.fullmatch("a" * 32) and not REQUEST_ID_RE.fullmatch("a" * 32 + "\n")
