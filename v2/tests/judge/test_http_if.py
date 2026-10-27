"""判断モデルの HTTP I/F（docs/redesign-v2.md「判断モデルの HTTP I/F」）の検証規則と 2 段構え。"""
from __future__ import annotations

import httpx
import pytest

from common.schemas import Message, SendPayload
from judge.base import C1Decision, C2Verdict, guarded_c1, guarded_c2
from judge.http_adapter import (
    C1_QUESTION_VERSION,
    C2_QUESTION_VERSION,
    ROUTES,
    CascadeC1Router,
    CascadeC2Gate,
    HttpC1Router,
    HttpC2Gate,
    MalformedResponse,
    validate_answer,
)


def payload() -> SendPayload:
    return SendPayload(contract="B-csv-codegen@1", destination="codex",
                       messages=(Message(role="user", content="ok"),), max_output_chars=3000)


def route_probs(top: str, p: float = 0.9) -> dict[str, float]:
    rest = (1 - p) / (len(ROUTES) - 1)
    return {r: (p if r == top else rest) for r in ROUTES}


def c2_body(p: float, **over):
    body = {"model": "m", "revision": "r1",
            "answers": [{"id": "forbidden", "answer": "yes" if p >= .5 else "no", "probs": {"yes": p, "no": 1 - p}}]}
    body.update(over)
    return body


def gate(body=None, *, status=200, **kw) -> HttpC2Gate:
    return HttpC2Gate("http://j", "m", transport=httpx.MockTransport(lambda req: httpx.Response(status, json=body)), **kw)


def router(top: str, p: float = 0.9, *, model: str = "m", **kw) -> HttpC1Router:
    body = {"model": model, "revision": "r1", "answers": [{"id": "route", "answer": top, "probs": route_probs(top, p)}]}
    return HttpC1Router("http://j", model, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)), **kw)


# ---- 1 問分の検証 ----

NOUL = {"id": "q", "type": "noul", "prompt": "?"}
CHOICE = {"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b", "c"]}
SCORE = {"id": "q", "type": "score", "prompt": "?", "min": 1, "max": 3}


@pytest.mark.parametrize("q,a,expected", [
    (NOUL, {"answer": "no", "probs": {"yes": 0.2, "no": 0.8}}, "no"),
    (CHOICE, {"answer": "b", "probs": {"a": 0.1, "b": 0.6, "c": 0.3}}, "b"),
    (SCORE, {"answer": 2, "probs": {"1": 0.1, "2": 0.7, "3": 0.2}}, "2"),
    (NOUL, {"answer": "yes", "probs": {"yes": 0.5, "no": 0.5}}, "yes"),  # 同率の最大はどちらも可
    (NOUL, {"answer": "yes", "probs": {"yes": 0.504, "no": 0.5}}, "yes"),  # 合計 1±0.01 まで許す
])
def test_validate_answer_ok(q, a, expected):
    assert validate_answer(q, a)[0] == expected


@pytest.mark.parametrize("q,a", [
    (NOUL, {"answer": "yes", "probs": {"yes": 0.9}}),  # ラベル不足
    (NOUL, {"answer": "yes", "probs": {"yes": 0.9, "no": 0.05, "maybe": 0.05}}),  # 余分なラベル
    (NOUL, {"answer": "yes", "probs": {"yes": 0.9, "no": 0.9}}),  # 合計が 1 でない
    (NOUL, {"answer": "yes", "probs": {"yes": 0.3, "no": 0.7}}),  # answer が最大確率でない
    (NOUL, {"answer": "yes", "probs": {"yes": float("nan"), "no": 0.0}}),
    (NOUL, {"answer": "yes", "probs": {"yes": True, "no": 0.0}}),
    (NOUL, {"answer": "yes", "probs": {"yes": 1.2, "no": -0.2}}),
    (NOUL, {"answer": "maybe", "probs": {"yes": 0.5, "no": 0.5}}),
    (NOUL, {"probs": {"yes": 0.5, "no": 0.5}}),  # answer なし
    (SCORE, {"answer": "2", "probs": {"1": 0.1, "2": 0.7, "3": 0.2}}),  # score は整数で返す
    (SCORE, {"answer": True, "probs": {"1": 0.1, "2": 0.7, "3": 0.2}}),
    (CHOICE, "b"),
])
def test_validate_answer_rejects(q, a):
    with pytest.raises(MalformedResponse):
        validate_answer(q, a)


# ---- 応答全体の検証（どれも guard で hold / human に倒れる） ----

@pytest.mark.parametrize("body", [
    c2_body(0.0, model="other-model"),  # 頼んだモデル以外が答えた
    c2_body(0.0, truncated="no"),
    c2_body(0.0, revision=""),
    c2_body(0.0, answers=[{"id": "forbidden", "answer": "no", "probs": {"yes": 0.0, "no": 1.0}}] * 2),  # 重複
    c2_body(0.0, answers=[]),
])
def test_malformed_body_holds(body):
    with pytest.raises(MalformedResponse):
        gate(body).check(payload())
    assert guarded_c2(gate(body), payload(), 2.0, 1000).decision == "hold"


@pytest.mark.parametrize("status", [301, 400, 404, 429, 500, 503])
def test_non_2xx_holds_without_retry(status):
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(status, json=c2_body(0.0), headers={"location": "http://elsewhere/"})

    g = HttpC2Gate("http://j", "m", transport=httpx.MockTransport(h))
    assert guarded_c2(g, payload(), 2.0, 1000).decision == "hold"
    assert len(calls) == 1  # 再試行もリダイレクト追従もしない


def test_http_timeout_holds():
    def h(req):
        raise httpx.ReadTimeout("slow", request=req)
    g = HttpC2Gate("http://j", "m", transport=httpx.MockTransport(h))
    v = guarded_c2(g, payload(), 2.0, 1000)
    assert v.decision == "hold" and "ReadTimeout" in v.reason


def test_model_absent_in_response_is_recorded_as_requested():
    body = c2_body(0.0)
    del body["model"]
    v = gate(body).check(payload())
    assert v.model == "m" and v.decision == "allow"


# ---- C1 の確信度下限 ----

def test_c1_min_confidence_routes_low_confidence_to_human():
    d = router("A", 0.4, min_confidence=0.6).route("u", "q", None)
    assert d.route == "human" and d.destination is None and "low confidence" in d.reason
    assert d.probs["A"] == pytest.approx(0.4)


def test_c1_min_confidence_keeps_confident_route():
    assert router("local", 0.9, min_confidence=0.6).route("u", "q", None).route == "local"


# ---- 2 段構え ----

class Spy:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def check(self, p):
        self.calls += 1
        return self.inner.check(p)

    def route(self, *a):
        self.calls += 1
        return self.inner.route(*a)


@pytest.mark.parametrize("p1,expected,escalated", [
    (0.9, "block", False),
    (0.05, "allow", False),
    (0.3, "allow", True),  # 前段が hold → 後段（p=0.0）が allow
])
def test_cascade_c2_escalates_only_on_hold(p1, expected, escalated):
    second = Spy(HttpC2Gate("http://j2", "clef", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=c2_body(0.0, model="clef", revision="c1")))))
    v = CascadeC2Gate(gate(c2_body(p1)), second).check(payload())
    assert v.decision == expected and second.calls == int(escalated)
    if escalated:
        assert (v.model, v.revision) == ("clef", "c1") and "escalated from m@r1" in v.reason
        assert v.question_version == C2_QUESTION_VERSION


def test_cascade_c2_truncated_first_escalates():
    second = Spy(gate(c2_body(0.0)))
    v = CascadeC2Gate(gate(c2_body(0.0, truncated=True)), second).check(payload())
    assert second.calls == 1 and v.decision == "allow" and not v.truncated


def test_cascade_c2_second_block_wins_and_hold_stays_hold():
    assert CascadeC2Gate(gate(c2_body(0.3)), gate(c2_body(0.95))).check(payload()).decision == "block"
    assert CascadeC2Gate(gate(c2_body(0.3)), gate(c2_body(0.3))).check(payload()).decision == "hold"


def test_cascade_c2_first_error_does_not_fall_back():
    second = Spy(gate(c2_body(0.0)))
    v = guarded_c2(CascadeC2Gate(gate({}, status=500), second), payload(), 2.0, 1000)
    assert v.decision == "hold" and second.calls == 0


def test_cascade_c2_second_error_holds():
    v = guarded_c2(CascadeC2Gate(gate(c2_body(0.3)), gate({}, status=500)), payload(), 2.0, 1000)
    assert v.decision == "hold"


def test_cascade_c1_confident_first_is_used():
    second = Spy(router("A", model="clef"))
    d = CascadeC1Router(router("local", 0.9), second, escalate_below=0.7).route("u", "q", None)
    assert d.route == "local" and second.calls == 0


def test_cascade_c1_low_confidence_escalates():
    second = Spy(router("reject", 0.95, model="clef"))
    d = CascadeC1Router(router("local", 0.5), second, escalate_below=0.7).route("u", "q", None)
    assert (d.route, d.model, second.calls) == ("reject", "clef", 1)
    assert "escalated from m@r1 route=local top=0.500" in d.reason and d.question_version == C1_QUESTION_VERSION


def test_cascade_c1_second_also_unsure_goes_human():
    d = CascadeC1Router(router("A", 0.5), router("A", 0.5, model="clef", min_confidence=0.7),
                        escalate_below=0.7).route("u", "q", None)
    assert d.route == "human"


def test_cascade_c1_through_guard_first_error_is_human():
    class Boom:
        def route(self, *a):
            raise RuntimeError("down")
    second = Spy(router("local"))
    d = guarded_c1(CascadeC1Router(Boom(), second, escalate_below=0.7), "u", "q", None, 2.0, 1000)
    assert d.route == "human" and second.calls == 0


def test_cascade_accepts_plain_schema_results():
    class P1:
        def check(self, p):
            return C2Verdict(decision="hold", reason="x", model="a", revision="1", question_version="q")
    class P2:
        def check(self, p):
            return {"decision": "block", "reason": "y", "model": "b", "revision": "2", "question_version": "q"}
    assert CascadeC2Gate(P1(), P2()).check(payload()).decision == "block"

    class R:
        def route(self, *a):
            return C1Decision(route="local", reason="r", model="a", revision="1", probs={}, question_version="q")
    # probs が空なら確信度 0 とみなして後段へ
    second = Spy(router("code"))
    assert CascadeC1Router(R(), second, escalate_below=0.5).route("u", "q", None).route == "code"


@pytest.mark.parametrize("kw", [{"escalate_below": 1.5}, {"escalate_below": -0.1}])
def test_cascade_c1_rejects_bad_threshold(kw):
    with pytest.raises(ValueError):
        CascadeC1Router(router("local"), router("local"), **kw)


def test_c1_rejects_bad_min_confidence():
    with pytest.raises(ValueError):
        router("local", min_confidence=2.0)
