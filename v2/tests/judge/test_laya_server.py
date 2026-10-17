"""Laya 変換サーバ（judge/laya_server.py）。偽の Laya を置き、本物の C1 / C2 アダプタから HTTP で通して確かめる。

偽の Laya の出力は laya 0.4.9 + laya-multilingual（1720e3e）の実測の形をそのまま写している
（probabilities は小数 4 桁・noul は P(true)・切り捨ては usage.truncated と truncated_questions）。
"""
from __future__ import annotations

import io
import json
import sys
import types

import httpx
import pytest
from fastapi.testclient import TestClient

from common.schemas import Message, SendPayload
from judge.base import guarded_c1, guarded_c2
from judge.http_adapter import C1_QUESTIONS, C2_QUESTIONS, ROUTES, HttpC1Router, HttpC2Gate
from judge.laya_server import SHIM_VERSION, create_app, load_agent

REV = "1720e3e3357cfe1e281542e223f8273b0890ca34"
OK_USAGE = {"input_tokens": 162, "output_tokens": 0, "state_tokens": 31, "state_tokens_dropped": 0,
            "truncated": False, "truncated_questions": []}
# 実測（2.1 万字・max_len=8192）の切り捨て時の usage
CUT_USAGE = {"input_tokens": 8192, "output_tokens": 0, "state_tokens": 16290, "state_tokens_dropped": 8178,
             "truncated": True, "truncated_questions": ["forbidden"]}


class FakeLaya:
    def __init__(self, *, route=None, noul=0.05, usage=None, raises=None, answers=None):
        self.route = route or {"code": 0.1466, "local": 0.8017, "A": 0.0045, "B": 0.0037, "human": 0.0213, "reject": 0.0222}
        self.noul, self.usage, self.raises, self.answers = noul, usage, raises, answers
        self.calls: list[tuple[str, dict, dict]] = []

    def system_one(self, state, questions, **kw):
        self.calls.append((state, questions, kw))
        if self.raises:
            raise self.raises
        if self.answers is not None:
            return {"model": "laya-rl-agent", "answers": self.answers, "usage": self.usage or OK_USAGE}
        ans = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                ans[qid] = {"type": "choice", "choice": max(self.route, key=self.route.get),
                            "probabilities": dict(self.route), "confidence": 0.3}
            else:
                ans[qid] = {"type": "noul", "noul": self.noul, "confidence": max(self.noul, 1 - self.noul)}
        out = {"model": "laya-rl-agent", "answers": ans}
        if self.usage != "missing":
            out["usage"] = self.usage or OK_USAGE
        return out


def serve(agent, **kw):
    log = io.StringIO()
    app = create_app(agent, model_name="laya-multilingual", revision=REV, log=log, **kw)
    tc = TestClient(app)

    def handler(req: httpx.Request) -> httpx.Response:
        r = tc.request(req.method, req.url.path, content=req.content, headers={"content-type": "application/json"})
        return httpx.Response(r.status_code, content=r.content, headers={"content-type": r.headers.get("content-type", "")})

    return httpx.MockTransport(handler), tc, log


def c1(agent, **kw) -> HttpC1Router:
    return HttpC1Router("http://judge", "laya-multilingual", transport=serve(agent)[0], **kw)


def c2(agent, **kw) -> HttpC2Gate:
    return HttpC2Gate("http://judge", "laya-multilingual", transport=serve(agent)[0], **kw)


def payload(text: str = "ok") -> SendPayload:
    return SendPayload(contract="B-csv-codegen@1", destination="codex",
                       messages=(Message(role="user", content=text),), max_output_chars=3000)


def post(tc, body):
    return tc.post("/v1/systemone", json=body)


def body(questions=None, **over):
    b = {"model": "laya-multilingual", "context": "架空の依頼",
         "questions": questions if questions is not None else [{"id": "q", "type": "noul", "prompt": "?"}]}
    b.update(over)
    return b


# ---- 写し方 ----

def test_c1_route_through_adapter():
    fake = FakeLaya()
    d = c1(fake).route("u1", "4月1日から6月30日までの日数を数えて", None)
    assert d.route == "local" and d.probs["local"] == 0.8017 and set(d.probs) == set(ROUTES)
    assert d.revision == f"{REV}+{SHIM_VERSION}" and d.truncated is False


def test_c1_descriptions_reach_laya_as_criteria():
    fake = FakeLaya()
    c1(fake).route("u1", "依頼", None)
    state, qs, kw = fake.calls[0]
    want = C1_QUESTIONS[0]
    assert qs["route"] == {"type": "choice", "instructions": want["prompt"],
                           "criteria": {c: want["descriptions"][c] for c in want["choices"]}}
    assert json.loads(state)["request"] == "依頼"
    assert kw == {"max_len": 1024}


def test_head_max_len_is_passed_only_when_set():
    fake = FakeLaya()
    HttpC1Router("http://judge", "laya-multilingual", transport=serve(fake, max_len=2048, head_max_len=300)[0]).route("u", "r", None)
    assert fake.calls[0][2] == {"max_len": 2048, "head_max_len": 300}


def test_choice_without_descriptions_sends_none():
    fake = FakeLaya(route={"a": 0.3, "b": 0.7})
    _, tc, _ = serve(fake)
    r = post(tc, body([{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b"]}]))
    assert r.status_code == 200 and fake.calls[0][1]["q"]["criteria"] == {"a": None, "b": None}
    assert r.json()["answers"] == [{"id": "q", "answer": "b", "probs": {"a": 0.3, "b": 0.7}}]


def test_answer_is_recomputed_and_ties_follow_choice_order():
    # 丸めで同点になった場合、Laya の choice ではなく選択肢の順で先のものを答える（検証規則 5 は同点ならどれでも可）
    route = {"human": 0.4, "local": 0.4, "code": 0.05, "A": 0.05, "B": 0.05, "reject": 0.05}
    fake = FakeLaya(route=route)  # 偽の Laya 自身は dict 順で human を選ぶ
    d = c1(fake).route("u", "r", None)
    assert d.route == "local"


@pytest.mark.parametrize("p,decision", [(0.7894, "block"), (0.5, "block"), (0.205, "hold"), (0.05, "allow")])
def test_c2_noul_is_p_true(p, decision):
    v = c2(FakeLaya(noul=p)).check(payload())
    assert v.decision == decision and v.prob_block == p and v.revision == f"{REV}+{SHIM_VERSION}"


def test_noul_probs_sum_to_one():
    _, tc, _ = serve(FakeLaya(noul=0.7894))
    a = post(tc, body()).json()["answers"][0]
    assert a["answer"] == "yes" and a["probs"] == {"yes": 0.7894, "no": 0.2106}


def test_c2_context_is_role_tagged():
    fake = FakeLaya()
    c2(fake).check(payload("本文"))
    assert fake.calls[0][0] == "[user]\n本文" and fake.calls[0][1]["forbidden"]["instructions"] == C2_QUESTIONS[0]["prompt"]


# ---- 切り捨て → human / hold ----

@pytest.mark.parametrize("usage", [
    CUT_USAGE,
    {**OK_USAGE, "truncated": False, "truncated_questions": ["forbidden"]},
    {**OK_USAGE, "options": {"route": ["A", "B"]}},  # 選択肢が head 予算に収まらず畳まれた
    "missing",
])
def test_truncation_holds_c2(usage):
    v = guarded_c2(c2(FakeLaya(noul=0.01, usage=usage)), payload(), 2.0, 1000)
    assert v.decision == "hold" and v.truncated is True


def test_truncation_sends_c1_to_human():
    d = guarded_c1(c1(FakeLaya(usage=CUT_USAGE)), "u", "r", None, 2.0, 1000)
    assert d.route == "human" and d.truncated is True


# ---- 受けない要求（422）→ アダプタは例外 → guard が止める ----

def test_model_mismatch_is_rejected_and_guard_stops():
    transport, tc, _ = serve(FakeLaya())
    assert post(tc, body(model="other")).status_code == 422
    d = guarded_c1(HttpC1Router("http://judge", "other", transport=transport), "u", "r", None, 2.0, 1000)
    assert d.route == "human"


@pytest.mark.parametrize("questions", [
    [{"id": "q", "type": "score", "prompt": "?", "min": 1, "max": 3}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b"], "descriptions": {"c": "x"}}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b"], "descriptions": {"a": ""}}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "a"]}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a"]}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", 1]}],
    [{"id": "q", "type": "noul", "prompt": "?", "choices": ["a", "b"]}],
    [{"id": "q", "type": "noul", "prompt": "?", "negate": True}],
    [{"id": "q", "type": "noul", "prompt": "?"}, {"id": "q", "type": "noul", "prompt": "!"}],
    [],
])
def test_bad_questions_are_rejected(questions):
    fake = FakeLaya()
    _, tc, _ = serve(fake)
    assert post(tc, body(questions)).status_code == 422 and fake.calls == []


@pytest.mark.parametrize("over", [{"context": ""}, {"context": "x" * 50_001}, {"extra": 1}])
def test_bad_requests_are_rejected(over):
    _, tc, _ = serve(FakeLaya())
    assert post(tc, body(**over)).status_code == 422


# ---- 推論の失敗・想定外の出力 → 500（本文を漏らさない）→ guard が止める ----

def test_inference_error_hides_text():
    transport, tc, log = serve(FakeLaya(raises=RuntimeError("社員番号 E12345")))
    r = post(tc, body())
    assert r.status_code == 500 and "E12345" not in r.text and "E12345" not in log.getvalue()
    assert guarded_c2(HttpC2Gate("http://judge", "laya-multilingual", transport=transport), payload(), 2.0, 1000).decision == "hold"


@pytest.mark.parametrize("answers", [
    {},
    {"q": {"type": "noul", "noul": 1.2}},
    {"q": {"type": "noul", "noul": True}},
    {"q": {"type": "noul", "noul": float("nan")}},
    {"q": {"type": "choice", "probabilities": {"yes": 0.5, "no": 0.5}}},
])
def test_unexpected_model_output_is_500(answers):
    _, tc, _ = serve(FakeLaya(answers=answers))
    assert post(tc, body()).status_code == 500


def test_log_has_counts_not_text():
    _, tc, log = serve(FakeLaya(usage=CUT_USAGE))
    post(tc, body(context="社員番号 E12345"))
    line = json.loads(log.getvalue().strip())
    assert line["state_tokens"] == 16290 and line["truncated"] is True and len(line["ctx"]) == 12
    assert "E12345" not in log.getvalue()


def test_healthz_reports_revision():
    _, tc, _ = serve(FakeLaya())
    assert tc.get("/healthz").json() == {"model": "laya-multilingual", "revision": f"{REV}+{SHIM_VERSION}"}


# ---- 起動時の版の確認 ----

class _FakeLoaded:
    def __init__(self, revision):
        self.revision = revision


@pytest.fixture
def fake_laya_module(monkeypatch):
    calls = []

    def load(model, **kw):
        calls.append((model, kw))
        return _FakeLoaded(kw.get("revision") if model != "drifted" else "other-sha")

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=load))
    return calls


def test_load_pins_hf_revision(fake_laya_module):
    load_agent("org/model", REV, device="cpu", sha256=None)
    assert fake_laya_module[0] == ("org/model", {"device": "cpu", "revision": REV, "expected_sha256": None})


def test_load_refuses_revision_drift(fake_laya_module):
    with pytest.raises(SystemExit):
        load_agent("drifted", REV, device="cpu", sha256=None)


def test_load_requires_revision(fake_laya_module):
    with pytest.raises(SystemExit):
        load_agent("org/model", "", device="cpu", sha256=None)


def test_local_dir_requires_digests(fake_laya_module, tmp_path):
    with pytest.raises(SystemExit):
        load_agent(str(tmp_path), REV, device="cpu", sha256=None)
    load_agent(str(tmp_path), REV, device="cpu", sha256={"model.safetensors": "ab" * 32})
    assert fake_laya_module[-1][1]["expected_sha256"] == {"model.safetensors": "ab" * 32}
