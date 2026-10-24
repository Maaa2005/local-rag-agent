"""Clef 変換サーバ（judge/clef_server.py）。偽の llama-server を置き、本物の C1 / C2 アダプタから HTTP で通して確かめる。

偽の llama-server の形は llama.cpp b11669（620d2225c）+ Clef-Flash Q4_K_M の実測を写している
（/props の欄・noul は P(true)・usage は input_tokens と output_tokens=0・エラーは本文付きの JSON）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from common.schemas import Message, SendPayload
from judge.base import guarded_c1, guarded_c2
from judge.clef_server import (SHIM_VERSION, Upstream, create_app, main, runtime_revision, upstream_client,
                               verify_upstream)
from judge.http_adapter import C1_QUESTIONS, C2_QUESTIONS, ROUTES, HttpC1Router, HttpC2Gate

REV = "ggml-org/Clef-Flash-GGUF@6b4fe4c/Q4_K_M"
UP = Upstream(model_path="/models/clef/Clef-Flash-Q4_K_M.gguf", build="b11669-620d2225c", n_ctx=8192, alias="clef-flash")
OK_USAGE = {"input_tokens": 175, "output_tokens": 0}


def props_for(up: Upstream) -> dict:
    return {"model_path": up.model_path, "build_info": up.build, "model_alias": up.alias, "total_slots": 1,
            "default_generation_settings": {"n_ctx": up.n_ctx}}


class FakeLlama:
    def __init__(self, *, route=None, noul=0.0302, answers=None, usage=None, status=200, raises=None,
                 model="clef-flash", props=None):
        self.route = route or {"code": 0.11, "local": 0.82, "A": 0.01, "B": 0.01, "human": 0.03, "reject": 0.02}
        self.noul, self.answers, self.usage, self.status, self.raises, self.model = noul, answers, usage, status, raises, model
        self.props = props if props is not None else props_for(UP)
        self.posts: list[dict] = []
        self.props_calls = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == "/props":
            self.props_calls += 1
            return httpx.Response(200, json=self.props)
        assert req.method == "POST" and req.url.path == "/v1/systemone"
        self.posts.append(json.loads(req.content))
        if self.raises:
            raise self.raises
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"code": self.status, "type": "server_error",
                                                               "message": "社員番号 E12345 を含む入力が長すぎる"}})
        if self.answers is not None:
            ans = self.answers
        else:
            ans = {}
            for qid, q in self.posts[-1]["questions"].items():
                if q["type"] == "choice":
                    ans[qid] = {"type": "choice", "choice": max(self.route, key=self.route.get),
                                "probabilities": dict(self.route), "confidence": 0.3}
                else:
                    ans[qid] = {"type": "noul", "noul": self.noul}
        out = {"model": self.model, "answers": ans}
        if self.usage != "missing":
            out["usage"] = self.usage or OK_USAGE
        return httpx.Response(200, json=out)


def serve(fake: FakeLlama):
    log = io.StringIO()
    upstream = upstream_client("http://127.0.0.1:8081", transport=httpx.MockTransport(fake.handler))
    app = create_app(upstream, UP, model_name="clef-flash", revision=REV, log=log)
    tc = TestClient(app)

    def handler(req: httpx.Request) -> httpx.Response:
        r = tc.request(req.method, req.url.path, content=req.content, headers={"content-type": "application/json"})
        return httpx.Response(r.status_code, content=r.content, headers={"content-type": r.headers.get("content-type", "")})

    return httpx.MockTransport(handler), tc, log


def c1(fake) -> HttpC1Router:
    return HttpC1Router("http://judge", "clef-flash", transport=serve(fake)[0])


def c2(fake) -> HttpC2Gate:
    return HttpC2Gate("http://judge", "clef-flash", transport=serve(fake)[0])


def payload(text: str = "ok") -> SendPayload:
    return SendPayload(contract="B-csv-codegen@1", destination="codex",
                       messages=(Message(role="user", content=text),), max_output_chars=3000)


def post(tc, body):
    return tc.post("/v1/systemone", json=body)


def body(questions=None, **over):
    b = {"model": "clef-flash", "context": "架空の依頼",
         "questions": questions if questions is not None else [{"id": "q", "type": "noul", "prompt": "?"}]}
    b.update(over)
    return b


# ---- 写し方 ----

def test_c1_route_through_adapter():
    fake = FakeLlama()
    d = c1(fake).route("u1", "4月1日から6月30日までの日数を数えて", None)
    assert d.route == "local" and d.probs["local"] == 0.82 and set(d.probs) == set(ROUTES)
    assert d.revision == f"{REV}+{SHIM_VERSION}" and d.truncated is False


def test_c1_descriptions_reach_clef_as_criteria():
    fake = FakeLlama()
    c1(fake).route("u1", "依頼", None)
    sent = fake.posts[0]
    want = C1_QUESTIONS[0]
    assert set(sent) == {"state", "questions"}
    assert sent["questions"] == {"route": {"type": "choice", "instructions": want["prompt"],
                                           "criteria": {c: want["descriptions"][c] for c in want["choices"]}}}
    assert json.loads(sent["state"])["request"] == "依頼"


def test_choice_without_descriptions_sends_none():
    fake = FakeLlama(route={"a": 0.3, "b": 0.7})
    _, tc, _ = serve(fake)
    r = post(tc, body([{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b"]}]))
    assert r.status_code == 200 and fake.posts[0]["questions"]["q"]["criteria"] == {"a": None, "b": None}
    assert r.json()["answers"] == [{"id": "q", "answer": "b", "probs": {"a": 0.3, "b": 0.7}}]


def test_answer_is_recomputed_and_ties_follow_choice_order():
    route = {"human": 0.4, "local": 0.4, "code": 0.05, "A": 0.05, "B": 0.05, "reject": 0.05}
    d = c1(FakeLlama(route=route)).route("u", "r", None)  # 偽物自身は dict 順で human を選ぶ
    assert d.route == "local"


@pytest.mark.parametrize("p,decision", [(0.7894, "block"), (0.5, "block"), (0.205, "hold"), (0.0302, "allow")])
def test_c2_noul_is_p_true(p, decision):
    v = c2(FakeLlama(noul=p)).check(payload())
    assert v.decision == decision and v.prob_block == p and v.revision == f"{REV}+{SHIM_VERSION}"


def test_noul_probs_sum_to_one():
    _, tc, _ = serve(FakeLlama(noul=0.7894))
    a = post(tc, body()).json()["answers"][0]
    assert a["answer"] == "yes" and a["probs"] == {"yes": 0.7894, "no": 0.2106}


def test_c2_context_is_role_tagged():
    fake = FakeLlama()
    c2(fake).check(payload("本文"))
    sent = fake.posts[0]
    assert sent["state"] == "[user]\n本文" and sent["questions"]["forbidden"]["instructions"] == C2_QUESTIONS[0]["prompt"]


# ---- 1 要求 1 問（Clef は質問をまとめて判断するので、測っていない組み合わせでは答えない）----

def test_more_than_one_question_is_rejected_before_inference():
    fake = FakeLlama()
    _, tc, _ = serve(fake)
    qs = [{"id": "a", "type": "noul", "prompt": "?"}, {"id": "b", "type": "noul", "prompt": "!"}]
    assert post(tc, body(qs)).status_code == 422 and fake.posts == []


# ---- 切り捨て → human / hold ----

def test_truncation_flag_holds_c2():
    v = guarded_c2(c2(FakeLlama(noul=0.01, usage={**OK_USAGE, "truncated": True})), payload(), 2.0, 1000)
    assert v.decision == "hold" and v.truncated is True


def test_truncation_flag_sends_c1_to_human():
    d = guarded_c1(c1(FakeLlama(usage={**OK_USAGE, "truncated": True})), "u", "r", None, 2.0, 1000)
    assert d.route == "human" and d.truncated is True


def test_too_long_input_error_holds_and_hides_text():
    # 実測: プロンプトが --ubatch-size を超えると llama-server は 500 と本文付きのエラーを返す（切り捨てない）
    fake = FakeLlama(status=500)
    transport, tc, log = serve(fake)
    r = post(tc, body())
    assert r.status_code == 500 and "E12345" not in r.text and "E12345" not in log.getvalue()
    assert json.loads(log.getvalue().strip())["status"] == 500
    assert guarded_c2(HttpC2Gate("http://judge", "clef-flash", transport=transport), payload(), 2.0, 1000).decision == "hold"


# ---- 受けない要求（422）→ アダプタは例外 → guard が止める ----

def test_model_mismatch_is_rejected_and_guard_stops():
    fake = FakeLlama()
    transport, tc, _ = serve(fake)
    assert post(tc, body(model="other")).status_code == 422
    d = guarded_c1(HttpC1Router("http://judge", "other", transport=transport), "u", "r", None, 2.0, 1000)
    assert d.route == "human" and fake.posts == []


@pytest.mark.parametrize("questions", [
    [{"id": "q", "type": "score", "prompt": "?", "min": 1, "max": 3}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b"], "descriptions": {"c": "x"}}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "b"], "descriptions": {"a": ""}}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", "a"]}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a"]}],
    [{"id": "q", "type": "choice", "prompt": "?", "choices": ["a", 1]}],
    [{"id": "q", "type": "noul", "prompt": "?", "choices": ["a", "b"]}],
    [{"id": "q", "type": "noul", "prompt": "?", "min": 0}],
    [{"id": "q", "type": "noul", "prompt": "?", "negate": True}],
    [],
])
def test_bad_questions_are_rejected(questions):
    fake = FakeLlama()
    _, tc, _ = serve(fake)
    assert post(tc, body(questions)).status_code == 422 and fake.posts == []


@pytest.mark.parametrize("over", [{"context": ""}, {"context": "x" * 50_001}, {"extra": 1}])
def test_bad_requests_are_rejected(over):
    fake = FakeLlama()
    _, tc, _ = serve(fake)
    assert post(tc, body(**over)).status_code == 422 and fake.posts == []


# ---- llama-server の失敗・想定外の出力 → 500 / 504（本文を漏らさない）→ guard が止める ----

@pytest.mark.parametrize("exc,status", [
    (httpx.ReadTimeout("社員番号 E12345"), 504),
    (httpx.ConnectError("社員番号 E12345"), 500),
])
def test_transport_errors_hide_text_and_hold(exc, status):
    transport, tc, log = serve(FakeLlama(raises=exc))
    r = post(tc, body())
    assert r.status_code == status and "E12345" not in r.text and "E12345" not in log.getvalue()
    assert guarded_c2(HttpC2Gate("http://judge", "clef-flash", transport=transport), payload(), 2.0, 1000).decision == "hold"


@pytest.mark.parametrize("status", [400, 307, 501])
def test_non_200_upstream_is_500(status):
    fake = FakeLlama(status=status)
    _, tc, _ = serve(fake)
    assert post(tc, body()).status_code == 500 and len(fake.posts) == 1  # 307 の転送先へは送らない


@pytest.mark.parametrize("over", [
    {"answers": {}},
    {"answers": {"q": {"type": "noul", "noul": 1.2}}},
    {"answers": {"q": {"type": "noul", "noul": True}}},
    {"answers": {"q": {"type": "noul", "noul": "0.3"}}},
    {"answers": {"q": {"type": "choice", "probabilities": {"yes": 0.5, "no": 0.5}}}},
    {"answers": []},
    {"model": "other-model"},
    {"usage": "missing"},
    {"usage": {"output_tokens": 0}},
    {"usage": {"input_tokens": True, "output_tokens": 0}},
])
def test_unexpected_model_output_is_500(over):
    _, tc, _ = serve(FakeLlama(**over))
    assert post(tc, body()).status_code == 500


def test_upstream_restarted_with_other_setup_is_refused():
    fake = FakeLlama()
    _, tc, log = serve(fake)
    assert post(tc, body()).status_code == 200
    fake.props = {**fake.props, "build_info": "b99999-0000000"}
    assert post(tc, body()).status_code == 500 and len(fake.posts) == 1
    assert json.loads(log.getvalue().strip().splitlines()[-1]) == {"event": "upstream_changed", "fields": ["build_info"]}


def test_log_has_counts_not_text():
    _, tc, log = serve(FakeLlama())
    post(tc, body(context="社員番号 E12345"))
    line = json.loads(log.getvalue().strip())
    assert line["event"] == "answered" and line["input_tokens"] == 175 and line["truncated"] is False
    assert len(line["ctx"]) == 12 and "E12345" not in log.getvalue()


def test_healthz_reports_revision():
    _, tc, _ = serve(FakeLlama())
    assert tc.get("/healthz").json() == {"model": "clef-flash", "revision": f"{REV}+{SHIM_VERSION}"}


# ---- 起動時の版の確認 ----

@pytest.fixture
def weights(tmp_path):
    f = tmp_path / "Clef-Flash-Q4_K_M.gguf"
    f.write_bytes(b"fake weights")
    return f, hashlib.sha256(b"fake weights").hexdigest()


def props_client(props, status=200):
    return upstream_client("http://127.0.0.1:8081",
                           transport=httpx.MockTransport(lambda req: httpx.Response(status, json=props)))


def test_verify_accepts_matching_upstream_via_symlink(weights, tmp_path):
    f, sha = weights
    link = tmp_path / "link.gguf"
    os.symlink(f, link)
    up = Upstream(model_path=str(link), build=UP.build, n_ctx=8192, alias="clef-flash")
    verify_upstream(props_client(props_for(Upstream(str(f), UP.build, 8192, "clef-flash"))), up, model_sha256=sha.upper())


@pytest.mark.parametrize("over", [
    {"model_path": "/other/model.gguf"},
    {"build_info": "b11670-aaaaaaa"},
    {"default_generation_settings": {"n_ctx": 4096}},
    {"default_generation_settings": None},
    {"total_slots": 2},
    {"model_alias": "other"},
])
def test_verify_refuses_other_setup(weights, over):
    f, sha = weights
    up = Upstream(model_path=str(f), build=UP.build, n_ctx=8192, alias="clef-flash")
    with pytest.raises(SystemExit):
        verify_upstream(props_client({**props_for(up), **over}), up, model_sha256=sha)


def test_verify_refuses_other_weights(weights):
    f, _ = weights
    up = Upstream(model_path=str(f), build=UP.build, n_ctx=8192, alias="clef-flash")
    with pytest.raises(SystemExit):
        verify_upstream(props_client(props_for(up)), up, model_sha256="0" * 64)


def test_verify_refuses_unreadable_props(weights):
    f, sha = weights
    up = Upstream(model_path=str(f), build=UP.build, n_ctx=8192, alias="clef-flash")
    with pytest.raises(SystemExit):
        verify_upstream(props_client({}, status=503), up, model_sha256=sha)


def test_runtime_revision_names_every_pinned_part():
    sha = "3626116623ca8d89641952471fac9ce12af84fb3517520029a2ac6f5146868a2"
    assert runtime_revision(REV, model_sha256=sha, up=UP) == f"{REV};sha256=3626116623ca8d89;llama.cpp=b11669-620d2225c;n_ctx=8192"


@pytest.mark.parametrize("url", [
    "http://localhost:8081", "https://127.0.0.1:8081", "http://10.0.0.5:8081", "http://user:pw@127.0.0.1:8081",
    "http://127.0.0.1:8081/v1", "http://127.0.0.1:8081/?a=1",
])
def test_upstream_must_be_loopback_ip(url):
    with pytest.raises(SystemExit):
        upstream_client(url)


def test_upstream_client_ignores_proxy_env_and_redirects():
    for url in ("http://127.0.0.1:8081", "http://[::1]:8081"):
        c = upstream_client(url)
        assert c.trust_env is False and c.follow_redirects is False


def test_main_requires_pinned_env(monkeypatch):
    for k in ("JUDGE_MODEL_PATH", "JUDGE_MODEL_SHA256", "JUDGE_MODEL_REVISION", "JUDGE_LLAMA_BUILD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        main([])
