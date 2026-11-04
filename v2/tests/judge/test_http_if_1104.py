"""判断モデルの HTTP I/F に 2026-11-04 に足した決まり（docs/redesign-v2.md「判断モデルの HTTP I/F」検証規則 6〜8）。

- 6: 要求ごとの番号（X-Request-Id）。返すなら同じ値。判定の結果と監査 DB に残す
- 7: expected_revision を渡したら、その版の応答だけを受ける
- 8: 応答は MAX_RESPONSE_BYTES まで
どれも外れたら不正な応答として guard が C1=human / C2=hold に倒す。
"""
from __future__ import annotations

import io
import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from common.schemas import Message, SendPayload
from judge.base import C1Decision, C2Verdict, guarded_c1, guarded_c2
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


def payload() -> SendPayload:
    return SendPayload(contract="B-csv-codegen@1", destination="codex",
                       messages=(Message(role="user", content="ok"),), max_output_chars=3000)


def c2_body(p: float = 0.05, revision: str = "r1", model: str = "m") -> dict:
    return {"model": model, "revision": revision,
            "answers": [{"id": "forbidden", "answer": "yes" if p >= .5 else "no", "probs": {"yes": p, "no": 1 - p}}]}


def c1_body(top: str = "local", p: float = 0.9, revision: str = "r1") -> dict:
    rest = (1 - p) / (len(ROUTES) - 1)
    return {"model": "m", "revision": revision,
            "answers": [{"id": "route", "answer": top, "probs": {r: (p if r == top else rest) for r in ROUTES}}]}


class Recorder:
    """送られた要求の番号を覚え、応答の作り方を差し替えられる偽の判断サーバ。"""

    def __init__(self, body=None, *, echo: str | None = "same", status=200, extra_headers=None, raw: bytes | None = None,
                 stream_chunks: list[bytes] | None = None):
        self.body = body if body is not None else c2_body()
        self.echo, self.status, self.extra, self.raw, self.chunks = echo, status, extra_headers or {}, raw, stream_chunks
        self.seen: list[str | None] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        rid = req.headers.get(REQUEST_ID_HEADER)
        self.seen.append(rid)
        headers = dict(self.extra)
        if self.echo == "same" and rid is not None:
            headers[REQUEST_ID_HEADER] = rid
        elif self.echo not in (None, "same"):
            headers[REQUEST_ID_HEADER] = self.echo
        if self.chunks is not None:
            return httpx.Response(self.status, headers=headers, content=iter(self.chunks))
        if self.raw is not None:
            return httpx.Response(self.status, headers=headers, content=self.raw)
        return httpx.Response(self.status, headers=headers, json=self.body)


def c2_gate(rec: Recorder, **kw) -> HttpC2Gate:
    return HttpC2Gate("http://j", "m", transport=httpx.MockTransport(rec), **kw)


# ---- 6. 要求の番号 ----

def test_each_call_sends_a_fresh_request_id_and_records_it():
    rec = Recorder()
    gate = c2_gate(rec)
    v1, v2 = gate.check(payload()), gate.check(payload())
    assert all(REQUEST_ID_RE.match(r or "") for r in rec.seen)
    assert rec.seen[0] != rec.seen[1]
    assert (v1.judge_request_id, v2.judge_request_id) == tuple(rec.seen)


def test_c1_records_request_id():
    rec = Recorder(c1_body())
    d = HttpC1Router("http://j", "m", transport=httpx.MockTransport(rec)).route("u", "r", None)
    assert d.judge_request_id == rec.seen[0] and d.route == "local"


def test_server_that_does_not_echo_is_accepted():
    rec = Recorder(echo=None)
    assert c2_gate(rec).check(payload()).judge_request_id == rec.seen[0]


def test_echoed_other_request_id_is_malformed_and_guard_holds():
    rec = Recorder(echo="0" * 32)
    with pytest.raises(MalformedResponse, match="request id"):
        c2_gate(rec).check(payload())
    v = guarded_c2(c2_gate(Recorder(echo="f" * 32)), payload(), 5.0, 10_000)
    assert v.decision == "hold" and "request id mismatch" in v.reason


def test_cascade_keeps_both_request_ids():
    first = c2_gate(Recorder(c2_body(0.3)))  # 中間帯 → 後段へ
    second_rec = Recorder(c2_body(0.01))
    v = CascadeC2Gate(first, c2_gate(second_rec)).check(payload())
    assert v.decision == "allow" and v.judge_request_id == second_rec.seen[0]
    assert "; first req=" in v.reason and v.reason.split("; first req=")[1] != v.judge_request_id

    r1 = HttpC1Router("http://j", "m", transport=httpx.MockTransport(Recorder(c1_body("local", 0.4))))
    rec2 = Recorder(c1_body("code", 0.9))
    d = CascadeC1Router(r1, HttpC1Router("http://j", "m", transport=httpx.MockTransport(rec2)),
                        escalate_below=0.6).route("u", "r", None)
    assert d.judge_request_id == rec2.seen[0] and "; first req=" in d.reason


# ---- 7. 版の固定 ----

def test_expected_revision_accepts_the_same():
    assert c2_gate(Recorder(c2_body(revision="sha-1")), expected_revision="sha-1").check(payload()).revision == "sha-1"


def test_expected_revision_mismatch_is_malformed_and_guard_holds():
    with pytest.raises(MalformedResponse, match="revision mismatch"):
        c2_gate(Recorder(c2_body(revision="sha-2")), expected_revision="sha-1").check(payload())
    v = guarded_c2(c2_gate(Recorder(c2_body(revision="sha-2")), expected_revision="sha-1"), payload(), 5.0, 10_000)
    assert v.decision == "hold"
    d = guarded_c1(HttpC1Router("http://j", "m", expected_revision="sha-1",
                                transport=httpx.MockTransport(Recorder(c1_body(revision="sha-2")))),
                   "u", "r", None, 5.0, 10_000)
    assert d.route == "human" and "revision mismatch" in d.reason


def test_expected_revision_must_not_be_empty():
    with pytest.raises(ValueError):
        c2_gate(Recorder(), expected_revision="")


def test_without_expected_revision_any_nonempty_revision_is_recorded():
    assert c2_gate(Recorder(c2_body(revision="whatever"))).check(payload()).revision == "whatever"


# ---- 8. 応答の大きさ ----

def test_declared_too_large_response_is_malformed():
    rec = Recorder(raw=b"{}", extra_headers={"content-length": str(MAX_RESPONSE_BYTES + 1)})
    with pytest.raises(MalformedResponse, match="too large"):
        c2_gate(rec).check(payload())


def test_streamed_too_large_response_is_cut_and_malformed():
    rec = Recorder(stream_chunks=[b"x" * 1024] * 64)
    with pytest.raises(MalformedResponse, match="too large"):
        c2_gate(rec, max_response_bytes=10_000).check(payload())


def test_response_at_the_limit_is_read():
    raw = json.dumps(c2_body()).encode()
    v = c2_gate(Recorder(stream_chunks=[raw]), max_response_bytes=len(raw)).check(payload())
    assert v.decision == "allow"


def test_non_2xx_still_holds_without_retry():
    rec = Recorder(status=503, extra_headers={"retry-after": "1"})
    v = guarded_c2(c2_gate(rec), payload(), 5.0, 10_000)
    assert v.decision == "hold" and len(rec.seen) == 1  # Retry-After があっても送り直さない


# ---- 起動時の確かめ（/healthz） ----

def _health(model="m", revision="sha-1", status=200):
    def h(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET" and req.url.path == "/healthz"
        return httpx.Response(status, json={"model": model, "revision": revision})
    return httpx.MockTransport(h)


def test_ready_reports_model_and_revision():
    assert HttpC2Gate("http://j", "m", transport=_health()).ready() == {"model": "m", "revision": "sha-1"}
    assert HttpC1Router("http://j", "m", expected_revision="sha-1", transport=_health()).ready()["revision"] == "sha-1"


@pytest.mark.parametrize("kw, transport", [
    ({"expected_revision": "sha-1"}, _health(revision="sha-2")),
    ({}, _health(model="other")),
    ({}, _health(revision="")),
])
def test_ready_rejects_wrong_identity(kw, transport):
    with pytest.raises(MalformedResponse):
        HttpC2Gate("http://j", "m", transport=transport, **kw).ready()


def test_ready_http_error_raises():
    with pytest.raises(httpx.HTTPStatusError):
        HttpC2Gate("http://j", "m", transport=_health(status=503)).ready()


# ---- 変換サーバが番号を返す ----

def _forwarding(app):
    tc = TestClient(app)

    def handler(req: httpx.Request) -> httpx.Response:
        hdr = {"content-type": "application/json"}
        if REQUEST_ID_HEADER in req.headers:
            hdr[REQUEST_ID_HEADER] = req.headers[REQUEST_ID_HEADER]
        r = tc.request(req.method, req.url.path, content=req.content, headers=hdr)
        return httpx.Response(r.status_code, content=r.content, headers=dict(r.headers))

    return tc, handler


def test_laya_shim_echoes_and_logs_request_id():
    from tests.judge.test_laya_server import REV, FakeLaya
    from judge.laya_server import create_app

    log = io.StringIO()
    tc, handler = _forwarding(create_app(FakeLaya(), model_name="laya-multilingual", revision=REV, log=log))
    v = HttpC2Gate("http://j", "laya-multilingual", transport=httpx.MockTransport(handler)).check(payload())
    assert v.judge_request_id and json.loads(log.getvalue().splitlines()[-1])["req"] == v.judge_request_id
    # 形の違う値は返さない・ログに残さない
    r = tc.post("/v1/systemone", headers={"X-Request-Id": "bad id\nINJECT"},
                json={"model": "laya-multilingual", "context": "x", "questions": [{"id": "q", "type": "noul", "prompt": "?"}]})
    assert r.status_code == 200 and "x-request-id" not in r.headers
    assert json.loads(log.getvalue().splitlines()[-1])["req"] is None


def test_clef_shim_echoes_and_logs_request_id():
    from tests.judge.test_clef_server import REV, UP, FakeLlama
    from judge.clef_server import create_app

    log = io.StringIO()
    upstream = httpx.Client(base_url="http://llama", transport=httpx.MockTransport(FakeLlama().handler))
    tc, handler = _forwarding(create_app(upstream, UP, model_name="clef-flash", revision=REV, log=log))
    v = HttpC2Gate("http://j", "clef-flash", transport=httpx.MockTransport(handler)).check(payload())
    assert v.judge_request_id and json.loads(log.getvalue().splitlines()[-1])["req"] == v.judge_request_id


# ---- 監査 DB に残る ----

class _C1:
    def route(self, user, request, input):  # noqa: A002
        return C1Decision(route="B", destination="codex", reason="stub", model="c1m", revision="r1",
                          question_version="c1q@1", probs={"B": 0.9}, judge_request_id="a" * 32)


class _C2:
    def check(self, payload):
        return C2Verdict(decision="allow", reason="stub", model="c2m", revision="r2", truncated=False,
                         prob_block=0.01, question_version="c2q@1", judge_request_id="b" * 32)


def test_audit_store_records_judge_request_ids(tmp_path):
    from integration.flow_b import run_contract_b
    from orchestrator.audit_store import AuditStore
    from tests.orchestrator.test_pipeline import B_TEXT, NOW, FakeConfirmer, FakeGateway

    store = AuditStore(tmp_path / "audit.db")
    try:
        run_contract_b("u_manager", "spec-csv-001@1", B_TEXT, {"言語": "Python 3.11"}, "codex",
                       c2=_C2(), confirmer=FakeConfirmer(), gateway=FakeGateway(), now=NOW, audit_sink=store, c1=_C1())
        (row,) = store.runs()
    finally:
        store.close()
    assert (row["c1_route"], row["c1_judge_request_id"], row["c2_judge_request_id"]) == ("B", "a" * 32, "b" * 32)


def test_old_audit_db_gets_the_new_columns(tmp_path):
    from orchestrator.audit_store import SCHEMA, AuditStore

    old = "\n".join(line for line in SCHEMA.splitlines() if "judge_request_id" not in line)
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(old)
    con.close()
    AuditStore(path).close()
    con = sqlite3.connect(path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    con.close()
    assert {"c1_judge_request_id", "c2_judge_request_id"} <= cols
