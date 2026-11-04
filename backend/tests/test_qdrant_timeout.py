"""Qdrant の時間切れの設定と、検索だけの送り直し（Issue #14）の試験。

確かめること:
- 既定（QDRANT_TIMEOUT_S=0・QDRANT_SEARCH_RETRIES=0）は今までと同じ: timeout を渡さず、送り直さない
- 時間切れ（httpx の TimeoutException と、それを包んだ ResponseHandlingException）のときだけ送り直す
- 接続拒否・応答の誤りなど時間切れ以外は送り直さない
- 本物の qdrant-client 1.12.1 と httpx で、遅い相手に対して timeout が効き、送り直しで答えが取れる
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings, settings
from app.services import qdrant as q


@pytest.fixture(autouse=True)
def _fresh_client():
    q._client = None
    yield
    q._client = None


def test_defaults_keep_previous_behavior():
    s = Settings()
    assert s.qdrant_timeout_s == 0 and s.qdrant_search_retries == 0


@pytest.mark.parametrize("env, value", [("QDRANT_TIMEOUT_S", "-1"), ("QDRANT_SEARCH_RETRIES", "3"),
                                        ("QDRANT_SEARCH_RETRIES", "-1")])
def test_bad_settings_stop_at_startup(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValidationError):
        Settings()


def test_env_sets_timeout_and_retries(monkeypatch):
    monkeypatch.setenv("QDRANT_TIMEOUT_S", "10")
    monkeypatch.setenv("QDRANT_SEARCH_RETRIES", "1")
    s = Settings()
    assert (s.qdrant_timeout_s, s.qdrant_search_retries) == (10, 1)


def test_get_client_passes_timeout_only_when_set(monkeypatch):
    seen = []
    monkeypatch.setattr(q, "AsyncQdrantClient", lambda **kw: seen.append(kw) or object())
    monkeypatch.setattr(settings, "qdrant_timeout_s", 0)
    q.get_client()
    q._client = None
    monkeypatch.setattr(settings, "qdrant_timeout_s", 10)
    q.get_client()
    assert seen == [{"url": settings.qdrant_url}, {"url": settings.qdrant_url, "timeout": 10}]


@pytest.mark.parametrize("exc, expected", [
    (httpx.ReadTimeout("t"), True),
    (httpx.ConnectTimeout("t"), True),
    (httpx.PoolTimeout("t"), True),
    (ResponseHandlingException(httpx.ReadTimeout("t")), True),
    (ResponseHandlingException(httpx.ConnectError("refused")), False),
    (ResponseHandlingException(ValueError("bad json")), False),
    (httpx.ConnectError("refused"), False),
    (TimeoutError(), False),  # 外側の打ち切り（rag_backend の 30 秒）は送り直さない
    (UnexpectedResponse(500, "err", b"", httpx.Headers()), False),
])
def test_is_timeout(exc, expected):
    assert q.is_timeout(exc) is expected


class _Flaky:
    def __init__(self, fails: list[BaseException]):
        self.fails = list(fails)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.fails:
            raise self.fails.pop(0)
        return "ok"


def _timeout():
    return ResponseHandlingException(httpx.ReadTimeout("slow"))


def test_no_retry_by_default(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_search_retries", 0)
    f = _Flaky([_timeout()])
    with pytest.raises(ResponseHandlingException):
        asyncio.run(q.read_with_retry(f))
    assert f.calls == 1


def test_one_retry_recovers_from_one_timeout(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_search_retries", 1)
    before = q.retry_count
    f = _Flaky([_timeout()])
    assert asyncio.run(q.read_with_retry(f)) == "ok"
    assert f.calls == 2 and q.retry_count == before + 1


def test_gives_up_after_the_set_number(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_search_retries", 1)
    f = _Flaky([_timeout(), _timeout()])
    with pytest.raises(ResponseHandlingException):
        asyncio.run(q.read_with_retry(f))
    assert f.calls == 2


@pytest.mark.parametrize("exc", [ResponseHandlingException(httpx.ConnectError("refused")),
                                 UnexpectedResponse(503, "busy", b"", httpx.Headers()), ValueError("x")])
def test_other_errors_are_not_retried(monkeypatch, exc):
    monkeypatch.setattr(settings, "qdrant_search_retries", 2)
    f = _Flaky([exc])
    with pytest.raises(type(exc)):
        asyncio.run(q.read_with_retry(f))
    assert f.calls == 1


# ─── retriever が送り直しを使う ───

class _Point:
    def __init__(self, i):
        self.payload = {"content": f"c{i}", "source_file": "f.md", "document_id": f"d{i}", "access_level": 1}
        self.score = 1.0


class _Resp:
    points = [_Point(0)]


class _FlakyClient:
    def __init__(self, fails: int):
        self.fails, self.calls, self.kwargs = fails, 0, []

    async def query_points(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        if self.fails:
            self.fails -= 1
            raise _timeout()
        return _Resp()


class _Db:
    async def fetchall(self, sql, params=()):
        return [{"id": d, "access_level": 1, "status": "done", "unclassified": 0} for d in params]


def _stub_retriever(monkeypatch, client):
    from app.services import retriever

    async def embed(question):
        return [0.0] * 4

    monkeypatch.setattr(retriever, "embed_query", embed)
    monkeypatch.setattr(retriever, "build_sparse", lambda qq: ([], []))
    monkeypatch.setattr(retriever, "get_client", lambda: client)
    monkeypatch.setattr(retriever, "db", _Db())
    monkeypatch.setattr(settings, "rerank_enabled", False)
    return retriever


def test_retrieve_retries_the_same_query_with_the_same_filter(monkeypatch):
    client = _FlakyClient(fails=1)
    retriever = _stub_retriever(monkeypatch, client)
    monkeypatch.setattr(settings, "qdrant_search_retries", 1)
    got = asyncio.run(retriever.retrieve("q", user_access_level=1))
    assert [c["content"] for c in got] == ["c0"]
    assert client.calls == 2
    # 送り直しても権限の絞り込みは同じ要求のまま
    f0 = client.kwargs[0]["prefetch"][0].filter
    f1 = client.kwargs[1]["prefetch"][0].filter
    assert f0 == f1 and f0.must[0].key == "access_level"


def test_retrieve_default_still_fails_on_timeout(monkeypatch):
    client = _FlakyClient(fails=1)
    retriever = _stub_retriever(monkeypatch, client)
    monkeypatch.setattr(settings, "qdrant_search_retries", 0)
    with pytest.raises(ResponseHandlingException):
        asyncio.run(retriever.retrieve("q", user_access_level=1))
    assert client.calls == 1


# ─── 本物の qdrant-client と httpx で ───

QUERY_OK = {"result": {"points": [{"id": 1, "version": 0, "score": 0.5,
                                   "payload": {"content": "c", "document_id": "d"}}]},
            "status": "ok", "time": 0.001}


class _SlowFirst(BaseHTTPRequestHandler):
    """最初の検索だけ遅く答える偽の Qdrant（REST）。"""

    n = 0
    delay = 2.5

    def do_POST(self):
        type(self).n += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if type(self).n == 1:
            time.sleep(self.delay)
        body = json.dumps(QUERY_OK).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 相手が時間切れで切った後

    def log_message(self, *a):
        pass


@pytest.fixture
def slow_qdrant():
    _SlowFirst.n = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowFirst)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _query(url, timeout_s, retries, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", url)
    monkeypatch.setattr(settings, "qdrant_timeout_s", timeout_s)
    monkeypatch.setattr(settings, "qdrant_search_retries", retries)

    async def run():
        try:
            return await q.read_with_retry(lambda: q.get_client().query_points(
                collection_name="c", query=[0.0, 0.0], limit=1, with_payload=True))
        finally:
            await q.close_client()

    return asyncio.run(run())


def test_real_client_times_out_and_is_detected(slow_qdrant, monkeypatch):
    with pytest.raises(ResponseHandlingException) as ei:
        _query(slow_qdrant, 1, 0, monkeypatch)
    assert q.is_timeout(ei.value)
    assert isinstance(ei.value.source, httpx.TimeoutException)


def test_real_client_retry_gets_the_answer(slow_qdrant, monkeypatch):
    t = time.time()
    res = _query(slow_qdrant, 1, 1, monkeypatch)
    assert [p.id for p in res.points] == [1]
    assert _SlowFirst.n == 2
    assert time.time() - t < 2.4  # 1 秒で切って送り直し、遅い 1 回目を待たない


def test_real_client_without_timeout_waits_past_one_second(slow_qdrant, monkeypatch):
    # 既定（timeout を渡さない）は httpx の 5 秒なので、2.5 秒の遅れは待って答えが取れる
    res = _query(slow_qdrant, 0, 0, monkeypatch)
    assert [p.id for p in res.points] == [1] and _SlowFirst.n == 1
