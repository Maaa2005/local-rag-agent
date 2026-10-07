"""レビュー指摘 1〜9 の回帰テスト。"""
from __future__ import annotations

import logging
import math
import sys
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common.schemas import (
    CommitRequest,
    FailureKind,
    Message,
    PrepareRequest,
    SendPayload,
    SendState,
)
from gateway import app as app_mod
from gateway.adapters import ClaudeAdapter, CodexAdapter, FakeAdapter, NotSentError
from gateway.app import create_app
from gateway.contracts import CONTRACTS
from gateway.service import Expired, GatewayService, InvalidRequest, UnknownContract
from gateway.store import SendStore, StoreError
from tests.gateway.test_gateway import _fake_sdk, key_file, make_payload, make_req  # noqa: F401

U = Message(role="user", content="u")
S = Message(role="system", content="s")


def _payload(messages, contract="A-faq-format@1", dest="claude", max_out=100):
    return SendPayload(contract=contract, destination=dest, messages=tuple(messages), max_output_chars=max_out)


@pytest.fixture
def svc(tmp_path):
    f = FakeAdapter()
    return GatewayService(SendStore(tmp_path / "db"), {"claude": f, "codex": f})


# ---- 1. expires_at ----
@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_expires_at_rejects_inf_nan(bad):
    with pytest.raises(ValidationError):
        PrepareRequest(request_id="req-00000001", payload=make_payload(), expires_at=bad)


def test_expires_at_ttl_upper_bound(tmp_path):
    now = 1000.0
    svc = GatewayService(SendStore(tmp_path / "db"), {"claude": FakeAdapter()}, clock=lambda: now)
    ttl = CONTRACTS["A-faq-format@1"].max_ttl_seconds
    svc.prepare(PrepareRequest(request_id="req-ttl-ok01", payload=make_payload(), expires_at=now + ttl))
    with pytest.raises(InvalidRequest):
        svc.prepare(PrepareRequest(request_id="req-ttl-ng01", payload=make_payload(), expires_at=now + ttl + 1))
    with pytest.raises(Expired):
        svc.prepare(PrepareRequest(request_id="req-ttl-ng02", payload=make_payload(), expires_at=now))


# ---- 2. finish 失敗 ----
class FlakyFinishStore(SendStore):
    def __init__(self, path, fail_times):
        super().__init__(path)
        self.fail_times = fail_times
        self.finish_calls = 0

    def finish(self, *a, **kw):
        self.finish_calls += 1
        if self.finish_calls <= self.fail_times:
            raise StoreError("finish failed")
        return super().finish(*a, **kw)


def test_finish_retried_then_succeeds(tmp_path):
    fake = FakeAdapter(reply="ok")
    store = FlakyFinishStore(tmp_path / "db", fail_times=2)
    svc = GatewayService(store, {"claude": fake}, sleep=lambda s: None)
    req = make_req()
    d = svc.prepare(req).digest
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st.state == SendState.SUCCEEDED and st.output_text == "ok"
    assert store.finish_calls == 3 and len(fake.received) == 1


def test_finish_gives_up_returns_unknown_without_resend(tmp_path, caplog):
    fake = FakeAdapter(reply="secret-output-text")
    store = FlakyFinishStore(tmp_path / "db", fail_times=99)
    svc = GatewayService(store, {"claude": fake}, sleep=lambda s: None)
    req = make_req(text="secret-body-text")
    d = svc.prepare(req).digest
    with caplog.at_level(logging.WARNING, logger="gateway"):
        st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st.state == SendState.FAILED and st.failure == FailureKind.unknown and st.output_text is None
    assert len(fake.received) == 1
    assert store.finish_calls == 3
    assert "secret-body-text" not in caplog.text and "secret-output-text" not in caplog.text
    assert req.request_id in caplog.text
    # 再 commit しても送らない（ATTEMPTING のまま）
    st2 = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st2.state == SendState.ATTEMPTING and len(fake.received) == 1


def test_finish_rowcount_mismatch_logged(tmp_path, caplog):
    store = SendStore(tmp_path / "db")
    fake = FakeAdapter()
    svc = GatewayService(store, {"claude": fake})
    req = make_req()
    d = svc.prepare(req).digest
    # 送信中に別経路で終端化された状況を作る
    fake.on_send = lambda p: store.fail_attempting_as_unknown()
    with caplog.at_level(logging.ERROR, logger="gateway"):
        st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert "finish rowcount=0" in caplog.text
    assert st.state == SendState.FAILED and st.failure == FailureKind.unknown


def test_store_finish_returns_rowcount(tmp_path):
    store = SendStore(tmp_path / "db")
    store.insert_prepared("req-00000001", "d", "{}", time.time() + 60)
    assert store.finish("req-00000001", SendState.FAILED, FailureKind.unknown) == 0
    assert store.claim_attempt("req-00000001")
    assert store.finish("req-00000001", SendState.SUCCEEDED, None, "x") == 1


# ---- 3. 契約レジストリ ----
@pytest.mark.parametrize("contract", ["A-faq-format@2", "A-other@1", "B-spec-code@1", "a-faq-format@1",
                                      "A-faq-format@1 "])
def test_contract_exact_match(svc, contract):
    with pytest.raises(UnknownContract):
        svc.prepare(make_req(contract=contract))


def test_contract_destination_not_allowed(svc, monkeypatch):
    from dataclasses import replace

    monkeypatch.setitem(CONTRACTS, "A-faq-format@1",
                        replace(CONTRACTS["A-faq-format@1"], destinations=frozenset({"claude"})))
    svc.prepare(make_req(rid="req-dest-ok1"))
    with pytest.raises(InvalidRequest):
        svc.prepare(make_req(rid="req-dest-ng1", dest="codex"))


def test_registry_values():
    a, b = CONTRACTS["A-faq-format@1"], CONTRACTS["B-csv-codegen@1"]
    assert (a.payload_chars, a.output_chars, b.payload_chars, b.output_chars) == (2000, 1500, 1500, 3000)
    assert a.destinations == b.destinations == {"claude", "codex"}


# ---- 4. messages の構造 ----
@pytest.mark.parametrize("msgs", [
    [U, S],            # system が先頭以外
    [S, S, U],         # system が 2 件
    [S, U, S, U],
    [S],               # user なし
])
def test_message_structure_rejected(svc, msgs):
    req = PrepareRequest(request_id="req-struct-1", payload=_payload(msgs), expires_at=time.time() + 60)
    with pytest.raises(InvalidRequest):
        svc.prepare(req)


@pytest.mark.parametrize("msgs", [[U], [S, U], [S, U, U]])
def test_message_structure_ok(svc, msgs):
    svc.prepare(PrepareRequest(request_id="req-struct-2", payload=_payload(msgs), expires_at=time.time() + 60))


def test_assistant_role_rejected():
    with pytest.raises(ValidationError):
        Message(role="assistant", content="x")


def test_claude_adapter_keeps_order_no_concat(monkeypatch, key_file):  # noqa: F811
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk("anthropic", captured=cap))
    p = _payload([S, Message(role="user", content="1"), Message(role="user", content="2")])
    ClaudeAdapter(model="m", key_path=key_file).send(p)
    assert cap["system"] == "s"
    assert cap["messages"] == [{"role": "user", "content": "1"}, {"role": "user", "content": "2"}]


def test_claude_adapter_refuses_bad_structure(monkeypatch, key_file):  # noqa: F811
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk("anthropic", captured=cap))
    with pytest.raises(NotSentError):
        ClaudeAdapter(model="m", key_path=key_file).send(_payload([U, S]))
    assert cap == {}


# ---- 5. サロゲート ----
def test_surrogate_rejected_model():
    with pytest.raises(ValidationError):
        Message(role="user", content="a\ud800b")


def test_surrogate_rejected_http(tmp_path):
    svc = GatewayService(SendStore(tmp_path / "db"), {"claude": FakeAdapter()})
    body = '{"request_id":"req-sur-0001","expires_at":%f,"payload":{"contract":"A-faq-format@1",' \
           '"destination":"claude","max_output_chars":10,"messages":[{"role":"user","content":"a\\ud800b"}]}}' \
           % (time.time() + 60)
    with TestClient(create_app(svc)) as c:
        r = c.post("/prepare", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


# ---- 6. 件数・出力上限・トークン上限 ----
def test_messages_max_length():
    with pytest.raises(ValidationError):
        _payload([U] * 17)
    _payload([U] * 16)


def test_max_output_chars_contract_limit(svc):
    svc.prepare(make_req(rid="req-out-ok01", max_out=1500))
    with pytest.raises(InvalidRequest):
        svc.prepare(make_req(rid="req-out-ng01", max_out=1501))
    svc.prepare(make_req(rid="req-out-ok02", contract="B-csv-codegen@1", max_out=3000))
    with pytest.raises(InvalidRequest):
        svc.prepare(make_req(rid="req-out-ng02", contract="B-csv-codegen@1", max_out=3001))


@pytest.mark.parametrize("contract", ["A-faq-format@1", "B-csv-codegen@1"])
def test_adapter_token_limits(monkeypatch, key_file, contract):  # noqa: F811
    ca: dict = {}
    co: dict = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk("anthropic", captured=ca))
    monkeypatch.setitem(sys.modules, "openai", _fake_sdk("openai", captured=co))
    ClaudeAdapter(model="m", key_path=key_file).send(make_payload(contract=contract))
    CodexAdapter(model="m", key_path=key_file).send(make_payload(contract=contract, dest="codex"))
    expected = CONTRACTS[contract].max_output_tokens
    assert ca["max_tokens"] == expected
    assert co["max_completion_tokens"] == expected


# ---- 7. 422 応答に入力値を含めない ----
def test_validation_error_hides_input(tmp_path):
    svc = GatewayService(SendStore(tmp_path / "db"), {"claude": FakeAdapter()})
    with TestClient(create_app(svc)) as c:
        r = c.post("/prepare", json={"request_id": "req-00000001", "expires_at": time.time() + 60,
                                     "payload": {"contract": "A-faq-format@1", "destination": "claude",
                                                 "max_output_chars": 10,
                                                 "messages": [{"role": "assistant", "content": "TOP-SECRET-BODY"}]}})
    assert r.status_code == 422
    assert "TOP-SECRET-BODY" not in r.text
    for e in r.json()["detail"]:
        assert set(e) == {"loc", "type"}


# ---- 8. 単一ワーカー ----
def test_main_runs_single_worker(monkeypatch):
    import types

    called: dict = {}
    fake_uvicorn = types.SimpleNamespace(run=lambda app, **kw: called.update(kw))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(app_mod, "build_service_from_env",
                        lambda: GatewayService.__new__(GatewayService))
    app_mod.main()
    assert called["workers"] == 1


# ---- 9. request_id の形式 ----
@pytest.mark.parametrize("rid", ["req 0000001", "req/000001", "req.000001", "../../etc1", "リクエスト00001"])
def test_request_id_pattern(rid):
    with pytest.raises(ValidationError):
        PrepareRequest(request_id=rid, payload=make_payload(), expires_at=time.time() + 60)
    with pytest.raises(ValidationError):
        CommitRequest(request_id=rid, expected_digest="x")


def test_request_id_uuid_hex_ok():
    import uuid

    PrepareRequest(request_id=uuid.uuid4().hex, payload=make_payload(), expires_at=time.time() + 60)


def test_status_path_pattern(tmp_path):
    svc = GatewayService(SendStore(tmp_path / "db"), {"claude": FakeAdapter()})
    with TestClient(create_app(svc)) as c:
        assert c.get("/status/bad.id%20x").status_code == 422
        assert c.get("/status/" + "a" * 65).status_code == 422
        assert c.get("/status/req-none-001").status_code == 404
