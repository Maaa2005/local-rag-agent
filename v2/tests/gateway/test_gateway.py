from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from common.schemas import (
    CommitRequest,
    FailureKind,
    Message,
    PrepareRequest,
    SendPayload,
    SendState,
    payload_digest,
)
from gateway.adapters import ClaudeAdapter, CodexAdapter, FakeAdapter, NotSentError
from gateway.service import (
    Conflict,
    DigestMismatch,
    Expired,
    GatewayService,
    NotFound,
    PayloadTooLarge,
    UnknownContract,
)
from gateway.store import SendStore, StoreError


def make_payload(text: str = "hello", contract: str = "A-faq-format@1", dest: str = "claude",
                 max_out: int = 100) -> SendPayload:
    return SendPayload(
        contract=contract,
        destination=dest,
        messages=(Message(role="system", content="sys"), Message(role="user", content=text)),
        max_output_chars=max_out,
    )


def make_req(rid: str = "req-00000001", ttl: float = 60, **kw) -> PrepareRequest:
    return PrepareRequest(request_id=rid, payload=make_payload(**kw), expires_at=time.time() + ttl)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "gw.db"


@pytest.fixture
def fake():
    return FakeAdapter(reply="answer")


@pytest.fixture
def svc(db, fake):
    return GatewayService(SendStore(db), {"claude": fake, "codex": fake})


def test_happy_path(svc, fake):
    req = make_req()
    prep = svc.prepare(req)
    assert prep.state == SendState.PREPARED
    assert prep.digest == payload_digest(req.payload)
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=prep.digest))
    assert st.state == SendState.SUCCEEDED
    assert st.output_text == "answer"
    assert fake.received == [req.payload]
    assert svc.status(req.request_id).state == SendState.SUCCEEDED


def test_output_truncated(db):
    fake = FakeAdapter(reply="x" * 50)
    svc = GatewayService(SendStore(db), {"claude": fake})
    req = make_req(max_out=10)
    prep = svc.prepare(req)
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=prep.digest))
    assert st.output_text == "x" * 10


def test_reprepare_same_content_returns_existing(svc, fake):
    req = make_req()
    d = svc.prepare(req).digest
    svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    again = svc.prepare(req)
    assert again.state == SendState.SUCCEEDED
    assert len(fake.received) == 1


def test_same_id_different_content_rejected(svc):
    svc.prepare(make_req(text="one"))
    with pytest.raises(Conflict):
        svc.prepare(make_req(text="two"))


def test_digest_mismatch(svc, fake):
    req = make_req()
    svc.prepare(req)
    with pytest.raises(DigestMismatch):
        svc.commit(CommitRequest(request_id=req.request_id, expected_digest="0" * 64))
    assert fake.received == []
    assert svc.status(req.request_id).state == SendState.PREPARED


def test_commit_unknown_id(svc):
    with pytest.raises(NotFound):
        svc.commit(CommitRequest(request_id="nope-0000", expected_digest="x"))


def test_prepare_expired_rejected(svc):
    with pytest.raises(Expired):
        svc.prepare(make_req(ttl=-1))


def test_commit_expired_rejected(db, fake):
    now = [1000.0]
    svc = GatewayService(SendStore(db), {"claude": fake}, clock=lambda: now[0])
    req = PrepareRequest(request_id="req-00000001", payload=make_payload(), expires_at=1010.0)
    d = svc.prepare(req).digest
    now[0] = 1011.0
    with pytest.raises(Expired):
        svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert fake.received == []


def test_size_limit_contract_a(svc):
    # system "sys"(3) + user
    svc.prepare(make_req(rid="req-ok-a-01", text="a" * 1997))
    with pytest.raises(PayloadTooLarge):
        svc.prepare(make_req(rid="req-ng-a-01", text="a" * 1998))


def test_size_limit_contract_b(svc):
    svc.prepare(make_req(rid="req-ok-b-01", text="b" * 1497, contract="B-spec-code@1"))
    with pytest.raises(PayloadTooLarge):
        svc.prepare(make_req(rid="req-ng-b-01", text="b" * 1498, contract="B-spec-code@1"))


@pytest.mark.parametrize("contract", ["C-x@1", "A-faq", "A@1", "", "Z-foo@2"])
def test_unknown_contract(svc, contract):
    with pytest.raises(UnknownContract):
        svc.prepare(make_req(contract=contract))


def test_concurrent_commit_sends_once(db):
    fake = FakeAdapter(reply="ok", delay=0.2)
    svc = GatewayService(SendStore(db), {"claude": fake})
    req = make_req()
    d = svc.prepare(req).digest
    barrier = threading.Barrier(8)
    results, errors = [], []

    def worker():
        barrier.wait()
        try:
            results.append(svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d)))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(fake.received) == 1
    assert svc.status(req.request_id).state == SendState.SUCCEEDED


def test_second_commit_does_not_resend(svc, fake):
    req = make_req()
    d = svc.prepare(req).digest
    svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st.state == SendState.SUCCEEDED
    assert len(fake.received) == 1


class FailingClaimStore(SendStore):
    def claim_attempt(self, request_id):
        raise StoreError("disk full")


def test_record_failure_does_not_call_adapter(db, fake):
    svc = GatewayService(FailingClaimStore(db), {"claude": fake})
    req = make_req()
    d = svc.prepare(req).digest
    with pytest.raises(StoreError):
        svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert fake.received == []


@pytest.mark.parametrize(
    "mode,kind",
    [("not_sent", FailureKind.not_sent), ("rejected", FailureKind.rejected),
     ("unknown", FailureKind.unknown), ("timeout", FailureKind.unknown)],
)
def test_failure_kinds(db, mode, kind):
    fake = FakeAdapter(mode=mode)
    svc = GatewayService(SendStore(db), {"claude": fake})
    req = make_req()
    d = svc.prepare(req).digest
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st.state == SendState.FAILED
    assert st.failure == kind
    assert st.output_text is None


def test_unknown_is_not_resent(db):
    fake = FakeAdapter(mode="timeout")
    svc = GatewayService(SendStore(db), {"claude": fake})
    req = make_req()
    d = svc.prepare(req).digest
    svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    fake.mode = "ok"
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st.state == SendState.FAILED and st.failure == FailureKind.unknown
    assert svc.prepare(req).state == SendState.FAILED
    assert len(fake.received) == 1


def test_recover_on_startup(db):
    # 送信中にプロセスが落ちた状況を再現: ATTEMPTING のまま残す
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter()})
    req = make_req()
    d = svc.prepare(req).digest
    assert store.claim_attempt(req.request_id)

    fake2 = FakeAdapter()
    svc2 = GatewayService(SendStore(db), {"claude": fake2})
    assert svc2.recover_on_startup() == 1
    st = svc2.status(req.request_id)
    assert st.state == SendState.FAILED and st.failure == FailureKind.unknown
    svc2.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert fake2.received == []


def test_destination_not_configured(db):
    from gateway.service import InvalidRequest

    svc = GatewayService(SendStore(db), {"claude": FakeAdapter()})
    with pytest.raises(InvalidRequest):
        svc.prepare(make_req(dest="codex"))


# ---- 実アダプタ（SDK はダミーモジュールで差し替え。ネットワークに出ない） ----

SECRET = "sk-test-SECRET-abcdef0123456789"


@pytest.fixture
def key_file(tmp_path):
    p = tmp_path / "key"
    p.write_text(SECRET + "\n")
    return p


def _fake_sdk(name: str, raise_exc: Exception | None = None, init_exc: Exception | None = None,
              captured: dict | None = None):
    mod = types.ModuleType(name)

    class APIStatusError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    class APITimeoutError(Exception):
        pass

    def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        if mod.raise_exc is not None:
            raise mod.raise_exc
        if name == "anthropic":
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="claude-out")])
        msg = types.SimpleNamespace(content="codex-out")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class Client:
        def __init__(self, api_key, max_retries, timeout):
            if init_exc is not None:
                raise init_exc
            assert max_retries == 0
            if captured is not None:
                captured["_max_retries"] = max_retries
            self.messages = types.SimpleNamespace(create=create)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    mod.raise_exc = raise_exc
    mod.APIStatusError = APIStatusError
    mod.APITimeoutError = APITimeoutError
    mod.Anthropic = Client
    mod.OpenAI = Client
    return mod


def test_claude_adapter_ok(monkeypatch, key_file):
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk("anthropic", captured=cap))
    out = ClaudeAdapter(model="m", key_path=key_file).send(make_payload(text="hi"))
    assert out == "claude-out"
    assert cap["system"] == "sys"
    assert cap["messages"] == [{"role": "user", "content": "hi"}]
    assert cap["_max_retries"] == 0


def test_codex_adapter_ok(monkeypatch, key_file):
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_sdk("openai", captured=cap))
    out = CodexAdapter(model="m", key_path=key_file).send(make_payload(dest="codex"))
    assert out == "codex-out"
    assert cap["messages"][0] == {"role": "system", "content": "sys"}


def test_adapter_sdk_missing_is_not_sent(monkeypatch, key_file):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(NotSentError):
        ClaudeAdapter(model="m", key_path=key_file).send(make_payload())


def test_adapter_missing_key_file_is_not_sent(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openai", _fake_sdk("openai"))
    with pytest.raises(NotSentError):
        CodexAdapter(model="m", key_path=tmp_path / "missing").send(make_payload(dest="codex"))


@pytest.mark.parametrize("which", ["anthropic", "openai"])
@pytest.mark.parametrize("where", ["init", "call_generic", "call_status", "call_timeout"])
def test_api_key_not_in_exception(monkeypatch, key_file, db, which, where):
    leak = f"invalid key {SECRET}"
    sdk = _fake_sdk(which, init_exc=ValueError(leak) if where == "init" else None)
    if where == "call_generic":
        sdk.raise_exc = RuntimeError(leak)
    elif where == "call_status":
        sdk.raise_exc = sdk.APIStatusError(leak, 401)
    elif where == "call_timeout":
        sdk.raise_exc = sdk.APITimeoutError(leak)
    monkeypatch.setitem(sys.modules, which, sdk)
    adapter = (ClaudeAdapter if which == "anthropic" else CodexAdapter)(model="m", key_path=key_file)
    dest = "claude" if which == "anthropic" else "codex"

    with pytest.raises(Exception) as ei:
        adapter.send(make_payload(dest=dest))
    e = ei.value
    assert SECRET not in str(e) and SECRET not in repr(e)
    assert e.__cause__ is None or SECRET not in str(e.__cause__)
    assert e.__suppress_context__

    svc = GatewayService(SendStore(db), {dest: adapter})
    req = make_req(dest=dest)
    d = svc.prepare(req).digest
    st = svc.commit(CommitRequest(request_id=req.request_id, expected_digest=d))
    assert st.state == SendState.FAILED
    expected = {"init": FailureKind.not_sent, "call_generic": FailureKind.unknown,
                "call_status": FailureKind.rejected, "call_timeout": FailureKind.unknown}[where]
    assert st.failure == expected
    assert SECRET not in st.model_dump_json()
