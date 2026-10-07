from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
import uvicorn

from common.schemas import C1Decision, C2Verdict, FailureKind, SendPayload, SendState, payload_digest
from gateway.adapters import FakeAdapter
from gateway.app import create_app
from gateway.service import GatewayService
from gateway.store import SendStore
from integration.flow import contract_a_input, default_c1, default_c2, run_contract_a
from integration.gateway_client import GatewayClient
from integration.observe import AttemptingProbe, observe
from orchestrator.candidate import build_candidate
from orchestrator.policy import JST, check_eligibility, load_policy

from datetime import datetime

USER = "u_general"
REF = "expl-keihi-001@1"
OPTS = {"文体": "です・ます調", "長さ": "400字以内"}
DEST = "claude"
POLICY = load_policy()


# ---- スタブ（judge が無い場合の C1/C2） ----

_DEST_DEFAULT = object()


class StubC1:
    def __init__(self, route: str = "A", destination=_DEST_DEFAULT) -> None:
        self.route_value = route
        self.destination = DEST if destination is _DEST_DEFAULT else destination

    def route(self, user, request, input):  # noqa: A002
        return C1Decision(route=self.route_value, destination=self.destination, reason="stub", model="stub", revision="0")


class StubC2:
    def __init__(self, decision: str = "allow") -> None:
        self.decision = decision
        self.calls = 0

    def check(self, payload: SendPayload) -> C2Verdict:
        self.calls += 1
        return C2Verdict(decision=self.decision, reason="stub", model="stub", revision="0")


class DigestConfirmer:
    """利用者が候補の本文を見て確定した。見た候補（digest）を記録する。"""

    def __init__(self) -> None:
        self.seen: list = []

    def confirm(self, candidate):
        self.seen.append(candidate)
        return candidate.digest


class FixedConfirmer:
    def __init__(self, digest: str) -> None:
        self.digest = digest

    def confirm(self, candidate):
        return self.digest


class CountingGateway:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def prepare(self, req):
        self.calls.append("prepare")
        return self.inner.prepare(req)

    def commit(self, req):
        self.calls.append("commit")
        return self.inner.commit(req)

    def status(self, rid):
        self.calls.append("status")
        return self.inner.status(rid)


def c1():
    return default_c1() or StubC1()


def c2_allow():
    return default_c2() or StubC2()


def _rows(store: SendStore) -> int:
    conn = store._connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM sends").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def gw(tmp_path):
    store = SendStore(tmp_path / "gateway.db")
    fake = FakeAdapter(reply="Q. 期限は？ A. 翌月10日まで")
    probe = AttemptingProbe(store)
    fake.on_send = probe
    svc = GatewayService(store, {"claude": fake, "codex": FakeAdapter()})
    return svc, store, fake, probe


def _assert_sent(res, store, fake, probe, confirmer, before=0):
    assert res.c1 is not None and res.c1.route == "A"
    assert res.outcome == "sent", res.reason
    run = res.run
    assert run.gateway_state == SendState.SUCCEEDED
    # 受信先が受け取った payload ＝ 利用者が確認した候補（digest 再計算で照合）
    assert len(fake.received) == before + 1
    got = fake.received[-1]
    confirmed = confirmer.seen[-1]
    assert payload_digest(got) == confirmed.digest == run.digest
    # 本文に承認済み説明文以外（ユーザー由来の自由文など）が入っていない
    approved = POLICY.approved[REF].body
    assert got == confirmed.payload
    user_msgs = [m.content for m in got.messages if m.role == "user"]
    assert len(user_msgs) == 1
    expected_user = f"文体: {OPTS['文体']}\n長さ: {OPTS['長さ']}\n形式: 質問と回答の組を3〜5個\n\n説明文:\n{approved}"
    assert user_msgs[0] == expected_user
    assert got.contract == "A-faq-format@1" and got.destination == DEST
    # 観測項目を分けて読める
    obs = observe(run, store=store, probe=probe, received_before=before, received_after=len(fake.received))
    assert obs.gateway_received and obs.attempting_recorded and obs.destination_received and obs.result_saved
    stages = [e.stage for e in run.audit]
    assert stages.index("gateway_received") < stages.index("attempted")
    assert run.gateway_received and run.attempted


# 1. in-process
def test_contract_a_in_process(gw):
    svc, store, fake, probe = gw
    conf = DigestConfirmer()
    res = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=conf, gateway=svc,
                         request_text="経費精算の説明を FAQ にして")
    _assert_sent(res, store, fake, probe, conf)
    assert res.run.output_text == fake.reply


# 2. UDS 実通信
@pytest.fixture
def uds_server(gw):
    svc, store, fake, probe = gw
    d = tempfile.mkdtemp(prefix="gw", dir="/tmp" if Path("/tmp").is_dir() else None)
    sock = str(Path(d) / "g.sock")
    server = uvicorn.Server(uvicorn.Config(create_app(svc), uds=sock, log_level="warning", access_log=False))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline and th.is_alive(), "uvicorn が起動しない"
        time.sleep(0.02)
    try:
        yield sock, store, fake, probe
    finally:
        server.should_exit = True
        th.join(timeout=10)
        shutil.rmtree(d, ignore_errors=True)


def test_contract_a_over_uds(uds_server):
    sock, store, fake, probe = uds_server
    conf = DigestConfirmer()
    with GatewayClient(sock, timeout_s=10) as client:
        res = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=conf, gateway=client)
        _assert_sent(res, store, fake, probe, conf)
        st = client.status(res.run.request_id)
        assert st.state == SendState.SUCCEEDED and st.digest == res.run.digest


def test_uds_http_errors_are_mapped(uds_server):
    from integration.gateway_client import GatewayHTTPError

    sock, *_ = uds_server
    with GatewayClient(sock, timeout_s=10) as client:
        with pytest.raises(GatewayHTTPError) as ei:
            client.status("no-such-request")
        assert ei.value.status_code == 404 and ei.value.detail == "NotFound"


# 3. 二重 commit
def test_double_run_same_request_id_sends_once(gw):
    svc, store, fake, probe = gw
    rid = uuid.uuid4().hex
    conf = DigestConfirmer()
    r1 = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=conf, gateway=svc, request_id=rid)
    r2 = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=conf, gateway=svc, request_id=rid)
    assert r1.outcome == "sent" and r2.outcome == "sent"
    assert r1.run.digest == r2.run.digest
    assert len(fake.received) == 1
    assert _rows(store) == 1


# 4. unknown 注入
def test_unknown_outcome_is_not_sent_and_not_resent(gw):
    svc, store, fake, probe = gw
    fake.mode = "unknown"
    rid = uuid.uuid4().hex
    conf = DigestConfirmer()
    r1 = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=conf, gateway=svc, request_id=rid)
    assert r1.outcome != "sent" and r1.outcome == "held"
    st = svc.status(rid)
    assert st.state == SendState.FAILED and st.failure == FailureKind.unknown
    obs = observe(r1.run, store=store, probe=probe, received_after=len(fake.received))
    assert obs.gateway_received and obs.attempting_recorded and obs.destination_received and obs.result_saved
    fake.mode = "ok"
    r2 = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=conf, gateway=svc, request_id=rid)
    assert r2.outcome != "sent"
    assert len(fake.received) == 1


# 5. C2 block
def test_c2_block_creates_no_gateway_row(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    res = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=StubC2("block"), confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "held" and res.run.stopped_at == "C2"
    assert counting.calls == []
    assert _rows(store) == 0 and fake.received == []
    obs = observe(res.run, store=store, probe=probe)
    assert not (obs.gateway_received or obs.attempting_recorded or obs.destination_received or obs.result_saved)


# 6. 確認後に選択値を変更
def test_options_changed_after_confirm_sends_nothing(gw):
    svc, store, fake, probe = gw
    now = datetime.now(JST)
    elig = check_eligibility(POLICY, USER, contract_a_input(REF, OPTS), DEST, now)
    confirmed = build_candidate(user_id=USER, contract=elig.contract, approved=elig.approved,
                                inp=contract_a_input(REF, OPTS), destination=DEST, now=now)
    changed = {**OPTS, "長さ": "200字以内"}
    counting = CountingGateway(svc)
    res = run_contract_a(USER, REF, changed, DEST, c1=c1(), c2=c2_allow(),
                         confirmer=FixedConfirmer(confirmed.digest), gateway=counting, now=now)
    assert res.outcome == "rejected" and res.run.stopped_at == "digest"
    assert counting.calls == [] and _rows(store) == 0 and fake.received == []


# C1 が A 以外 → 送らない
def test_c1_non_a_route_sends_nothing(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    c2 = StubC2()
    res = run_contract_a(USER, REF, OPTS, DEST, c1=StubC1("human"), c2=c2, confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "not_routed"
    run = res.run
    assert (run.outcome, run.stopped_at) == ("held", "C1")
    assert not run.gateway_received and not run.attempted and run.request_id is None
    c1_ev = [e for e in run.audit if e.stage == "C1"]
    assert len(c1_ev) == 1 and c1_ev[0].result == "human"
    for part in ("route=human", f"destination={DEST}", "model=stub@0"):
        assert part in c1_ev[0].detail
    assert [e.stage for e in run.audit] == ["C1", "result"]
    assert c2.calls == 0 and counting.calls == [] and _rows(store) == 0
    obs = observe(run, store=store)
    assert not (obs.gateway_received or obs.result_saved)


def test_c1_reject_route_is_rejected_at_c1(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    res = run_contract_a(USER, REF, OPTS, DEST, c1=StubC1("reject"), c2=StubC2(), confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "not_routed" and res.c1.route == "reject"
    assert (res.run.outcome, res.run.stopped_at) == ("rejected", "C1")
    assert counting.calls == [] and _rows(store) == 0


class BrokenC1:
    def route(self, user, request, input):  # noqa: A002
        raise RuntimeError("model down")


def test_c1_failure_held_at_c1(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    c2 = StubC2()
    res = run_contract_a(USER, REF, OPTS, DEST, c1=BrokenC1(), c2=c2, confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "held" and res.c1 is None
    assert (res.run.outcome, res.run.stopped_at) == ("held", "C1")
    assert any(e.stage == "C1" and e.result == "error" and "model down" in e.detail for e in res.run.audit)
    assert c2.calls == 0 and counting.calls == []


# Gateway に接続できない → 本文は未到達と確定（gateway_received=False）
def test_gateway_unreachable_marks_not_received(tmp_path):
    from integration.gateway_client import GatewayUnavailable
    from orchestrator.pipeline import GatewayNotReached

    assert issubclass(GatewayUnavailable, GatewayNotReached)
    sock = str(tmp_path / "missing.sock")
    with GatewayClient(sock, timeout_s=0.5) as client:
        res = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=DigestConfirmer(), gateway=client)
    assert res.outcome == "held" and res.run.stopped_at == "none"
    assert "GatewayUnavailable" in res.reason
    assert not res.run.gateway_received and not res.run.attempted


def test_judge_used_when_available():
    try:
        import judge.rules  # noqa: F401
    except ImportError:
        pytest.skip("judge 未作成: スタブ C1/C2 で実行")
    assert type(c1()).__name__ == "RuleC1Router" and type(c2_allow()).__name__ == "RuleC2Gate"


# commit がタイムアウト → 自動再試行せず status で確定。未確定なら held、送信は 1 回だけ
def test_commit_timeout_resolves_by_status_without_retry(uds_server):
    sock, store, fake, probe = uds_server
    fake.delay = 1.0
    with GatewayClient(sock, timeout_s=0.3) as client:
        res = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=DigestConfirmer(), gateway=client)
    assert res.outcome == "held"
    assert any(e.stage == "attempted" and e.result == "error" for e in res.run.audit)
    assert res.run.gateway_state == SendState.ATTEMPTING
    time.sleep(1.2)
    assert len(fake.received) == 1
    assert store.get(res.run.request_id).state == SendState.SUCCEEDED


# ---- C1 の宛先照合（設計書 決定事項 5: 宛先は C1 が決める。不一致時の扱いは規定なし→安全側で保留） ----

def test_c1_destination_mismatch_is_held_and_not_sent(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    c2 = StubC2()
    res = run_contract_a(USER, REF, OPTS, DEST, c1=StubC1("A", destination="codex"), c2=c2,
                         confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "held"
    assert (res.run.outcome, res.run.stopped_at) == ("held", "C1")
    assert any(e.stage == "C1" and e.result == "destination_mismatch" for e in res.run.audit)
    assert not res.run.gateway_received and c2.calls == 0 and counting.calls == [] and _rows(store) == 0


def test_c1_destination_none_uses_caller_and_is_audited(gw):
    svc, store, fake, probe = gw
    conf = DigestConfirmer()
    res = run_contract_a(USER, REF, OPTS, DEST, c1=StubC1("A", destination=None), c2=c2_allow(), confirmer=conf, gateway=svc)
    assert res.outcome == "sent", res.reason
    assert fake.received[-1].destination == DEST
    first = res.run.audit[0]
    assert first.stage == "C1" and "destination=None" in first.detail and DEST in first.detail


# ---- 応答の閲覧制約（契約 A = source_level_requester_only） ----

def test_contract_a_response_visibility_source_level_requester_only(gw):
    svc, store, fake, probe = gw
    res = run_contract_a(USER, REF, OPTS, DEST, c1=c1(), c2=c2_allow(), confirmer=DigestConfirmer(), gateway=svc)
    assert res.outcome == "sent", res.reason
    run = res.run
    assert run.response_visibility == "source_level_requester_only"
    assert run.visible_to == USER and run.no_exec is True
    assert run.min_view_level == POLICY.approved[REF].source_level
    assert run.can_view(USER, 1)
    assert not run.can_view("u_manager", 2)  # 依頼者以外は不可
    assert not run.can_view(USER, None)  # レベル不明なら見せない
