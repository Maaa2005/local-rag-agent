"""契約 B（B-csv-codegen@1）の通し試験。test_contract_a_e2e.py と同じ流儀。"""
from __future__ import annotations

import dataclasses
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
import uvicorn

from common.schemas import C1Decision, C2Verdict, SendPayload, SendState, payload_digest
from gateway.adapters import FakeAdapter
from gateway.app import create_app
from gateway.service import GatewayService
from gateway.store import SendStore
from integration.flow import default_c1, default_c2
from integration.flow_b import CONTRACT_B, contract_b_input, run_contract_b
from integration.gateway_client import GatewayClient
from integration.observe import AttemptingProbe, observe
from orchestrator.candidate import B_SYSTEM, build_candidate
from orchestrator.policy import JST, check_eligibility, load_policy

USER = "u_manager"
SPEC = "spec-csv-001@1"
OPTS = {"言語": "Python 3.11"}
DEST = "codex"
FREE = "category ごとに amount を合計し、合計の大きい順に並べた結果を新しい CSV に書き出す関数を書いてください。"
POLICY = load_policy()

# 機密語・個人情報を含む依頼文（ルール C2 の person_name / employee_id / counterparty に当たる）
FREE_SECRET = "田中さん（社員番号 E12345）が株式会社サンプルに出した経費だけを category ごとに合計する関数を書いてください。"


# ---- スタブ（judge が無い場合の C1/C2） ----

class StubC1:
    def __init__(self, route: str = "B") -> None:
        self.route_value = route

    def route(self, user, request, input):  # noqa: A002
        return C1Decision(route=self.route_value, destination=DEST, reason="stub", model="stub", revision="0")


class StubC2:
    """judge が無いときの代替。ルール C2 と同じ趣旨の最小パターンで block する。"""

    _PAT = re.compile(r"([一-龥]{1,4}さん|[EＥ]\d{3,}|株式会社)")

    def __init__(self, decision: str | None = None) -> None:
        self.decision = decision
        self.calls = 0

    def check(self, payload: SendPayload) -> C2Verdict:
        self.calls += 1
        if self.decision is not None:
            return C2Verdict(decision=self.decision, reason="stub", model="stub", revision="0")
        text = "\n".join(m.content for m in payload.messages)
        d = "block" if self._PAT.search(text) else "allow"
        return C2Verdict(decision=d, reason="stub", model="stub", revision="0")


class CountingC2:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def check(self, payload):
        self.calls += 1
        return self.inner.check(payload)


class DigestConfirmer:
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


def c2():
    return default_c2() or StubC2()


def _rows(store: SendStore) -> int:
    conn = store._connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM sends").fetchone()[0]
    finally:
        conn.close()


def _expected_user(free_text: str) -> str:
    return f"言語: {OPTS['言語']}\n仕様:\n{POLICY.approved[SPEC].body}\n依頼:\n{free_text}"


@pytest.fixture
def gw(tmp_path):
    store = SendStore(tmp_path / "gateway.db")
    fake = FakeAdapter(reply="def summarize(path_in, path_out): ...")
    probe = AttemptingProbe(store)
    fake.on_send = probe
    svc = GatewayService(store, {"claude": FakeAdapter(), "codex": fake})
    return svc, store, fake, probe


def _assert_nothing_sent(counting, store, fake):
    assert counting.calls == []
    assert _rows(store) == 0 and fake.received == []


def _assert_sent(res, store, fake, probe, confirmer, before=0):
    assert res.c1 is not None and res.c1.route == "B"
    assert res.outcome == "sent", res.reason
    run = res.run
    assert run.gateway_state == SendState.SUCCEEDED
    assert len(fake.received) == before + 1
    got = fake.received[-1]
    confirmed = confirmer.seen[-1]
    assert payload_digest(got) == confirmed.digest == run.digest
    assert got == confirmed.payload
    assert got.contract == CONTRACT_B and got.destination == DEST
    assert got.max_output_chars == POLICY.contracts[CONTRACT_B].output_chars
    obs = observe(run, store=store, probe=probe, received_before=before, received_after=len(fake.received))
    assert obs.gateway_received and obs.attempting_recorded and obs.destination_received and obs.result_saved
    stages = [e.stage for e in run.audit]
    assert stages.index("gateway_received") < stages.index("attempted")


# 正常通し（in-process）
def test_contract_b_in_process(gw):
    svc, store, fake, probe = gw
    conf = DigestConfirmer()
    res = run_contract_b(USER, SPEC, FREE, OPTS, DEST, c1=c1(), c2=c2(), confirmer=conf, gateway=svc)
    _assert_sent(res, store, fake, probe, conf)
    assert res.run.output_text == fake.reply


# 正常通し（UDS 実通信）
@pytest.fixture
def uds_server(gw):
    svc, store, fake, probe = gw
    d = tempfile.mkdtemp(prefix="gwb", dir="/tmp" if Path("/tmp").is_dir() else None)
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


def test_contract_b_over_uds(uds_server):
    sock, store, fake, probe = uds_server
    conf = DigestConfirmer()
    with GatewayClient(sock, timeout_s=10) as client:
        res = run_contract_b(USER, SPEC, FREE, OPTS, DEST, c1=c1(), c2=c2(), confirmer=conf, gateway=client)
        _assert_sent(res, store, fake, probe, conf)
        st = client.status(res.run.request_id)
        assert st.state == SendState.SUCCEEDED and st.digest == res.run.digest


# 本文は 仕様 + テンプレート + 依頼文 だけ
def test_body_is_only_spec_template_and_free_text(gw):
    svc, store, fake, probe = gw
    conf = DigestConfirmer()
    res = run_contract_b(USER, SPEC, FREE, OPTS, DEST, c1=c1(), c2=c2(), confirmer=conf, gateway=svc)
    assert res.outcome == "sent", res.reason
    got = fake.received[-1]
    assert [(m.role, m.content) for m in got.messages] == [("system", B_SYSTEM), ("user", _expected_user(FREE))]
    # 依頼文は 1 回だけ、加工されずに入る
    assert sum(m.content.count(FREE) for m in got.messages) == 1
    assert sum(len(m.content) for m in got.messages) <= POLICY.contracts[CONTRACT_B].payload_chars


# 依頼文 200 字超は送らない（C2・確認・Gateway のどれにも渡らない）
def test_free_text_over_200_chars_is_not_sent(gw):
    svc, store, fake, probe = gw
    counting, c2c = CountingGateway(svc), CountingC2(c2())
    long_text = "category ごとに amount を合計する関数を書いてください。" + "あ" * 200
    assert len(long_text) > 200
    res = run_contract_b(USER, SPEC, long_text, OPTS, DEST, c1=c1(), c2=c2c, confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "rejected" and res.run.stopped_at == "contract"
    assert "上限 200 字" in res.reason
    assert c2c.calls == 0
    _assert_nothing_sent(counting, store, fake)


def test_free_text_exactly_200_chars_is_sent(gw):
    svc, store, fake, probe = gw
    text = (FREE + "い" * 200)[:200]
    res = run_contract_b(USER, SPEC, text, OPTS, DEST, c1=c1(), c2=c2(), confirmer=DigestConfirmer(), gateway=svc)
    assert res.outcome == "sent", res.reason


def test_empty_free_text_is_not_sent(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    res = run_contract_b(USER, SPEC, "  ", OPTS, DEST, c1=c1(), c2=c2(), confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "rejected" and res.run.stopped_at == "contract"
    _assert_nothing_sent(counting, store, fake)


# 依頼文に機密語・個人情報 → C2 で止まる（確認にも Gateway にも進まない）
def test_secret_in_free_text_stops_at_c2(gw):
    svc, store, fake, probe = gw
    counting, conf = CountingGateway(svc), DigestConfirmer()
    res = run_contract_b(USER, SPEC, FREE_SECRET, OPTS, DEST, c1=c1(), c2=c2(), confirmer=conf, gateway=counting)
    assert res.outcome == "held" and res.run.stopped_at == "C2"
    assert res.run.c2_verdict is not None and res.run.c2_verdict.decision == "block"
    assert conf.seen == []
    _assert_nothing_sent(counting, store, fake)
    obs = observe(res.run, store=store, probe=probe)
    assert not (obs.gateway_received or obs.attempting_recorded or obs.destination_received or obs.result_saved)


def test_c2_hold_is_not_sent(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    res = run_contract_b(USER, SPEC, FREE, OPTS, DEST, c1=c1(), c2=StubC2("hold"), confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "held" and res.run.stopped_at == "C2"
    _assert_nothing_sent(counting, store, fake)


# 確認後に依頼文を変更 → digest 不一致で送らない
def test_free_text_changed_after_confirm_sends_nothing(gw):
    svc, store, fake, probe = gw
    now = datetime.now(JST)
    inp = contract_b_input(SPEC, FREE, OPTS)
    elig = check_eligibility(POLICY, USER, inp, DEST, now)
    assert elig.ok, elig.reason
    confirmed = build_candidate(user_id=USER, contract=elig.contract, approved=elig.approved,
                                inp=inp, destination=DEST, now=now)
    changed = FREE + "列名は変えないでください。"
    counting = CountingGateway(svc)
    res = run_contract_b(USER, SPEC, changed, OPTS, DEST, c1=c1(), c2=c2(),
                         confirmer=FixedConfirmer(confirmed.digest), gateway=counting, now=now)
    assert res.outcome == "rejected" and res.run.stopped_at == "digest"
    assert res.run.digest != confirmed.digest
    _assert_nothing_sent(counting, store, fake)


# 権限不足: 契約 B の利用権限がない利用者
def test_user_without_contract_b_is_not_sent(gw):
    svc, store, fake, probe = gw
    counting, c2c = CountingGateway(svc), CountingC2(c2())
    res = run_contract_b("u_general", SPEC, FREE, OPTS, DEST, c1=c1(), c2=c2c, confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "rejected" and res.run.stopped_at == "eligibility"
    assert "利用権限がない" in res.reason
    assert c2c.calls == 0
    _assert_nothing_sent(counting, store, fake)


# 権限不足: 契約 B は使えるが依頼文の外部利用を承認できない利用者（policies に該当者がいないので差し替え）
def test_user_without_free_text_approval_is_not_sent(gw):
    svc, store, fake, probe = gw
    u = POLICY.users[USER]
    pol = dataclasses.replace(POLICY, users={**POLICY.users, "u_nofree": dataclasses.replace(u, id="u_nofree", free_text_approval=())})
    counting = CountingGateway(svc)
    res = run_contract_b("u_nofree", SPEC, FREE, OPTS, DEST, c1=c1(), c2=c2(), confirmer=DigestConfirmer(),
                         gateway=counting, policy=pol)
    assert res.outcome == "rejected" and res.run.stopped_at == "eligibility"
    assert "外部利用承認できない" in res.reason
    _assert_nothing_sent(counting, store, fake)


# C1 が B 以外 → 送らない
def test_c1_non_b_route_sends_nothing(gw):
    svc, store, fake, probe = gw
    counting, c2c = CountingGateway(svc), CountingC2(c2())
    res = run_contract_b(USER, SPEC, FREE, OPTS, DEST, c1=StubC1("human"), c2=c2c, confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "not_routed" and res.run is None
    assert c2c.calls == 0
    _assert_nothing_sent(counting, store, fake)


# 仕様の代わりに契約 A 用の説明文を指定 → 送らない
def test_spec_ref_must_be_approved_for_contract_b(gw):
    svc, store, fake, probe = gw
    counting = CountingGateway(svc)
    res = run_contract_b(USER, "expl-keihi-001@1", FREE, OPTS, DEST, c1=c1(), c2=c2(), confirmer=DigestConfirmer(), gateway=counting)
    assert res.outcome == "rejected" and res.run.stopped_at == "eligibility"
    _assert_nothing_sent(counting, store, fake)
