"""監査ストア（orchestrator.audit_store）の回帰試験。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from common.schemas import C1Decision, SendState
from integration.flow import run_contract_a
from integration.flow_b import run_contract_b
from orchestrator.audit_store import AuditRecord, AuditStore
from orchestrator.pipeline import GatewayNotReached
from orchestrator.policy import load_policy
from tests.orchestrator.test_pipeline import (
    A_INPUT,
    B_INPUT,
    B_TEXT,
    NOW,
    FakeC2,
    FakeConfirmer,
    FakeGateway,
    RaisingPrepareGateway,
    run,
)

POLICY = load_policy()
SECRET_REQ = "依頼文の秘密断片ZQX-7781を含む本人の依頼です"
OUTPUT = "Q1. 出力テキストの秘密断片OUT-5521"


@pytest.fixture
def store(tmp_path: Path):
    s = AuditStore(tmp_path / "audit.db")
    yield s
    s.close()


def all_text(path: str) -> str:
    """DB の全テーブル全行を文字列化する。"""
    con = sqlite3.connect(path)
    try:
        out = []
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            for row in con.execute(f"SELECT * FROM {t}"):
                out.append(" ".join("" if v is None else str(v) for v in row))
        return "\n".join(out)
    finally:
        con.close()


class SuccessGateway(FakeGateway):
    def commit(self, req):
        st = super().commit(req)
        return st.model_copy(update={"output_text": OUTPUT})


class LeakyC2(FakeC2):
    """理由に本文をそのまま引用する判定（除去されるべき）。"""

    def check(self, payload):
        v = super().check(payload)
        text = payload.messages[-1].content
        return v.model_copy(update={"reason": f"引用: {text}"})


class StubC1:
    def __init__(self, route: str, destination: str | None, reason: str = "stub"):
        self.d = C1Decision(route=route, destination=destination, reason=reason, model="c1m", revision="r1",
                            probs={route: 0.9})

    def route(self, user, request, input):  # noqa: A002
        return self.d


class RaisingC1:
    def route(self, user, request, input):  # noqa: A002
        raise RuntimeError("c1 down")


# ---- 本文非保存 ----

def test_body_not_stored_contract_b_sent(store: AuditStore):
    gw = SuccessGateway()
    res = run_contract_b("u_manager", "spec-csv-001@1", B_TEXT, {"言語": "Python 3.11"}, "codex",
                         c2=LeakyC2(), confirmer=FakeConfirmer(), gateway=gw, now=NOW, audit_sink=store,
                         c1=StubC1("B", "codex", reason=f"依頼文: {B_TEXT}"))
    assert res.outcome == "sent" and res.run.audit_persisted is True
    dump = all_text(store.path)
    payload_texts = [m.content for m in gw.prepared[0].payload.messages]
    for body in [B_TEXT, OUTPUT, *payload_texts]:
        assert body not in dump
        for line in body.splitlines():
            if len(line.strip()) >= 8:
                assert line.strip() not in dump
    # 対応付けに必要な項目は残る
    (row,) = store.runs()
    assert row["request_id"] == res.run.request_id and row["digest"] == res.run.digest
    assert row["user_id"] == "u_manager" and row["destination"] == "codex" and row["contract"] == "B-csv-codegen@1"
    assert row["outcome"] == "sent" and row["gateway_state"] == SendState.SUCCEEDED.value
    assert row["c1_model"] == "c1m" and row["c1_revision"] == "r1" and row["c2_model"] == "fake"
    assert row["output_sha256"] and row["output_chars"] == len(OUTPUT)
    assert store.events(row["run_id"])[0]["stage"] == "C1"


def test_body_not_stored_contract_a_sent(store: AuditStore):
    gw = SuccessGateway()
    res = run_contract_a("u_general", "expl-keihi-001@1", A_INPUT["options"], "claude",
                         c2=LeakyC2(), confirmer=FakeConfirmer(), gateway=gw, now=NOW, audit_sink=store,
                         c1=StubC1("A", "claude", reason=SECRET_REQ), request_text=SECRET_REQ)
    assert res.outcome == "sent" and res.run.audit_persisted is True
    dump = all_text(store.path)
    for body in [SECRET_REQ, OUTPUT, *(m.content for m in gw.prepared[0].payload.messages)]:
        assert body not in dump


# ---- 各停止点で記録される ----

def _only_run(store: AuditStore) -> tuple[dict, list[dict]]:
    (row,) = store.runs()
    return row, store.events(row["run_id"])


def test_stop_eligibility(store):
    res, _ = run(user="u_general", inp=B_INPUT, audit_sink=store)
    row, ev = _only_run(store)
    assert res.stopped_at == "eligibility" and row["stopped_at"] == "eligibility" and row["outcome"] == "rejected"
    assert [e["stage"] for e in ev] == ["eligibility", "result"]
    assert B_TEXT not in all_text(store.path)


@pytest.mark.parametrize("c1,expected", [
    (StubC1("reject", None), "rejected"),
    (StubC1("A", "codex"), "held"),  # 宛先不一致
    (RaisingC1(), "held"),
])
def test_stop_c1(store, c1, expected):
    res = run_contract_a("u_general", "expl-keihi-001@1", A_INPUT["options"], "claude",
                         c2=FakeC2(), confirmer=FakeConfirmer(), gateway=FakeGateway(), now=NOW,
                         audit_sink=store, c1=c1, request_text=SECRET_REQ)
    row, ev = _only_run(store)
    assert res.run.stopped_at == "C1" and row["stopped_at"] == "C1" and row["outcome"] == expected
    assert ev[0]["stage"] == "C1" and ev[-1]["stage"] == "result"
    assert res.run.audit_persisted is True
    assert SECRET_REQ not in all_text(store.path)


def test_stop_c1_contract_b(store):
    res = run_contract_b("u_manager", "spec-csv-001@1", B_TEXT, {"言語": "Python 3.11"}, "codex",
                         c2=FakeC2(), confirmer=FakeConfirmer(), gateway=FakeGateway(), now=NOW,
                         audit_sink=store, c1=StubC1("local", None, reason=B_TEXT))
    row, _ = _only_run(store)
    assert res.outcome == "not_routed" and row["stopped_at"] == "C1"
    assert B_TEXT not in all_text(store.path)


def test_stop_c2(store):
    res, gw = run(user="u_manager", inp=B_INPUT, c2=FakeC2("block"), audit_sink=store)
    row, ev = _only_run(store)
    assert row["stopped_at"] == "C2" and row["c2_decision"] == "block" and gw.calls == 0
    assert row["digest"] == res.digest and row["request_id"] == res.request_id


def test_stop_confirm(store):
    run(confirmer=FakeConfirmer("none"), audit_sink=store)
    row, ev = _only_run(store)
    assert row["stopped_at"] == "confirm" and any(e["stage"] == "confirm" for e in ev)


def test_stop_digest(store):
    run(confirmer=FakeConfirmer("other", other="f" * 64), audit_sink=store)
    row, _ = _only_run(store)
    assert row["stopped_at"] == "digest" and row["outcome"] == "rejected"


def test_stop_gateway_digest_mismatch(store):
    run(gw=FakeGateway(tamper_digest=True), audit_sink=store)
    row, _ = _only_run(store)
    assert row["stopped_at"] == "digest" and row["gateway_received"] == 1 and row["attempted"] == 0


@pytest.mark.parametrize("gw,received", [
    (RaisingPrepareGateway(GatewayNotReached("refused")), 0),
    (RaisingPrepareGateway(TimeoutError("read timeout")), 1),
])
def test_stop_gateway_prepare_failure(store, gw, received):
    run(gw=gw, audit_sink=store)
    row, ev = _only_run(store)
    assert row["outcome"] == "held" and row["gateway_received"] == received
    assert any(e["stage"] == "gateway_received" for e in ev)


def test_gateway_failed_recorded(store):
    run(gw=FakeGateway(final=SendState.FAILED), audit_sink=store)
    row, _ = _only_run(store)
    assert row["attempted"] == 1 and row["gateway_state"] == SendState.FAILED.value and row["outcome"] == "held"


def test_run_id_groups_each_run(store):
    run(audit_sink=store)
    run(audit_sink=store)
    rows = store.runs()
    assert len(rows) == 2 and rows[0]["run_id"] != rows[1]["run_id"]
    for r in rows:
        assert {e["run_id"] for e in store.events(r["run_id"])} == {r["run_id"]}


# ---- 追記専用 ----

def test_no_update_delete_api():
    names = {n for n in dir(AuditStore) if not n.startswith("_")}
    assert not names & {"update", "delete", "remove", "clear", "purge", "execute"}
    assert names == {"append", "runs", "events", "close"}


@pytest.mark.parametrize("sql", [
    "UPDATE runs SET outcome='sent'",
    "DELETE FROM runs",
    "UPDATE events SET detail='x'",
    "DELETE FROM events",
])
def test_db_rejects_update_delete(store, sql):
    run(audit_sink=store)
    con = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(sql)
    finally:
        con.close()


def test_wal_and_synchronous_full(store):
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


def test_duplicate_run_id_rejected_atomically(store):
    rec = AuditRecord(run={"recorded_at": "2026-10-07T00:00:00+00:00", "outcome": "held", "stopped_at": "none", "reason": "r",
                           "gateway_received": 0, "attempted": 0}, events=(("x", "y", "z"),))
    store.append(rec)
    with pytest.raises(sqlite3.IntegrityError):
        store.append(rec)
    assert len(store.runs()) == 1 and len(store.events(rec.run_id)) == 1


# ---- 書き込み失敗 ----

class BrokenSink:
    def __init__(self):
        self.calls = 0

    def append(self, record):
        self.calls += 1
        raise sqlite3.OperationalError("disk I/O error")


def test_write_failure_after_send_does_not_change_outcome_or_resend():
    sink = BrokenSink()
    gw = FakeGateway()
    res, _ = run(gw=gw, audit_sink=sink)
    assert res.outcome == "sent" and res.audit_persisted is False and res.audit_error == "OperationalError"
    assert len(gw.prepared) == 1 and len(gw.commits) == 1 and sink.calls == 1


def test_write_failure_on_stop_keeps_stop():
    res, gw = run(c2=FakeC2("block"), audit_sink=BrokenSink())
    assert (res.outcome, res.stopped_at) == ("held", "C2") and res.audit_persisted is False and gw.calls == 0


def test_write_failure_on_c1_stop():
    res = run_contract_a("u_general", "expl-keihi-001@1", A_INPUT["options"], "claude",
                         c2=FakeC2(), confirmer=FakeConfirmer(), gateway=FakeGateway(), now=NOW,
                         audit_sink=BrokenSink(), c1=StubC1("reject", None))
    assert res.run.stopped_at == "C1" and res.run.audit_persisted is False


def test_no_sink_keeps_previous_behavior():
    res, _ = run()
    assert res.outcome == "sent" and res.audit_persisted is None and res.audit_run_id is None
