"""監査ストア（orchestrator.audit_store）の回帰試験。"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from common.schemas import C1Decision, SendState
from integration.flow import run_contract_a
from integration.flow_b import run_contract_b
from judge.base import GUARD_QUESTION_VERSION, guarded_c2
from judge.rules import C1_QUESTION_VERSION, C2_QUESTION_VERSION, RuleC1Router, RuleC2Gate
from orchestrator.audit_store import AuditRecord, AuditStore, PurgeResult
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
    assert not names & {"update", "delete", "remove", "clear", "execute"}
    # 削除は保持期限切れの purge だけ
    assert names == {"append", "runs", "events", "close", "purge"}


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


# ---- 質問の版（F9） ----

def test_question_versions_recorded_with_rules(store):
    res = run_contract_b("u_manager", "spec-csv-001@1", B_TEXT, {"言語": "Python 3.11"}, "codex",
                         c2=RuleC2Gate(), confirmer=FakeConfirmer(), gateway=FakeGateway(), now=NOW,
                         audit_sink=store, c1=RuleC1Router())
    assert res.outcome == "sent"
    (row,) = store.runs()
    assert row["c1_question_version"] == C1_QUESTION_VERSION
    assert row["c2_question_version"] == C2_QUESTION_VERSION


class GuardedC2:
    """guard を通した C2（判定器が質問の版を返さない → hold）。"""

    def check(self, payload):
        return guarded_c2(FakeC2(), payload, 1.0, 10_000)


def test_guard_hold_records_guard_question_version(store):
    res, _ = run(c2=GuardedC2(), audit_sink=store)
    assert res.outcome == "held" and res.stopped_at == "C2"
    (row,) = store.runs()
    assert row["c2_decision"] == "hold" and row["c2_question_version"] == GUARD_QUESTION_VERSION
    assert row["c1_question_version"] is None  # C1 を通していない実行


# ---- 承認者と対象（F9） ----

def test_approver_recorded_contract_b(store):
    res, _ = run(user="u_manager", inp=B_INPUT, dest="codex", audit_sink=store)
    assert res.outcome == "sent"
    (row,) = store.runs()
    assert row["approver"] == "u_manager" and row["approver_authority"] == 1
    assert row["approved_ref"] == "spec-csv-001@1" and row["approved_by"] == "情報システム部長（架空）"
    assert row["digest"] == res.digest
    ok = [e for e in store.events(row["run_id"]) if e["stage"] == "confirm"]
    assert ok[0]["result"] == "ok" and "approver=u_manager" in ok[0]["detail"]


def test_approver_contract_a_has_no_free_text_authority(store):
    res, _ = run(audit_sink=store)
    assert res.outcome == "sent"
    (row,) = store.runs()
    # 契約 A は依頼文を取らないので権限欄は NULL（未記録ではなく対象外）
    assert row["approver"] == "u_general" and row["approver_authority"] is None
    assert row["approved_ref"] == "expl-keihi-001@1" and row["approved_by"] == "総務部長（架空）"


def test_unconfirmed_run_has_no_approver(store):
    res, _ = run(user="u_manager", inp=B_INPUT, dest="codex", confirmer=FakeConfirmer("none"), audit_sink=store)
    assert res.outcome == "held"
    (row,) = store.runs()
    assert row["approver"] is None and row["approver_authority"] is None
    assert row["approved_ref"] == "spec-csv-001@1"  # 対象は確定している


# ---- 保持期限（N7: メタデータ 90 日） ----

def _rec(recorded_at: datetime, n_events: int = 2) -> AuditRecord:
    return AuditRecord(run={"recorded_at": recorded_at.isoformat(), "outcome": "held", "stopped_at": "none",
                            "reason": "r", "gateway_received": 0, "attempted": 0},
                       events=tuple(("s", "r", "d") for _ in range(n_events)))


def test_purge_boundary_89_90_91_days(store):
    now = datetime.now(timezone.utc)
    recs = {d: _rec(now - timedelta(days=d)) for d in (89, 90, 91)}
    for r in recs.values():
        store.append(r)
    assert store.purge(now) == PurgeResult(runs=1, events=2)
    left = {r["run_id"] for r in store.runs()}
    # 90 日ちょうどは期限内として残す。91 日だけ消える
    assert left == {recs[89].run_id, recs[90].run_id}
    assert store.events(recs[91].run_id) == [] and len(store.events(recs[90].run_id)) == 2
    assert store.purge(now) == PurgeResult(runs=0, events=0)


def test_purge_with_future_now_is_refused_by_db(store):
    now = datetime.now(timezone.utc)
    young = _rec(now - timedelta(days=10))
    old = _rec(now - timedelta(days=200))
    store.append(young)
    store.append(old)
    # DB トリガーは実時刻で判定するので、未来の now で期限内の行を消すことはできない（全体を中止）
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.purge(now + timedelta(days=95))
    assert len(store.runs()) == 2 and len(store.events(old.run_id)) == 2


def test_purge_requires_aware_now(store):
    with pytest.raises(ValueError):
        store.purge(datetime(2026, 1, 1))


@pytest.mark.parametrize("table", ["runs", "events"])
def test_db_rejects_delete_within_retention(store, table):
    now = datetime.now(timezone.utc)
    rec = _rec(now - timedelta(days=89))
    store.append(rec)
    con = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", (rec.run_id,))
    finally:
        con.close()


def test_db_allows_direct_delete_only_after_retention_but_never_update(store):
    rec = _rec(datetime.now(timezone.utc) - timedelta(days=91))
    store.append(rec)
    con = sqlite3.connect(store.path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE runs SET outcome='sent' WHERE run_id = ?", (rec.run_id,))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE events SET detail='x' WHERE run_id = ?", (rec.run_id,))
        con.execute("DELETE FROM events WHERE run_id = ?", (rec.run_id,))
        con.execute("DELETE FROM runs WHERE run_id = ?", (rec.run_id,))
    finally:
        con.close()
    assert store.runs() == []


def test_unparsable_recorded_at_never_deleted(store):
    rec = AuditRecord(run={"recorded_at": "not-a-date", "outcome": "held", "stopped_at": "none", "reason": "r",
                           "gateway_received": 0, "attempted": 0}, events=())
    store.append(rec)
    con = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("DELETE FROM runs")
    finally:
        con.close()
    assert store.purge(datetime.now(timezone.utc)) == PurgeResult(runs=0, events=0)


# ---- 旧版 DB の移行 ----

OLD_SCHEMA = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, run_at TEXT, user_id TEXT, contract TEXT,
    destination TEXT, request_id TEXT, digest TEXT, outcome TEXT NOT NULL, stopped_at TEXT NOT NULL,
    reason TEXT NOT NULL, gateway_received INTEGER NOT NULL, attempted INTEGER NOT NULL, gateway_state TEXT,
    c1_route TEXT, c1_model TEXT, c1_revision TEXT, c1_probs TEXT, c2_decision TEXT, c2_model TEXT,
    c2_revision TEXT, c2_prob_block REAL, c2_truncated INTEGER, response_visibility TEXT,
    output_sha256 TEXT, output_chars INTEGER
);
CREATE TABLE events (
    run_id TEXT NOT NULL REFERENCES runs(run_id), seq INTEGER NOT NULL, stage TEXT NOT NULL,
    result TEXT NOT NULL, detail TEXT NOT NULL, PRIMARY KEY (run_id, seq)
);
CREATE TRIGGER runs_no_update BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
"""


def test_old_db_migrated_on_open(tmp_path: Path):
    path = tmp_path / "old.db"
    old_at = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    con = sqlite3.connect(path)
    con.executescript(OLD_SCHEMA)
    con.execute("INSERT INTO runs (run_id, recorded_at, outcome, stopped_at, reason, gateway_received, attempted) "
                "VALUES ('old1', ?, 'held', 'none', 'r', 0, 0)", (old_at,))
    con.execute("INSERT INTO events VALUES ('old1', 0, 's', 'r', 'd')")
    con.commit()
    with pytest.raises(sqlite3.IntegrityError):  # 旧トリガーは期限切れでも消させない
        con.execute("DELETE FROM runs")
    con.close()

    s = AuditStore(path)
    try:
        cols = {r[1] for r in s._conn.execute("PRAGMA table_info(runs)")}
        assert {"c1_question_version", "c2_question_version", "approver", "approver_authority",
                "approved_ref", "approved_by"} <= cols
        # 新しい列で書ける・旧行は残っている
        run(user="u_manager", inp=B_INPUT, dest="codex", audit_sink=s)
        assert len(s.runs()) == 2 and s.runs()[0]["approver"] is None
        # 置き換え後のトリガーで、期限切れの旧行だけ消せる
        assert s.purge() == PurgeResult(runs=1, events=1)
        (row,) = s.runs()
        assert row["approver"] == "u_manager"
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            s._conn.execute("DELETE FROM runs")
    finally:
        s.close()
    # 2 回目の起動でも壊れない（冪等）
    AuditStore(path).close()
