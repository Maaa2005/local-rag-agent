"""保持期限（N7: 本文 30 日・メタデータ 90 日）の削除処理。"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from common.schemas import CommitRequest, FailureKind, Message, PrepareRequest, SendPayload, SendState
from gateway import purge as purge_cli
from gateway.adapters import FakeAdapter
from gateway.app import create_app
from gateway.service import Expired, GatewayService, NotFound
from gateway.store import (
    BODY_RETENTION_DAYS,
    META_RETENTION_DAYS,
    PURGED_PAYLOAD,
    PurgeResult,
    SendStore,
    StoreError,
)

DAY = 86400.0
NOW = 2_000_000_000.0
BODY = "本文-保持-BODY-41d2"
OUT = "出力-保持-OUT-77ae"


def _set_times(db, rid: str, updated_at: float, expires_at: float | None = None) -> None:
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE sends SET updated_at=?, created_at=? WHERE request_id=?", (updated_at, updated_at, rid))
        if expires_at is not None:
            conn.execute("UPDATE sends SET expires_at=? WHERE request_id=?", (expires_at, rid))
    conn.close()


def _make(store: SendStore, rid: str, state: SendState, failure: FailureKind | None = None,
          age_days: float = 0.0, expires_in: float = 60.0) -> None:
    """state の行を作り、updated_at を NOW - age_days にする。"""
    store.insert_prepared(rid, "d-" + rid, json.dumps({"body": BODY}, ensure_ascii=False), NOW + expires_in)
    if state != SendState.PREPARED:
        assert store.claim_attempt(rid)
    if state in (SendState.SUCCEEDED, SendState.FAILED):
        assert store.finish(rid, state, failure, OUT if state == SendState.SUCCEEDED else None) == 1
    _set_times(store._path, rid, NOW - age_days * DAY,
               expires_at=None if state != SendState.PREPARED else NOW - age_days * DAY + expires_in)


def _raw(db, rid: str):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT payload_json, output_text, digest, state, failure FROM sends "
                            "WHERE request_id=?", (rid,)).fetchone()
    finally:
        conn.close()


def _hist_count(db, rid: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM send_history WHERE request_id=?", (rid,)).fetchone()[0]
    finally:
        conn.close()


def test_constants():
    assert BODY_RETENTION_DAYS == 30
    assert META_RETENTION_DAYS == 90


@pytest.mark.parametrize("state,failure", [(SendState.SUCCEEDED, None),
                                           (SendState.FAILED, FailureKind.unknown),
                                           (SendState.FAILED, FailureKind.not_sent)])
@pytest.mark.parametrize("age,purged", [(29, False), (29.999, False), (30, True), (31, True)])
def test_body_boundary(tmp_path, state, failure, age, purged):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    _make(store, "r-1", state, failure, age_days=age)
    res = store.purge_expired(NOW)
    assert res == PurgeResult(bodies_purged=1 if purged else 0, sends_deleted=0, history_deleted=0)
    payload, out, digest, st, fl = _raw(db, "r-1")
    if purged:
        assert payload == PURGED_PAYLOAD and out is None
    else:
        assert BODY in payload
        assert out == (OUT if state == SendState.SUCCEEDED else None)
    # メタデータは残る
    assert digest == "d-r-1" and st == state.value and fl == (failure.value if failure else None)
    assert _hist_count(db, "r-1") == 3


@pytest.mark.parametrize("age,deleted", [(89, False), (89.999, False), (90, True), (91, True)])
def test_meta_boundary(tmp_path, age, deleted):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    _make(store, "r-1", SendState.SUCCEEDED, age_days=age)
    res = store.purge_expired(NOW)
    if deleted:
        assert res == PurgeResult(bodies_purged=0, sends_deleted=1, history_deleted=3)
        assert store.get("r-1") is None
        assert store.history("r-1") == []
    else:
        assert res == PurgeResult(bodies_purged=1, sends_deleted=0, history_deleted=0)
        assert store.get("r-1") is not None
        assert _hist_count(db, "r-1") == 3


def test_unterminated_not_touched(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    # 期限内の PREPARED（作成から 200 日経過していても expires_at が未来なら消さない）
    _make(store, "prep-live", SendState.PREPARED, age_days=200, expires_in=200 * DAY + 60)
    _make(store, "attempting", SendState.ATTEMPTING, age_days=200)
    res = store.purge_expired(NOW)
    assert res == PurgeResult(0, 0, 0)
    for rid in ("prep-live", "attempting"):
        payload, *_ = _raw(db, rid)
        assert BODY in payload
    assert _hist_count(db, "attempting") == 2


def test_expired_prepared_uses_expiry_as_start(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    # 期限切れから 29 日 / 30 日 / 90 日
    _make(store, "p29", SendState.PREPARED, age_days=29, expires_in=0)
    _make(store, "p30", SendState.PREPARED, age_days=30, expires_in=0)
    _make(store, "p90", SendState.PREPARED, age_days=90, expires_in=0)
    # 作成は 100 日前だが期限が 20 日前 → 起点は期限なので何もしない
    _make(store, "p-late-exp", SendState.PREPARED, age_days=100, expires_in=80 * DAY)
    res = store.purge_expired(NOW)
    assert res == PurgeResult(bodies_purged=1, sends_deleted=1, history_deleted=1)
    assert BODY in _raw(db, "p29")[0]
    assert _raw(db, "p30")[0] == PURGED_PAYLOAD and _raw(db, "p30")[3] == SendState.PREPARED.value
    assert store.get("p90") is None
    assert BODY in _raw(db, "p-late-exp")[0]


def test_counts_mixed_and_idempotent(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    _make(store, "s-new", SendState.SUCCEEDED, age_days=1)
    _make(store, "s-body1", SendState.SUCCEEDED, age_days=40)
    _make(store, "f-body2", SendState.FAILED, FailureKind.rejected, age_days=60)
    _make(store, "s-del1", SendState.SUCCEEDED, age_days=100)
    _make(store, "f-del2", SendState.FAILED, FailureKind.unknown, age_days=95)
    _make(store, "att", SendState.ATTEMPTING, age_days=100)
    res = store.purge_expired(NOW)
    assert res == PurgeResult(bodies_purged=2, sends_deleted=2, history_deleted=6)
    # 2 回目は既に消した行を数えない
    assert store.purge_expired(NOW) == PurgeResult(0, 0, 0)
    assert _hist_count(db, "s-new") == 3 and _hist_count(db, "att") == 2


def test_recovered_unknown_starts_from_recovery_time(tmp_path):
    """起動時に ATTEMPTING→unknown になった行は、その時刻から数える（早く消さない）。"""
    db = tmp_path / "gw.db"
    store = SendStore(db)
    _make(store, "att", SendState.ATTEMPTING, age_days=200)
    svc = GatewayService(store, {}, clock=lambda: NOW)
    svc.recover_on_startup()  # updated_at は実時間（time.time()）で記録される
    res = store.purge_expired(time.time())
    assert res == PurgeResult(0, 0, 0)
    payload, _, _, st, fl = _raw(db, "att")
    assert BODY in payload and st == "FAILED" and fl == "unknown"


def test_purge_rollback_on_error(tmp_path, monkeypatch):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    _make(store, "old", SendState.SUCCEEDED, age_days=100)
    _make(store, "mid", SendState.SUCCEEDED, age_days=40)
    real = store._connect

    class Boom:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *a):
            if sql.startswith("UPDATE sends SET payload_json"):
                raise sqlite3.OperationalError("boom")
            return self.conn.execute(sql, *a)

        def close(self):
            self.conn.close()

    monkeypatch.setattr(store, "_connect", lambda: Boom(real()))
    with pytest.raises(StoreError):
        store.purge_expired(NOW)
    monkeypatch.setattr(store, "_connect", real)
    # 途中の DELETE も巻き戻っている
    assert store.get("old") is not None and _hist_count(db, "old") == 3
    assert BODY in _raw(db, "mid")[0]


# ---- 照会・サービス・起動時・CLI ----

def _req(rid: str, now: float) -> PrepareRequest:
    payload = SendPayload(
        contract="A-faq-format@1",
        destination="claude",
        messages=(Message(role="system", content="sys"), Message(role="user", content=BODY)),
        max_output_chars=100,
    )
    return PrepareRequest(request_id=rid, payload=payload, expires_at=now + 60)


def test_queries_after_purge(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    clock = [time.time()]
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)}, clock=lambda: clock[0])
    prep_body = svc.prepare(_req("req-ret-0001", clock[0]))
    svc.commit(CommitRequest(request_id="req-ret-0001", expected_digest=prep_body.digest))
    prep_meta = svc.prepare(_req("req-ret-0002", clock[0]))
    svc.commit(CommitRequest(request_id="req-ret-0002", expected_digest=prep_meta.digest))
    _set_times(db, "req-ret-0001", clock[0] - 31 * DAY)
    _set_times(db, "req-ret-0002", clock[0] - 91 * DAY)

    res = svc.purge_expired(now=clock[0])
    assert res == PurgeResult(bodies_purged=1, sends_deleted=1, history_deleted=3)

    # 本文を消した行: status / history / 同一内容の prepare / commit が壊れない
    st = svc.status("req-ret-0001")
    assert st.state == SendState.SUCCEEDED and st.output_text is None and st.digest == prep_body.digest
    assert [t.to_state for t in svc.history("req-ret-0001").transitions] == [
        SendState.PREPARED, SendState.ATTEMPTING, SendState.SUCCEEDED]
    again = svc.prepare(_req("req-ret-0001", clock[0]))
    assert again.state == SendState.SUCCEEDED and again.digest == prep_body.digest
    c = svc.commit(CommitRequest(request_id="req-ret-0001", expected_digest=prep_body.digest))
    assert c.state == SendState.SUCCEEDED and c.output_text is None

    with pytest.raises(NotFound):
        svc.status("req-ret-0002")
    with pytest.raises(NotFound):
        svc.history("req-ret-0002")

    client = TestClient(create_app(svc))
    r = client.get("/status/req-ret-0001")
    assert r.status_code == 200 and r.json()["output_text"] is None
    assert BODY not in r.text and OUT not in r.text
    r = client.get("/history/req-ret-0001")
    assert r.status_code == 200 and len(r.json()["transitions"]) == 3
    assert client.get("/status/req-ret-0002").status_code == 404
    assert client.get("/history/req-ret-0002").status_code == 404


def test_commit_refuses_purged_prepared(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    clock = [time.time()]
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)}, clock=lambda: clock[0])
    prep = svc.prepare(_req("req-ret-0003", clock[0]))
    # 期限切れから 30 日後に本文が消える。時計がずれて期限内に見えても送らない
    assert store.purge_expired(clock[0] + 60 + 30 * DAY).bodies_purged == 1
    with pytest.raises(Expired):
        svc.commit(CommitRequest(request_id="req-ret-0003", expected_digest=prep.digest))
    assert svc.status("req-ret-0003").state == SendState.PREPARED


def test_startup_runs_purge_after_recover(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    now = time.time()
    _make_rid = "old"
    store.insert_prepared(_make_rid, "d", "{}", now + 60)
    store.claim_attempt(_make_rid)
    store.finish(_make_rid, SendState.SUCCEEDED, None, OUT)
    _set_times(db, _make_rid, now - 91 * DAY)
    store.insert_prepared("att", "d2", json.dumps({"body": BODY}, ensure_ascii=False), now + 60)
    store.claim_attempt("att")
    _set_times(db, "att", now - 200 * DAY)

    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)})
    with TestClient(create_app(svc)):
        pass
    assert store.get("old") is None
    rec = store.get("att")  # 先に unknown 化され、起点が起動時刻になるので残る
    assert rec is not None and rec.state == SendState.FAILED and rec.failure == FailureKind.unknown
    assert BODY in rec.payload_json


def test_startup_continues_when_purge_fails(tmp_path, monkeypatch):
    store = SendStore(tmp_path / "gw.db")
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)})

    def fail(now=None):
        raise StoreError("purge failed")

    monkeypatch.setattr(svc, "purge_expired", fail)
    with TestClient(create_app(svc)) as client:
        assert client.get("/status/req-none-0001").status_code == 404


def test_existing_db_with_not_null_payload(tmp_path):
    """既存 DB（payload_json NOT NULL のまま）でスキーマ変更なしに動く。"""
    db = tmp_path / "gw.db"
    conn = sqlite3.connect(db)
    with conn:
        conn.execute(
            "CREATE TABLE sends (request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, expires_at REAL NOT NULL, state TEXT NOT NULL, "
            "failure TEXT, output_text TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute("INSERT INTO sends VALUES ('legacy','d',?,?, 'SUCCEEDED', NULL, ?, ?, ?)",
                     (BODY, NOW - 50 * DAY, OUT, NOW - 50 * DAY, NOW - 40 * DAY))
    conn.close()
    store = SendStore(db)
    assert store.purge_expired(NOW).bodies_purged == 1
    rec = store.get("legacy")
    assert rec.payload_json == PURGED_PAYLOAD and rec.output_text is None and rec.state == SendState.SUCCEEDED


def test_cli(tmp_path, capsys):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    now = time.time()
    store.insert_prepared("old", "d", json.dumps({"body": BODY}, ensure_ascii=False), now + 60)
    store.claim_attempt("old")
    store.finish("old", SendState.SUCCEEDED, None, OUT)
    _set_times(db, "old", now - 31 * DAY)
    assert purge_cli.main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"bodies_purged": 1, "sends_deleted": 0, "history_deleted": 0}
    assert BODY not in out and "old" not in out
