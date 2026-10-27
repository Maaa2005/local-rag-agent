"""状態遷移履歴（send_history）と request_id 付き送信の回帰テスト。"""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from common.schemas import CommitRequest, FailureKind, Message, PrepareRequest, SendPayload, SendState
from gateway.adapters import FakeAdapter
from gateway.app import create_app
from gateway.service import GatewayService, NotFound
from gateway.store import SendStore, StoreError
from integration.observe import AttemptingProbe

SECRET_IN = "本文-秘密-INPUT-7f3a"
SECRET_OUT = "出力-秘密-OUTPUT-9c1b"


def _req(rid: str, text: str = SECRET_IN) -> PrepareRequest:
    payload = SendPayload(
        contract="A-faq-format@1",
        destination="claude",
        messages=(Message(role="system", content="sys"), Message(role="user", content=text)),
        max_output_chars=100,
    )
    return PrepareRequest(request_id=rid, payload=payload, expires_at=time.time() + 60)


def _steps(store: SendStore, rid: str):
    return [(t.from_state, t.to_state, t.failure) for t in store.history(rid)]


def _commit(svc: GatewayService, rid: str, text: str = SECRET_IN):
    prep = svc.prepare(_req(rid, text))
    return svc.commit(CommitRequest(request_id=rid, expected_digest=prep.digest))


def test_success_transition_order(tmp_path):
    store = SendStore(tmp_path / "gw.db")
    svc = GatewayService(store, {"claude": FakeAdapter(reply=SECRET_OUT)})
    _commit(svc, "req-hist-0001")
    assert _steps(store, "req-hist-0001") == [
        (None, SendState.PREPARED, None),
        (SendState.PREPARED, SendState.ATTEMPTING, None),
        (SendState.ATTEMPTING, SendState.SUCCEEDED, None),
    ]
    ats = [t.at for t in store.history("req-hist-0001")]
    assert ats == sorted(ats)


@pytest.mark.parametrize("mode,kind", [("not_sent", FailureKind.not_sent),
                                       ("rejected", FailureKind.rejected),
                                       ("timeout", FailureKind.unknown)])
def test_failure_transition_records_kind(tmp_path, mode, kind):
    store = SendStore(tmp_path / "gw.db")
    svc = GatewayService(store, {"claude": FakeAdapter(mode=mode)})
    _commit(svc, "req-hist-0002")
    assert _steps(store, "req-hist-0002")[-1] == (SendState.ATTEMPTING, SendState.FAILED, kind)


def test_history_has_no_body_or_output(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter(reply=SECRET_OUT)})
    _commit(svc, "req-hist-0003")
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(send_history)")]
        rows = conn.execute("SELECT * FROM send_history").fetchall()
    finally:
        conn.close()
    assert cols == ["seq", "request_id", "from_state", "to_state", "failure", "at"]
    dump = repr(rows)
    assert SECRET_IN not in dump and SECRET_OUT not in dump and "sys" not in dump
    # API 応答にも本文・出力を載せない
    h = svc.history("req-hist-0003").model_dump_json()
    assert SECRET_IN not in h and SECRET_OUT not in h


def test_restart_unknown_is_recorded(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter()})
    prep = svc.prepare(_req("req-hist-0004"))
    svc.prepare(_req("req-hist-0005"))
    assert store.claim_attempt("req-hist-0004")  # 送信中に落ちた想定
    store2 = SendStore(db)  # 再起動
    svc2 = GatewayService(store2, {"claude": FakeAdapter()})
    assert svc2.recover_on_startup() == 1
    assert _steps(store2, "req-hist-0004") == [
        (None, SendState.PREPARED, None),
        (SendState.PREPARED, SendState.ATTEMPTING, None),
        (SendState.ATTEMPTING, SendState.FAILED, FailureKind.unknown),
    ]
    # ATTEMPTING でなかった行には追記しない
    assert _steps(store2, "req-hist-0005") == [(None, SendState.PREPARED, None)]
    # 2 回目の復旧では何も追記しない
    assert svc2.recover_on_startup() == 0
    assert len(store2.history("req-hist-0004")) == 3
    # 確定後の commit は再送せず、履歴も増えない
    svc2.commit(CommitRequest(request_id="req-hist-0004", expected_digest=prep.digest))
    assert len(store2.history("req-hist-0004")) == 3


def test_double_commit_records_once(tmp_path):
    store = SendStore(tmp_path / "gw.db")
    fake = FakeAdapter(delay=0.1)
    svc = GatewayService(store, {"claude": fake})
    prep = svc.prepare(_req("req-hist-0006"))
    svc.prepare(_req("req-hist-0006"))  # 再 prepare でも PREPARED は 1 回
    cr = CommitRequest(request_id="req-hist-0006", expected_digest=prep.digest)
    ts = [threading.Thread(target=svc.commit, args=(cr,)) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    svc.commit(cr)
    assert len(fake.received) == 1
    assert [s[1] for s in _steps(store, "req-hist-0006")] == [
        SendState.PREPARED, SendState.ATTEMPTING, SendState.SUCCEEDED]


def test_transition_and_history_are_atomic(tmp_path, monkeypatch):
    """履歴の追記に失敗したら状態遷移も巻き戻る（同一トランザクション）。"""
    db = tmp_path / "gw.db"
    store = SendStore(db)
    store.insert_prepared("req-hist-0007", "d", "{}", time.time() + 60)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TRIGGER block_hist BEFORE INSERT ON send_history "
                 "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError):
        store.claim_attempt("req-hist-0007")
    assert store.get("req-hist-0007").state == SendState.PREPARED
    with pytest.raises(StoreError):
        store.insert_prepared("req-hist-0008", "d", "{}", time.time() + 60)
    assert store.get("req-hist-0008") is None


def test_existing_db_gets_history_table(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sends (request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, "
                 "payload_json TEXT NOT NULL, expires_at REAL NOT NULL, state TEXT NOT NULL, "
                 "failure TEXT, output_text TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
    conn.execute("INSERT INTO sends VALUES ('req-old-0001','d','{}',0,'SUCCEEDED',NULL,'x',0,0)")
    conn.commit()
    conn.close()
    store = SendStore(db)
    assert store.get("req-old-0001").state == SendState.SUCCEEDED
    assert store.history("req-old-0001") == []
    SendStore(db)  # 2 回目の初期化も通る


def test_history_unknown_request(tmp_path):
    svc = GatewayService(SendStore(tmp_path / "gw.db"), {"claude": FakeAdapter()})
    with pytest.raises(NotFound):
        svc.history("req-none-0001")


def test_http_history(tmp_path):
    svc = GatewayService(SendStore(tmp_path / "gw.db"), {"claude": FakeAdapter(reply=SECRET_OUT)})
    _commit(svc, "req-hist-0009")
    with TestClient(create_app(svc)) as client:
        r = client.get("/history/req-hist-0009")
        assert r.status_code == 200
        body = r.json()
        assert [t["to_state"] for t in body["transitions"]] == ["PREPARED", "ATTEMPTING", "SUCCEEDED"]
        assert SECRET_IN not in r.text and SECRET_OUT not in r.text
        assert client.get("/history/req-none-0001").status_code == 404
        # 形式検査は /status と同じ
        for bad in ["bad id", "x" * 65, "a;b"]:
            assert client.get(f"/history/{bad}").status_code == client.get(f"/status/{bad}").status_code == 422


def test_fake_adapter_receives_request_id(tmp_path):
    store = SendStore(tmp_path / "gw.db")
    fake = FakeAdapter()
    svc = GatewayService(store, {"claude": fake})
    _commit(svc, "req-hist-0010")
    assert fake.received_ids == ["req-hist-0010"]


def test_probe_does_not_confuse_concurrent_sends(tmp_path):
    """同時送信中でも、プローブは各送信自身の request_id だけを記録する。"""
    store = SendStore(tmp_path / "gw.db")
    probe = AttemptingProbe(store)
    barrier = threading.Barrier(4)

    def on_send(payload, rid):
        barrier.wait(timeout=5)  # 全員が ATTEMPTING の状態で揃ってから観測
        probe(payload, rid)

    fake = FakeAdapter(on_send=on_send)
    svc = GatewayService(store, {"claude": fake})
    rids = [f"req-conc-000{i}" for i in range(4)]
    ts = [threading.Thread(target=_commit, args=(svc, rid, f"t{rid}")) for rid in rids]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(probe.seen) == sorted((rid, SendState.ATTEMPTING) for rid in rids)
    assert len(probe.seen) == 4
    # request_id なしの呼び出しは何も記録しない（推測で拾わない）
    probe(fake.received[0], None)
    assert len(probe.seen) == 4
