"""Gateway 常駐中の保持期限削除の定期実行（N7）と、送信処理との同時実行の安全性。"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from common.schemas import CommitRequest, Message, PrepareRequest, SendPayload, SendState
from gateway import purge as purge_mod
from gateway.adapters import FakeAdapter
from gateway.app import create_app
from gateway.purge import (
    DEFAULT_PURGE_INTERVAL_SECONDS,
    MIN_PURGE_INTERVAL_SECONDS,
    PURGE_INTERVAL_ENV,
    parse_purge_interval,
    purge_interval_from_env,
    purge_periodically,
)
from gateway.service import GatewayService
from gateway.store import PURGED_PAYLOAD, PurgeResult, SendStore, StoreError

DAY = 86400.0
BODY = "本文-定期-BODY-9c1e"
OUT = "出力-定期-OUT-3f0a"


# ---- 間隔の解釈 ----

@pytest.mark.parametrize("raw,expected", [
    (None, DEFAULT_PURGE_INTERVAL_SECONDS),
    ("", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("   ", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("3600", 3600.0),
    (" 7200.5 ", 7200.5),
    ("abc", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("0", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("-5", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("nan", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("inf", DEFAULT_PURGE_INTERVAL_SECONDS),
    ("1", MIN_PURGE_INTERVAL_SECONDS),
    ("59.9", MIN_PURGE_INTERVAL_SECONDS),
    ("60", 60.0),
])
def test_parse_interval(raw, expected):
    assert parse_purge_interval(raw) == expected


def test_default_is_24h():
    assert DEFAULT_PURGE_INTERVAL_SECONDS == 24 * 3600


def test_invalid_interval_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="gateway"):
        parse_purge_interval("0")
    assert PURGE_INTERVAL_ENV in caplog.text


def test_interval_from_env(monkeypatch):
    monkeypatch.setenv(PURGE_INTERVAL_ENV, "120")
    assert purge_interval_from_env() == 120.0
    monkeypatch.delenv(PURGE_INTERVAL_ENV)
    assert purge_interval_from_env() == DEFAULT_PURGE_INTERVAL_SECONDS


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_create_app_rejects_nonpositive_explicit_interval(tmp_path, bad):
    svc = GatewayService(SendStore(tmp_path / "gw.db"), {})
    with pytest.raises(ValueError):
        create_app(svc, purge_interval=bad)


# ---- 定期ループ ----

def test_loop_calls_purge_repeatedly_and_stops():
    calls: list[float] = []

    async def run():
        stop = asyncio.Event()

        def purge():
            calls.append(time.monotonic())
            if len(calls) >= 3:
                stop.set()

        await asyncio.wait_for(purge_periodically(purge, 0.01, stop), timeout=5)

    asyncio.run(run())
    assert len(calls) == 3


def test_loop_continues_after_exceptions(caplog):
    calls = 0

    async def run():
        stop = asyncio.Event()

        def purge():
            nonlocal calls
            calls += 1
            if calls >= 3:
                stop.set()
            if calls == 1:
                raise StoreError("purge failed") from sqlite3.OperationalError("database is locked")
            raise RuntimeError("boom")

        await asyncio.wait_for(purge_periodically(purge, 0.01, stop), timeout=5)

    with caplog.at_level(logging.ERROR, logger="gateway"):
        asyncio.run(run())
    assert calls == 3
    assert "periodic purge failed kind=OperationalError" in caplog.text
    assert "periodic purge failed kind=RuntimeError" in caplog.text


def test_loop_stop_before_first_interval_never_purges():
    calls = 0

    async def run():
        nonlocal calls
        stop = asyncio.Event()

        def purge():
            nonlocal calls
            calls += 1

        task = asyncio.create_task(purge_periodically(purge, 3600, stop))
        await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())
    assert calls == 0


def test_loop_stop_waits_for_inflight_purge():
    started, release = threading.Event(), threading.Event()
    finished = []

    def purge():
        started.set()
        release.wait(5)
        finished.append(True)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(purge_periodically(purge, 0.01, stop))
        while not started.is_set():
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.sleep(0.05)
        assert not task.done()  # 実行中の purge を打ち切らない
        release.set()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(run())
    assert finished == [True]


# ---- アプリのライフサイクル ----

def _wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_app_runs_periodic_purge_and_stops_on_shutdown(tmp_path, monkeypatch):
    svc = GatewayService(SendStore(tmp_path / "gw.db"), {"claude": FakeAdapter(reply=OUT)})
    calls = 0
    real = svc.purge_expired

    def counting(now=None):
        nonlocal calls
        calls += 1
        return real(now)

    monkeypatch.setattr(svc, "purge_expired", counting)
    with TestClient(create_app(svc, purge_interval=0.02)) as client:
        assert _wait_until(lambda: calls >= 3)  # 起動時 1 回 + 定期 2 回以上
        assert client.get("/status/req-none-0001").status_code == 404
    after = calls
    time.sleep(0.1)
    assert calls == after  # 停止後は呼ばれない


def test_app_periodic_purge_failure_keeps_serving(tmp_path, monkeypatch, caplog):
    svc = GatewayService(SendStore(tmp_path / "gw.db"), {"claude": FakeAdapter(reply=OUT)})
    calls = 0

    def fail(now=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return PurgeResult(0, 0, 0)  # 起動時は成功させる
        raise StoreError("purge failed") from sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(svc, "purge_expired", fail)
    with caplog.at_level(logging.ERROR, logger="gateway"):
        with TestClient(create_app(svc, purge_interval=0.02)) as client:
            assert _wait_until(lambda: calls >= 3)
            assert client.get("/status/req-none-0001").status_code == 404
    assert "periodic purge failed kind=OperationalError" in caplog.text


def test_app_uses_env_interval(tmp_path, monkeypatch):
    seen = []

    async def fake_loop(purge, interval, stop):
        seen.append(interval)
        await stop.wait()

    monkeypatch.setattr("gateway.app.purge_periodically", fake_loop)
    monkeypatch.setenv(PURGE_INTERVAL_ENV, "not-a-number")
    svc = GatewayService(SendStore(tmp_path / "gw.db"), {})
    with TestClient(create_app(svc)):
        pass
    monkeypatch.setenv(PURGE_INTERVAL_ENV, "7200")
    with TestClient(create_app(svc)):
        pass
    assert seen == [DEFAULT_PURGE_INTERVAL_SECONDS, 7200.0]


# ---- 送信処理との同時実行 ----

def _req(rid: str, text: str = BODY) -> PrepareRequest:
    payload = SendPayload(contract="A-faq-format@1", destination="claude",
                          messages=(Message(role="user", content=text),), max_output_chars=50)
    return PrepareRequest(request_id=rid, payload=payload, expires_at=time.time() + 60)


def _age(db, rid: str, days: float) -> None:
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE sends SET updated_at=?, created_at=? WHERE request_id=?",
                     (time.time() - days * DAY,) * 2 + (rid,))
    conn.close()


def _hist(db, rid: str) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute(
            "SELECT to_state FROM send_history WHERE request_id=? ORDER BY seq", (rid,))]
    finally:
        conn.close()


def _make_old_terminal(svc: GatewayService, db, rid: str, days: float) -> None:
    prep = svc.prepare(_req(rid))
    svc.commit(CommitRequest(request_id=rid, expected_digest=prep.digest))
    _age(db, rid, days)


def test_purge_during_send_keeps_attempting_row_and_finish_lands(tmp_path):
    """送信中（ATTEMPTING）に purge が走っても行・本文・履歴を消さず、送信後の finish が記録される。"""
    db = tmp_path / "gw.db"
    entered, release = threading.Event(), threading.Event()

    def block(_payload, rid):
        if rid == "req-inflight":
            entered.set()
            release.wait(5)

    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT, on_send=block)})
    _make_old_terminal(svc, db, "req-old-done", 91)  # 対照: これは消える

    prep = svc.prepare(_req("req-inflight"))
    result: dict = {}
    t = threading.Thread(target=lambda: result.setdefault(
        "r", svc.commit(CommitRequest(request_id="req-inflight", expected_digest=prep.digest))))
    t.start()
    assert entered.wait(5)
    _age(db, "req-inflight", 200)  # 起点が古くても ATTEMPTING は対象外であることを確かめる
    assert store.get("req-inflight").state == SendState.ATTEMPTING

    res = svc.purge_expired(time.time())
    assert res.sends_deleted == 1 and res.bodies_purged == 0
    assert store.get("req-old-done") is None
    rec = store.get("req-inflight")
    assert rec.state == SendState.ATTEMPTING and BODY in rec.payload_json

    release.set()
    t.join(5)
    assert result["r"].state == SendState.SUCCEEDED and result["r"].output_text == OUT
    assert _hist(db, "req-inflight") == ["PREPARED", "ATTEMPTING", "SUCCEEDED"]


def test_purge_keeps_unexpired_prepared_even_if_old_and_commit_works_after(tmp_path):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)})
    prep = svc.prepare(_req("req-pending"))
    _age(db, "req-pending", 200)  # updated_at が古くても送信期限前なら対象外
    assert svc.purge_expired(time.time()) == PurgeResult(0, 0, 0)
    rec = store.get("req-pending")
    assert rec.state == SendState.PREPARED and rec.payload_json != PURGED_PAYLOAD
    r = svc.commit(CommitRequest(request_id="req-pending", expected_digest=prep.digest))
    assert r.state == SendState.SUCCEEDED


def test_concurrent_purge_and_sends_stay_consistent(tmp_path):
    """purge を繰り返しながら prepare/commit を並行に流しても、エラー・欠損・半端な状態が出ない。"""
    db = tmp_path / "gw.db"
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)})
    for i in range(5):
        _make_old_terminal(svc, db, f"req-old-{i}", 91)
    for i in range(5):
        _make_old_terminal(svc, db, f"req-mid-{i}", 31)

    errors: list[BaseException] = []
    stop = threading.Event()
    purge_runs = 0

    def purger():
        nonlocal purge_runs
        while not stop.is_set():
            try:
                svc.purge_expired(time.time())
                purge_runs += 1
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

    def sender(w: int):
        try:
            for j in range(8):
                rid = f"req-w{w}-{j}"
                prep = svc.prepare(_req(rid, f"{BODY}-{w}-{j}"))
                r = svc.commit(CommitRequest(request_id=rid, expected_digest=prep.digest))
                assert r.state == SendState.SUCCEEDED
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    pt = threading.Thread(target=purger)
    pt.start()
    senders = [threading.Thread(target=sender, args=(w,)) for w in range(4)]
    for s in senders:
        s.start()
    for s in senders:
        s.join(30)
    stop.set()
    pt.join(30)

    assert errors == []
    assert purge_runs >= 1
    for i in range(5):
        assert store.get(f"req-old-{i}") is None and _hist(db, f"req-old-{i}") == []
        rec = store.get(f"req-mid-{i}")
        assert rec.payload_json == PURGED_PAYLOAD and rec.output_text is None
        assert rec.state == SendState.SUCCEEDED
    for w in range(4):
        for j in range(8):
            rec = store.get(f"req-w{w}-{j}")
            assert rec.state == SendState.SUCCEEDED and rec.output_text == OUT and BODY in rec.payload_json
            assert _hist(db, f"req-w{w}-{j}") == ["PREPARED", "ATTEMPTING", "SUCCEEDED"]


def test_purge_waits_for_concurrent_writer_lock(tmp_path):
    """別接続が書き込みトランザクション中なら purge は待ち、解放後に完了する（途中で割り込まない）。"""
    db = tmp_path / "gw.db"
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)})
    _make_old_terminal(svc, db, "req-old-0001", 91)

    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("r", store.purge_expired(time.time())))
    t.start()
    time.sleep(0.2)
    assert t.is_alive()  # ロック解放待ち
    holder.execute("COMMIT")
    holder.close()
    t.join(10)
    assert out["r"].sends_deleted == 1


# ---- CLI ----

def test_cli_dry_run_does_not_delete(tmp_path, capsys):
    db = tmp_path / "gw.db"
    store = SendStore(db)
    svc = GatewayService(store, {"claude": FakeAdapter(reply=OUT)})
    _make_old_terminal(svc, db, "req-old-0001", 91)
    _make_old_terminal(svc, db, "req-mid-0001", 31)
    assert purge_mod.main(["--db", str(db), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"dry_run": True, "bodies_purged": 1, "sends_deleted": 1, "history_deleted": 3}
    assert BODY not in out and "req-" not in out
    assert store.get("req-old-0001") is not None and BODY in store.get("req-mid-0001").payload_json
    assert purge_mod.main(["--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out) == {"bodies_purged": 1, "sends_deleted": 1, "history_deleted": 3}
    assert store.get("req-old-0001") is None


def test_cli_missing_db_does_not_create(tmp_path, capsys):
    db = tmp_path / "missing.db"
    assert purge_mod.main(["--db", str(db)]) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "DatabaseNotFound"}
    assert not db.exists()
