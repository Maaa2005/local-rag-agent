"""Gateway の単一ワーカー前提の強制（同じ DB を使う 2 つ目の起動を失敗させる）。"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common.schemas import SendState
from gateway.app import GatewayAlreadyRunning, acquire_single_worker_lock, create_app, lock_path_for
from gateway.service import GatewayService
from gateway.store import SendStore

V2 = Path(__file__).resolve().parents[2]


def _svc(db: Path) -> GatewayService:
    return GatewayService(SendStore(db), {})


def test_second_app_on_same_db_fails_to_start(tmp_path):
    db = tmp_path / "gw.db"
    with TestClient(create_app(_svc(db))):
        with pytest.raises(GatewayAlreadyRunning, match="単一ワーカー"):
            with TestClient(create_app(_svc(db))):
                pass  # pragma: no cover


def test_second_start_does_not_recover_first_ones_attempting(tmp_path):
    """2 つ目は recover_on_startup より前に止まる（1 つ目の送信中の行を unknown にしない）。"""
    db = tmp_path / "gw.db"
    with TestClient(create_app(_svc(db))):
        store = SendStore(db)
        assert store.insert_prepared("r1", "d" * 64, "{}", 1e12)
        assert store.claim_attempt("r1")
        with pytest.raises(GatewayAlreadyRunning):
            with TestClient(create_app(_svc(db))):
                pass  # pragma: no cover
        assert store.get("r1").state is SendState.ATTEMPTING


def test_different_dbs_both_start(tmp_path):
    with TestClient(create_app(_svc(tmp_path / "a.db"))) as a, \
            TestClient(create_app(_svc(tmp_path / "b.db"))) as b:
        assert a.get("/status/x").status_code == 404
        assert b.get("/status/x").status_code == 404


def test_lock_reacquired_after_first_stops(tmp_path):
    db = tmp_path / "gw.db"
    with TestClient(create_app(_svc(db))):
        pass
    with TestClient(create_app(_svc(db))) as c:
        assert c.get("/status/x").status_code == 404
    assert Path(lock_path_for(str(db))).exists()  # ロックファイルは消さない


def test_lock_conflicts_across_processes(tmp_path):
    """別プロセスが保持中なら取れず、そのプロセスが終われば取れる。"""
    db = str(tmp_path / "gw.db")
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys
            from gateway.app import acquire_single_worker_lock
            acquire_single_worker_lock({db!r})
            print("locked", flush=True)
            sys.stdin.read()
        """)],
        cwd=V2, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(GatewayAlreadyRunning):
            acquire_single_worker_lock(db)
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
    fd = acquire_single_worker_lock(db)
    os.close(fd)
