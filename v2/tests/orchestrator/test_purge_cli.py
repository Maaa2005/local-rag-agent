"""監査ストアの保持期限削除 CLI（python -m orchestrator.purge）と dry_run。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from orchestrator import purge as purge_cli
from orchestrator.audit_store import AuditRecord, AuditStore, PurgeResult


def _rec(recorded_at: datetime, n_events: int = 2) -> AuditRecord:
    return AuditRecord(run={"recorded_at": recorded_at.isoformat(), "outcome": "held", "stopped_at": "none",
                            "reason": "r", "gateway_received": 0, "attempted": 0},
                       events=tuple(("s", "r", "d") for _ in range(n_events)))


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "audit.db"
    s = AuditStore(path)
    now = datetime.now(timezone.utc)
    for d in (10, 89, 91, 120):
        s.append(_rec(now - timedelta(days=d)))
    s.close()
    return path


def _counts(path) -> tuple[int, int]:
    conn = sqlite3.connect(path)
    try:
        return (conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    finally:
        conn.close()


def test_cli_purges_and_prints_counts(db, capsys):
    assert purge_cli.main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"runs": 2, "events": 4}
    assert _counts(db) == (2, 4)  # 10 日と 89 日が残る
    assert purge_cli.main(["--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out) == {"runs": 0, "events": 0}


def test_cli_dry_run_counts_without_deleting(db, capsys):
    assert purge_cli.main(["--db", str(db), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"dry_run": True, "runs": 2, "events": 4}
    assert _counts(db) == (4, 8)


def test_cli_missing_db_does_not_create(tmp_path, capsys):
    path = tmp_path / "nope.db"
    assert purge_cli.main(["--db", str(path)]) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "DatabaseNotFound"}
    assert not path.exists()


def test_cli_requires_db_or_env(monkeypatch, capsys):
    monkeypatch.delenv("ORCHESTRATOR_AUDIT_DB", raising=False)
    assert purge_cli.main([]) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "AuditDbNotConfigured"}


def test_cli_uses_env_when_db_omitted(db, monkeypatch, capsys):
    monkeypatch.setenv("ORCHESTRATOR_AUDIT_DB", str(db))
    assert purge_cli.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"dry_run": True, "runs": 2, "events": 4}
    assert purge_cli.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"runs": 2, "events": 4}
    assert _counts(db) == (2, 4)


def test_cli_db_flag_overrides_env(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORCHESTRATOR_AUDIT_DB", str(tmp_path / "other.db"))
    assert purge_cli.main(["--db", str(db), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 2
    assert not (tmp_path / "other.db").exists()


@pytest.mark.parametrize("value", ["", "   ", "relative/audit.db"])
def test_cli_rejects_blank_or_relative_env(value, monkeypatch, capsys):
    monkeypatch.setenv("ORCHESTRATOR_AUDIT_DB", value)
    assert purge_cli.main([]) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "AuditDbNotConfigured"}


def test_cli_env_missing_file_does_not_create(tmp_path, monkeypatch, capsys):
    path = tmp_path / "nope.db"
    monkeypatch.setenv("ORCHESTRATOR_AUDIT_DB", str(path))
    assert purge_cli.main([]) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "DatabaseNotFound"}
    assert not path.exists()


def test_cli_db_error_returns_1(tmp_path, capsys):
    path = tmp_path / "broken.db"
    path.write_bytes(b"not a sqlite database" * 100)
    assert purge_cli.main(["--db", str(path)]) == 1
    assert "error" in json.loads(capsys.readouterr().err)


def test_store_dry_run_matches_real_purge(db):
    s = AuditStore(db)
    try:
        now = datetime.now(timezone.utc)
        assert s.purge(now, dry_run=True) == PurgeResult(runs=2, events=4)
        assert len(s.runs()) == 4
        assert s.purge(now) == PurgeResult(runs=2, events=4)
        assert len(s.runs()) == 2
    finally:
        s.close()
