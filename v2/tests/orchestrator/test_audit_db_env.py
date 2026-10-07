"""監査 DB の置き場所（環境変数 ORCHESTRATOR_AUDIT_DB）。既定パスは持たない。"""
from __future__ import annotations

import sqlite3

import pytest

from orchestrator.audit_store import (
    AUDIT_DB_ENV,
    AuditDbNotConfigured,
    AuditStore,
    audit_db_path_from_env,
    open_audit_store_from_env,
)


def test_env_name():
    assert AUDIT_DB_ENV == "ORCHESTRATOR_AUDIT_DB"


def test_unset_raises(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_AUDIT_DB", raising=False)
    with pytest.raises(AuditDbNotConfigured):
        audit_db_path_from_env()
    with pytest.raises(AuditDbNotConfigured):
        open_audit_store_from_env()


@pytest.mark.parametrize("value", ["", "  ", "audit.db", "./data/audit.db"])
def test_blank_or_relative_raises(value, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AuditDbNotConfigured):
        open_audit_store_from_env({"ORCHESTRATOR_AUDIT_DB": value})
    assert list(tmp_path.iterdir()) == []  # どこにも DB を作らない


def test_reads_os_environ(tmp_path, monkeypatch):
    path = tmp_path / "audit.db"
    monkeypatch.setenv("ORCHESTRATOR_AUDIT_DB", str(path))
    assert audit_db_path_from_env() == str(path)


def test_opens_store_at_env_path(tmp_path):
    path = tmp_path / "audit.db"
    store = open_audit_store_from_env({"ORCHESTRATOR_AUDIT_DB": f" {path} "})
    try:
        assert isinstance(store, AuditStore)
        assert store.path == str(path)
        assert store.runs() == []
    finally:
        store.close()
    assert path.is_file()


def test_missing_parent_dir_fails_without_creating_it(tmp_path):
    path = tmp_path / "not-mounted" / "audit.db"
    with pytest.raises(sqlite3.OperationalError):
        open_audit_store_from_env({"ORCHESTRATOR_AUDIT_DB": str(path)})
    assert not path.parent.exists()
