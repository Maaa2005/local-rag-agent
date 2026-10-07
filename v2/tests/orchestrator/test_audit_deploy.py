"""監査 DB の配置（compose の ORCHESTRATOR_AUDIT_DB・専用 volume）と purge の cron 定義例の静的検査。"""
from __future__ import annotations

import posixpath
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((DEPLOY / "compose.yml").read_text(encoding="utf-8"))


def _mounts(svc: dict) -> dict[str, str]:
    out = {}
    for v in svc.get("volumes", []):
        src, dst = str(v).split(":")[:2]
        out[src] = dst
    return out


def test_internal_app_audit_db_on_dedicated_volume(compose):
    app = compose["services"]["internal-app"]
    path = app["environment"]["ORCHESTRATOR_AUDIT_DB"]
    assert posixpath.isabs(path)
    assert "audit-data" in compose["volumes"]
    mount = _mounts(app)["audit-data"]
    assert posixpath.dirname(path) == mount
    assert all(not str(v).endswith(":ro") for v in app["volumes"] if str(v).startswith("audit-data:"))


def test_audit_volume_only_on_internal_app(compose):
    for name, svc in compose["services"].items():
        if name != "internal-app":
            assert "audit-data" not in _mounts(svc), name


def test_cron_runs_purge_daily_in_internal_app_without_db_flag():
    lines = [ln for ln in (DEPLOY / "audit-purge.cron").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#") and "=" not in ln.split()[0]]
    assert len(lines) == 1
    fields = lines[0].split(None, 5)
    minute, hour, dom, month, dow, cmd = fields
    assert minute.isdigit() and hour.isdigit() and (dom, month, dow) == ("*", "*", "*")
    assert "exec -T internal-app python -m orchestrator.purge" in cmd
    assert "--db" not in cmd  # 場所は compose の ORCHESTRATOR_AUDIT_DB だけで決める
