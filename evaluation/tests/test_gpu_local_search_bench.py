"""search_bench（Issue #14 E4 の検索だけを繰り返す道具）の試験。Qdrant・埋め込み・GPU は使わない。"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from qdrant_client.http.exceptions import ResponseHandlingException

from evaluation.gpu_local import search_bench as sb

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))  # search_bench の子と同じく backend の app を使う


def test_load_questions_reads_only_id_question_level():
    qs = sb.load_questions(REPO)
    assert len(qs) == 62  # Q 50 問 + N 12 問
    assert all(set(q) == {"id", "question", "level"} and q["level"] in (1, 2, 3) for q in qs)


def test_run_queries_cycles_and_classifies_failures():
    calls = []

    async def retrieve(question, level):
        calls.append((question, level))
        if len(calls) == 2:
            raise ResponseHandlingException(httpx.ReadTimeout("slow"))
        if len(calls) == 3:
            raise ResponseHandlingException(httpx.ConnectError("refused"))
        return []

    from app.services.qdrant import is_timeout

    qs = [{"id": "a", "question": "qa", "level": 1}, {"id": "b", "question": "qb", "level": 2}]
    rows = asyncio.run(sb.run_queries(qs, 5, retrieve, is_timeout))
    assert [r["id"] for r in rows] == ["a", "b", "a", "b", "a"]
    assert [(r["ok"], r["timeout"]) for r in rows] == [(True, False), (False, True), (False, False),
                                                        (True, False), (True, False)]
    assert rows[1]["error"] == "ResponseHandlingException"
    # 記録に問の文を書かない
    assert all("question" not in r and "qa" not in json.dumps(r) for r in rows)


def _rows(times, fail_at=()):
    return [{"i": i, "id": "x", "seconds": t, "ok": i not in fail_at, "error": "E" if i in fail_at else None,
             "timeout": i in fail_at} for i, t in enumerate(times)]


def test_summarize_counts_failures_as_infinite():
    s = sb.summarize(_rows([3.0] + [0.1] * 99, fail_at=(50,)), retry_count=2)
    assert (s["n"], s["failed"], s["timeouts"], s["retries"]) == (100, 1, 1, 2)
    assert s["max"] == float("inf") and s["first"] == 3.0 and s["p50"] == 0.1


def test_e4_lines_pass_and_fail_at_the_lines():
    ok = sb.summarize(_rows([0.2] * 297 + [1.0, 3.0, 3.0]), 0)
    assert ok["p95"] <= 1.0 and ok["p99"] <= 3.0
    assert sb.e4_line_checks(ok, {"valid": True})["verdict"] == "通る"
    slow = sb.summarize(_rows([0.2] * 280 + [1.5] * 20), 0)
    got = sb.e4_line_checks(slow, {"valid": True})
    assert got["lines"]["L2_p95"] is False and got["verdict"] == "落ちる"
    failed = sb.summarize(_rows([0.2] * 300, fail_at=(10,)), 1)
    assert sb.e4_line_checks(failed, {"valid": True})["lines"]["L1_no_failure"] is False
    short = sb.summarize(_rows([0.2] * 299), 0)
    assert sb.e4_line_checks(short, {"valid": True})["lines"]["L1_no_failure"] is False


@pytest.mark.parametrize("pressure", [None, {"valid": False}])
def test_e4_invalid_pressure_gives_no_verdict(pressure):
    s = sb.summarize(_rows([0.2] * 300), 0)
    assert sb.e4_line_checks(s, pressure)["verdict"] == "無効"


def test_pressure_window():
    rows = [{"t": float(t), "free_mib": f, "action": a} for t, f, a in
            [(0, 3000, "grow 8"), (1, 1100, "keep"), (2, 1000, "keep"), (3, 950, "keep"), (9, 5000, "keep")]]
    w = sb.pressure_window(rows, 1, 3, 1024, 256)
    assert (w["n"], w["free_median"], w["valid"]) == (3, 1000, True)
    rows[2]["action"] = "floor"
    assert sb.pressure_window(rows, 1, 3, 1024, 256)["valid"] is False
    assert sb.pressure_window(rows, 20, 30, 1024, 256) == {"n": 0, "valid": False}


def test_parse_args_defaults_and_requirements():
    a = sb.parse_args(["--work", "w", "--out", "o"])
    assert (a.n, a.qdrant_timeout_s, a.qdrant_retries, a.pressure_cmd) == (300, 0, 0, "")
    assert sb.parse_args(["--child", "--rows", "r"]).child is True
    with pytest.raises(SystemExit):
        sb.parse_args(["--n", "3"])


def test_lines_are_the_research_values():
    assert (sb.E4_N, sb.E4_P95, sb.E4_P99) == (300, 1.0, 3.0)


# ─── 子が終わること（aiosqlite 0.20.0 の接続は非 daemon スレッド。閉じないとプロセスが終わらない） ───

_CHILD_DRIVER = """
import sys, types
sys.path.insert(0, {backend!r})
stub = types.ModuleType("app.services.retriever")  # 埋め込み・Qdrant は使わない
async def retrieve(question, level):
    return []
stub.retrieve = retrieve
sys.modules["app.services.retriever"] = stub
from evaluation.gpu_local import search_bench as sb
if {boom_close_client!r}:
    from app.services import qdrant as qmod
    async def boom():
        raise RuntimeError("close_client failed")
    qmod.close_client = boom
sys.exit(sb.child_main(sb.parse_args(["--child", "--repo", {repo!r}, "--n", "3", "--rows", {rows!r}])))
"""


def _run_child(tmp_path, repo, boom_close_client=False):
    """本物の app.database（本物の aiosqlite）で子を動かす。終わらなければ親が待ち続ける事になるので落とす。"""
    rows = tmp_path / "rows.jsonl"
    code = _CHILD_DRIVER.format(backend=str(REPO / "backend"), repo=str(repo), rows=str(rows),
                                boom_close_client=boom_close_client)
    env = dict(os.environ, DATA_DIR=str(tmp_path / "data"))
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("子のプロセスが 60 秒で終わらない（SQLite の接続が閉じられていない）")
    return p, rows


def test_child_exits_after_queries(tmp_path):
    p, rows = _run_child(tmp_path, REPO)
    assert p.returncode == 0, p.stderr[-2000:]
    got = sb._read_jsonl(rows)
    assert [r["i"] for r in got if "i" in r] == [0, 1, 2]
    assert got[-1] == {"retry_count": 0}


def test_child_exits_when_questions_cannot_be_read(tmp_path):
    p, rows = _run_child(tmp_path, tmp_path / "no_such_repo")  # 測る途中の例外でも閉じる
    assert p.returncode != 0
    assert not rows.exists()


def test_child_exits_when_closing_qdrant_client_fails(tmp_path):
    p, _ = _run_child(tmp_path, REPO, boom_close_client=True)  # Qdrant 側を閉じ損ねても SQLite は閉じる
    assert p.returncode != 0
    assert "close_client failed" in p.stderr


# ─── 親の流れ（Qdrant・索引・絞る道具・子を偽物にして、起動の環境変数と判定までを通す） ───

class _FakeProc:
    def __init__(self, rc=None):
        self.rc = rc

    def poll(self):
        return self.rc

    def terminate(self):
        self.rc = 0 if self.rc is None else self.rc

    def wait(self, timeout=None):
        return self.rc


class _Clock:
    """測定の区間を 1000〜1100 秒にする時計。子を動かす間に進む。"""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now


def _fake_parent(monkeypatch, tmp_path, n_rows=300, pressure_until=1100, pressure_rc=None):
    """parent_main を偽物で動かす用意。launch に渡った {プログラム名: env} と結果の置き場を返す。"""
    from evaluation.gpu_local import step2_local

    launched = {}
    clock = _Clock()
    out = tmp_path / "out"

    def launch(argv, log_path, env=None):
        launched[Path(argv[0]).stem] = env
        return _FakeProc(pressure_rc if argv[0] == "pressure" else None)

    def run(argv, **kw):
        if "--child" not in argv:  # 索引
            (out / "index_work").mkdir(parents=True)
            (out / "index_work" / "index.json").write_text(json.dumps({"env": {}}))
            return subprocess.CompletedProcess(argv, 0, "", "")
        with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
            for r in _rows([0.2] * n_rows):
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"retry_count": 0}) + "\n")
        with open(out / "pressure.jsonl", "w", encoding="utf-8") as f:
            for t in range(1000, pressure_until + 1):
                f.write(json.dumps({"t": float(t), "free_mib": 1024, "action": "keep"}) + "\n")
        clock.now = 1100.0
        kw["stdout"].close()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(step2_local, "launch", launch)
    monkeypatch.setattr(step2_local, "kill", lambda p: None)
    monkeypatch.setattr(step2_local, "wait_up", lambda *a: True)
    monkeypatch.setattr(sb, "time", clock)
    monkeypatch.setattr(sb.subprocess, "run", run)
    return launched, out


def _parent_args(tmp_path, out, *extra):
    return sb.parse_args(["--work", str(tmp_path / "work"), "--out", str(out), "--pressure-settle-s", "0",
                          "--pressure-cmd", "pressure --log {log}", *extra])


def _summary(out):
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("use_existing_storage", [False, True])
def test_qdrant_listens_only_on_loopback(monkeypatch, tmp_path, use_existing_storage):
    monkeypatch.setenv("QDRANT__SERVICE__HOST", "0.0.0.0")  # 外の環境変数でも広がらない
    launched, out = _fake_parent(monkeypatch, tmp_path)
    existing = tmp_path / "existing_storage"
    extra = ["--qdrant-storage", str(existing)] if use_existing_storage else []
    assert sb.parent_main(_parent_args(tmp_path, out, *extra)) == 0
    env = launched["qdrant"]
    assert env["QDRANT__SERVICE__HOST"] == "127.0.0.1"
    assert env["QDRANT__STORAGE__STORAGE_PATH"] == str(existing if use_existing_storage else out / "qdrant_storage")


def test_parent_full_coverage_gives_a_verdict(monkeypatch, tmp_path):
    _, out = _fake_parent(monkeypatch, tmp_path)
    assert sb.parent_main(_parent_args(tmp_path, out)) == 0
    got = _summary(out)
    assert got["pressure"]["valid"] is True and got["lines"]["verdict"] == "通る"


def test_parent_pressure_tool_dying_midway_is_invalid(monkeypatch, tmp_path):
    # 前半の記録の中央値は線の中。途中で落ちた回を有効にしない
    _, out = _fake_parent(monkeypatch, tmp_path, pressure_until=1040, pressure_rc=1)
    assert sb.parent_main(_parent_args(tmp_path, out)) == 0
    got = _summary(out)
    assert got["pressure"]["free_median"] == 1024
    assert got["pressure"]["valid"] is False and got["lines"]["verdict"] == "無効"


def test_parent_tool_exit_just_before_the_end_is_invalid(monkeypatch, tmp_path):
    # 記録は終わりまであるが、測り終えた時に道具が居ない
    _, out = _fake_parent(monkeypatch, tmp_path, pressure_rc=1)
    assert sb.parent_main(_parent_args(tmp_path, out)) == 0
    got = _summary(out)
    assert got["pressure"]["valid"] is False and got["lines"]["verdict"] == "無効"


@pytest.mark.parametrize("n", [299, 301])
def test_parent_n_other_than_300_does_not_pass(monkeypatch, tmp_path, n):
    _, out = _fake_parent(monkeypatch, tmp_path, n_rows=n)
    assert sb.parent_main(_parent_args(tmp_path, out, "--n", str(n))) == 0
    got = _summary(out)
    assert got["numbers"]["n"] == n and got["pressure"]["valid"] is True
    assert got["lines"]["lines"]["L1_no_failure"] is False and got["lines"]["verdict"] == "落ちる"


def _window_rows(ts):
    return [{"t": float(t), "free_mib": 1024, "action": "keep"} for t in ts]


@pytest.mark.parametrize("ts, valid", [
    (range(0, 101), True),
    (range(0, 41), False),  # 40 秒で記録が止まる（途中で落ちた）
    (range(60, 101), False),  # 60 秒まで記録が無い（遅れて始まった）
    ([*range(0, 41), *range(60, 101)], False),  # 途中が 20 秒抜ける
    ([*range(0, 41), *range(50, 101)], True),  # 10 秒の抜けまでは有効（ちょうど）
])
def test_pressure_window_requires_coverage(ts, valid):
    w = sb.pressure_window(_window_rows(ts), 0, 100, 1024, 256)
    assert w["free_median"] == 1024
    assert w["valid"] is valid


def test_pressure_window_tool_gone_at_end_is_invalid():
    w = sb.pressure_window(_window_rows(range(0, 101)), 0, 100, 1024, 256, tool_alive=False)
    assert w["valid"] is False and w["tool_alive"] is False


def test_e4_line_uses_fixed_300():
    s = sb.summarize(_rows([0.2] * 299), 0)
    got = sb.e4_line_checks(s, {"valid": True})
    assert got["lines"]["L1_no_failure"] is False and got["verdict"] == "落ちる"
    with pytest.raises(TypeError):  # 変えられる値を渡す道を残さない
        sb.e4_line_checks(s, {"valid": True}, 299)


# ─── docker-compose.yml の backend に時間切れの設定が届くこと ───

@pytest.mark.parametrize("name", ["QDRANT_TIMEOUT_S", "QDRANT_SEARCH_RETRIES"])
def test_compose_forwards_qdrant_settings_with_same_default(name):
    from app.config import Settings

    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    env = dict(e.split("=", 1) for e in compose["services"]["backend"]["environment"])
    default = str(Settings.model_fields[name.lower()].default)
    assert default == "0"  # 既定は今までと同じ（時間切れは qdrant-client の既定・送り直しなし）
    assert env[name] == "${%s:-%s}" % (name, default)
