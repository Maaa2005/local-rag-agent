"""search_bench（Issue #14 E4 の検索だけを繰り返す道具）の試験。Qdrant・埋め込み・GPU は使わない。"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
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
    assert sb.e4_line_checks(ok, {"valid": True}, 300)["verdict"] == "通る"
    slow = sb.summarize(_rows([0.2] * 280 + [1.5] * 20), 0)
    got = sb.e4_line_checks(slow, {"valid": True}, 300)
    assert got["lines"]["L2_p95"] is False and got["verdict"] == "落ちる"
    failed = sb.summarize(_rows([0.2] * 300, fail_at=(10,)), 1)
    assert sb.e4_line_checks(failed, {"valid": True}, 300)["lines"]["L1_no_failure"] is False
    short = sb.summarize(_rows([0.2] * 299), 0)
    assert sb.e4_line_checks(short, {"valid": True}, 300)["lines"]["L1_no_failure"] is False


@pytest.mark.parametrize("pressure", [None, {"valid": False}])
def test_e4_invalid_pressure_gives_no_verdict(pressure):
    s = sb.summarize(_rows([0.2] * 300), 0)
    assert sb.e4_line_checks(s, pressure, 300)["verdict"] == "無効"


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
    assert (sb.E4_P95, sb.E4_P99) == (1.0, 3.0)
