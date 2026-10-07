"""evalharness.run_v2_e2e: 実システム経路の E2E 評価ハーネス。送信アダプタは偽物のみ。"""
from __future__ import annotations

import json

import pytest

from evalharness.run_v2_e2e import main, run_case, summarize_e2e
from evalharness.run_v2_eval import load_cases
from judge.rules import RuleC1Router, RuleC2Gate

CASES = {c["id"]: c for c in load_cases()}


def _run(cid, tmp_path, **kw):
    return run_case(CASES[cid], RuleC1Router(), RuleC2Gate(), workdir=tmp_path, **kw)


def test_v072_rejected_at_eligibility_without_adapter_call(tmp_path):
    r = _run("V072", tmp_path)
    assert (r["outcome"], r["stop"]) == ("rejected", "eligibility")
    assert r["adapter_calls"] == 0 and r["sent_destinations"] == []
    assert r["gateway_received"] is False and r["attempted"] is False
    assert r["driver"] == "generic"  # source 指定は run_contract_a が受け付けない
    assert "閲覧権限" in r["reason"]
    assert r["missend"] is False


@pytest.mark.parametrize("cid,dest", [("V001", "claude"), ("V003", "claude"), ("V011", "codex"), ("V012", "codex")])
def test_allowed_cases_sent_once_to_expected_destination(tmp_path, cid, dest):
    r = _run(cid, tmp_path)
    assert (r["outcome"], r["stop"]) == ("sent", "none")
    assert r["adapter_calls"] == 1
    assert r["sent_destinations"] == [dest]
    assert r["sent_payloads"][0]["payload_destination"] == dest
    assert r["wrong_destination"] is False and r["false_block"] is False
    assert r["driver"] == ("contract_a" if dest == "claude" else "contract_b")


@pytest.mark.parametrize("cid", ["V051", "V053", "V061"])
def test_c2_hold_never_calls_adapter(tmp_path, cid):
    r = _run(cid, tmp_path)
    assert (r["outcome"], r["stop"]) == ("held", "C2")
    assert r["adapter_calls"] == 0 and r["gateway_received"] is False
    assert r["c2"] in ("block", "hold")


def test_c2_timeout_holds_without_adapter_call(tmp_path):
    r = _run("V083", tmp_path, timeout_s=0.2, timeout_delay_s=0.5)
    assert (r["outcome"], r["stop"]) == ("held", "C2")
    assert r["adapter_calls"] == 0 and r["gateway_received"] is False


@pytest.mark.parametrize("cid,stop,outcome", [
    ("V081", "eligibility", "rejected"), ("V082", "eligibility", "rejected"),
    ("V084", "digest", "rejected"), ("V085", "confirm", "held"),
    ("V071", "eligibility", "rejected"), ("V073", "contract", "rejected"),
])
def test_process_conditions_stop_before_gateway(tmp_path, cid, stop, outcome):
    r = _run(cid, tmp_path)
    assert (r["outcome"], r["stop"]) == (outcome, stop)
    assert r["adapter_calls"] == 0 and r["gateway_received"] is False


def test_null_input_routes_without_gateway(tmp_path):
    r = _run("V041", tmp_path)
    assert r["driver"] == "c1_only" and (r["outcome"], r["stop"]) == ("human", "route")
    assert r["adapter_calls"] == 0


def _row(i, data_class="send_ok", exp_out="sent", exp_stop="none", out="sent", stop="none", calls=0,
         dests=None, exp_dest="claude"):
    dests = dests if dests is not None else []
    return {
        "id": i, "data_class": data_class, "expected_outcome": exp_out, "expected_stop": exp_stop,
        "expected_destination": exp_dest, "outcome": out, "stop": stop,
        "outcome_ok": out == exp_out, "stop_ok": stop == exp_stop, "adapter_calls": calls,
        "sent_destinations": dests, "gateway_received": calls > 0, "driver": "contract_a",
        "missend": data_class in ("local_only", "forbidden") and calls > 0,
        "false_block": exp_out == "sent" and calls == 0,
        "wrong_destination": calls > 0 and exp_dest is not None and any(d != exp_dest for d in dests),
        "duplicate_send": calls > 1,
    }


def test_summarize_counts():
    rows = [
        _row("ok", calls=1, dests=["claude"]),
        _row("blocked", out="held", stop="C2"),
        _row("wrong", calls=1, dests=["codex"]),
        _row("dup", calls=2, dests=["claude", "claude"]),
        _row("leak", data_class="forbidden", exp_out="held", exp_stop="C2", calls=1, dests=["codex"], exp_dest="codex"),
        _row("safe", data_class="forbidden", exp_out="rejected", exp_stop="eligibility", out="rejected", stop="eligibility"),
        _row("local", data_class="local_only", exp_out="local_answer", exp_stop="none", out="local_answer", exp_dest=None),
    ]
    s = summarize_e2e(rows)
    assert s["cases"] == 7
    assert s["system_missend"] == {"n": 1, "of": 3, "rate": round(1 / 3, 4)}
    assert s["system_missend_ids"] == ["leak"]
    assert (s["false_block"]["n"], s["false_block"]["of"], s["false_block_ids"]) == (1, 4, ["blocked"])
    assert (s["wrong_destination"]["n"], s["wrong_destination"]["of"], s["wrong_destination_ids"]) == (1, 4, ["wrong"])
    assert s["duplicate_send_ids"] == ["dup"]
    assert s["adapter_calls_total"] == 5 and s["sent_cases"] == 4
    assert s["mismatch_ids"] == ["blocked", "leak"]
    assert s["outcome_accuracy"]["n"] == 5 and s["stop_accuracy"]["n"] == 5
    assert s["gateway_reached_unexpected_ids"] == ["leak"]


def test_cli_writes_report_and_results(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["--out", str(out), "--timeout", "0.3"]) == 0
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["summary"]["cases"] == len(CASES)
    assert "judge_layer_summary" in res and res["confirm_policy"]
    # V072 は判断層では誤送信に数えられるが、実システムでは送信資格で止まる
    assert "V072" in res["judge_layer_summary"]["missend_ids"]
    assert "V072" not in res["summary"]["system_missend_ids"]
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "E2E" in md and "判断層" in md
