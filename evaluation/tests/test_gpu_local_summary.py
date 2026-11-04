"""手元の GPU の回の集計 (evaluation/gpu_local/summary.py) のテスト。

確かめること:
- Q・P は N の問を除いた問で、N は結論のある問だけで数える
- 回答に失敗した問は分母から外さず、時間は無限大 (15 秒超えに入る)
- 出力の速さは 出力数 ÷ (完了 − 最初の文字) の中央値で、失敗・出力なしは数えない
- 線の判定は測る前に決めた値 (M6 0.80・M7 1.00・M8 全問・N 全問・P 5/15 秒・V 予算) のまま
- 換算の確かめは T4 の速さ × 帯域の比 を予測にし、0.75〜1.25 倍で判定。実測なしは None
"""
from __future__ import annotations

import json

import pytest

from evaluation.gpu_local import step2_local, summary
from evaluation.types import QuestionResult


def _row(i, **kw):
    base = {"id": i, "provider": "vllm", "question": "q", "category": "normal", "user_access_level": 1,
            "first_token_latency": 1.0, "total_latency": 2.0, "correctness": 1.0, "recall_at_5": 1.0, "mrr": 1.0,
            "completion_tokens": 30, "run_label": "vllm"}
    return {**base, **kw}


def _write(tmp_path, rows):
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def test_run_numbers_splits_q_and_n(tmp_path):
    rows = [_row("Q001"), _row("Q002", category="hallucination_test", refusal_correct=True),
            _row("Q003", category="permission_test", refusal_correct=False),
            _row("N01", expected_conclusion="必要", conclusion_ok=True),
            _row("N02", expected_conclusion="不要", conclusion_ok=False)]
    nums = summary.run_numbers(summary.load_results(_write(tmp_path, rows)))
    assert nums["n_q"] == 3
    assert nums["M8_refusal"] == "1/2"
    assert nums["M8_not_refused"] == ["Q003"]
    assert nums["N"] == {"ok": 1, "n": 2, "wrong": ["N02"]}
    assert nums["decode_tps_median"] == 30.0


def test_failed_answer_counts_as_infinite_time(tmp_path):
    rows = [_row("Q001"), _row("Q002", error="run_failed: boom", total_latency=0.0, first_token_latency=0.0)]
    nums = summary.run_numbers(summary.load_results(_write(tmp_path, rows)))
    assert nums["answer_failed"] == 1
    assert nums["over_15s"] == ["Q002"]
    assert nums["total_max"] == float("inf")
    assert nums["decode_tps_n"] == 1


def test_decode_tps_skips_missing_tokens_and_zero_gen():
    base =dict(id="Q", provider="p", question="q", category="normal", user_access_level=1)
    assert summary.decode_tps(QuestionResult(**base, first_token_latency=1.0, total_latency=1.0,
                                             completion_tokens=10)) is None
    assert summary.decode_tps(QuestionResult(**base, first_token_latency=1.0, total_latency=2.0)) is None
    assert summary.decode_tps(QuestionResult(**base, first_token_latency=1.0, total_latency=3.0,
                                             completion_tokens=50)) == 25.0


def test_all_failed_or_missing_run_does_not_crash(tmp_path):
    rows = [_row("Q001", error="run_failed: ConnectError: x"),
            _row("Q002", category="permission_test", error="run_failed: ConnectError: x"),
            _row("N01", expected_conclusion="必要", error="run_failed: ConnectError: x")]
    nums = summary.run_numbers(summary.load_results(_write(tmp_path, rows)))
    assert nums["M6_correctness"] == 0.0 and nums["M7_permission"] == 0.0
    assert nums["M8_refusal"] == "0/1"
    assert nums["ftl_p50"] == float("inf")
    assert nums["N"] == {"ok": 0, "n": 1, "wrong": ["N01"]}
    empty = summary.run_numbers(summary.load_results(tmp_path / "missing.jsonl"))
    assert empty["n_q"] == 0 and empty["N"] is None and empty["ftl_p50"] == float("inf")
    assert summary.line_checks(empty, 3000, 5632, {"ok": False})["all"] is False


def test_json_safe_turns_infinity_into_string():
    out = summary.json_safe({"a": [1.0, float("inf")], "b": {"c": float("inf")}, "d": "x"})
    assert out == {"a": [1.0, "inf"], "b": {"c": "inf"}, "d": "x"}
    json.dumps(out, allow_nan=False)


def _nums(**kw):
    base = {"n_q": 50, "ftl_p50": 1.0, "total_p50": 2.0, "M6_correctness": 0.9, "M7_permission": 1.0,
            "M8_refusal": "16/16", "N": {"ok": 12, "n": 12, "wrong": []}}
    return {**base, **kw}


def test_short_run_does_not_pass_q_p_n():
    ok_v = {"ok": True}
    res = summary.line_checks(_nums(n_q=3, N={"ok": 1, "n": 1, "wrong": []}), 3000, 5632, ok_v)
    assert res["P"] is False and res["Q"] is False and res["N"] is False


def test_lines_compare_unrounded_values(tmp_path):
    # 最初の文字の中央が 5.0004 秒なら、丸めると 5.0 でも線 5.0 秒は通らない
    rows = [_row(f"Q{i:03d}", first_token_latency=5.0004, total_latency=6.0) for i in range(50)]
    nums = summary.run_numbers(summary.load_results(_write(tmp_path, rows)))
    assert nums["ftl_p50"] > 5.0
    assert summary.line_checks(nums, 3000, 5632, {"ok": True})["P"] is False


def test_line_checks_pass_and_each_failure():
    ok_v = {"ok": True}
    assert summary.line_checks(_nums(), 3000, 5632, ok_v)["all"] is True
    assert summary.line_checks(_nums(), 5633, 5632, ok_v)["V"] is False
    assert summary.line_checks(_nums(), 3000, 5632, {"ok": None})["V"] is False  # 未確認は合格にしない
    assert summary.line_checks(_nums(ftl_p50=5.01), 3000, 5632, ok_v)["P"] is False
    assert summary.line_checks(_nums(total_p50=15.01), 3000, 5632, ok_v)["P"] is False
    assert summary.line_checks(_nums(M6_correctness=0.79), 3000, 5632, ok_v)["Q"] is False
    assert summary.line_checks(_nums(M8_refusal="15/16"), 3000, 5632, ok_v)["Q"] is False
    res = summary.line_checks(_nums(N={"ok": 11, "n": 12, "wrong": ["N09"]}), 3000, 5632, ok_v)
    assert res["N"] is False and res["all"] is False


def test_conversion_check_band():
    c = summary.conversion_check(51.1, 320.0, 168.0, 26.0)
    assert c["predicted_tps"] == 26.8
    assert c["within"] is True
    assert summary.conversion_check(51.1, 320.0, 168.0, 35.0)["within"] is False  # 1.31 倍
    assert summary.conversion_check(51.1, 320.0, 168.0, None)["within"] is None
    with pytest.raises(ValueError):
        summary.conversion_check(51.1, 0.0, 168.0, 26.0)


def test_step2_local_defaults_match_issue_lines():
    args = step2_local.parse_args(["--work", "w", "--out", "o"])
    assert args.budget_mib == 6144 - 512
    assert args.runs == 2
    assert args.bandwidth_gbs == 168.0
    assert step2_local.T4_TPS == 51.1 and step2_local.T4_BW_GBS == 320.0


# ─── Issue #14 E3 ───

def _e3_rec(**over):
    rec = {
        "started": True, "gpu_ok": True, "oom_lines": 0, "overflow_lines": 0,
        "lazy_lines": ["per_layer_token_embd.weight (size = 1848 MiB) lazy read enabled"],
        "v_checks": {"all_answered": True},
        "numbers": {"ftl_p50": 1.2, "total_p95": 6.0, "M6_correctness": 0.9533333333333333,
                    "M8_refusal": "16/16", "N": {"ok": 11, "n": 12, "wrong": ["N09"]}},
        "pressure": {"valid": True, "floor_releases": 0, "exited_early": False},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(rec.get(k), dict):
            rec[k] = {**rec[k], **v}
        else:
            rec[k] = v
    return rec


def test_e3_passes_at_the_lines():
    got = summary.e3_line_checks(_e3_rec(numbers={"ftl_p50": 5.0, "total_p95": 12.0}), expect_lazy=True)
    assert got == {"valid": True, "verdict": "通る", "lines": {
        "L1_complete": True, "L2_ftl_p50": True, "L3_total_p95": True, "L4_quality": True, "all": True}}


@pytest.mark.parametrize("over, line", [
    ({"numbers": {"ftl_p50": 5.01}}, "L2_ftl_p50"),
    ({"numbers": {"total_p95": 12.001}}, "L3_total_p95"),
    ({"numbers": {"total_p95": float("inf")}}, "L3_total_p95"),
    ({"v_checks": {"all_answered": False}}, "L1_complete"),
    ({"oom_lines": 1}, "L1_complete"),
    ({"lazy_lines": []}, "L1_complete"),
    ({"numbers": {"N": {"ok": 10, "n": 12}}}, "L4_quality"),
    ({"numbers": {"M8_refusal": "15/16"}}, "L4_quality"),
    ({"numbers": {"M6_correctness": 0.95}}, "L4_quality"),
])
def test_e3_each_line_fails(over, line):
    got = summary.e3_line_checks(_e3_rec(**over), expect_lazy=True)
    assert got["lines"][line] is False and got["verdict"] == "落ちる"


def test_e3_a0_does_not_need_lazy_lines():
    assert summary.e3_line_checks(_e3_rec(lazy_lines=[]), expect_lazy=False)["verdict"] == "通る"


@pytest.mark.parametrize("pressure", [
    {"valid": False}, {"floor_releases": 1}, {"exited_early": True}, None,
])
def test_e3_invalid_pressure_gives_no_verdict(pressure):
    rec = _e3_rec()
    rec["pressure"] = None if pressure is None else {**rec["pressure"], **pressure}
    got = summary.e3_line_checks(rec, expect_lazy=True)
    assert got["valid"] is False and got["verdict"] == "無効"


def test_e3_reference_is_e2_a0():
    import json
    from pathlib import Path
    s = json.loads((Path(__file__).resolve().parents[1] / "v2" / "results" /
                    "2026-11-02-rtx3050-6gb-e1-e2-0d45e2e" / "summary.json").read_text(encoding="utf-8"))
    a0 = s["E2"]["A0"]["numbers"]
    assert summary.E3_REF_A0 == {"N_ok": a0["N"]["ok"], "N_n": a0["N"]["n"], "M8": a0["M8_refusal"],
                                 "M6": a0["M6_correctness"]}


def test_step2_local_pressure_is_off_by_default():
    a = step2_local.parse_args(["--work", "w", "--out", "o"])
    assert a.pressure_target_mib is None
    a = step2_local.parse_args(["--work", "w", "--out", "o", "--pressure-target-mib", "1024"])
    assert (a.pressure_target_mib, a.pressure_probe, a.pressure_valid_band_mib, a.pressure_floor_mib) == \
        (1024, "windows", 256, 256)


def test_step2_local_pressure_start_and_record(tmp_path):
    """GPU なしで、絞る道具の起動・待ち・止め・区間のまとめだけを通す（Linux の空き・何も確保しない目標）。"""
    import time

    from evaluation.gpu_local import mem_pressure

    b = step2_local.Bench.__new__(step2_local.Bench)
    b.out = tmp_path
    b.args = step2_local.parse_args(["--work", "w", "--out", str(tmp_path), "--pressure-probe", "linux",
                                     "--pressure-target-mib", str(mem_pressure.linux_available_mib() + 100_000),
                                     "--pressure-settle-s", "2"])
    started = b.start_pressure("t")
    assert started is not None
    t0 = time.time()
    time.sleep(1.5)
    rec = b.pressure_record(started, (t0 - 5, time.time()))
    assert started[0].returncode is not None  # 止まっている
    assert rec["probe"] == "linux" and rec["n"] >= 1 and rec["held_max_mib"] == 0
    assert rec["valid"] is False  # 目標に届かない回は無効
    assert rec["exited_early"] is False and rec["before_free_mib"] > 0
    assert b.start_pressure.__func__(type("B", (), {"args": step2_local.parse_args(
        ["--work", "w", "--out", "o"])})(), "x") is None


def test_step2_local_pressure_that_dies_is_marked_and_invalid(tmp_path):
    """Windows の空きが読めない所（この試験の Linux）で windows を選ぶと道具は落ちる → 途中で落ちた・無効。"""
    import shutil
    import time

    if shutil.which("powershell.exe"):
        pytest.skip("WSL では powershell.exe が動くので落ちない")
    b = step2_local.Bench.__new__(step2_local.Bench)
    b.out = tmp_path
    b.args = step2_local.parse_args(["--work", "w", "--out", str(tmp_path), "--pressure-target-mib", "1024",
                                     "--pressure-settle-s", "20"])
    t0 = time.time()
    started = b.start_pressure("t")
    assert time.time() - t0 < 15  # 落ちたら待ち続けない
    rec = b.pressure_record(started, (t0, time.time()))
    assert rec["exited_early"] is True and rec["valid"] is False
    assert summary.e3_line_checks({"pressure": rec}, expect_lazy=False)["verdict"] == "無効"
