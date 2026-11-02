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


def _nums(**kw):
    base = {"ftl_p50": 1.0, "total_p50": 2.0, "M6_correctness": 0.9, "M7_permission": 1.0, "M8_refusal": "16/16",
            "N": {"ok": 12, "n": 12, "wrong": []}}
    return {**base, **kw}


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
