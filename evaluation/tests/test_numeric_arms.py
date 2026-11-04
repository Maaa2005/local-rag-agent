"""数の比べの腕の比較 (evaluation/gpu_local/numeric_arms.py) の集計のテスト。

確かめること:
- 保留・失敗・誤った断定を分けて数え、どれも分母から外さない
- 値の一致は枠ごとに見て、値が無いはずの枠に値を入れた時も違いに数える
- 腕の設定は 1 つずつで、カード方式だけが下書きのカードを使う
"""
from __future__ import annotations

from evaluation.gpu_local import numeric_arms as na
from evaluation.types import QuestionResult


def _r(id_: str, answer: str, expected: str = "必要", error: str | None = None) -> QuestionResult:
    from evaluation import metrics

    r = QuestionResult(id=id_, question="q", provider="vllm", category="c", user_access_level=1)
    r.answer, r.expected_conclusion, r.error = answer, expected, error
    r.total_latency, r.first_token_latency, r.retrieval_latency = 3.0, 1.0, 0.5
    r.conclusion_ok = None if error else metrics.numeric_conclusion_ok(answer, expected)
    return r


def test_round_numbers_separate_held_failed_and_wrong():
    rows = [_r("a", "x\n結論: 必要"), _r("b", "x\n結論: 不要"), _r("c", "x\n結論: 保留"),
            _r("d", "", error="run_failed: Timeout")]
    n = na.round_numbers(rows)
    assert (n["ok"], n["n"]) == (1, 4)
    assert n["held"] == ["c"]
    assert n["failed"] == ["d"]
    assert n["wrong_decided"] == ["b"]
    assert n["total_max"] == float("inf")


def test_slot_matches_counts_invented_values_as_wrong():
    gold = {"金額": {"value": 49000, "quote": "49,000円"}, "日数": {"value": None, "quote": None}}
    assert na.slot_matches(gold, {"金額": {"value": 49000.0}, "日数": {"value": None}}) == {"金額": True, "日数": True}
    assert na.slot_matches(gold, {"金額": {"value": 49001}, "日数": {"value": 3}}) == {"金額": False, "日数": False}
    assert na.slot_matches(gold, None) == {"金額": False, "日数": True}


def test_arms_change_one_thing_at_a_time():
    assert na.ARMS["c0"] == {}
    for arm in ("norm", "contrast", "codeform"):
        assert len(na.ARMS[arm]) == 1
    assert na.ARMS["cards"]["RULE_CARDS_ALLOW_DRAFT"] == "true"
    assert all(ds in (na.N12, na.DEV24) for _, ds, _ in na.ROUNDS)
