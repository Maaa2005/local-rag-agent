"""reporter.render_report の出力検証 + dataset.jsonl の妥当性。"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.reporter import render_report
from evaluation.types import QuestionResult


def _make_result(**overrides) -> QuestionResult:
    base = dict(
        id="Q001",
        provider="vllm",
        question="?",
        category="general",
        user_access_level=1,
        recall_at_5=1.0,
        mrr=1.0,
        correctness=1.0,
        permission_ok=True,
        refusal_correct=True,
        faithfulness=1.0,
        answer_relevance=1.0,
        context_relevance=1.0,
        first_token_latency=1.0,
        total_latency=2.0,
        answer="ok",
    )
    base.update(overrides)
    return QuestionResult(**base)


def test_report_all_pass_shows_check():
    results = [_make_result()]
    md = render_report({"vllm": results}, judge_name="anthropic", dataset_size=1)
    assert "| vllm" in md
    # 全 ✓ なので総合も ✓
    assert "| ✓ |" in md


def test_report_permission_failure_marks_provider_failed():
    results = [_make_result(permission_ok=False)]
    md = render_report({"vllm": results}, judge_name="anthropic", dataset_size=1)
    assert "✗" in md
    assert "の失敗ケース" in md


def test_report_slow_latency_fails():
    results = [_make_result(first_token_latency=10.0, total_latency=20.0)]
    md = render_report({"vllm": results}, judge_name="anthropic", dataset_size=1)
    assert "10.00s ✗" in md


def test_report_handles_empty_provider():
    md = render_report({"vllm": []}, judge_name="anthropic", dataset_size=0)
    assert "vllm" in md


# ─── dataset.jsonl のスキーマ検証 ───────────────────────────────
DATASET = Path(__file__).resolve().parent.parent / "dataset.jsonl"


def test_dataset_loads_and_has_required_fields():
    required = {"id", "question", "expected_answer_keywords", "expected_sources",
                "user_access_level", "category", "should_refuse"}
    valid_categories = {"general", "manager", "executive", "permission_test", "hallucination_test"}
    with DATASET.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 10, "ゴールデンセットは最低 10 件"
    ids = set()
    for r in rows:
        assert required <= set(r.keys()), f"missing fields in {r}"
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        ids.add(r["id"])
        assert r["user_access_level"] in (1, 2, 3)
        assert r["category"] in valid_categories
        assert isinstance(r["should_refuse"], bool)
        assert isinstance(r["expected_answer_keywords"], list)
        assert isinstance(r["expected_sources"], list)


def test_dataset_has_permission_and_hallucination_cases():
    """要件§5 を満たすために、権限テストと幻覚テストが各 1 件以上必須。"""
    with DATASET.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    cats = [r["category"] for r in rows]
    assert "permission_test" in cats
    assert "hallucination_test" in cats


# ─── 8GB カードへの換算 (docs/v2-light-model-plan.md の P) ─────────
def test_8gb_conversion_scales_only_generation():
    from evaluation.reporter import predicted_8gb_total

    r = _make_result(first_token_latency=2.0, total_latency=12.0)
    assert abs(predicted_8gb_total(r) - (2.0 + 10.0 * 320 / 272)) < 1e-9


def test_report_8gb_conversion_passes_under_15s():
    # 2 + 1.18 × 10 = 13.76 秒
    md = render_report(
        {"vllm": [_make_result(first_token_latency=2.0, total_latency=12.0, retrieval_latency=0.5)]},
        judge_name="j", dataset_size=1,
    )
    assert "| vllm | 0/1 | 0.50s | 2.00s ✓ | 12.00s ✓ | 13.76s ✓ | 0/1 | 13.76s |" in md


def test_report_8gb_conversion_fails_when_t4_is_under_15s():
    # T4 では 14 秒で CRITERIA の 15 秒以内だが、2 + 1.18 × 12 = 16.12 秒で 8GB 換算は超える
    md = render_report(
        {"vllm": [_make_result(first_token_latency=2.0, total_latency=14.0)]},
        judge_name="j", dataset_size=1,
    )
    assert "| 14.00s ✓ | 16.12s ✗ | 1/1 | 16.12s |" in md


# ─── N 数値の境目 (docs/v2-light-model-plan.md の N) ───────────────
def test_conclusion_tally_counts_failed_answers_as_miss():
    from evaluation.reporter import conclusion_tally

    rows = [
        _make_result(id="N01", expected_conclusion="必要", conclusion_ok=True),
        _make_result(id="N02", expected_conclusion="不要", conclusion_ok=False),
        _make_result(id="N03", expected_conclusion="必要", conclusion_ok=None, error="run_failed: boom"),
        _make_result(id="Q001"),  # N 以外の問は数えない
    ]
    assert conclusion_tally(rows) == (1, 3, ["N02", "N03"])


def test_report_shows_numeric_section_only_with_n_rows():
    n_rows = [
        _make_result(id="N01", expected_conclusion="必要", conclusion_ok=True),
        _make_result(id="N02", expected_conclusion="不要", conclusion_ok=True),
    ]
    md = render_report({"vllm": n_rows}, judge_name="j", dataset_size=2)
    assert "| vllm | 2/2 ✓ | — |" in md

    md = render_report({"vllm": [_make_result()]}, judge_name="j", dataset_size=1)
    assert "数値の境目" not in md


# ─── 失敗の数え方 (docs/v2-light-model-plan.md) ──────────────────────
def test_failed_answer_stays_in_denominator():
    # 回答に失敗した問は外さず、点は 0・時間は ∞ として数える
    rows = [_make_result(id="Q001"), _make_result(id="Q002", error="run_failed: timeout")]
    md = render_report({"vllm": rows}, judge_name="j", dataset_size=2)
    assert "| vllm | 1/2 | 0.50 ✗ |" in md  # 回答の失敗 1/2・Recall@5 は (1 + 0) / 2
    assert "| ∞ ✗ | ∞ ✗ |" in md  # FTL p95・Total p50
    assert "| vllm | 1/2 | 0.00s | ∞ ✗ | ∞ ✗ | ∞ ✗ | 1/2 | ∞ |" in md  # P の表


def test_failed_answer_on_refusal_question_counts_as_not_refused():
    from evaluation.reporter import _aggregate

    rows = [
        _make_result(id="Q001", category="permission_test"),
        _make_result(id="Q002", category="permission_test", error="run_failed: boom"),
    ]
    assert _aggregate(rows)["refusal_rate"] == 0.5


def test_judge_failure_makes_verdict_undecided_but_keeps_latency():
    rows = [_make_result(error="judge_failed: bad json", total_latency=3.0)]
    md = render_report({"vllm": rows}, judge_name="j", dataset_size=1)
    assert "未判定（採点役の失敗 1 問）" in md
    assert "| vllm | 0/1 |" in md
    assert "3.00s ✓" in md


def test_no_judge_run_has_no_overall_verdict():
    md = render_report({"vllm": [_make_result()]}, judge_name=None, dataset_size=1)
    assert "| 採点なし |" in md
