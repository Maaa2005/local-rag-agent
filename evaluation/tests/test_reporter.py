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
