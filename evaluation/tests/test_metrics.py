"""evaluation.metrics と evaluation.judges のユニットテスト。"""
from __future__ import annotations

import pytest

from evaluation import metrics
from evaluation.judges import _extract_json


# ─── recall / MRR ─────────────────────────────────────────────
def test_recall_hit():
    assert metrics.recall_at_k(["a", "b", "c"], ["b"], k=5) == 1.0


def test_recall_miss():
    assert metrics.recall_at_k(["a", "b", "c"], ["z"], k=5) == 0.0


def test_recall_respects_k():
    assert metrics.recall_at_k(["a", "b", "c", "d", "e", "target"], ["target"], k=5) == 0.0
    assert metrics.recall_at_k(["a", "b", "c", "d", "target"], ["target"], k=5) == 1.0


def test_recall_empty_expected_returns_1():
    assert metrics.recall_at_k(["a"], [], k=5) == 1.0


def test_mrr_rank_1():
    assert metrics.reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_mrr_rank_3():
    assert metrics.reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)


def test_mrr_not_found():
    assert metrics.reciprocal_rank(["a", "b"], ["z"]) == 0.0


# ─── 権限 / 拒否 ──────────────────────────────────────────────
def test_permission_ok_pass():
    chunks = [{"access_level": 1}, {"access_level": 2}]
    assert metrics.permission_ok(chunks, user_access_level=2) is True


def test_permission_ok_fails_on_level_violation():
    chunks = [{"access_level": 1}, {"access_level": 3}]
    assert metrics.permission_ok(chunks, user_access_level=2) is False


def test_is_refusal_detects_template():
    assert metrics.is_refusal("その情報は現在の資料には含まれていません。") is True
    assert metrics.is_refusal("これは10日です。") is False


# ─── correctness ─────────────────────────────────────────────
def test_correctness_partial():
    score = metrics.correctness_keyword_score("有給は10日付与されます", ["10日", "勤続6ヶ月"])
    assert score == 0.5


def test_correctness_full():
    score = metrics.correctness_keyword_score("10日かつ勤続6ヶ月", ["10日", "勤続6ヶ月"])
    assert score == 1.0


def test_correctness_no_expected():
    assert metrics.correctness_keyword_score("any", []) == 1.0


# ─── percentile / mean ───────────────────────────────────────
def test_percentile_basic():
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert metrics.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_percentile_empty():
    assert metrics.percentile([], 50) == 0.0


def test_mean_empty():
    assert metrics.mean([]) == 0.0


# ─── 文分割 ──────────────────────────────────────────────────
def test_split_sentences_jp():
    assert metrics.split_sentences("こんにちは。今日は晴れです。") == ["こんにちは。", "今日は晴れです。"]


def test_split_sentences_en_mix():
    assert metrics.split_sentences("Hello. これはテスト!") == ["Hello.", "これはテスト!"]


# ─── ジャッジ JSON 抽出 ─────────────────────────────────────
def test_extract_json_clean():
    text = '{"faithfulness": 1.0, "answer_relevance": 0.5}'
    assert _extract_json(text) == {"faithfulness": 1.0, "answer_relevance": 0.5}


def test_extract_json_with_preamble():
    text = '評価結果を返します:\n{"faithfulness": 0.8}\n以上です。'
    assert _extract_json(text) == {"faithfulness": 0.8}


def test_extract_json_malformed_returns_empty():
    assert _extract_json("not json at all") == {}
    assert _extract_json("{ broken json") == {}
