"""evaluation/v2/numeric_boundary.jsonl (合否の線 N) の整合性検証。

N は 12/12 で合格の線なので、正解表に 1 問でも誤りがあると全候補が落ちる。
ここでは正解を人の読みではなく境目の不等号から導き直して照合し、
規則の原文がコーパスに実在すること、run_eval でそのまま流せる形であることを確かめる。
"""
from __future__ import annotations

import json
import operator
from collections import defaultdict
from pathlib import Path

import pytest

from evaluation import metrics

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "v2" / "numeric_boundary.jsonl"
CORPUS_ROOT = ROOT / "corpus"

CATEGORY_ACCESS_LEVEL = {"general": 1, "manager": 2, "executive": 3}
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
RUN_EVAL_FIELDS = (
    "id", "question", "user_access_level", "category", "should_refuse",
    "expected_sources", "expected_answer_keywords",
)


def _load() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


ROWS = _load()


def _corpus_text(source: str) -> str:
    rel = [p for p in Path(source).parts if p not in ("/", "watched")]
    return CORPUS_ROOT.joinpath(*rel).read_text(encoding="utf-8")


def test_twelve_questions_with_unique_ids():
    assert len(ROWS) == 12
    assert len({r["id"] for r in ROWS}) == 12


def test_each_rule_has_below_at_above():
    positions = defaultdict(list)
    for r in ROWS:
        positions[r["rule_id"]].append(r["position"])
    assert len(positions) == 4
    for rule_id, ps in positions.items():
        assert sorted(ps) == ["above", "at", "below"], rule_id


def test_inclusive_and_exclusive_bounds_are_both_covered():
    ops = {r["boundary"]["op"] for r in ROWS}
    assert ops & {">=", "<="} and ops & {">", "<"}


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_run_eval_can_read_row(row):
    for key in RUN_EVAL_FIELDS:
        assert key in row, key
    assert row["should_refuse"] is False


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_expected_conclusion_follows_from_boundary(row):
    b = row["boundary"]
    holds = OPS[b["op"]](b["value"], b["threshold"])
    assert row["expected_conclusion"] == (b["then"] if holds else b["else"])
    assert sorted(row["conclusion_options"]) == sorted([b["then"], b["else"]])


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_value_sits_on_the_named_side(row):
    b = row["boundary"]
    v, t = b["value"], b["threshold"]
    expected = {"below": v < t, "at": v == t, "above": v > t}[row["position"]]
    assert expected, (row["position"], v, t)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_question_asks_for_conclusion_line(row):
    for option in row["conclusion_options"]:
        assert f"「結論: {option}」" in row["question"]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_rule_and_keywords_exist_in_source(row):
    (source,) = row["expected_sources"]
    text = _corpus_text(source)
    assert row["rule_quote"] in text
    for kw in row["expected_answer_keywords"]:
        assert kw in text, kw


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_access_level_matches_source_folder(row):
    (source,) = row["expected_sources"]
    folder = Path(source).parts[2]
    assert row["category"] == folder
    assert row["user_access_level"] == CATEGORY_ACCESS_LEVEL[folder]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_scorer_accepts_expected_and_rejects_other(row):
    other = next(o for o in row["conclusion_options"] if o != row["expected_conclusion"])
    assert metrics.numeric_conclusion_ok(f"説明 [1]\n結論: {row['expected_conclusion']}", row["expected_conclusion"])
    assert not metrics.numeric_conclusion_ok(f"説明 [1]\n結論: {other}", row["expected_conclusion"])
