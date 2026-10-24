"""evaluation/v2/numeric_dev24.jsonl・numeric_holdout48.jsonl（数の比べの別の問）の整合性検証。

合否の線は今までどおり N（numeric_boundary.jsonl）。この 2 組は C（数の比べをコードで）と
考え ON の補助の証拠で、開発用 24 問は直すのに使い、確かめ用 48 問は直した後に 1 回だけ流す。
正解は境目の不等号から導き直して照合し、規程の文がそれぞれの別コーパス（開発用は corpus_nx_dev・確かめ用は corpus_nx_holdout）に実在すること、
「いつも同じ向きに答えれば当たる」形になっていないことを確かめる。
"""
from __future__ import annotations

import datetime as dt
import json
import operator
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from evaluation import metrics

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "v2"

SETS = {"dev24": (V2 / "numeric_dev24.jsonl", 24, "D"), "holdout48": (V2 / "numeric_holdout48.jsonl", 48, "H")}
CORPUS_ROOTS = {"D": V2 / "corpus_nx_dev", "H": V2 / "corpus_nx_holdout"}

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
RUN_EVAL_FIELDS = (
    "id", "question", "user_access_level", "category", "should_refuse",
    "expected_sources", "expected_answer_keywords",
)


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


DATA = {name: _load(path) for name, (path, _, _) in SETS.items()}
ALL_ROWS = [r for rows in DATA.values() for r in rows]


def _corpus_text(source: str, root: Path) -> str:
    rel = [p for p in Path(source).parts if p not in ("/", "watched")]
    return root.joinpath(*rel).read_text(encoding="utf-8")


def _shown(row: dict) -> str:
    v = row["boundary"]["value"]
    if row["kind"] == "date":
        d = dt.date.fromisoformat(v)
        return f"{d.year}年{d.month}月{d.day}日"
    return f"{v:,}" if isinstance(v, int) and v >= 1000 else f"{v:g}"


@pytest.mark.parametrize("name", SETS)
def test_size_ids_and_prefix(name):
    rows = DATA[name]
    _, size, prefix = SETS[name]
    assert len(rows) == size
    assert len({r["id"] for r in rows}) == size
    assert all(r["id"].startswith(prefix) for r in rows)


@pytest.mark.parametrize("name", SETS)
def test_kind_by_op_by_position_is_covered_evenly(name):
    cells = Counter((r["kind"], r["boundary"]["op"], r["position"]) for r in DATA[name])
    per_cell = SETS[name][1] // 24
    assert len(cells) == 2 * 4 * 3
    assert set(cells.values()) == {per_cell}


@pytest.mark.parametrize("name", SETS)
def test_each_rule_has_below_at_above(name):
    positions = defaultdict(list)
    for r in DATA[name]:
        positions[r["rule_id"]].append(r["position"])
    for rule_id, ps in positions.items():
        assert sorted(ps) == ["above", "at", "below"], rule_id


def test_dev_and_holdout_share_no_rule_or_document():
    dev, hold = DATA["dev24"], DATA["holdout48"]
    assert not {r["rule_id"] for r in dev} & {r["rule_id"] for r in hold}
    assert not {r["rule_quote"] for r in dev} & {r["rule_quote"] for r in hold}
    assert not {r["expected_sources"][0] for r in dev} & {r["expected_sources"][0] for r in hold}


def test_dev_and_holdout_documents_live_in_separate_corpora():
    dev_docs = {p.name for p in (CORPUS_ROOTS["D"] / "general").glob("*.md")}
    hold_docs = {p.name for p in (CORPUS_ROOTS["H"] / "general").glob("*.md")}
    assert dev_docs == {Path(r["expected_sources"][0]).name for r in DATA["dev24"]}
    assert hold_docs == {Path(r["expected_sources"][0]).name for r in DATA["holdout48"]}


def test_no_overlap_with_the_n_questions():
    n_rows = _load(V2 / "numeric_boundary.jsonl")
    assert not {r["id"] for r in n_rows} & {r["id"] for r in ALL_ROWS}
    assert not {r["rule_quote"] for r in n_rows} & {r["rule_quote"] for r in ALL_ROWS}


@pytest.mark.parametrize("name", SETS)
def test_answers_are_not_one_sided(name):
    """「いつも否定で答える」「いつも 1 つ目の選択肢を選ぶ」では 6 割を超えて当たらない。"""
    rows = DATA[name]
    negative = sum(r["expected_conclusion"].endswith("ない") or r["expected_conclusion"] == "不要" for r in rows)
    first = sum(r["expected_conclusion"] == r["conclusion_options"][0] for r in rows)
    for count in (negative, first):
        assert 0.4 <= count / len(rows) <= 0.6, (name, negative, first)


@pytest.mark.parametrize("row", ALL_ROWS, ids=lambda r: r["id"])
def test_run_eval_can_read_row(row):
    for key in RUN_EVAL_FIELDS:
        assert key in row, key
    assert row["should_refuse"] is False


@pytest.mark.parametrize("row", ALL_ROWS, ids=lambda r: r["id"])
def test_expected_conclusion_follows_from_boundary(row):
    b = row["boundary"]
    holds = OPS[b["op"]](b["value"], b["threshold"])
    assert row["expected_conclusion"] == (b["then"] if holds else b["else"])
    assert sorted(row["conclusion_options"]) == sorted([b["then"], b["else"]])


@pytest.mark.parametrize("row", ALL_ROWS, ids=lambda r: r["id"])
def test_value_sits_on_the_named_side(row):
    b = row["boundary"]
    v, t = b["value"], b["threshold"]
    expected = {"below": v < t, "at": v == t, "above": v > t}[row["position"]]
    assert expected, (row["position"], v, t)


@pytest.mark.parametrize("row", ALL_ROWS, ids=lambda r: r["id"])
def test_question_shows_value_and_asks_for_conclusion_line(row):
    assert _shown(row) in row["question"]
    for option in row["conclusion_options"]:
        assert f"「結論: {option}」" in row["question"]


@pytest.mark.parametrize("row", ALL_ROWS, ids=lambda r: r["id"])
def test_rule_and_keywords_exist_in_source(row):
    (source,) = row["expected_sources"]
    text = _corpus_text(source, CORPUS_ROOTS[row["id"][0]])
    assert row["rule_quote"] in text
    for kw in row["expected_answer_keywords"]:
        assert kw in text, kw
    assert row["category"] == "general" and row["user_access_level"] == 1


@pytest.mark.parametrize("row", ALL_ROWS, ids=lambda r: r["id"])
def test_scorer_accepts_expected_and_rejects_other(row):
    other = next(o for o in row["conclusion_options"] if o != row["expected_conclusion"])
    assert metrics.numeric_conclusion_ok(f"説明 [1]\n結論: {row['expected_conclusion']}", row["expected_conclusion"])
    assert not metrics.numeric_conclusion_ok(f"説明 [1]\n結論: {other}", row["expected_conclusion"])
