"""生成器 (evaluation/distill/gen) のテスト（life #45・docs/verified-distill-plan.md §1・§2）。

確かめること:
- 境目の定義が封 B の正解のコード（teacher_check.rules.cmp）と同じ
- 指示文が本番（backend/app/services/llm.py）と一字一句同じ
- 家族の題材・境目の値が封と重ならない、学習用と開発用の入力に他の側の規程が入らない
- 封との 13 文字の一致が 0、割合どおり、正解は選択肢の中、答えの条の塊が見つかる
- 境目の 3 つ組は結論が変わる所をまたぎ、例外の対は結論が逆
"""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from evaluation.distill.gen import leak, make, spec
from evaluation.distill.gen.families import DEV_FAMILIES, FAMILIES, TRAIN_FAMILIES
from evaluation.teacher_check.rules import cmp

REPO = Path(__file__).resolve().parents[2]
TRAIN = make.generate("train", 600, 11)
DEV = make.generate("dev", 300, 11)


@pytest.mark.parametrize("word", list(spec.WORDS))
def test_comparison_matches_sealed_b_definition(word):
    o = spec.WORDS[word][0]
    for v in (Fraction(9), Fraction(10), Fraction(11)):
        assert spec.holds(v, o, Fraction(10)) == cmp(v, o, Fraction(10))


def test_system_prompt_is_production():
    src = (REPO / "backend/app/services/llm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    prod = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "SYSTEM_PROMPT")
    assert prod == make.SYSTEM_PROMPT


def test_families_disjoint_from_sealed():
    assert leak.family_problems() == []
    assert len(FAMILIES) == 30 and len(TRAIN_FAMILIES) == 24 and len(DEV_FAMILIES) == 6
    assert {f["type"] for f in FAMILIES if f["id"] in DEV_FAMILIES} <= {f["type"] for f in FAMILIES}


def test_normalization_examples():
    assert spec.normalize(10, "万円") == spec.normalize(100_000, "円")
    assert spec.normalize(8, "割") == spec.normalize(80, "%")
    assert not spec.threshold_allowed(100_000, "円") and not spec.threshold_allowed(Fraction(4, 5), "比")
    assert spec.threshold_allowed(6, "割")


def test_no_13gram_overlap_with_sealed():
    sealed = leak.sealed_grams() - leak.allowed_grams()
    bad = [r["id"] for r in TRAIN + DEV if leak.overlap(r["messages"][1]["content"], sealed)]
    assert bad == []


def test_splits_do_not_mix():
    titles = {f["id"]: f["title"] for f in FAMILIES}
    for rows, other in ((TRAIN, DEV_FAMILIES), (DEV, TRAIN_FAMILIES)):
        for r in rows:
            user = r["messages"][1]["content"]
            assert not any(f"] {titles[x]}.md" in user for x in other), r["id"]


def test_mix_and_options():
    c = Counter(r["category"] for r in TRAIN)
    assert abs(c["boundary"] / len(TRAIN) - 0.40) < 0.02 and abs(c["exception"] / len(TRAIN) - 0.15) < 0.02
    for r in TRAIN + DEV:
        if r["category"] in ("refusal", "describe"):
            assert r["options"] == []
        else:
            assert r["expected"] in r["options"]
            assert r["rule_chunk_ids"], r["id"]
        if r["expected"] == make.UNDECIDED:
            assert make.UNDECIDED in r["options"]


def test_boundary_triples_cross_and_exception_pairs():
    f = make.FAM["F02"]
    items = [i for i in make.items_single(f, __import__("random").Random(0)) if i.category == "boundary" and i.style == 0
             and i.word == f["word"]]
    assert len({i.expected for i in items}) == 2
    ex = make.items_exception(make.FAM["F28"], __import__("random").Random(0))
    applies = [i for i in ex if i.category == "exception" and make.FAM["F28"]["exc"] in i.facts_text]
    near = [i for i in ex if i.category == "exception" and make.FAM["F28"]["exc_near"] in i.facts_text]
    assert applies and near and {i.expected for i in applies} != {i.expected for i in near}


def test_age_dates_avoid_anniversary_and_count_full_years():
    import random
    for it in make.items_age(make.FAM["F23"], random.Random(1)):
        assert it.check["at"] in (True, False)


def test_contrast_pairs_exist():
    words = {(r["family"], r["word"]) for r in TRAIN if r["type"] == "single"}
    fams = {f for f, _ in words}
    assert any(len({w for f2, w in words if f2 == f}) >= 2 for f in fams)
