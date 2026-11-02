"""教師の確認の問題集 (evaluation/teacher_check) のテスト。

確かめること:
- 96 問・区分 48/24/24・先に流す 12 問は区分ごとに 4 問
- 正解は選択肢の中にあり、境目の 3 問は結論が変わる所をまたいでいる
- 例外が当たる問と当たらない問は対で結論が逆、判断できない問は 6 問だけ
- 根拠の条は規程に実在し、規程は今までの評価で使っていない
- 3 値の組み合わせと採点の読み方
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.teacher_check import build as tb
from evaluation.teacher_check import rules as tr
from evaluation.teacher_check import score as ts
from evaluation.teacher_check.questions import build

ROWS = build()
REPO = Path(__file__).resolve().parents[2]


def test_counts_and_ids():
    assert len(ROWS) == 96
    assert Counter(r["category"] for r in ROWS) == {"boundary": 48, "paraphrase": 24, "exception": 24}
    assert len({r["id"] for r in ROWS}) == 96
    stage1 = [r for r in ROWS if r["stage1"]]
    assert Counter(r["category"] for r in stage1) == {"boundary": 4, "paraphrase": 4, "exception": 4}


def test_expected_is_one_of_options_and_footer_lists_them():
    for r in ROWS:
        assert r["expected_conclusion"] in r["options"]
        assert len(r["options"]) == 3 and r["options"][2] == tr.UNDECIDED
        for opt in r["options"]:
            assert f"「{opt}」" in r["question"]


def test_boundary_triples_cross_the_line():
    for cat in ("boundary", "paraphrase"):
        rows = [r for r in ROWS if r["category"] == cat]
        for i in range(0, len(rows), 3):
            triple = rows[i:i + 3]
            assert len({r["outcome"] for r in triple}) == 1
            assert len({r["expected_conclusion"] for r in triple}) == 2, [r["id"] for r in triple]
            assert tr.UNDECIDED not in {r["expected_conclusion"] for r in triple}


def test_exception_pairs_and_holds():
    ex = [r for r in ROWS if r["category"] == "exception"]
    applies, near, hold, decided = ex[0:6], ex[6:12], ex[12:18], ex[18:24]
    for a, n in zip(applies, near):
        assert a["outcome"] == n["outcome"]
        assert a["expected_conclusion"] != n["expected_conclusion"]
    assert all(r["expected_conclusion"] == tr.UNDECIDED for r in hold)
    assert all(r["expected_conclusion"] != tr.UNDECIDED for r in decided)
    assert sum(r["expected_conclusion"] == tr.UNDECIDED for r in ROWS) == 6


def test_articles_exist_in_docs():
    for r in ROWS:
        text = tb.doc_text(r["doc"])
        for a in r["expected_articles"]:
            assert f"第{a}条" in text, (r["id"], a)


def test_corpus_is_unused_in_earlier_evaluations():
    earlier = {p.name for d in ("evaluation/corpus", "evaluation/v2") for p in (REPO / d).rglob("*.md")
               if "teacher_check" not in str(p) and "results" not in str(p)}
    new = {p.name for p in tb.CORPUS.glob("*.md")}
    assert len(new) == 6 and not (new & earlier)


def test_three_valued_logic():
    assert tr.all3(True, None) is None and tr.all3(False, None) is False and tr.all3(True, True) is True
    assert tr.any3(False, None) is None and tr.any3(True, None) is True and tr.any3(False, False) is False
    assert tr.not3(None) is None


@pytest.mark.parametrize("facts,expected", [
    ({"amount": 2_000_000}, True),                               # 200万円ちょうどは目的が分からなくても範囲内
    ({"amount": 3_000_000}, None),                               # 住宅の購入か分からない
    ({"amount": 5_000_001, "housing_purchase": None}, False),    # どちらでも範囲外
])
def test_loan_limit_with_unknown_purpose(facts, expected):
    assert tr.loan_within_limit(facts) is expected


def test_party_size_with_unknown_department():
    assert tr.party_size_ok({"participants": 10}) is True
    assert tr.party_size_ok({"participants": 7}) is None
    assert tr.party_size_ok({"participants": 4}) is False


def test_full_years_counts_anniversary():
    from datetime import date
    assert tr.full_years(date(2017, 4, 2), date(2027, 4, 1)) == 9
    assert tr.full_years(date(2017, 4, 1), date(2027, 4, 1)) == 10


def test_grade_reads_last_lines():
    ans = "説明…\n根拠: 第三条、第5条\n結論: 必要"
    g = ts.grade(ans, "必要", [3])
    assert g["ok"] and g["articles"] == [3, 5]
    assert not ts.grade(ans, "不要", [3])["ok"]
    assert not ts.grade(ans, "必要", [4])["ok"]                       # 条が違う
    assert not ts.grade("結論: 必要", "必要", [3])["ok"]              # 根拠の行が無い
    assert ts.grade("根拠: 第１０条\n結論: 判断できない", "判断できない", [10])["ok"]
    assert ts.grade("根拠: 第十条\n結論: **判断できない**", "判断できない", [10])["ok"]
    assert not ts.grade(None, "必要", [3])["ok"]


def test_sealed_manifest_matches_files(tmp_path):
    m = tb.write_sealed(tmp_path)
    assert m["n"] == 96 and len(m["stage1_ids"]) == 12
    assert m["questions_sha256"] == tb.sha256((tmp_path / "questions.jsonl").read_bytes())
    rows = [json.loads(l) for l in (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [r["id"] for r in ROWS]


def test_blind_set_has_no_answers(tmp_path):
    tb.write_blind(tmp_path)
    text = (tmp_path / "questions_blind.jsonl").read_text(encoding="utf-8")
    for key in ("expected_conclusion", "facts", "category", "stage1", "expected_articles"):
        assert key not in text
    assert len(list((tmp_path / "corpus").glob("*.md"))) == 6


def test_prompt_gives_whole_doc_and_no_answer():
    r = ROWS[0]
    msgs = tb.prompt_messages(r)
    assert tb.doc_text(r["doc"]) in msgs[0]["content"]
    assert r["expected_conclusion"] not in msgs[0]["content"] or r["expected_conclusion"] in tb.doc_text(r["doc"])
    assert msgs[1]["content"] == r["question"]


def test_sealed_set_matches_current_build():
    # 封をした後に問・規程・指示文が 1 文字でも変われば落ちる
    sealed = tb.CORPUS.parent / "sealed"
    manifest = json.loads((sealed / "manifest.json").read_text(encoding="utf-8"))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS).encode("utf-8")
    assert tb.sha256(data) == manifest["questions_sha256"] == tb.sha256((sealed / "questions.jsonl").read_bytes())
    assert manifest["corpus_sha256"] == {p.name: tb.sha256(p.read_bytes()) for p in sorted(tb.CORPUS.glob("*.md"))}
    assert manifest["system_prompt_sha256"] == tb.sha256(tb.SYSTEM_PROMPT.encode("utf-8"))
