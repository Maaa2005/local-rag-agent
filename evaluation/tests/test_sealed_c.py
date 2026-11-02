"""封 C（evaluation/distill/sealed_c）のテスト。

確かめること:
- 24 問・区分 12/4/4/4・正解は選択肢の中・境目の 3 つ組は結論が変わる所をまたぐ
- 判断できない問は 4 問だけ・根拠の条は規程に実在する
- 題材は封 A・封 B・今までの評価の文書と重ならない
- 封の後に問・規程が 1 文字でも変われば落ちる
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.distill.sealed_c import build as cb
from evaluation.distill.sealed_c.questions import build
from evaluation.teacher_check.rules import UNDECIDED

ROWS = build()
REPO = Path(__file__).resolve().parents[2]


def test_counts():
    assert len(ROWS) == 24 and len({r["id"] for r in ROWS}) == 24
    assert Counter(r["category"] for r in ROWS) == {"boundary": 12, "paraphrase": 4, "exception": 4, "undecidable": 4}


def test_expected_in_options_and_footer():
    for r in ROWS:
        assert r["expected_conclusion"] in r["options"]
        for opt in r["options"]:
            assert f"「{opt}」" in r["question"]


def test_boundary_triples_cross():
    rows = [r for r in ROWS if r["category"] == "boundary"]
    for i in range(0, 12, 3):
        assert len({r["expected_conclusion"] for r in rows[i:i + 3]}) == 2


def test_undecidable_only_in_its_category():
    assert {r["id"] for r in ROWS if r["expected_conclusion"] == UNDECIDED} == \
           {r["id"] for r in ROWS if r["category"] == "undecidable"}


def test_articles_exist():
    for r in ROWS:
        for a in r["expected_articles"]:
            assert f"第{a}条" in cb.doc_text(r["doc"])


def test_domains_unused_elsewhere():
    earlier = {p.name for d in ("evaluation/corpus", "evaluation/v2", "evaluation/teacher_check/corpus")
               for p in (REPO / d).rglob("*.md") if "results" not in str(p)}
    mine = {p.name for p in cb.CORPUS.glob("*.md")}
    assert len(mine) == 4 and not (mine & earlier)


def test_sealed_matches_build():
    sealed = cb.CORPUS.parent / "sealed"
    if not (sealed / "manifest.json").exists():
        return   # 封の前
    m = json.loads((sealed / "manifest.json").read_text(encoding="utf-8"))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS).encode("utf-8")
    assert cb.sha256(data) == m["questions_sha256"] == cb.sha256((sealed / "questions.jsonl").read_bytes())
    assert m["corpus_sha256"] == {p.name: cb.sha256(p.read_bytes()) for p in sorted(cb.CORPUS.glob("*.md"))}
