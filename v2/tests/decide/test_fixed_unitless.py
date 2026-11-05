"""計画 7.3 の固定の確認例（10/25 に凍結前の試験として固定。後から変えない）。

単位のない量「評価」（境目 5 以上・relation if_then）について、単位・日付・条番号を取り違えて決めない事。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from decide.engine import DecisionEngine
from decide.records import RecordError, build_record, sha256_text

LINES = [
    "# 評価規程",                              # 1
    "",                                        # 2
    "賞与は評価が5以上の社員に支給する。",     # 3
    "賞与は評価が5点以上の社員に支給する。",   # 4
]
DOC = "\n".join(LINES) + "\n"
PATH = "/watched/general/評価規程.md"
OPT = "最後の行には「結論: 支給される」または「結論: 支給されない」のどちらかだけを書いてください。"


class Docs:
    def lookup(self, document_id):
        return (PATH, sha256_text(DOC)) if document_id == PATH else None

    def read_text(self, doc_path):
        return DOC if doc_path == PATH else None


@dataclass(frozen=True)
class Ch:
    document_id: str = PATH
    content: str = "x"
    access_level: int = 1
    source: str = "評価規程.md"


def rec(line=3, quote="賞与は評価が5以上の社員に支給する"):
    assert quote in LINES[line - 1]
    e = {"quote": quote, "line": line}
    return {"rule_id": "R-score", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "d",
            "supported": True, "relation": "if_then",
            "subject": {"groups": [["賞与"]]},
            "facts": [{"role": "score", "label": "評価", "kind": "quantity", "unit": None, "cues": ["評価"]}],
            "conditions": [{"fact": "score", "op": ">=", "threshold": {"type": "number", "value": 5}, "evidence": e}],
            "outcomes": {"all_true": {"value": "支給される", "basis": "explicit", "evidence": e},
                         "any_false": {"value": "unknown", "basis": "unknown"}},
            "options": ["支給される", "支給されない"]}


def decide(q):
    docs = Docs()
    e = DecisionEngine([build_record(copy.deepcopy(rec()), docs)], docs)
    return e.decide("賞与について。" + q + OPT, [Ch()])


def test_unit_ful_quote_for_unitless_fact_is_rejected():
    with pytest.raises(RecordError):
        build_record(rec(line=4, quote="賞与は評価が5点以上の社員に支給する"), Docs())


@pytest.mark.parametrize("q", [
    "評価は5点です。",
    "評価は5メートルです。",
    "評価は 5 点です。",
    "評価が決まったのは2026年5月1日です。",
    "評価は第5条で決まる。",
])
def test_held(q):
    res = decide(q)
    assert res.kind == "held", getattr(res, "answer_text", res)


@pytest.mark.parametrize("q", [
    "第7条により、評価は5です。",
    "評価は5です。",
    "評価が5の社員です。",
])
def test_decided_with_5(q):
    res = decide(q)
    assert res.kind == "decided", getattr(res, "reason", res)
    assert res.answer_text.strip().split("\n")[-1] == "結論: 支給される"
