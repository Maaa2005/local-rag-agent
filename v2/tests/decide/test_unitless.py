"""単位のない量（「評価が5以上」の 5）の記録と取り出し。合成の文書だけを使う。

- 記録: unit: null と明示した量の事実は、引用の境目が単位なしで、直後の比べ方の語の向きが op と合えば作れる。境目に単位が付いていれば（「5点以上」）拒否
- 取り出し: 質問の数が単位なしの型（後ろが文の終わり・句読点・閉じ括弧・です／の／以上 等）なら使い、
  単位が付いている・型に合わない数は、手がかりと同じ節にあれば保留（facts:unit_mismatch）
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from decide.engine import DecisionEngine
from decide.records import RecordError, build_record, sha256_text

LINES = [
    "# 評価規程",                                  # 1
    "",                                            # 2
    "賞与は評価が5以上の社員に支給する。",         # 3
    "賞与は評価が5以下の社員に支給する。",         # 4
    "賞与は評価が5点以上の社員に支給する。",       # 5
    "賞与は評価5の社員に支給する。",               # 6
    "賞与は評価が5以上5以下の社員に支給する。",    # 7
    "勤続3年以上の社員は住宅手当の対象とする。",   # 8
    "賞与の評価が5以上の社員には、3年以内の勤続を問わず支給する。",  # 9
    "賞与は評価が5メートル以上の社員に支給する。",  # 10
]
DOC = "\n".join(LINES) + "\n"
PATH = "/watched/general/評価規程.md"
OPT = "最後の行には「結論: {a}」または「結論: {b}」のどちらかだけを書いてください。"


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


def ev(line, quote):
    assert quote in LINES[line - 1]
    return {"quote": quote, "line": line}


SCORE = {"role": "score", "label": "評価", "kind": "quantity", "unit": None, "cues": ["評価"]}


def rec(line=3, quote="賞与は評価が5以上の社員に支給する", op=">=", value=5, facts=None, **over):
    e = ev(line, quote)
    d = {"rule_id": "R-score", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "d", "supported": True,
         "relation": "if_else",  # 取り出しの試験用に両側へ結論を持たせる合成の記録
         "subject": {"groups": [["賞与"], ["支給"]]}, "facts": facts or [SCORE],
         "conditions": [{"fact": "score", "op": op, "threshold": {"type": "number", "value": value}, "evidence": e}],
         "outcomes": {"all_true": {"value": "支給される", "basis": "explicit", "evidence": e},
                      "any_false": {"value": "支給されない", "basis": "explicit", "evidence": e}},
         "options": ["支給される", "支給されない"]}
    d.update(over)
    return d


def decide(q, data=None):
    docs = Docs()
    e = DecisionEngine([build_record(copy.deepcopy(data or rec()), docs)], docs)
    return e.decide(q + OPT.format(a="支給される", b="支給されない"), [Ch()])


def last(res):
    assert res.kind == "decided", getattr(res, "reason", res)
    return res.answer_text.strip().split("\n")[-1]


# ---------------------------------------------------------------- 記録

def test_unitless_record_loads():
    r = build_record(rec(), Docs())
    assert not r.stale and r.fact("score").unit == ""


def test_unitless_record_rejects_unit_attached_quote():
    # 変更（計画 7.3）: 旧は「5点以上」の引用でも作れた。今は単位のない量の境目に単位が付いていれば拒否する
    with pytest.raises(RecordError, match="単位"):
        build_record(rec(line=5, quote="賞与は評価が5点以上の社員に支給する"), Docs())
    # 向きの合う引用でも、単位として読めない語が続く数（5メートル以上）は単位付きとして拒否
    with pytest.raises(RecordError, match="単位"):
        build_record(rec(line=10, quote="賞与は評価が5メートル以上の社員に支給する"), Docs())


def test_unitless_boundary_with_other_comparison_word():
    assert build_record(rec(line=4, quote="賞与は評価が5以下の社員に支給する", op="<="), Docs())


@pytest.mark.parametrize("line,quote,op", [
    (4, "賞与は評価が5以下の社員に支給する", ">="),     # 語の向きが逆
    (3, "賞与は評価が5以上の社員に支給する", "<="),
    (3, "賞与は評価が5以上の社員に支給する", ">"),     # 以上は >=
    (6, "賞与は評価5の社員に支給する", ">="),           # 比べ方の語が無い
])
def test_unitless_record_rejects_wrong_or_missing_comparison_word(line, quote, op):
    with pytest.raises(RecordError):
        build_record(rec(line=line, quote=quote, op=op), Docs())


def test_unitless_record_rejects_value_not_in_quote():
    with pytest.raises(RecordError):
        build_record(rec(value=6), Docs())


def test_same_value_twice_in_quote_must_all_agree():
    q = "賞与は評価が5以上5以下の社員に支給する"
    for op in (">=", "<="):
        with pytest.raises(RecordError):
            build_record(rec(line=7, quote=q, op=op), Docs())


def test_quantity_without_unit_key_is_rejected_but_null_is_explicit():
    f = {k: v for k, v in SCORE.items() if k != "unit"}
    with pytest.raises(RecordError):
        build_record(rec(facts=[f]), Docs())
    assert build_record(rec(facts=[{**SCORE, "unit": ""}]), Docs())


def test_unitful_fact_still_needs_unit_in_quote():
    # 単位 点 の事実で、引用が「5以上」（単位なし）なら拒否のまま。引用が「5点以上」なら通る
    with pytest.raises(RecordError):
        build_record(rec(facts=[{**SCORE, "unit": "点"}]), Docs())
    assert build_record(rec(line=5, quote="賞与は評価が5点以上の社員に支給する", facts=[{**SCORE, "unit": "点"}]), Docs())


# ---------------------------------------------------------------- 取り出し・結論

@pytest.mark.parametrize("q,want", [
    ("評価は5です。賞与は支給されますか？", "支給される"),
    ("評価は5の社員の賞与は支給されますか？", "支給される"),
    ("評価が6の社員の賞与は支給されますか？", "支給される"),
    ("評価は4です。賞与は支給されますか？", "支給されない"),
    ("評価が4.5の社員の賞与は支給されますか？", "支給されない"),
])
def test_unitless_value_decides(q, want):
    res = decide(q)
    assert last(res) == f"結論: {want}"
    assert res.decision_record["extracted"][0]["value"] in ("5", "6", "4", "4.5")
    assert "5点" not in res.answer_text.split("\n")[-1]


@pytest.mark.parametrize("q", [
    "評価は5点です。賞与は支給されますか？",
    "評価が4点の社員の賞与は支給されますか？",
    "評価は5ヶ月です。賞与は支給されますか？",
])
def test_unit_attached_question_value_is_held_as_unit_mismatch(q):
    res = decide(q)
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:unit_mismatch"


def test_missing_value_is_held():
    res = decide("賞与は支給されますか？評価は分かりません。")
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:missing"


def test_two_different_unitless_values_conflict():
    res = decide("評価は4でした。評価は5に上がりました。賞与は支給されますか？")
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:conflict"


def test_unrelated_unit_in_other_clause_is_not_a_mismatch():
    res = decide("勤続3年です。評価は5です。賞与は支給されますか？")
    assert last(res) == "結論: 支給される"


def test_numbers_inside_dates_and_ordinals_are_not_values():
    res = decide("2026年9月1日時点で、第3条の対象です。評価は4です。賞与は支給されますか？")
    assert last(res) == "結論: 支給されない"


def test_unitful_records_ignore_bare_numbers():
    # 単位のない数を候補に入れるのは、単位のない量の事実がある記録だけ
    from decide import parse
    assert [c.kind for c in parse.extract("評価は5、勤続3年").cands] == ["年"]
    assert [c.kind for c in parse.extract("評価は5、勤続3年", bare=True).cands] == ["", "年"]


# ---------------------------------------------------------------- 質問から取る数の型（計画 7.3）

def test_bare_number_forms_accepted_and_rejected():
    from decide import parse
    kinds = lambda q: [c.kind for c in parse.extract(q, bare=True).cands]  # noqa: E731
    for q in ("評価は5です", "評価は5。", "評価は5", "評価は5、", "（評価5）", "評価は5 です", "評価5以上", "評価5未満", "評価が5より上",
              "評価は5を超える", "評価は5と6", "評価は5だ", "評価は5に上がった", "評価5の人", "評価5が", "評価5は"):
        assert kinds(q)[:1] == [""], q
    for q in ("評価は5メートルです", "評価は 5 点です", "評価は5段階です", "評価は5条で決まる"):
        assert kinds(q) == [parse.UNKNOWN_UNIT], q
    # 日付・第N条・条の枝番は値として使わない
    assert kinds("評価が決まったのは2026年5月1日です") == ["date"]
    assert kinds("第7条により、評価は5です") == [""]
    assert kinds("第7条の2により") == []


def test_number_with_unit_in_other_clause_is_ignored_but_same_clause_holds():
    assert last(decide("勤続3メートルです。評価は5です。賞与は支給されますか？")) == "結論: 支給される"
    res = decide("評価は5メートルです。賞与は支給されますか？")
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:unit_mismatch"
    res = decide("評価は 5 点です。賞与は支給されますか？")
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:unit_mismatch"
