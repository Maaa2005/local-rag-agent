"""decide.records: 構造の検査と、規程の本文との照合（F7 の機械検査）。

分かる事: 構造の違反・引用が本文に一字一句在るか・境目の数と比べ方の向きが引用と合うか・版。
分からない事（既知の穴。test_reversed_outcome_record_loads_known_gap で明示）:
結論側の取り違え、対象や手がかりの語の取り違え、結論の値の肯定・否定の取り違え。
"""
from __future__ import annotations

import copy
import json

import pytest

from decide.records import MAX_CONDITIONS, RecordError, build_record, load_records, parse_record, sha256_text

DOC = "\n".join([
    "# 購買規程",                                            # 1
    "",                                                      # 2
    "10万円を超える購入は部長の承認を要する。",              # 3
    "出張に伴う旅費は別の規程による。",                      # 4
    "",                                                      # 5
    "購入日の翌日から数えて14日を過ぎた申請は受け付けない。",  # 6
    "10万円未満は課長が承認し、10万円以上は部長が承認する。",  # 7
    "翌月10日を過ぎた申請は原則として受理しない。",          # 8
    "人間ドックの補助は、2026年12月1日より前に受診した分を対象とする。",  # 9
    "勤続3年以上の社員は住宅手当の対象とする。",            # 10
]) + "\n"
PATH = "/watched/general/購買規程.md"


class Source:
    def __init__(self, text=DOC):
        self.text = text

    def read_text(self, doc_path):
        return self.text if doc_path == PATH else None


def ev(line, quote=None):
    return {"quote": quote if quote is not None else DOC.split("\n")[line - 1].rstrip("。"), "line": line}


def money_record(**over):
    d = {
        "rule_id": "R-money", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "d", "supported": True,
        "relation": "if_else",  # 両側に結論を持つ合成の記録
        "subject": {"groups": [["購入"], ["承認"]]},
        "facts": [{"role": "amount", "label": "金額", "kind": "money", "cues": ["購入"]}],
        "conditions": [{"fact": "amount", "op": ">", "threshold": {"type": "number", "value": 100000},
                        "evidence": ev(3, "10万円を超える購入は部長の承認を要する")}],
        "outcomes": {"all_true": {"value": "必要", "basis": "explicit", "evidence": ev(3, "10万円を超える購入は部長の承認を要する")},
                     "any_false": {"value": "不要", "basis": "explicit",
                                   "evidence": ev(3, "10万円を超える購入は部長の承認を要する")}},
        "options": ["必要", "不要"],
    }
    d.update(over)
    return d


def test_valid_record_loads_and_validates():
    rec = build_record(money_record(), Source())
    assert rec.rule_id == "R-money" and not rec.stale and len(rec.conditions) == 1


def test_all_shipped_records_load_against_their_documents():
    from pathlib import Path

    from decide.engine import FileDocLookup
    root = Path(__file__).resolve().parents[3]
    lookup = FileDocLookup([root / "evaluation/corpus", root / "evaluation/v2/corpus_nx_dev"])
    recs = load_records(root / "v2/decide/records", lookup)
    assert len(recs) >= 12 and not [r.rule_id for r in recs if r.stale]


# ---- 構造の拒否

def test_more_than_three_conditions_rejected():
    d = money_record()
    d["conditions"] = [copy.deepcopy(d["conditions"][0]) for _ in range(MAX_CONDITIONS + 1)]
    with pytest.raises(RecordError):
        parse_record(d)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(or_=[1]),                                       # 知らない欄
    lambda d: d.update(**{"or": [{"x": 1}]}),                          # 「または」
    lambda d: d["conditions"][0].update(any_of=[1]),                   # 条件の中の知らない欄
    lambda d: d["conditions"][0].update(conditions=[{"fact": "amount"}]),  # 入れ子
    lambda d: d["conditions"][0].update(op="!="),                      # 知らない比べ方
    lambda d: d["conditions"][0].update(op="~"),
    lambda d: d["conditions"][0]["threshold"].update(type="regex"),    # 知らない境目の型
    lambda d: d["conditions"][0]["threshold"].update(value="100000"),  # 数でない
    lambda d: d["facts"][0].update(kind="text"),
    lambda d: d["outcomes"]["all_true"].update(basis="guess"),
    lambda d: d["outcomes"]["all_true"].update(value="どちらでもない"),  # options に無い
    lambda d: d["options"].append("保留"),                             # 2 つでない
    lambda d: d["subject"].update(groups=[]),
    lambda d: d.update(supported="yes"),
])
def test_structural_violations_rejected(mutate):
    d = money_record()
    mutate(d)
    with pytest.raises(RecordError):
        parse_record(d)


def test_duplicate_rule_id_rejected(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(json.dumps(money_record(), ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RecordError):
        load_records(tmp_path, Source())


def test_load_records_reads_only_json(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(money_record(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "ignored.jsonl").write_text("{broken", encoding="utf-8")
    assert [r.rule_id for r in load_records(tmp_path, Source())] == ["R-money"]


# ---- relation（計画 7.1）

def _with_sides(relation, all_true, any_false):
    d = money_record(relation=relation)
    e = ev(3, "10万円を超える購入は部長の承認を要する")
    d["outcomes"] = {
        "all_true": {"value": "必要", "basis": "explicit", "evidence": e} if all_true else {"value": "unknown", "basis": "unknown"},
        "any_false": {"value": "不要", "basis": "explicit", "evidence": e} if any_false else {"value": "unknown", "basis": "unknown"}}
    return d


@pytest.mark.parametrize("relation,all_true,any_false,ok", [
    ("if_then", True, False, True), ("if_then", False, True, False), ("if_then", True, True, False),
    ("if_then_not", True, False, True), ("if_then_not", False, True, False), ("if_then_not", True, True, False),
    ("only_if", False, True, True), ("only_if", True, False, False), ("only_if", True, True, False),
    ("if_else", True, True, True), ("if_else", True, False, True), ("if_else", False, True, True),
])
def test_relation_allows_only_its_sides(relation, all_true, any_false, ok):
    d = _with_sides(relation, all_true, any_false)
    if ok:
        assert parse_record(d).relation == relation
    else:
        with pytest.raises(RecordError, match="relation"):
            parse_record(d)
        with pytest.raises(RecordError, match="relation"):
            build_record(d, Source())  # 文書の照合より前（構造の段）で止まる


def test_both_sides_unknown_is_allowed_for_every_relation():
    for rel in ("if_then", "if_then_not", "only_if", "if_else"):
        assert parse_record(_with_sides(rel, False, False)).all_true.value == "unknown"


def test_inferred_negation_basis_is_rejected():
    d = money_record()
    d["outcomes"]["any_false"]["basis"] = "inferred_negation"
    with pytest.raises(RecordError, match="basis"):
        parse_record(d)


def test_supported_record_without_relation_is_rejected_and_unknown_relation_too():
    d = money_record()
    del d["relation"]
    with pytest.raises(RecordError, match="relation"):
        parse_record(d)
    with pytest.raises(RecordError, match="relation"):
        parse_record(money_record(relation="only_then"))


def test_unsupported_record_needs_no_relation_but_a_stated_outcome_does():
    d = money_record(supported=False, scope_reason="対応外", conditions=[], outcomes={})
    del d["relation"]
    assert parse_record(d).relation is None
    d = _with_sides("if_then", True, False)
    d.update(supported=False, scope_reason="対応外")
    del d["relation"]
    with pytest.raises(RecordError, match="relation"):
        parse_record(d)


# ---- 文書との照合（F7）

def test_quote_not_verbatim_rejected():
    d = money_record()
    d["conditions"][0]["evidence"] = ev(3, "10万円を超える購入は部長の承認が必要")
    with pytest.raises(RecordError, match="そのまま無い"):
        build_record(d, Source())


def test_quote_on_wrong_line_rejected():
    d = money_record()
    d["conditions"][0]["evidence"] = {"quote": "10万円を超える購入は部長の承認を要する", "line": 4}
    with pytest.raises(RecordError):
        build_record(d, Source())
    d["conditions"][0]["evidence"] = {"quote": "10万円を超える購入は部長の承認を要する", "line": 99}
    with pytest.raises(RecordError):
        build_record(d, Source())


def test_wrong_number_caught():
    d = money_record()
    d["conditions"][0]["threshold"]["value"] = 200000  # 引用は 10万円
    with pytest.raises(RecordError):
        build_record(d, Source())


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_wrong_comparison_direction_caught(op):
    d = money_record()
    d["conditions"][0]["op"] = op  # 引用は「超える」= >
    with pytest.raises(RecordError, match="言い回し"):
        build_record(d, Source())


def test_threshold_token_with_conflicting_phrases_caught():
    # 引用に 10万円 が 2 回在り、「未満」と「以上」。全部の言い回しが比べ方と一致しなければならない
    q = "10万円未満は課長が承認し、10万円以上は部長が承認する"
    for op in (">=", "<"):
        d = money_record()
        d["conditions"][0].update(op=op, evidence=ev(7, q))
        with pytest.raises(RecordError):
            build_record(d, Source())


def test_comparison_word_missing_from_quote_caught():
    d = money_record()
    d["conditions"][0]["evidence"] = ev(3, "10万円の購入は部長の承認を要する")  # 本文には無い言い方（引用の不一致）
    with pytest.raises(RecordError):
        build_record(d, Source())
    d = money_record()
    d["conditions"][0]["evidence"] = ev(3, "購入は部長の承認を要する")  # 本文の一部だが境目の数が無い
    with pytest.raises(RecordError, match="境目の語"):
        build_record(d, Source())


def test_offset_days_requires_next_day_phrase_in_quote():
    fact_d = {"role": "p", "label": "購入日", "kind": "date", "cues": ["購入"]}
    fact_s = {"role": "s", "label": "申請日", "kind": "date", "cues": ["申請"]}
    cond = {"fact": "s", "op": ">", "threshold": {"type": "offset_days", "base": "p", "days": 14, "count_from": "next_day"},
            "evidence": ev(6, "購入日の翌日から数えて14日を過ぎた申請は受け付けない")}
    d = money_record(facts=[fact_d, fact_s], conditions=[cond])
    assert build_record(d, Source()).rule_id == "R-money"
    # 「翌日から数えて」の無い引用に対して next_day と書いたら拒否
    doc2 = DOC.replace("購入日の翌日から数えて14日を過ぎた", "購入日から14日を過ぎた")
    d = money_record(facts=[fact_d, fact_s], conditions=[{**cond, "evidence": ev(6, "購入日から14日を過ぎた申請は受け付けない")}],
                     doc_version=sha256_text(doc2))
    with pytest.raises(RecordError, match="翌日から数えて"):
        build_record(d, Source(doc2))


def test_month_day_and_date_thresholds_checked():
    f_e = {"role": "e", "label": "発生日", "kind": "date", "cues": ["発生"]}
    f_s = {"role": "s", "label": "申請日", "kind": "date", "cues": ["申請"]}
    good = {"fact": "s", "op": ">", "threshold": {"type": "month_day", "base": "e", "month_offset": 1, "day": 10},
            "evidence": ev(8, "翌月10日を過ぎた申請は原則として受理しない")}
    assert build_record(money_record(facts=[f_e, f_s], conditions=[good]), Source())
    for patch in ({"op": "<="}, {"threshold": {"type": "month_day", "base": "e", "month_offset": 1, "day": 20}}):
        with pytest.raises(RecordError):
            build_record(money_record(facts=[f_e, f_s], conditions=[{**good, **patch}]), Source())
    f_x = {"role": "x", "label": "受診日", "kind": "date", "cues": ["受診"]}
    date_c = {"fact": "x", "op": "<", "threshold": {"type": "date", "value": "2026-12-01"},
              "evidence": ev(9, "2026年12月1日より前に受診した分を対象とする")}
    assert build_record(money_record(facts=[f_x], conditions=[date_c]), Source())
    for patch in ({"op": "<="}, {"threshold": {"type": "date", "value": "2026-12-02"}}):
        with pytest.raises(RecordError):
            build_record(money_record(facts=[f_x], conditions=[{**date_c, **patch}]), Source())


def test_quantity_threshold_unit_must_match_quote():
    f = {"role": "t", "label": "勤続", "kind": "quantity", "unit": "年", "cues": ["勤続"]}
    c = {"fact": "t", "op": ">=", "threshold": {"type": "number", "value": 3},
         "evidence": ev(10, "勤続3年以上の社員は住宅手当の対象とする")}
    assert build_record(money_record(facts=[f], conditions=[c]), Source())
    with pytest.raises(RecordError):  # 年でなく日の記録にすると引用に 3日 が無い
        build_record(money_record(facts=[{**f, "unit": "日"}], conditions=[c]), Source())


def test_wrong_version_loads_as_stale_without_document_check():
    d = money_record(doc_version="0" * 64)
    d["conditions"][0]["threshold"]["value"] = 200000  # 版が違えば引用との照合はしない（engine が版で保留する）
    rec = build_record(d, Source())
    assert rec.stale
    assert build_record(money_record(), Source("# 別の文書\n")).stale  # 本文が違う
    assert build_record(money_record(), None).stale  # 文書が引けない


def test_reversed_outcome_record_loads_known_gap():
    """既知の穴: 結論の向きを取り違えた記録（成り立つ時に「不要」）は F7 の機械検査を通る。
    これを止めるのは人の確認（作問役とは別の採点）で、この検査では止まらない。"""
    d = money_record()
    d["outcomes"]["all_true"]["value"], d["outcomes"]["any_false"]["value"] = "不要", "必要"
    rec = build_record(d, Source())
    assert rec.all_true.value == "不要" and not rec.stale


def test_wrong_subject_or_cues_load_known_gap():
    """既知の穴: 対象の語・手がかりの語の取り違えも、この検査では見つからない。"""
    d = money_record(subject={"groups": [["旅費"], ["精算"]]})
    d["facts"][0]["cues"] = ["無関係な語"]
    assert not build_record(d, Source()).stale
