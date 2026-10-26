"""境目に名前を付ける言い方（「…を〈名前〉とし、申請日が〈名前〉以前の場合は…」）の検査と判定。

対象の形:
  (a) 「購入日の翌月12日を期限日とし、…」 … 翌月N日を名前にする（month_day）
  (b) 「破損発生日の翌日を数え始めとして10日目を期限日とし、…」 … 基準日 + N 日（offset_days）
  (c) 「結果通知日の翌日を数え始めとして7日目を受付開始日とし、…以後／それより前」 … 開始日の形
規程の文書は、この試験の中で作った架空の 1 本。.jsonl・開発用の問は開かない。
"""
from __future__ import annotations

import copy

import pytest

from decide import parse
from decide.engine import DecisionEngine
from decide.records import RecordError, build_record, sha256_text

LINES = [
    "# 申請規程",                                                                                   # 1
    "",                                                                                             # 2
    "購入日の翌月12日を期限日とし、申請日が期限日以前の場合は受付可、期限日を過ぎた場合は受付不可とする。",  # 3
    "返品は、購入日の翌月15日を期限日とし、申請日が期限日以前の場合は受付可、期限日を過ぎた場合は受付不可とする。",  # 4
    "破損発生日の翌日を数え始めとして10日目を期限日とし、報告日が期限日を過ぎた場合は原則として受理不可、期限日以前の場合は受理可とする。",  # 5
    "結果通知日の翌日を数え始めとして7日目を受付開始日とし、再審査申請日が受付開始日以後の場合は申請可、それより前の場合は申請不可とする。",  # 6
    "購入日の翌月12日を期限日とし、申請日が期限日以前の場合は受付可、期限日以前の場合は受付不可とする。",  # 7  同じ比べ方が両側
    "購入日の翌月12日を期限日とし、申請日が期限日以前の場合は受付可、期限日以後の場合は受付不可とする。",  # 8  反対ではない組
    "申請日が期限日以前の場合は受付可とし、購入日の翌月12日を期限日とする。",                       # 9  名前が定義より前
    "購入日の翌月12日を期限日とし、申請日が締切日以前の場合は受付可とする。",                       # 10 名前が違う
    "購入日の翌月12日を期限日とし、申請日が期限日の場合は受付可とする。",                           # 11 名前の直後に比べ方が無い
    "破損発生日の翌日を数え始めとして10日目を過ぎた報告は受理不可とする。",                       # 12 形 (b) に名前が無い
    "購入日の翌月12日をもって、申請日が期限日以前の場合は受付可とする。",                           # 13 名付けが無い
    "破損発生日の翌日を数え始めとして10日目を期限日とし、報告日が締切日を過ぎた場合は受理不可とする。",  # 14 形(b)で名前が違う
    "結果通知日の翌日を数え始めとして7日目を受付開始日とし、再審査申請日が受付日以後の場合は申請可とする。",  # 15 形(c)で名前が違う
    "破損発生日の翌日を数え始めとして10日目を期限日とし、報告日が期限日の場合は受理不可とする。",    # 16 形(b)で名前の直後に比べ方が無い
]
DOC = "\n".join(LINES) + "\n"
PATH = "/watched/general/申請規程.md"
OPT_RECEIPT = "最後の行には「結論: 受付可」または「結論: 受付不可」のどちらかだけを書いてください。"
OPT_ACCEPT = "最後の行には「結論: 受理可」または「結論: 受理不可」のどちらかだけを書いてください。"
OPT_APPLY = "最後の行には「結論: 申請可」または「結論: 申請不可」のどちらかだけを書いてください。"

Q3 = LINES[2].rstrip("。")
Q4 = LINES[3].rstrip("。")
Q5 = LINES[4].rstrip("。")
Q6 = LINES[5].rstrip("。")


def ev(line, quote=None):
    return {"quote": quote if quote is not None else LINES[line - 1].rstrip("。"), "line": line}


class Docs:
    def __init__(self, text=DOC):
        self.text = text

    def lookup(self, document_id):
        return (PATH, sha256_text(self.text)) if document_id == PATH else None

    def read_text(self, doc_path):
        return self.text if doc_path == PATH else None


class Ch:
    def __init__(self, document_id):
        self.document_id = document_id
        self.content = "x"
        self.access_level = 1
        self.source = "申請規程.md"


CHUNKS = [Ch(PATH)]


def record(rule_id, groups, facts, cond, a, b, options, line, quote=None, **extra):
    q = ev(line, quote)
    d = {"rule_id": rule_id, "doc_path": PATH, "doc_version": sha256_text(DOC), "description": rule_id, "supported": True,
         "relation": "if_else", "subject": {"groups": groups}, "facts": facts,
         "conditions": [{**cond, "evidence": q}],
         "outcomes": {"all_true": {"value": a, "basis": "explicit", "evidence": q},
                      "any_false": {"value": b, "basis": "explicit", "evidence": q}},
         "options": options}
    d.update(extra)
    return d


F_BUY = {"role": "p", "label": "購入日", "kind": "date", "cues": ["購入"]}
F_APPLY = {"role": "s", "label": "申請日", "kind": "date", "cues": ["申請"]}


def month_day_rec(rule_id, day, line, op="<=", quote=None):
    return record(rule_id, [["返品"], ["受付"]], [F_BUY, F_APPLY],
                  {"fact": "s", "op": op, "threshold": {"type": "month_day", "base": "p", "month_offset": 1, "day": day}},
                  "受付可", "受付不可", ["受付可", "受付不可"], line, quote)


F_DMG = {"role": "d", "label": "破損発生日", "kind": "date", "cues": ["破損"]}
F_REP = {"role": "r", "label": "報告日", "kind": "date", "cues": ["報告"]}
F_NOTICE = {"role": "n", "label": "結果通知日", "kind": "date", "cues": ["通知"]}
F_REAPPLY = {"role": "a", "label": "再審査申請日", "kind": "date", "cues": ["申請"]}


def damage_rec(op=">", a="受理不可", b="受理可", quote=None, line=5):
    return record("R-damage", [["破損"], ["受理"]], [F_DMG, F_REP],
                  {"fact": "r", "op": op, "threshold": {"type": "offset_days", "base": "d", "days": 10, "count_from": "next_day"}},
                  a, b, ["受理可", "受理不可"], line, quote)


def start_rec(op=">=", quote=None):
    return record("R-start", [["再審査"], ["申請"]], [F_NOTICE, F_REAPPLY],
                  {"fact": "a", "op": op, "threshold": {"type": "offset_days", "base": "n", "days": 7, "count_from": "next_day"}},
                  "申請可", "申請不可", ["申請可", "申請不可"], 6, quote)


def decide(q, *datas, docs=None):
    docs = docs or Docs()
    return DecisionEngine([build_record(copy.deepcopy(d), docs) for d in datas], docs).decide(q, CHUNKS)


def last(res):
    assert res.kind == "decided", (res.kind, getattr(res, "reason", None), res.decision_record)
    return res.answer_text.strip().split("\n")[-1]


def refund_q(buy, apply_):
    return f"{buy}に商品を購入しました。返品を{apply_}に申請します。受付されますか？" + OPT_RECEIPT


# ---------------------------------------------------------------- 翌月N日を名前にする（月日）

@pytest.mark.parametrize("day,line", [(12, 3), (15, 4)])
def test_month_day_named_boundary_loads(day, line):
    assert build_record(month_day_rec("R-ret", day, line), Docs()).rule_id == "R-ret"


@pytest.mark.parametrize("apply_,want", [
    ("2026年10月11日", "受付可"), ("2026年10月12日", "受付可"), ("2026年10月13日", "受付不可"),
])
def test_month_day_named_boundary_12(apply_, want):
    assert last(decide(refund_q("2026年9月3日", apply_), month_day_rec("R-ret", 12, 3))) == f"結論: {want}"


@pytest.mark.parametrize("apply_,want", [
    ("2026年10月14日", "受付可"), ("2026年10月15日", "受付可"), ("2026年10月16日", "受付不可"),
])
def test_month_day_named_boundary_15(apply_, want):
    assert last(decide(refund_q("2026年9月3日", apply_), month_day_rec("R-ret", 15, 4))) == f"結論: {want}"


def test_month_day_named_boundary_year_end_and_missing_date_held():
    r = month_day_rec("R-ret", 12, 3)
    assert last(decide(refund_q("2026年12月20日", "2027年1月12日"), r)) == "結論: 受付可"
    assert last(decide(refund_q("2026年12月20日", "2027年1月13日"), r)) == "結論: 受付不可"
    res = decide("2026年9月3日に商品を購入しました。返品は受付されますか？" + OPT_RECEIPT, r)
    assert res.kind == "held"


def test_month_day_named_one_sided_quote():
    # 名付けから最初の比べ方の語までを引用する書き方（引用は本文の連続した一続きでなければならない）
    one = "購入日の翌月12日を期限日とし、申請日が期限日以前の場合は受付可"
    assert build_record(month_day_rec("R-ret", 12, 3, quote=one), Docs())
    with pytest.raises(RecordError):  # 片側の引用に反対の比べ方を書いたら拒否
        build_record(month_day_rec("R-ret", 12, 3, op=">", quote=one), Docs())
    with pytest.raises(RecordError):  # 補集合でない比べ方
        build_record(month_day_rec("R-ret", 12, 3, op="<", quote=one), Docs())


# ---------------------------------------------------------------- 「翌日を数え始めとしてN日目」（期限）

def dmg_q(dmg, rep):
    return f"{dmg}に機器が破損しました。{rep}に報告します。受理されますか？" + OPT_ACCEPT


def test_start_count_tokens():
    assert parse.start_count_tokens(Q5) == [(Q5.index("10日目") + len("10日目"), 10)]
    assert parse.start_count_tokens("翌日から数えて10日") == []


@pytest.mark.parametrize("rep,want", [
    ("2026年9月12日", "受理可"), ("2026年9月13日", "受理可"), ("2026年9月14日", "受理不可"),
])
def test_start_count_deadline(rep, want):
    assert last(decide(dmg_q("2026年9月3日", rep), damage_rec())) == f"結論: {want}"


def test_start_count_deadline_crosses_month_and_year():
    r = damage_rec()
    assert last(decide(dmg_q("2026年12月25日", "2027年1月4日"), r)) == "結論: 受理可"
    assert last(decide(dmg_q("2026年12月25日", "2027年1月5日"), r)) == "結論: 受理不可"
    assert last(decide(dmg_q("2028年2月20日", "2028年3月1日"), r)) == "結論: 受理可"  # うるう年
    assert last(decide(dmg_q("2028年2月20日", "2028年3月2日"), r)) == "結論: 受理不可"


def test_start_count_with_genzoku_between_reference_and_outcome():
    # 「原則として」が比べ方の語と結論の語の間に在っても読める
    assert build_record(damage_rec(), Docs())
    assert "原則として" in Q5


# ---------------------------------------------------------------- 開始日の形（以後／それより前）

def start_q(notice, apply_):
    return f"{notice}に結果が通知されました。再審査を{apply_}に申請します。申請できますか？" + OPT_APPLY


@pytest.mark.parametrize("apply_,want", [
    ("2026年9月9日", "申請不可"), ("2026年9月10日", "申請可"), ("2026年9月11日", "申請可"),
])
def test_start_date_form(apply_, want):
    assert last(decide(start_q("2026年9月3日", apply_), start_rec())) == f"結論: {want}"


def test_start_date_form_one_sided_quote():
    assert build_record(start_rec(), Docs())
    one = "結果通知日の翌日を数え始めとして7日目を受付開始日とし、再審査申請日が受付開始日以後の場合は申請可"
    assert build_record(start_rec(quote=one), Docs())
    with pytest.raises(RecordError):  # 以後（>=）の側の引用に < と書いたら拒否
        build_record(start_rec(op="<", quote=one), Docs())


# ---------------------------------------------------------------- 拒否

@pytest.mark.parametrize("day,line", [(12, 3), (15, 4)])
def test_reversed_comparison_month_day_rejected(day, line):
    with pytest.raises(RecordError, match="結論"):
        build_record(month_day_rec("R-ret", day, line, op=">"), Docs())


def test_reversed_comparison_with_full_quote_rejected():
    # 引用は両側が書かれていて、(<=, >) の組は (>, <=) と同じに見える。結論との組み合わせで取り違えを拒否する
    with pytest.raises(RecordError, match="結論"):
        build_record(damage_rec(op="<=", a="受理不可", b="受理可"), Docs())
    with pytest.raises(RecordError, match="結論"):
        build_record(start_rec(op="<"), Docs())
    # 結論も一緒に入れ替えた記録は同じ意味なので通る
    assert build_record(damage_rec(op="<=", a="受理可", b="受理不可"), Docs())


@pytest.mark.parametrize("op", [">=", "<"])
@pytest.mark.parametrize("day,line", [(12, 3), (15, 4)])
def test_non_complementary_comparison_rejected(op, day, line):
    with pytest.raises(RecordError, match="どちらでもない"):
        build_record(month_day_rec("R-ret", day, line, op=op), Docs())


@pytest.mark.parametrize("op", [">=", "<"])
def test_non_complementary_comparison_rejected_for_start_count(op):
    with pytest.raises(RecordError, match="どちらでもない"):
        build_record(damage_rec(op=op), Docs())
    other = {">=": "<=", "<": ">"}[op]  # 開始日の形（以後 / それより前）にも同じ検査
    with pytest.raises(RecordError, match="どちらでもない"):
        build_record(start_rec(op=other), Docs())


def test_same_comparison_on_both_sides_rejected():
    with pytest.raises(RecordError, match="矛盾"):
        build_record(month_day_rec("R-ret", 12, 7), Docs())


def test_non_complementary_pair_rejected():
    with pytest.raises(RecordError):
        build_record(month_day_rec("R-ret", 12, 8), Docs())
    with pytest.raises(RecordError):
        build_record(month_day_rec("R-ret", 12, 8, op=">"), Docs())


def test_reference_before_definition_rejected():
    with pytest.raises(RecordError, match="名前"):
        build_record(month_day_rec("R-ret", 12, 9), Docs())


def test_reference_reads_op_per_phrase():
    # 以後 = >=、それより前 = <、を過ぎた = >、以前 = <=
    assert [op for op, _ in parse.named_boundary_refs(Q6, Q6.index("7日目") + 3)] == [">=", "<"]
    assert [op for op, _ in parse.named_boundary_refs(Q5, Q5.index("10日目") + 4)] == [">", "<="]
    assert [op for op, _ in parse.named_boundary_refs(Q3, Q3.index("12日") + 3)] == ["<=", ">"]


def test_undefined_or_different_name_rejected_for_start_count_forms():
    with pytest.raises(RecordError, match="名前"):
        build_record(damage_rec(line=14, quote=LINES[13].rstrip("。")), Docs())
    with pytest.raises(RecordError, match="名前"):
        build_record(start_rec(quote=LINES[14].rstrip("。")) | {"conditions": [
            {**start_rec()["conditions"][0], "evidence": ev(15)}]}, Docs())
    with pytest.raises(RecordError, match="比べ方"):
        build_record(damage_rec(line=16, quote=LINES[15].rstrip("。")), Docs())


def test_undefined_or_different_name_rejected():
    with pytest.raises(RecordError, match="名前"):
        build_record(month_day_rec("R-ret", 12, 10), Docs())
    with pytest.raises(RecordError):  # 名付けが無い
        build_record(month_day_rec("R-ret", 12, 13), Docs())


def test_name_without_comparison_word_rejected():
    with pytest.raises(RecordError, match="比べ方"):
        build_record(month_day_rec("R-ret", 12, 11), Docs())


def test_start_count_without_name_is_rejected():
    r = record("R-damage", [["破損"], ["受理"]], [F_DMG, F_REP],
               {"fact": "r", "op": ">", "threshold": {"type": "offset_days", "base": "d", "days": 10, "count_from": "next_day"}},
               "受理不可", "受理可", ["受理可", "受理不可"], 12)
    with pytest.raises(RecordError, match="名前"):
        build_record(r, Docs())


def test_wrong_day_count_or_day_rejected():
    r = damage_rec()
    r["conditions"][0]["threshold"]["days"] = 11
    with pytest.raises(RecordError):
        build_record(r, Docs())
    with pytest.raises(RecordError):
        build_record(month_day_rec("R-ret", 13, 3), Docs())


def test_existing_direct_forms_are_unchanged():
    # 名付けでない今までの言い方は今まで通り（過ぎた = >）
    doc = DOC.replace("# 申請規程", "# 申請規程\n翌月10日を過ぎた申請は受理しない。", 1)
    lines = doc.split("\n")
    d = month_day_rec("R-ret", 10, 2, op=">", quote="翌月10日を過ぎた申請は受理しない")
    d["doc_version"] = sha256_text(doc)
    d["conditions"][0]["evidence"]["line"] = 2
    d["outcomes"]["all_true"]["evidence"] = d["outcomes"]["any_false"]["evidence"] = d["conditions"][0]["evidence"]
    assert lines[1].startswith("翌月10日")
    assert build_record(copy.deepcopy(d), Docs(doc))
    d["conditions"][0]["op"] = "<="
    with pytest.raises(RecordError):
        build_record(d, Docs(doc))
