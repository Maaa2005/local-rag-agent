"""decide.engine: 記録の選択 → 事実の取り出し → 計算 → 結論。各段の保留と、計算の境目。

規程の文書は、この試験の中で作った架空の 1 本（購買規程）。.jsonl・開発用の問は開かない。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date

import pytest

from decide.engine import Decided, DecisionEngine, FileDocLookup, Held, NoRecord
from decide.records import build_record, sha256_text

LINES = [
    "# 購買規程",                                                    # 1
    "",                                                              # 2
    "10万円を超える購入は部長の承認を要する。",                      # 3
    "出張に伴う旅費は別の規程による。",                              # 4
    "",                                                              # 5
    "購入日の翌日から数えて14日を過ぎた申請は受け付けない。",        # 6
    "申請には領収書を添える。",                                      # 7
    "",                                                              # 8
    "勤続3年以上の社員は住宅手当の対象とする。",                    # 9
    "",                                                              # 10
    "全労働日の8割以上出勤した社員に特別休暇を付与する。",          # 11
    "",                                                              # 12
    "翌月10日を過ぎた申請は原則として受理しない。",                  # 13
    "人間ドックの補助は、2026年12月1日より前に受診した分を対象とする。",  # 14
    "賞与は評価が5点以上の社員に支給する。",                          # 15
]
DOC = "\n".join(LINES) + "\n"
PATH = "/watched/general/購買規程.md"
OPT = "最後の行には「結論: {a}」または「結論: {b}」のどちらかだけを書いてください。"


def ev(line, quote):
    assert quote in LINES[line - 1]
    return {"quote": quote, "line": line}


class Docs:
    def __init__(self, text=DOC):
        self.text = text

    def lookup(self, document_id):
        return (PATH, sha256_text(self.text)) if document_id == PATH else None

    def read_text(self, doc_path):
        return self.text if doc_path == PATH else None


@dataclass(frozen=True)
class Ch:
    document_id: str
    content: str = "x"
    access_level: int = 1
    source: str = "購買規程.md"


CHUNKS = [Ch(PATH)]


def rec(rule_id, groups, facts, conditions, a, b, options, **extra):
    q = conditions[0]["evidence"] if conditions else None
    d = {"rule_id": rule_id, "doc_path": PATH, "doc_version": sha256_text(DOC), "description": rule_id, "supported": True,
         "subject": {"groups": groups}, "facts": facts, "conditions": conditions,
         "outcomes": {"all_true": {"value": a, "basis": "explicit", "evidence": q},
                      "any_false": {"value": b, "basis": "inferred_negation", "evidence": q}},
         "options": options}
    d.update(extra)
    return d


AMOUNT = [{"role": "amount", "label": "金額", "kind": "money", "cues": ["購入", "金額"]}]
MONEY = rec("R-money", [["購入", "備品"], ["承認"]], AMOUNT,
            [{"fact": "amount", "op": ">", "threshold": {"type": "number", "value": 100000},
              "evidence": ev(3, "10万円を超える購入は部長の承認を要する")}], "必要", "不要", ["必要", "不要"])
MONEY["subject"]["exclude"] = [{"terms": ["出張"], "reason": "別規程", "evidence": ev(4, "出張に伴う旅費は別の規程による")}]

DEADLINE = rec(
    "R-deadline", [["購入", "備品"], ["申請", "期限"]],
    [{"role": "p", "label": "購入日", "kind": "date", "cues": ["購入"]},
     {"role": "s", "label": "申請日", "kind": "date", "cues": ["申請"]}],
    [{"fact": "s", "op": ">", "threshold": {"type": "offset_days", "base": "p", "days": 14, "count_from": "next_day"},
      "evidence": ev(6, "購入日の翌日から数えて14日を過ぎた申請は受け付けない")}],
    "受け付けられない", "受け付けられる", ["受け付けられる", "受け付けられない"],
    other_requirements=[{"description": "領収書の添付", "cues": ["領収書"], "assume_met_if_unstated": True,
                         "matters_for": "any_false", "evidence": ev(7, "申請には領収書を添える")}])

MONTHDAY = rec(
    "R-monthday", [["精算"], ["受理"]],
    [{"role": "e", "label": "発生日", "kind": "date", "cues": ["発生"]},
     {"role": "s", "label": "申請日", "kind": "date", "cues": ["申請"]}],
    [{"fact": "s", "op": ">", "threshold": {"type": "month_day", "base": "e", "month_offset": 1, "day": 10},
      "evidence": ev(13, "翌月10日を過ぎた申請は原則として受理しない")}],
    "受理されない", "受理される", ["受理される", "受理されない"],
    exceptions=[{"description": "原則の例外", "cues": ["やむを得"], "evidence": ev(13, "原則として受理しない")}])

TENURE = rec("R-tenure", [["住宅手当"], ["対象"]],
             [{"role": "t", "label": "勤続", "kind": "quantity", "unit": "年", "cues": ["勤続"]}],
             [{"fact": "t", "op": ">=", "threshold": {"type": "number", "value": 3},
               "evidence": ev(9, "勤続3年以上の社員は住宅手当の対象とする")}], "対象になる", "対象にならない",
             ["対象になる", "対象にならない"])

RATIO = rec("R-ratio", [["特別休暇"], ["付与"]],
            [{"role": "att", "label": "出勤日数", "kind": "quantity", "unit": "日", "cues": ["出勤"]},
             {"role": "wk", "label": "全労働日数", "kind": "quantity", "unit": "日", "cues": ["全労働日"]},
             {"role": "rate", "label": "出勤率", "kind": "ratio", "cues": [], "percent_cues": ["出勤率"],
              "numerator": "att", "denominator": "wk"}],
            [{"fact": "rate", "op": ">=", "threshold": {"type": "number", "value": 80},
              "evidence": ev(11, "全労働日の8割以上出勤した社員に特別休暇を付与する")}], "付与される", "付与されない",
            ["付与される", "付与されない"])

DOCK = rec("R-dock", [["人間ドック"], ["補助", "対象"]],
           [{"role": "x", "label": "受診日", "kind": "date", "cues": ["受診"]}],
           [{"fact": "x", "op": "<", "threshold": {"type": "date", "value": "2026-12-01"},
             "evidence": ev(14, "2026年12月1日より前に受診した分を対象とする")}], "対象になる", "対象にならない",
           ["対象になる", "対象にならない"])

UNSUPPORTED = {"rule_id": "R-unsup", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "d",
               "supported": False, "scope_reason": "回数を数える計算は対応外",
               "subject": {"groups": [["回数"], ["補助"]]}, "facts": [], "conditions": [], "outcomes": {},
               "options": ["a", "b"]}

# 規程から導けない側を unknown にした記録（賞与）。条件の引用は「5以上」
UNKNOWN = rec("R-unknown", [["賞与"], ["支給"]],
              [{"role": "score", "label": "評価", "kind": "quantity", "unit": "点", "cues": ["評価"]}],
              [{"fact": "score", "op": ">=", "threshold": {"type": "number", "value": 5},
                "evidence": ev(15, "賞与は評価が5点以上の社員に支給する")}], "支給される", "unknown", ["支給される", "支給されない"])
UNKNOWN["outcomes"]["any_false"] = {"value": "unknown", "basis": "unknown"}


def engine(*datas, docs=None, on_record=None):
    docs = docs or Docs()
    return DecisionEngine([build_record(copy.deepcopy(d), docs) for d in datas], docs, on_record=on_record)


ALL = (MONEY, DEADLINE, MONTHDAY, TENURE, RATIO, DOCK, UNSUPPORTED)


def decide(q, *datas, chunks=CHUNKS, **kw):
    return engine(*(datas or ALL), **kw).decide(q, chunks)


def last(res):
    assert res.kind == "decided", getattr(res, "reason", res)
    return res.answer_text.strip().split("\n")[-1]


def money_q(amount, tail=""):
    return f"{amount}の備品を購入します。部長の承認は必要ですか？{tail}" + OPT.format(a="必要", b="不要")


# ---------------------------------------------------------------- 金額

@pytest.mark.parametrize("amount,want", [
    ("99,999円", "不要"), ("100,000円", "不要"), ("100,001円", "必要"), ("10万円", "不要"), ("10万1円", "必要"),
    ("9万9千円", "不要"), ("12万円", "必要"),
])
def test_money_boundary(amount, want):
    assert last(decide(money_q(amount), MONEY)) == f"結論: {want}"


def test_decided_result_shape_and_text_template():
    res = decide(money_q("120,000円"), MONEY)
    assert isinstance(res, Decided) and res.kind == "decided"
    t = res.answer_text
    assert t.startswith("【規程にもとづく判定】") and "規程の記載:" in t and "質問から取り出した値:" in t and "計算:" in t
    assert "10万円を超える購入は部長の承認を要する" in t and "3行目" in t
    assert res.used_chunks == tuple(CHUNKS)
    assert res.decision_record["rule_id"] == "R-money" and res.decision_record["outcome_key"] == "all_true"


def test_excluded_subject_is_no_record():
    res = decide("出張で12万円の備品を購入します。部長の承認は必要ですか？", MONEY)
    assert isinstance(res, NoRecord) and res.kind == "no_record"
    assert res.decision_record["route"] == "model_only" and res.decision_record["excluded"]


def test_unrelated_question_is_no_record_and_unknown_document_too():
    assert decide("今日の昼食は何にしますか？").kind == "no_record"
    assert engine(*ALL).decide(money_q("12万円"), [Ch("/watched/general/別の文書.md")]).kind == "no_record"
    assert engine(*ALL).decide(money_q("12万円"), []).kind == "no_record"


# ---------------------------------------------------------------- 期限

def dq(buy, submit, tail=""):
    return (f"{buy}に備品を購入しました。{submit}に申請します。{tail}受け付けられますか？"
            + OPT.format(a="受け付けられる", b="受け付けられない"))


@pytest.mark.parametrize("buy,submit,want", [
    ("2026年9月1日", "2026年9月15日", "受け付けられる"),     # 翌日から数えて 14 日目
    ("2026年9月1日", "2026年9月16日", "受け付けられない"),
    ("2026年12月25日", "2027年1月8日", "受け付けられる"),    # 年またぎ
    ("2026年12月25日", "2027年1月9日", "受け付けられない"),
    ("2028年2月20日", "2028年3月5日", "受け付けられる"),     # うるう年
    ("2028年2月20日", "2028年3月6日", "受け付けられない"),
    ("2027年2月20日", "2027年3月6日", "受け付けられる"),     # 平年
    ("2027年2月20日", "2027年3月7日", "受け付けられない"),
    ("2026年3月31日", "2026年4月14日", "受け付けられる"),    # 月末
])
def test_offset_days_deadline(buy, submit, want):
    assert last(decide(dq(buy, submit), DEADLINE)) == f"結論: {want}"


def mq(expense, submit, tail=""):
    return (f"{expense}に発生した費用の精算を、{submit}に申請しました。{tail}受理されますか？"
            + OPT.format(a="受理される", b="受理されない"))


@pytest.mark.parametrize("expense,submit,want", [
    ("2026年9月15日", "2026年10月10日", "受理される"),
    ("2026年9月15日", "2026年10月11日", "受理されない"),
    ("2026年12月20日", "2027年1月10日", "受理される"),      # 年またぎ
    ("2026年12月20日", "2027年1月11日", "受理されない"),
    ("2026年1月31日", "2026年2月10日", "受理される"),       # 月末の発生
    ("2026年1月31日", "2026年2月11日", "受理されない"),
])
def test_month_day_deadline(expense, submit, want):
    assert last(decide(mq(expense, submit), MONTHDAY)) == f"結論: {want}"


@pytest.mark.parametrize("exam,want", [("2026年11月30日", "対象になる"), ("2026年12月1日", "対象にならない"),
                                         ("2026年12月2日", "対象にならない")])
def test_date_threshold_strictly_before(exam, want):
    q = f"{exam}に人間ドックを受診しました。補助の対象になりますか？" + OPT.format(a="対象になる", b="対象にならない")
    assert last(decide(q, DOCK)) == f"結論: {want}"


# ---------------------------------------------------------------- 量・割合

@pytest.mark.parametrize("period,want", [("3年", "対象になる"), ("36か月", "対象になる"), ("36ヶ月", "対象になる"),
                                          ("35か月", "対象にならない"), ("2年11か月", None)])
def test_quantity_with_year_month_conversion(period, want):
    q = f"勤続{period}の社員は住宅手当の対象になりますか？" + OPT.format(a="対象になる", b="対象にならない")
    res = decide(q, TENURE)
    if want is None:
        assert res.kind == "held"  # 「2年」と「11か月」は同じ役割に違う値
    else:
        assert last(res) == f"結論: {want}"


def test_unit_without_conversion_is_held():
    res = decide("勤続1000日の社員は住宅手当の対象になりますか？", TENURE)
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:unit_mismatch"


def rq(att, wk):
    return (f"この期間の全労働日{wk}日のうち、出勤したのは{att}日です。特別休暇は付与されますか？"
            + OPT.format(a="付与される", b="付与されない"))


@pytest.mark.parametrize("att,wk,want", [(79, 100, "付与されない"), (80, 100, "付与される"), (90, 100, "付与される"),
                                          (4, 5, "付与される"), (3, 5, "付与されない")])
def test_ratio_from_numerator_and_denominator(att, wk, want):
    assert last(decide(rq(att, wk), RATIO)) == f"結論: {want}"


def test_ratio_from_percent_only():
    q = "出勤率は80%でした。特別休暇は付与されますか？" + OPT.format(a="付与される", b="付与されない")
    assert last(decide(q, RATIO)) == "結論: 付与される"
    q = "出勤率は79%でした。特別休暇は付与されますか？"
    assert last(decide(q, RATIO)) == "結論: 付与されない"


def test_denominator_zero_numerator_over_denominator_and_conflict_are_held():
    assert decide(rq(0, 0), RATIO).decision_record["hold_code"] == "facts:denominator_zero"
    assert decide(rq(11, 10), RATIO).decision_record["hold_code"] == "facts:numerator_gt_denominator"
    q = "出勤率は95%です。全労働日100日のうち、出勤したのは70日です。特別休暇は付与されますか？"
    assert decide(q, RATIO).decision_record["hold_code"] == "facts:conflict"


def test_half_point_difference_between_percent_and_computed_is_tolerated():
    q = "出勤率は80%です。全労働日1000日のうち、出勤したのは798日です。特別休暇は付与されますか？"
    assert decide(q, RATIO).kind == "decided"


# ---------------------------------------------------------------- 保留の段

def held(res, code):
    assert res.kind == "held", res
    assert res.decision_record["hold_code"] == code, res.decision_record["hold_code"]
    assert res.decision_record["final_answer"] is None and res.decision_record["conclusion"] is None
    return res


def test_hold_facts_missing_conflict_unresolved_date():
    held(decide("備品を購入します。部長の承認は必要ですか？", MONEY), "facts:missing")
    held(decide("購入金額は98,000円と購入金額は120,000円です。部長の承認は必要ですか？", MONEY), "facts:conflict")
    held(decide(mq("9月15日", "10月9日"), MONTHDAY), "facts:unresolved_date")


def test_unassigned_amount_is_role_undecided():
    # 手がかりの無い別の金額（役割が決まらない）が混ざる問は言い切らない
    held(decide("備品を購入します。余った予算は50,000円です。120,000円で部長の承認は必要ですか？", MONEY),
         "facts:role_undecided")


def test_hold_version_when_document_changed():
    new = Docs(DOC + "\n追記。\n")
    e = DecisionEngine([build_record(copy.deepcopy(MONEY), Docs())], new)
    held(e.decide(money_q("12万円"), CHUNKS), "version:stale")


def test_hold_version_when_record_version_wrong():
    bad = copy.deepcopy(MONEY)
    bad["doc_version"] = "0" * 64
    held(decide(money_q("12万円"), bad), "version:stale")


def test_hold_scope_unsupported_and_business_day():
    held(decide("人間ドックの補助は何回まで受けられますか？回数を教えてください。", UNSUPPORTED), "scope:unsupported")
    held(decide(dq("2026年9月1日", "2026年9月3日", "3営業日で数える場合、"), DEADLINE), "scope:business_day")


def test_hold_exception_cue():
    held(decide(mq("2026年9月15日", "2026年10月11日", "やむを得ない事情がありました。"), MONTHDAY), "exception:cue")


def test_other_requirement_assumed_met_is_stated_only_on_the_side_it_matters():
    res = decide(dq("2026年9月1日", "2026年9月10日"), DEADLINE)  # any_false 側
    assert last(res) == "結論: 受け付けられる" and "領収書の添付など）他の要件は満たしている前提" in res.answer_text
    res = decide(dq("2026年9月1日", "2026年9月20日"), DEADLINE)  # all_true 側では書かない
    assert last(res) == "結論: 受け付けられない" and "他の要件は満たしている前提" not in res.answer_text


def test_hold_other_requirement_unmet():
    held(decide(dq("2026年9月1日", "2026年9月10日", "領収書を紛失しました。"), DEADLINE), "other_requirements:unmet")
    held(decide(dq("2026年9月1日", "2026年9月10日", "領収書が分からない状態です。"), DEADLINE), "other_requirements:unmet")


def test_hold_other_requirement_unstated_when_not_assumable():
    d = copy.deepcopy(DEADLINE)
    d["other_requirements"][0]["assume_met_if_unstated"] = False
    held(decide(dq("2026年9月1日", "2026年9月10日"), d), "other_requirements:unstated")


def test_hold_outcome_unknown_only_on_the_unknown_side():
    q = "評価が5点の賞与は支給されますか？"
    assert last(decide(q, UNKNOWN)) == "結論: 支給される"
    held(decide("評価が4点の賞与は支給されますか？", UNKNOWN), "outcome:unknown")


def test_hold_outcome_option_mismatch_for_unexpected_wording():
    q = "12万円の備品を購入します。部長の承認は必要ですか？最後の行には「結論: はい」または「結論: いいえ」のどちらかだけを書いてください。"
    held(decide(q, MONEY), "outcome:option_mismatch")


def test_no_options_in_question_still_decides():
    assert last(decide("12万円の備品を購入します。部長の承認は必要ですか？", MONEY)) == "結論: 必要"


def test_subject_tie_is_held():
    twin = copy.deepcopy(MONEY)
    twin["rule_id"] = "R-money-twin"
    held(decide(money_q("12万円"), MONEY, twin), "subject:tie")


def test_more_specific_record_wins_over_general_one():
    specific = copy.deepcopy(MONEY)
    specific["rule_id"] = "R-money-specific"
    specific["subject"]["groups"].append(["特別"])
    q = "特別に12万円の備品を購入します。部長の承認は必要ですか？"
    assert decide(q, MONEY, specific).decision_record["rule_id"] == "R-money-specific"
    assert decide(money_q("12万円"), MONEY, specific).decision_record["rule_id"] == "R-money"


def test_engine_error_is_held_not_raised():
    class Boom(Docs):
        def lookup(self, document_id):
            raise RuntimeError("disk")
    e = DecisionEngine([build_record(copy.deepcopy(MONEY), Docs())], Boom())
    res = e.decide(money_q("12万円"), CHUNKS)
    held(res, "engine:error")
    assert "disk" not in res.reason  # 例外の中身（本文を含みうる）は出さない


# ---------------------------------------------------------------- 判断の記録

def test_decision_record_is_made_once_per_question_and_has_all_fields():
    seen = []
    e = engine(*ALL, on_record=seen.append)
    r1 = e.decide(money_q("12万円"), CHUNKS)
    e.decide("今日の昼食は？", CHUNKS)
    e.decide("備品を購入します。部長の承認は必要ですか？", CHUNKS)
    assert len(seen) == 3 and seen[0] is r1.decision_record
    for k in ("route", "rule_id", "doc_path", "doc_version", "evidence", "extracted", "computation", "conclusion",
              "outcome_key", "outcome_basis", "stage", "reason", "hold_code", "final_answer", "correct", "excluded",
              "candidates"):
        assert k in seen[0]
    assert seen[0]["correct"] is None and "id" not in seen[0]
    assert seen[0]["doc_version"] == sha256_text(DOC) and seen[0]["final_answer"] == r1.answer_text
    assert seen[1]["route"] == "model_only" and seen[2]["route"] == "held"
    assert seen[0]["extracted"][0]["sources"][0]["text"] == "12万円"


def test_basis_inferred_negation_is_stated_in_the_answer():
    t = decide(money_q("5万円"), MONEY).answer_text
    assert "推論" in t
    t = decide(money_q("12万円"), MONEY).answer_text
    assert "推論" not in t


# ---------------------------------------------------------------- F7: 敵対的な記録

def test_f7_wrong_number_wrong_op_wrong_version_are_stopped():
    from decide.records import RecordError
    bad = copy.deepcopy(MONEY)
    bad["conditions"][0]["threshold"]["value"] = 200000
    with pytest.raises(RecordError):
        build_record(bad, Docs())
    bad = copy.deepcopy(MONEY)
    bad["conditions"][0]["op"] = ">="
    with pytest.raises(RecordError):
        build_record(bad, Docs())
    bad = copy.deepcopy(MONEY)
    bad["doc_version"] = "f" * 64
    held(DecisionEngine([build_record(bad, Docs())], Docs()).decide(money_q("12万円"), CHUNKS), "version:stale")


def test_f7_known_gap_reversed_outcome_is_not_stopped():
    """既知の穴: 結論の向きを取り違えた記録は読み込まれ、逆の結論を出す。F7 の機械検査では止まらない。
    止める手段は人による記録の確認（作問役とは別の採点）。ここでは「止まらない」ことを事実として固定する。"""
    bad = copy.deepcopy(MONEY)
    bad["outcomes"]["all_true"]["value"], bad["outcomes"]["any_false"]["value"] = "不要", "必要"
    assert last(decide(money_q("12万円"), bad)) == "結論: 不要"


# ---------------------------------------------------------------- 実ファイルの文書の置き場

def test_file_doc_lookup_resolves_only_md_under_given_dirs(tmp_path):
    (tmp_path / "general").mkdir()
    (tmp_path / "general" / "a.md").write_text("本文\n", encoding="utf-8")
    (tmp_path / "general" / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path.parent / "outside.md").write_text("外", encoding="utf-8")
    fl = FileDocLookup([tmp_path])
    assert fl.lookup("/watched/general/a.md")[0] == "/watched/general/a.md"
    assert fl.lookup("general/a.md")[1] == sha256_text("本文\n")
    assert fl.read_text("/watched/general/a.md") == "本文\n"
    for bad in ("/watched/general/b.txt", "/watched/../outside.md", "/watched/general/../../outside.md",
                "/etc/passwd", "../outside.md", None, "/watched/general/missing.md", "/watched/a/b/c.md"):
        assert fl.lookup(bad) is None, bad
        assert fl.read_text(bad) is None if isinstance(bad, str) else True
