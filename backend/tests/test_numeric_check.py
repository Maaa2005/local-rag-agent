"""numeric_check (数の比べをコードでやる) の単体テスト。

仕様: docs/v2-code-compare-design.md。例文はこのテストのために作った物
(開発用の文を含む。確かめ用の文書・問は使わない)。
"""
from __future__ import annotations

import inspect
import logging
from datetime import date
from decimal import Decimal

import pytest

from app.services import llm, numeric_check
from app.services.numeric_check import CheckResult, check


def _run(question: str, *rules: str) -> CheckResult:
    return check(question, list(rules))


# ---------------------------------------------------------------- 関数の形


def test_check_signature_is_question_and_chunk_texts_only():
    params = list(inspect.signature(check).parameters)
    assert params == ["question", "chunk_texts"]


def test_result_has_comparison_fields():
    r = _run("1件98,000円の備品を買います。", "1件あたり10万円を超える備品は承認を要する。")
    assert isinstance(r, CheckResult)
    assert r.value == Decimal(98000)
    assert r.boundary == Decimal(100000)
    assert r.op == ">"
    assert r.applies is False
    assert r.hold_reason is None


def test_exact_sentence_form():
    r = _run("1件98,000円の備品を買います。", "1件あたり10万円を超える備品は承認を要する。")
    assert r.sentence == "規程の「10万円を超える」に対し、質問の値 98,000円 は当てはまらない（98,000 ≤ 100,000）。"


def test_applies_sentence_uses_the_operator_itself():
    r = _run("1件120,000円の備品を買います。", "1件あたり10万円を超える備品は承認を要する。")
    assert r.applies is True
    assert r.sentence == "規程の「10万円を超える」に対し、質問の値 120,000円 は当てはまる（120,000 > 100,000）。"


# ---------------------------------------------------------------- 一覧: 数 × 4 向き

# (規程の言い回し, 向き, 当てはまる値, 当てはまらない値, 境目ちょうどで当てはまるか)
_NUM_ROWS = [
    # >
    ("{n}人を超える", ">", 11, 10),
    ("{n}人を超えた", ">", 11, 10),
    ("{n}人を超えて", ">", 11, 10),
    ("{n}人超過", ">", 11, 10),
    ("{n}人を上回る", ">", 11, 10),
    ("{n}人を上回った", ">", 11, 10),
    ("{n}人より多い", ">", 11, 10),
    ("{n}人を超す", ">", 11, 10),
    # >=
    ("{n}人以上", ">=", 10, 9),
    ("{n}人に達した", ">=", 10, 9),
    ("{n}人に達する", ">=", 10, 9),
    ("{n}人を下回らない", ">=", 10, 9),
    # <
    ("{n}人未満", "<", 9, 10),
    ("{n}人を下回る", "<", 9, 10),
    ("{n}人を下回った", "<", 9, 10),
    ("{n}人に満たない", "<", 9, 10),
    ("{n}人より少ない", "<", 9, 10),
    ("{n}人に届かない", "<", 9, 10),
    # <=
    ("{n}人以下", "<=", 10, 11),
    ("{n}人以内", "<=", 10, 11),
    ("{n}人を超えない", "<=", 10, 11),
    ("{n}人を上回らない", "<=", 10, 11),
    ("{n}人まで", "<=", 10, 11),
]


@pytest.mark.parametrize("phrase,op,yes,no", _NUM_ROWS)
def test_number_phrases(phrase, op, yes, no):
    rule = f"参加者が{phrase.format(n=10)}の場合は申請を要する。"
    r_yes = _run(f"参加者は{yes}人です。", rule)
    r_no = _run(f"参加者は{no}人です。", rule)
    assert r_yes.hold_reason is None, r_yes.hold_reason
    assert r_yes.op == op
    assert r_yes.boundary == Decimal(10)
    assert r_yes.applies is True
    assert r_no.applies is False


@pytest.mark.parametrize(
    "rule,op",
    [
        ("10万円を超えない", "<="),
        ("10万円を超えなかった", "<="),
        ("10万円を超えていない", "<="),
        ("10万円を上回らない", "<="),
        ("10万円を下回らない", ">="),
        ("10万円に達しない", "<"),
    ],
)
def test_negated_forms_flip_direction(rule, op):
    r = _run("金額は10万円です。", f"購入額が{rule}場合は所属長が承認する。")
    assert r.op == op
    assert r.boundary == Decimal(100000)


def test_negative_sentence_ending_does_not_flip():
    # 「2km以下」は否定されていない。文末の「支給しない」は言い回しの否定ではない。
    r = _run("通勤距離は1.8kmです。", "通勤距離が2km以下の社員には、通勤手当を支給しない。")
    assert r.op == "<="
    assert r.applies is True
    assert r.sentence == "規程の「2km以下」に対し、質問の値 1.8km は当てはまる（1.8 ≤ 2）。"


def test_boundary_exact_values():
    rule = "1件あたり5万円を超える備品の購入は、総務部長の承認を要する。"
    assert _run("1件50,000円の備品を買います。", rule).applies is False
    assert _run("1件50,001円の備品を買います。", rule).applies is True
    rule2 = "勤続年数が3年以上の社員は、住宅手当の対象とする。"
    assert _run("勤続年数はちょうど3年です。", rule2).applies is True
    assert _run("勤続年数はちょうど2年です。", rule2).applies is False


# ---------------------------------------------------------------- 一覧: 日付 × 4 向き

_DATE_ROWS = [
    # >
    ("2026年10月31日を過ぎた", ">", date(2026, 11, 1), date(2026, 10, 31)),
    ("2026年10月31日を過ぎて", ">", date(2026, 11, 1), date(2026, 10, 31)),
    ("2026年10月31日より後", ">", date(2026, 11, 1), date(2026, 10, 31)),
    ("2026年10月31日に遅れた", ">", date(2026, 11, 1), date(2026, 10, 31)),
    ("2026年10月31日を経過した後", ">", date(2026, 11, 1), date(2026, 10, 31)),
    # >=
    ("2026年10月31日以降", ">=", date(2026, 10, 31), date(2026, 10, 30)),
    ("2026年10月31日以後", ">=", date(2026, 10, 31), date(2026, 10, 30)),
    ("2026年10月31日から", ">=", date(2026, 10, 31), date(2026, 10, 30)),
    # <
    ("2026年10月31日より前", "<", date(2026, 10, 30), date(2026, 10, 31)),
    ("2026年10月31日の前日まで", "<", date(2026, 10, 30), date(2026, 10, 31)),
    ("2026年10月31日に先立って", "<", date(2026, 10, 30), date(2026, 10, 31)),
    # <=
    ("2026年10月31日まで", "<=", date(2026, 10, 31), date(2026, 11, 1)),
    ("2026年10月31日までに", "<=", date(2026, 10, 31), date(2026, 11, 1)),
    ("2026年10月31日以前", "<=", date(2026, 10, 31), date(2026, 11, 1)),
    ("2026年10月31日を過ぎない", "<=", date(2026, 10, 31), date(2026, 11, 1)),
]


def _q_date(d: date) -> str:
    return f"私の手続の日は{d.year}年{d.month}月{d.day}日です。"


@pytest.mark.parametrize("phrase,op,yes,no", _DATE_ROWS)
def test_date_phrases(phrase, op, yes, no):
    rule = f"本手続は{phrase}の場合に行う。"
    r_yes = _run(_q_date(yes), rule)
    r_no = _run(_q_date(no), rule)
    assert r_yes.hold_reason is None, r_yes.hold_reason
    assert r_yes.op == op
    assert r_yes.boundary == date(2026, 10, 31)
    assert r_yes.applies is True
    assert r_no.applies is False


def test_date_sentence_form():
    r = _run("私の入社日は2026年10月31日です。", "2026年11月1日以降に入社した社員は、表彰の対象外とする。")
    assert r.sentence == "規程の「2026年11月1日以降」に対し、質問の値 2026年10月31日 は当てはまらない（2026-10-31 < 2026-11-01）。"


# ---------------------------------------------------------------- 単位の換算


def test_man_yen_vs_yen():
    r = _run("購入額は98,000円です。", "10万円を超える購入は承認を要する。")
    assert r.boundary == Decimal(100000) and r.value == Decimal(98000)


def test_value_in_man_yen_against_yen_rule():
    r = _run("購入額は12万円です。", "100,000円を超える購入は承認を要する。")
    assert r.value == Decimal(120000) and r.applies is True


def test_sen_yen():
    r = _run("購入額は50千円です。", "5万円を超える購入は承認を要する。")
    assert r.value == Decimal(50000) and r.applies is False
    r2 = _run("購入額は5万2千円です。", "5万円を超える購入は承認を要する。")
    assert r2.value == Decimal(52000) and r2.applies is True


def test_wari_to_percent():
    r = _run("出席率は8割でした。", "出席率が90%未満の部署には改善計画を求める。")
    assert r.value == Decimal(80)
    assert r.boundary == Decimal(90)
    assert r.applies is True
    r2 = _run("出席率は92%でした。", "出席率が9割以上の部署は表彰する。")
    assert r2.boundary == Decimal(90) and r2.applies is True


def test_wari_with_bu():
    r = _run("出席率は8割5分でした。", "出席率が90%未満の部署には改善計画を求める。")
    assert r.value == Decimal(85)


def test_full_width_digits_and_percent():
    r = _run("受診率は８８％でした。", "受診率が90%未満の部署には改善計画を求める。")
    assert r.value == Decimal(88) and r.applies is True


def test_decimal_values_are_exact():
    r = _run("距離は0.3kmです。", "距離が0.1km以上の社員は対象とする。")
    assert r.value == Decimal("0.3") and r.applies is True


def test_hundred_point_denominator_is_not_a_competing_value():
    r = _run("100点満点で80点でした。", "70点以上の者は合格とする。")
    assert r.hold_reason is None
    assert r.value == Decimal(80) and r.applies is True


# ---------------------------------------------------------------- 相対日付


def test_relative_days_from_next_day_counts_base_plus_n():
    rule = "購入日の翌日から数えて14日を過ぎた購入申請は受け付けない。"
    q = "2026年9月1日に購入した備品の購入申請を2026年9月{d}日に出しました。"
    r14, r15, r16 = (_run(q.format(d=d), rule) for d in (14, 15, 16))
    assert r14.boundary == date(2026, 9, 15) and r14.applies is False
    assert r15.applies is False
    assert r16.applies is True
    assert r16.op == ">"


def test_relative_within_n_days():
    rule = "育児支援金の申請は、子の誕生日の翌日から数えて60日以内に行ったものに限り受け付ける。"
    q = "子どもが2026年8月1日に生まれ、育児支援金の申請を2026年{md}に出しました。"
    r = _run(q.format(md="9月30日"), rule)
    assert r.boundary == date(2026, 9, 30) and r.op == "<=" and r.applies is True
    assert _run(q.format(md="10月1日"), rule).applies is False


def test_relative_n_days_before():
    rule = "イベントの申込は、開催日の3日前までに行う。"
    q = "イベントは2026年10月20日に開催されます。申込を2026年10月{d}日に出しました。"
    r = _run(q.format(d=17), rule)
    assert r.boundary == date(2026, 10, 17) and r.op == "<=" and r.applies is True
    assert _run(q.format(d=18), rule).applies is False


def test_relative_next_month_day():
    rule = "経費の請求は、対象月の翌月10日までに行う。"
    r = _run("10月分の経費請求を2026年11月10日に出しました。", rule)
    assert r.boundary == date(2026, 11, 10) and r.op == "<=" and r.applies is True
    assert _run("10月分の経費請求を2026年11月11日に出しました。", rule).applies is False


def test_relative_next_month_across_year_end():
    rule = "経費の請求は、翌月5日までに行う。"
    r = _run("12月分の経費請求を2027年1月6日に出しました。", rule)
    assert r.boundary == date(2027, 1, 5) and r.applies is False


def test_relative_month_end_from_base_date():
    rule = "立替経費は、発生日の当月末までに申請する。"
    r = _run("立替が2026年2月10日に発生し、申請を2026年3月1日に出しました。", rule)
    assert r.boundary == date(2026, 2, 28) and r.applies is False


def test_relative_next_month_end():
    rule = "請求書は、発行日の翌月末日までに支払う。"
    r = _run("請求書を2026年1月20日に発行し、支払を2026年2月28日に行いました。", rule)
    assert r.boundary == date(2026, 2, 28) and r.applies is True


def test_relative_base_and_target_not_competing_values():
    # 基準日 (購入日) は競合する値に数えない
    r = _run(
        "2026年9月1日に購入した備品の購入申請を2026年9月10日に出しました。",
        "購入日の翌日から数えて14日を過ぎた購入申請は受け付けない。",
    )
    assert r.hold_reason is None and r.applies is False


# ---------------------------------------------------------------- 保留


def test_hold_no_value_in_question():
    r = _run("備品の購入に承認は必要ですか。", "5万円を超える購入は承認を要する。")
    assert r.sentence is None and r.hold_reason


def test_hold_two_different_values_of_same_kind():
    r = _run("5万円の備品と8万円の備品を買います。", "5万円を超える購入は承認を要する。")
    assert r.sentence is None and "複数" in r.hold_reason


def test_same_value_repeated_is_not_a_conflict():
    r = _run("5万円の備品を買います。予算は5万円です。", "3万円を超える購入は承認を要する。")
    assert r.sentence is not None


def test_hold_no_candidate():
    r = _run("購入額は5万円です。", "勤続年数が3年以上の社員は住宅手当の対象とする。")
    assert r.sentence is None and r.hold_reason


def test_hold_conflicting_candidates_with_no_clear_winner():
    # 向きも境目も違う 2 つの条件が同じ程度に質問と関係する
    rule = "購入額が3万円を超える場合は承認を要し、購入額が1万円未満の場合は承認を要しない。"
    r = _run("購入額は2万円です。", rule)
    assert r.sentence is None and r.hold_reason


def test_hold_range_in_one_sentence():
    r = _run("購入額は2万円です。", "購入額が1万円以上3万円未満の場合は所属長が承認する。")
    assert r.sentence is None


def test_conflict_resolved_by_unique_word_overlap_is_marked():
    rules = [
        "出張費の精算は、3万円を超える場合に領収書の添付を要する。",
        "会議費の精算は、10万円を超える場合に部長承認を要する。",
    ]
    r = _run("会議費の精算は12万円でした。部長承認が必要ですか。", *rules)
    assert r.boundary == Decimal(100000)
    assert r.tiebreak is True


def test_hold_phrase_not_in_the_list():
    r = _run("購入額は5万円です。", "購入額が5万円ちょうどの場合は承認を要する。")
    assert r.sentence is None and r.hold_reason


def test_hold_business_days():
    r = _run("2026年9月1日に購入し、2026年9月9日に申請しました。", "購入日から5営業日を過ぎた申請は受け付けない。")
    assert r.sentence is None and "営業日" in r.hold_reason
    r2 = _run("5営業日後に申請します。", "申請は10日以内に行う。")
    assert r2.sentence is None and "営業日" in r2.hold_reason


def test_hold_unit_mismatch():
    r = _run("勤続は18か月です。", "勤続年数が3年以上の社員は住宅手当の対象とする。")
    assert r.sentence is None and "単位" in r.hold_reason


def test_hold_two_dates_without_roles():
    r = _run(
        "2026年9月1日と2026年9月10日があります。",
        "購入日の翌日から数えて14日を過ぎた申請は受け付けない。",
    )
    assert r.sentence is None and r.hold_reason


def test_hold_two_dates_for_absolute_rule():
    r = _run("2026年9月1日と2026年9月10日があります。", "2026年11月1日以降に入社した社員は対象外とする。")
    assert r.sentence is None and r.hold_reason


def test_hold_relative_rule_without_base():
    r = _run("申請を2026年9月10日に出しました。", "購入日の翌日から数えて14日を過ぎた申請は受け付けない。")
    assert r.sentence is None and r.hold_reason


def test_hold_invalid_day_of_next_month():
    r = _run("1月分の経費請求を2026年3月1日に出しました。", "請求は翌月31日までに行う。")
    assert r.sentence is None and r.hold_reason


def test_hold_has_no_comparison_fields():
    r = _run("備品の購入に承認は必要ですか。", "5万円を超える購入は承認を要する。")
    assert r.applies is None and r.value is None and r.boundary is None and r.op is None


# ---------------------------------------------------------------- llm._build_messages

_OLD_FORMAT = "【参考情報】\n{context}\n\n【質問】\n{question}"
_CHUNKS = [
    {"source_file": "/watched/general/備品購入規程.md", "content": "1件あたり5万円を超える備品の購入は、総務部長の承認を要する。"},
]
_Q = "1件49,000円の備品を購入します。総務部長の承認は必要ですか？"


def _set_flag(monkeypatch, value: bool):
    monkeypatch.setattr(llm.settings, "numeric_check_enabled", value)


def test_setting_default_is_off():
    from app.config import Settings

    assert Settings.model_fields["numeric_check_enabled"].default is False


def test_off_build_messages_is_identical_to_before(monkeypatch):
    _set_flag(monkeypatch, False)
    msgs = llm._build_messages(_Q, _CHUNKS)
    context = llm._build_context(_CHUNKS)
    assert msgs == [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": _OLD_FORMAT.format(context=context, question=_Q)},
    ]
    assert "【計算の結果】" not in msgs[1]["content"]


def test_off_does_not_call_check(monkeypatch):
    _set_flag(monkeypatch, False)

    def boom(*a, **k):
        raise AssertionError("OFF の時は check を呼ばない")

    monkeypatch.setattr(numeric_check, "check", boom)
    llm._build_messages(_Q, _CHUNKS)


def test_on_inserts_calculation_between_context_and_question(monkeypatch):
    _set_flag(monkeypatch, True)
    msgs = llm._build_messages(_Q, _CHUNKS)
    context = llm._build_context(_CHUNKS)
    sentence = check(_Q, [c["content"] for c in _CHUNKS]).sentence
    assert sentence
    assert msgs[1]["content"] == f"【参考情報】\n{context}\n\n【計算の結果】\n{sentence}\n\n【質問】\n{_Q}"
    assert msgs[0] == {"role": "system", "content": llm.SYSTEM_PROMPT}


def test_on_hold_is_same_as_off_and_logs_reason(monkeypatch, caplog):
    q = "備品の購入に承認は必要ですか？"
    _set_flag(monkeypatch, False)
    off = llm._build_messages(q, _CHUNKS)
    _set_flag(monkeypatch, True)
    with caplog.at_level(logging.INFO, logger="app.services.llm"):
        on = llm._build_messages(q, _CHUNKS)
    assert on == off
    assert any("保留" in rec.getMessage() and "値が無い" in rec.getMessage() for rec in caplog.records)


def test_on_check_exception_falls_back_to_off(monkeypatch, caplog):
    _set_flag(monkeypatch, False)
    off = llm._build_messages(_Q, _CHUNKS)
    _set_flag(monkeypatch, True)

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(numeric_check, "check", boom)
    with caplog.at_level(logging.ERROR, logger="app.services.llm"):
        assert llm._build_messages(_Q, _CHUNKS) == off
