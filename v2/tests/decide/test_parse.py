"""decide.parse: 質問・引用からの値の読み取り、単位の換算、暦の計算。"""
from __future__ import annotations

from datetime import date
from fractions import Fraction
from pathlib import Path

import pytest

from decide import parse


def kinds(q):
    return [(c.kind, c.value) for c in parse.extract(q).cands]


def test_money_with_comma_and_man_yen():
    assert kinds("98,000円の申請") == [("円", Fraction(98000))]
    assert kinds("10万円を超える") == [("円", Fraction(100000))]
    assert kinds("12万5千円") == [("円", Fraction(125000))]


def test_full_dates_and_partial_date_need_one_year():
    c = parse.extract("2026年9月15日に発生し、10月9日に申請しました")
    assert [x.value for x in c.cands] == [date(2026, 9, 15), date(2026, 10, 9)] and not c.unresolved_date
    c = parse.extract("9月15日に発生し、10月9日に申請しました")
    assert c.unresolved_date and c.cands == ()
    c = parse.extract("2026年9月15日に発生し、2027年10月9日に申請し、11月1日に提出")
    assert c.unresolved_date  # 年が 2 種類なら決めない


def test_quantities_and_percent():
    assert kinds("勤続6ヶ月") == [("か月", Fraction(6))]
    assert kinds("勤続3年") == [("年", Fraction(3))]
    assert kinds("2週間と3日") == [("週間", Fraction(2)), ("日", Fraction(3))]
    assert kinds("距離は2kmです") == [("km", Fraction(2))]
    assert kinds("出勤率は79%") == [("%", Fraction(79))]


def test_format_instruction_after_last_line_is_not_read():
    q = "5万円の備品。最後の行には「結論: 必要」または「結論: 不要」のどちらかだけを書いてください。"
    assert kinds(q) == [("円", Fraction(50000))]
    assert "最後の行" not in parse.clean_question(q)


def test_business_days_flag():
    assert parse.extract("3営業日以内に申請").business_days
    assert not parse.extract("3日以内に申請").business_days


def test_convert_only_year_month_and_week_day():
    one = Fraction(1)
    assert parse.convert(Fraction(3), "年", "か月") == 36
    assert parse.convert(Fraction(36), "か月", "年") == 3
    assert parse.convert(Fraction(2), "週間", "日") == 14
    assert parse.convert(Fraction(14), "日", "週間") == 2
    assert parse.convert(one, "日", "か月") is None
    assert parse.convert(one, "年", "日") is None
    assert parse.convert(one, "km", "日") is None
    assert parse.convert(one, "日", "日") == one


@pytest.mark.parametrize("v,op,b,want", [
    (5, ">", 5, False), (5, ">=", 5, True), (5, "<", 5, False), (5, "<=", 5, True),
    (6, ">", 5, True), (4, "<", 5, True), (date(2026, 1, 2), ">", date(2026, 1, 1), True),
    (date(2026, 1, 1), ">", date(2026, 1, 1), False), (date(2026, 1, 1), "<=", date(2026, 1, 1), True),
])
def test_holds_boundaries(v, op, b, want):
    assert parse.holds(v, op, b) is want


def test_add_months_crosses_year():
    assert parse.add_months(date(2026, 12, 20), 1) == (2027, 1)
    assert parse.add_months(date(2026, 11, 30), 2) == (2027, 1)
    assert parse.add_months(date(2026, 1, 31), 1) == (2026, 2)


def test_quote_tokens_and_phrase_op():
    q = "10万円を超える申請は部門長の承認を要する"
    (end, v), = parse.quote_tokens(q, "円")
    assert v == 100000 and parse.phrase_op(q, end, "num") == ">"
    q = "勤続6ヶ月を経過し"
    (end, v), = parse.quote_tokens(q, "か月")
    assert v == 6 and parse.phrase_op(q, end, "num") == ">="  # 経過し は表に足した行
    q = "全労働日の8割以上出勤した"
    (end, v), = parse.quote_tokens(q, "%")
    assert v == 80 and parse.phrase_op(q, end, "num") == ">="
    q = "購入日の翌日から数えて14日を過ぎた申請"
    (end, n), = parse.day_count_tokens(q)
    assert n == 14 and parse.phrase_op(q, end, "date") == ">"
    q = "翌月10日を過ぎた申請"
    (end, k, day, mode, _), = parse.month_day_tokens(q)
    assert (k, day, mode) == (1, 10, "day") and parse.phrase_op(q, end, "date") == ">"


def test_phrase_table_is_a_copy_plus_one_added_row():
    """判定の言い回しの表は backend の numeric_check.py の写し。足した行は「経過し」の 1 行だけ。
    backend は import せず、本文を文字列として読んで各行の書き方が在るかを見る（無ければ飛ばす）。"""
    src_path = Path(__file__).resolve().parents[3] / "backend/app/services/numeric_check.py"
    if not src_path.exists():
        pytest.skip("backend の numeric_check.py が無い")
    src = src_path.read_text(encoding="utf-8")
    added = {r"経過(?:し|した|して)"}
    for table in (parse._NUM_PHRASES, parse._DATE_PHRASES):
        for pat, op in table:
            if pat in added:
                assert pat not in src
                continue
            # _NEG を展開した形は元の文字列に直接出ないので、_NEG より前の部分で比べる
            head = pat.split(parse._NEG)[0] if parse._NEG in pat else pat
            assert head in src, (pat, op)
