"""numnorm (数・日付の表記そろえ) の単体テスト。

例文はこのテストのために、一般的な規程の書き方から作った物。
"""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from app.services.numnorm import normalize_numbers, parse_number

# (入力, 期待の出力)。変わる物。
CHANGED: list[tuple[str, str]] = [
    # 全角 (NFKC)
    ("申請は３０日以内に行う", "申請は30日以内に行う"),
    ("上限は５％とする", "上限は5%とする"),
    ("費用は１０００００円を超えない", "費用は100,000円を超えない"),
    ("２万円", "20,000円"),
    ("５，０００円", "5,000円"),
    # 漢数字 + 数え方の語
    ("二十五日以内に提出する", "25日以内に提出する"),
    ("百二十日を超える", "120日を超える"),
    ("三千五百円", "3,500円"),
    ("一万二千円", "12,000円"),
    ("十万円以上", "100,000円以上"),
    ("一億円", "100,000,000円"),
    ("千円", "1,000円"),
    ("百円", "100円"),
    ("三か月以内", "3か月以内"),
    ("六ヶ月", "6ヶ月"),
    ("一年間", "1年間"),
    ("二週間以内", "2週間以内"),
    ("八時間", "8時間"),
    ("三名以上", "3名以上"),
    ("五件", "5件"),
    ("二回まで", "2回まで"),
    ("六十五歳", "65歳"),
    ("十点", "10点"),
    ("三個", "3個"),
    ("十枚", "10枚"),
    ("二台", "2台"),
    ("五社", "5社"),
    ("五キロ", "5キロ"),
    ("十km", "10km"),
    ("五十%", "50%"),
    ("三十パーセント", "30パーセント"),
    ("年二回", "年2回"),
    ("月三回以上", "月3回以上"),
    ("第三回", "第3回"),
    ("千人", "1,000人"),
    # 割・分・厘
    ("8割", "80%"),
    ("8割5分", "85%"),
    ("1割2分5厘", "12.5%"),
    ("八割五分", "85%"),
    ("二割以上", "20%以上"),
    ("2.5割", "25%"),
    ("３割以上を負担する", "30%以上を負担する"),
    # 万・千・億と桁区切り
    ("10万円", "100,000円"),
    ("1億2000万円", "120,000,000円"),
    ("3.5万円", "35,000円"),
    ("0.5万円", "5,000円"),
    ("5千円", "5,000円"),
    ("1万2千円", "12,000円"),
    ("100000円", "100,000円"),
    ("5000円", "5,000円"),
    ("10万人", "100,000人"),
    ("約3万円から", "約30,000円から"),
    ("5万円を超える場合は、10万円以上", "50,000円を超える場合は、100,000円以上"),
    # 和暦
    ("令和8年11月15日", "2026年11月15日"),
    ("令和元年5月1日", "2019年5月1日"),
    ("平成31年4月30日", "2019年4月30日"),
    ("昭和64年1月7日", "1989年1月7日"),
    ("平成元年", "1989年"),
    ("令和８年４月１日", "2026年4月1日"),
    ("令和8年度", "2026年度"),
    ("令和八年十一月十五日", "2026年11月15日"),
    # 区切りの日付
    ("2026/11/15", "2026年11月15日"),
    ("2026-11-15", "2026年11月15日"),
    ("2026.11.15", "2026年11月15日"),
    ("2026/1/5", "2026年1月5日"),
    ("2026-01-05", "2026年1月5日"),
    ("２０２６／１１／１５", "2026年11月15日"),
    ("期限は2026/11/15までに提出", "期限は2026年11月15日までに提出"),
    # 漢数字の日付
    ("二〇二六年十一月十五日", "2026年11月15日"),
    ("三月十五日", "3月15日"),
    ("十二月三十一日", "12月31日"),
    # 組み合わせ
    (
        "令和8年11月15日までに10万円以上",
        "2026年11月15日までに100,000円以上",
    ),
    (
        "申請は令和8年11月15日までに、10万円を超える場合は３割以上",
        "申請は2026年11月15日までに、100,000円を超える場合は30%以上",
    ),
]

# 変えてはいけない物。
UNCHANGED: list[str] = [
    "一般に認められる",
    "一部を除く",
    "一様に扱う",
    "一時金を支給する",
    "十分に確認する",
    "万一の場合",
    "千葉県に勤務する",
    "一層の努力",
    "一定の期間",
    "唯一の例外",
    "万年筆",
    "一律に適用する",
    "一方で",
    "第一条",
    "二重払いを防ぐ",
    "五分五分",
    "八月に実施する",
    "数十人が参加",
    "数万円程度",
    "三十分以内に連絡",
    "5分以内に連絡",
    "12時30分",
    "10営業日",
    "三営業日以内",
    "3.11.4版",
    "ver.2026.1.1.3",
    "11/15に開催",
    "1/2以上",
    "2026/2/30",
    "2026-13-01",
    "1,2,3円",
    "1,000円",
    "100,000円",
    "80%",
    "3.5%",
    "1年6か月",
    "2026年11月15日",
    "令和の時代",
    "10万を超える",
    "40歳以上",
    "1万二千円",
    "",
]

# 冪等性を見る追加の入力 (表に載らない書き方)
EXTRA_IDEMPOTENT: list[str] = [
    "令和0年4月1日",
    "数千円から数十万円",
    "1億2万円",
    "三千五",
    "二十五三日",
    "8割5分と5分",
    "10割",
    "0.50万円",
    "1,000.50円",
    "2026/11/15-2026/11/20",
    "令和8年11月15日（2026/11/15）",
    "十人十色",
    "万円",
    "一万",
]


def test_table_size() -> None:
    assert len(CHANGED) + len(UNCHANGED) >= 60
    assert len(UNCHANGED) >= 15


@pytest.mark.parametrize(("src", "want"), CHANGED)
def test_changed(src: str, want: str) -> None:
    assert normalize_numbers(src) == want


@pytest.mark.parametrize("src", UNCHANGED)
def test_unchanged(src: str) -> None:
    assert normalize_numbers(src) == src


def test_idempotent_all_table_inputs() -> None:
    inputs = [s for s, _ in CHANGED] + UNCHANGED + [w for _, w in CHANGED] + EXTRA_IDEMPOTENT
    for x in inputs:
        once = normalize_numbers(x)
        assert normalize_numbers(once) == once, x


def test_kanji_only_before_counter() -> None:
    # 数え方の語が続かない漢数字は、数であっても変えない
    assert normalize_numbers("二十五") == "二十五"
    assert normalize_numbers("三千五百") == "三千五百"


def test_number_before_kanji_unit_not_merged() -> None:
    # 5千円 の「千」を 1000 にして 51000 にしない
    assert normalize_numbers("5千円") == "5,000円"


def test_year_is_not_given_commas() -> None:
    assert normalize_numbers("2026年に制定") == "2026年に制定"
    assert normalize_numbers("二〇二六年に制定") == "2026年に制定"


def test_invalid_wareki_year_zero() -> None:
    assert normalize_numbers("令和0年4月1日") == "令和0年4月1日"


def test_long_digits_is_fast() -> None:
    t = time.perf_counter()
    normalize_numbers("1" * 20000 + "x" + "1,000" * 2000)
    assert time.perf_counter() - t < 2.0


PARSE: list[tuple[str, Decimal | None]] = [
    ("10万", Decimal(100000)),
    ("1,000", Decimal(1000)),
    ("二十五", Decimal(25)),
    ("3.5万", Decimal(35000)),
    ("八割", Decimal(80)),
    ("8割5分", Decimal(85)),
    ("一割二分五厘", Decimal("12.5")),
    ("１，０００", Decimal(1000)),
    ("1億2000万", Decimal(120000000)),
    ("三千五百", Decimal(3500)),
    ("百二十", Decimal(120)),
    ("一万二千", Decimal(12000)),
    ("十万円", Decimal(100000)),
    ("2.5", Decimal("2.5")),
    ("30%", Decimal(30)),
    ("二〇二六", Decimal(2026)),
    ("5千", Decimal(5000)),
    ("0.5万", Decimal(5000)),
    ("  42  ", Decimal(42)),
    ("", None),
    ("abc", None),
    ("万", None),
    ("1,00", None),
    ("5分", None),
    ("十分", None),
    ("一般", None),
    ("1万2万", None),
    ("3.11.4", None),
    ("三千五", None),
    ("割", None),
]


def test_parse_table_size() -> None:
    assert len(PARSE) >= 15


@pytest.mark.parametrize(("src", "want"), PARSE)
def test_parse_number(src: str, want: Decimal | None) -> None:
    got = parse_number(src)
    if want is None:
        assert got is None
    else:
        assert isinstance(got, Decimal)
        assert got == want
