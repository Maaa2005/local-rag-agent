"""封 C: 人が書いた確認問 24 問（life #45・docs/verified-distill-plan.md §1）。

生成器を作る前に固定する。生成器と別の題材 4 つ（会議室予約・社内図書貸出・自転車通勤手当・退職金）。
区分: 境目 12（4 条件 × 下・ちょうど・上）・換算 4・例外 4・判断できない 4。
正解は facts から 3 値でコードが出す（evaluation.teacher_check.rules の 3 値の部品を使う）。
文は生成器のひな形と違う、ふだんの相談の書き方にする。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from fractions import Fraction

from evaluation.teacher_check.rules import UNDECIDED, all3, any3, cmp, full_years, not3

Tri = bool | None


# ─── 条件 ──────────────────────────────────────────────────────
def room_length_ok(f: dict) -> Tri:            # 会議室 第3条: 連続4時間まで
    return cmp(f.get("hours"), "<=", 4)


def room_window_ok(f: dict) -> Tri:            # 会議室 第2条: 利用日の30日前から受付
    reserve, use = f.get("reserve_on"), f.get("use_on")
    if reserve is None or use is None:
        return None
    return (use - reserve).days <= 30


def room_big_needed(f: dict) -> Tri:           # 会議室 第4条: 20名を超えると大会議室。オンラインは数えない
    total, online = f.get("participants"), f.get("online", 0)
    if total is None or online is None:
        return None
    return total - online > 20


def book_count_ok(f: dict) -> Tri:             # 図書 第2条: 5冊まで
    return cmp(f.get("books"), "<=", 5)


def book_lendable(f: dict) -> Tri:             # 図書 第5条: 定価1万円を超える本は貸し出さない
    return not3(cmp(f.get("price"), ">", 10_000))


def book_can_borrow(f: dict) -> Tri:           # 図書 第4条: 7日以上遅れたら30日停止。図書係が認めた理由は除く
    late, excused = f.get("late_days"), f.get("excused")
    suspended = all3(cmp(late, ">=", 7), not3(excused), cmp(f.get("days_since_return"), "<", 30))
    return all3(not3(suspended), book_count_ok(f) if "books" in f else True)


def bike_paid(f: dict) -> Tri:                 # 自転車 第2条 かつ 第4条
    km, insured = f.get("km"), f.get("insured")
    days, workdays = f.get("bike_days"), f.get("workdays")
    ratio_ok = None if days is None or workdays is None else not (Fraction(days, workdays) < Fraction(7, 10))
    return all3(cmp(km, ">=", 3), insured, ratio_ok)


def bike_amount_6000(f: dict) -> Tri:          # 自転車 第3条: 片道10km以上は6,000円（支給される前提の問だけで使う）
    return cmp(f.get("km"), ">=", 10)


def retire_paid(f: dict) -> Tri:               # 退職金 第2条 かつ 第3条
    years = f.get("years")
    if years is None and f.get("entry_on") and f.get("retire_on"):
        years = full_years(f["entry_on"], f["retire_on"])
    return all3(cmp(years, ">=", 2), not3(f.get("disciplinary")))


@dataclass(frozen=True)
class Outcome:
    name: str
    doc: str
    fn: object
    yes: str
    no: str
    articles: tuple[int, ...]


OUTCOMES = {o.name: o for o in (
    Outcome("room_length_ok", "会議室予約規程", room_length_ok, "予約できる", "予約できない", (3,)),
    Outcome("room_window_ok", "会議室予約規程", room_window_ok, "受け付けられる", "受け付けられない", (2,)),
    Outcome("room_big_needed", "会議室予約規程", room_big_needed, "大会議室が必要", "大会議室でなくてよい", (4,)),
    Outcome("book_count_ok", "社内図書貸出規程", book_count_ok, "借りられる", "借りられない", (2,)),
    Outcome("book_lendable", "社内図書貸出規程", book_lendable, "貸し出される", "貸し出されない", (5,)),
    Outcome("book_can_borrow", "社内図書貸出規程", book_can_borrow, "借りられる", "借りられない", (2, 4)),
    Outcome("bike_paid", "自転車通勤手当規程", bike_paid, "支給される", "支給されない", (2, 4)),
    Outcome("bike_amount_6000", "自転車通勤手当規程", bike_amount_6000, "月額6,000円", "月額4,000円", (3,)),
    Outcome("retire_paid", "退職金規程", retire_paid, "支給される", "支給されない", (2, 3)),
)}


def expected(outcome: str, facts: dict) -> str:
    o = OUTCOMES[outcome]
    v = o.fn(facts)
    return UNDECIDED if v is None else (o.yes if v else o.no)


def footer(outcome: str) -> str:
    o = OUTCOMES[outcome]
    return ("答えの最後の2行は、次の形で書いてください。\n"
            "根拠: 判断に使った条（第○条の形で）\n"
            f"結論: 「{o.yes}」「{o.no}」「{UNDECIDED}」のどれか1つ")


def _specs() -> list[tuple[str, str, dict, str]]:
    """(区分, outcome, facts, 文)。"""
    s = []
    # 境目 12
    for h, t in ((Fraction(7, 2), "3時間半"), (Fraction(4), "4時間ちょうど"), (Fraction(9, 2), "4時間半")):
        s.append(("boundary", "room_length_ok", {"hours": h},
                  f"来週の研修で会議室を使いたいです。準備から片付けまで通しで{t}押さえたいのですが、1回の予約でこの長さを取れますか？"))
    for n in (4, 5, 6):
        s.append(("boundary", "book_count_ok", {"books": n},
                  f"図書室で資格試験の参考書をまとめて{n}冊借りたいです。今は1冊も借りておらず、貸出の停止も受けていません。どの本も定価は3千円ほどです。一度にこの冊数を借りられますか？"))
    for km, t in ((Fraction(29, 10), "2.9km"), (Fraction(3), "ちょうど3km"), (Fraction(31, 10), "3.1km")):
        s.append(("boundary", "bike_paid", {"km": km, "insured": True, "bike_days": 20, "workdays": 20},
                  f"家から会社まで片道{t}で、毎日自転車で来ています（今月は出勤20日すべて自転車）。自転車保険にも入っています。自転車通勤手当は出ますか？"))
    for months, t in ((23, "1年11か月"), (24, "ちょうど2年"), (25, "2年1か月")):
        s.append(("boundary", "retire_paid", {"years": months // 12, "disciplinary": False},
                  f"会社都合で退職することになりました。入社してから退職日までで勤続{t}です。懲戒の処分は受けていません。退職金は支給されますか？"))
    # 換算 4
    s.append(("paraphrase", "room_window_ok", {"reserve_on": date(2027, 3, 1), "use_on": date(2027, 3, 31)},
              "2027年3月31日に使う会議室を、今日（2027年3月1日）のうちに予約しておきたいです。もう受け付けてもらえますか？"))
    s.append(("paraphrase", "book_lendable", {"price": 12_000},
              "図書室にある定価1万2千円の専門書を、家で読みたいので借りたいです。この本は貸し出してもらえますか？"))
    s.append(("paraphrase", "bike_paid", {"km": 5, "insured": True, "bike_days": 14, "workdays": 20},
              "片道5kmを自転車で通っていて、保険にも入っています。今月は出勤が20日あり、そのうち14日は自転車、6日は雨で電車でした。今月の自転車通勤手当は出ますか？"))
    s.append(("paraphrase", "retire_paid", {"entry_on": date(2025, 6, 1), "retire_on": date(2027, 3, 31), "disciplinary": False},
              "2025年6月1日に入社し、2027年3月31日付けで自己都合で退職します。懲戒の処分は受けていません。退職金は支給されますか？"))
    # 例外 4
    s.append(("exception", "bike_amount_6000", {"km": 12},
              "自転車通勤手当をもらっています。家から会社までは片道12kmです。手当の月額はいくらになりますか？"))
    s.append(("exception", "retire_paid", {"years": 5, "disciplinary": True},
              "勤続5年の社員が懲戒解雇になりました。この社員に退職金は支給されますか？"))
    s.append(("exception", "room_big_needed", {"participants": 24, "online": 8},
              "全社の定例会議を開きます。参加者は24名ですが、そのうち8名は支社からオンラインで参加します。大会議室を予約しないといけませんか？"))
    s.append(("exception", "book_can_borrow", {"late_days": 10, "excused": True, "days_since_return": 5, "books": 1, "price": 3_000},
              "入院していたため、本の返却が期限より10日遅れてしまいました。図書係には事情を伝えて認めてもらっています。返却したのは5日前です。今は1冊も借りておらず、今度借りたいのは定価3,000円の本1冊です。今、また本を借りられますか？"))
    # 判断できない 4
    s.append(("undecidable", "bike_paid", {"km": 5, "bike_days": 20, "workdays": 20},
              "片道5kmの道のりを、今月は出勤20日すべて自転車で通いました。自転車通勤手当は出ますか？"))
    s.append(("undecidable", "retire_paid", {"disciplinary": False},
              "自己都合で退職する予定です。懲戒の処分は受けていません。退職金は支給されますか？"))
    s.append(("undecidable", "book_can_borrow", {"books": 3, "price_max": 5_000},
              "図書室で3冊借りたいです。どれも定価は5千円以下です。前に返却が遅れた事があったかどうかは覚えていません。今、借りられますか？"))
    s.append(("undecidable", "room_length_ok", {},
              "午後に会議室を予約したいのですが、会議が何時に終わるかはまだ決まっていません。1回の予約で取れる長さに収まりますか？"))
    return s


def build() -> list[dict]:
    rows = []
    for category, outcome, facts, text in _specs():
        o = OUTCOMES[outcome]
        rows.append({
            "id": f"C{len(rows) + 1:02d}",
            "category": category,
            "doc": o.doc,
            "outcome": outcome,
            "question": text + "\n" + footer(outcome),
            "options": [o.yes, o.no, UNDECIDED],
            "expected_conclusion": expected(outcome, facts),
            "expected_articles": list(o.articles),
            "facts": {k: (str(v) if isinstance(v, (date, Fraction)) else v) for k, v in facts.items()},
        })
    return rows
