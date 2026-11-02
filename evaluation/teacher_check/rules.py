"""教師の確認（life #24）の正解を、規程の条件からコードで出す。

corpus/ の 6 規程の数の条件を 1 つずつ関数にした。問の文ではなく、問に入れた事実（facts）だけから結論を出す。
分からない事実は None で渡し、3 値（True / False / None）で組み合わせる。None が残れば「判断できない」。

- AND: 1 つでも False なら False。False が無く None があれば None。全部 True なら True
- OR: 1 つでも True なら True。True が無く None があれば None。全部 False なら False
- 比べる時は整数か Fraction で行い、小数の誤差を入れない
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction

UNDECIDED = "判断できない"
Tri = bool | None


def all3(*vals: Tri) -> Tri:
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


def any3(*vals: Tri) -> Tri:
    if any(v is True for v in vals):
        return True
    if any(v is None for v in vals):
        return None
    return False


def not3(v: Tri) -> Tri:
    return None if v is None else not v


def cmp(value, op: str, threshold) -> Tri:
    """value が分からなければ None。op は ">=" ">" "<=" "<"。"""
    if value is None:
        return None
    return {">=": value >= threshold, ">": value > threshold, "<=": value <= threshold, "<": value < threshold}[op]


def full_years(start: date | None, end: date | None) -> int | None:
    """start から end までの満年数（誕生日・入社日の応当日を迎えたら 1 年）。"""
    if start is None or end is None:
        return None
    return end.year - start.year - ((end.month, end.day) < (start.month, start.day))


def base_date(year: int) -> date:
    """永年勤続表彰規程・住宅手当規程の基準日（毎年 4 月 1 日）。"""
    return date(year, 4, 1)


# ─── 社内貸付規程 ──────────────────────────────────────────────
def loan_eligible(f: dict) -> Tri:                        # 第2条: 勤続12か月以上
    return cmp(f.get("tenure_months"), ">=", 12)


def loan_within_limit(f: dict) -> Tri:                    # 第3条: 200万円まで。住宅の購入なら500万円まで
    amount, housing = f.get("amount"), f.get("housing_purchase")
    return any3(cmp(amount, "<=", 2_000_000), all3(housing, cmp(amount, "<=", 5_000_000)))


def loan_repay_ok(f: dict) -> Tri:                        # 第4条: 60か月を超えない
    return cmp(f.get("repay_months"), "<=", 60)


def loan_review_needed(f: dict) -> Tri:                   # 第5条: 申込額 > 年収 / 3
    amount, income = f.get("amount"), f.get("annual_income")
    if amount is None or income is None:
        return None
    return Fraction(amount) > Fraction(income, 3)


def loan_available(f: dict) -> Tri:                       # 第2条 かつ 第3条
    return all3(loan_eligible(f), loan_within_limit(f))


# ─── 副業規程 ──────────────────────────────────────────────────
MAIN_HOURS = 40                                           # 第2条


def side_permission_needed(f: dict) -> Tri:               # 第3条: 週10時間を超える
    return cmp(f.get("side_hours"), ">", 10)


def side_allowed(f: dict) -> Tri:                         # 第4条: 合計55時間以上は不可。自宅の執筆・講演の準備で時間を自由に決められるものは除く
    hours = f.get("side_hours")
    over = None if hours is None else cmp(MAIN_HOURS + hours, ">=", 55)
    kind = f.get("kind")
    kind_ok = None if kind is None else kind in ("執筆", "講演の準備")
    exception = all3(f.get("at_home"), kind_ok, f.get("self_scheduled"))
    return any3(not3(over), exception)


def side_report_needed(f: dict) -> Tri:                   # 第5条: 月額20万円を超える
    return cmp(f.get("monthly_income"), ">", 200_000)


# ─── 永年勤続表彰規程 ──────────────────────────────────────────
def tenure_years(f: dict) -> int | None:                  # 第2条: 入社日から基準日までの満年数
    if f.get("tenure_years") is not None:
        return f["tenure_years"]
    return full_years(f.get("entry_date"), f.get("base_date"))


def long_service_leave(f: dict) -> Tri:                   # 第3条 かつ 第4条
    return all3(cmp(tenure_years(f), ">=", 10), not3(cmp(f.get("leave_days"), ">", 365)))


# ─── 住宅手当規程 ──────────────────────────────────────────────
def age_at_base(f: dict) -> int | None:
    if f.get("age") is not None:
        return f["age"]
    return full_years(f.get("birth_date"), f.get("base_date"))


def housing_paid(f: dict) -> Tri:                         # 第2条 一・二 かつ 第3条
    rent = f.get("monthly_rent")
    rent_ok = cmp(rent, ">=", 50_000)
    age_ok = any3(cmp(age_at_base(f), "<", 35), f.get("dependent_child"))
    near = cmp(f.get("distance_m"), "<", 2_000)
    return all3(rent_ok, age_ok, not3(near))


# ─── 取引先審査規程 ────────────────────────────────────────────
def trade_forbidden(f: dict) -> Tri:                      # 第2条: 自己資本比率が20%を下回る会社とは始めない
    return cmp(f.get("equity_ratio_pct"), "<", 20)


def trade_review_needed(f: dict) -> Tri:                  # 第3条: 300万円以上。上場会社は除く
    return all3(cmp(f.get("deal_amount"), ">=", 3_000_000), not3(f.get("listed")))


def trade_prepay_needed(f: dict) -> Tri:                  # 第4条: 設立から3年未満
    return cmp(f.get("years_since_founding"), "<", 3)


def trade_stop(f: dict) -> Tri:                           # 第5条: 入金がない日数が30日を超える
    return cmp(f.get("overdue_days"), ">", 30)


def trade_credit_ok(f: dict) -> Tri:                      # 第6条: 与信限度額 ≤ 純資産 × 10%
    limit, assets = f.get("credit_limit"), f.get("net_assets")
    if limit is None or assets is None:
        return None
    return Fraction(limit) <= Fraction(assets, 10)


# ─── 懇親会費補助規程 ──────────────────────────────────────────
def party_size_ok(f: dict) -> Tri:                        # 第2条: 10名以上。部署をまたぐ交流会は5名以上
    n = f.get("participants")
    return any3(cmp(n, ">=", 10), all3(f.get("cross_dept"), cmp(n, ">=", 5)))


def party_deadline_ok(f: dict) -> Tri:                    # 第3条: 開催日の14日前までに申請
    applied, event = f.get("applied_on"), f.get("event_on")
    if applied is None or event is None:
        return None
    return applied <= event - timedelta(days=14)


def party_full_subsidy(f: dict) -> Tri:                   # 第4条: 1人あたり5,000円以下なら全額
    return cmp(f.get("per_person"), "<=", 5_000)


def party_alcohol_ok(f: dict) -> Tri:                     # 第5条: 酒類の代金が総額の5割を超えれば補助しない
    alcohol, total = f.get("alcohol"), f.get("total")
    if alcohol is None or total is None:
        return None
    return not (Fraction(alcohol) > Fraction(total, 2))


def party_subsidy(f: dict) -> Tri:                        # 第2条 かつ 第3条 かつ 第5条
    return all3(party_size_ok(f), party_deadline_ok(f), party_alcohol_ok(f))


# ─── 問の型 ────────────────────────────────────────────────────
@dataclass(frozen=True)
class Outcome:
    """問で聞く結論。fn が True なら yes、False なら no、None なら判断できない。"""
    name: str
    doc: str
    fn: Callable[[dict], Tri]
    yes: str
    no: str
    articles: tuple[int, ...]   # 判断に使う条（根拠の照合に使う）


OUTCOMES = {o.name: o for o in (
    Outcome("loan_eligible", "社内貸付規程", loan_eligible, "対象", "対象外", (2,)),
    Outcome("loan_within_limit", "社内貸付規程", loan_within_limit, "範囲内", "範囲外", (3,)),
    Outcome("loan_repay_ok", "社内貸付規程", loan_repay_ok, "認められる", "認められない", (4,)),
    Outcome("loan_review_needed", "社内貸付規程", loan_review_needed, "必要", "不要", (5,)),
    Outcome("loan_available", "社内貸付規程", loan_available, "受けられる", "受けられない", (2, 3)),
    Outcome("side_permission_needed", "副業規程", side_permission_needed, "許可が必要", "届出のみ", (3,)),
    Outcome("side_allowed", "副業規程", side_allowed, "認められる", "認められない", (2, 4)),
    Outcome("side_report_needed", "副業規程", side_report_needed, "必要", "不要", (5,)),
    Outcome("long_service_leave", "永年勤続表彰規程", long_service_leave, "付与される", "付与されない", (2, 3, 4)),
    Outcome("housing_paid", "住宅手当規程", housing_paid, "支給される", "支給されない", (2, 3)),
    Outcome("trade_forbidden", "取引先審査規程", trade_forbidden, "禁じられている", "禁じられていない", (2,)),
    Outcome("trade_review_needed", "取引先審査規程", trade_review_needed, "必要", "不要", (3,)),
    Outcome("trade_prepay_needed", "取引先審査規程", trade_prepay_needed, "必要", "不要", (4,)),
    Outcome("trade_stop", "取引先審査規程", trade_stop, "停止する", "停止しない", (5,)),
    Outcome("trade_credit_ok", "取引先審査規程", trade_credit_ok, "設定できる", "設定できない", (6,)),
    Outcome("party_size_ok", "懇親会費補助規程", party_size_ok, "対象", "対象外", (2,)),
    Outcome("party_deadline_ok", "懇親会費補助規程", party_deadline_ok, "受け付けられる", "受け付けられない", (3,)),
    Outcome("party_full_subsidy", "懇親会費補助規程", party_full_subsidy, "全額補助される", "全額補助されない", (4,)),
    Outcome("party_alcohol_ok", "懇親会費補助規程", party_alcohol_ok, "補助される", "補助されない", (5,)),
    Outcome("party_subsidy", "懇親会費補助規程", party_subsidy, "受けられる", "受けられない", (2, 3, 5)),
)}


def expected_conclusion(outcome: str, facts: dict) -> str:
    o = OUTCOMES[outcome]
    v = o.fn(facts)
    return UNDECIDED if v is None else (o.yes if v else o.no)
