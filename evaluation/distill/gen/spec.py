"""生成器の土台: 比べる語・値の正規化・使ってはいけない題材と境目（life #45・docs/verified-distill-plan.md §1）。

比べる語は記号にそろえる。否定を介した同じ条件（「80%を下回ると不支給」と「80%以上で支給」）は、
境目の値と単位が同じなので同じ境目として扱う（向きは問わない）。
"""
from __future__ import annotations

from fractions import Fraction

# 比べる語 → (記号, 規程の中の書き方)。書き方の {t} に境目の値の文字が入る
WORDS: dict[str, tuple[str, str]] = {
    "以上": (">=", "{t}以上の"),
    "以下": ("<=", "{t}以下の"),
    "未満": ("<", "{t}未満の"),
    "超える": (">", "{t}を超える"),
    "以内": ("<=", "{t}以内の"),
    "を超えない": ("<=", "{t}を超えない"),
    "を下回る": ("<", "{t}を下回る"),
    "に満たない": ("<", "{t}に満たない"),
    "上限とする": ("<=", "{t}を上限とする"),
}


def holds(value, op: str, threshold) -> bool:
    """value と境目を比べる。teacher_check.rules.cmp と同じ定義（別の実装）を試験で突き合わせる。"""
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    raise ValueError(op)


# 単位の次元と、基準の単位への倍率
UNITS: dict[str, tuple[str, Fraction]] = {
    "円": ("yen", Fraction(1)), "万円": ("yen", Fraction(10_000)),
    "%": ("ratio", Fraction(1, 100)), "割": ("ratio", Fraction(1, 10)), "比": ("ratio", Fraction(1)),
    "日": ("day", Fraction(1)), "か月": ("month", Fraction(1)), "年": ("year", Fraction(1)),
    "時間": ("hour", Fraction(1)), "分": ("hour", Fraction(1, 60)),
    "km": ("km", Fraction(1)), "m": ("km", Fraction(1, 1000)),
    "名": ("person", Fraction(1)), "人": ("person", Fraction(1)),
    "件": ("count", Fraction(1)), "回": ("count", Fraction(1)), "冊": ("count", Fraction(1)), "台": ("count", Fraction(1)),
    "歳": ("age", Fraction(1)), "点": ("score", Fraction(1)), "kg": ("kg", Fraction(1)), "枚": ("count", Fraction(1)),
}


def normalize(value, unit: str) -> tuple[str, Fraction]:
    dim, k = UNITS[unit]
    return dim, Fraction(value) * k


# 封 A・封 B・封 C の境目（値と単位）。docs/verified-distill-plan.md §1 の 3
SEALED_THRESHOLDS: list[tuple[object, str]] = [
    # 封 A（numeric_boundary.jsonl・evaluation/corpus）
    (10, "万円"), (8, "割"), (80, "%"),
    # 封 B（teacher_check/corpus）
    (200, "万円"), (500, "万円"), (60, "か月"), (12, "か月"), (10, "時間"), (55, "時間"), (20, "万円"),
    (10, "年"), (365, "日"), (5, "万円"), (35, "歳"), (2, "km"), (300, "万円"), (3, "年"), (20, "%"),
    (30, "日"), (10, "%"), (5000, "円"), (10, "名"), (5, "名"), (14, "日"), (5, "割"), (7, "日"),
    # 封 C（distill/sealed_c/corpus）
    (30, "日"), (4, "時間"), (20, "名"), (15, "分"), (5, "冊"), (21, "日"), (1, "万円"), (3, "km"),
    (10, "km"), (7, "割"), (2, "年"), (25, "年"), (60, "日"), (4000, "円"), (6000, "円"), (8, "割"),
]
SEALED_NORMALIZED = {normalize(v, u) for v, u in SEALED_THRESHOLDS}
# どの単位でも使わない割合と期限（§1 の 3）
BANNED_RATIOS = {Fraction(4, 5), Fraction(1, 2), Fraction(1, 3)}
BANNED_DEADLINE_PHRASES = ("翌月10日", "14日前")


def threshold_allowed(value, unit: str) -> bool:
    dim, v = normalize(value, unit)
    if (dim, v) in SEALED_NORMALIZED:
        return False
    if dim == "ratio" and v in BANNED_RATIOS:
        return False
    return True


# 題材の一覧（封 A・B・C と今までの評価の文書）。生成器の家族の題材と規程名に、この語を含めない
BLOCKED_TOPICS = (
    "経費", "精算", "出張", "旅費", "在宅", "就業規則", "情報セキュリティ", "研修", "育児", "介護", "役員報酬",
    "人事評価", "採用", "面接", "目標管理", "M&A", "経営計画", "備品", "福利厚生", "慶弔", "見舞", "時間外",
    "社用車", "資格", "貸付", "副業", "永年勤続", "住宅手当", "取引先", "与信", "懇親会", "会議室", "図書",
    "自転車", "退職金", "有給", "賞与", "部門長",
)


def topic_allowed(text: str) -> bool:
    return not any(t in text for t in BLOCKED_TOPICS)
