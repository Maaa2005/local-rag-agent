"""質問の値の取り出し（金額・日付・量・割合）と、規程の引用の言い回しの読み取り。

backend/app/services/numeric_check.py から次の定義を、本文を変えずにそのまま写した（v2 は backend を import できないため）:
  _NEG, _NUM_PHRASES, _DATE_PHRASES, _PRE, _compile, _NUM_RE, _DATE_RE, _BOTH_RE, _match_phrase,
  _holds, _FULL_DATE, _PART_DATE, _MONTH_REF, _YEAR_MONTH, _MONEY, _MONEY_TOKEN, _UNITS, _QTY, _BIZDAY,
  _UNIT_KIND, _MULT, _Qty, _normalize, _dec, _mask, _to_date, _full_dates, _quantities,
  _REL_MONTH, _N_DAYS, _add_months
写しと違う点は 1 つだけ: _NUM_PHRASES に「経過」の 1 行を足した（「勤続6ヶ月を経過し」を >= と読む。
numeric_check には無い。足した行には ADDED の印を付けてある）。
写しが元と一致している事は v2/tests/decide/test_parse.py が確かめる（backend が在る環境のみ）。
標準ライブラリだけで動く。「ここから新規」より下は v2 側で新しく書いた部分。
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any

# ---------------------------------------------------------------- numeric_check.py からの写し

_NEG = r"(?:ない|なかった|なく|ず|なけれ|ません)"

# (正規表現, 向き)。否定形は向きを反転した形で持つ。一番長く合った物を採る。
_NUM_PHRASES: list[tuple[str, str]] = [
    # >
    (r"超(?:え(?:る|た|て)|過|す)", ">"),
    (r"上回(?:る|った|って)", ">"),
    (r"より多い", ">"),
    # >=
    (r"以上", ">="),
    (r"達(?:した|して|する)", ">="),
    (r"経過(?:し|した|して)", ">="),  # ADDED (numeric_check には無い): 「勤続6ヶ月を経過し」= 6ヶ月に達している
    (r"下回ら" + _NEG, ">="),  # 下回らない
    # <
    (r"未満", "<"),
    (r"下回(?:る|った|って)", "<"),
    (r"満たない|満たなかった|満たず", "<"),
    (r"より少ない", "<"),
    (r"届かない|届かなかった|届かず", "<"),
    (r"達し(?:て(?:い)?)?" + _NEG, "<"),  # 達しない (達するの否定)
    # <=
    (r"以下", "<="),
    (r"以内", "<="),
    (r"まで(?:に)?", "<="),
    (r"超え(?:て(?:い)?)?" + _NEG, "<="),  # 超えない
    (r"超過(?:しない|しなかった|せず|しなく)", "<="),
    (r"超さ(?:ない|なかった|ず)", "<="),
    (r"上回ら" + _NEG, "<="),  # 上回らない
]

_DATE_PHRASES: list[tuple[str, str]] = [
    # >
    (r"過ぎ(?:た|て)", ">"),
    (r"より後", ">"),
    (r"遅れ(?:た|て)", ">"),
    (r"経過した後", ">"),
    # >=
    (r"以降", ">="),
    (r"以後", ">="),
    (r"から(?!数え)", ">="),
    # <
    (r"より前", "<"),
    (r"の前日まで(?:に)?", "<"),
    (r"先立(?:って|ち)", "<"),
    # <=
    (r"まで(?:に)?", "<="),
    (r"以前", "<="),
    (r"以内", "<="),
    (r"過ぎ" + _NEG, "<="),  # を過ぎない
    (r"遅れ" + _NEG, "<="),
]

_PRE = r"(?:ちょうど|きっかり)?(?:を|が|は|に|も)?"


def _compile(table: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(_PRE + p), op) for p, op in table]


_NUM_RE = _compile(_NUM_PHRASES)
_DATE_RE = _compile(_DATE_PHRASES)
_BOTH_RE = _NUM_RE + _DATE_RE


def _match_phrase(
    text: str, pos: int, table: list[tuple[re.Pattern[str], str]]
) -> tuple[int, str] | None:
    """pos から始まる言い回し。一番長く合った物の (終わり位置, 向き)。"""
    best: tuple[int, str] | None = None
    for pat, op in table:
        m = pat.match(text, pos)
        if m and m.end() > pos and (best is None or m.end() > best[0]):
            best = (m.end(), op)
    return best


_SYM = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}
_NEG_OP = {">": "≤", ">=": "<", "<": "≥", "<=": ">"}


def _holds(value: Any, op: str, boundary: Any) -> bool:
    if op == ">":
        return value > boundary
    if op == ">=":
        return value >= boundary
    if op == "<":
        return value < boundary
    return value <= boundary


_FULL_DATE = re.compile(
    r"(?<!\d)(?P<y>\d{4})\s?[年/\-.]\s?(?P<m>\d{1,2})\s?[月/\-.]\s?(?P<d>\d{1,2})\s?日?(?!\d)"
)
_PART_DATE = re.compile(r"(?<![\d/\-.])(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
_MONTH_REF = re.compile(r"(?:(?P<y>\d{4})年)?(?P<m>\d{1,2})月(?:分|度)")
_YEAR_MONTH = re.compile(r"\d{4}年\d{1,2}月")

_MONEY = re.compile(r"(?P<t>(?:\d[\d,]*(?:\.\d+)?(?:億|万|千)?)+)円")
_MONEY_TOKEN = re.compile(r"(\d[\d,]*(?:\.\d+)?)(億|万|千)?")
_UNITS = (
    r"%|割|日間|日|回|時間|人|名|km|キロメートル|キロ|件|個|台|枚|歳|年間|年|"
    r"[かヶカケヵ]月間?|週間|点|kg"
)
_QTY = re.compile(r"(?P<n>\d[\d,]*(?:\.\d+)?)(?P<u>" + _UNITS + r")")
_BIZDAY = re.compile(r"\d+営業日")

_UNIT_KIND = {
    "%": "%", "割": "%", "日間": "日", "日": "日", "回": "回", "時間": "時間",
    "人": "人", "名": "人", "km": "km", "キロメートル": "km", "キロ": "km",
    "件": "件", "個": "個", "台": "台", "枚": "枚", "歳": "歳", "年間": "年",
    "年": "年", "週間": "週間", "点": "点", "kg": "kg",
}
_KIND_LABEL = {"%": "%", "円": "円"}
# 単位が違っても同じ家族なら「単位がそろわない」として保留の理由にする
_TIME_FAMILY = {"日", "年", "か月", "週間", "時間"}
_MULT = {"億": Decimal(10**8), "万": Decimal(10**4), "千": Decimal(10**3), None: Decimal(1), "": Decimal(1)}


@dataclass
class _Qty:
    kind: str
    value: Decimal
    start: int
    end: int
    bizday: bool = False


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _dec(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = "□"
    return "".join(chars)


def _to_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _full_dates(text: str) -> list[tuple[int, int, date]]:
    out = []
    for m in _FULL_DATE.finditer(text):
        d = _to_date(int(m["y"]), int(m["m"]), int(m["d"]))
        if d is not None:
            out.append((m.start(), m.end(), d))
    return out


def _quantities(text: str, masked_spans: list[tuple[int, int]]) -> list[_Qty]:
    """数量を取る。日付などでマスクした範囲は見ない。"""
    t = _mask(text, masked_spans)
    out: list[_Qty] = []
    taken: list[tuple[int, int]] = []

    for m in _BIZDAY.finditer(t):
        out.append(_Qty("営業日", Decimal(0), m.start(), m.end(), bizday=True))
        taken.append((m.start(), m.end()))

    for m in _MONEY.finditer(t):
        total = Decimal(0)
        for tok in _MONEY_TOKEN.finditer(m["t"]):
            total += _dec(tok[1]) * _MULT[tok[2]]
        out.append(_Qty("円", total, m.start(), m.end()))
        taken.append((m.start(), m.end()))

    for m in _QTY.finditer(t):
        if any(a <= m.start() < b for a, b in taken):
            continue
        u = m["u"]
        rest = t[m.end():]
        if u == "点" and rest.startswith("満点"):
            continue  # 「100点満点」の分母は比べる値ではない
        if u == "割":
            if rest[:1] in ("引", "増"):
                continue
            val = _dec(m["n"]) * 10
            mm = re.match(r"(\d)分", rest)
            if mm:
                val += Decimal(mm[1])
                out.append(_Qty("%", val, m.start(), m.end() + mm.end()))
                continue
            out.append(_Qty("%", val, m.start(), m.end()))
            continue
        if u in ("年", "年間"):
            if rest.startswith("度") or _dec(m["n"]) >= 1900:
                continue
        if re.fullmatch(r"[かヶカケヵ]月間?", u):
            kind = "か月"
        else:
            kind = _UNIT_KIND[u]
        out.append(_Qty(kind, _dec(m["n"]), m.start(), m.end()))
    out.sort(key=lambda q: q.start)
    return out


_REL_MONTH = re.compile(r"(?P<w>翌々月|翌月|当月)(?:(?P<eom>末日|末)|(?P<d>\d{1,2})日)")
_N_DAYS_BEFORE = re.compile(r"(?P<n>\d+)日前")
_N_DAYS = re.compile(r"(?P<n>\d+)日間?")


def _add_months(y: int, m: int, k: int) -> tuple[int, int]:
    idx = y * 12 + (m - 1) + k
    return idx // 12, idx % 12 + 1


# ---------------------------------------------------------------- ここから新規（v2 側で書いた部分）

CLAUSE_BREAKS = "。！？!?\n、，,"


@dataclass(frozen=True)
class Cand:
    """質問の中の 1 つの値の候補。kind は 'date'・'円'・'%'・'日'・'か月'・'年'・'km' など。"""
    kind: str
    value: Any  # date か Fraction
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Extracted:
    text: str  # 正規化し、書式の指示（「最後の行」以降）を除いた質問
    cands: tuple[Cand, ...]  # 開始位置順
    unresolved_date: bool  # 年の分からない「M月D日」がある
    business_days: bool  # 営業日が出てくる


def clean_question(question: str) -> str:
    """NFKC に正規化し、書式の指示（「最後の行には…」）以降は値を取る対象から外す。"""
    text = _normalize(question)
    cut = text.find("最後の行")
    return text[:cut] if cut != -1 else text


def to_fraction(v: Decimal | int | str) -> Fraction:
    return Fraction(Decimal(v)) if not isinstance(v, Fraction) else v


# 数の直後にこれが付く物は「単位のない数」ではない（日付・金額・順序・番号などの一部）。単位の一覧に無い物もここで外す
_COUNTER_AFTER = "年月日時分秒円割%点人名回件個台枚歳週号条項章節番階つ倍期度万千億"
_BARE_NUM = re.compile(
    r"(?<![\d.,:/\-A-Za-z_第])(?P<n>\d+(?:,\d{3})*(?:\.\d+)?)(?![\d:/\-A-Za-z_" + _COUNTER_AFTER + r"])(?!\.\d)"
)


def _bare_numbers(text: str, consumed: list[tuple[int, int]]) -> list[tuple[int, int, Decimal]]:
    """単位の付かない数 [(開始, 終わり, 値)]。日付・金額・量などで読めた範囲（consumed）と重なる物は除く。"""
    out = []
    for m in _BARE_NUM.finditer(text):
        if any(a < m.end() and m.start() < b for a, b in consumed):
            continue
        out.append((m.start(), m.end(), _dec(m["n"])))
    return out


def extract(question: str, base_year: int | None = None, bare: bool = False) -> Extracted:
    """質問から日付・金額・量・割合の候補を全部取る。役割（どの事実か）はここでは決めない。

    年の無い「M月D日」は、質問の中の完全な日付の年が 1 種類だけ（または base_year の指定）のときだけ日付にする。
    bare=True の時は、単位の付かない数も kind="" の候補として取る（記録に単位のない量の事実がある時だけ使う）。
    """
    text = clean_question(question)
    spans: list[tuple[int, int]] = []
    fulls = _full_dates(text)
    spans += [(a, b) for a, b, _ in fulls]
    years = {d.year for _, _, d in fulls}
    if base_year is not None:
        years = {base_year}
    unresolved = False
    partial: list[tuple[int, int, date]] = []
    for m in _PART_DATE.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        spans.append((m.start(), m.end()))
        if len(years) == 1:
            d = _to_date(next(iter(years)), int(m["m"]), int(m["d"]))
            if d is not None:
                partial.append((m.start(), m.end(), d))
                continue
        unresolved = True
    for m in _MONTH_REF.finditer(text):
        if not any(a <= m.start() < b for a, b in spans):
            spans.append((m.start(), m.end()))
    for m in _YEAR_MONTH.finditer(text):
        if not any(a <= m.start() < b for a, b in spans):
            spans.append((m.start(), m.end()))
    cands: list[Cand] = []
    for a, b, d in sorted(fulls + partial, key=lambda x: x[0]):
        cands.append(Cand("date", d, a, b, text[a:b]))
    qtys = _quantities(text, spans)
    for q in qtys:
        if q.bizday:
            continue
        cands.append(Cand(q.kind, to_fraction(q.value), q.start, q.end, text[q.start:q.end]))
    if bare:
        consumed = spans + [(q.start, q.end) for q in qtys]
        for a, b, v in _bare_numbers(text, consumed):
            cands.append(Cand("", to_fraction(v), a, b, text[a:b]))
    cands.sort(key=lambda c: (c.start, c.end))
    return Extracted(text, tuple(cands), unresolved, any(q.bizday for q in qtys) or "営業日" in text)


# ---------------------------------------------------------------- 単位

_FAMILY = {"年": "time", "か月": "time", "日": "time", "週間": "time", "時間": "time"}
# 換算表を持つ組だけ換算する。それ以外（年と日など）は換算せず「単位のずれ」として保留する。
_CONVERT = {
    ("年", "か月"): Fraction(12),
    ("か月", "年"): Fraction(1, 12),
    ("週間", "日"): Fraction(7),
    ("日", "週間"): Fraction(1, 7),
}


def family(kind: str) -> str:
    return _FAMILY.get(kind, kind)


def convert(value: Fraction, src: str, dst: str) -> Fraction | None:
    """src の単位の値を dst の単位にする。換算表に無ければ None。"""
    if src == dst:
        return value
    f = _CONVERT.get((src, dst))
    return None if f is None else value * f


# ---------------------------------------------------------------- 表示

def fmt_number(v: Fraction) -> str:
    if v.denominator == 1:
        return f"{v.numerator:,}" if abs(v) >= 10000 else str(v.numerator)
    d = Decimal(v.numerator) / Decimal(v.denominator)
    s = format(d.quantize(Decimal("0.001")).normalize(), "f")
    return s


def fmt_date(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"


def fmt_value(kind: str, v: Any) -> str:
    if isinstance(v, date):
        return fmt_date(v)
    n = fmt_number(v)
    if kind == "円":
        return f"{n}円"
    return f"{n}{kind}" if kind else n


# ---------------------------------------------------------------- 規程の引用の読み取り（F7 の検証用）

def quote_tokens(quote: str, kind: str, unit: str | None = None) -> list[tuple[int, Any]]:
    """引用文の中の値の候補 [(終わり位置, 値)] を、質問側と同じ読み取りで取る。

    kind: 'date'（完全な日付）・'円'・'%'・単位名（'年'・'か月'・'日'・'km' など）
    """
    text = _normalize(quote)
    if kind == "date":
        return [(b, d) for _, b, d in _full_dates(text)]
    spans = [(a, b) for a, b, _ in _full_dates(text)]
    out = []
    for q in _quantities(text, spans):
        if q.bizday:
            continue
        if q.kind == kind:
            out.append((q.end, to_fraction(q.value)))
    return out


def quote_number_tokens(quote: str) -> list[tuple[int, Any]]:
    """単位のない量の境目用: 引用の中の数 [(終わり位置, 値)]。単位が付いていれば単位の後ろが終わり位置。

    日付の一部の数は除く。単位の無い数（「5以上」の 5）も取る。
    """
    text = _normalize(quote)
    spans = [(a, b) for a, b, _ in _full_dates(text)]
    spans += [(m.start(), m.end()) for m in _PART_DATE.finditer(text)]
    spans += [(m.start(), m.end()) for m in _MONTH_REF.finditer(text)]
    spans += [(m.start(), m.end()) for m in _YEAR_MONTH.finditer(text)]
    qtys = [q for q in _quantities(text, spans) if not q.bizday]
    out = [(q.end, to_fraction(q.value)) for q in qtys]
    consumed = spans + [(q.start, q.end) for q in _quantities(text, spans)]
    out += [(b, to_fraction(v)) for _, b, v in _bare_numbers(text, consumed)]
    out.sort(key=lambda t: t[0])
    return out


def phrase_op(quote: str, pos: int, kind: str) -> str | None:
    """pos（値の終わり）の直後の比べ方の言い回し。日付なら日付の表、それ以外は数の表で読む。"""
    text = _normalize(quote)
    r = _match_phrase(text, pos, _DATE_RE if kind == "date" else _NUM_RE)
    return None if r is None else r[1]


def month_day_tokens(quote: str) -> list[tuple[int, int, int, str, int]]:
    """「翌月10日」の類 [(終わり位置, 月の足し幅, 日, 'day'|'eom', 0)]。"""
    text = _normalize(quote)
    out = []
    for m in _REL_MONTH.finditer(text):
        k = {"当月": 0, "翌月": 1, "翌々月": 2}[m["w"]]
        if m["eom"]:
            out.append((m.end(), k, 0, "eom", 0))
        else:
            out.append((m.end(), k, int(m["d"]), "day", 0))
    return out


def day_count_tokens(quote: str) -> list[tuple[int, int]]:
    """「14日」「60日以内」の類 [(終わり位置, 日数)]。「翌月10日」「3営業日」「N日前」は除く。"""
    text = _normalize(quote)
    consumed = [(m.start(), m.end()) for m in _REL_MONTH.finditer(text)]
    consumed += [(m.start(), m.end()) for m in _BIZDAY.finditer(text)]
    out = []
    for m in _N_DAYS.finditer(text):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        if m.start() > 0 and text[m.start() - 1] in "月営":
            continue
        out.append((m.end(), int(m["n"])))
    return out


def add_months(d: date, k: int) -> tuple[int, int]:
    return _add_months(d.year, d.month, k)


def holds(value: Any, op: str, boundary: Any) -> bool:
    return _holds(value, op, boundary)
