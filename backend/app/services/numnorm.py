"""数・日付の表記をそろえる前処理 (Issue #16 の「数の正規化」)。

質問と規程の文に出る数・日付の書き方 (全角・漢数字・割分厘・万千億・和暦・
区切りの日付) を、モデルに渡す前とコードで比べる前に 1 つの形へそろえる。
標準ライブラリだけで動く純関数。

- ``normalize_numbers(text)``: 文を書き換えて返す。冪等。数・日付以外は変えない。
- ``parse_number(s)``: 1 つの数の表記を ``Decimal`` にする。読めなければ ``None``。

書き換えの順 (``normalize_numbers``):
  1. NFKC
  2. 漢数字 → 算用数字 (直後が数え方の語の時だけ)
  3. 割・分・厘 → %
  4. 万・千・億つきの数 (と円の付く数) → 3 桁区切り
  5. 和暦 → 西暦
  6. 区切りの日付 → 年月日
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal

__all__ = ["normalize_numbers", "parse_number"]

# ---------------------------------------------------------------- 共通の部品

_DIGIT = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNIT = {"十": 10, "百": 100, "千": 1000}
_KN = "〇零一二三四五六七八九十百千万億"  # 漢数字の並びに使う字
_KUNITS = "十百千万億"

# 数え方の語。長い物を先に置く。「分」は時間と紛れるので入れない。
_COUNTER = (
    r"(?:日間|日|年間|年|[かヶカケヵ箇]月間?|週間|時間|人|名|件|回|歳|才|点|個|枚|本|台|社"
    r"|km|キロ|割|%|パーセント|円)"
)

# 算用数字の 1 つの数 (3 桁区切りか、区切り無し。小数も可)
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MULT_VALUE = {"億": Decimal(10**8), "万": Decimal(10**4), "千": Decimal(10**3)}
_MULT_RANK = {"億": 3, "万": 2, "千": 1}


def _fmt(v: Decimal) -> str:
    """指数表記にならない文字列。"""
    return format(v.normalize(), "f")


def _fmt_comma(v: Decimal) -> str:
    return format(v.normalize(), ",f")


# ---------------------------------------------------------------- 漢数字

def _under_10000(s: str) -> int | None:
    """万未満の漢数字 (千百十と一〜九)。「二十五」「三千五百」。不正なら None。"""
    total = 0
    cur: int | None = None
    last = 10000
    for ch in s:
        if ch in _DIGIT and _DIGIT[ch] > 0:
            if cur is not None:
                return None
            cur = _DIGIT[ch]
        elif ch in _SMALL_UNIT:
            v = _SMALL_UNIT[ch]
            if v >= last:
                return None
            total += (1 if cur is None else cur) * v
            cur = None
            last = v
        else:
            return None
    if cur is not None:
        if last not in (10000, 10):  # 「三千五」のような略し方は読まない
            return None
        total += cur
    return total if total > 0 else None


_KANJI_SPLIT = re.compile(r"(?:([^億万]+)億)?(?:([^億万]+)万)?([^億万]*)")


def _kanji_value(s: str) -> int | None:
    """漢数字の並びの値。「二〇二六」は桁ごとに読む。読めなければ None。"""
    if not s or any(c not in _KN for c in s):
        return None
    if not any(c in _KUNITS for c in s):
        return int("".join(str(_DIGIT[c]) for c in s))
    m = _KANJI_SPLIT.fullmatch(s)
    if not m:
        return None
    total = 0
    for part, mul in ((m[1], 10**8), (m[2], 10**4), (m[3], 1)):
        if part:
            v = _under_10000(part)
            if v is None:
                return None
            total += v * mul
    return total if total > 0 else None


# 漢数字の並び + 数え方の語。数・何・幾の直後 (数十人・数万円) と、算用数字の直後
# (5千円 の 千) は対象外。「月」は日付 (十一月十五日) の時だけ。
_KANJI_RE = re.compile(
    rf"(?<![\d.,{_KN}数何幾])(?P<n>[{_KN}]+)"
    rf"(?:(?P<c>{_COUNTER})(?!十色)|(?P<mo>月)(?=[\d{_KN}]+日))"
)


def _sub_kanji(m: re.Match[str]) -> str:
    v = _kanji_value(m["n"])
    if v is None:
        return m[0]
    if m["mo"]:
        return f"{v}月" if 1 <= v <= 12 else m[0]
    n = m["n"]
    # 千・万・億を含む大きい数は 3 桁区切りにして、算用数字の「5千」と形をそろえる
    if v >= 1000 and any(c in n for c in "千万億"):
        return f"{v:,}{m['c']}"
    return f"{v}{m['c']}"


# ---------------------------------------------------------------- 割分厘

_D1 = r"[\d一二三四五六七八九〇零]"
_WARI_RE = re.compile(
    rf"(?<![\d.])(?P<w>\d+(?:\.\d+)?)割"
    rf"(?:(?P<b>{_D1})分(?:(?P<r>{_D1})厘)?|(?P<r2>{_D1})厘)?"
)


def _d1(ch: str) -> int:
    return _DIGIT[ch] if ch in _DIGIT else int(ch)


def _wari_value(w: Decimal, b: str | None, r: str | None) -> Decimal:
    v = w * 10
    if b:
        v += _d1(b)
    if r:
        v += Decimal(_d1(r)) / 10
    return v


def _sub_wari(m: re.Match[str]) -> str:
    v = _wari_value(Decimal(m["w"]), m["b"], m["r"] or m["r2"])
    return f"{_fmt(v)}%"


# ---------------------------------------------------------------- 万千億と桁区切り

_MONEY_RE = re.compile(
    rf"(?<!\d)(?<!\d[.,])(?=\d)"
    rf"(?P<t>(?:{_NUM}[億万千])*(?:{_NUM})?)(?P<u>{_COUNTER})"
)
_TOKEN_RE = re.compile(r"(?P<n>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<m>[億万千])?")


def _chain_value(t: str) -> tuple[Decimal, bool] | None:
    """「1億2000万」「3.5万」「100,000」の値と、万千億を含んだか。不正なら None。"""
    total = Decimal(0)
    prev = 4
    has_mult = False
    pos = 0
    for k in _TOKEN_RE.finditer(t):
        if k.start() != pos:
            return None
        pos = k.end()
        mult = k["m"]
        rank = _MULT_RANK[mult] if mult else 0
        if rank >= prev:
            return None
        prev = rank
        has_mult = has_mult or bool(mult)
        total += Decimal(k["n"].replace(",", "")) * (_MULT_VALUE[mult] if mult else 1)
    if pos != len(t) or not t:
        return None
    return total, has_mult


def _sub_money(m: re.Match[str]) -> str:
    r = _chain_value(m["t"])
    if r is None:
        return m[0]
    total, has_mult = r
    if not has_mult and m["u"] != "円":
        return m[0]  # 2026年 のような数に区切りは付けない
    return f"{_fmt_comma(total)}{m['u']}"


# ---------------------------------------------------------------- 和暦

_WAREKI_BASE = {"令和": 2018, "平成": 1988, "昭和": 1925}
_WAREKI_RE = re.compile(r"(令和|平成|昭和)(元|\d{1,3})年")


def _sub_wareki(m: re.Match[str]) -> str:
    n = 1 if m[2] == "元" else int(m[2])
    if n < 1:
        return m[0]
    return f"{_WAREKI_BASE[m[1]] + n}年"


# ---------------------------------------------------------------- 区切りの日付

_SEP_DATE_RE = re.compile(
    r"(?<![\d./])(?P<y>\d{4})(?P<s>[/\-.])(?P<m>\d{1,2})(?P=s)(?P<d>\d{1,2})"
    r"(?!\d)(?![./]\d)"
)


def _sub_sep_date(m: re.Match[str]) -> str:
    y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
    try:
        date(y, mo, d)
    except ValueError:
        return m[0]
    return f"{y}年{mo}月{d}日"


# ---------------------------------------------------------------- 本体

def normalize_numbers(text: str) -> str:
    """数・日付の表記をそろえた文を返す (冪等)。"""
    s = unicodedata.normalize("NFKC", text)
    s = _KANJI_RE.sub(_sub_kanji, s)
    s = _WARI_RE.sub(_sub_wari, s)
    s = _MONEY_RE.sub(_sub_money, s)
    s = _WAREKI_RE.sub(_sub_wareki, s)
    s = _SEP_DATE_RE.sub(_sub_sep_date, s)
    return s


# ---------------------------------------------------------------- 1 つの数

_PLAIN_ARABIC = re.compile(rf"(?:{_NUM}[億万千])*(?:{_NUM})?")
_PLAIN_KANJI = re.compile(rf"[{_KN}]+")
_PARSE_WARI = re.compile(
    rf"(?P<w>[^割]+)割(?:(?P<b>{_D1})分(?:(?P<r>{_D1})厘)?|(?P<r2>{_D1})厘)?"
)
_TRAIL_COUNTER = re.compile(rf"{_COUNTER}$")


def _plain(t: str) -> Decimal | None:
    if not t:
        return None
    if _PLAIN_ARABIC.fullmatch(t):
        r = _chain_value(t)
        return None if r is None else r[0]
    if _PLAIN_KANJI.fullmatch(t):
        v = _kanji_value(t)
        return None if v is None else Decimal(v)
    return None


def parse_number(s: str) -> Decimal | None:
    """1 つの数の表記を Decimal にする。「10万」「1,000」「二十五」「八割」(=80)。

    末尾の数え方の語 (円・人・%…) は読み飛ばす。読めなければ None。
    """
    if not isinstance(s, str):
        return None
    t = unicodedata.normalize("NFKC", s).strip()
    m = _PARSE_WARI.fullmatch(t)
    if m:
        w = _plain(m["w"])
        if w is None:
            return None
        return _wari_value(w, m["b"], m["r"] or m["r2"])
    t = _TRAIL_COUNTER.sub("", t, count=1).strip()
    return _plain(t)
