"""数の比べをコードでやる (設計: docs/v2-code-compare-design.md)。

質問の値と、検索で取れた規程の文の条件 (「5万円を超える」など) を比べ、
「当てはまる/当てはまらない」を表す一文を作る純関数 ``check`` を提供する。
結論 (どの選択肢か) は決めない。決められない時は保留 (``sentence is None``) とし、
理由を ``hold_reason`` に入れる。

読むのは質問の文と検索で取れた文だけ。評価用の boundary / expected_* は引数に無い。
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

# ---------------------------------------------------------------- 結果の型


@dataclass
class CheckResult:
    sentence: str | None = None  # 注入する文 (保留なら None)
    hold_reason: str | None = None  # 保留の理由 (注入するなら None)
    kind: str | None = None  # 比べた値の種類 (円・日・日付など)
    value: Any = None  # 質問の値 (Decimal か date)
    boundary: Any = None  # 規程の境目 (Decimal か date)
    op: str | None = None  # 規程の向き (">" ">=" "<" "<=")
    applies: bool | None = None  # 規程の条件に当てはまるか
    rule_span: str | None = None  # 規程の文のうち比べた部分
    tiebreak: bool = False  # 候補が競合し、言葉の重なりで 1 つに絞った時 True


# ---------------------------------------------------------------- 言い回しの一覧

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


# ---------------------------------------------------------------- 値の取り出し

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


# ---------------------------------------------------------------- 質問の整理

_BASE_WORDS = (
    "購入", "発生", "生まれ", "誕生", "開催", "発行", "受領", "受け取", "受取", "到着",
    "納品", "納入", "入社", "退職", "着任", "契約", "締結", "出張", "出発", "開始",
    "発覚", "判明", "完了", "終了", "入院", "退院", "事故", "検収", "発送", "出荷",
    "採用", "異動", "就任",
)
_TARGET_WORDS = (
    "出し", "提出", "申請", "申し込", "申込", "届け", "届出", "報告", "請求", "連絡",
    "相談", "変更", "取消", "取り消", "キャンセル", "受診", "利用", "使用", "支払",
    "精算", "返却", "返送", "回答", "応募", "予約", "依頼",
)
_COMPOUND = re.compile(r"(?:購入|発行|請求|契約|出張|開催|入社|退職|発生)(?=申請|届|依頼|手続|報告)")


def _role(text: str, start: int, end: int, other_starts: list[int]) -> str | None:
    """日付の役割: 'base' (発生日・購入日など)・'target' (申請日など)・None。"""
    after = text[end:end + 9]
    cut = len(after)
    for ch in "、。\n，,":
        i = after.find(ch)
        if i != -1:
            cut = min(cut, i + (1 if ch in "、，," else 0))
    for s in other_starts:
        if s >= end:
            cut = min(cut, s - end)
    after = _COMPOUND.sub("", after[:cut])
    before = text[max(0, start - 9):start]
    for ch in "、。\n，,":
        i = before.rfind(ch)
        if i != -1:
            before = before[i + 1:]
    before = _COMPOUND.sub("", before)
    has_base = any(w in after for w in _BASE_WORDS)
    has_target = any(w in after for w in _TARGET_WORDS)
    label = re.search(r"([^\s、。]{1,6}日)\s*(?:は|が|:|：)?\s*$", before)
    if label:
        has_base = has_base or any(w in label[1] for w in _BASE_WORDS)
        has_target = has_target or any(w in label[1] for w in _TARGET_WORDS)
    else:
        tail = re.search(r"([^\s、。]{1,6})を\s*$", before)
        if tail:
            has_target = has_target or any(w in tail[1] for w in _TARGET_WORDS)
    if has_base and not has_target:
        return "base"
    if has_target and not has_base:
        return "target"
    return None


@dataclass
class _QDate:
    d: date
    role: str | None


@dataclass
class _Question:
    text: str
    qtys: list[_Qty]
    dates: list[_QDate]  # 重複を除いた完全な日付
    unresolved_date: bool  # 年の分からない日付がある
    month_refs: list[tuple[int | None, int]]  # (年, 月) 「10月分」など
    biz: bool


def _parse_question(question: str) -> _Question:
    text = _normalize(question)
    # 書式の指示 (「最後の行には…」) は値を取る対象から外す
    cut = text.find("最後の行")
    if cut != -1:
        text = text[:cut]
    spans: list[tuple[int, int]] = []
    fulls = _full_dates(text)
    spans += [(a, b) for a, b, _ in fulls]
    years = {d.year for _, _, d in fulls}
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
    month_refs: list[tuple[int | None, int]] = []
    for m in _MONTH_REF.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        month_refs.append((int(m["y"]) if m["y"] else None, int(m["m"])))
        spans.append((m.start(), m.end()))
    for m in _YEAR_MONTH.finditer(text):
        if not any(a <= m.start() < b for a, b in spans):
            spans.append((m.start(), m.end()))
    all_dates = sorted(fulls + partial, key=lambda x: x[0])
    starts = [a for a, _, _ in all_dates]
    dates: list[_QDate] = []
    seen: dict[date, int] = {}
    for a, b, d in all_dates:
        role = _role(text, a, b, [s for s in starts if s != a])
        if d in seen:
            prev = dates[seen[d]]
            if prev.role is None:
                prev.role = role
            continue
        seen[d] = len(dates)
        dates.append(_QDate(d, role))
    qtys = _quantities(text, spans)
    return _Question(
        text=text,
        qtys=[q for q in qtys if not q.bizday],
        dates=dates,
        unresolved_date=unresolved,
        month_refs=sorted(set(month_refs), key=lambda x: (x[0] or 0, x[1])),
        biz=any(q.bizday for q in qtys) or "営業日" in text,
    )


# ---------------------------------------------------------------- 規程の文

_SPLIT = re.compile(r"[。！？!?\n]+")


def _sentences(chunk_texts: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in chunk_texts:
        for s in _SPLIT.split(_normalize(t or "")):
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


@dataclass
class _Cand:
    group: str
    op: str
    boundary: Any
    span: str
    sentence: str
    note: str | None = None
    unsupported: str | None = None


_REL_MONTH = re.compile(r"(?P<w>翌々月|翌月|当月)(?:(?P<eom>末日|末)|(?P<d>\d{1,2})日)")
_N_DAYS_BEFORE = re.compile(r"(?P<n>\d+)日前")
_N_DAYS = re.compile(r"(?P<n>\d+)日間?")
_LEAD = re.compile(r"([^、。，\s]{0,10}(?:から数えて|から))$")


def _fd(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"


def _add_months(y: int, m: int, k: int) -> tuple[int, int]:
    idx = y * 12 + (m - 1) + k
    return idx // 12, idx % 12 + 1


@dataclass
class _RelCtx:
    base: date | None  # 基準日
    target: date | None  # 比べる日
    base_month: tuple[int, int] | None  # (年, 月)
    base_label: str
    reason: str | None  # 決められない時の理由


def _rel_context(q: _Question) -> _RelCtx:
    ds = q.dates
    if len(ds) > 2 or (len(ds) == 2 and q.month_refs) or len(q.month_refs) > 1:
        return _RelCtx(None, None, None, "", "基準になる日・月が決められない（日付が多い）")
    if q.unresolved_date:
        return _RelCtx(None, None, None, "", "日付の年が分からない")
    if len(ds) == 1 and len(q.month_refs) == 1:
        target = ds[0].d
        y, m = q.month_refs[0]
        if y is None:
            y = target.year if m <= target.month else target.year - 1
        return _RelCtx(None, target, (y, m), f"{y}年{m}月分", None)
    if len(ds) == 2:
        a, b = ds
        if a.role == "base" and b.role != "base":
            base, target = a.d, b.d
        elif b.role == "base" and a.role != "base":
            base, target = b.d, a.d
        elif a.role == "target" and b.role != "target":
            base, target = b.d, a.d
        elif b.role == "target" and a.role != "target":
            base, target = a.d, b.d
        else:
            return _RelCtx(None, None, None, "", "基準日と比べる日の区別が付かない")
        return _RelCtx(base, target, (base.year, base.month), f"基準日 {_fd(base)}", None)
    return _RelCtx(None, None, None, "", "基準になる日・月が無い")


def _lead_extend(s: str, start: int) -> int:
    m = _LEAD.search(s[:start])
    return m.start() if m else start


def _abs_date_cands(s: str) -> tuple[list[_Cand], bool]:
    """文中の絶対日付 + 言い回し。第 2 値は『言い回しの無い日付があった』。"""
    out: list[_Cand] = []
    bare = False
    for a, b, d in _full_dates(s):
        r = _match_phrase(s, b, _DATE_RE)
        if r is None:
            bare = True
            continue
        out.append(_Cand("date_abs", r[1], d, s[a:r[0]], s))
    return out, bare


def _rel_cands(s: str, ctx: _RelCtx) -> tuple[list[_Cand], list[str], bool]:
    """文中の相対日付。戻り値: (候補, 決められなかった理由, 営業日の候補があったか)"""
    cands: list[_Cand] = []
    reasons: list[str] = []
    consumed: list[tuple[int, int]] = [(a, b) for a, b, _ in _full_dates(s)]
    biz = False

    for m in _BIZDAY.finditer(s):
        if _match_phrase(s, m.end(), _BOTH_RE):
            biz = True
        consumed.append((m.start(), m.end()))

    for m in _REL_MONTH.finditer(s):
        consumed.append((m.start(), m.end()))
        r = _match_phrase(s, m.end(), _DATE_RE)
        if r is None:
            continue
        if ctx.base_month is None:
            reasons.append(ctx.reason or "基準の月が無い")
            continue
        k = {"当月": 0, "翌月": 1, "翌々月": 2}[m["w"]]
        y, mo = _add_months(*ctx.base_month, k)
        if m["eom"]:
            thr = date(y, mo, calendar.monthrange(y, mo)[1])
        else:
            thr = _to_date(y, mo, int(m["d"]))
            if thr is None:
                reasons.append("翌月の日が暦に無い")
                continue
        note = f"{ctx.base_label}の{m['w']}{m['eom'] or m['d'] + '日'} = {_fd(thr)}"
        cands.append(_Cand("date_rel", r[1], thr, s[m.start():r[0]], s, note))

    for m in _N_DAYS_BEFORE.finditer(s):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        r = _match_phrase(s, m.end(), _DATE_RE)
        if r is None:
            continue
        consumed.append((m.start(), m.end()))
        if ctx.base is None:
            reasons.append(ctx.reason or "基準日が無い")
            continue
        n = int(m["n"])
        thr = ctx.base - timedelta(days=n)
        note = f"{ctx.base_label} の {n}日前 = {_fd(thr)}"
        cands.append(_Cand("date_rel", r[1], thr, s[m.start():r[0]], s, note))

    for m in _N_DAYS.finditer(s):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        if m.start() > 0 and s[m.start() - 1] in "月営":
            continue
        r = _match_phrase(s, m.end(), _BOTH_RE)
        if r is None:
            continue
        if ctx.base is None:
            reasons.append(ctx.reason or "基準日が無い")
            continue
        n = int(m["n"])
        thr = ctx.base + timedelta(days=n)
        note = f"{ctx.base_label} + {n}日 = {_fd(thr)}"
        cands.append(_Cand("date_rel", r[1], thr, s[_lead_extend(s, m.start()):r[0]], s, note))
    return cands, reasons, biz


def _num_cands(s: str, spans_of_dates: list[tuple[int, int]]) -> tuple[list[_Qty], list[tuple[_Qty, _Cand]]]:
    qs = _quantities(s, spans_of_dates)
    cands = []
    for q in qs:
        if q.bizday:
            continue
        r = _match_phrase(s, q.end, _NUM_RE)
        if r is None:
            continue
        cands.append((q, _Cand(f"num:{q.kind}", r[1], q.value, s[q.start:r[0]], s)))
    return qs, cands


# ---------------------------------------------------------------- 競合の整理

_JA = re.compile(r"[一-龥ァ-ヶー々A-Za-z]")


def _bigrams(text: str) -> set[str]:
    chars = [c if _JA.match(c) else " " for c in text]
    out = set()
    for i in range(len(chars) - 1):
        a, b = chars[i], chars[i + 1]
        if a == " " or b == " ":
            continue
        if "ぁ" <= a <= "ん" and "ぁ" <= b <= "ん":
            continue  # ひらがなだけの組は助詞・活用で、話題を表さない
        out.add(a + b)
    return out


def _overlap(question_text: str, sentence: str) -> int:
    return len(_bigrams(question_text) & _bigrams(sentence))


def _fmt_num(kind: str, v: Decimal) -> str:
    v = v.normalize() if v != v.to_integral() else v.quantize(Decimal(1))
    s = f"{v:,}" if kind == "円" else f"{v}"
    if "E" in s or "e" in s:
        s = format(v, "f")
    return s


def _fmt_value(kind: str, v: Any) -> str:
    if isinstance(v, date):
        return _fd(v)
    s = _fmt_num(kind, v)
    return f"{s}{_KIND_LABEL.get(kind, kind)}"


def _build(kind: str, cand: _Cand, value: Any, tiebreak: bool) -> CheckResult:
    ok = _holds(value, cand.op, cand.boundary)
    sym = _SYM[cand.op] if ok else _NEG_OP[cand.op]
    glue = " "
    if isinstance(value, date):
        vtxt, ttxt = value.isoformat(), cand.boundary.isoformat()
        shown = _fmt_value(kind, value) + (f"（{cand.note}）" if cand.note else "")
        glue = "" if cand.note else " "
    else:
        vtxt, ttxt = _fmt_num(kind, value), _fmt_num(kind, cand.boundary)
        shown = _fmt_value(kind, value)
    sentence = (
        f"規程の「{cand.span}」に対し、質問の値 {shown}{glue}は"
        f"{'当てはまる' if ok else '当てはまらない'}（{vtxt} {sym} {ttxt}）。"
    )
    return CheckResult(
        sentence=sentence, kind=kind, value=value, boundary=cand.boundary,
        op=cand.op, applies=ok, rule_span=cand.span, tiebreak=tiebreak,
    )


def _hold(reason: str) -> CheckResult:
    return CheckResult(hold_reason=reason)


# ---------------------------------------------------------------- 本体


def check(question: str, chunk_texts: list[str]) -> CheckResult:
    q = _parse_question(question)
    sents = _sentences(chunk_texts)

    # 質問の数量: 種類ごとの値の集合
    by_kind: dict[str, set[Decimal]] = {}
    for qt in q.qtys:
        by_kind.setdefault(qt.kind, set()).add(qt.value)
    has_dates = bool(q.dates) or q.unresolved_date
    if not by_kind and not has_dates and not q.month_refs and not q.biz:
        return _hold("質問に値が無い")
    if q.biz:
        return _hold("営業日（暦が要る）")

    groups: dict[str, list[_Cand]] = {}
    unsupported_hit: list[str] = []
    bare_same_kind: set[str] = set()
    family_hit = False
    rel_failures: list[str] = []
    abs_bare = False

    # 日付の方式: 日付が 2 つ以上か「N月分」があれば相対、1 つだけなら絶対
    date_mode: str | None = None
    ctx: _RelCtx | None = None
    if has_dates or q.month_refs:
        if len(q.dates) == 1 and not q.month_refs and not q.unresolved_date:
            date_mode = "abs"
        else:
            date_mode = "rel"
            ctx = _rel_context(q)

    for s in sents:
        date_spans = [(a, b) for a, b, _ in _full_dates(s)]
        qs, ncands = _num_cands(s, date_spans)
        for qt, c in ncands:
            if qt.kind in by_kind:
                groups.setdefault(c.group, []).append(c)
            elif by_kind.keys() & _TIME_FAMILY and qt.kind in _TIME_FAMILY:
                family_hit = True
        phrased = {id(qt) for qt, _ in ncands}
        for qt in qs:
            if not qt.bizday and id(qt) not in phrased and qt.kind in by_kind:
                bare_same_kind.add(qt.kind)
        if date_mode == "abs":
            ac, bare = _abs_date_cands(s)
            abs_bare = abs_bare or bare
            groups.setdefault("date_abs", []).extend(ac)
            for m in _BIZDAY.finditer(s):
                if _match_phrase(s, m.end(), _BOTH_RE):
                    groups.setdefault("date_abs", []).append(
                        _Cand("date_abs", "<=", None, m.group(0), s, unsupported="営業日（暦が要る）")
                    )
        elif date_mode == "rel":
            rc, reasons, biz = _rel_cands(s, ctx)  # type: ignore[arg-type]
            groups.setdefault("date_rel", []).extend(rc)
            rel_failures += reasons
            if biz:
                groups.setdefault("date_rel", []).append(
                    _Cand("date_rel", "<=", None, "営業日", s, unsupported="営業日（暦が要る）")
                )
            ac, _ = _abs_date_cands(s)
            if ac:
                unsupported_hit.append("abs")
        elif by_kind:
            for m in _BIZDAY.finditer(s):
                if _match_phrase(s, m.end(), _BOTH_RE) and "日" in by_kind:
                    groups.setdefault("num:日", []).append(
                        _Cand("num:日", "<=", None, m.group(0), s, unsupported="営業日（暦が要る）")
                    )

    groups = {k: v for k, v in groups.items() if v}

    # 同じ種類で違う値が 2 つ以上ある数量は保留 (候補のある種類だけ見る)
    for g in groups:
        if g.startswith("num:") and len(by_kind[g[4:]]) > 1:
            return _hold(f"質問に同じ種類（{g[4:]}）の違う値が複数ある")

    if not groups:
        if date_mode == "rel" and rel_failures:
            return _hold(rel_failures[0])
        if date_mode == "rel" and "abs" in unsupported_hit:
            return _hold("質問に違う日付が複数あり、絶対日付の条件とは比べない")
        if family_hit:
            return _hold("単位がそろわない")
        if bare_same_kind or abs_bare:
            return _hold("一覧に無い言い回し")
        return _hold("条件の候補が無い")

    if len(groups) > 1:
        return _hold("複数の種類に条件の候補がある")

    (gname, cands), = groups.items()
    tiebreak = False
    uniq: dict[tuple[str, Any], list[_Cand]] = {}
    for c in cands:
        uniq.setdefault((c.op, c.boundary if not c.unsupported else c.unsupported + c.sentence), []).append(c)
    if len(uniq) > 1:
        scored = sorted(
            ((max(_overlap(q.text, c.sentence) for c in v), k) for k, v in uniq.items()),
            key=lambda x: -x[0],
        )
        if scored[0][0] <= 0 or scored[0][0] == scored[1][0]:
            return _hold("違う境目・違う向きの候補が複数ある")
        winner_key = scored[0][1]
        tiebreak = True
    else:
        winner_key = next(iter(uniq))
    cand = uniq[winner_key][0]
    if cand.unsupported:
        return _hold(cand.unsupported)

    if gname.startswith("num:"):
        kind = gname[4:]
        (value,) = by_kind[kind]
        return _build(kind, cand, value, tiebreak)
    if gname == "date_abs":
        return _build("日付", cand, q.dates[0].d, tiebreak)
    assert ctx is not None and ctx.target is not None
    return _build("日付", cand, ctx.target, tiebreak)
