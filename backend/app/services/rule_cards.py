"""規程カード方式（Issue #16・docs/research/2026-11-02/numeric-rules.md の順位 1）。

規程ごとのカード（対象・式・比べ方・境目・例外・結論・文書の版）を持ち、
- 読むのは小型モデル: カードの項目名を鍵に、質問から値とその根拠の引用だけを JSON で拾う（json_schema で形を固定）
- 決めるのはコード: 引用が質問に実在し、引用の数が値と一致するかを確かめ、式と比べ方で結論を出す
各条件は 成立 / 不成立 / 不明 の 3 つで扱い（不明は Kleene の論理で伝える）、結論が決まらなければ保留する。
説明は定型文。モデルの自由な文は使わない。

カードは強いモデルが規程の文から下書きし（status: draft）、人が確認して status: confirmed にする。
実行時は confirmed だけを使う（allow_draft は開発の比較用）。カードの形は load_card が確かめ、
おかしなカードは読み込みで止める（停止試験）。

カードの形（JSON・format: rule-card@1）:
  id, status, drafted_by, confirmed_by, doc{source, section, quote}, target,
  match{conclusions: [then, else], keywords: [...]},
  slots{名前: {type: number|date|time|bool, unit?, describe}}（time の値は "HH:MM"・00:00〜23:59）,
  derive{名前: 式}（拾えなかった項目を他の項目から計算する。例: 出勤日数 = 全労働日数 - 欠勤日数）,
  constraints[{name, left, op, right}]（値の筋が通らなければ保留。例: 出勤日数 <= 全労働日数）,
  conditions[{name, left, op, right[, quote]}], combine: all|any, conclusion{then, else},
  exceptions[{name, left, op, right, conclusion, if_unknown: hold|ignore[, quote]}]
  条件・例外は left/op/right の代わりに入れ子 {name, all|any: [条件...]} も書ける（3 値で評価: all は 1 つでも不成立なら不成立、
  any は 1 つでも成立なら成立、残りは不明）。quote は根拠の規程の文（任意。下書きの道具が規程に実在するかを確かめる）。
式: 数・項目名・{"date": "YYYY-MM-DD"}・{"time": "HH:MM"}・{"add": [...]}・{"sub": [a, b]}・{"mul": [...]}・{"div": [a, b]}・
    {"add_days": [日付, n]}・{"month_day": [日付, 何か月後, 日]}・{"min": [...]}・{"max": [...]}（2 個以上。1 つでも不明なら不明）・
    {"month_end": 日付}（その月の末日）・{"add_months": [日付, 月数]}（行き先の月に同じ日が無ければ月末に丸める）・
    {"minutes_between": [開始の時刻, 終了の時刻]}（終了が開始より前なら夜をまたぐとして 24 時間足す）
時刻は 0 時からの分の数として比べる。
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.services import numnorm

FORMAT = "rule-card@1"
OPS = (">", ">=", "<", "<=", "==")
SLOT_TYPES = ("number", "date", "time", "bool")
BOOL_VALUES = ("あり", "なし")
# 数え方の語の言い換え（引用の数の直後にあるかを見る）
UNIT_ALIASES = {"か月": ("か月", "ヶ月", "カ月", "箇月", "ケ月"), "%": ("%", "パーセント")}
_EXPR_FUNCS = {"add", "sub", "mul", "div", "add_days", "month_day", "date", "time", "min", "max", "month_end",
               "add_months", "minutes_between"}
_MAX_NEST = 5  # 条件の入れ子の深さの上限
_OPTIONS = re.compile(r"「結論\s*[:：]\s*(.+?)」")
_LAST_LINE = re.compile(r"最後の行")
_NUM = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_TIME_VALUE = re.compile(r"^(\d{2}):(\d{2})$")
TIME_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"
# 引用の中の時刻: 22:30・午後10時・22時30分・22時半。「3時間」は時刻ではない
_TIME_COLON = re.compile(r"(?<![\d:.])(午前|午後)?\s*(\d{1,2}):(\d{2})(?![\d:])")
_TIME_JI = re.compile(r"(?<![\d.])(午前|午後)?\s*(\d{1,2})時(?!間)(?:\s*(\d{1,2})分|(半))?")
_MIDDAY = "正午"
# 引用の中の時間の長さ: 6時間10分・3時間半・8時間（枠の単位が 分 なら分に、時間なら時間にして照合する）
_DURATION = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)時間(?:(\d+)分|(半))?")
DURATION_UNITS = ("分", "時間")
# 同じ量の別の言い方（verify="equiv" の時だけ見る）: 3年 = 36か月・1年1か月 = 13か月（2023年11月 のような年月は除く）・4千6百円
_YEARS = re.compile(r"(?<![\d.,年])(\d{1,2})年(?!\d+月)(?:(\d+)(?:か月|ヶ月|カ月|箇月|ケ月))?")
_SENN_HYAKU = re.compile(r"(?<![\d.,])(\d+)千(\d+)百")
ZERO_WORDS = ("ない", "なし", "ありません", "持っていない", "無い")  # 数の枠の 0 は、引用にこの語がある時だけ通す
# 年のない日付: 12月18日・1/30（直前が「年」や数字・斜線の物は年つきの日付の一部なので除く）
_MD_KANJI = re.compile(r"(?<![年\d])(\d{1,2})月(\d{1,2})日")
_MD_SLASH = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
YEARLESS_BASE_YEAR = 2024  # 年のない日付だけの問で、最初の 1 つを置く年（2 月 29 日を作れる閏年）
YEARLESS_AMBIGUOUS_DAYS = 170  # 基準の日付との差がこれ以上なら、近い方の年が定まらないので保留


class CardError(ValueError):
    """カードの形がおかしい（読み込みで止める）。"""


@dataclass(frozen=True)
class Slot:
    name: str
    type: str
    unit: str | None
    describe: str


@dataclass(frozen=True, order=True)
class Clock:
    """時刻。0 時からの分の数（0〜1439）。日付・数とは別の型なので、混ぜて比べるとカードの誤りとして保留になる。"""
    minutes: int

    def __str__(self) -> str:
        return f"{self.minutes // 60:02d}:{self.minutes % 60:02d}"


@dataclass(frozen=True)
class Card:
    id: str
    status: str
    doc: Mapping[str, str]
    target: str
    conclusions: tuple[str, str]  # (then, else)
    keywords: tuple[str, ...]
    slots: Mapping[str, Slot]
    derive: Mapping[str, Any]
    constraints: tuple[Mapping[str, Any], ...]
    conditions: tuple[Mapping[str, Any], ...]
    combine: str
    exceptions: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any] = field(repr=False, compare=False, default_factory=dict)


# ---------------------------------------------------------------- 読み込み（形の検査）


def parse_time(text: str) -> Clock | None:
    """"HH:MM"（00:00〜23:59）を Clock にする。24:00 や形の違う物は None。"""
    m = _TIME_VALUE.match(text) if isinstance(text, str) else None
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        return None
    return Clock(int(m[1]) * 60 + int(m[2]))


def _check_expr(card_id: str, e: Any, slots: Mapping[str, Slot]) -> None:
    if isinstance(e, bool):
        raise CardError(f"{card_id}: bool is not an expression")
    if isinstance(e, (int, float)):
        return
    if isinstance(e, str):
        if e not in slots and e not in BOOL_VALUES:
            raise CardError(f"{card_id}: unknown slot {e!r}")
        return
    if isinstance(e, dict) and len(e) == 1:
        (fn, args), = e.items()
        if fn not in _EXPR_FUNCS:
            raise CardError(f"{card_id}: unknown function {fn!r}")
        if fn == "date":
            if not isinstance(args, str):
                raise CardError(f"{card_id}: date literal must be YYYY-MM-DD")
            try:
                date.fromisoformat(args)
            except ValueError as ex:
                raise CardError(f"{card_id}: bad date {args!r}") from ex
            return
        if fn == "time":
            if parse_time(args) is None:
                raise CardError(f"{card_id}: bad time {args!r} (HH:MM, 00:00-23:59)")
            return
        if fn == "month_end":  # 引数は式 1 つ（リストではない）
            _check_expr(card_id, args, slots)
            return
        if not isinstance(args, list) or not args:
            raise CardError(f"{card_id}: {fn} needs a list of arguments")
        arity = {"sub": 2, "div": 2, "add_days": 2, "month_day": 3, "add_months": 2, "minutes_between": 2}.get(fn)
        if arity is not None and len(args) != arity:
            raise CardError(f"{card_id}: {fn} takes {arity} arguments")
        if fn in ("min", "max") and len(args) < 2:
            raise CardError(f"{card_id}: {fn} takes 2 or more arguments")
        for a in args:
            _check_expr(card_id, a, slots)
        return
    raise CardError(f"{card_id}: bad expression {e!r}")


def _check_cond(card_id: str, c: Any, slots: Mapping[str, Slot], extra: Sequence[str] = (), depth: int = 0) -> None:
    """条件 1 つ。left/op/right の形か、入れ子 {name, all|any: [条件...]}。quote は任意。"""
    if not isinstance(c, dict) or not isinstance(c.get("name"), str):
        raise CardError(f"{card_id}: condition needs name: {c!r}")
    if "quote" in c and not (isinstance(c["quote"], str) and c["quote"].strip()):
        raise CardError(f"{card_id}: condition quote must be a non-empty string: {c!r}")
    keys = {"name", "quote", *extra}
    group = [k for k in ("all", "any") if k in c]
    if group:
        if len(group) != 1 or set(c) - {*keys, group[0]} or not isinstance(c[group[0]], list) or not c[group[0]]:
            raise CardError(f"{card_id}: nested condition needs name and one non-empty all|any list: {c!r}")
        if depth >= _MAX_NEST:
            raise CardError(f"{card_id}: conditions are nested too deep")
        for child in c[group[0]]:
            _check_cond(card_id, child, slots, depth=depth + 1)
        return
    if not {"left", "op", "right"} <= set(c) or set(c) - {*keys, "left", "op", "right"}:
        raise CardError(f"{card_id}: condition needs name/left/op/right: {c!r}")
    if c["op"] not in OPS:
        raise CardError(f"{card_id}: unknown op {c['op']!r}")
    _check_expr(card_id, c["left"], slots)
    _check_expr(card_id, c["right"], slots)


def parse_card(doc: Mapping[str, Any]) -> Card:
    cid = str(doc.get("id") or "?")
    if doc.get("format") != FORMAT:
        raise CardError(f"{cid}: format must be {FORMAT}")
    if doc.get("status") not in ("draft", "confirmed"):
        raise CardError(f"{cid}: status must be draft or confirmed")
    if doc["status"] == "confirmed" and not doc.get("confirmed_by"):
        raise CardError(f"{cid}: confirmed card needs confirmed_by")
    d = doc.get("doc") or {}
    if not all(isinstance(d.get(k), str) and d[k] for k in ("source", "quote")):
        raise CardError(f"{cid}: doc needs source and quote")
    m = doc.get("match") or {}
    concl = m.get("conclusions")
    if not (isinstance(concl, list) and len(concl) == 2 and all(isinstance(x, str) and x for x in concl)
            and concl[0] != concl[1]):
        raise CardError(f"{cid}: match.conclusions must be two different strings")
    if doc.get("conclusion") != {"then": concl[0], "else": concl[1]}:
        raise CardError(f"{cid}: conclusion must be {{then: {concl[0]}, else: {concl[1]}}}")
    slots: dict[str, Slot] = {}
    for name, s in (doc.get("slots") or {}).items():
        if not isinstance(s, dict) or s.get("type") not in SLOT_TYPES or not s.get("describe"):
            raise CardError(f"{cid}: slot {name} needs type ({'/'.join(SLOT_TYPES)}) and describe")
        slots[name] = Slot(name, s["type"], s.get("unit"), s["describe"])
    if not slots:
        raise CardError(f"{cid}: no slots")
    derive = doc.get("derive") or {}
    for name, e in derive.items():
        if name not in slots:
            raise CardError(f"{cid}: derive target {name} is not a slot")
        _check_expr(cid, e, slots)
    for c in doc.get("constraints") or []:
        _check_cond(cid, c, slots)
    conds = doc.get("conditions") or []
    if not conds:
        raise CardError(f"{cid}: no conditions")
    for c in conds:
        _check_cond(cid, c, slots)
    if doc.get("combine", "all") not in ("all", "any"):
        raise CardError(f"{cid}: combine must be all or any")
    for x in doc.get("exceptions") or []:
        _check_cond(cid, x, slots, extra=("conclusion", "if_unknown"))
        if x.get("conclusion") not in concl or x.get("if_unknown", "hold") not in ("hold", "ignore"):
            raise CardError(f"{cid}: exception needs a conclusion from match.conclusions and if_unknown hold|ignore")
    return Card(id=cid, status=doc["status"], doc=d, target=str(doc.get("target") or ""), conclusions=(concl[0], concl[1]),
                keywords=tuple(m.get("keywords") or ()), slots=slots, derive=derive,
                constraints=tuple(doc.get("constraints") or ()), conditions=tuple(conds),
                combine=doc.get("combine", "all"), exceptions=tuple(doc.get("exceptions") or ()), raw=doc)


def load_card(path: Path | str) -> Card:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CardError(f"{path}: {e}") from e
    return parse_card(doc)


def load_cards(directory: Path | str, *, allow_draft: bool = False) -> list[Card]:
    cards = [load_card(p) for p in sorted(Path(directory).glob("*.json"))]
    if len({c.id for c in cards}) != len(cards):
        raise CardError(f"{directory}: duplicate card ids")
    _check_pair_clash(cards, directory)
    return [c for c in cards if allow_draft or c.status == "confirmed"]


def _keywords_overlap(a: Card, b: Card) -> bool:
    """手がかりの語が重なるか（同じか、一方がもう一方に含まれる）。語が無いカードは何にでも当たるので重なる扱い。"""
    if not a.keywords or not b.keywords:
        return True
    return any(x in y or y in x for x in a.keywords for y in b.keywords)


def _check_pair_clash(cards: Sequence[Card], where: Path | str) -> None:
    """同じ文書で結論の語の組が同じカードが 2 枚あれば止める。
    同じ組のカードが並ぶと select_card が 1 枚に決められず None になるため、規程の定めは組ごとに 1 枚へまとめる。
    ただし keywords が重ならない物は許す: 質問にどちらか一方の語だけが出る時は 1 枚に決まり、
    両方の語が出る時は select_card が None を返して保留になるだけで、誤った答えにはならない
    （例: 福利厚生規程の住宅手当と人間ドックは、どちらも 対象になる / 対象にならない）。"""
    for i, a in enumerate(cards):
        for b in cards[i + 1:]:
            if (a.doc["source"] == b.doc["source"] and set(a.conclusions) == set(b.conclusions)
                    and _keywords_overlap(a, b)):
                raise CardError(f"{where}: cards {a.id} and {b.id} have the same conclusions pair in {a.doc['source']}"
                                " (merge them into one card, or make their keywords distinct)")


# ---------------------------------------------------------------- 質問の扱い


def question_body(question: str) -> str:
    """答えの書式の指示（「最後の行」以降）を除いた質問。"""
    m = _LAST_LINE.search(question)
    return question[: m.start()] if m else question


def conclusion_options(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(o.strip() for o in _OPTIONS.findall(question)))


# カードの選び方の定数。値は「正しい規程を選んだ・誤選択・答えない」を GPU なしで数えて決めた
# （開発用 90 問・N 12 問・開発用 24 問。誤選択 0 の範囲で選択数が最大の点のうち、最も厳しい点）。変えたら数え直す
SELECT_MIN_SCORE = 14  # 1 位の点の下限。語の組に合うカードが 1 枚だけの時も要る
SELECT_MIN_MARGIN = 8  # 1 位と 2 位の点の差の下限（2 枚以上の時）
SELECT_KEYWORD_WEIGHT = 3  # keywords の文字 2-gram の重み（他は 1）
_NOT_GRAM = re.compile(r"[\s、。，．,.!?！？「」『』（）()・:：;；/／]")


def _bigrams(text: str) -> set[str]:
    s = _NOT_GRAM.sub("", unicodedata.normalize("NFKC", text))
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _cond_texts(c: Mapping[str, Any]) -> list[str]:
    out = [str(c.get("name") or "")]
    for k in ("all", "any"):
        for child in c.get(k) or ():
            out += _cond_texts(child)
    return out


def card_select_weights(card: Card) -> dict[str, int]:
    """カードの語（doc.source・doc.quote・枠の名前と describe・条件と例外の name・keywords）の文字 2-gram → 重み。"""
    texts = [card.doc["source"], card.doc["quote"]]
    for s in card.slots.values():
        texts += [s.name, s.describe]
    for c in (*card.conditions, *card.exceptions):
        texts += _cond_texts(c)
    weights: dict[str, int] = {}
    for t in texts:
        for g in _bigrams(t):
            weights[g] = 1
    for k in card.keywords:
        for g in _bigrams(k):
            weights[g] = SELECT_KEYWORD_WEIGHT
    return weights


def select_score(card: Card, question: str) -> int:
    """質問本文の文字 2-gram とカードの語の 2-gram の重なり（重みの合計）。"""
    qg = _bigrams(question_body(question))
    return sum(w for g, w in card_select_weights(card).items() if g in qg)


def select_card(cards: Sequence[Card], question: str) -> Card | None:
    """質問の結論の選択肢がカードの結論と同じカードのうち、質問との語の重なりの点が最も高い 1 枚。
    1 位が SELECT_MIN_SCORE 以上で、2 位との差が SELECT_MIN_MARGIN 以上の時だけ返す。それ以外は None（答えない）。
    誤ったカードを選ぶと誤断定になるので、迷う時は選ばない。"""
    opts = set(conclusion_options(question))
    ranked = sorted(((select_score(c, question), i) for i, c in enumerate(cards) if opts == set(c.conclusions)),
                    key=lambda x: (-x[0], x[1]))
    if not ranked or ranked[0][0] < SELECT_MIN_SCORE:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < SELECT_MIN_MARGIN:
        return None
    return cards[ranked[0][1]]


# ---------------------------------------------------------------- 抽出の形


def extraction_schema(card: Card, *, reverse: bool = False) -> dict[str, Any]:
    """llama.cpp / vLLM の json_schema に渡す形。項目ごとに quote（質問からそのまま抜き出した部分）→ value の順。
    reverse=True は枠の順を逆にする（2 回目の読み直し用）。"""
    props: dict[str, Any] = {}
    slots = list(card.slots.values())
    for s in reversed(slots) if reverse else slots:
        if s.type == "number":
            value: dict[str, Any] = {"type": ["number", "null"]}
        elif s.type == "date":
            value = {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"}
        elif s.type == "time":
            value = {"type": ["string", "null"], "pattern": TIME_PATTERN}
        else:
            value = {"type": ["string", "null"], "enum": [*BOOL_VALUES, None]}
        # quote を value より先に出させる（先に質問の文を抜き、その文から値を決める）。properties と required の両方で
        props[s.name] = {"type": "object", "properties": {"quote": {"type": ["string", "null"]}, "value": value},
                         "required": ["quote", "value"], "additionalProperties": False}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def extraction_messages(card: Card, question: str, *, reverse: bool = False) -> list[dict[str, str]]:
    lines = []
    slots = list(card.slots.values())
    for s in reversed(slots) if reverse else slots:
        kind = {"number": f"数（単位 {s.unit}）" if s.unit else "数", "date": "日付（YYYY-MM-DD）",
                "time": "時刻（HH:MM・24 時間表記。午後10時は 22:00）", "bool": "「あり」か「なし」"}[s.type]
        lines.append(f"- {s.name}: {s.describe}。{kind}")
    system = ("質問の文から、次の項目について、先に根拠になる質問の文をそのまま抜き出し（quote）、次にその文から値（value）を決めて、"
              "JSON で答えてください。\n"
              "quote には質問の文を言い換えずにそのまま入れます。value は quote に書かれている通りに決めます。\n"
              "質問に書かれていない項目は quote と value を null にします。計算・推測・換算はしません。"
              "ただし、数の単位が 分 や 時間 で quote が「6時間10分」「3時間半」のような時間の長さの時だけは、その単位の数に直します。\n"
              + "\n".join(lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": question_body(question).strip()}]


# ---------------------------------------------------------------- 抽出の検査


def _norm(s: str) -> str:
    return numnorm.normalize_numbers(s)


def _durations(text: str) -> list[tuple[int, int, Decimal]]:
    """引用の時間の言い方 6時間10分・3時間半・8時間 → (始まり, 終わり, 分の数)。"""
    out = []
    for m in _DURATION.finditer(text):
        minutes = Decimal(m[1]) * 60 + (Decimal(m[2]) if m[2] else Decimal(30) if m[3] else Decimal(0))
        out.append((m.start(), m.end(), minutes))
    return out


def _numbers_with_unit(text: str, unit: str | None) -> list[Decimal]:
    out = []
    aliases = UNIT_ALIASES.get(unit or "", (unit,)) if unit else ()
    durations = _durations(text) if unit in DURATION_UNITS else []
    # 「6時間10分」の 6 や 10 は、複合の値の一部。単独の数としては数えない（6 時間と読み違えない）
    compound = [(a, b) for a, b, _ in durations if text[a:b].endswith(("分", "半"))]
    for m in _NUM.finditer(text):
        if aliases and not any(text[m.end():].lstrip().startswith(a) for a in aliases):
            continue
        if any(a <= m.start() < b for a, b in compound):
            continue
        try:
            out.append(Decimal(m.group().replace(",", "")))
        except InvalidOperation:
            continue
    for _, _, minutes in durations:
        out.append(minutes if unit == "分" else minutes / 60)
    return out


def _equiv_numbers(quote: str, unit: str | None) -> list[Decimal]:
    """引用の数と同じ量だとコードで確かめられる別の値。年→か月・千百の混じり・単位の合う数が 2 つ以上ある時の合計。"""
    out: list[Decimal] = []
    if unit == "か月":
        out += [Decimal(m[1]) * 12 + Decimal(m[2] or 0) for m in _YEARS.finditer(quote)]
    aliases = UNIT_ALIASES.get(unit or "", (unit,)) if unit else ()
    mixed = [(m, Decimal(m[1]) * 1000 + Decimal(m[2]) * 100) for m in _SENN_HYAKU.finditer(quote)]
    out += [v for m, v in mixed if not aliases or any(quote[m.end():].lstrip().startswith(a) for a in aliases)]
    same = _numbers_with_unit(quote, unit) if unit else []
    if len(same) >= 2:
        out.append(sum(same, Decimal(0)))
    return out


def _number_in_quote(dv: Decimal, quote: str, unit: str | None, *, equiv: bool = False) -> bool:
    if equiv and dv in _equiv_numbers(quote, unit):
        return True
    cands = _numbers_with_unit(quote, unit)
    if unit == "時間":  # 6時間10分 = 6.1666… に対して 6.17 を通す（±0.01）
        return any(abs(dv - c) <= Decimal("0.01") for c in cands if c != c.to_integral_value()) or dv in cands
    return dv in cands


def _zero_in_quote(quote: str) -> bool:
    """数の枠の 0: 引用に「ない・なし・ありません・持っていない・無い」のどれかがある。"""
    return any(w in quote for w in ZERO_WORDS)


def _dates(text: str) -> list[date]:
    out = []
    for m in _DATE.finditer(text):
        try:
            out.append(date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            continue
    return out


def _month_days(text: str) -> list[tuple[int, int]]:
    """引用の中の年のない日付（月, 日）。"""
    out = []
    for rx in (_MD_KANJI, _MD_SLASH):
        for m in rx.finditer(text):
            mo, d = int(m[1]), int(m[2])
            if 1 <= mo <= 12 and 1 <= d <= 31:
                out.append((mo, d))
    return out


def _place_yearless(month: int, day: int, anchors: Sequence[date]) -> tuple[date | None, str | None]:
    """年のない日付に年を付ける: 基準の日付のどれかとの差の絶対値が最小になる年。
    差が YEARLESS_AMBIGUOUS_DAYS 日以上なら、近い方の年が定まらないので (None, 理由)。"""
    best: tuple[int, date] | None = None
    for a in anchors:
        for y in range(a.year - 4, a.year + 5):
            try:
                d = date(y, month, day)
            except ValueError:  # 2 月 29 日は閏年だけ
                continue
            diff = abs((d - a).days)
            if best is None or diff < best[0]:
                best = (diff, d)
    if best is None:
        return None, f"{month}月{day}日 の年を決められない（そういう日付が基準の近くに無い）"
    if best[0] >= YEARLESS_AMBIGUOUS_DAYS:
        return None, (f"{month}月{day}日 の年を決められない（基準の日付との差が {YEARLESS_AMBIGUOUS_DAYS} 日以上で、"
                      "近い方の年が定まらない）")
    return best[1], None


def _clock(ampm: str | None, hour: int, minute: int) -> Clock | None:
    if minute > 59:
        return None
    if ampm == "午前":
        h = hour if hour <= 11 else None  # 午前12時は 0 時とも正午とも読まれるので読まない
    elif ampm == "午後":
        h = hour + 12 if hour <= 11 else (12 if hour == 12 else None)
    else:
        h = hour if hour <= 23 else None  # 24 時は無効
    return None if h is None else Clock(h * 60 + minute)


def _times(text: str) -> list[Clock]:
    """引用に書かれた時刻（22時・22時30分・22:30・午後10時・午前9時・正午・22時半）。"""
    out: list[Clock] = []
    for m in _TIME_COLON.finditer(text):
        c = _clock(m[1], int(m[2]), int(m[3]))
        if c is not None:
            out.append(c)
    for m in _TIME_JI.finditer(text):
        c = _clock(m[1], int(m[2]), 30 if m[4] else int(m[3] or 0))
        if c is not None:
            out.append(c)
    if _MIDDAY in text:
        out.append(Clock(12 * 60))
    return out


def _names(e: Any, slots: Mapping[str, Slot]) -> set[str]:
    """式の中の枠の名前。"""
    if isinstance(e, str):
        return {e} if e in slots else set()
    if isinstance(e, dict):
        out: set[str] = set()
        for fn, args in e.items():
            if isinstance(args, list):
                out |= {n for a in args for n in _names(a, slots)}
            elif fn == "month_end":
                out |= _names(args, slots)
        return out
    return set()


def _cond_names(c: Mapping[str, Any], slots: Mapping[str, Slot]) -> set[str]:
    """条件（入れ子も）の中の枠の名前。"""
    for k in ("all", "any"):
        if k in c:
            return {n for child in c[k] for n in _cond_names(child, slots)}
    return _names(c["left"], slots) | _names(c["right"], slots)


def _unused_slots(card: Card, values: Mapping[str, Any]) -> set[str]:
    """判定に使わない枠。derive の元の枠で、derive の先の値が書かれている物（出勤日数があれば欠勤日数は使わない）。"""
    used: set[str] = set()
    for c in (*card.constraints, *card.conditions, *card.exceptions):
        used |= _cond_names(c, card.slots)
    for target, e in card.derive.items():
        if values.get(target) is None:
            used |= _names(e, card.slots)
    return {n for n in card.slots if n not in used and n not in card.derive}


def validate_extraction(card: Card, question: str, raw: Any, *, verify: str = "strict",
                        problem_as_unknown: bool = False) -> tuple[dict[str, Any], list[str]]:
    """抽出の結果を確かめ、(値, 問題の一覧) を返す。問題が 1 つでもあれば保留にする。

    値は 数=Decimal・日付=date・bool=「あり」/「なし」・不明=None。
    判定に使わない枠の問題だけは保留の理由にしない（検査に通らない値は使わず、不明として扱う）。
    verify="equiv" は同じ量の別の言い方も照合で通す。problem_as_unknown は問題のある枠を不明にして残りの枠で判定に進む
    （結論に効かない枠の読み落としで保留にしない。結論に効けば decide が保留にする）。どちらも既定は今の動き。
    """
    problems: list[str] = []
    values: dict[str, Any] = {}
    yearless: list[tuple[str, int, int]] = []  # 年のない日付の枠（名前, 月, 日）
    if not isinstance(raw, dict):
        return {}, ["extraction is not an object"]
    body = _norm(question_body(question))
    for name, s in card.slots.items():
        item = raw.get(name)
        if not isinstance(item, dict) or set(item) != {"value", "quote"}:
            problems.append(f"{name}: needs value and quote")
            continue
        v, q = item["value"], item["quote"]
        if v is None:
            values[name] = None
            continue
        if not isinstance(q, str) or not q.strip():
            problems.append(f"{name}: value without quote")
            continue
        nq = _norm(q).strip()
        if nq not in body:
            problems.append(f"{name}: quote is not in the question")
            continue
        if s.type == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                problems.append(f"{name}: value is not a number")
                continue
            dv = Decimal(str(v))
            if dv < 0:
                problems.append(f"{name}: negative value")
                continue
            if dv == 0 and _zero_in_quote(nq):  # 「ない」「ありません」など。0 という数字は引用に無くてよい
                values[name] = dv
                continue
            if not _number_in_quote(dv, nq, s.unit, equiv=verify == "equiv"):
                problems.append(f"{name}: value {dv} is not the number in its quote")
                continue
            values[name] = dv
        elif s.type == "date":
            try:
                dd = date.fromisoformat(v) if isinstance(v, str) else None
            except ValueError:
                dd = None
            if dd is not None and dd in _dates(nq):
                values[name] = dd
            elif dd is not None and (dd.month, dd.day) in _month_days(nq):
                yearless.append((name, dd.month, dd.day))  # 年はモデルの値を使わず、後でコードが決める
            else:
                problems.append(f"{name}: date {v!r} is not the date in its quote")
                continue
        elif s.type == "time":
            tv = parse_time(v)
            if tv is None or tv not in _times(nq):
                problems.append(f"{name}: time {v!r} is not the time in its quote")
                continue
            values[name] = tv
        else:
            if v not in BOOL_VALUES:
                problems.append(f"{name}: value must be あり or なし")
                continue
            values[name] = v
    if yearless:
        anchors = _dates(body)  # 年つきの日付が同じ問にあれば、その年を基準にする
        todo = list(yearless)
        if not anchors:  # 年のない日付だけ: 最初の 1 つを固定の年に置き、他はそれに最も近い年にする
            name, mo, d = todo.pop(0)
            try:
                values[name] = date(YEARLESS_BASE_YEAR, mo, d)
                anchors = [values[name]]
            except ValueError:
                problems.append(f"{name}: {mo}月{d}日 という日付は無い")
        for name, mo, d in todo:
            placed, why = _place_yearless(mo, d, anchors) if anchors else (None, "基準の日付が決まらない")
            if placed is None:
                problems.append(f"{name}: {why}")
            else:
                values[name] = placed
    # derive の先の枠が照合に落ちた時は、保留にせず捨てる。元の枠が全部読めていればコードが計算する（decide）。
    # 元の枠が無い・落ちた時は、従来どおり保留（先の問題を残す）
    for target, e in card.derive.items():
        sources = _names(e, card.slots)
        if sources and all(values.get(n) is not None for n in sources):
            problems = [p for p in problems if p.split(":", 1)[0] != target or p.endswith("needs value and quote")]
    unused = _unused_slots(card, values)
    problems = [p for p in problems if p.split(":", 1)[0] not in unused]
    for name in unused:
        values.setdefault(name, None)
    if problem_as_unknown:
        for p in problems:
            values[p.split(":", 1)[0]] = None
        problems = []
    return values, problems


# ---------------------------------------------------------------- 式と判定


def _need(fn: str, xs: Sequence[Any], kind: type) -> None:
    if not all(type(x) is kind for x in xs):
        raise CardError(f"{fn} needs {kind.__name__} values")


def _last_day(year: int, month: int) -> int:
    nxt = date(year + month // 12, month % 12 + 1, 1)
    return (nxt - timedelta(days=1)).day


def _eval(e: Any, values: Mapping[str, Any]) -> Any:
    """式の値。拾えていない項目を含めば None（不明）。"""
    if isinstance(e, (int, float)) and not isinstance(e, bool):
        return Decimal(str(e))
    if isinstance(e, str):
        return e if e in BOOL_VALUES else values.get(e)
    (fn, args), = e.items()
    if fn == "date":
        return date.fromisoformat(args)
    if fn == "time":
        c = parse_time(args)
        if c is None:
            raise CardError(f"bad time {args!r}")
        return c
    xs = [_eval(a, values) for a in (args if isinstance(args, list) else [args])]
    if any(x is None for x in xs):
        return None
    if fn == "add":
        if isinstance(xs[0], date):
            raise CardError("use add_days for dates")
        _need(fn, xs, Decimal)
        return sum(xs[1:], xs[0])
    if fn == "sub":
        a, b = xs
        if isinstance(a, date) and isinstance(b, date):
            return Decimal((a - b).days)
        _need(fn, xs, Decimal)
        return a - b
    if fn == "mul":
        _need(fn, xs, Decimal)
        out = xs[0]
        for x in xs[1:]:
            out = out * x
        return out
    if fn == "div":
        _need(fn, xs, Decimal)
        a, b = xs
        if b == 0:
            raise CardError("division by zero")
        return a / b
    if fn in ("min", "max"):
        if len({type(x) for x in xs}) != 1 or not isinstance(xs[0], (Decimal, date, Clock)):
            raise CardError(f"{fn} needs values of one kind (number, date or time)")
        return min(xs) if fn == "min" else max(xs)
    if fn == "add_days":
        d, n = xs
        if type(d) is not date or type(n) is not Decimal:
            raise CardError("add_days needs a date and a number")
        return d + timedelta(days=int(n))
    if fn == "month_day":
        d, k, day = xs
        if type(d) is not date or type(k) is not Decimal or type(day) is not Decimal:
            raise CardError("month_day needs a date and numbers")
        y, m = divmod(d.month - 1 + int(k), 12)
        return date(d.year + y, m + 1, int(day))
    if fn == "month_end":
        (d,) = xs
        if type(d) is not date:
            raise CardError("month_end needs a date")
        return date(d.year, d.month, _last_day(d.year, d.month))
    if fn == "add_months":
        d, k = xs
        if type(d) is not date or type(k) is not Decimal or k != k.to_integral_value():
            raise CardError("add_months needs a date and a whole number of months")
        y, m = divmod(d.month - 1 + int(k), 12)
        return date(d.year + y, m + 1, min(d.day, _last_day(d.year + y, m + 1)))
    if fn == "minutes_between":
        _need(fn, xs, Clock)
        a, b = xs
        diff = b.minutes - a.minutes
        return Decimal(diff if diff >= 0 else diff + 24 * 60)  # 終了が開始より前なら夜をまたぐ
    raise CardError(f"unknown function {fn}")


def _compare(a: Any, op: str, b: Any) -> bool | None:
    if a is None or b is None:
        return None
    if type(a) is not type(b):
        raise CardError(f"cannot compare {type(a).__name__} with {type(b).__name__}")
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b, "==": a == b}[op]


def _kleene(results: Sequence[bool | None], combine: str) -> bool | None:
    if combine == "all":
        if any(r is False for r in results):
            return False
        return True if all(r is True for r in results) else None
    if any(r is True for r in results):
        return True
    return False if all(r is False for r in results) else None


def _eval_cond(c: Mapping[str, Any], vals: Mapping[str, Any]) -> bool | None:
    """条件 1 つの 3 値（True 成立 / False 不成立 / None 不明）。入れ子は Kleene の all / any。"""
    for k in ("all", "any"):
        if k in c:
            return _kleene([_eval_cond(child, vals) for child in c[k]], k)
    return _compare(_eval(c["left"], vals), c["op"], _eval(c["right"], vals))


@dataclass
class CardDecision:
    card_id: str
    conclusion: str | None  # None は保留
    hold_reason: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    conditions: list[tuple[str, bool | None]] = field(default_factory=list)
    raw: Any = None  # モデルが拾ったままの値（検査の前。値の一致率を測るため）
    # 2 回読みの記録（確認なしの成績も同じ記録から数えられるように）。first_conclusion は 1 回目だけの結論（保留は None）、
    # second_conclusion は 2 回目の結論（読み直さなかった・保留は None）。pass_stats は各回の extractor.last_stats
    first_conclusion: str | None = None
    second_conclusion: str | None = None
    second_hold_reason: str | None = None
    second_raw: Any = None
    pass_stats: list[dict[str, Any]] = field(default_factory=list)


def decide(card: Card, values: Mapping[str, Any]) -> CardDecision:
    vals = dict(values)
    for name, e in card.derive.items():
        if vals.get(name) is None:
            try:
                vals[name] = _eval(e, vals)
            except (CardError, ArithmeticError, ValueError) as ex:
                return CardDecision(card.id, None, f"派生量 {name} を計算できない（{ex}）", vals)
    try:
        for c in card.constraints:
            if _compare(_eval(c["left"], vals), c["op"], _eval(c["right"], vals)) is False:
                return CardDecision(card.id, None, f"値の筋が通らない（{c['name']}）", vals)
        # 結論になりうる語を集める。例外は順に見て、成立すればその語で終わり（以降は見ない）。
        # 不明の例外が hold なら「その例外の結論になるかもしれない」として加え、先へ進む。
        # 成立する例外が無ければ、条件の結論（不明なら両方）も加える。語が 1 つに決まれば決め、2 つなら保留。
        possible: list[str] = []
        first_unknown: str | None = None
        fired: Mapping[str, Any] | None = None
        for x in card.exceptions:
            r = _eval_cond(x, vals)
            if r is True:
                possible.append(x["conclusion"])
                fired = x
                break
            if r is None and x.get("if_unknown", "hold") == "hold":
                possible.append(x["conclusion"])
                first_unknown = first_unknown or x["name"]
        results: list[tuple[str, bool | None]] = []
        if fired is None:
            results = [(c["name"], _eval_cond(c, vals)) for c in card.conditions]
            combined = _kleene([r for _, r in results], card.combine)
            possible += list(card.conclusions) if combined is None else [card.conclusions[0] if combined else card.conclusions[1]]
    except (CardError, ArithmeticError, ValueError) as ex:
        return CardDecision(card.id, None, f"式を計算できない（{ex}）", vals)
    if len(set(possible)) == 1:
        return CardDecision(card.id, possible[0], None, vals, [(fired["name"], True)] if fired else results)
    if first_unknown is not None:
        return CardDecision(card.id, None, f"例外（{first_unknown}）に当たるか分からない", vals, [(first_unknown, None)])
    unknown = [n for n, r in results if r is None]
    return CardDecision(card.id, None, f"条件（{'・'.join(unknown)}）が質問から決まらない", vals, results)


# ---------------------------------------------------------------- 定型の説明


def _fmt(v: Any) -> str:
    if v is None:
        return "不明"
    if isinstance(v, date):
        return f"{v.year}年{v.month}月{v.day}日"
    if isinstance(v, Clock):
        return str(v)
    if isinstance(v, Decimal):
        s = format(v.normalize(), "f")
        if "." in s:
            return s
        return f"{int(s):,}"
    return str(v)


def render_answer(card: Card, decision: CardDecision) -> str:
    src = card.doc["source"] + (f" {card.doc['section']}" if card.doc.get("section") else "")
    lines = [f"規程（{src}）の「{card.doc['quote']}」に当てはめました。"]
    shown = [f"{n} {_fmt(decision.values.get(n))}{card.slots[n].unit or ''}" if decision.values.get(n) is not None
             else f"{n} 不明" for n in card.slots]
    lines.append("質問から読み取った値: " + "、".join(shown) + "。")
    mark = {True: "成立", False: "不成立", None: "不明"}
    for name, r in decision.conditions:
        lines.append(f"- {name}: {mark[r]}")
    if decision.conclusion is None:
        lines.append(f"判断に必要な事実が足りないため保留します（{decision.hold_reason}）。")
        lines.append("結論: 保留")
    else:
        lines.append(f"結論: {decision.conclusion}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 抽出役（小型モデル）


class Extractor(Protocol):
    async def extract(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> Any: ...


class OpenAICompatExtractor:
    """llama.cpp（llama-server）/ vLLM の OpenAI 互換 API で、json_schema に沿った JSON を 1 回だけ拾う。温度 0。"""

    def __init__(self, base_url: str, model: str, *, timeout_s: float = 30.0, client: Any = None) -> None:
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=base_url, api_key="NONE", timeout=timeout_s)
        self._client = client
        self._model = model
        self.last_stats: dict[str, Any] = {}  # 直前の 1 回の数（測定用）

    async def extract(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> Any:
        self.last_stats = {}  # 通信で落ちた問に前の問の数が残らないように
        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0, max_tokens=512,
            response_format={"type": "json_schema", "json_schema": {"name": "slots", "schema": schema, "strict": True}},
        )
        msg = resp.choices[0].message
        # Gemma 4 は json_schema でも考える文が出る事がある (llama.cpp #30256)。枠を食うので測れるように残す
        extra = getattr(msg, "model_extra", None) or {}
        self.last_stats = {"finish_reason": getattr(resp.choices[0], "finish_reason", None),
                           "completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
                           "reasoning_chars": len(extra.get("reasoning_content") or "")}
        return json.loads(msg.content or "null")


CONFIRM_PASSES = 2  # 1 回目で結論が決まった時、枠の順を逆にした別の呼び出しで読み直す回数（1 なら確認しない）
HOLD_MISMATCH = "2 回の読みが一致しない"


async def _read_once(card: Card, question: str, extractor: Extractor, *, reverse: bool) -> tuple[CardDecision, dict[str, Any]]:
    """抽出 1 回 → 検査 → decide。抽出の失敗・形の崩れ・検査に通らない値は保留。(判定, その回の last_stats)。"""
    try:
        raw = await extractor.extract(extraction_messages(card, question, reverse=reverse),
                                      extraction_schema(card, reverse=reverse))
    except Exception as e:  # noqa: BLE001  抽出の失敗はすべて保留
        return CardDecision(card.id, None, f"抽出に失敗した（{type(e).__name__}）"), dict(getattr(extractor, "last_stats", {}) or {})
    stats = dict(getattr(extractor, "last_stats", {}) or {})
    values, problems = validate_extraction(card, question, raw)
    if problems:
        return CardDecision(card.id, None, "抽出の検査に通らない: " + "; ".join(problems[:3]), values, raw=raw), stats
    d = decide(card, values)
    d.raw = raw
    return d, stats


async def answer_with_card(card: Card, question: str, extractor: Extractor) -> tuple[str, CardDecision]:
    """カードで答える。抽出の失敗・形の崩れ・検査に通らない値は保留（モデルだけの答えには戻さない）。
    1 回目で結論が決まった時だけ、枠の順を逆にした 2 回目の抽出を別の呼び出しで行い、2 回の結論が同じ時だけ決める
    （問い方を変えた呼び出しの不一致を保留の信号に使う）。"""
    d, stats = await _read_once(card, question, extractor, reverse=False)
    d.pass_stats = [stats]
    d.first_conclusion = d.conclusion
    if d.conclusion is not None and CONFIRM_PASSES >= 2:
        d2, stats2 = await _read_once(card, question, extractor, reverse=True)
        d.pass_stats.append(stats2)
        d.second_conclusion, d.second_hold_reason, d.second_raw = d2.conclusion, d2.hold_reason, d2.raw
        if d2.conclusion != d.conclusion:
            d.conclusion, d.hold_reason, d.conditions = None, HOLD_MISMATCH, []
    return render_answer(card, d), d
