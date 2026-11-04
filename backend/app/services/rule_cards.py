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
  slots{名前: {type: number|date|bool, unit?, describe}},
  derive{名前: 式}（拾えなかった項目を他の項目から計算する。例: 出勤日数 = 全労働日数 - 欠勤日数）,
  constraints[{name, left, op, right}]（値の筋が通らなければ保留。例: 出勤日数 <= 全労働日数）,
  conditions[{name, left, op, right}], combine: all|any, conclusion{then, else},
  exceptions[{name, left, op, right, conclusion, if_unknown: hold|ignore}]
式: 数・項目名・{"date": "YYYY-MM-DD"}・{"add": [...]}・{"sub": [a, b]}・{"mul": [...]}・{"div": [a, b]}・
    {"add_days": [日付, n]}・{"month_day": [日付, 何か月後, 日]}
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
SLOT_TYPES = ("number", "date", "bool")
BOOL_VALUES = ("あり", "なし")
# 数え方の語の言い換え（引用の数の直後にあるかを見る）
UNIT_ALIASES = {"か月": ("か月", "ヶ月", "カ月", "箇月", "ケ月"), "%": ("%", "パーセント")}
_EXPR_FUNCS = {"add", "sub", "mul", "div", "add_days", "month_day", "date"}
_OPTIONS = re.compile(r"「結論\s*[:：]\s*(.+?)」")
_LAST_LINE = re.compile(r"最後の行")
_NUM = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


class CardError(ValueError):
    """カードの形がおかしい（読み込みで止める）。"""


@dataclass(frozen=True)
class Slot:
    name: str
    type: str
    unit: str | None
    describe: str


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
        if not isinstance(args, list) or not args:
            raise CardError(f"{card_id}: {fn} needs a list of arguments")
        arity = {"sub": 2, "div": 2, "add_days": 2, "month_day": 3}.get(fn)
        if arity is not None and len(args) != arity:
            raise CardError(f"{card_id}: {fn} takes {arity} arguments")
        for a in args:
            _check_expr(card_id, a, slots)
        return
    raise CardError(f"{card_id}: bad expression {e!r}")


def _check_cond(card_id: str, c: Any, slots: Mapping[str, Slot], extra: Sequence[str] = ()) -> None:
    if not isinstance(c, dict) or not {"name", "left", "op", "right"} <= set(c) or set(c) - {"name", "left", "op", "right", *extra}:
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
    return [c for c in cards if allow_draft or c.status == "confirmed"]


# ---------------------------------------------------------------- 質問の扱い


def question_body(question: str) -> str:
    """答えの書式の指示（「最後の行」以降）を除いた質問。"""
    m = _LAST_LINE.search(question)
    return question[: m.start()] if m else question


def conclusion_options(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(o.strip() for o in _OPTIONS.findall(question)))


def select_card(cards: Sequence[Card], question: str) -> Card | None:
    """質問の結論の選択肢がカードの結論と同じで、カードの手がかりの語が質問にあるカード。ただ 1 枚の時だけ返す。"""
    opts = set(conclusion_options(question))
    body = unicodedata.normalize("NFKC", question_body(question))
    hits = [c for c in cards if opts == set(c.conclusions) and (not c.keywords or any(k in body for k in c.keywords))]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------- 抽出の形


def extraction_schema(card: Card) -> dict[str, Any]:
    """llama.cpp / vLLM の json_schema に渡す形。項目ごとに value と quote（質問からそのまま抜き出した部分）。"""
    props: dict[str, Any] = {}
    for s in card.slots.values():
        if s.type == "number":
            value: dict[str, Any] = {"type": ["number", "null"]}
        elif s.type == "date":
            value = {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"}
        else:
            value = {"type": ["string", "null"], "enum": [*BOOL_VALUES, None]}
        props[s.name] = {"type": "object", "properties": {"value": value, "quote": {"type": ["string", "null"]}},
                         "required": ["value", "quote"], "additionalProperties": False}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def extraction_messages(card: Card, question: str) -> list[dict[str, str]]:
    lines = []
    for s in card.slots.values():
        kind = {"number": f"数（単位 {s.unit}）" if s.unit else "数", "date": "日付（YYYY-MM-DD）",
                "bool": "「あり」か「なし」"}[s.type]
        lines.append(f"- {s.name}: {s.describe}。{kind}")
    system = ("質問の文から、次の項目の値と、その値が書かれた質問の部分を JSON で答えてください。\n"
              "quote には質問の文をそのまま抜き出して入れます（言い換えない）。\n"
              "質問に書かれていない項目は value と quote を null にします。計算・推測・換算はしません。\n"
              + "\n".join(lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": question_body(question).strip()}]


# ---------------------------------------------------------------- 抽出の検査


def _norm(s: str) -> str:
    return numnorm.normalize_numbers(s)


def _numbers_with_unit(text: str, unit: str | None) -> list[Decimal]:
    out = []
    aliases = UNIT_ALIASES.get(unit or "", (unit,)) if unit else ()
    for m in _NUM.finditer(text):
        if aliases and not any(text[m.end():].lstrip().startswith(a) for a in aliases):
            continue
        try:
            out.append(Decimal(m.group().replace(",", "")))
        except InvalidOperation:
            continue
    return out


def _dates(text: str) -> list[date]:
    out = []
    for m in _DATE.finditer(text):
        try:
            out.append(date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            continue
    return out


def validate_extraction(card: Card, question: str, raw: Any) -> tuple[dict[str, Any], list[str]]:
    """抽出の結果を確かめ、(値, 問題の一覧) を返す。問題が 1 つでもあれば保留にする。

    値は 数=Decimal・日付=date・bool=「あり」/「なし」・不明=None。
    """
    problems: list[str] = []
    values: dict[str, Any] = {}
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
            if dv not in _numbers_with_unit(nq, s.unit):
                problems.append(f"{name}: value {dv} is not the number in its quote")
                continue
            values[name] = dv
        elif s.type == "date":
            try:
                dd = date.fromisoformat(v) if isinstance(v, str) else None
            except ValueError:
                dd = None
            if dd is None or dd not in _dates(nq):
                problems.append(f"{name}: date {v!r} is not the date in its quote")
                continue
            values[name] = dd
        else:
            if v not in BOOL_VALUES:
                problems.append(f"{name}: value must be あり or なし")
                continue
            values[name] = v
    return values, problems


# ---------------------------------------------------------------- 式と判定


def _eval(e: Any, values: Mapping[str, Any]) -> Any:
    """式の値。拾えていない項目を含めば None（不明）。"""
    if isinstance(e, (int, float)) and not isinstance(e, bool):
        return Decimal(str(e))
    if isinstance(e, str):
        return e if e in BOOL_VALUES else values.get(e)
    (fn, args), = e.items()
    if fn == "date":
        return date.fromisoformat(args)
    xs = [_eval(a, values) for a in args]
    if any(x is None for x in xs):
        return None
    if fn == "add":
        if isinstance(xs[0], date):
            raise CardError("use add_days for dates")
        return sum(xs[1:], xs[0])
    if fn == "sub":
        a, b = xs
        if isinstance(a, date) and isinstance(b, date):
            return Decimal((a - b).days)
        return a - b
    if fn == "mul":
        out = xs[0]
        for x in xs[1:]:
            out = out * x
        return out
    if fn == "div":
        a, b = xs
        if b == 0:
            raise CardError("division by zero")
        return a / b
    if fn == "add_days":
        d, n = xs
        return d + timedelta(days=int(n))
    if fn == "month_day":
        d, k, day = xs
        y, m = divmod(d.month - 1 + int(k), 12)
        return date(d.year + y, m + 1, int(day))
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


@dataclass
class CardDecision:
    card_id: str
    conclusion: str | None  # None は保留
    hold_reason: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    conditions: list[tuple[str, bool | None]] = field(default_factory=list)
    raw: Any = None  # モデルが拾ったままの値（検査の前。値の一致率を測るため）


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
        for x in card.exceptions:
            r = _compare(_eval(x["left"], vals), x["op"], _eval(x["right"], vals))
            if r is True:
                return CardDecision(card.id, x["conclusion"], None, vals, [(x["name"], True)])
            if r is None and x.get("if_unknown", "hold") == "hold":
                return CardDecision(card.id, None, f"例外（{x['name']}）に当たるか分からない", vals, [(x["name"], None)])
        results = [(c["name"], _compare(_eval(c["left"], vals), c["op"], _eval(c["right"], vals)))
                   for c in card.conditions]
    except (CardError, ArithmeticError, ValueError) as ex:
        return CardDecision(card.id, None, f"式を計算できない（{ex}）", vals)
    combined = _kleene([r for _, r in results], card.combine)
    if combined is None:
        unknown = [n for n, r in results if r is None]
        return CardDecision(card.id, None, f"条件（{'・'.join(unknown)}）が質問から決まらない", vals, results)
    return CardDecision(card.id, card.conclusions[0] if combined else card.conclusions[1], None, vals, results)


# ---------------------------------------------------------------- 定型の説明


def _fmt(v: Any) -> str:
    if v is None:
        return "不明"
    if isinstance(v, date):
        return f"{v.year}年{v.month}月{v.day}日"
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


async def answer_with_card(card: Card, question: str, extractor: Extractor) -> tuple[str, CardDecision]:
    """カードで答える。抽出の失敗・形の崩れ・検査に通らない値は保留（モデルだけの答えには戻さない）。"""
    try:
        raw = await extractor.extract(extraction_messages(card, question), extraction_schema(card))
    except Exception as e:  # noqa: BLE001  抽出の失敗はすべて保留
        d = CardDecision(card.id, None, f"抽出に失敗した（{type(e).__name__}）")
        return render_answer(card, d), d
    values, problems = validate_extraction(card, question, raw)
    if problems:
        d = CardDecision(card.id, None, "抽出の検査に通らない: " + "; ".join(problems[:3]), values, raw=raw)
        return render_answer(card, d), d
    d = decide(card, values)
    d.raw = raw
    return render_answer(card, d), d
