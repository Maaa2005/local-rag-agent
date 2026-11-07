"""既定値を使わない読みの試作（Issue #16・2026-11-07 の事前登録）。

判定の木は規程カードのまま。枠を 値/範囲/分からないと言う/触れていない で読み、全部の値で結論が 1 つの時だけ決める。
腕 A: 全部の枠を 1 回で読む。腕 B: 数と日付は 1 回で読み、あり/なしは結論が割れている間だけ 1 つずつ聞く。

    python -m evaluation.gpu_local.open_read upper --cards <dir> --questions <jsonl> --gold <json>
    python -m evaluation.gpu_local.open_read run --arm A --cards <dir> --questions <jsonl> --out <dir> --server <exe> --gguf <gguf>
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import subprocess
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

os.environ.setdefault("RULE_CARDS_V2", "true")
from app.services import rule_cards as rc  # noqa: E402

BIG = Decimal(10) ** 13  # 上限が言われない範囲の上の端
MAX_COMBOS = 512
GUIDE = """質問の文を読み、聞かれた項目を次のどれかで答えます。
- 値: 質問が値を言っている、または文から必ず言える。
- 範囲: 上限か下限か幅だけ言っている（「多く見積もっても20時間」は hi=20。「10万円はありません」は hi=100000。「28万円から32万円」は lo と hi）。
- 不明: 質問者が分からない・確認していない・まだ聞いていない・測っていないと言っている。
- 触れない: その話題に質問が触れていない。

読み方の決まり:
- 「〜はなく」「〜はありません」は なし、数なら 0。
- 質問者が別の種類の物だと言っていれば、その種類ではないと読みます（「通信の仕事です」なら 電気・ガス・水道の復旧か は なし。「購入します」「発注します」なら リースか は なし）。
- 「か月」と「年」、「千円」と「円」のような同じ量の言い換えは項目の単位に直します。「入社5年目」は lo=48 hi=59（か月）のように範囲です。
- 今日・明日・来週のように今からの言い方だけで日付が無い時は日付を作らず、差の項目があれば差で答えます（明日は 1、明後日は 2、来週は lo=1 hi=13）。
- 年や月までしか言っていない日付は範囲です（「2024年6月」は lo=2024-06-01 hi=2024-06-30。年の無い「4月」は lo=2024-04-01 hi=2024-04-30）。「先週」のような言い方は日付を作らず 触れない です。「午前中」は時刻の hi=12:00、「午後」は lo=12:00 です。
- 項目の説明に表（「議事録は3年」）があり質問がその種類を言っていれば、その数が値です。説明に月日が決まっていて（「毎年10月1日」）質問が年を言っていれば、その年の日付が値です。
- 項目は質問者本人（同僚の話なら同僚）の事です。家族など別の人の事を当てはめません。
- 計算が要る数は 式 で答えます。terms に [数, 掛ける数] を並べるとコードが足します（「2時間を3回と10時間」は [[2,3],[10,1]]、「4年を毎月」は [[4,12]]、「11月と7月の差」は [[11,1],[-7,1]]）。数は質問の文にある物だけです。
- 推測で埋めません。
quote には値を含む一番短い部分を、質問の文から言い換えずにそのまま入れます。式は使う数が全部入る部分です。触れない時は quote を空にします。"""

_ZERO_MORE = ("なく", "無く", "いません", "ていない", "無し", "ゼロ", "まだ")
_RELATIVE = {"明後日": (2, 2), "明日": (1, 1), "翌日": (1, 1), "来週": (1, 13), "今日": (0, 0), "当日": (0, 0)}
_TABLE = re.compile(r"([^、。（）\s]{2,}?)は(\d+(?:\.\d+)?)")
_DESC_MD = re.compile(r"(\d{1,2})月(\d{1,2})日")


def _value_schema(s: rc.Slot) -> dict[str, Any]:
    if s.type == "number":
        return {"type": "number"}
    if s.type == "date":
        return {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
    if s.type == "time":
        return {"type": "string", "pattern": rc.TIME_PATTERN}
    return {"type": "string", "enum": list(rc.BOOL_VALUES)}


def _slot_schema(s: rc.Slot) -> dict[str, Any]:
    v = _value_schema(s)

    def obj(status: str, **more: Any) -> dict[str, Any]:
        props = {"quote": {"type": "string"}, "status": {"const": status}, **more}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    forms = [obj("値", value=v)]
    if s.type != "bool":
        forms.append(obj("範囲", lo={"anyOf": [v, {"type": "null"}]}, hi={"anyOf": [v, {"type": "null"}]}))
    if s.type == "number":
        pair = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
        forms.append(obj("式", terms={"type": "array", "items": pair, "minItems": 1, "maxItems": 6}))
    forms += [obj("不明"), obj("触れない")]
    return {"anyOf": forms}


def read_schema(slots: list[rc.Slot]) -> dict[str, Any]:
    props = {s.name: _slot_schema(s) for s in slots}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _kind(s: rc.Slot) -> str:
    return {"number": f"数（単位 {s.unit}）" if s.unit else "数", "date": "日付（YYYY-MM-DD）",
            "time": "時刻（HH:MM・24 時間表記）", "bool": "「あり」か「なし」"}[s.type]


def read_messages(slots: list[rc.Slot], question: str) -> list[dict[str, str]]:
    lines = "\n".join(f"- {s.name}: {s.describe}。{_kind(s)}" for s in slots)
    return [{"role": "system", "content": GUIDE + "\n\n項目:\n" + lines},
            {"role": "user", "content": rc.question_body(question).strip()}]


def used_slots(card: rc.Card) -> set[str]:
    used = {n for c in (*card.constraints, *card.conditions, *card.exceptions) for n in rc._cond_names(c, card.slots)}
    changed = True
    while changed:
        changed = False
        for target, e in card.derive.items():
            if target in used:
                new = rc._names(e, card.slots) - used
                if new:
                    used |= new
                    changed = True
    return used


def claim_only(card: rc.Card) -> set[str]:
    return {k for k, v in (card.raw.get("claim_only") or {}).items()
            if k in card.slots and card.slots[k].type == "bool" and isinstance(v, dict) and v.get("basis")}


def _years_ordinal(d: Decimal, nq: str, unit: str | None) -> bool:
    """「入社5年目」の か月 の端（48 と 59）だけ通す。"""
    if unit != "か月":
        return False
    return any(d in ((int(y) - 1) * 12, int(y) * 12 - 1) for y in re.findall(r"(\d+)年目", nq))


def _table_value(s: rc.Slot, d: Decimal, nq: str) -> bool:
    """項目の説明の表（「稟議書は7年」）の数で、その種類の語が引用にある時だけ通す。別の種類の語も引用にあれば通さない。"""
    def hit(key: str) -> bool:
        return key in nq or any(key[i:i + 3] in nq for i in range(len(key) - 2))
    rows = [(k, Decimal(n)) for k, n in _TABLE.findall(s.describe or "")]
    return any(n == d for k, n in rows if hit(k)) and len({n for k, n in rows if hit(k)}) == 1


def _extra_number(s: rc.Slot, d: Decimal, nq: str) -> bool:
    """本体の照合に落ちた数のうち、同じ量だとコードで確かめられる物。"""
    if d == 0 and any(w in nq for w in _ZERO_MORE):
        return True
    if s.unit == "時間" and d * 60 in rc._numbers_with_unit(nq, "分"):
        return True
    if s.unit == "km/h" and re.search(rf"時速\s*{d.normalize():f}\s*(キロ|km)", nq):
        return True
    if s.unit == "日" and any(w in nq and a == b == d for w, (a, b) in _RELATIVE.items()):
        return True
    return _table_value(s, d, nq)


def _extra_date(s: rc.Slot, d: date, nq: str) -> bool:
    """説明に月日が決まっていて、引用が年を言っている時（「2026年の基準日」と「毎年10月1日」）。"""
    return str(d.year) + "年" in nq and any((int(m), int(x)) == (d.month, d.day) for m, x in _DESC_MD.findall(s.describe or ""))


def _partial_date(d: date, nq: str) -> bool:
    """「2024年6月」「2010年」の範囲の端（その月・その年の最初と最後の日）。"""
    if re.search(rf"{d.year}年0?{d.month}月(?!\d)", nq):
        return d.day == 1 or d.day == rc._last_day(d.year, d.month)
    if re.search(rf"{d.year}年(?!\s*\d+月)", nq):
        return (d.month, d.day) in ((1, 1), (12, 31))
    return False


def _bound(s: rc.Slot, v: Any, nq: str, anchors: list[date], body: str = "") -> Any:
    """範囲の端を引用と照合して値にする。通らなければ None。"""
    if v is None:
        return None
    if s.type == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            return None
        d = Decimal(str(v))
        ok = (rc._number_in_quote(d, nq, s.unit, equiv=True, same_units=s.same_units) or _years_ordinal(d, nq, s.unit)
              or (s.unit == "日" and any(w in nq and a <= d <= b for w, (a, b) in _RELATIVE.items())))
        return d if ok else None
    if s.type == "date":
        try:
            d = date.fromisoformat(v)
        except (TypeError, ValueError):
            return None
        if d in rc._dates(nq) or _partial_date(d, nq):
            return d
        if (d.month, d.day) in rc._month_days(nq):
            if f"{d.year}年" in body:  # 年は別の文で言っている（「2016年の秋に入社」「入社日は9月1日」）
                return d
            return rc._place_yearless(d.month, d.day, anchors)[0] if anchors else date(rc.YEARLESS_BASE_YEAR, d.month, d.day)
        if re.search(rf"(?<!\d)0?{d.month}月(?!\d)", nq) and "年" not in nq and (d.day == 1 or d.day == rc._last_day(d.year, d.month)):
            return date(rc.YEARLESS_BASE_YEAR, d.month, d.day)  # 「4月」だけ。年は使わない
        return None
    if s.type == "time":
        t = rc.parse_time(v)
        if t is None:
            return None
        half = (v == "12:00" and ("午前中" in nq or "午後" in nq))
        return t if t in rc._times(nq) or half else None
    return None


def _terms_value(s: rc.Slot, terms: Any, nq: str) -> Decimal | None:
    """式の値。使う数が全部引用にあり（掛ける数の 1 と 12 は別）、合計が 0 以上の時だけ。"""
    if not isinstance(terms, list) or not terms:
        return None
    have = {Decimal(m.group().replace(",", "")) for m in rc._NUM.finditer(nq)}
    total = Decimal(0)
    for t in terms:
        if not (isinstance(t, list) and len(t) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in t)):
            return None
        a, b = Decimal(str(t[0])), Decimal(str(t[1]))
        if abs(a) not in have or (b not in (1, 12) and b not in have):
            return None
        total += a * b
    return total if total >= 0 else None


def to_candidates(card: rc.Card, question: str, reading: dict[str, Any]) -> dict[str, list[Any] | None]:
    """読みを枠ごとの候補にする。None は不明。引用が問に無い・照合に落ちた値は不明へ。"""
    body = rc._norm(rc.question_body(question))
    claim = claim_only(card)
    plain: dict[str, Any] = {}
    for name in card.slots:
        it = reading.get(name)
        if isinstance(it, dict) and it.get("status") == "値":
            plain[name] = {"quote": it.get("quote"), "value": it.get("value")}
        else:
            plain[name] = {"quote": None, "value": None}
    values, _ = rc.validate_extraction(card, question, plain, verify="equiv", problem_as_unknown=True)
    anchors = rc._dates(body) or [v for v in values.values() if isinstance(v, date)]
    out: dict[str, list[Any] | None] = {}
    for name, s in card.slots.items():
        it = reading.get(name)
        st = it.get("status") if isinstance(it, dict) else None
        out[name] = None
        q = rc._norm(str(it.get("quote") or "")).strip() if isinstance(it, dict) else ""
        quoted = bool(q) and q in body and not rc._HEDGE.search(q)
        if st == "値":
            if values.get(name) is not None:
                out[name] = [values[name]]
            elif quoted and s.type == "number" and isinstance(it.get("value"), (int, float)) and not isinstance(it.get("value"), bool):
                d = Decimal(str(it["value"]))
                if d >= 0 and _extra_number(s, d, q):
                    out[name] = [d]
            elif quoted and s.type == "date":
                try:
                    d = date.fromisoformat(str(it.get("value")))
                except ValueError:
                    continue
                if _extra_date(s, d, q):
                    out[name] = [d]
        elif st == "式" and s.type == "number":
            d = _terms_value(s, it.get("terms"), q) if quoted else None
            if d is not None:
                out[name] = [d]
        elif st == "範囲" and s.type != "bool":
            if not q or q not in body:
                continue
            lo, hi = _bound(s, it.get("lo"), q, anchors, body), _bound(s, it.get("hi"), q, anchors, body)
            if (it.get("lo") is not None and lo is None) or (it.get("hi") is not None and hi is None) or (lo is None and hi is None):
                continue
            if s.type == "number":
                out[name] = [lo if lo is not None else Decimal(0), hi if hi is not None else BIG]
            elif s.type == "time":
                out[name] = [lo or rc.parse_time("00:00"), hi or rc.parse_time("23:59")]
            elif lo is not None and hi is not None:
                out[name] = [lo, hi]
        elif st == "触れない" and name in claim:
            out[name] = ["なし"]
    # 年を補った日付と本当の年の日付が混じると差が嘘になる。年を補った方を、言っていれば本当の年に、無ければ不明に。
    real_year = any(
        c and isinstance(x, date) and x.year != rc.YEARLESS_BASE_YEAR for c in out.values() if c for x in c)
    if real_year and f"{rc.YEARLESS_BASE_YEAR}年" not in body:
        for name, c in out.items():
            if not c or not all(isinstance(x, date) for x in c) or all(x.year != rc.YEARLESS_BASE_YEAR for x in c):
                continue
            it = reading.get(name) if isinstance(reading.get(name), dict) else {}
            raw = [it.get("value")] if it.get("status") == "値" else [it.get("lo"), it.get("hi")]
            try:
                fixed = [date.fromisoformat(r) for r in raw if r]
            except (TypeError, ValueError):
                fixed = []
            if fixed and all(f"{x.year}年" in body and (x.month, x.day) == (y.month, y.day) for x, y in zip(fixed, c)):
                out[name] = fixed
            else:
                out[name] = None
    return out


def evaluate(card: rc.Card, cand: dict[str, list[Any] | None]) -> tuple[str | None, list[str], str | None]:
    """全部の組で結論が 1 つなら結論。戻りは (結論, 結論を割っている不明のあり/なしの枠, 保留の理由)。"""
    used = used_slots(card)
    fixed = {n: (c[0] if c and len(c) == 1 else None) for n, c in cand.items()}
    multi = [n for n, c in cand.items() if c and len(c) > 1 and n in used]
    unknown_bool = [n for n, s in card.slots.items() if s.type == "bool" and cand.get(n) is None and n in used]
    axes: list[list[Any]] = [cand[n] for n in multi] + [list(rc.BOOL_VALUES)] * len(unknown_bool)
    names = multi + unknown_bool
    total = 1
    for a in axes:
        total *= len(a)
    if total > MAX_COMBOS:
        return None, unknown_bool, "組が多すぎる"
    results: collections.Counter = collections.Counter()
    why = None
    for combo in itertools.product(*axes):
        d = rc.decide(card, {**fixed, **dict(zip(names, combo))})
        results[d.conclusion] += 1
        why = why or d.hold_reason
    if len(results) == 1 and None not in results:
        return next(iter(results)), [], None
    split = []
    for b in unknown_bool:
        rest = [x for x in unknown_bool if x != b]
        per_value = []
        for v in rc.BOOL_VALUES:
            got = set()
            for combo in itertools.product(*([cand[m] for m in multi] + [list(rc.BOOL_VALUES)] * len(rest))):
                got.add(rc.decide(card, {**fixed, **dict(zip(multi + rest, combo)), b: v}).conclusion)
            per_value.append(got)
        if per_value[0] != per_value[1]:
            split.append(b)
    return None, split or unknown_bool, why or "結論が割れる"


class LLM:
    def __init__(self, port: int, temperature: float = 0, seed: int = 1) -> None:
        self.url, self.calls, self.tokens = f"http://127.0.0.1:{port}/v1/chat/completions", 0, 0
        self.temperature, self.seed = temperature, seed

    def __call__(self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int) -> Any:
        import urllib.request
        body = {"model": "local-llm", "messages": messages, "temperature": self.temperature, "seed": self.seed,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema, "strict": True}}}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        self.calls += 1
        self.tokens += d.get("usage", {}).get("completion_tokens", 0)
        return json.loads(d["choices"][0]["message"]["content"] or "null")


def answer(arm: str, card: rc.Card, question: str, llm: LLM) -> dict[str, Any]:
    used = used_slots(card)
    slots = [s for s in card.slots.values() if s.name in used]
    first = [s for s in slots if arm == "A" or s.type != "bool"]
    reading: dict[str, Any] = {}
    if first:
        reading.update(llm(read_messages(first, question), read_schema(first), 1500) or {})
    cand = to_candidates(card, question, reading)
    concl, split, why = evaluate(card, cand)
    asked: set[str] = set()
    while arm == "B" and concl is None:
        todo = [b for b in split if b not in asked]
        if not todo:
            break
        s = card.slots[todo[0]]
        asked.add(s.name)
        reading[s.name] = (llm(read_messages([s], question), read_schema([s]), 200) or {}).get(s.name)
        cand = to_candidates(card, question, reading)
        concl, split, why = evaluate(card, cand)
    return {"conclusion": concl, "hold_reason": why, "reading": reading}


def load_cards(d: str) -> list[rc.Card]:
    cards = rc.load_cards(d, allow_draft=True)
    for c in cards:
        if "unless_mentioned" in json.dumps(c.raw, ensure_ascii=False) or any(s.absent is not None for s in c.slots.values()):
            raise SystemExit(f"{c.id}: 既定の値が残っている")
    return cards


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def score(rows: list[dict]) -> dict[str, Any]:
    s: dict[str, Any] = {"correct": 0, "wrong": 0, "hold_wrong": 0, "held": 0, "stopped": 0}
    pairs: dict[str, list[bool]] = collections.defaultdict(list)
    rules: collections.Counter = collections.Counter()
    for r in rows:
        c = r["conclusion"]
        if r["should_hold"]:
            s["held"] += c is None
            s["hold_wrong"] += c is not None
        elif c == r["expected"]:
            s["correct"] += 1
            rules[r["rule_id"]] += 1
        elif c:
            s["wrong"] += 1
        else:
            s["stopped"] += 1
        if r.get("pair_id"):
            pairs[r["pair_id"]].append(c == r["expected"])
    s["pairs"] = sum(all(v) for v in pairs.values())
    s["per_rule_min"] = min(rules.values()) if len(rules) == 6 else 0
    s["per_rule"] = dict(rules)
    s["ok"] = bool(s["correct"] >= 65 and s["wrong"] == 0 and s["hold_wrong"] == 0 and s["held"] == 18
                   and s["pairs"] >= 11 and s["per_rule_min"] >= 10)
    secs = sorted(r["seconds"] for r in rows if r.get("seconds") is not None)
    if secs:
        s["sec_p50"], s["sec_p95"], s["sec_max"] = secs[len(secs) // 2], secs[int(len(secs) * 0.95)], secs[-1]
        s["calls_mean"] = round(sum(r["calls"] for r in rows) / len(rows), 2)
    return s


def row_base(q: dict) -> dict[str, Any]:
    return {"id": q["id"], "rule_id": q["rule_id"], "group": q["group"], "should_hold": q["should_hold"],
            "expected": q["expected_conclusion"], "pair_id": q.get("pair_id")}


def print_misses(rows: list[dict]) -> None:
    for r in rows:
        if (r["should_hold"] and r["conclusion"]) or (not r["should_hold"] and r["conclusion"] != r["expected"]):
            print(r["id"], r["group"], r["rule_id"][:4], r["conclusion"], "exp", r["expected"], "|", (r.get("hold_reason") or "")[:70])


def cmd_upper(a: argparse.Namespace) -> None:
    cards = load_cards(a.cards)
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    rows = []
    for q in read_jsonl(a.questions):
        card = rc.select_card(cards, q["question"])
        r = {**row_base(q), "conclusion": None, "hold_reason": "カード無し", "card_id": card.id if card else None}
        if card:
            r["conclusion"], _, r["hold_reason"] = evaluate(card, to_candidates(card, q["question"], gold.get(q["id"]) or {}))
        rows.append(r)
    print(json.dumps(score(rows), ensure_ascii=False))
    print_misses(rows)


def cmd_run(a: argparse.Namespace) -> None:
    cards = load_cards(a.cards)
    qs = read_jsonl(a.questions)
    from evaluation.colab import step05 as s5
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    argv = s5.llama_server_argv(a.server, a.gguf, a.port, a.ctx, extra=["--cache-ram", "1024"])
    with open(out / f"server_{a.arm}{a.round}.log", "w") as log:
        srv = subprocess.Popen(argv, stdout=log, stderr=log)
    try:
        import urllib.request
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        llm = LLM(a.port, a.temperature, a.seed)
        rows = []
        with open(out / f"raw_{a.arm}{a.round}.jsonl", "w", encoding="utf-8") as f:
            for q in qs:
                card = rc.select_card(cards, q["question"])
                r = {**row_base(q), "conclusion": None, "hold_reason": "カード無し", "card_id": card.id if card else None}
                c0, t0 = llm.calls, time.time()
                if card:
                    try:
                        r.update(answer(a.arm, card, q["question"], llm))
                    except Exception as e:  # noqa: BLE001  読みの失敗は保留
                        r["hold_reason"] = f"読みの失敗 {type(e).__name__}: {e}"[:200]
                r["seconds"], r["calls"] = round(time.time() - t0, 3), llm.calls - c0
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        s = score(rows)
        (out / f"summary_{a.arm}{a.round}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(s, ensure_ascii=False))
        print_misses(rows)
    finally:
        srv.terminate()
        srv.wait(30)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upper")
    r = sub.add_parser("run")
    for x in (u, r):
        x.add_argument("--cards", required=True)
        x.add_argument("--questions", required=True)
    u.add_argument("--gold", required=True)
    r.add_argument("--arm", choices=("A", "B"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--server", required=True)
    r.add_argument("--gguf", required=True)
    r.add_argument("--port", type=int, default=8098)
    r.add_argument("--ctx", type=int, default=4096)
    r.add_argument("--round", default="1")
    r.add_argument("--temperature", type=float, default=0)
    r.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    (cmd_upper if a.cmd == "upper" else cmd_run)(a)


if __name__ == "__main__":
    main()
