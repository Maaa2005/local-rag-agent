"""聞き返しの試作（Issue #29）。判定の木はそのまま、足りない値をコードが選んで聞き、読んだ値を確かめる。

答えは選択か数字の入力で受ける想定なので、モデルには読ませない。模擬の利用者は隠した本当の値から答える。
最初の読みは open_read の記録を使うので、ここでは推論しない。

    python -m evaluation.gpu_local.ask_back run --cards <dir> --questions <jsonl> --reads <raw.jsonl か gold.json> --truth <json> --out <json>
    python -m evaluation.gpu_local.ask_back devtruth --cards <dir> --questions <jsonl> --gold <json> --out <json>
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from evaluation.gpu_local import open_read as o  # 設定を入れてから判定の部品を読むので、これを先に

rc = o.rc

ARMS = {  # 名: (読むか, 確かめるか, 選び方, 聞くか)
    "基準": (True, False, "greedy", False),
    "聞く": (True, False, "greedy", True),
    "本命": (True, True, "greedy", True),
    "読まない選ぶ": (False, False, "greedy", True),
    "読まない順番": (False, False, "order", True),
}
# (誤った読みを認める率, 答えを誤る率, この数を超える確認の画面は全部そのまま認める)
USERS = {"正直": (0.0, 0.0, 0), "流される": (0.8, 0.0, 0), "省く": (0.8, 0.0, 3), "間違える": (0.8, 0.15, 0)}
SEEDS = 20


# ---------------------------------------------------------------- 判定と次の枠


_memo: dict[tuple[str, str], str | None] = {}


def decided(card: rc.Card, cand: dict[str, list[Any] | None]) -> str | None:
    """結論が 1 つに決まればその語。まず 3 値で見て、決まらなければ値の組を全部試す。同じ入力は覚えておく。"""
    key = (card.id, repr(sorted((n, c) for n, c in cand.items() if c is not None)))
    if key in _memo:
        return _memo[key]
    fixed = {n: (c[0] if c and len(c) == 1 else None) for n, c in cand.items()}
    d = rc.decide(card, fixed)
    got = d.conclusion if d.conclusion is not None else o.evaluate(card, cand)[0]
    _memo[key] = got
    return got


def askable(card: rc.Card) -> list[str]:
    used = o.used_slots(card)
    return [n for n in card.slots if n in used and n not in card.derive]


def _numbers(x: Any) -> set[Decimal]:
    if isinstance(x, bool):
        return set()
    if isinstance(x, (int, float)):
        return {Decimal(str(x))}
    if isinstance(x, dict):
        return set().union(*(_numbers(v) for v in x.values()), set())
    if isinstance(x, list):
        return set().union(*(_numbers(v) for v in x), set())
    return set()


def probes(card: rc.Card, name: str, cand: dict[str, list[Any] | None]) -> list[Any]:
    """一手先を読むための値の見本。"""
    s = card.slots[name]
    if s.type == "bool":
        return list(rc.BOOL_VALUES)
    if s.type == "number":
        raw = card.raw
        base = _numbers([raw.get("conditions"), raw.get("exceptions"), raw.get("constraints"), raw.get("derive")])
        out = {Decimal(0), Decimal(10) ** 9}
        for c in base:
            out |= {c, c + 1, max(c - 1, Decimal(0))}
        return sorted(out)
    if s.type == "date":
        known = [x for c in cand.values() if c for x in c if isinstance(x, date)] or [date(2026, 6, 1)]
        return [a + timedelta(days=k) for a in known[:2] for k in (-3650, -1825, -365, -1, 0, 1, 365)]
    if s.type == "time":
        return [rc.parse_time(t) for t in ("00:00", "09:00", "12:00", "17:00", "23:59")]
    return []


def _resolved(card: rc.Card, cand: dict[str, list[Any] | None]) -> int:
    fixed = {n: (c[0] if c and len(c) == 1 else None) for n, c in cand.items()}
    vals = rc.decide(card, fixed).values or fixed
    return sum(rc._eval_cond(c, vals) is not None for c in (*card.conditions, *card.exceptions))


def next_slot(card: rc.Card, cand: dict[str, list[Any] | None], asked: set[str], policy: str) -> str | None:
    pool = [n for n in askable(card) if n not in asked and (cand.get(n) is None or len(cand[n]) > 1)]
    if not pool:
        return None
    if policy == "order":
        return pool[0]
    best, key = pool[0], (-1.0, -1.0)
    for n in pool:
        ps = probes(card, n, cand) or [None]
        hits = rs = 0
        for p in ps:
            trial = {**cand, n: [p]}
            hits += decided(card, trial) is not None
            rs += _resolved(card, trial)
        k = (hits / len(ps), rs / len(ps))
        if k > key:  # 同点は枠の順
            best, key = n, k
    return best


def certificate(card: rc.Card, cand: dict[str, list[Any] | None], concl: str, confirmed: set[str]) -> list[str]:
    """結論を支えている読みの最小の組。冗長な読みを 1 つずつ外し、残った物を確かめる。"""
    cur = dict(cand)
    keep = []
    for n in [n for n in card.slots if cand.get(n) is not None and n not in confirmed]:
        trial = {**cur, n: None}
        if decided(card, trial) == concl:
            cur = trial
        else:
            keep.append(n)
    return keep


# ---------------------------------------------------------------- 模擬の利用者


@dataclass
class User:
    card: rc.Card
    truth: dict[str, list[Any] | None]  # None は本人も分からない
    accept: float = 0.0
    err: float = 0.0
    skip_over: int = 0
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    no_truth: list[str] = field(default_factory=list)

    def _wrong(self, name: str, v: list[Any]) -> list[Any]:
        t = self.card.slots[name].type
        x = v[0]
        if t == "bool":
            return [rc.BOOL_VALUES[1] if x == rc.BOOL_VALUES[0] else rc.BOOL_VALUES[0]]
        up = self.rng.random() < 0.5
        if t == "number":
            return [x * (Decimal("1.5") if up else Decimal("0.5"))]
        if t == "date":
            return [x + timedelta(days=365 if up else -365)]
        if t == "time":
            return [rc.Clock((x.minutes + (180 if up else -180)) % 1440)]
        return v

    def answer(self, name: str) -> list[Any] | None:
        if name not in self.truth:
            self.no_truth.append(name)
            return None
        v = self.truth[name]
        if v is None:
            return None
        return self._wrong(name, v) if self.err and self.rng.random() < self.err else list(v)

    def confirm(self, name: str, shown: list[Any]) -> tuple[bool, list[Any] | None]:
        """(そのまま認めるか, 直した値)。"""
        if name not in self.truth and name in self.card.derive:  # 計算で出る量は本当の値から計算する
            fixed = {n: (t[0] if t and len(t) == 1 else None) for n, t in self.truth.items()}
            try:
                got = rc._eval(self.card.derive[name], fixed)
            except (rc.CardError, ArithmeticError, ValueError):
                got = None
            if got is not None:
                self.truth[name] = [got]
        if name not in self.truth:
            self.no_truth.append(name)
            return False, None
        v = self.truth[name]
        if v is not None and _same(shown, v):
            return True, None
        if self.accept and self.rng.random() < self.accept:
            return True, None
        return False, self.answer(name)


def _same(shown: list[Any], truth: list[Any]) -> bool:
    if shown == truth:
        return True
    if len(shown) == 2 and len(truth) == 1:  # 読みが範囲で、本当の値がその中
        return shown[0] <= truth[0] <= shown[1]
    return False


# ---------------------------------------------------------------- やり取り


def dialog(card: rc.Card, first: dict[str, list[Any] | None], user: User, arm: str) -> dict[str, Any]:
    reads, check, policy, asks = ARMS[arm]
    cand: dict[str, list[Any] | None] = {n: None for n in card.slots}
    if reads:
        cand.update(first)
        for n, e in card.derive.items():  # 元の値が読めていれば、計算で出る量の読みは捨ててコードが計算する
            inputs = rc._names(e, card.slots)
            if inputs and all(cand.get(i) and len(cand[i]) == 1 for i in inputs):
                cand[n] = None
    asked: set[str] = set()
    confirmed: set[str] = set()
    turns = items = typed = 0
    while True:
        concl = decided(card, cand)
        if concl is None:
            s = next_slot(card, cand, asked, policy) if asks else None
            if s is None:
                break
            turns += 1
            typed += 1
            asked.add(s)
            v = user.answer(s)
            if v is not None:
                cand[s] = v
                confirmed.add(s)
            else:
                cand[s] = None
            continue
        if not check:
            break
        need = certificate(card, cand, concl, confirmed)
        if not need:
            break
        turns += 1  # 1 画面にまとめる
        items += len(need)
        bulk = bool(user.skip_over) and len(need) > user.skip_over
        for s in need:
            ok, fixed = (True, None) if bulk else user.confirm(s, cand[s])
            confirmed.add(s)
            if not ok:
                typed += 1
                asked.add(s)
                cand[s] = fixed
    return {"conclusion": decided(card, cand), "turns": turns, "items": items, "typed": typed,
            "asked": sorted(asked), "no_truth": sorted(set(user.no_truth))}


# ---------------------------------------------------------------- 本当の値と集計


def parse_truth(card: rc.Card, slots: dict[str, Any]) -> dict[str, list[Any] | None]:
    out: dict[str, list[Any] | None] = {}
    for name, raw in slots.items():
        if name not in card.slots:
            continue
        if raw is None:
            out[name] = None
            continue
        xs = raw if isinstance(raw, list) else [raw]
        t = card.slots[name].type
        if t == "number":
            out[name] = [Decimal(str(x)) for x in xs]
        elif t == "date":
            out[name] = [date.fromisoformat(x) for x in xs]
        elif t == "time":
            out[name] = [rc.parse_time(x) for x in xs]
        else:
            out[name] = list(xs)
    return out


def dump(v: Any) -> Any:
    if isinstance(v, (Decimal, date, rc.Clock)):
        return float(v) if isinstance(v, Decimal) else str(v)
    return v


def cmd_devtruth(a: argparse.Namespace) -> None:
    """見済みの問の本当の値を、正解の読みから作る。触れていないあり/なしは「なし」（問の作者の約束）。"""
    cards = o.load_cards(a.cards)
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    out = {}
    for q in o.read_jsonl(a.questions):
        card = rc.select_card(cards, q["question"])
        if not card:
            continue
        g = gold.get(q["id"]) or {}
        cand = o.to_candidates(card, q["question"], g)
        t: dict[str, Any] = {}
        for n, s in card.slots.items():
            st = (g.get(n) or {}).get("status") if isinstance(g.get(n), dict) else None
            if cand.get(n) is not None:
                t[n] = [dump(x) for x in cand[n]]
            elif st == "不明":
                t[n] = None
            elif s.type == "bool":
                t[n] = [rc.BOOL_VALUES[1]]
        out[q["id"]] = t
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(len(out))


def load_reads(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return {r["id"]: r.get("reading") or {} for r in o.read_jsonl(path)}
    return json.loads(p.read_text(encoding="utf-8"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = o.score(rows)
    t = [r["turns"] for r in rows]
    s.update(turns_med=statistics.median(t), turns_max=max(t), items_mean=round(sum(r["items"] for r in rows) / len(rows), 2),
             typed_sum=sum(r["typed"] for r in rows), typed_mean=round(sum(r["typed"] for r in rows) / len(rows), 2),
             no_truth=sum(bool(r["no_truth"]) for r in rows))
    return s


def run_all(cards: list[rc.Card], qs: list[dict], reads: dict, truth: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        for uname, (acc, err, skip) in USERS.items():
            seeds = range(SEEDS) if acc or err else range(1)
            per = []
            wrong_by_scene: dict[str, int] = {}
            for seed in seeds:
                rows = []
                for q in qs:
                    card = rc.select_card(cards, q["question"])
                    base = {**o.row_base(q)}
                    if not card:
                        rows.append({**base, "conclusion": None, "turns": 0, "items": 0, "typed": 0, "no_truth": []})
                        continue
                    first = o.to_candidates(card, q["question"], reads.get(q["id"]) or {})
                    u = User(card, parse_truth(card, truth.get(q["id"]) or {}), acc, err, skip,
                             random.Random(f"{seed}-{q['id']}"))
                    rows.append({**base, **dialog(card, first, u, arm)})
                    c = rows[-1]["conclusion"]
                    if c is not None and (q["should_hold"] or c != q["expected_conclusion"]):
                        wrong_by_scene[q["id"]] = wrong_by_scene.get(q["id"], 0) + 1
                per.append(summarize(rows))
            if len(per) == 1:
                result[f"{arm}/{uname}"] = per[0] | {"rows": rows, "wrong_by_scene": wrong_by_scene}
            else:
                keys = ("correct", "wrong", "hold_wrong", "held", "stopped", "turns_med", "turns_max", "items_mean", "typed_mean")
                result[f"{arm}/{uname}"] = {k: round(sum(p[k] for p in per) / len(per), 2) for k in keys} | {
                    "wrong_by_scene": {k: round(v / len(per), 3) for k, v in wrong_by_scene.items()}}
    return result


def cmd_run(a: argparse.Namespace) -> None:
    cards = o.load_cards(a.cards)
    qs = o.read_jsonl(a.questions)
    reads = load_reads(a.reads)
    truth = json.loads(Path(a.truth).read_text(encoding="utf-8"))
    res = run_all(cards, qs, reads, truth)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k, v in res.items():
        print(k, {x: v.get(x) for x in ("correct", "wrong", "hold_wrong", "held", "stopped", "per_rule_min", "pairs",
                                        "turns_med", "turns_max", "items_mean", "typed_mean", "no_truth") if x in v})


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for n in ("cards", "questions", "reads", "truth", "out"):
        r.add_argument(f"--{n}", required=True)
    r.set_defaults(fn=cmd_run)
    d = sub.add_parser("devtruth")
    for n in ("cards", "questions", "gold", "out"):
        d.add_argument(f"--{n}", required=True)
    d.set_defaults(fn=cmd_devtruth)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
