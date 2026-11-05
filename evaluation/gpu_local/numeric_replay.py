"""保存した読み (raw.jsonl) を、推論せずに検査の条件を変えて判定し直す (Issue #16)。

厳密/同じ量の別の言い方 × 全枠必須/問題の枠は不明、の 4 通りの数を比べる。2 回読みの確認は入らない (1 回目の読みだけ)。

    python -m evaluation.gpu_local.numeric_replay --raw <raw.jsonl> --questions <questions.jsonl> --cards <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services import rule_cards as rc

ARMS = {"厳密・全枠": ("strict", False), "厳密・不明": ("strict", True),
        "同量・全枠": ("equiv", False), "同量・不明": ("equiv", True)}


def load_cards(directory: str) -> dict:
    cards = {}
    for p in Path(directory).glob("*.json"):
        if p.name != "unable.json":
            c = rc.load_card(p)
            cards[c.id] = c
    return cards


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def conclude(card, question: str, raw, verify: str, unknown: bool) -> str | None:
    values, problems = rc.validate_extraction(card, question, raw, verify=verify, problem_as_unknown=unknown)
    return None if problems else rc.decide(card, values).conclusion


def replay(rows: list[dict], qs: dict, cards: dict, verify: str, unknown: bool) -> dict:
    n = {"answerable": 0, "correct": 0, "wrong": 0, "hold_n": 0, "held": 0, "hold_wrong": 0}
    ok: dict[str, list[bool]] = {}
    changed = []
    for r in rows:
        q, card = qs[r["id"]], cards.get(r.get("card_id"))
        c = conclude(card, q["question"], r["raw"], verify, unknown) if card and r.get("raw") is not None else None
        if q["should_hold"]:
            n["hold_n"] += 1
            n["held"] += c is None
            n["hold_wrong"] += c is not None
        else:
            n["answerable"] += 1
            n["correct"] += c == q["expected_conclusion"]
            n["wrong"] += c is not None and c != q["expected_conclusion"]
        if q.get("pair_id"):
            ok.setdefault(q["pair_id"], []).append(c == q["expected_conclusion"])
        changed.append((r["id"], c))
    n["pairs"] = sum(all(v) and len(v) >= 2 for v in ok.values())
    n["pair_n"] = len(ok)
    n["decisions"] = dict(changed)
    return n


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--raw", required=True)
    a.add_argument("--questions", required=True)
    a.add_argument("--cards", required=True)
    a.add_argument("--run", type=int, default=1)
    args = a.parse_args()
    rows = [r for r in read_jsonl(args.raw) if r.get("run", 1) == args.run]
    qs = {q["id"]: q for q in read_jsonl(args.questions)}
    cards = load_cards(args.cards)
    print("| 条件 | 正しく決めた | 誤り(答えられる問) | 誤り(保留すべき問) | 保留 | 対 |\n|---|---|---|---|---|---|")
    for name, (verify, unknown) in ARMS.items():
        n = replay(rows, qs, cards, verify, unknown)
        print(f"| {name} | {n['correct']}/{n['answerable']} | {n['wrong']} | {n['hold_wrong']} | "
              f"{n['held']}/{n['hold_n']} | {n['pairs']}/{n['pair_n']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
