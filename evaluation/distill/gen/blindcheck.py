"""生成の前の点検（docs/verified-distill-plan.md §2）。

生成器が作れる（型・比べる語）の組・問の書き方・規程の書き方（家族 × 3）・（型・区分）の組を、
それぞれ 1 回以上含む見本を貪欲に選ぶ（上限 300 問）。正解を伏せて確認役に解かせ、コードの正解と照合する。

    python -m evaluation.distill.gen.blindcheck --blind <dir>
    python -m evaluation.distill.gen.blindcheck --check <dir>/sample.jsonl <確認役の出力>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation.distill.gen.make import generate
from evaluation.metrics import is_refusal

CAP = 300
_LINE = re.compile(r"^\s*(S\d{3})\s*\|\s*(.+?)\s*$")


def keys(r: dict) -> set[tuple]:
    k = {("wrap", r["wrap"]), ("style", r["family"], r["style"]), ("cat", r["type"], r["category"])}
    if r["word"]:
        k.add(("tw", r["type"], r["word"]))
    return k


def coverage_sample(seed: int = 20261102) -> list[dict]:
    cands = generate("train", 6000, seed) + generate("dev", 1500, seed)
    need = set().union(*(keys(r) for r in cands))
    picked, covered = [], set()
    while covered != need and len(picked) < CAP:
        best = max(cands, key=lambda r: len(keys(r) - covered))
        gain = keys(best) - covered
        if not gain:
            break
        picked.append(best)
        covered |= gain
    for i, r in enumerate(picked, 1):
        r["sid"] = f"S{i:03d}"
    return picked


def write_blind(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = coverage_sample()
    (out / "sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    blind = [{"id": r["sid"], "system": r["messages"][0]["content"], "user": r["messages"][1]["content"]} for r in rows]
    (out / "blind.jsonl").write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in blind), encoding="utf-8")
    return {"n": len(rows), "keys": len(set().union(*(keys(r) for r in rows)))}


def compare(sample: list[dict], text: str) -> dict:
    got = {m.group(1): m.group(2) for m in map(_LINE.match, text.splitlines()) if m}
    diff = []
    for r in sample:
        a = got.get(r["sid"])
        if a is None:
            diff.append((r["sid"], r["expected"], None))
        elif r["category"] == "refusal":
            if not ("資料なし" in a or is_refusal(a)):
                diff.append((r["sid"], "資料なし", a))
        elif r["category"] == "describe":
            if not all(k in a for k in r["keywords"]):
                diff.append((r["sid"], r["keywords"], a))
        elif a.strip(" 「」*") != r["expected"]:
            diff.append((r["sid"], r["expected"], a))
    return {"n": len(sample), "answered": len(got), "diff": diff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", type=Path)
    ap.add_argument("--check", nargs=2, type=Path)
    a = ap.parse_args()
    if a.blind:
        print(write_blind(a.blind))
    if a.check:
        sample = [json.loads(l) for l in a.check[0].read_text(encoding="utf-8").splitlines() if l.strip()]
        print(json.dumps(compare(sample, a.check[1].read_text(encoding="utf-8")), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
