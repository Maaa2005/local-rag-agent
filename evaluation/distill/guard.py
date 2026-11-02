"""守りの問 40 問（docs/verified-distill-plan.md §1・§5）。開発用の家族から、開発用と別の種で作って固定する。

- refusal 10: 資料に無い事を聞く → 断る（metrics.is_refusal）
- describe 10: 資料の文を答える → 断らない・正解の語句を含む
- two 10: 2 択の数の問 → 最後の結論が 2 つの選択肢のどちらか（正誤は記録）
- three 10: 3 択の数の問 → 最後の結論が選択肢のどれか（正誤は記録）

    python -m evaluation.distill.guard --out evaluation/distill/guard
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from pathlib import Path

from evaluation.distill.gen import make
from evaluation.metrics import final_conclusion, is_refusal

SEED = 4242
DEV_SEED = 20261102


def build() -> list[dict]:
    dev_keys = {r["messages"][1]["content"] for r in make.generate("dev", 300, DEV_SEED)}
    rng = random.Random(SEED)
    pool = make.pool("dev", SEED)
    want = {"refusal": 10, "describe": 10, "two": 10, "three": 10}
    rows, seen = [], set(dev_keys)
    for kind, n in want.items():
        cands = pool[kind] if kind in ("refusal", "describe") else pool["boundary"] + pool["conversion"]
        if kind == "two":   # 2 択にできるのは、段階の型でなく、結論が決まる問だけ
            cands = [c for c in cands if make.FAM[c.family]["type"] != "tier" and c.expected != make.UNDECIDED]
        got = 0
        while got < n:
            it = dataclasses.replace(rng.choice(cands), three=kind == "three")
            r = make.assemble(it, "dev", rng)
            key = r["messages"][1]["content"]
            if key in seen:
                continue
            seen.add(key)
            r["guard"] = kind
            r["id"] = f"G{len(rows) + 1:02d}"
            rows.append(r)
            got += 1
    return rows


def judge(row: dict, answer: str) -> bool:
    """守りの条件を満たしたか（正誤ではなく、断り・形の決まりどおりか）。"""
    if row["guard"] == "refusal":
        return is_refusal(answer)
    if row["guard"] == "describe":
        return not is_refusal(answer) and all(k in answer for k in row["keywords"])
    return final_conclusion(answer) in row["options"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build()).encode("utf-8")
    (a.out / "guard.jsonl").write_bytes(data)
    (a.out / "manifest.json").write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "n": 40}) + "\n")
    print(hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
