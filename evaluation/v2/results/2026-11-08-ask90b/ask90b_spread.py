"""読みのばらつき。合否用 f1 と比べて、枠の読みが変わった割合と、結論に効く候補が変わった割合を出す。引数: run の親の場所"""
import json
import os
import sys

from evaluation.gpu_local import open_read as o

rc = o.rc
base = sys.argv[1]
cards = o.load_cards(os.path.join(base, "cards"))


def load(name, qfile):
    qs = {q["id"]: q for q in o.read_jsonl(qfile)}
    rows = o.read_jsonl(os.path.join(base, f"run_{name}", "raw_A1.jsonl"))
    out = {}
    for r in rows:
        q = qs[r["id"]]
        card = rc.select_card(cards, q["question"])
        out[r["id"]] = {n: repr(v) for n, v in o.to_candidates(card, q["question"], r.get("reading") or {}).items()}
    return out


Q = "evaluation/v2/ask90b/questions.jsonl"
ref = load("f1", Q)
result = {}
for name in ("f2", "t11", "t12", "t13", "n1", "n2"):
    qf = os.path.join(base, f"questions_{name}.jsonl") if name.startswith("n") else Q
    cur = load(name, qf)
    slots = changed = scenes = 0
    for i, a in ref.items():
        diff = sum(a[n] != cur[i][n] for n in a)
        slots += len(a)
        changed += diff
        scenes += diff > 0
    result[name] = {"slots_changed": f"{changed}/{slots}", "scenes_changed": f"{scenes}/{len(ref)}"}
    print(name, result[name])
json.dump(result, open(os.path.join(base, "spread.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
