"""線 4 の誤りの出どころ。探索なので線の判定には使わない。"""
import collections
import json
import os
import random
import statistics

from evaluation.gpu_local import ask_back as ab

rc, o = ab.rc, ab.o
L = os.path.expanduser("~/.cache/lr-gpu/ask90")
qs = o.read_jsonl("evaluation/v2/ask90/questions.jsonl")
kind = {q["id"]: q["kind"] for q in qs}
cards = o.load_cards(os.path.join(L, "cards"))
truth = json.load(open(os.path.join(L, "truth.json"), encoding="utf-8"))


def load(n):
    return json.load(open(os.path.join(L, n), encoding="utf-8"))


res, gold = load("res_r1.json"), load("upper_v2.json")


def bykind(d):
    c = collections.Counter()
    for k, v in d.items():
        c[kind[k]] += v
    return {k: round(v, 2) for k, v in sorted(c.items())}


for name, r in (("実測", res), ("正解の読み", gold)):
    for arm in ("基準/流される", "本命/流される", "本命/省く", "本命/間違える", "読まない選ぶ/間違える"):
        w = r[arm]["wrong_by_scene"]
        print(name, arm, round(sum(w.values()), 2), bykind(w))

top = sorted(res["本命/流される"]["wrong_by_scene"].items(), key=lambda x: -x[1])[:12]
print("多い場面", [(k, kind[k], v) for k, v in top])

# 読みのうち「触れない」から「なし」と決めた枠に由来する誤りか: 触れない枠を捨てて、確かめずに聞く
reads = {r["id"]: r.get("reading") or {} for r in o.read_jsonl(os.path.join(L, "run_r1", "raw_A1.jsonl"))}


def run(drop_untouched, user):
    acc, err, skip = ab.USERS[user]
    out = []
    for seed in range(ab.SEEDS if acc or err else 1):
        rows = []
        for q in qs:
            card = rc.select_card(cards, q["question"])
            first = o.to_candidates(card, q["question"], reads.get(q["id"]) or {})
            if drop_untouched:
                for n in card.slots:
                    it = reads[q["id"]].get(n)
                    if isinstance(it, dict) and it.get("status") == "触れない":
                        first[n] = None
            u = ab.User(card, ab.parse_truth(card, truth[q["id"]]), acc, err, skip, random.Random(f"{seed}-{q['id']}"))
            rows.append({**o.row_base(q), **ab.dialog(card, first, u, "本命")})
        out.append(ab.summarize(rows))
    return {k: round(statistics.mean(s[k] for s in out), 2) for k in ("correct", "wrong", "hold_wrong", "held", "stopped", "turns_med", "turns_max", "items_mean", "typed_mean")}


for d in (False, True):
    for u in ("正直", "流される", "間違える"):
        print("触れない枠を聞く" if d else "今の本命", u, run(d, u))
