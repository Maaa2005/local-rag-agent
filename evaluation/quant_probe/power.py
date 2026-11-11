"""判定の線 qp-judge-3 の検出力と誤りの率を、測る前に見積もる。judge_model を端から端まで通す。結果は計画 6 章に写す。

    python -m evaluation.quant_probe.power --sims 2000

作り（問の並び・規程・保留すべき問は本物の 360 問。答えは作り物）:
- 基準: 規程ごとに誤った断定（WA）の率 p_b ~ N(0.10, 0.06)（0〜0.5 に切る）。答えられる問は WA p_b・保留しすぎ 0.03・
  形式違反 0.005・残り正答。保留すべき問は WA p_b・形式違反 0.005・残り保留できた。
- 揺れの組（基準の 2 回目・Q4_K_M-en の 2 回目・Q6_K-none の載せ方違い）: 元から各問を 1% の確率で同じ分布から引き直す。
- 変種: 基準 1 回目から、bit ごとの churn（Q6 .01・Q5 .02・Q4 .04・Q3 .08・Q2 .15）の確率で各問を同じ分布から
  引き直す（平均では増えも減りもしない揺れ）。さらに WA でない問から Δ 問を WA に変える（24 規程か 8 規程に散らす）。
- ずれ（全部の物差しで同じ値）: bit ごと Q6 .004・Q5 .010・Q4 .030・Q3 .090・Q2 .35 に校正の倍率 none 1.25・en 1.0・ja 1.08。
  安全域（<= Q4_K_M-en）は Q6 と Q5 の 6 つと Q4_K_M-en。10% 以内の近い対は同じ bit の en と ja。
場面:
- notice（気づくモデル）: Δ = Q6/Q5/Q4_K_M-en 0・Q4_K_M-ja/none 2・Q3 12・Q2 40。
- miss_A: notice に加え Q5_K_M-ja（安全域）に +d。
- miss_B: notice に加え Q3_K_M-ja に +d（Q3_K_M-en と KLD が 10% 以内）。
- null: Δ 全部 0。
- skew（偏った帰無）: notice に加え、Q5_K_M-ja（安全域）だけ規程ごとに 1/4 の確率で WA を 3 問足し、3/4 の確率で
  1 問消す（平均の増えはほぼ 0 だが、規程の差の分布が 0 の回りで対称でない。消す WA が無い規程では消さないので、
  平均はわずかに正で、誤りの率は高めに出る）。符号の入れ替えの前提が崩れた時の、誤った「気づかない」の率を見る。
3 モデルの全体の答えは、場面ごとの 1 モデルの判定の割合から、3 つが独立として数え上げる。
"""
from __future__ import annotations

import argparse
import itertools
import math
import random

from .judge import ANCHOR, METRICS, PLACE, PRIMARY, RERUN, judge_model, judge_overall
from .pipeline import ALL_VARIANTS
from .questions import load_questions

KLD_BIT = {"Q6_K": 0.004, "Q5_K_M": 0.010, "Q4_K_M": 0.030, "Q3_K_M": 0.090, "Q2_K": 0.35}
CAL = {"none": 1.25, "en": 1.0, "ja": 1.08}
CHURN = {"Q6_K": 0.01, "Q5_K_M": 0.02, "Q4_K_M": 0.04, "Q3_K_M": 0.08, "Q2_K": 0.15}
NOTICE = {"Q6_K": 0, "Q5_K_M": 0, "Q4_K_M": 2, "Q3_K_M": 12, "Q2_K": 40}
RERUN_NOISE = 0.01
VERDICTS = ("気づかない", "気づく", "決められない", "差が出ない", "測れない")
OVERALL = ("はい", "モデルによる", "いいえ（この 360 問で）", "測れず")


def _split(name: str) -> tuple[str, str]:
    q, c = name.rsplit("-", 1)
    return q, c


def deltas(scene: str, d: int) -> dict[str, int]:
    out = {}
    for n in ALL_VARIANTS:
        q, _ = _split(n)
        out[n] = 0 if scene == "null" or n == ANCHOR else NOTICE[q]
    if scene == "miss_A":
        out["Q5_K_M-ja"] += d
    elif scene == "miss_B":
        out["Q3_K_M-ja"] += d
    return out


def skew(w: "World", o: dict[str, str]) -> dict[str, str]:
    o = dict(o)
    for b in w.order:
        ids = [i for i in w.ids if w.block[i] == b]
        if w.rng.random() < 0.25:
            for i in w.rng.sample([i for i in ids if o[i] != "wrong_assert"], 3):
                o[i] = "wrong_assert"
        else:
            on = [i for i in ids if o[i] == "wrong_assert"]
            if on:
                i = w.rng.choice(on)
                o[i] = "held_ok" if w.hold[i] else "correct"
    return o


def _metrics(name: str) -> dict:
    q, c = _split(name)
    x = KLD_BIT[q] * CAL[c]
    return {s: {"kld_mean": x, "ln_ppl_ratio": x} for s in METRICS}


class World:
    def __init__(self, rng: random.Random):
        self.rng = rng
        qs = load_questions()
        self.ids = [q.id for q in qs]
        self.hold = {q.id: q.should_hold for q in qs}
        self.block = {q.id: f"{q.set}/{q.rule_id}" for q in qs}
        names = sorted(set(self.block.values()))
        self.order = {b: i for i, b in enumerate(names)}

    def draw(self, i: str, p: float) -> str:
        r = self.rng.random()
        if r < 0.005:
            return "format"
        if r < 0.005 + p:
            return "wrong_assert"
        if self.hold[i]:
            return "held_ok"
        return "over_hold" if r < 0.035 + p else "correct"

    def ref(self) -> tuple[dict[str, float], dict[str, str]]:
        p = {b: min(0.5, max(0.0, self.rng.gauss(0.10, 0.06))) for b in self.order}
        rate = {i: p[self.block[i]] for i in self.ids}
        return rate, {i: self.draw(i, rate[i]) for i in self.ids}

    def redraw(self, base: dict[str, str], rate: dict[str, float], frac: float) -> dict[str, str]:
        return {i: (self.draw(i, rate[i]) if self.rng.random() < frac else o) for i, o in base.items()}

    def add_wa(self, o: dict[str, str], delta: int, spread: int) -> dict[str, str]:
        if delta <= 0:
            return o
        o = dict(o)
        blocks = set(self.rng.sample(sorted(self.order), spread))
        slots = [i for i in self.ids if self.block[i] in blocks and o[i] != "wrong_assert"]
        for i in self.rng.sample(slots, min(delta, len(slots))):
            o[i] = "wrong_assert"
        return o


def one_model(w: World, scene: str, d: int, spread: int) -> str:
    rate, r1 = w.ref()
    r2 = w.redraw(r1, rate, RERUN_NOISE)
    dl = deltas(scene, d)
    variants, outs = [], {}
    for n in ALL_VARIANTS:
        q, _ = _split(n)
        o = w.add_wa(w.redraw(r1, rate, CHURN[q]), dl[n], spread if n in ("Q5_K_M-ja", "Q3_K_M-ja") else 24)
        if scene == "skew" and n == "Q5_K_M-ja":
            o = skew(w, o)
        outs[n] = o
        variants.append({"name": n, "outcomes": o, "metrics": _metrics(n)})
    noise = [(RERUN, outs[ANCHOR], w.redraw(outs[ANCHOR], rate, RERUN_NOISE)),
             (PLACE, outs["Q6_K-none"], w.redraw(outs["Q6_K-none"], rate, RERUN_NOISE))]
    j = judge_model(r1, r2, variants, w.hold, w.block, noise, strict=True, need_place=True, boot_n=0)
    return j["by_metric"][PRIMARY]["verdict"] if j["valid"] else "測れない"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    den = 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n) - c) / den), min(1.0, (p + z * z / (2 * n) + c) / den)


def rates(scene: str, d: int, spread: int, sims: int, seed: int) -> dict[str, int]:
    w = World(random.Random(f"{seed}-{scene}-{d}-{spread}"))
    cnt = {v: 0 for v in VERDICTS}
    for _ in range(sims):
        cnt[one_model(w, scene, d, spread)] += 1
    return cnt


def combine(probs: list[dict[str, float]]) -> dict[str, float]:
    """3 モデルそれぞれの判定の割合から、全体の答えの割合（独立とみなす）。"""
    out = {o: 0.0 for o in OVERALL}
    for combo in itertools.product(VERDICTS, repeat=len(probs)):
        pr = math.prod(p[v] for p, v in zip(probs, combo))
        if pr == 0:
            continue
        res = {f"m{i}": {"valid": v != "測れない", "by_metric": {PRIMARY: {"verdict": v}}} for i, v in enumerate(combo)}
        out[judge_overall(res, PRIMARY)] += pr
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args(argv)
    scenes = [("notice", 0, 24), ("null", 0, 24), ("skew", 0, 24)]
    scenes += [(s, d, sp) for s in ("miss_A", "miss_B") for sp in (24, 8) for d in (10, 20, 30)]
    table = {}
    print(f"1 モデルの判定の割合（主の物差し {PRIMARY}・{a.sims} 回・括弧は Wilson 95%）")
    print("| 場面 | 増え d | 広さ | " + " | ".join(VERDICTS) + " |")
    print("|---|---|---|" + "---|" * len(VERDICTS))
    for s, d, sp in scenes:
        cnt = rates(s, d, sp, a.sims, a.seed)
        table[(s, d, sp)] = {v: c / a.sims for v, c in cnt.items()}
        cells = []
        for v in VERDICTS:
            lo, hi = wilson(cnt[v], a.sims)
            cells.append(f"{cnt[v] / a.sims:.3f} ({lo:.3f}-{hi:.3f})")
        print(f"| {s} | {d or '-'} | {sp if s.startswith('miss') else '-'} | " + " | ".join(cells) + " |", flush=True)
    print("\n3 モデルの全体の答えの割合（本当に見逃すモデルの数 k。見逃すモデルは miss_A か miss_B・24 規程）")
    print("| 見逃しの型 | d | k | " + " | ".join(OVERALL) + " |")
    print("|---|---|---|" + "---|" * len(OVERALL))
    for s in ("miss_A", "miss_B"):
        for d in (10, 20, 30):
            for k in range(4):
                probs = [table[(s, d, 24)]] * k + [table[("notice", 0, 24)]] * (3 - k)
                o = combine(probs)
                print(f"| {s} | {d} | {k} | " + " | ".join(f"{o[x]:.3f}" for x in OVERALL) + " |")
    o = combine([table[("null", 0, 24)]] * 3)
    print("| null（3 つとも増えなし） | - | 0 | " + " | ".join(f"{o[x]:.3f}" for x in OVERALL) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
