"""判定の線（qp-judge-2）の検出力を、測る前に見積もる。結果は計画の「検出力」に写す。

    python -m evaluation.quant_probe.power --sims 400

作り: 24 規程 × 15 問。基準の誤った断定（WA）の率は規程ごとに違う（平均 base・規程の間のばらつき het）。
変種は基準から churn の割合の問がでたらめに入れ替わり（増えも減りもしない揺れ）、さらに Δ 問が新しく WA になる。
増えは 24・8・4 規程のどれかに散る（数が少ないほど集まる）。
- 悪化を見つける率: 片側の入れ替え検定の p < ALPHA_A/15（Holm の最初の段。ほかの変種が悪くない最悪の時）かつ Δ >= 3。
  揺れの判定（Δ > 2×揺れ）は 0 と置く（決まった出し方で 2 回とも同じ答えが出る前提）。
- 同じと言える率: Δ = 0 の時、95% 上側 < EQUIV_MARGIN。
"""
from __future__ import annotations

import argparse
import random

from .judge import ALPHA_A, EQUIV_MARGIN, MIN_DELTA, N_VARIANTS
from .stats import block_bootstrap_upper, signflip_exact

N_BLOCKS, PER_BLOCK = 24, 15


def _ref(rng: random.Random, base: float, het: float) -> list[list[int]]:
    out = []
    for _ in range(N_BLOCKS):
        p = min(0.9, max(0.0, rng.gauss(base, het)))
        out.append([int(rng.random() < p) for _ in range(PER_BLOCK)])
    return out


def _variant(rng: random.Random, ref: list[list[int]], delta: int, spread: int, churn: float) -> list[list[int]]:
    v = [row[:] for row in ref]
    # 揺れ: 全体で m = churn×360 問が入れ替わる。半分は WA が消え、半分は WA が出る（合計は変わらない）。
    # どの規程で起きるかはでたらめなので、規程ごとの差は 0 にならない。
    on = [(b, i) for b, row in enumerate(v) for i, x in enumerate(row) if x]
    off = [(b, i) for b, row in enumerate(v) for i, x in enumerate(row) if not x]
    k = min(len(on), round(churn * N_BLOCKS * PER_BLOCK / 2))
    for b, i in rng.sample(on, k):
        v[b][i] = 0
    for b, i in rng.sample(off, k):
        v[b][i] = 1
    targets = rng.sample(range(N_BLOCKS), spread)
    slots = [(b, i) for b in targets for i, x in enumerate(v[b]) if not x]
    for b, i in rng.sample(slots, min(delta, len(slots))):
        v[b][i] = 1
    return v


def _diffs(v: list[list[int]], ref: list[list[int]]) -> list[int]:
    return [sum(a) - sum(b) for a, b in zip(v, ref)]


def detect_rate(delta: int, spread: int, churn: float, sims: int, seed: int,
                base: float = 0.10, het: float = 0.06) -> float:
    rng = random.Random(seed)
    hit = 0
    for _ in range(sims):
        ref = _ref(rng, base, het)
        d = _diffs(_variant(rng, ref, delta, spread, churn), ref)
        hit += sum(d) >= MIN_DELTA and signflip_exact(d, "greater") < ALPHA_A / N_VARIANTS
    return hit / sims


def same_rate(churn: float, sims: int, seed: int, boot_n: int = 2000, margin: int = EQUIV_MARGIN,
              base: float = 0.10, het: float = 0.06) -> float:
    rng = random.Random(seed)
    hit = 0
    for k in range(sims):
        ref = _ref(rng, base, het)
        d = _diffs(_variant(rng, ref, 0, N_BLOCKS, churn), ref)
        hit += block_bootstrap_upper(d, 0.95, boot_n, seed + k) < margin
    return hit / sims


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args(argv)
    print(f"悪化を見つける率（片側 p < {ALPHA_A:.4f}/{N_VARIANTS}・Δ>={MIN_DELTA}・{a.sims} 回）")
    print("| 揺れ | 広さ | " + " | ".join(f"Δ={d}" for d in (5, 10, 15, 20, 30)) + " |")
    print("|---|---|" + "---|" * 5)
    for churn in (0.02, 0.06, 0.10):
        print(f"| {churn:.2f} | 増えなし（誤って悪化と言う率） | {detect_rate(0, N_BLOCKS, churn, a.sims * 5, a.seed):.4f} |||||")
        for spread in (24, 8, 4):
            rates = [detect_rate(d, spread, churn, a.sims, a.seed) for d in (5, 10, 15, 20, 30)]
            print(f"| {churn:.2f} | {spread} 規程 | "
                  + " | ".join(f"{r:.2f}" for r in rates) + " |")
    print(f"\n増えていない時に「同じ」（95% 上側 < 幅）と言える率（ブートストラップは 2000 回で近似・{a.sims // 2} 回）")
    print("| 揺れ（入れ替わる問の割合） | 幅 10 | 幅 15 |")
    print("|---|---|---|")
    for churn in (0.02, 0.06, 0.10, 0.15):
        r10 = same_rate(churn, a.sims // 2, a.seed)
        r15 = same_rate(churn, a.sims // 2, a.seed, margin=15)
        print(f"| {churn:.2f} | {r10:.2f} | {r15:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
