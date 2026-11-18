"""検定と順位の計算（標準ライブラリだけ）。

- 主の検定: 規程ごとのまとまり（1 規程 15 問）の差の符号を入れ替える正確検定。
  同じ規程の問は互いに独立でないため、問ごとの検定は甘くなる（Sol との協議 2026-11-11）。
  前提: 差が無いなら、各規程の差の符号は独立に入れ替えてよい。
- 参考: 問ごとの McNemar の正確検定（不一致の問だけで二項検定 p=0.5）。
  量子化の劣化の検出に片側の正確 McNemar を使う例: Kübler ら arXiv 2602.10144。
- 多重比較: Holm（Bonferroni より強く、家族の第一種の誤りを抑える）。
- 「増えていない」を言う時の上限: 規程のまとまりを引き直すブートストラップの 95% 上側（種は固定）。
"""
from __future__ import annotations

import random
from math import comb


def signflip_exact(diffs: list[int], alternative: str = "greater") -> float:
    """各まとまりの整数の差 d_r の符号を全部の組で入れ替えた時、合計が観測以上（両側は絶対値で以上）になる割合。
    2^k 通りを数え上げず、合計の分布を足し込みで求める（k = 差が 0 でないまとまりの数）。"""
    d = [int(x) for x in diffs if x != 0]
    obs = sum(d)
    dist = {0: 1}
    for x in d:
        nxt: dict[int, int] = {}
        for s, c in dist.items():
            nxt[s + x] = nxt.get(s + x, 0) + c
            nxt[s - x] = nxt.get(s - x, 0) + c
        dist = nxt
    total = 2 ** len(d)
    if alternative == "greater":
        hit = sum(c for s, c in dist.items() if s >= obs)
    elif alternative == "two-sided":
        hit = sum(c for s, c in dist.items() if abs(s) >= abs(obs))
    else:
        raise ValueError(alternative)
    return hit / total


def block_bootstrap_upper(diffs: list[int], q: float = 0.95, n: int = 10000, seed: int = 20261111) -> float:
    """まとまりを重複ありで引き直した合計の q 分位（増えた数の上側の限界）。"""
    rng = random.Random(seed)
    k = len(diffs)
    sums = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return float(sums[min(n - 1, int(q * n))])


def binom_tail_ge(k: int, n: int) -> float:
    """P(X >= k), X ~ Bin(n, 0.5)。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def mcnemar_one_sided(b: int, c: int) -> float:
    """b: 比べる側だけが悪い問の数、c: 基準の側だけが悪い問の数。H1: b > c。"""
    return binom_tail_ge(b, b + c)


def mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_tail_ge(max(b, c), n))


def discordant(x: dict[str, bool], y: dict[str, bool]) -> tuple[int, int]:
    """x だけ真の数、y だけ真の数（同じ問の集合で比べる）。"""
    if set(x) != set(y):
        raise ValueError("比べる 2 つの問の集合が違う")
    b = sum(x[i] and not y[i] for i in x)
    c = sum(y[i] and not x[i] for i in x)
    return b, c


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm で直した p（単調にそろえる）。"""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5
