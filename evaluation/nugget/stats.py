"""対応ありの差・規程でまとめた SE・A/A の揺れ幅・「信じる差」の判定。標準ライブラリのみ。

式（設計 5 章）:
  d_i = s_B,i - s_A,i,  d̄ = mean(d)
  SE_CLT        = sqrt( Var(d, ddof=1) / n )
  SE_clustered² = SE_CLT² + (1/n²) Σ_c Σ_i Σ_{j≠i} (d_ic - d̄)(d_jc - d̄)   (Error Bars 論文 式 4)
  95% 区間      = d̄ ± 1.96·SE
  w             = A/A の区間の端の絶対値の大きい方
  信じる差      = B-A の区間が 0 を含まず、かつ |d̄| > w

決め事:
  n=0 は平均も出さない(None)。n=1 は平均だけ出し、SE・区間は None。
  全部同じ値なら SE_CLT=0（区間は d̄ の 1 点）。
  SE_clustered² は d̄ からの偏差を使うので理論上は負にならない。丸め誤差で負になった時だけ 0 に切り上げる。
  まとまりが指定されない時は CLT の SE を主に使う。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, Mapping, Sequence

Z95 = 1.96

VERDICT_BELIEVE = "信じる差"
VERDICT_NOISE = "差は揺れの中"
VERDICT_NO_AA = "A/Aなし(判定保留)"
VERDICT_TOO_FEW = "判定不能(n<2)"


def mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def var_ddof1(xs: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    m = mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (n - 1)


def se_clt(xs: Sequence[float]) -> float | None:
    """SE = sqrt(Var(ddof=1) / n)。n<2 は None。"""
    v = var_ddof1(xs)
    if v is None:
        return None
    return math.sqrt(v / len(xs))


def se_clustered(xs: Sequence[float], groups: Sequence[Hashable] | None) -> float | None:
    """まとまりを考えた SE（式 4）。n<2 は None。groups が無ければ None。"""
    if groups is None:
        return None
    if len(groups) != len(xs):
        raise ValueError("groups と値の数が違う")
    n = len(xs)
    base = se_clt(xs)
    if base is None:
        return None
    m = mean(xs)
    by_c: dict[Hashable, list[float]] = defaultdict(list)
    for x, g in zip(xs, groups):
        by_c[g].append(x - m)
    cross = 0.0
    for devs in by_c.values():
        # Σ_i Σ_{j≠i} a_i a_j = (Σa)² - Σa²
        cross += math.fsum(devs) ** 2 - math.fsum(a * a for a in devs)
    var = base * base + cross / (n * n)
    return math.sqrt(max(var, 0.0))


def ci95(m: float, se: float | None) -> tuple[float, float] | None:
    if se is None:
        return None
    return (m - Z95 * se, m + Z95 * se)


def paired_diff(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[list[str], list[float]]:
    """両方にある qid だけ（qid 順）で d = b - a。"""
    qids = sorted(set(a) & set(b))
    return qids, [b[q] - a[q] for q in qids]


def compare(
    a: Mapping[str, float],
    b: Mapping[str, float],
    clusters: Mapping[str, Hashable] | None = None,
) -> dict:
    """B-A の対応ありの比較。主の SE は、まとまりがあれば clustered、無ければ CLT。"""
    qids, d = paired_diff(a, b)
    n = len(d)
    out: dict = {
        "n": n,
        "qids": qids,
        "d": d,
        "dbar": mean(d) if n else None,
        "se_clt": None,
        "se_cluster": None,
        "ci_clt": None,
        "ci_cluster": None,
        "main": "cluster" if clusters is not None else "clt",
        "se": None,
        "ci": None,
    }
    if n < 2:
        return out
    out["se_clt"] = se_clt(d)
    out["ci_clt"] = ci95(out["dbar"], out["se_clt"])
    if clusters is not None:
        out["se_cluster"] = se_clustered(d, [clusters[q] for q in qids])
        out["ci_cluster"] = ci95(out["dbar"], out["se_cluster"])
    main = "se_cluster" if clusters is not None else "se_clt"
    out["se"] = out[main]
    out["ci"] = ci95(out["dbar"], out["se"])
    return out


def aa_width(aa: Mapping) -> float | None:
    """A/A の比較結果から揺れの幅 w（区間の端の絶対値の大きい方）。区間が無ければ None。"""
    ci = aa.get("ci")
    if ci is None:
        return None
    return max(abs(ci[0]), abs(ci[1]))


def verdict(cmp: Mapping, w: float | None) -> str:
    """信じる差 = 区間が 0 を含まず、かつ |d̄| > w。"""
    ci = cmp.get("ci")
    if cmp["n"] < 2 or ci is None:
        return VERDICT_TOO_FEW
    if w is None:
        return VERDICT_NO_AA
    contains_zero = ci[0] <= 0.0 <= ci[1]
    if not contains_zero and abs(cmp["dbar"]) > w:
        return VERDICT_BELIEVE
    return VERDICT_NOISE


def is_believed(cmp: Mapping, w: float | None) -> bool:
    return verdict(cmp, w) == VERDICT_BELIEVE
