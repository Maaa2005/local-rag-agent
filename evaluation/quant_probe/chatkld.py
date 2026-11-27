"""会話の形の KLD（kld_chat・探索）。計画 版 8 の 5 章。

run_answers.py の確かめの依頼で取った、「結論:」の直後の次の 1 語の確率の上位 100（行の "probe": [{id, p}, ...]）から、
問 i ごとに基準 1 回目の分布 P と変種の分布 Q の KLD(P‖Q) を取り、使えた問で平均する。
- 問 i の共通の語 S_i = 基準 1 回目と、そのモデルの変種（15 変種全部）の上位 100 の全部に出る語の番号の集合。
  どの変種も同じ S_i を使う（変種ごとに区分が変わらない）。
- S_i の語はそのまま。残りは「その他」の 1 つにまとめる（確率 = 1 − Σ S_i の確率。下限 1e-12）。P・Q とも和が 1 の分布になり、
  まとめた分布の KLD は元の KLD 以下で負にならない（計算の誤差で負になった分は 0 に直す）。確率の下限は全部 1e-12。
- 使わない問: 基準 1 回目の probe が無い（「結論:」が無かった・確かめが無い）問。
- 変種に probe が無い問が 1 つでもあれば、その変種は値なし（kld_mean None）。
- 使えた問 n が MIN_USABLE（342・95%）未満なら、そのモデルの kld_chat は全変種で値なし。
- 返す物（変種ごと）: kld_mean・same_top（一番の候補が同じだった問の割合 0〜1）・n（使えた問の数）・
  common_mean（S_i の大きさの平均）・other_p_mean（基準の「その他」の確率の平均）。値なしの時は reason も付く。
KLD は自然対数（nat）。共通の語が無い問は「その他」だけの分布で 0 になるが、分布が同じという意味ではない
（common_mean を並べて読む）。
"""
from __future__ import annotations

import math
from typing import Iterable

FLOOR = 1e-12
MIN_USABLE = 342  # 360 問の 95%


def _usable(row: dict | None) -> list[dict] | None:
    top = (row or {}).get("probe")
    return top if isinstance(top, list) and top else None


def _probs(probe: list[dict]) -> dict[int, float]:
    """probe → {語の番号: 確率}。同じ語が重なっていれば最初の物を使う。形が違えば ValueError。"""
    out: dict[int, float] = {}
    for t in probe:
        if not isinstance(t, dict) or type(t.get("id")) is not int or "p" not in t:
            raise ValueError(f"probe の要素の形が違う: {t!r}")
        p = t["p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0:
            raise ValueError(f"probe の確率が数でない: {p!r}")
        out.setdefault(t["id"], float(p))
    return out


def question_kld(p: dict[int, float], q: dict[int, float], common: Iterable[int]) -> float:
    """共通の語 common はそのまま・残りを「その他」にまとめた分布の KLD(P‖Q)。p・q は {語の番号: 確率}。"""
    common = list(common)
    kld = 0.0
    for i in common:
        pi, qi = max(p[i], FLOOR), max(q[i], FLOOR)
        kld += pi * math.log(pi / qi)
    p_o = max(1.0 - sum(p[i] for i in common), FLOOR)
    q_o = max(1.0 - sum(q[i] for i in common), FLOOR)
    kld += p_o * math.log(p_o / q_o)
    return max(kld, 0.0)


def _top(d: dict[int, float]) -> int:
    return max(d, key=d.get)  # 同じ確率なら先に出た語


def chat_kld(ref_rows: dict[str, dict], variant_rows: dict[str, dict[str, dict]], ids: Iterable[str] | None = None,
             min_usable: int = MIN_USABLE) -> dict[str, dict]:
    """ref_rows: 基準 1 回目の答えの行（id → 行）。variant_rows: 変種名 → その答えの行（id → 行）。
    ids を渡すとその問だけ見る。返り値: 変種名 → {kld_mean, same_top, n, common_mean, other_p_mean[, reason]}。"""
    ids = sorted(ref_rows) if ids is None else list(ids)
    names = list(variant_rows)
    usable = [i for i in ids if _usable(ref_rows.get(i)) is not None]
    n = len(usable)

    def none(name: str, reason: str, n_here: int = n) -> dict:
        return {"kld_mean": None, "same_top": None, "n": n_here, "common_mean": None, "other_p_mean": None,
                "reason": reason}

    if n < min_usable:
        return {v: none(v, f"基準 1 回目の確かめが使えた問が {n}（{min_usable} 未満）") for v in names}
    ref_p = {i: _probs(ref_rows[i]["probe"]) for i in usable}
    var_p: dict[str, dict[str, dict[int, float]]] = {}
    lacking: dict[str, list[str]] = {}
    for v in names:
        var_p[v], lacking[v] = {}, []
        for i in usable:
            pr = _usable(variant_rows[v].get(i))
            if pr is None:
                lacking[v].append(i)
            else:
                var_p[v][i] = _probs(pr)
    common: dict[str, set[int]] = {}
    for i in usable:
        s = set(ref_p[i])
        for v in names:
            if i in var_p[v]:
                s &= set(var_p[v][i])
        common[i] = s
    common_mean = sum(len(s) for s in common.values()) / n
    other_mean = sum(max(1.0 - sum(ref_p[i][t] for t in common[i]), FLOOR) for i in usable) / n
    out = {}
    for v in names:
        if lacking[v]:
            out[v] = none(v, f"{v} の確かめが無い問が {len(lacking[v])}", n - len(lacking[v]))
            continue
        klds = [question_kld(ref_p[i], var_p[v][i], common[i]) for i in usable]
        same = sum(_top(ref_p[i]) == _top(var_p[v][i]) for i in usable)
        out[v] = {"kld_mean": sum(klds) / n, "same_top": same / n, "n": n,
                  "common_mean": common_mean, "other_p_mean": other_mean}
    return out
