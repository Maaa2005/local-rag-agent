"""測る前に固定した判定の線 qp-judge-2（docs/quant-probe-plan.md の 5 章と同じ。変えるなら測る前に版を上げる）。

まとまり = 規程（24 規程・1 規程 15 問）。誤った断定（WA）の差は規程ごとに足し、規程を単位に検定する。

0. 基準が測れる: 基準 Q8_0 の 2 回とも、答えられる問の正答 >= 70%・保留すべき問の保留 >= 50%・形式違反 <= 5%。
1. 揺れ: 同じ物をもう 1 回流した組（基準の 2 回・Q4_K_M-en の 2 回・置き方だけ変えた Q6_K-none）で、
   WA が食い違った問の数の最大。
2. 悪化した変種 W（家族 A）: 基準 1 回目との WA の差の合計 Δ >= 3、規程ごとの差の符号を入れ替える正確検定（片側）の
   p を変種の数で Holm に直して < ALPHA_A、かつ Δ > 2×揺れ。
3. 物差し s の安全域 Safe_s: s のずれ <= ANCHOR（Q4_K_M-en）のずれ。
4. 型 A（気づかない）: Safe_s ∩ W が空でない。
5. 型 B（気づかない・家族 B）: s のずれの差が大きい方の 10% 以内の対（KLD だけで選び、答えは見ない）。
   WA の差の符号入れ替え正確検定（両側）の p を対の数で Holm に直して < ALPHA_B、|Δ| >= 3、|Δ| > 2×揺れ、
   WA の多い方が W に入り、少ない方が答えている（形式違反 <= 5%・保留しすぎの率の増え <= 5 点）。
6. 答えていない変種（形式違反 > 5% か 保留しすぎの増え > 5 点）は「応答不能が増えた」と印を付ける。W からは外さない。
7. 同じと言える: 規程を引き直すブートストラップ（種 20261111・1 万回）で WA の増えの 95% 上側 < 10 問。
8. モデルごと・物差しごと: 0 を満たさない →「測れない」。型 A か B →「気づかない」。W があり Safe_s が全部同じと言える
   →「気づく」。W があり Safe_s に同じと言えない物がある →「決められない」。W が無い →「差が出ない」。
9. α: 1 モデル 0.10/3 を A と B に半分ずつ。3 モデルで「気づかない」を 2 つ以上誤って出す確率は
   依存によらず <= 0.05（P(N>=2) <= E[N]/2）。主の物差しは kld_en だけ。ほかの物差しは同じ線で出すが探索の扱い。
10. 全体: 「気づかない」2 つ以上 →「はい」、1 つ →「モデルによる」、0 で「気づく」2 つ以上 →「いいえ」、ほかは「測れず」。
11. 厳しく見る時（本番）: 答えの欠け・変種が 15 でない・ずれが数でない → 判定せずに止める。
"""
from __future__ import annotations

import math
from itertools import combinations

from .stats import block_bootstrap_upper, discordant, holm, mcnemar_one_sided, signflip_exact, spearman

JUDGE_VERSION = "qp-judge-2"
MIN_DELTA = 3
ALPHA_MODEL = 0.10 / 3
ALPHA_A = ALPHA_MODEL / 2
ALPHA_B = ALPHA_MODEL / 2
NEAR_REL = 0.10
REF_MIN_CORRECT = 0.70
REF_MIN_HELD = 0.50
REF_MAX_FORMAT = 0.05
RESP_MAX_FORMAT = 0.05
RESP_MAX_OVERHOLD_UP = 0.05
EQUIV_MARGIN = 10
BOOT_N = 10000
BOOT_SEED = 20261111
ANCHOR = "Q4_K_M-en"
N_VARIANTS = 15
METRICS = ("kld_en", "ppl_en", "kld_ja", "ppl_ja", "kld_dom")
PRIMARY = "kld_en"


class StrictError(ValueError):
    """厳しく見る時に、判定の前提が欠けている。"""


def deviation(metrics: dict, s: str) -> float | None:
    """物差し s のずれ。KLD は平均 KLD、PPL は ln(PPL(Q)/PPL(base))。"""
    m = metrics.get(s)
    if m is None:
        return None
    if s.startswith("kld"):
        return m.get("kld_mean")
    if "ln_ppl_ratio" in m:
        return m["ln_ppl_ratio"]
    r = m.get("ppl_ratio")
    return math.log(r) if r else None


def _wa(outcomes: dict[str, str]) -> dict[str, bool]:
    return {k: v == "wrong_assert" for k, v in outcomes.items()}


def block_diffs(x: dict[str, bool], y: dict[str, bool], blocks: dict[str, str]) -> list[int]:
    """規程ごとの (x の WA の数 - y の WA の数)。規程の並びは名前の順。"""
    if set(x) != set(y):
        raise ValueError("比べる 2 つの問の集合が違う")
    d = {b: 0 for b in sorted({blocks[i] for i in x})}
    for i in x:
        d[blocks[i]] += int(x[i]) - int(y[i])
    return list(d.values())


def quality(o: dict[str, str], should_hold: dict[str, bool]) -> dict[str, float]:
    ans = [k for k in o if not should_hold[k]]
    hold = [k for k in o if should_hold[k]]
    return {"correct": sum(o[k] == "correct" for k in ans) / max(1, len(ans)),
            "held": sum(o[k] == "held_ok" for k in hold) / max(1, len(hold)),
            "format": sum(v in ("format", "missing") for v in o.values()) / max(1, len(o)),
            "over_hold": sum(o[k] == "over_hold" for k in ans) / max(1, len(ans))}


def ref_ok(q: dict[str, float]) -> bool:
    return q["correct"] >= REF_MIN_CORRECT and q["held"] >= REF_MIN_HELD and q["format"] <= REF_MAX_FORMAT


def _check_strict(ref1, ref2, variants, noise_pairs) -> None:
    from .pipeline import ALL_VARIANTS
    names = {v["name"] for v in variants}
    if names != set(ALL_VARIANTS):
        raise StrictError(f"変種が 15 そろっていない: 足りない {sorted(set(ALL_VARIANTS) - names)}")
    runs = [("ref-r1", ref1), ("ref-r2", ref2), *((v["name"], v["outcomes"]) for v in variants),
            *((f"{n}:a", a) for n, a, _ in noise_pairs), *((f"{n}:b", b) for n, _, b in noise_pairs)]
    for n, o in runs:
        miss = sum(v == "missing" for v in o.values())
        if miss:
            raise StrictError(f"{n} に答えの無い問が {miss}")
    for v in variants:
        for s in METRICS:
            x = deviation(v["metrics"], s)
            if x is None or not math.isfinite(x):
                raise StrictError(f"{v['name']} の {s} が数でない: {x}")


def judge_model(ref1: dict[str, str], ref2: dict[str, str], variants: list[dict],
                should_hold: dict[str, bool], blocks: dict[str, str],
                noise_pairs: list[tuple[str, dict[str, str], dict[str, str]]] = (),
                strict: bool = False) -> dict:
    """ref1/ref2: id → 結果の語。variants: [{name, outcomes, metrics: {s: parse_log の辞書}}]。
    blocks: id → 規程の名前。noise_pairs: [(名前, 1 回目の結果, 2 回目の結果)]（同じ物を 2 回流した組）。"""
    names = [v["name"] for v in variants]
    if ANCHOR not in names:
        raise ValueError(f"{ANCHOR} が変種に無い")
    if len(set(names)) != len(names):
        raise ValueError("変種の名前が重なっている")
    if strict:
        _check_strict(ref1, ref2, variants, noise_pairs)
    w_ref = _wa(ref1)
    q1, q2 = quality(ref1, should_hold), quality(ref2, should_hold)
    valid = ref_ok(q1) and ref_ok(q2)
    noise_each = {"ref": sum(discordant(_wa(ref1), _wa(ref2)))}
    for n, a, b in noise_pairs:
        noise_each[n] = sum(discordant(_wa(a), _wa(b)))
    noise = max(noise_each.values())

    per, p1 = {}, {}
    for v in variants:
        w = _wa(v["outcomes"])
        d = block_diffs(w, w_ref, blocks)
        b, c = discordant(w, w_ref)
        vq = quality(v["outcomes"], should_hold)
        responsive = vq["format"] <= RESP_MAX_FORMAT and vq["over_hold"] - q1["over_hold"] <= RESP_MAX_OVERHOLD_UP
        upper = block_bootstrap_upper(d, 0.95, BOOT_N, BOOT_SEED)
        per[v["name"]] = {"wa": sum(w.values()), "delta": sum(d), "b": b, "c": c,
                          "p_mcnemar_ref": mcnemar_one_sided(b, c), "format_rate": vq["format"],
                          "over_hold_up": vq["over_hold"] - q1["over_hold"], "responsive": responsive,
                          "upper95": upper, "same": upper < EQUIV_MARGIN, "_w": w}
        p1[v["name"]] = signflip_exact(d, "greater")
    ph = holm(p1)
    worse = []
    for n in names:
        d = per[n]
        d["p"], d["p_holm"] = p1[n], ph[n]
        d["worse"] = d["delta"] >= MIN_DELTA and ph[n] < ALPHA_A and d["delta"] > 2 * noise
        if d["worse"]:
            worse.append(n)
    unresponsive = [n for n in names if not per[n]["responsive"]]

    by_metric = {}
    for s in METRICS:
        dev = {n: deviation(v["metrics"], s) for n, v in zip(names, variants)}
        if any(x is None or not math.isfinite(x) for x in dev.values()):
            by_metric[s] = {"verdict": "値が無い"}
            continue
        safe = [n for n in names if dev[n] <= dev[ANCHOR]]
        pattern_a = [n for n in safe if n in worse]
        near = {}
        for a, b in combinations(names, 2):
            if abs(dev[a] - dev[b]) <= NEAR_REL * max(abs(dev[a]), abs(dev[b])):
                d = block_diffs(per[a]["_w"], per[b]["_w"], blocks)
                near[f"{a}|{b}"] = (a, b, sum(d), signflip_exact(d, "two-sided"))
        nh = holm({k: t[3] for k, t in near.items()})
        pattern_b = []
        for k, (a, b, delta, _) in near.items():
            hi, lo = (a, b) if delta > 0 else (b, a)
            if (abs(delta) >= MIN_DELTA and nh[k] < ALPHA_B and abs(delta) > 2 * noise
                    and hi in worse and per[lo]["responsive"]):
                pattern_b.append(k)
        if not valid:
            verdict = "測れない"
        elif pattern_a or pattern_b:
            verdict = "気づかない"
        elif worse:
            verdict = "気づく" if all(per[n]["same"] for n in safe) else "決められない"
        else:
            verdict = "差が出ない"
        rho = spearman([dev[n] for n in names], [per[n]["wa"] for n in names])
        by_metric[s] = {"verdict": verdict, "exploratory": s != PRIMARY, "anchor": dev[ANCHOR], "safe": safe,
                        "safe_not_same": [n for n in safe if not per[n]["same"]], "pattern_a": pattern_a,
                        "near_pairs": {k: {"delta": t[2], "p": t[3], "p_holm": nh[k]} for k, t in near.items()},
                        "pattern_b": pattern_b, "spearman_dev_wa": rho, "deviation": dev}
    for d in per.values():
        d.pop("_w")
    return {"version": JUDGE_VERSION, "valid": valid, "ref_quality": [q1, q2], "ref_wa": sum(w_ref.values()),
            "noise": noise, "noise_each": noise_each, "variants": per, "worse": worse,
            "unresponsive": unresponsive, "by_metric": by_metric}


def judge_overall(model_results: dict[str, dict], metric: str = PRIMARY) -> str:
    verdicts = [r["by_metric"][metric]["verdict"] for r in model_results.values()
                if r["valid"] and r["by_metric"][metric]["verdict"] != "値が無い"]
    if len(verdicts) < 2:
        return "測れず"
    miss = verdicts.count("気づかない")
    if miss >= 2:
        return "はい"
    if miss == 1:
        return "モデルによる"
    if verdicts.count("気づく") >= 2:
        return "いいえ"
    return "測れず"
