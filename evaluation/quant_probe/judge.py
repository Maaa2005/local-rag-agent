"""測る前に固定した判定の線 qp-judge-8（docs/quant-probe-plan.md の 5 章を docs/quant-probe-plan-v8.md の 2・4・5・6 章で替えた物。
変えるなら測る前に版を上げる）。

まとまり = 規程（24 規程・1 規程 15 問）。誤った断定（WA）の差は規程ごとに足し、規程を単位に検定する。
検定の前提（比べる 2 つ a・b ごと）: モデル・量子化したファイル・評価の文章・KLD の値（まとめて F）を固定した上で、
規程 r の差を D_r = μ + ε_r と書き、ε_r は F の下で 0 の回りで対称、規程どうしは独立とする。家族 A（基準と変種）の
帰無は μ <= 0（片側）、家族 B（変種どうし）の帰無は μ = 0（両側）。μ = 0 なら、|D_r| を条件にした符号の入れ替えの
分布が正確な帰無の分布で、μ < 0 では片側の p は保守的（各 D_r を下げても p は下がらない）。平均が 0 なだけでは足りない。
近い対の集まりは F だけで決まり、F の下で各本当の帰無の p が有効と仮定するので、選んだ対だけを Holm に直してよい。
モデルと校正のやり方は規程の成績で選ばない。

0. 基準が測れる: 基準 Q8_0 の 2 回とも、次の 3 つを全部満たす（満たさなければ「測れない」）。
   0a 形式: 形式違反と欠けが 5% 以下。
   0b 向きを見分けている: 答えられる問で、規程 r ごとに c_r =（正答の数）−（逆の結論を言い切った数）。
      規程を単位にした符号入れ替えの正確検定（片側・c > 0 の向き）で p < REF_ALPHA（0.001）。
   0c 保留を見分けている: 規程 r ごとに h_r =（保留すべき問で保留した数）×（その規程の答えられる問の数）
      −（答えられる問で保留した数）×（その規程の保留すべき問の数）。同じ検定で p < REF_ALPHA。
   当てずっぽう（いつも同じ語・いつも保留・でたらめ）は c も h も 0 の回りで、通らない（全部 0 の差は p = 1）。
1. 揺れ: 同じ物をもう 1 回流した組（基準の 2 回・Q4_K_M-en の 2 回）で、WA が食い違った問の数の最大。
   基準が GPU に載り切らない時は、答えの回を全部同じ `-ngl` で流す（pipeline.py・report.py が載った層の数を照合する）。
2. 悪化した変種 W（家族 A）: 基準 1 回目との WA の差の合計 Δ >= 3、規程ごとの差の符号を入れ替える正確検定（片側）の
   p を 15 変種で Holm に直して < ALPHA_A、かつ Δ > 2×揺れ。
3. 物差し s の安全域 Safe_s: s のずれ <= ANCHOR（Q4_K_M-en）のずれ。
4. 型 A（気づかない）: Safe_s ∩ W が空でない。
5. 型 B（気づかない・家族 B）: s のずれの差が大きい方の 10% 以内の対。WA の差の符号入れ替え正確検定（両側）の p を
   対の数で Holm に直して < ALPHA_B、|Δ| >= 3、|Δ| > 2×揺れ、WA の多い方が W に入り、少ない方が答えている
   （形式違反 <= 5%・保留しすぎの率の増え <= 5 点）。
6. 答えていない変種（形式違反 > 5% か 保留しすぎの増え > 5 点）は「応答不能が増えた」と印を付ける。W からは外さない。
7. 小さい: この 360 問の上で WA の増え Δ < 3（悪化の線と同じ幅。規程一般への保証は無い・記述）。
   規程を引き直すブートストラップの 95% 上側は参考に出すだけで、線には使わない（24 規程では覆う確率を保証できない）。
8. モデルごと・物差しごと: 0 を割る →「測れない」。型 A か B →「気づかない」。
   W があり、Safe_s の全部が「答えている」かつ「小さい」、かつ「近い対で差が残る」物が無い →「気づく」。
   「近い対で差が残る」= 近い対で |Δ| >= 3 なのに型 B と決まらなかった（両方が W の対も含む。少数の規程に
   集まった型 B を「気づく」と取り違えないため）。
   W があり、それ以外 →「決められない」（安全域で増えが残る・安全域が答えていない・近い対に差が残る）。W が無い →「差が出ない」。
9. α: 1 モデル 0.0125 を A と B に半分ずつ。「はい」（5 モデル中 2 つ以上で気づかない）を誤って出す確率は、
   測る前に決めた 5 モデルについて、本当に見逃すモデルが 0 なら 5×0.0125/2 = 0.03125 以下、1 なら残り 4 つの
   どれかの誤り 4×0.0125 = 0.05 以下（主の物差し・上の前提の下で）。基準の成績でモデルを替えるとこの保証が崩れるので、
   替えは「変換できない」時だけで、どのモデルの答えも流す前に決める（report.py・計画 4 章）。基準不適格は「測れない」で
   「気づかない」を出さないので上限を増やさない。ただし適格になったことを条件にした誤りの率の保証ではない。
   主の物差しは kld_en だけ。ほかの物差し（kld_chat を含む）は同じ線で出すが探索の扱い。「いいえ」は 8 の記述の線に依り、誤りの確率を保証しない。
10. 全体: 「気づかない」2 つ以上 →「はい」、1 つ →「モデルによる」、0 で「気づく」3 つ以上 →「いいえ（この 360 問で）」、ほかは「測れず」。
11. 厳しく見る時（本番）: 答えの欠け・問が 360 でない・規程が 24 でない・変種が 15 でない・ずれが数でない・
    揺れの組（Q4_K_M-en の 2 回目）が無い → 判定せずに止める。
    kld_chat（探索）だけは、値が無くても止めない。その物差しの判定が「値が無い」になり、ほかの物差しと全体の答えは出す。
"""
from __future__ import annotations

import math
from itertools import combinations

from .pipeline import ALL_VARIANTS, RERUN
from .stats import block_bootstrap_upper, discordant, holm, mcnemar_one_sided, signflip_exact, spearman

JUDGE_VERSION = "qp-judge-8"
MIN_DELTA = 3
ALPHA_MODEL = 0.0125
ALPHA_A = ALPHA_MODEL / 2
ALPHA_B = ALPHA_MODEL / 2
NEAR_REL = 0.10
REF_ALPHA = 0.001
REF_MAX_FORMAT = 0.05
RESP_MAX_FORMAT = 0.05
RESP_MAX_OVERHOLD_UP = 0.05
BOOT_N = 10000
BOOT_SEED = 20261111
ANCHOR = "Q4_K_M-en"
N_QUESTIONS = 360
N_BLOCKS = 24
METRICS = ("kld_en", "ppl_en", "kld_ja", "ppl_ja", "kld_dom", "kld_chat")
PRIMARY = "kld_en"
OPTIONAL_METRICS = ("kld_chat",)  # 取れなくても全体の答えを止めない（計画 版 8 の 5 章）


class StrictError(ValueError):
    """厳しく見る時に、判定の前提が欠けている（基準が弱い「測れない」とは別）。"""


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


def ref_blocks(o: dict[str, str], should_hold: dict[str, bool], blocks: dict[str, str]) -> tuple[list[int], list[int]]:
    """規程ごと（名前の順）の c_r・h_r（線 0 の 0b・0c）。
    c_r = 答えられる問の（正答 − 逆の結論を言い切った数）。
    h_r = （保留すべき問で保留した数）×（答えられる問の数）−（答えられる問で保留した数）×（保留すべき問の数）。"""
    rules = sorted({blocks[i] for i in o})
    c, held, over, n_ans, n_hold = ({r: 0 for r in rules} for _ in range(5))
    for i, v in o.items():
        r = blocks[i]
        if should_hold[i]:
            n_hold[r] += 1
            held[r] += v == "held_ok"
        else:
            n_ans[r] += 1
            c[r] += (v == "correct") - (v == "wrong_assert")
            over[r] += v == "over_hold"
    return [c[r] for r in rules], [held[r] * n_ans[r] - over[r] * n_hold[r] for r in rules]


def quality(o: dict[str, str], should_hold: dict[str, bool], blocks: dict[str, str]) -> dict[str, float]:
    """基準 1 回分の成績。率の 4 つ（correct・held・format・over_hold）に、c・h の合計と線 0 の 2 つの p を足す。"""
    ans = [k for k in o if not should_hold[k]]
    hold = [k for k in o if should_hold[k]]
    c, h = ref_blocks(o, should_hold, blocks)
    return {"correct": sum(o[k] == "correct" for k in ans) / max(1, len(ans)),
            "held": sum(o[k] == "held_ok" for k in hold) / max(1, len(hold)),
            "format": sum(v in ("format", "missing") for v in o.values()) / max(1, len(o)),
            "over_hold": sum(o[k] == "over_hold" for k in ans) / max(1, len(ans)),
            "c_sum": sum(c), "h_sum": sum(h),
            "p_direction": signflip_exact(c, "greater"), "p_hold": signflip_exact(h, "greater")}


def ref_ok(q: dict[str, float]) -> bool:
    return q["format"] <= REF_MAX_FORMAT and q["p_direction"] < REF_ALPHA and q["p_hold"] < REF_ALPHA


def _check_strict(ref1, ref2, variants, noise_pairs, blocks) -> None:
    if len(ref1) != N_QUESTIONS or len({blocks[i] for i in ref1}) != N_BLOCKS:
        raise StrictError(f"問が {len(ref1)}・規程が {len({blocks[i] for i in ref1})}（{N_QUESTIONS}・{N_BLOCKS} であるべき）")
    names = {v["name"] for v in variants}
    if names != set(ALL_VARIANTS):
        raise StrictError(f"変種が 15 そろっていない: 足りない {sorted(set(ALL_VARIANTS) - names)}")
    have = {n for n, _, _ in noise_pairs}
    if RERUN not in have:
        raise StrictError(f"揺れの組が無い: {[RERUN]}")
    runs = [("ref-r1", ref1), ("ref-r2", ref2), *((v["name"], v["outcomes"]) for v in variants),
            *((f"{n}:a", a) for n, a, _ in noise_pairs), *((f"{n}:b", b) for n, _, b in noise_pairs)]
    for n, o in runs:
        miss = sum(v == "missing" for v in o.values())
        if miss:
            raise StrictError(f"{n} に答えの無い問が {miss}")
    for v in variants:
        for s in METRICS:
            if s in OPTIONAL_METRICS:
                continue  # 値が無くても止めない（その物差しの判定だけ「値が無い」になる）
            x = deviation(v["metrics"], s)
            if x is None or not math.isfinite(x):
                raise StrictError(f"{v['name']} の {s} が数でない: {x}")


def judge_model(ref1: dict[str, str], ref2: dict[str, str], variants: list[dict],
                should_hold: dict[str, bool], blocks: dict[str, str],
                noise_pairs: list[tuple[str, dict[str, str], dict[str, str]]] = (),
                strict: bool = False, boot_n: int = BOOT_N) -> dict:
    """ref1/ref2: id → 結果の語。variants: [{name, outcomes, metrics: {s: parse_log の辞書}}]。
    blocks: id → 規程の名前。noise_pairs: [(名前, 1 回目の結果, 2 回目の結果)]（同じ物を 2 回流した組）。
    揺れの組は基準の 2 回と RERUN だけ（載せ方の組は無い）。boot_n=0 で参考の上側を省く。"""
    names = [v["name"] for v in variants]
    if ANCHOR not in names:
        raise ValueError(f"{ANCHOR} が変種に無い")
    if len(set(names)) != len(names):
        raise ValueError("変種の名前が重なっている")
    if strict:
        _check_strict(ref1, ref2, variants, noise_pairs, blocks)
    w_ref = _wa(ref1)
    q1, q2 = quality(ref1, should_hold, blocks), quality(ref2, should_hold, blocks)
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
        vq = quality(v["outcomes"], should_hold, blocks)
        responsive = vq["format"] <= RESP_MAX_FORMAT and vq["over_hold"] - q1["over_hold"] <= RESP_MAX_OVERHOLD_UP
        per[v["name"]] = {"wa": sum(w.values()), "delta": sum(d), "b": b, "c": c,
                          "p_mcnemar_ref": mcnemar_one_sided(b, c), "format_rate": vq["format"],
                          "over_hold_up": vq["over_hold"] - q1["over_hold"], "responsive": responsive,
                          "small": sum(d) < MIN_DELTA,
                          "upper95_ref": block_bootstrap_upper(d, 0.95, boot_n, BOOT_SEED) if boot_n else None,
                          "_w": w}
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

    pair_cache: dict[tuple[str, str], tuple[int, float]] = {}  # 近い対の（増え・p）は物差しによらないので、1 回だけ数える

    def pair_test(a: str, b: str) -> tuple[int, float]:
        if (a, b) not in pair_cache:
            d = block_diffs(per[a]["_w"], per[b]["_w"], blocks)
            pair_cache[(a, b)] = (sum(d), signflip_exact(d, "two-sided"))
        return pair_cache[(a, b)]

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
                near[f"{a}|{b}"] = (a, b, *pair_test(a, b))
        nh = holm({k: t[3] for k, t in near.items()})
        pattern_b = []
        for k, (a, b, delta, _) in near.items():
            hi, lo = (a, b) if delta > 0 else (b, a)
            if (abs(delta) >= MIN_DELTA and nh[k] < ALPHA_B and abs(delta) > 2 * noise
                    and hi in worse and per[lo]["responsive"]):
                pattern_b.append(k)
        safe_left = [n for n in safe if not (per[n]["responsive"] and per[n]["small"])]
        near_left = [k for k, t in near.items() if abs(t[2]) >= MIN_DELTA and k not in pattern_b]
        if not valid:
            verdict = "測れない"
        elif pattern_a or pattern_b:
            verdict = "気づかない"
        elif worse:
            verdict = "気づく" if not safe_left and not near_left else "決められない"
        else:
            verdict = "差が出ない"
        rho = spearman([dev[n] for n in names], [per[n]["wa"] for n in names])
        by_metric[s] = {"verdict": verdict, "exploratory": s != PRIMARY, "anchor": dev[ANCHOR], "safe": safe,
                        "safe_left": safe_left, "near_left": near_left, "pattern_a": pattern_a,
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
    if verdicts.count("気づく") >= 3:
        return "いいえ（この 360 問で）"
    return "測れず"
