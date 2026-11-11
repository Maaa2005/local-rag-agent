"""判定の線 qp-judge-5 の検出力と誤りの率を、測る前に見積もる。judge_model を端から端まで通す。結果は計画 6 章に写す。

    python -m evaluation.quant_probe.power --sims 2000 [--joint-sims 2000]

作り（問の並び・規程・保留すべき問は本物の 360 問。答えは作り物）:
- 基準: 規程ごとに誤った断定（WA）の率 p_b ~ N(0.10, 0.06)（0〜0.5 に切る）。答えられる問は WA p_b・保留しすぎ 0.03・
  形式違反 0.005・残り正答。保留すべき問は WA p_b・形式違反 0.005・残り保留できた。
- 揺れの組（基準の 2 回目・Q4_K_M-en の 2 回目・Q6_K-none の載せ方違い）: 元から各問を 1% の確率で同じ分布から引き直す。
- 変種: 基準 1 回目から、bit ごとの churn（Q6 .01・Q5 .02・Q4 .04・Q3 .08・Q2 .15）の確率で各問を同じ分布から
  引き直す（平均では増えも減りもしない揺れ）。次に、気づくモデルの増え（NOTICE）を 24 規程に散らして WA に変える。
  見逃しの場面では、そのあと狙いの変種にだけ d 問を、広さ（24 か 8 規程）の規程に足す。
- ずれ（全部の物差しで同じ値）: bit ごと Q6 .004・Q5 .010・Q4 .030・Q3 .090・Q2 .35 に校正の倍率 none 1.25・en 1.0・ja 1.08。
  安全域（<= Q4_K_M-en）は Q6 と Q5 の 6 つと Q4_K_M-en。10% 以内の近い対は同じ bit の en と ja。
場面:
- notice（気づくモデル・本当の見逃しは 0）: Δ = Q6・Q5・Q4 0・Q3 12・Q2 40。安全域と近い対（同じ bit の en と ja）の
  Δ はどれも同じなので、家族 A の安全域と家族 B の近い対は全部が本当の帰無（μ = 0）。
- miss_A: notice に加え Q5_K_M-ja（安全域）に d。
- miss_B: notice に加え Q3_K_M-ja に d（Q3_K_M-en と KLD が 10% 以内）。Q3_K_M-ja の NOTICE 12 は 24 規程に散らし、
  d だけを広さの規程に集める。
- null: Δ 全部 0。
- skew（偏った帰無）: notice に加え、Q5_K_M-ja（安全域）だけ、WA が 1 問以上あり WA でない問が 3 問以上ある規程ごとに、
  1/4 の確率で WA を 3 問足し、3/4 の確率で 1 問消す。どの規程でも増えの平均はちょうど 0 で、分布は 0 の回りで対称でない。
- skew_rev（逆向きの偏った帰無）: 同じ規程の選び方を逆にして、WA が 3 問以上あり WA でない問が 1 問以上ある規程ごとに、
  3/4 の確率で 1 問足し、1/4 の確率で 3 問消す（小さな正が多く、まれに大きな負。符号の入れ替えに危ない向き）。
  どちらも、符号の入れ替えの前提が崩れた時の、誤った「気づかない」の率を見る。
1 モデルの表のあと、3 モデルを一緒に作る見積り（joint）も出す。3 モデルは規程の難しさを分け合う（p_b の 0.10 からの
ずれの半分の分散を共通に引き、残りの半分を各モデルで引く。1 モデルの分布は独立の時と同じ）。見逃しの d を足す規程と、
skew の規程ごとの向きも 3 モデルで同じにする（相関を強める向きの一例で、最も悪い場合とは限らない）。
足した問の数と触れた規程の数は、実際に足せた数を数える（規程の空きが足りない時は少なくなる）。差が正／負の規程の数は、
狙いの変種と基準 1 回目の差と、狙いの変種と同じ bit の en の変種（近い対の相手）の差の両方を数える。
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
NOTICE = {"Q6_K": 0, "Q5_K_M": 0, "Q4_K_M": 0, "Q3_K_M": 12, "Q2_K": 40}
TARGET = {"miss_A": "Q5_K_M-ja", "miss_B": "Q3_K_M-ja"}
SKEWED = "Q5_K_M-ja"
RERUN_NOISE = 0.01
P_MEAN, P_SD = 0.10, 0.06
VERDICTS = ("気づかない", "気づく", "決められない", "差が出ない", "測れない")
OVERALL = ("はい", "モデルによる", "いいえ（この 360 問で）", "測れず")


def _split(name: str) -> tuple[str, str]:
    q, c = name.rsplit("-", 1)
    return q, c


def base_delta(scene: str, name: str) -> int:
    """気づくモデルの増え（24 規程に散らす）。null と錨は 0。"""
    return 0 if scene == "null" or name == ANCHOR else NOTICE[_split(name)[0]]


def skew(w: "World", o: dict[str, str], coins: dict[str, bool] | None = None,
         reverse: bool = False) -> dict[str, str]:
    """規程ごとの増えの平均がちょうど 0 で、0 の回りで対称でない揺れ。coins は規程ごとの、確率 1/4 の向きを引いたか。
    reverse でない時: 1/4 で 3 問足す・3/4 で 1 問消す。reverse の時: 1/4 で 3 問消す・3/4 で 1 問足す。"""
    o = dict(o)
    big, small = (-3, 1) if reverse else (3, -1)
    for b in w.order:
        ids = [i for i in w.ids if w.block[i] == b]
        on = [i for i in ids if o[i] == "wrong_assert"]
        off = [i for i in ids if o[i] != "wrong_assert"]
        need_on, need_off = (3, 1) if reverse else (1, 3)
        if len(on) < need_on or len(off) < need_off:
            continue  # 両方の向きに動かせる規程だけ動かす（向きは答えと無関係に引くので、平均はちょうど 0）
        rare = coins[b] if coins is not None else w.rng.random() < 0.25
        k = big if rare else small
        if k > 0:
            for i in w.rng.sample(off, k):
                o[i] = "wrong_assert"
        else:
            for i in w.rng.sample(on, -k):
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

    def shared(self, spread: int) -> dict:
        """3 モデルで分け合う物: 規程の難しさの共通の分・d を足す規程・skew の向き。"""
        return {"diff": {b: self.rng.gauss(0.0, P_SD / math.sqrt(2)) for b in self.order},
                "blocks": self.pick(spread),
                "coins": {b: self.rng.random() < 0.25 for b in self.order}}

    def ref(self, diff: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, str]]:
        if diff is None:
            p = {b: self.rng.gauss(P_MEAN, P_SD) for b in self.order}
        else:
            p = {b: P_MEAN + diff[b] + self.rng.gauss(0.0, P_SD / math.sqrt(2)) for b in self.order}
        p = {b: min(0.5, max(0.0, x)) for b, x in p.items()}
        rate = {i: p[self.block[i]] for i in self.ids}
        return rate, {i: self.draw(i, rate[i]) for i in self.ids}

    def redraw(self, base: dict[str, str], rate: dict[str, float], frac: float) -> dict[str, str]:
        return {i: (self.draw(i, rate[i]) if self.rng.random() < frac else o) for i, o in base.items()}

    def pick(self, spread: int) -> set[str]:
        return set(self.rng.sample(sorted(self.order), spread))

    def add_wa(self, o: dict[str, str], delta: int, blocks: set[str]) -> tuple[dict[str, str], int, int]:
        """blocks の規程の WA でない問から delta 問を WA に変える。（新しい答え・足せた問の数・触れた規程の数）"""
        if delta <= 0:
            return o, 0, 0
        o = dict(o)
        slots = [i for i in self.ids if self.block[i] in blocks and o[i] != "wrong_assert"]
        chosen = self.rng.sample(slots, min(delta, len(slots)))
        for i in chosen:
            o[i] = "wrong_assert"
        return o, len(chosen), len({self.block[i] for i in chosen})


def one_model(w: World, scene: str, d: int, spread: int, shared: dict | None = None,
              info: dict | None = None) -> str:
    """1 モデルを作って判定する。shared があれば規程の難しさ・d の規程・skew の向きをそれに合わせる。
    info を渡すと、狙いの変種に実際に足せた問の数（added）と触れた規程の数（touched）を入れる。"""
    rate, r1 = w.ref(shared["diff"] if shared else None)
    r2 = w.redraw(r1, rate, RERUN_NOISE)
    everywhere = set(w.order)
    variants, outs = [], {}
    for n in ALL_VARIANTS:
        q, _ = _split(n)
        o, _, _ = w.add_wa(w.redraw(r1, rate, CHURN[q]), base_delta(scene, n), everywhere)
        if TARGET.get(scene) == n:
            blocks = shared["blocks"] if shared else w.pick(spread)
            o, added, touched = w.add_wa(o, d, blocks)
            if info is not None:
                info["added"], info["touched"] = added, touched
        if scene in ("skew", "skew_rev") and n == SKEWED:
            o = skew(w, o, shared["coins"] if shared else None, reverse=scene == "skew_rev")
        outs[n] = o
        variants.append({"name": n, "outcomes": o, "metrics": _metrics(n)})
    tgt = TARGET.get(scene, SKEWED if scene in ("skew", "skew_rev") else None)
    if info is not None and tgt:
        partner = _split(tgt)[0] + "-en"
        for key, other in (("", r1), ("pair_", outs[partner])):
            diff = {b: 0 for b in w.order}
            for i in w.ids:
                diff[w.block[i]] += (outs[tgt][i] == "wrong_assert") - (other[i] == "wrong_assert")
            info[key + "pos"] = sum(v > 0 for v in diff.values())
            info[key + "neg"] = sum(v < 0 for v in diff.values())
    noise = [(RERUN, outs[ANCHOR], w.redraw(outs[ANCHOR], rate, RERUN_NOISE)),
             (PLACE, outs["Q6_K-none"], w.redraw(outs["Q6_K-none"], rate, RERUN_NOISE))]
    j = judge_model(r1, r2, variants, w.hold, w.block, noise, strict=True, need_place=True, boot_n=0)
    return j["by_metric"][PRIMARY]["verdict"] if j["valid"] else "測れない"


def joint(w: World, scenes: list[tuple[str, int]], spread: int) -> str:
    """3 モデルを、規程の難しさ・d の規程・skew の向きを分け合って作り、全体の答えを返す。"""
    sh = w.shared(spread)
    res = {}
    for k, (s, d) in enumerate(scenes):
        v = one_model(w, s, d, spread, sh)
        res[f"m{k}"] = {"valid": v != "測れない", "by_metric": {PRIMARY: {"verdict": v}}}
    return judge_overall(res, PRIMARY)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    den = 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n) - c) / den), min(1.0, (p + z * z / (2 * n) + c) / den)


def rates(scene: str, d: int, spread: int, sims: int, seed: int) -> tuple[dict[str, int], dict[str, float]]:
    w = World(random.Random(f"{seed}-{scene}-{d}-{spread}"))
    cnt = {v: 0 for v in VERDICTS}
    keys = ("added", "touched", "pos", "neg", "pair_pos", "pair_neg")
    tot = {k: 0 for k in keys}
    for _ in range(sims):
        info = {k: 0 for k in keys}
        cnt[one_model(w, scene, d, spread, info=info)] += 1
        for x in tot:
            tot[x] += info[x]
    return cnt, {x: v / sims for x, v in tot.items()}


def joint_rates(label: str, scenes: list[tuple[str, int]], spread: int, sims: int, seed: int) -> dict[str, int]:
    w = World(random.Random(f"{seed}-joint-{label}-{spread}"))
    cnt = {o: 0 for o in OVERALL}
    for _ in range(sims):
        cnt[joint(w, scenes, spread)] += 1
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


def _cells(cnt: dict[str, int], keys, n: int) -> str:
    out = []
    for v in keys:
        lo, hi = wilson(cnt[v], n)
        out.append(f"{cnt[v] / n:.3f} ({lo:.3f}-{hi:.3f})")
    return " | ".join(out)


def joint_scenes() -> list[tuple[str, list[tuple[str, int]], int]]:
    """（名前・3 モデルの場面・広さ）。見逃すモデルの数 k は 0〜3。"""
    out = [("notice k=0", [("notice", 0)] * 3, 24), ("null", [("null", 0)] * 3, 24),
           ("skew×3（向きをそろえる）", [("skew", 0)] * 3, 24),
           ("skew_rev×3（向きをそろえる）", [("skew_rev", 0)] * 3, 24)]
    for s in ("miss_A", "miss_B"):
        for d in (10, 20, 30):
            for k in (1, 2, 3):
                out.append((f"{s} d={d} k={k}", [(s, d)] * k + [("notice", 0)] * (3 - k), 24))
    for s, d in (("miss_A", 20), ("miss_B", 30)):
        for k in (2, 3):
            out.append((f"{s} d={d} k={k} 8 規程", [(s, d)] * k + [("notice", 0)] * (3 - k), 8))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--joint-sims", type=int, default=None, help="3 モデル一緒の見積りの回数（既定は --sims と同じ。0 で出さない）")
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args(argv)
    js = a.sims if a.joint_sims is None else a.joint_sims
    scenes = [("notice", 0, 24), ("null", 0, 24), ("skew", 0, 24), ("skew_rev", 0, 24)]
    scenes += [(s, d, sp) for s in ("miss_A", "miss_B") for sp in (24, 8) for d in (10, 20, 30)]
    table = {}
    print(f"1 モデルの判定の割合（主の物差し {PRIMARY}・{a.sims} 回・括弧は Wilson 95%。足した問・触れた規程・差が正／負の規程は"
          "狙いの変種（skew は Q5_K_M-ja）の平均。差は規程ごとの WA の差で、基準 1 回目との差と、同じ bit の en との差）")
    print("| 場面 | 増え d | 広さ | 足した問 | 触れた規程 | 基準との差 正／負 | en との差 正／負 | " + " | ".join(VERDICTS) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(VERDICTS))
    for s, d, sp in scenes:
        cnt, info = rates(s, d, sp, a.sims, a.seed)
        table[(s, d, sp)] = {v: c / a.sims for v, c in cnt.items()}
        miss = s.startswith("miss")
        print(f"| {s} | {d or '-'} | {sp if miss else '-'} | {info['added']:.1f} | {info['touched']:.1f} | "
              f"{info['pos']:.1f}／{info['neg']:.1f} | {info['pair_pos']:.1f}／{info['pair_neg']:.1f} | "
              + _cells(cnt, VERDICTS, a.sims) + " |", flush=True)
    print("\n3 モデルの全体の答えの割合（独立とみなした数え上げ。見逃すモデルの数 k・見逃すモデルは 24 規程）")
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
    if js > 0:
        print(f"\n3 モデルを一緒に作った全体の答えの割合（規程の難しさ・d の規程・skew の向きを分け合う・{js} 回・括弧は Wilson 95%）")
        print("| 場面 | " + " | ".join(OVERALL) + " |")
        print("|---|" + "---|" * len(OVERALL))
        for label, sc, sp in joint_scenes():
            cnt = joint_rates(label, sc, sp, js, a.seed)
            print(f"| {label} | " + _cells(cnt, OVERALL, js) + " |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
