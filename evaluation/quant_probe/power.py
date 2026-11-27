"""判定の線 qp-judge-8 の検出力と誤りの率を、測る前に見積もる。judge_model を端から端まで通す。結果は docs/quant-probe-power-qp-judge-8.md。

    python -m evaluation.quant_probe.power [--sims 10000] [--joint-sims 4000] [--jobs N] [--seed 20261111]

本番の回数は 1 モデル 10000 回・5 モデル一緒 4000 回（既定）。--jobs は並列の数（既定は コア数 − 2）。
種は試行ごとに決める（"種|場面|試行の番号"）ので、同じ種なら --jobs の数で結果は変わらない（数えるのは整数だけで、足し合わせの順に依らない）。
速さ: 時間の大半は judge_model の中の符号入れ替えの検定だった（1 試行 0.031 秒のうち 7 割）。stats.signflip_exact を、合計の分布を
大きな整数の桁で数える形に直した（元の書き方は signflip_exact_reference に残し、テストで同じ値になる事を確かめる）。judge_model の中で、
近い対の検定が物差しごとに同じ対を数え直していたのを 1 回にした（結果は同じ）。判定の結果は変えず、judge_model をそのまま呼ぶ。

作り（問の並び・規程・保留すべき問は本物の 360 問。答えは作り物）:
- 問の並び: data_v8 の 4 組がそろっていればそれ、無ければ版 7 の 360 問（内訳が同じなので並びだけ使う）。どちらを使ったかを出力に書く。
- 基準は 2 種類。
  強い基準（版 7 と同じ作り）: 規程ごとに誤った断定（WA）の率 p_b ~ N(0.10, 0.06)（0〜0.5 に切る）。答えられる問は WA p_b・
  保留しすぎ 0.03・形式違反 0.005・残り正答。保留すべき問は WA p_b・形式違反 0.005・残り保留できた。
  弱い基準（8B 級・計画 7 章）: 答えられる問は WA（逆の結論）~ N(0.20, 0.06)・保留しすぎ 0.15・形式違反 0.005・残り正答（約 0.65）。
  保留すべき問は WA（言い切り）~ N(0.53, 0.10)（0〜0.95 に切る）・形式違反 0.005・残り保留できた（約 0.47）。規程ごとにばらつく。
- 揺れの組（基準の 2 回目・Q4_K_M-en の 2 回目）: 元から各問を 1% の確率で同じ分布から引き直す。
- 変種: 基準 1 回目から、bit ごとの churn（Q6 .01・Q5 .02・Q4 .04・Q3 .08・Q2 .15）の確率で各問を同じ分布から
  引き直す（平均では増えも減りもしない揺れ）。次に、気づくモデルの増え（NOTICE）を 24 規程に散らして WA に変える。
  見逃しの場面では、そのあと狙いの変種にだけ d 問を、広さ（24 か 8 規程・または同じ担当の 6 規程）の規程に足す。
- ずれ（全部の物差し・kld_chat を含む・kld_en と同じ形）: bit ごと Q6 .004・Q5 .010・Q4 .030・Q3 .090・Q2 .35 に校正の倍率
  none 1.25・en 1.0・ja 1.08。安全域（<= Q4_K_M-en）は Q6 と Q5 の 6 つと Q4_K_M-en。10% 以内の近い対は同じ bit の en と ja。
場面:
- notice（気づくモデル・本当の見逃しは 0）: Δ = Q6・Q5・Q4 0・Q3 12・Q2 40。安全域と近い対（同じ bit の en と ja）の
  Δ はどれも同じなので、家族 A の安全域と家族 B の近い対は全部が本当の帰無（μ = 0）。
- miss_A: notice に加え Q5_K_M-ja（安全域）に d。
- miss_B: notice に加え Q3_K_M-ja に d（Q3_K_M-en と KLD が 10% 以内）。Q3_K_M-ja の NOTICE 12 は 24 規程に散らし、
  d だけを広さの規程に集める。
- 広さ「同じ担当の 6 規程」: 4 組（numeric_v8a〜d）のうち 1 組の 6 規程だけに d を集める（型 A・d = 20 の 1 場面）。
- null: Δ 全部 0（本当の差が 0）。
- skew（偏った帰無）: notice に加え、Q5_K_M-ja（安全域）だけ、WA が 1 問以上あり WA でない問が 3 問以上ある規程ごとに、
  1/4 の確率で WA を 3 問足し、3/4 の確率で 1 問消す。どの規程でも増えの平均はちょうど 0 で、分布は 0 の回りで対称でない。
  Q5_K_M-ja は型 A の安全域（基準 1 回目との差）にも、型 B の近い対（Q5_K_M-en との差）にも入るので、両方の差に入る。
- skew_rev（逆向きの偏った帰無）: 同じ規程の選び方を逆にして、WA が 3 問以上あり WA でない問が 1 問以上ある規程ごとに、
  3/4 の確率で 1 問足し、1/4 の確率で 3 問消す（小さな正が多く、まれに大きな負。符号の入れ替えに危ない向き）。
  どちらも、符号の入れ替えの前提が崩れた時の、誤った「気づかない」の率を見る。強い基準でも弱い基準でも数える。
出す表:
1. 線 0 を通る割合: 強い基準・弱い基準・当てずっぽうの基準 4 種（いつも同じ語＝各問の先頭の選択肢・いつも保留・でたらめ・
   規程ごとに 15 問から 1 問だけでたらめに保留してほかはでたらめに 2 語）。同じ作り物に版 7 の線（正答 0.70・保留 0.50・形式 0.05）も当てる。
2. 1 モデルの判定の割合（強い基準・弱い基準それぞれ。型 A・B、24 規程・8 規程・同じ担当の 6 規程、d = 10・20・30）。
   差が正・負・0 の規程の数も出す。
2b. 本当の差が 0 の場面（null）で、全部の試行と線 0 を通った試行のそれぞれの、変種の増え Δ の平均と、誤った「気づかない」の率。
    線 0 を通らなかった試行の「気づかない」は、基準が適格だったとして判定した時の型 A・型 B（by_metric の pattern_a・pattern_b）で数える。
3. 5 モデルの全体の答えの割合（独立とみなした数え上げ。見逃すモデル k = 0〜5）。
4. 5 モデルを一緒に作った全体の答えの割合（見逃すモデル k = 0・1〜5。強い基準 5 つの場合と、弱い 4・強い 1 の場合）。
5 モデルは規程の難しさを分け合う（p_b の 0.10 からのずれの半分の分散を共通に引き、残りの半分を各モデルで引く。
1 モデルの分布は独立の時と同じ）。見逃しの d を足す規程と、skew の規程ごとの向きも 5 モデルで同じにする
（相関を強める向きの一例で、最も悪い場合とは限らない）。弱い 4・強い 1 の場合は、見逃すモデルを弱い基準の方から先に割り当てる
（見つけにくい向き）。足した問の数と触れた規程の数は、実際に足せた数を数える（規程の空きが足りない時は少なくなる）。
差が正／負の規程の数は、狙いの変種と基準 1 回目の差と、狙いの変種と同じ bit の en の変種（近い対の相手）の差の両方を数える。
"""
from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing
import os
import random
import sys
import time
from functools import lru_cache

from .judge import (ALPHA_MODEL, ANCHOR, JUDGE_VERSION, METRICS, PRIMARY, REF_ALPHA, REF_MAX_FORMAT, RERUN, judge_model,
                    judge_overall, quality)
from .pipeline import ALL_VARIANTS
from .questions import DATA_DIR, DATA_DIR_V7, SETS, SETS_V7, load_questions

KLD_BIT = {"Q6_K": 0.004, "Q5_K_M": 0.010, "Q4_K_M": 0.030, "Q3_K_M": 0.090, "Q2_K": 0.35}
CAL = {"none": 1.25, "en": 1.0, "ja": 1.08}
CHURN = {"Q6_K": 0.01, "Q5_K_M": 0.02, "Q4_K_M": 0.04, "Q3_K_M": 0.08, "Q2_K": 0.15}
NOTICE = {"Q6_K": 0, "Q5_K_M": 0, "Q4_K_M": 0, "Q3_K_M": 12, "Q2_K": 40}
TARGET = {"miss_A": "Q5_K_M-ja", "miss_B": "Q3_K_M-ja"}
SKEWED = "Q5_K_M-ja"
RERUN_NOISE = 0.01
P_MEAN, P_SD = 0.10, 0.06
N_MODELS = 5
SETSPREAD = "set"  # 広さ: 同じ担当の 6 規程（4 組のうち 1 組）
# 版 7 の線（判定の道具の版 7 で固定だった数。比べるためだけに残す）
V7_MIN_CORRECT, V7_MIN_HELD, V7_MAX_FORMAT = 0.70, 0.50, 0.05
# 基準の作り: ans = 答えられる問の WA（平均・ばらつき）、hold = 保留すべき問の WA（None なら ans と同じ率）、over = 保留しすぎの率
REF_KINDS = {"strong": {"ans": (P_MEAN, P_SD), "hold": None, "over": 0.03},
             "weak": {"ans": (0.20, 0.06), "hold": (0.53, 0.10), "over": 0.15}}
GUESSES = {"same_word": "いつも同じ語（各問の先頭の選択肢）", "always_hold": "いつも保留", "random": "でたらめ（3 択）",
           "one_hold": "規程ごとに 15 問から 1 問だけでたらめに保留・ほかはでたらめに 2 語"}
VERDICTS = ("気づかない", "気づく", "決められない", "差が出ない", "測れない")
OVERALL = ("はい", "モデルによる", "いいえ（この 360 問で）", "測れず")
INFO_KEYS = ("added", "touched", "pos", "neg", "zero", "pair_pos", "pair_neg", "pair_zero")


def _split(name: str) -> tuple[str, str]:
    q, c = name.rsplit("-", 1)
    return q, c


@lru_cache(maxsize=None)
def world_questions() -> tuple[tuple, str]:
    """問の並び。data_v8 がそろっていればそれ、無ければ版 7 の 360 問。（問・使った物の説明）"""
    try:
        qs = load_questions(DATA_DIR, SETS)
        if len(qs) == 360 and len({(q.set, q.rule_id) for q in qs}) == 24 and sum(q.should_hold for q in qs) == 72:
            return tuple(qs), "data_v8（版 8 の新しい 360 問）"
    except (OSError, ValueError, KeyError):
        pass
    return tuple(load_questions(DATA_DIR_V7, SETS_V7)), "data（版 7 の 360 問。data_v8 がそろっていないので、内訳が同じ版 7 の並びを使った）"


@lru_cache(maxsize=None)
def _static() -> dict:
    """問の並びから作る、試行で変わらない物（試行ごとに作り直さない）。"""
    qs, source = world_questions()
    block = {q.id: f"{q.set}/{q.rule_id}" for q in qs}
    names = sorted(set(block.values()))
    ids_by_block = {b: [q.id for q in qs if block[q.id] == b] for b in names}
    sets: dict[str, list[str]] = {}
    for b in names:
        sets.setdefault(b.split("/")[0], []).append(b)
    return {"source": source, "ids": [q.id for q in qs], "hold": {q.id: q.should_hold for q in qs},
            "first": {q.id: q.expected == q.options[0] for q in qs},  # 答えられる問で、正解が先頭の選択肢か
            "block": block, "order": {b: i for i, b in enumerate(names)}, "ids_by_block": ids_by_block,
            "sets": [sorted(v) for _, v in sorted(sets.items())]}


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
        ids = w.ids_by_block[b]
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


@lru_cache(maxsize=None)
def _metrics(name: str) -> dict:
    q, c = _split(name)
    x = KLD_BIT[q] * CAL[c]
    return {s: {"kld_mean": x, "ln_ppl_ratio": x} for s in METRICS}


class World:
    def __init__(self, rng: random.Random):
        self.rng = rng
        st = _static()
        self.source, self.ids, self.hold, self.first = st["source"], st["ids"], st["hold"], st["first"]
        self.block, self.order, self.ids_by_block, self.sets = st["block"], st["order"], st["ids_by_block"], st["sets"]

    def draw(self, i: str, pr: tuple[float, float]) -> str:
        """pr = (WA の率, 保留しすぎの率)。"""
        p, over = pr
        r = self.rng.random()
        if r < 0.005:
            return "format"
        if r < 0.005 + p:
            return "wrong_assert"
        if self.hold[i]:
            return "held_ok"
        return "over_hold" if r < 0.005 + p + over else "correct"

    def shared(self, spread) -> dict:
        """5 モデルで分け合う物: 規程の難しさの共通の分・d を足す規程・skew の向き。"""
        return {"diff": {b: self.rng.gauss(0.0, P_SD / math.sqrt(2)) for b in self.order},
                "blocks": self.pick(spread),
                "coins": {b: self.rng.random() < 0.25 for b in self.order}}

    def _rate(self, mean: float, sd: float, hi: float, diff: float | None) -> float:
        x = self.rng.gauss(mean, sd) if diff is None else mean + diff + self.rng.gauss(0.0, sd / math.sqrt(2))
        return min(hi, max(0.0, x))

    def ref(self, diff: dict[str, float] | None = None, kind: str = "strong") -> tuple[dict, dict[str, str]]:
        k = REF_KINDS[kind]
        pa = {b: self._rate(*k["ans"], 0.5, None if diff is None else diff[b]) for b in self.order}
        ph = pa if k["hold"] is None else {b: self._rate(*k["hold"], 0.95, None if diff is None else diff[b])
                                           for b in self.order}
        rate = {i: ((ph if self.hold[i] else pa)[self.block[i]], k["over"]) for i in self.ids}
        return rate, {i: self.draw(i, rate[i]) for i in self.ids}

    def redraw(self, base: dict[str, str], rate: dict, frac: float) -> dict[str, str]:
        return {i: (self.draw(i, rate[i]) if self.rng.random() < frac else o) for i, o in base.items()}

    def pick(self, spread) -> set[str]:
        if spread == SETSPREAD:
            return set(self.rng.choice(self.sets))  # 同じ担当（1 組）の 6 規程
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

    def guess(self, mode: str) -> dict[str, str]:
        """当てずっぽうの 1 回分。choice 0・1 は 2 つの選択肢、2 は保留。"""
        held = {self.rng.choice(ids) for ids in self.ids_by_block.values()} if mode == "one_hold" else set()
        out = {}
        for i in self.ids:
            if mode == "same_word":
                c = 0
            elif mode == "always_hold":
                c = 2
            elif mode == "one_hold":
                c = 2 if i in held else self.rng.randrange(2)
            else:
                c = self.rng.randrange(3)
            if self.hold[i]:
                out[i] = "held_ok" if c == 2 else "wrong_assert"
            elif c == 2:
                out[i] = "over_hold"
            else:
                out[i] = "correct" if (c == 0) == self.first[i] else "wrong_assert"
        return out


def one_model(w: World, scene: str, d: int, spread, shared: dict | None = None,
              info: dict | None = None, kind: str = "strong") -> str:
    """1 モデルを作って判定する。shared があれば規程の難しさ・d の規程・skew の向きをそれに合わせる。kind は基準の作り。
    info を渡すと、狙いの変種に実際に足せた問の数（added）と触れた規程の数（touched）・差が正・負・0 の規程の数・
    判定が有効か（valid）・15 変種の増えの合計（delta_sum）・基準が適格だったとした時の「気づかない」（miss_if_valid）を入れる。"""
    rate, r1 = w.ref(shared["diff"] if shared else None, kind)
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
            info[key + "zero"] = sum(v == 0 for v in diff.values())
    noise = [(RERUN, outs[ANCHOR], w.redraw(outs[ANCHOR], rate, RERUN_NOISE))]
    j = judge_model(r1, r2, variants, w.hold, w.block, noise, strict=True, boot_n=0)
    if info is not None:
        bm = j["by_metric"][PRIMARY]
        info["valid"] = int(j["valid"])
        info["delta_sum"] = sum(j["variants"][n]["delta"] for n in ALL_VARIANTS)
        info["miss_if_valid"] = int(bool(bm["pattern_a"] or bm["pattern_b"]))
    return j["by_metric"][PRIMARY]["verdict"] if j["valid"] else "測れない"


def joint(w: World, scenes: list[tuple[str, int]], spread, kinds: list[str] | None = None) -> str:
    """N 個のモデルを、規程の難しさ・d の規程・skew の向きを分け合って作り、全体の答えを返す。kinds はモデルごとの基準の作り。"""
    sh = w.shared(spread)
    res = {}
    for k, (s, d) in enumerate(scenes):
        v = one_model(w, s, d, spread, sh, kind=kinds[k] if kinds else "strong")
        res[f"m{k}"] = {"valid": v != "測れない", "by_metric": {PRIMARY: {"verdict": v}}}
    return judge_overall(res, PRIMARY)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    den = 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n) - c) / den), min(1.0, (p + z * z / (2 * n) + c) / den)


# ---- 試行ごとの種・並列 ----
def trial_rng(seed: int, uid: str, i: int) -> random.Random:
    """試行 i の乱数。場面（uid）と番号だけで決まるので、並列の数・チャンクの切り方に依らない。"""
    return random.Random(f"{seed}|{uid}|{i}")


def v7_line_ok(q: dict) -> bool:
    return q["correct"] >= V7_MIN_CORRECT and q["held"] >= V7_MIN_HELD and q["format"] <= V7_MAX_FORMAT


def _ref_trial(w: World, params: tuple) -> dict[str, int]:
    """基準 2 回分が線 0 を通るか。all = 2 回とも 0a・0b・0c を全部満たす。fmt・dir・hold = 2 回とも 0a・0b・0c のそれぞれ。
    v7 = 2 回とも版 7 の線（正答 .70・保留 .50・形式 .05）を満たす。"""
    source = params[0]
    if source in REF_KINDS:
        rate, r1 = w.ref(None, source)
        runs = [r1, w.redraw(r1, rate, RERUN_NOISE)]
    elif source in ("same_word", "always_hold"):
        runs = [w.guess(source)] * 2
    else:
        runs = [w.guess(source), w.guess(source)]
    qs = [quality(o, w.hold, w.block) for o in runs]
    f = all(q["format"] <= REF_MAX_FORMAT for q in qs)
    dr = all(q["p_direction"] < REF_ALPHA for q in qs)
    h = all(q["p_hold"] < REF_ALPHA for q in qs)
    return {"all": int(f and dr and h), "fmt": int(f), "dir": int(dr), "hold": int(h), "v7": int(all(v7_line_ok(q) for q in qs))}


def _one_trial(w: World, p: tuple) -> dict[str, int]:
    kind, scene, d, spread = p
    info = {k: 0 for k in INFO_KEYS}
    v = one_model(w, scene, d, spread, info=info, kind=kind)
    out = {f"v:{v}": 1, **{f"t:{k}": info[k] for k in INFO_KEYS}}
    out.update({"all:n": 1, "all:delta": info["delta_sum"], "all:miss": info["miss_if_valid"]})
    if info["valid"]:
        out.update({"pass:n": 1, "pass:delta": info["delta_sum"], "pass:miss": info["miss_if_valid"]})
    return out


def _joint_trial(w: World, p: tuple) -> dict[str, int]:
    scenes, spread, kinds = p
    return {f"o:{joint(w, list(scenes), spread, list(kinds) if kinds else None)}": 1}


def _chunk(job: tuple) -> tuple[str, dict[str, int]]:
    uid, kind, params, lo, hi, seed = job
    fn = {"ref": _ref_trial, "one": _one_trial, "joint": _joint_trial}[kind]
    out: dict[str, int] = {}
    for i in range(lo, hi):
        for k, v in fn(World(trial_rng(seed, uid, i)), params).items():
            out[k] = out.get(k, 0) + v
    return uid, out


def run_units(units: dict[str, tuple[str, tuple, int]], seed: int, jobs: int = 1, chunk: int = 100,
              progress: bool = False) -> dict[str, dict[str, int]]:
    """units: uid → (種類, 引数, 試行の数)。返り値: uid → 整数の合計。jobs > 1 なら multiprocessing で並列に回す。"""
    todo = [(uid, kind, params, lo, min(n, lo + chunk), seed)
            for uid, (kind, params, n) in units.items() for lo in range(0, n, chunk)]
    results: dict[str, dict[str, int]] = {uid: {} for uid in units}
    done = 0

    def add(uid: str, part: dict[str, int]) -> None:
        nonlocal done
        for k, v in part.items():
            results[uid][k] = results[uid].get(k, 0) + v
        done += 1
        if progress and (done % max(1, len(todo) // 10) == 0 or done == len(todo)):
            print(f"  {done}/{len(todo)}", file=sys.stderr, flush=True)

    if jobs <= 1 or len(todo) <= 1:
        for j in todo:
            add(*_chunk(j))
    else:
        with multiprocessing.Pool(min(jobs, len(todo))) as pool:
            for uid, part in pool.imap_unordered(_chunk, todo):
                add(uid, part)
    return results


def _ints(raw: dict[str, int], prefix: str, keys) -> dict[str, int]:
    return {k: raw.get(f"{prefix}{k}", 0) for k in keys}


def ref_rates(source: str, sims: int, seed: int, jobs: int = 1) -> dict[str, int]:
    """基準 2 回分が線 0 を通る回数。source は REF_KINDS の名前か GUESSES の名前（_ref_trial を見よ）。"""
    raw = run_units({f"ref|{source}": ("ref", (source,), sims)}, seed, jobs)[f"ref|{source}"]
    return {k: raw.get(k, 0) for k in ("all", "fmt", "dir", "hold", "v7")}


def one_unit(kind: str, scene: str, d: int, spread) -> tuple[str, tuple]:
    return f"one|{kind}|{scene}|{d}|{spread}", (kind, scene, d, spread)


def summarize_one(raw: dict[str, int], sims: int) -> tuple[dict[str, int], dict[str, float]]:
    cnt = _ints(raw, "v:", VERDICTS)
    return cnt, {k: raw.get(f"t:{k}", 0) / sims for k in INFO_KEYS}


def rates(scene: str, d: int, spread, sims: int, seed: int, kind: str = "strong",
          jobs: int = 1) -> tuple[dict[str, int], dict[str, float]]:
    uid, params = one_unit(kind, scene, d, spread)
    raw = run_units({uid: ("one", params, sims)}, seed, jobs)[uid]
    return summarize_one(raw, sims)


def joint_unit(label: str, scenes, spread, kinds) -> tuple[str, tuple]:
    return f"joint|{label}|{spread}|{'/'.join(kinds or [])}", (tuple(scenes), spread, tuple(kinds or ()))


def joint_rates(label: str, scenes: list[tuple[str, int]], spread, sims: int, seed: int,
                kinds: list[str] | None = None, jobs: int = 1) -> dict[str, int]:
    uid, params = joint_unit(label, scenes, spread, kinds)
    raw = run_units({uid: ("joint", params, sims)}, seed, jobs)[uid]
    return _ints(raw, "o:", OVERALL)


def combine(probs: list[dict[str, float]]) -> dict[str, float]:
    """モデルそれぞれの判定の割合から、全体の答えの割合（独立とみなす）。"""
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


def joint_scenes(n: int = N_MODELS, ks: tuple[int, ...] = (1, 2, 3, 4, 5)) -> list[tuple[str, list[tuple[str, int]], object]]:
    """（名前・n モデルの場面・広さ）。見逃すモデルの数 k は 0（notice）・ks（1〜5）。見逃すモデルは先頭から k 個。"""
    out = [("notice k=0", [("notice", 0)] * n, 24), ("null", [("null", 0)] * n, 24),
           ("skew×5（向きをそろえる）", [("skew", 0)] * n, 24),
           ("skew_rev×5（向きをそろえる）", [("skew_rev", 0)] * n, 24)]
    for s in ("miss_A", "miss_B"):
        for d in (10, 20, 30):
            for k in ks:
                out.append((f"{s} d={d} k={k}", [(s, d)] * k + [("notice", 0)] * (n - k), 24))
    for s, d in (("miss_A", 20), ("miss_B", 30)):
        for k in (2, 3, 5):
            out.append((f"{s} d={d} k={k} 8 規程", [(s, d)] * k + [("notice", 0)] * (n - k), 8))
    for k in (1, 2, 3, 5):
        out.append((f"miss_A d=20 k={k} 同じ担当の 6 規程", [("miss_A", 20)] * k + [("notice", 0)] * (n - k), SETSPREAD))
    return out


# 5 モデルの組み立て: 見逃すモデルを先頭から割り当てるので、弱い基準を先に並べる（見つけにくい向き）
KIND_SETS = {"強い基準 5 つ": ["strong"] * N_MODELS, "弱い基準 4 つ・強い基準 1 つ": ["weak"] * 4 + ["strong"]}


def one_scenes() -> list[tuple[str, int, object]]:
    scenes = [("notice", 0, 24), ("null", 0, 24), ("skew", 0, 24), ("skew_rev", 0, 24)]
    scenes += [(s, d, sp) for s in ("miss_A", "miss_B") for sp in (24, 8) for d in (10, 20, 30)]
    scenes += [("miss_A", 20, SETSPREAD)]
    return scenes


def _spread_label(sp) -> str:
    return "同じ担当の 6 規程" if sp == SETSPREAD else str(sp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000, help="1 モデルの見積りの回数（本番は 10000）")
    ap.add_argument("--joint-sims", type=int, default=4000,
                    help="5 モデル一緒の見積りの回数（本番は 4000。0 で出さない）")
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 2),
                    help="並列の数（既定は コア数 − 2）。結果は並列の数で変わらない")
    a = ap.parse_args(argv)
    t0 = time.time()
    js, n, seed, jobs = a.joint_sims, a.sims, a.seed, a.jobs
    st = _static()
    print(f"# 量子化の物差し: 見積りの出力（判定 {JUDGE_VERSION}・測る前）\n")
    print(f"`python -m evaluation.quant_probe.power --sims {n} --joint-sims {js} --seed {seed}` の出力"
          f"（本番の回数は --sims 10000 --joint-sims 4000。この出力は {n}・{js} 回）。"
          "答えは作り物で、本物の答えも KLD も使っていない。読み方は docs/quant-probe-plan-v8.md の 7 章。\n")
    print(f"- 回数: 1 モデル {n} 回・5 モデル一緒 {js} 回（試行ごとに種を決めるので、並列の数 {jobs} で結果は変わらない）")
    print(f"- 問の並び: {st['source']}（{len(st['ids'])} 問・規程 {len(st['order'])}・保留すべき問 {sum(st['hold'].values())}）")
    print(f"- 1 モデルの α {ALPHA_MODEL}・モデル数 {N_MODELS}・線 0 の p < {REF_ALPHA}・形式違反 <= {REF_MAX_FORMAT}・"
          f"版 7 の線 正答 >= {V7_MIN_CORRECT}・保留 >= {V7_MIN_HELD}・形式違反 <= {V7_MAX_FORMAT}\n")

    # 1. 線 0
    sources = [("strong", "強い基準（WA 10%・保留しすぎ 3%）"), ("weak", "弱い基準（正答 .65・保留しすぎ .15・保留 .47）"),
               *((g, f"当てずっぽう: {t}") for g, t in GUESSES.items())]
    raw = run_units({f"ref|{s}": ("ref", (s,), n) for s, _ in sources}, seed, jobs, chunk=500)
    print(f"## 1. 基準が線 0 を通る割合（基準 2 回とも・{n} 回・括弧は Wilson 95%）\n")
    print("| 基準の作り | 線 0 を全部通る | 0a 形式だけ | 0b 向きだけ | 0c 保留だけ | 版 7 の線を通る |")
    print("|---|---|---|---|---|---|")
    for s, label in sources:
        print(f"| {label} | " + _cells({k: raw[f"ref|{s}"].get(k, 0) for k in ("all", "fmt", "dir", "hold", "v7")},
                                       ("all", "fmt", "dir", "hold", "v7"), n) + " |", flush=True)

    # 2. 1 モデル
    scenes = one_scenes()
    units = dict(one_unit(kind, s, d, sp) for kind in REF_KINDS for s, d, sp in scenes)
    raw = run_units({u: ("one", p, n) for u, p in units.items()}, seed, jobs, progress=True)
    table = {}
    for kind, title in (("strong", "強い基準"), ("weak", "弱い基準")):
        print(f"\n## 2. 1 モデルの判定の割合（{title}・主の物差し {PRIMARY}・{n} 回・括弧は Wilson 95%）\n")
        print("足した問・触れた規程・差が正／負／0 の規程は狙いの変種（skew は Q5_K_M-ja）の平均。差は規程ごとの WA の差で、"
              "基準 1 回目との差と、同じ bit の en との差。「測れない」は基準が線 0 を割った回。\n")
        print("| 場面 | 増え d | 広さ | 足した問 | 触れた規程 | 基準との差 正／負／0 | en との差 正／負／0 | " + " | ".join(VERDICTS) + " |")
        print("|---|---|---|---|---|---|---|" + "---|" * len(VERDICTS))
        for s, d, sp in scenes:
            uid, _ = one_unit(kind, s, d, sp)
            cnt, info = summarize_one(raw[uid], n)
            table[(kind, s, d, sp)] = {v: c / n for v, c in cnt.items()}
            miss = s.startswith("miss")
            print(f"| {s} | {d or '-'} | {_spread_label(sp) if miss else '-'} | {info['added']:.1f} | {info['touched']:.1f} | "
                  f"{info['pos']:.1f}／{info['neg']:.1f}／{info['zero']:.1f} | "
                  f"{info['pair_pos']:.1f}／{info['pair_neg']:.1f}／{info['pair_zero']:.1f} | " + _cells(cnt, VERDICTS, n) + " |")

    # 2b. 本当の差が 0 の場面: 線 0 を通った試行だけに絞ると、増えと誤りの率はどう変わるか
    print(f"\n## 2b. 本当の差が 0 の場面（null）で、線 0 を通った試行に絞った時（{n} 回・括弧は Wilson 95%）\n")
    print("増え Δ の平均は 15 変種の、基準 1 回目との WA の差の合計の平均（本当の差が 0 なので、0 から離れるほど絞り込みの偏り）。"
          "誤った「気づかない」の率は、その試行の型 A か型 B が立った割合"
          "（線 0 を通らなかった試行は、基準が適格だったとして判定した時の値。通った試行では実際の判定）。\n")
    print("| 基準 | 試行 | 試行の数 | 変種の増え Δ の平均 | 誤った「気づかない」の率 |")
    print("|---|---|---|---|---|")
    for kind, title in (("strong", "強い基準"), ("weak", "弱い基準")):
        r = raw[one_unit(kind, "null", 0, 24)[0]]
        for grp, label in (("all", "全部の試行"), ("pass", "線 0 を通った試行")):
            m = r.get(f"{grp}:n", 0)
            lo, hi = wilson(r.get(f"{grp}:miss", 0), m)
            delta = r.get(f"{grp}:delta", 0) / (m * len(ALL_VARIANTS)) if m else float("nan")
            rate = r.get(f"{grp}:miss", 0) / m if m else float("nan")
            print(f"| {title} | {label} | {m} | {delta:+.3f} | {rate:.3f} ({lo:.3f}-{hi:.3f}) |")

    # 3. 独立の数え上げ
    print(f"\n## 3. {N_MODELS} モデルの全体の答えの割合（独立とみなした数え上げ。見逃すモデルの数 k = 0〜{N_MODELS}・"
          "見逃すモデルは 24 規程。見逃すモデルは並びの先頭から）")
    for label, kinds in KIND_SETS.items():
        print(f"\n{label}\n")
        print("| 見逃しの型 | d | k | " + " | ".join(OVERALL) + " |")
        print("|---|---|---|" + "---|" * len(OVERALL))
        for s in ("miss_A", "miss_B"):
            for d in (10, 20, 30):
                for k in range(N_MODELS + 1):
                    probs = [table[(kinds[m], s if m < k else "notice", d if m < k else 0, 24)] for m in range(N_MODELS)]
                    o = combine(probs)
                    print(f"| {s} | {d} | {k} | " + " | ".join(f"{o[x]:.3f}" for x in OVERALL) + " |")
        o = combine([table[(kinds[m], "null", 0, 24)] for m in range(N_MODELS)])
        print("| null（全部増えなし） | - | 0 | " + " | ".join(f"{o[x]:.3f}" for x in OVERALL) + " |")

    # 4. 一緒に作る
    if js > 0:
        jscenes = joint_scenes()
        units = {}
        for label, kinds in KIND_SETS.items():
            for lab, sc, sp in jscenes:
                uid, p = joint_unit(lab, sc, sp, kinds)
                units[uid] = ("joint", p, js)
        raw = run_units(units, seed, jobs, chunk=50, progress=True)
        for label, kinds in KIND_SETS.items():
            print(f"\n## 4. {N_MODELS} モデルを一緒に作った全体の答えの割合（{label}・規程の難しさ・d の規程・skew の向きを分け合う・"
                  f"{js} 回・括弧は Wilson 95%）\n")
            print("| 場面 | " + " | ".join(OVERALL) + " |")
            print("|---|" + "---|" * len(OVERALL))
            for lab, sc, sp in jscenes:
                cnt = _ints(raw[joint_unit(lab, sc, sp, kinds)[0]], "o:", OVERALL)
                print(f"| {lab} | " + _cells(cnt, OVERALL, js) + " |")
    print(f"\nかかった秒: {time.time() - t0:.0f}（並列 {jobs}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
