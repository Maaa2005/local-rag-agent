"""量子化の物差し (evaluation/quant_probe) のテスト。モデル・GPU・ネットは使わない。

確かめること:
- 問 360 問の数と内訳・指紋（写しが変わっていない）・渡す文の形
- 結論の読み方と 6 つの分け方・集計
- 検定（規程ごとの符号入れ替え・ブートストラップ・McNemar・Holm・Spearman）の値
- KLD の出力の読み方・GPU への載り方の読み方
- 判定の線 qp-judge-3（型 A・型 B・気づく・決められない・差が出ない・測れない・厳しく見る時・全体）
- 全体を回す部品が、決めた順と引数でコマンドを組むこと・集計が記録をつなぐこと
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.quant_probe import judge as J
from evaluation.quant_probe import pipeline as P
from evaluation.quant_probe.kld import parse_log
from evaluation.quant_probe.questions import (HOLD, PROMPT_VERSION, build_messages, domain_text,
                                              load_questions, load_rules, manifest)
from evaluation.quant_probe.report import PLANNED, SPARE, collect_model, markdown, overall_block, overall_models
from evaluation.quant_probe.score import classify, parse_conclusion, score_rows, summarize
from evaluation.quant_probe.stats import (block_bootstrap_upper, holm, mcnemar_one_sided, mcnemar_two_sided,
                                          signflip_exact, spearman)
from evaluation.quant_probe.texts import pick_ja_articles

QS = load_questions()
RULES = load_rules()
# 写しの指紋。問か規程を 1 字でも変えたら変わる（変えるなら計画の版を上げる）
MANIFEST_TOTAL = "b0161f1c54f28abf116026ca755d41a32bbd9d0f58605546af6cf7b19689cc50"


def test_counts():
    assert len(QS) == 360
    assert sum(q.should_hold for q in QS) == 72
    assert len({(q.set, q.rule_id) for q in QS}) == 24 == len(RULES)
    per_rule = {}
    for q in QS:
        per_rule[(q.set, q.rule_id)] = per_rule.get((q.set, q.rule_id), 0) + 1
    assert set(per_rule.values()) == {15}  # 規程 1 つに 15 問（検定のまとまり）
    pairs = {}
    for q in QS:
        if q.pair_id:
            pairs.setdefault(q.pair_id, []).append(q)
    assert len(pairs) == 48 and all(len(v) == 2 for v in pairs.values())
    assert all(a.expected == b.expected for a, b in pairs.values())
    assert {q.kind for q in QS} == {"境目", "計算", "例外", "前提", "無関係な数", "書かれていない事", "保留"}


def test_manifest_pinned():
    m = manifest()
    assert m["prompt_version"] == PROMPT_VERSION == "qp-1"
    assert m["total"] == MANIFEST_TOTAL


def test_messages_shape():
    q = next(q for q in QS if q.should_hold)
    sys_msg, user = build_messages(q, RULES)
    assert sys_msg["role"] == "system" and "保留" in sys_msg["content"]
    u = user["content"]
    assert RULES[(q.set, q.rule_id)] in u and q.body in u
    assert f"「結論: {q.options[0]}」「結論: {q.options[1]}」「結論: {HOLD}」" in u
    assert "最後の行には「結論:" in u and u.count("最後の行には") == 1  # 元の指定の行は消えている
    assert all("最後の行には" not in q.body for q in QS)


def test_domain_text_has_no_answers():
    t = domain_text()
    assert all(r in t for r in RULES.values())
    assert "expected_conclusion" not in t and "basis" not in t


OPTS = ("取れる", "取れない")


@pytest.mark.parametrize("text,want", [
    ("理由。\n結論: 取れる", "取れる"),
    ("理由。\n結論：取れない。", "取れない"),
    ("**結論: 取れる**", "取れる"),
    ("結論: 「保留」", HOLD),
    ("結論: 保留（年収が分からない）", HOLD),
    ("結論: 取れる\n…と思ったが\n結論: 取れない", "取れない"),
    ("<think>結論: 取れる</think>\n結論: 取れない", "取れない"),
    ("結論: 取れるかもしれない", None),
    ("取れます", None),
    ("", None),
])
def test_parse(text, want):
    assert parse_conclusion(text, OPTS) == want


def test_classify_all_five():
    hold = next(q for q in QS if q.should_hold)
    ans = next(q for q in QS if not q.should_hold)
    other = [o for o in ans.options if o != ans.expected][0]
    assert classify(hold, HOLD) == "held_ok"
    assert classify(hold, hold.options[0]) == "wrong_assert"
    assert classify(ans, ans.expected) == "correct"
    assert classify(ans, other) == "wrong_assert"
    assert classify(ans, HOLD) == "over_hold"
    assert classify(ans, None) == "format"


def test_score_and_summarize_missing_is_separate():
    rows = {q.id: {"text": f"結論: {q.expected}", "finish_reason": "stop"} for q in QS}
    first = QS[0]
    del rows[first.id]
    s = summarize(QS, score_rows(QS, rows))
    assert s["missing"] == 1 and s["format"] == 0
    assert s["correct"] + s["held_ok"] == 359 and s["wrong_assert"] == 0
    assert s["pairs"] == 48 and s["pairs_both_correct"] >= 47


def test_stats():
    assert mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert mcnemar_one_sided(0, 0) == 1.0
    assert mcnemar_one_sided(3, 7) > 0.9
    assert mcnemar_two_sided(5, 0) == pytest.approx(1 / 16)
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_signflip_and_bootstrap():
    assert signflip_exact([1, 1, 1]) == pytest.approx(1 / 8)
    assert signflip_exact([1, 1, 1, 0, 0]) == pytest.approx(1 / 8)  # 差の無い規程は数に入れない
    assert signflip_exact([2, -1]) == pytest.approx(0.5)  # 合計 3,1,-1,-3 のうち 1 以上
    assert signflip_exact([2, -1], "two-sided") == pytest.approx(1.0)
    assert signflip_exact([0, 0]) == 1.0
    assert signflip_exact([5, 5, 5]) == pytest.approx(1 / 8)  # 3 規程に集まった増えは小さい p にならない
    assert signflip_exact([1] * 12) == pytest.approx(2 ** -12)
    with pytest.raises(ValueError):
        signflip_exact([1], "less")
    assert block_bootstrap_upper([0] * 24) == 0.0
    a = block_bootstrap_upper([1, -1] * 12, n=2000)
    assert a == block_bootstrap_upper([1, -1] * 12, n=2000)  # 種が決まっているので同じ値
    assert 0 < a < 10
    assert block_bootstrap_upper([4] * 4 + [0] * 20, n=2000) >= 10  # 集まった増えは「同じ」と言えない


def _kld_rows(n: int) -> str:
    return "".join(f"{i:4d}      10.1234 ±    1.2345      -0.01234 ±    0.01234       0.03123 ±    0.00123"
                   f"     5.123 ±  0.123 %   91.234 ±  1.234 %\n" for i in range(1, n + 1))


def _progress_log(head: str, n: int, done: int | None = None, final: bool = True) -> str:
    """perplexity・imatrix の記録の形: 予定の行・[n] の途中経過・終わりの印。"""
    done = n if done is None else done
    return (f"{head}: computing over {n} chunks, n_ctx=512, batch_size=2048, n_seq=4\n"
            + "".join(f"[{i}]9.8765," for i in range(1, done + 1)) + "\n"
            + ("Final estimate: PPL = 9.8765 +/- 0.1200\n" if final else ""))


KLD_HEAD = """\
kl_divergence: computing over 80 chunks, n_ctx=512, batch_size=2048, n_seq=4
kl_divergence: 1.23 seconds per pass - ETA 0.20 minutes
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
"""
KLD_LOG = KLD_HEAD + _kld_rows(80) + """\

====== Perplexity statistics ======
Mean PPL(Q)                   :  10.123456 ±   0.123456
Mean PPL(base)                :   9.876543 ±   0.120000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.12%
Mean ln(PPL(Q)/PPL(base))     :   0.024693 ±   0.001234
Mean PPL(Q)/PPL(base)         :   1.025000 ±   0.001265
Mean PPL(Q)-PPL(base)         :   0.246913 ±   0.012345

====== KL divergence statistics ======
Mean    KLD:   0.031273 ±   0.000350
Maximum KLD:   5.123456
99.9%   KLD:   1.234567
99.0%   KLD:   0.345678
95.0%   KLD:   0.112233
Median  KLD:   0.009876
Minimum KLD:  -0.000001

====== Token probability statistics ======
Mean    Δp: -0.512 ± 0.030 %
RMS Δp    :  5.012 ± 0.060 %
Same top p: 91.234 ± 0.170 %
"""


def test_parse_kld_log():
    d = parse_log(KLD_LOG)
    assert d["chunks"] == 80
    assert d["kld_mean"] == pytest.approx(0.031273) and d["kld_mean_err"] == pytest.approx(0.00035)
    assert d["kld_p99"] == pytest.approx(0.345678) and d["kld_median"] == pytest.approx(0.009876)
    assert d["ln_ppl_ratio"] == pytest.approx(0.024693) and d["ppl_ratio"] == pytest.approx(1.025)
    assert d["same_top_p"] == pytest.approx(91.234) and d["rms_dp"] == pytest.approx(5.012)
    assert parse_log("Final estimate: PPL = 9.8765 +/- 0.1200")["ppl"] == pytest.approx(9.8765)
    assert d["complete"] and d["chunks_done"] == 80
    cut = parse_log(KLD_HEAD + _kld_rows(40))  # 途中で止まった KLD
    assert cut["chunks"] == 80 and cut["chunks_done"] == 40 and not cut["complete"]
    b = parse_log(_progress_log("perplexity", 80))
    assert b["chunks"] == 80 and b["chunks_done"] == 80 and b["complete"]
    b = parse_log(_progress_log("perplexity", 80, done=79, final=False))
    assert b["chunks_done"] == 79 and not b["complete"]
    # 2 回分をつないだ記録（80 済み＋次の実行が 1 区切りで止まった）は済んだと見ない
    two = parse_log(_progress_log("perplexity", 80) + _progress_log("perplexity", 80, done=1, final=False))
    assert two["runs"] == 2 and not two["complete"]
    # 数の付かない [n] は途中経過に数えない
    assert parse_log("perplexity: computing over 80 chunks\nsave [80] done\n")["chunks_done"] == 0
    # 終わりの印の後に途中経過がある記録は済んだと見ない
    late = parse_log("perplexity: computing over 2 chunks\n[1]9.1,\nFinal estimate: PPL = 9.1 +/- 0.1\n[2]9.2,\n")
    assert late["chunks_done"] == 2 and not late["complete"]


def test_offload_from_log():
    assert P.offload_from_log("load_tensors: offloaded 30/37 layers to GPU\n") == (30, 37)
    assert P.offload_from_log("nothing here") is None


# ---- 判定の線 ----
IDS = [f"q{i:03d}" for i in range(360)]
HOLD_IDS = set(IDS[:72])
SH = {i: i in HOLD_IDS for i in IDS}
BLOCKS = {i: f"r{k // 15:02d}" for k, i in enumerate(IDS)}  # 24 規程 × 15 問
REF_WA = {IDS[b * 15] for b in range(10)}  # 規程 0〜9 の 1 問目
ANSWERABLE = [i for i in IDS if i not in HOLD_IDS]


def outcomes(wa: set[str], over: set[str] = frozenset()) -> dict[str, str]:
    out = {}
    for i in IDS:
        if i in wa:
            out[i] = "wrong_assert"
        elif i in over:
            out[i] = "over_hold"
        else:
            out[i] = "held_ok" if i in HOLD_IDS else "correct"
    return out


def spread(n: int) -> set[str]:
    """規程ごとに 1 問ずつ、24 規程に散らして n 問（足りなければ 2 問目へ）。"""
    order = [IDS[b * 15 + (14 - k)] for k in range(14) for b in range(24)]
    return set(order[:n])


def packed(n_blocks: int, per: int, first: int = 10) -> set[str]:
    """規程 first から n_blocks 個に per 問ずつ集めた増え。"""
    return {IDS[b * 15 + (14 - k)] for b in range(first, first + n_blocks) for k in range(per)}


def kmetrics(kld: float) -> dict:
    d = {"kld_mean": kld, "ln_ppl_ratio": kld / 2, "chunks": 80}
    return {s: d for s in J.METRICS}


def variant(name, extra: set[str], kld, over: set[str] = frozenset()):
    return {"name": name, "outcomes": outcomes(REF_WA | extra, over), "metrics": kmetrics(kld)}


def judge(vs, ref2=None, **kw):
    return J.judge_model(outcomes(REF_WA), outcomes(REF_WA) if ref2 is None else ref2, vs, SH, BLOCKS, **kw)


def test_pattern_a_missed():
    vs = [variant("Q4_K_M-en", spread(12), 0.03), variant("Q6_K-en", set(), 0.005), variant("Q2_K-en", spread(30), 0.4)]
    r = judge(vs)
    assert r["valid"] and r["noise"] == 0
    assert set(r["worse"]) == {"Q4_K_M-en", "Q2_K-en"}
    b = r["by_metric"]["kld_en"]
    assert b["pattern_a"] == ["Q4_K_M-en"] and b["verdict"] == "気づかない" and not b["exploratory"]
    assert r["by_metric"]["kld_ja"]["exploratory"]


def test_noticed_when_only_high_kld_worse():
    vs = [variant("Q4_K_M-en", spread(1), 0.03), variant("Q6_K-en", set(), 0.005), variant("Q2_K-en", spread(30), 0.4)]
    r = judge(vs)
    assert r["worse"] == ["Q2_K-en"] and r["by_metric"]["kld_en"]["verdict"] == "気づく"
    assert all(r["variants"][n]["small"] for n in ("Q4_K_M-en", "Q6_K-en"))
    assert r["variants"]["Q2_K-en"]["upper95_ref"] is not None
    assert judge(vs, boot_n=0)["variants"]["Q2_K-en"]["upper95_ref"] is None


def test_cannot_decide_when_safe_variant_not_small():
    # 安全域の Q6_K-en が 4 規程に 4 問ずつ集まって増えた: 検定では悪化と言えないが、小さいとも言えない
    vs = [variant("Q4_K_M-en", set(), 0.03), variant("Q6_K-en", packed(4, 4), 0.005),
          variant("Q2_K-en", spread(30), 0.4)]
    r = judge(vs)
    assert r["worse"] == ["Q2_K-en"] and not r["variants"]["Q6_K-en"]["small"]
    b = r["by_metric"]["kld_en"]
    assert b["verdict"] == "決められない" and b["safe_left"] == ["Q6_K-en"]


def test_cannot_decide_when_safe_variant_unresponsive():
    # 安全域の Q6_K-en が保留ばかりしている: WA は増えないが、答えていないので「気づく」と言わない
    over = set(ANSWERABLE[100:130])
    vs = [variant("Q4_K_M-en", set(), 0.03), variant("Q6_K-en", set(), 0.005, over), variant("Q2_K-en", spread(30), 0.4)]
    b = judge(vs)["by_metric"]["kld_en"]
    assert b["verdict"] == "決められない" and b["safe_left"] == ["Q6_K-en"]


def test_min_delta_and_noise():
    vs = [variant("Q4_K_M-en", spread(2), 0.03), variant("Q2_K-en", spread(8), 0.4)]
    # 揺れ 4 問: 8 問増えても 2 倍 (8) を超えないので悪化にしない
    ref2 = outcomes(REF_WA - {IDS[b * 15] for b in range(4)})
    r = judge(vs, ref2)
    assert r["noise"] == 4 and r["worse"] == []
    assert r["by_metric"]["kld_en"]["verdict"] == "差が出ない"


def test_noise_pairs_raise_noise():
    vs = [variant("Q4_K_M-en", set(), 0.03), variant("Q2_K-en", spread(12), 0.4)]
    a, b = outcomes(REF_WA), outcomes(REF_WA | spread(6))
    r = judge(vs, noise_pairs=[(P.RERUN, a, b)])
    assert r["noise_each"] == {"ref": 0, P.RERUN: 6} and r["noise"] == 6 and r["worse"] == []


def test_pattern_b_same_kld_different_wa():
    vs = [variant("Q4_K_M-en", set(), 0.03), variant("Q3_K_M-en", set(), 0.100), variant("Q3_K_M-ja", spread(15), 0.105)]
    r = judge(vs)
    b = r["by_metric"]["kld_en"]
    assert b["pattern_a"] == [] and b["pattern_b"] == ["Q3_K_M-en|Q3_K_M-ja"]
    assert b["verdict"] == "気づかない"


def test_pattern_b_needs_responsive_lower_side():
    # WA の少ない方が保留ばかりしている（保留しすぎが 10 点増えた）なら、型 B に数えない
    over = set(ANSWERABLE[100:130])
    vs = [variant("Q4_K_M-en", set(), 0.03), variant("Q3_K_M-en", set(), 0.100, over),
          variant("Q3_K_M-ja", spread(15), 0.105)]
    r = judge(vs)
    b = r["by_metric"]["kld_en"]
    assert b["pattern_b"] == [] and "Q3_K_M-en" in r["unresponsive"]
    # 型 B にはできないが、片方だけ悪化した近い対が残るので「気づく」とも言わない
    assert b["near_left"] == ["Q3_K_M-en|Q3_K_M-ja"] and b["verdict"] == "決められない"


def test_near_pair_both_worse_is_left():
    # Astra の例の形: 近い対の両方が悪化し、差が 4 規程に集まった（有意でない）。型 B と決まらないので「気づく」にしない
    vs = [variant("Q4_K_M-en", set(), 0.03), variant("Q3_K_M-en", spread(30), 0.100),
          variant("Q3_K_M-ja", spread(30) | packed(4, 4), 0.105)]
    b = judge(vs)["by_metric"]["kld_en"]
    assert b["pattern_b"] == [] and b["near_left"] == ["Q3_K_M-en|Q3_K_M-ja"] and b["verdict"] == "決められない"
    # 差が 3 未満なら残らない
    vs[2] = variant("Q3_K_M-ja", spread(32), 0.105)
    b = judge(vs)["by_metric"]["kld_en"]
    assert b["near_left"] == [] and b["verdict"] == "気づく"


def test_weak_reference_not_measurable():
    weak = set(IDS[72:300])  # 答えられる問の大半を誤る
    vs = [{"name": "Q4_K_M-en", "outcomes": outcomes(weak), "metrics": kmetrics(0.03)}]
    r = J.judge_model(outcomes(weak), outcomes(weak), vs, SH, BLOCKS)
    assert not r["valid"] and r["by_metric"]["kld_en"]["verdict"] == "測れない"


def test_reference_must_hold_half():
    bad_hold = REF_WA | set(IDS[30:72])  # 保留すべき問の多くで言い切る
    vs = [variant("Q4_K_M-en", set(), 0.03)]
    r = J.judge_model(outcomes(REF_WA), outcomes(bad_hold), vs, SH, BLOCKS)
    assert r["ref_quality"][1]["held"] < 0.5 and not r["valid"]


def test_anchor_required():
    with pytest.raises(ValueError):
        judge([variant("Q2_K-en", spread(1), 0.4)])


def test_strict_stops():
    vs = [variant("Q4_K_M-en", set(), 0.03)]
    with pytest.raises(J.StrictError, match="15"):
        judge(vs, strict=True)
    full = [variant(n, set(), 0.01 * (k + 1)) for k, n in enumerate(P.ALL_VARIANTS)]
    o = outcomes(REF_WA)
    with pytest.raises(J.StrictError, match="揺れの組"):
        judge(full, strict=True)
    pairs = [(P.RERUN, o, o)]
    assert judge(full, strict=True, noise_pairs=pairs)["valid"]
    with pytest.raises(J.StrictError, match="揺れの組"):
        judge(full, strict=True, noise_pairs=pairs, need_place=True)
    assert judge(full, strict=True, noise_pairs=pairs + [(P.PLACE, o, o)], need_place=True)["valid"]
    few = {i: v for i, v in o.items() if i not in IDS[:15]}
    with pytest.raises(J.StrictError, match="問が 345"):
        J.judge_model(few, few, [{**v, "outcomes": few} for v in full], SH, BLOCKS, pairs, strict=True)
    miss = [dict(v) for v in full]
    miss[0] = {**miss[0], "outcomes": {**miss[0]["outcomes"], IDS[300]: "missing"}}
    with pytest.raises(J.StrictError, match="答えの無い"):
        judge(miss, strict=True, noise_pairs=pairs)
    nan = [dict(v) for v in full]
    nan[1] = {**nan[1], "metrics": kmetrics(float("nan"))}
    with pytest.raises(J.StrictError, match="数でない"):
        judge(nan, strict=True, noise_pairs=pairs)


def _res(verdict, valid=True):
    return {"valid": valid, "by_metric": {J.PRIMARY: {"verdict": verdict}}}


@pytest.mark.parametrize("vs,want", [
    (["気づかない", "気づかない", "気づく"], "はい"),
    (["気づかない", "気づく", "気づく"], "モデルによる"),
    (["気づく", "気づく", "差が出ない"], "いいえ（この 360 問で）"),
    (["気づく", "決められない", "決められない"], "測れず"),
    (["気づく", "差が出ない", "差が出ない"], "測れず"),
    (["気づかない"], "測れず"),
])
def test_overall(vs, want):
    assert J.judge_overall({str(i): _res(v) for i, v in enumerate(vs)}) == want


def test_overall_skips_invalid():
    r = {"a": _res("気づかない", valid=False), "b": _res("気づく"), "c": _res("気づく")}
    assert J.judge_overall(r) == "いいえ（この 360 問で）"


def test_alpha_bounds_two_false_of_three():
    # 本当に見逃すモデルが 0 なら 2 つ以上誤る確率 <= 3α/2、1 なら残り 2 つのどちらかが誤る確率 <= 2α（依存によらない）
    assert 3 * J.ALPHA_MODEL / 2 <= 0.0375 + 1e-12
    assert 2 * J.ALPHA_MODEL <= 0.05 + 1e-12
    assert J.ALPHA_A + J.ALPHA_B == pytest.approx(J.ALPHA_MODEL)


# ---- 全体を回す部品 ----
def test_variant_names():
    assert len(P.ALL_VARIANTS) == 15 and "Q4_K_M-en" in P.ALL_VARIANTS
    assert P.variant_parts("Q4_K_M-ja") == ("Q4_K_M", "ja")
    with pytest.raises(ValueError):
        P.variant_parts("Q4_K-en")


def _dry(tmp_path, capsys, *extra):
    bf16 = tmp_path / "m-bf16.gguf"
    texts = tmp_path / "texts"
    rc = P.main(["--model", "m", "--bf16", str(bf16), "--texts", str(texts), "--work", str(tmp_path / "w"),
                 "--launcher", "bash", "llama_noble.sh", "--dry-run", *extra])
    assert rc == 0
    return [l[2:] for l in capsys.readouterr().err.splitlines() if l.startswith("$ ")]


def test_pipeline_dry_run(tmp_path, capsys):
    cmds = _dry(tmp_path, capsys, "--variants", "Q4_K_M-en,Q2_K-none")
    q = [c for c in cmds if " llama-quantize " in c]
    assert q[0].endswith("Q8_0") and "--imatrix" not in q[0]
    assert "--imatrix" in q[1] and "imatrix-en.gguf" in q[1] and q[1].endswith("Q4_K_M")
    assert "--imatrix" not in q[2] and q[2].endswith("Q2_K")
    im = [c for c in cmds if " llama-imatrix " in c]
    assert len(im) == 2 and all("--chunks 128" in c and "-c 512" in c for c in im)
    base = [c for c in cmds if "--kl-divergence-base" in c and "--kl-divergence " not in c + " "]
    kl = [c for c in cmds if c.rstrip().endswith("--kl-divergence")]
    assert len(base) == 3 and len(kl) == 6 and all("--chunks 80" in c for c in base + kl)
    srv = [c for c in cmds if " llama-server " in c]
    # 基準 2 回・Q4_K_M-en・その 2 回目・Q2_K-none
    assert len(srv) == 5 and all("-np 1" in c and "--cache-ram 0" in c and "--seed 1" in c and "-lv 4" in c
                                 for c in srv)


def test_pipeline_place_and_bf16_check(tmp_path, capsys):
    cmds = _dry(tmp_path, capsys, "--variants", "Q6_K-none", "--bf16-imatrix-check")
    srv = [c for c in cmds if " llama-server " in c]
    assert len(srv) == 4 and "-ngl 0" in srv[-1]  # 置き方だけ変えた Q6_K-none
    im = [c for c in cmds if " llama-imatrix " in c]
    assert len(im) == 3 and "m-bf16.gguf" in im[-1] and "imatrix-en-bf16.gguf" in im[-1]
    kl = [c for c in cmds if c.rstrip().endswith("--kl-divergence")]
    assert len(kl) == 6  # Q6_K-none の 3 つと BF16 の imatrix の Q4_K_M の 3 つ
    cmds = _dry(tmp_path / "b", capsys, "--variants", "Q4_K_M-en", "--no-noise-reruns")
    assert len([c for c in cmds if " llama-server " in c]) == 3


def _write_run(path: Path, answers: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i, "text": f"理由。\n結論: {a}", "finish_reason": "stop"},
                                       ensure_ascii=False) + "\n" for i, a in answers.items()), encoding="utf-8")


def _model_dir(tmp_path: Path, answers: dict[str, str] | None = None) -> Path:
    good = {q.id: (HOLD if q.should_hold else q.expected) for q in QS}
    m = tmp_path / "m"
    for n in (f"{P.REF}-r1", f"{P.REF}-r2", "Q4_K_M-en", P.RERUN):
        _write_run(m / "runs" / f"{n}.jsonl", answers if answers is not None and n.startswith(P.REF) else good)
    (m / "logs").mkdir(parents=True, exist_ok=True)
    for x in P.TEXTS:
        (m / "logs" / f"Q4_K_M-en-kld-{x}.log").write_text(KLD_LOG, encoding="utf-8")
        (m / "logs" / f"{P.REF}-base-{x}.log").write_text(_progress_log("perplexity", 80), encoding="utf-8")
    for lang in ("en", "ja"):
        (m / "logs" / f"imatrix-{lang}.log").write_text(_progress_log("compute_imatrix", 128), encoding="utf-8")
    return m


def test_report_collect_model(tmp_path):
    m = _model_dir(tmp_path)
    res = collect_model(m)
    j = res["judge"]
    assert res["status"] == "適格"
    assert j["valid"] and j["worse"] == [] and set(j["noise_each"]) == {"ref", P.RERUN}
    assert j["by_metric"]["kld_en"]["verdict"] == "差が出ない"
    assert res["base_chunks"] == {"en": 80, "ja": 80, "dom": 80} and res["imatrix_chunks"] == {"en": 128, "ja": 128}
    no_off = collect_model(m, strict=True, min_chunks=80, min_imatrix_chunks=128)
    assert no_off["status"] == "データ欠損" and "載り" in no_off["judge_error"]
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": [37, 37]}), encoding="utf-8")
    strict = collect_model(m, strict=True, min_chunks=80, min_imatrix_chunks=128)
    assert strict["status"] == "データ欠損" and "15" in strict["judge_error"]
    assert "区切り" in collect_model(m, strict=True, min_chunks=100)["judge_error"]
    assert "imatrix" in collect_model(m, strict=True, min_chunks=80, min_imatrix_chunks=200)["judge_error"]


def test_report_strict_checks_completion(tmp_path):
    m = _model_dir(tmp_path)
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": [37, 37]}), encoding="utf-8")
    kw = {"strict": True, "min_chunks": 80, "min_imatrix_chunks": 128}
    (m / "logs" / "Q4_K_M-en-kld-ja.log").write_text(KLD_HEAD + _kld_rows(79), encoding="utf-8")
    e = collect_model(m, **kw)["judge_error"]
    assert "Q4_K_M-en" in e and "済んでいない" in e
    (m / "logs" / "Q4_K_M-en-kld-ja.log").write_text(KLD_LOG, encoding="utf-8")
    (m / "logs" / f"{P.REF}-base-en.log").write_text(_progress_log("perplexity", 80, final=False), encoding="utf-8")
    assert "比べ元 en" in collect_model(m, **kw)["judge_error"]
    (m / "logs" / f"{P.REF}-base-en.log").write_text(_progress_log("perplexity", 80), encoding="utf-8")
    (m / "logs" / "imatrix-ja.log").write_text(_progress_log("compute_imatrix", 128, done=100), encoding="utf-8")
    assert "imatrix ja" in collect_model(m, **kw)["judge_error"]


def test_report_status_unconvertible_and_weak_ref(tmp_path):
    m = _model_dir(tmp_path, answers={q.id: ("A" if q.should_hold else HOLD) for q in QS})
    kw = {"strict": True, "min_chunks": 80, "min_imatrix_chunks": 128}
    weak = collect_model(m, **kw)
    assert weak["status"] == "基準不適格" and not weak["judge"]["valid"]
    assert all(v["verdict"] == "測れない" for v in weak["judge"]["by_metric"].values())
    # 基準不適格なら、壊れた変種の記録があっても読まずに返す
    (m / "runs" / "Q2_K-en.jsonl").write_text("{壊れた行\n", encoding="utf-8")
    assert collect_model(m, **kw)["status"] == "基準不適格"
    # 答えがあるのに置いた変換不能の印は受け付けない
    (m / "unconvertible.txt").write_text("変換で落ちる", encoding="utf-8")
    late = collect_model(m, **kw)
    assert late["status"] == "データ欠損" and "替え" in late["judge_error"]
    assert "データ欠損" in markdown("m", late, False)
    fresh = tmp_path / "fresh"
    (fresh / "runs").mkdir(parents=True)
    (fresh / "unconvertible.txt").write_text("変換で落ちる", encoding="utf-8")
    gone = collect_model(fresh, **kw)
    assert gone["status"] == "変換不能" and gone["reason"] == "変換で落ちる"
    assert collect_model(fresh)["status"] != "変換不能"  # 試しでは印を見ない


def test_report_broken_record_is_missing_data(tmp_path):
    m = _model_dir(tmp_path)
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": [37, 37]}), encoding="utf-8")
    (m / "runs" / "Q2_K-en.jsonl").write_text("{壊れた行\n", encoding="utf-8")
    res = collect_model(m, strict=True)
    assert res["status"] == "データ欠損" and "読めない" in res["judge_error"]
    assert "データ欠損" in markdown("m", res, False)
    (m / "runs" / "Q2_K-en.jsonl").unlink()
    (m / "ref_offload.json").write_text("{", encoding="utf-8")
    assert "読めない" in collect_model(m, strict=True)["judge_error"]


def test_report_strict_keeps_minimum_inside(tmp_path):
    m = _model_dir(tmp_path)
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": [37, 37]}), encoding="utf-8")
    for x in P.TEXTS:
        (m / "logs" / f"{P.REF}-base-{x}.log").write_text(_progress_log("perplexity", 40), encoding="utf-8")
    assert "区切り" in collect_model(m, strict=True, min_chunks=20, min_imatrix_chunks=20)["judge_error"]


@pytest.mark.parametrize("off", [[37], ["9", "37"], [38, 37], [-1, 37], [0, 0], [9.0, 37], "37/37", [True, 37]])
def test_report_bad_offload_is_missing_data(tmp_path, off):
    m = _model_dir(tmp_path)
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": off}), encoding="utf-8")
    for kw in ({"strict": True}, {}):
        res = collect_model(m, **kw)
        assert res["status"] == "データ欠損" and "載り" in res["judge_error"]
        assert "ref_offload" not in res and "データ欠損" in markdown("m", res, not kw)


@pytest.mark.parametrize("off,want", [([9, 37], True), ([37, 37], False)])
def test_report_partial_offload_needs_place_run(tmp_path, monkeypatch, off, want):
    from evaluation.quant_probe import report
    seen = {}

    def fake(*a, **k):
        seen["need_place"] = a[7]
        raise J.StrictError("止めた")

    monkeypatch.setattr(report, "judge_model", fake)
    m = _model_dir(tmp_path)
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": off}), encoding="utf-8")
    assert collect_model(m, strict=True)["status"] == "データ欠損"
    assert seen["need_place"] is want  # 一部だけ載った時は載せ方違いの組を求める


@pytest.mark.parametrize("name,line", [("timing.jsonl", "{}"), ("timing.jsonl", '{"step": "a", "sec": "9"}'),
                                       ("timing.jsonl", "[1]"), ("variants.jsonl", '{"name": "Q4_K_M-en"}'),
                                       ("timing.jsonl", '{"step": "a", "sec": NaN}'),
                                       ("timing.jsonl", '{"step": "a", "sec": Infinity}'),
                                       ("timing.jsonl", '{"step": "a", "sec": -1}'),
                                       ("variants.jsonl", '{"name": "Q4_K_M-en", "sha256": "", "bytes": 3}'),
                                       ("variants.jsonl", '{"name": "Q4_K_M-en", "sha256": "ab", "bytes": 3}'),
                                       ("variants.jsonl", '{"name": "Q4_K_M-en", "sha256": "' + "A" * 64 + '", "bytes": 3}'),
                                       ("variants.jsonl", '{"name": "Q4_K_M-en", "sha256": "' + "a" * 64 + '", "bytes": 0}')])
def test_report_bad_side_records_are_missing_data(tmp_path, name, line):
    m = _model_dir(tmp_path)
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": [37, 37]}), encoding="utf-8")
    (m / name).write_text(line + "\n", encoding="utf-8")
    for kw in ({"strict": True}, {}):
        res = collect_model(m, **kw)
        assert res["status"] == "データ欠損" and name in res["judge_error"]
        assert "データ欠損" in markdown("m", res, not kw)
    good = ('{"step": "a", "sec": 1.5, "at": "x"}' if name == "timing.jsonl"
            else '{"name": "Q4_K_M-en", "sha256": "' + "0f" * 32 + '", "bytes": 3}')
    (m / name).write_text(good + "\n", encoding="utf-8")
    assert collect_model(m)["status"] == "適格"


def test_report_unreadable_marker_is_missing_data(tmp_path):
    fresh = tmp_path / "fresh"
    (fresh / "runs").mkdir(parents=True)
    (fresh / "unconvertible.txt").write_bytes(b"\xff\xfe\xfa")
    res = collect_model(fresh, strict=True)
    assert res["status"] == "データ欠損" and "印" in res["judge_error"]
    assert "データ欠損" in markdown("fresh", res, False)


def test_report_missing_ref_answers(tmp_path):
    m = _model_dir(tmp_path, answers={q.id: (HOLD if q.should_hold else q.expected) for q in QS[:300]})
    res = collect_model(m, strict=True, min_chunks=80, min_imatrix_chunks=128)
    assert res["status"] == "データ欠損" and "360" in res["judge_error"]
    (m / "runs" / f"{P.REF}-r2.jsonl").unlink()
    assert collect_model(m)["status"] == "データ欠損"


def test_report_main_keeps_minimums(tmp_path):
    from evaluation.quant_probe import report
    with pytest.raises(SystemExit):
        report.main(["--work", str(tmp_path), "--models", "x", "--out", str(tmp_path / "o"), "--min-chunks", "20"])
    with pytest.raises(SystemExit):
        report.main(["--work", str(tmp_path), "--models", "x", "--out", str(tmp_path / "o"),
                     "--min-imatrix-chunks", "64"])


def _ok(verdict="気づく"):
    return {"status": "適格", "judge": _res(verdict)}


def test_overall_block_and_spare():
    ok = {m: _ok() for m in PLANNED}
    assert overall_block(ok) is None and set(overall_models(ok)) == set(PLANNED)
    lost = {**ok, PLANNED[0]: {"status": "データ欠損", "judge_error": "答えの無い問が 3"}}
    assert "データ欠損" in overall_block(lost) and "答えの無い" in overall_block(lost)
    assert "決めたモデル" in overall_block({m: ok[m] for m in PLANNED[:2]})
    assert "決めたモデル" in overall_block({**ok, "other": _ok()})
    # 替えは変換不能がちょうど 1 つの時だけ
    assert SPARE in overall_block({**ok, SPARE: _ok()})
    gone = {**ok, PLANNED[1]: {"status": "変換不能", "reason": "x"}}
    assert SPARE in overall_block(gone)
    swap = {**gone, SPARE: _ok()}
    assert overall_block(swap) is None
    use = overall_models(swap)
    assert PLANNED[1] not in use and SPARE in use and len(use) == 3
    two = {**swap, PLANNED[2]: {"status": "変換不能", "reason": "y"}}
    assert "2 つ以上" in overall_block(two)
    # 基準不適格は替えずに「測れない」として残す
    weak = {**ok, PLANNED[1]: {"status": "基準不適格", "judge": _res("測れない", valid=False)}}
    assert overall_block(weak) is None
    use = overall_models(weak)
    assert PLANNED[1] in use and not use[PLANNED[1]]["valid"] and len(use) == 3
    assert SPARE in overall_block({**weak, SPARE: _ok()})


def test_power_runs_end_to_end():
    from evaluation.quant_probe import power
    import random
    w = power.World(random.Random(1))
    assert power.one_model(w, "null", 0, 24) in ("差が出ない", "決められない", "気づく", "気づかない")
    o = power.combine([{v: (1.0 if v == "気づく" else 0.0) for v in power.VERDICTS}] * 3)
    assert o["いいえ（この 360 問で）"] == pytest.approx(1.0)
    info = {}
    assert power.one_model(w, "miss_B", 30, 8, info=info) in power.VERDICTS
    assert info["added"] == 30 and 1 <= info["touched"] <= 8
    assert power.joint(w, [("notice", 0)] * 3, 24) in power.OVERALL
    assert power.joint(w, [("miss_A", 20)] * 2 + [("notice", 0)], 8) in power.OVERALL


def test_power_skew_is_mean_zero():
    from evaluation.quant_probe import power
    import random
    w = power.World(random.Random(2))
    _, base = w.ref()
    tot, n = 0, 400
    for _ in range(n):
        o = power.skew(w, base)
        tot += sum(v == "wrong_assert" for v in o.values()) - sum(v == "wrong_assert" for v in base.values())
    # 動く規程ごとに +3（1/4）か -1（3/4）。平均 0、1 回あたりの標準偏差は高々 sqrt(3 * 24) ≒ 8.5
    assert abs(tot / n) < 3 * 8.5 / n ** 0.5
    up = {b: True for b in w.order}
    o = power.skew(w, base, coins=up)
    assert sum(v == "wrong_assert" for v in o.values()) > sum(v == "wrong_assert" for v in base.values())
    # 逆向き: 動かせる規程ごとに 3/4 で +1・1/4 で -3。入力を固定した期待値はちょうど 0
    rich = {i: ("wrong_assert" if k % 3 == 0 else v) for k, (i, v) in enumerate(base.items())}
    movable = 0
    for b in w.order:
        ids = [i for i in w.ids if w.block[i] == b]
        on = sum(rich[i] == "wrong_assert" for i in ids)
        movable += on >= 3 and len(ids) - on >= 1
    rare = power.skew(w, rich, coins=up, reverse=True)
    common = power.skew(w, rich, coins={b: False for b in w.order}, reverse=True)
    n_wa = lambda o: sum(v == "wrong_assert" for v in o.values())
    assert n_wa(rare) - n_wa(rich) == -3 * movable and n_wa(common) - n_wa(rich) == movable
    assert 0.25 * (-3 * movable) + 0.75 * movable == 0
    assert power.base_delta("notice", "Q4_K_M-ja") == 0 and power.base_delta("notice", "Q3_K_M-ja") == 12


def test_pick_ja_articles():
    rows = [{"title": "短い", "text": "あ" * 100}, {"title": "東京の一覧", "text": "い" * 3000},
            {"title": "A", "text": "う" * 2500}, {"title": "B", "text": "え" * 2500}, {"title": "C", "text": "お" * 9}]
    text, titles = pick_ja_articles(rows, min_chars=2000, total_chars=4000)
    assert titles == ["A", "B"] and text.count("う") == 2500
    with pytest.raises(ValueError):
        pick_ja_articles(rows, min_chars=2000, total_chars=10_000)
