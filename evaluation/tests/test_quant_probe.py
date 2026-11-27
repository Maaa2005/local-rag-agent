"""量子化の物差し (evaluation/quant_probe) のテスト。モデル・GPU・ネットは使わない。

確かめること:
- 問 360 問（版 7 の見済みの問 = DATA_DIR_V7。版 8 の data_v8 は、そろっていれば件数と内訳だけ）の数と内訳・指紋・渡す文の形
- 結論の読み方と 6 つの分け方・集計
- 検定（規程ごとの符号入れ替え・ブートストラップ・McNemar・Holm・Spearman）の値
- KLD の出力の読み方・GPU への載り方の読み方・会話の形の KLD（chatkld）・答えの依頼と最初の 1 語の確率（run_answers）
- 判定の線 qp-judge-8（線 0 の 0a・0b・0c・型 A・型 B・気づく・決められない・差が出ない・測れない・厳しく見る時・全体）
- 全体を回す部品が、決めた順と引数でコマンドを組むこと（載り切らない時の -ngl を含む）・集計が記録をつなぐこと
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.quant_probe import judge as J
from evaluation.quant_probe import pipeline as P
from evaluation.quant_probe import questions as Q
from evaluation.quant_probe import run_answers as RA
from evaluation.quant_probe.chatkld import chat_kld, question_kld
from evaluation.quant_probe.kld import parse_log
from evaluation.quant_probe.questions import (DATA_DIR, DATA_DIR_V7, HOLD, PROMPT_VERSION, SETS, SETS_V7, build_messages,
                                              domain_text, load_questions, load_rules, manifest)
from evaluation.quant_probe.report import PLANNED, SPARES, collect_model, markdown, overall_block, overall_models
from evaluation.quant_probe.score import classify, parse_conclusion, score_rows, summarize
from evaluation.quant_probe.stats import (block_bootstrap_upper, holm, mcnemar_one_sided, mcnemar_two_sided,
                                          signflip_exact, spearman)
from evaluation.quant_probe.texts import pick_ja_articles

QS = load_questions(DATA_DIR_V7, SETS_V7)
RULES = load_rules(DATA_DIR_V7, SETS_V7)
# 版 7 の写しの指紋。問か規程を 1 字でも変えたら変わる（変えるなら計画の版を上げる）
MANIFEST_TOTAL = "b0161f1c54f28abf116026ca755d41a32bbd9d0f58605546af6cf7b19689cc50"
# 渡す文 qp-1 の SYSTEM と USER（版 8 でも 1 字も変えない）
PROMPT_SHA = "0caf0a39a914ce2bf81520cfe600d70f561c45072088bf29ffcd0ac24a820587"


@pytest.fixture(autouse=True)
def _v7_questions(monkeypatch):
    """集計（report）と答えを流す道具（run_answers）の既定の問は data_v8 だが、テストでは版 7 の 360 問を読ませる。"""
    from evaluation.quant_probe import report
    v7 = lambda *a, **k: load_questions(DATA_DIR_V7, SETS_V7)
    monkeypatch.setattr(report, "load_questions", v7)
    monkeypatch.setattr(report, "manifest", lambda *a, **k: manifest(DATA_DIR_V7, SETS_V7))
    monkeypatch.setattr(RA, "load_questions", v7)
    monkeypatch.setattr(RA, "load_rules", lambda *a, **k: load_rules(DATA_DIR_V7, SETS_V7))
    monkeypatch.setattr(RA, "manifest", lambda *a, **k: manifest(DATA_DIR_V7, SETS_V7))


def test_default_questions_are_v8_and_v7_still_loads():
    assert DATA_DIR.name == "data_v8" and SETS == ("numeric_v8a", "numeric_v8b", "numeric_v8c", "numeric_v8d")
    assert DATA_DIR_V7.name == "data" and SETS_V7 == ("numeric_dev90", "numeric_dev90b", "numeric_dev90c", "numeric_dev90d")
    assert len(load_questions(DATA_DIR_V7, SETS_V7)) == 360
    assert hashlib.sha256((Q.SYSTEM + "\x00" + Q.USER).encode()).hexdigest() == PROMPT_SHA  # 渡す文は変えない


# 版 8 の問と規程の指紋（2026-11-28 の点検で 13 問と規程 5 か所を直した後・正解は変えていない）。変えるなら計画の版を上げる
MANIFEST_TOTAL_V8 = "d82935b365b30004e262826293e6b81e984fe0d530c4b7ab50586b40c801f667"
V8_READY = all((DATA_DIR / s / "questions.jsonl").exists() and (DATA_DIR / s / "corpus").is_dir() for s in SETS)


@pytest.mark.skipif(not V8_READY, reason="data_v8 の 4 組がまだそろっていない")
def test_data_v8_manifest_pinned():
    m = manifest()
    assert m["prompt_version"] == "qp-1" and m["total"] == MANIFEST_TOTAL_V8
    assert m["total"] != MANIFEST_TOTAL  # 版 7 の指紋とは別


@pytest.mark.skipif(not V8_READY, reason="data_v8 の 4 組がまだそろっていない")
def test_data_v8_counts():
    qs = load_questions()
    assert len(qs) == 360 and sum(q.should_hold for q in qs) == 72
    per_rule, hold_rule = {}, {}
    for q in qs:
        per_rule[(q.set, q.rule_id)] = per_rule.get((q.set, q.rule_id), 0) + 1
        hold_rule[(q.set, q.rule_id)] = hold_rule.get((q.set, q.rule_id), 0) + q.should_hold
    assert len(per_rule) == 24 == len(load_rules()) and set(per_rule.values()) == {15} and set(hold_rule.values()) == {3}
    pairs = {}
    for q in qs:
        if q.pair_id:
            pairs.setdefault(q.pair_id, []).append(q)
    assert len(pairs) == 48 and all(len(v) == 2 for v in pairs.values())


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
    m = manifest(DATA_DIR_V7, SETS_V7)
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
    t = domain_text(DATA_DIR_V7, SETS_V7)
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


def _top(*pairs: tuple[int, float]) -> list[dict]:
    """probe の形（語の番号 id・確率 p）。pairs は (id, 確率)。"""
    return [{"id": i, "p": p} for i, p in pairs]


def _dist(*pairs: tuple[int, float]) -> dict[int, float]:
    return {i: p for i, p in pairs}


def test_question_kld_by_hand():
    p = _dist((1, 0.6), (2, 0.3), (3, 0.05))  # その他 0.05（語 3 は共通でないので、その他に入る）
    q = _dist((1, 0.5), (2, 0.35), (4, 0.1))
    common = {1, 2}
    want = 0.6 * math.log(0.6 / 0.5) + 0.3 * math.log(0.3 / 0.35) + 0.1 * math.log(0.1 / 0.15)  # p_o = 0.1・q_o = 0.15
    assert question_kld(p, q, common) == pytest.approx(want)
    assert question_kld(p, p, set(p)) == pytest.approx(0.0, abs=1e-12)
    # 共通の語が無ければ、その他だけの分布（P・Q とも 1）で 0（分布が同じという意味ではない）
    assert question_kld(_dist((1, 0.9)), _dist((2, 0.9)), set()) == pytest.approx(0.0, abs=1e-12)
    # まとめた分布の KLD は、元の分布の KLD 以下
    full_p, full_q = _dist((1, 0.5), (2, 0.3), (3, 0.2)), _dist((1, 0.2), (2, 0.3), (3, 0.5))
    full = sum(full_p[i] * math.log(full_p[i] / full_q[i]) for i in full_p)
    assert 0.0 < question_kld(full_p, full_q, {1, 2}) <= full + 1e-12
    # 下限 1e-12: Q の共通の語が 0 でも有限・その他が 0 以下に落ちても対数が取れる
    assert math.isfinite(question_kld(_dist((1, 0.5), (2, 0.5)), _dist((1, 0.0), (2, 1.0)), {1, 2}))
    assert question_kld(_dist((1, 0.5), (2, 0.5)), _dist((1, 0.6), (2, 0.6)), {1, 2}) >= 0.0


def _rows(tops: dict[str, list[dict] | None]) -> dict[str, dict]:
    return {i: {"id": i, "probe": t} for i, t in tops.items()}


def test_chat_kld_shares_common_set_across_variants():
    ids = [f"a{k}" for k in range(4)]
    ref = _rows({i: _top((1, 0.6), (2, 0.3), (3, 0.05)) for i in ids})
    va = _rows({i: _top((1, 0.5), (2, 0.35), (4, 0.1)) for i in ids})
    vb = _rows({i: _top((1, 0.7), (5, 0.2), (3, 0.06)) for i in ids})
    both = chat_kld(ref, {"A": va, "B": vb}, ids, min_usable=4)
    # 共通の語は 3 つの上位の全部に出る語 = {1}（語 2 は B に無く、語 3 は A に無い）。A も B も同じ区分
    want_a = 0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5)
    want_b = 0.6 * math.log(0.6 / 0.7) + 0.4 * math.log(0.4 / 0.3)
    assert both["A"]["kld_mean"] == pytest.approx(want_a) and both["B"]["kld_mean"] == pytest.approx(want_b)
    assert both["A"]["common_mean"] == 1.0 and both["A"]["other_p_mean"] == pytest.approx(0.4) and both["A"]["n"] == 4
    # 変種を 1 つに減らすと共通の語が増え、A の値が変わる（変種の組を固定する意味）
    alone = chat_kld(ref, {"A": va}, ids, min_usable=4)
    assert alone["A"]["common_mean"] == 2.0
    assert alone["A"]["kld_mean"] == pytest.approx(0.6 * math.log(0.6 / 0.5) + 0.3 * math.log(0.3 / 0.35) + 0.1 * math.log(0.1 / 0.15))
    assert alone["A"]["kld_mean"] != both["A"]["kld_mean"]
    # 一番の候補が同じだった割合（全部の語で見る）
    vc = _rows({ids[0]: _top((2, 0.9), (1, 0.1)), **{i: _top((1, 0.8), (2, 0.2)) for i in ids[1:]}})
    assert chat_kld(ref, {"C": vc}, ids, min_usable=4)["C"]["same_top"] == pytest.approx(0.75)


def test_chat_kld_skips_and_thresholds():
    ids = [f"a{k}" for k in range(5)]
    top = _top((1, 0.6), (2, 0.4))
    ref = _rows({**{i: top for i in ids[:4]}, ids[4]: None})  # 5 問目は基準に「結論:」が無かった → 使わない
    ok = _rows({i: top for i in ids})
    r = chat_kld(ref, {"A": ok}, ids, min_usable=4)["A"]
    assert r["n"] == 4 and r["kld_mean"] == pytest.approx(0.0, abs=1e-12) and r["same_top"] == 1.0
    assert "reason" not in r
    # 使えた問が下限（既定 342）未満: 全変種で値なし・理由つき
    low = chat_kld(ref, {"A": ok, "B": ok}, ids)
    assert all(v["kld_mean"] is None and v["same_top"] is None and v["n"] == 4 and "342" in v["reason"] for v in low.values())
    assert chat_kld(ref, {"A": ok}, ids, min_usable=5)["A"]["kld_mean"] is None
    # 変種に確かめが無い問が 1 つでもあれば、その変種だけ値なし
    part = _rows({**{i: top for i in ids[:3]}, ids[3]: None, ids[4]: top})
    both = chat_kld(ref, {"A": ok, "P": part}, ids, min_usable=4)
    assert both["A"]["kld_mean"] == pytest.approx(0.0, abs=1e-12) and both["P"]["kld_mean"] is None
    assert both["P"]["n"] == 3 and "P" in both["P"]["reason"]
    # 360 問が既定の下限を超える
    full = [f"q{k:03d}" for k in range(360)]
    big = chat_kld(_rows({i: top for i in full}), {"A": _rows({i: top for i in full})})
    assert big["A"]["n"] == 360 and big["A"]["kld_mean"] == pytest.approx(0.0, abs=1e-12)
    drop = _rows({i: (top if k >= 18 else None) for k, i in enumerate(full)})  # 342 問
    assert chat_kld(drop, {"A": _rows({i: top for i in full})})["A"]["n"] == 342
    drop = _rows({i: (top if k >= 19 else None) for k, i in enumerate(full)})  # 341 問
    assert chat_kld(drop, {"A": _rows({i: top for i in full})})["A"]["kld_mean"] is None


@pytest.mark.parametrize("bad", [[{"id": 1}], [{"p": 0.5}], [{"id": "1", "p": 0.5}], [{"id": 1, "p": float("nan")}],
                                 [{"id": 1, "p": -0.1}], [{"id": 1, "p": True}], ["x"]])
def test_chat_kld_rejects_bad_probe(bad):
    ok = _rows({"a": _top((1, 1.0))})
    with pytest.raises(ValueError):
        chat_kld(_rows({"a": bad}), {"A": ok}, ["a"], min_usable=1)
    with pytest.raises(ValueError):
        chat_kld(ok, {"A": _rows({"a": bad})}, ["a"], min_usable=1)


# ---- run_answers: 答えの依頼は版 7 と同じ・確かめの依頼・取り違えの防止 ----
ANS = "理由。\n結論: 取れる"


def _fake_server(sent, texts=None, tokens=57):
    """URL の終わりで返事を分ける偽のサーバー。texts は id の順に答えの文を返す（無ければ ANS）。"""
    state = {"n": 0}

    def post(url, body, timeout):
        sent.append((url.rsplit("/", 1)[-1], body))
        if url.endswith("/v1/chat/completions"):
            text = (texts or {}).get(state["n"], ANS)
            state["n"] += 1
            return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": tokens},
                    "timings": {}}
        if url.endswith("/apply-template"):
            return {"prompt": f"<T>{body['messages'][1]['content'][:10]}"}
        if url.endswith("/tokenize"):
            return {"tokens": list(range(tokens))}
        assert url.endswith("/completion")
        return {"content": "x", "tokens_evaluated": tokens + 3, "completion_probabilities": [
            {"id": 5, "token": "取", "logprob": -0.1, "top_logprobs": [
                {"id": 5, "token": "取", "bytes": [1], "logprob": math.log(0.7)},
                {"id": 6, "token": "保", "bytes": [2], "logprob": math.log(0.2)}]}]}
    return post


def test_request_bodies_and_probe_helpers():
    q = QS[0]
    # 答えの依頼は版 7 と 1 字も同じ（確率を頼まない）
    assert RA.request_body(q, RULES, 512) == {"messages": build_messages(q, RULES), "max_tokens": 512, **RA.GEN}
    assert "logprobs" not in RA.request_body(q, RULES, 512) and not hasattr(RA, "TOP_LOGPROBS")
    assert RA.GEN == {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1, "cache_prompt": False,
                      "chat_template_kwargs": {"enable_thinking": False}}
    # /apply-template は messages と chat_template_kwargs だけ（答えの依頼と同じ）
    assert RA.apply_template_body(q, RULES) == {"messages": build_messages(q, RULES),
                                                "chat_template_kwargs": {"enable_thinking": False}}
    assert RA.completion_body("P") == {"prompt": "P", "n_predict": 1, "n_probs": 100, "post_sampling_probs": False,
                                       "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False,
                                       "grammar": "root ::= [ -~]"}
    assert RA.PROBE_GRAMMAR == "root ::= [ -~]" and RA.completion_body("P")["cache_prompt"] is False
    assert RA.probe_gen() == {k: v for k, v in RA.completion_body("P").items() if k != "prompt"} and "prompt" not in RA.probe_gen()
    # 書き出し: 最後の「結論:」の直後（「結論:」まで含む・後ろの空白は付けない）
    assert RA.probe_prefix("理由。\n結論: 取れる") == "理由。\n結論:"
    assert RA.probe_prefix("結論: 取れない\n前の話に戻る。\n結論: 取れる\n") == "結論: 取れない\n前の話に戻る。\n結論:"
    assert RA.probe_prefix("理由だけ") is None and RA.probe_prefix("") is None and RA.probe_prefix(None) is None
    assert RA.probe_prefix("結論：全角") is None  # 渡す文の形（半角のコロン）の「結論:」だけ
    # /completion の返事の読み方: logprob（自然対数）→ 確率・prob はそのまま・無ければ None
    resp = _fake_server([])("http://x/completion", {}, 1)
    assert RA.probe_top(resp) == [{"id": 5, "p": pytest.approx(0.7)}, {"id": 6, "p": pytest.approx(0.2)}]
    post = {"completion_probabilities": [{"id": 5, "top_probs": [{"id": 5, "token": "a", "prob": 0.4}]}]}
    assert RA.probe_top(post) == [{"id": 5, "p": 0.4}]
    assert RA.probe_top({}) is None and RA.probe_top({"completion_probabilities": []}) is None
    assert RA.probe_top({"completion_probabilities": [{"top_logprobs": []}]}) is None
    with pytest.raises(ValueError):
        RA.probe_top({"completion_probabilities": [{"top_logprobs": [{"token": "a", "logprob": -1.0}]}]})
    # 一番小さい logprob（浮動小数の下限）は確率 0 になる
    low = {"completion_probabilities": [{"top_logprobs": [{"id": 1, "logprob": -3.4028234663852886e38}]}]}
    assert RA.probe_top(low) == [{"id": 1, "p": 0.0}]


def _patch_server(monkeypatch, sent, **kw):
    monkeypatch.setattr(RA, "_post", _fake_server(sent, **kw))
    monkeypatch.setattr(RA, "_get", lambda *a, **k: {})
    monkeypatch.setattr(RA, "wait_ready", lambda *a, **k: None)


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_run_answers_without_probe_is_the_v7_request(tmp_path, monkeypatch):
    sent = []
    _patch_server(monkeypatch, sent)
    out = tmp_path / "a.jsonl"
    assert RA.main(["--out", str(out), "--limit", "3"]) == 0
    rows = _read(out)
    assert len(rows) == 3 and {u for u, _ in sent} == {"completions"}  # 確かめを挟まない
    assert all(b == RA.request_body(q, RULES, 512) for (_, b), q in zip(sent, QS))
    assert all("probe" not in r and "first_top" not in r for r in rows)
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["probe"] is None and meta["probe_gen"] is None and meta["smoke"] is False and meta["manifest"]["total"] == MANIFEST_TOTAL
    assert meta["prompt_version"] == "qp-1" and meta["max_tokens"] == 512 and meta["gen"] == RA.GEN


def test_run_answers_probe_self_and_ref(tmp_path, monkeypatch):
    sent = []
    texts = {1: "答えだけ", 2: "", 3: "理由。\n結論: 保留"}  # 2 問目は「結論:」が無い・3 問目は空・4 問目は保留
    _patch_server(monkeypatch, sent, texts=texts)
    out = tmp_path / "r1.jsonl"
    assert RA.main(["--out", str(out), "--limit", "4", "--probe-self"]) == 0
    rows = _read(out)
    assert [r["probe"] is None for r in rows] == [False, True, True, False]
    # 各問で、答えの依頼の直後に確かめの依頼（apply-template → tokenize → completion）を挟む
    kinds = [u for u, _ in sent]
    assert kinds == ["completions", "apply-template", "tokenize", "completion", "completions", "completions",
                     "completions", "apply-template", "tokenize", "completion"]
    templ = "<T>" + build_messages(QS[0], RULES)[1]["content"][:10]
    assert sent[1][1] == RA.apply_template_body(QS[0], RULES)
    assert sent[2][1] == {"content": templ, "add_special": True, "parse_special": True}
    assert sent[3][1] == RA.completion_body(templ + "理由。\n結論:")
    assert rows[0]["probe"] == [{"id": 5, "p": pytest.approx(0.7)}, {"id": 6, "p": pytest.approx(0.2)}]
    assert rows[0]["probe_prompt_tokens"] == 57
    assert rows[0]["probe_prefix_sha256"] == hashlib.sha256("理由。\n結論:".encode()).hexdigest()
    assert rows[1]["probe_prompt_tokens"] is None and rows[2]["text"] == ""
    assert sent[9][1]["prompt"].endswith("理由。\n結論:") and json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["probe"] == "self"
    # --probe-ref: 保存してある基準 1 回目の文の書き出しを使う（今の答えの文ではない）
    sent.clear()
    _patch_server(monkeypatch, sent)  # 今回の答えは全部 ANS
    out2 = tmp_path / "r2.jsonl"
    assert RA.main(["--out", str(out2), "--limit", "4", "--probe-ref", str(out)]) == 0
    rows2 = _read(out2)
    assert [r["probe"] is None for r in rows2] == [False, True, True, False]  # 基準 1 回目に「結論:」の無い問は確かめない
    assert [u for u, _ in sent].count("completion") == 2
    assert json.loads(out2.with_suffix(".meta.json").read_text(encoding="utf-8"))["probe"] == "ref"
    # 確かめの依頼の設定を meta に残す（確かめを挟んだ回だけ）。取り違えの比べる欄には入れない
    for o in (out, out2):
        assert json.loads(o.with_suffix(".meta.json").read_text(encoding="utf-8"))["probe_gen"] == RA.probe_gen()
    meta2 = json.loads(out2.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert RA.meta_mismatch({**meta2, "probe_gen": {"x": 1}}, 512) == []
    # 同時には指定できない
    with pytest.raises(SystemExit):
        RA.main(["--out", str(tmp_path / "x.jsonl"), "--probe-self", "--probe-ref", str(out)])


def test_run_answers_stops_on_mismatched_meta_or_ids(tmp_path, monkeypatch):
    sent = []
    _patch_server(monkeypatch, sent)
    out = tmp_path / "a.jsonl"
    assert RA.main(["--out", str(out), "--limit", "2"]) == 0
    sent.clear()
    assert RA.main(["--out", str(out), "--limit", "3"]) == 0  # 同じ指紋なら足せる
    assert len(_read(out)) == 3
    mp = out.with_suffix(".meta.json")
    good = json.loads(mp.read_text(encoding="utf-8"))
    for key, bad in (("prompt_version", "qp-2"), ("max_tokens", 256), ("gen", {**RA.GEN, "seed": 2}),
                     ("manifest", {**good["manifest"], "total": "0" * 64}), ("smoke", True)):
        mp.write_text(json.dumps({**good, key: bad}), encoding="utf-8")
        sent.clear()
        assert RA.main(["--out", str(out), "--limit", "5"]) == RA.EXIT_MISMATCH == 2, key
        assert sent == [] and len(_read(out)) == 3  # 答えを足さない
    mp.write_text(json.dumps(good), encoding="utf-8")
    # --max-tokens を変えて再開しても止まる
    assert RA.main(["--out", str(out), "--max-tokens", "256"]) == 2
    # 途中まである jsonl に、今の問の集まりに無い id がある
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "ZZZ999", "text": "x"}) + "\n")
    assert RA.main(["--out", str(out), "--limit", "5"]) == 2 and len(_read(out)) == 4
    # 古い meta（smoke の欄が無い）は smoke でない物として比べる
    old = tmp_path / "old.jsonl"
    assert RA.main(["--out", str(old), "--limit", "1"]) == 0
    m = json.loads(old.with_suffix(".meta.json").read_text(encoding="utf-8"))
    m.pop("smoke")
    old.with_suffix(".meta.json").write_text(json.dumps(m), encoding="utf-8")
    assert RA.main(["--out", str(old), "--limit", "2"]) == 0


def test_smoke_ids_and_v7_smoke(tmp_path, monkeypatch):
    ids = Q.smoke_ids_v7()
    assert ids == ["W001", "W037", "W073", "X019", "X055", "Y001", "Y037", "Y073", "Z019", "Z055"]
    allv7 = sorted(q.id for q in QS)
    assert ids == allv7[::36] and len(ids) == 10
    sent = []
    _patch_server(monkeypatch, sent)
    out = tmp_path / "smoke" / "s.jsonl"
    assert RA.main(["--out", str(out), "--v7-smoke", "--probe-self"]) == 0
    rows = _read(out)
    assert [r["id"] for r in rows] == [q.id for q in QS if q.id in set(ids)] and len(rows) == 10
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["smoke"] is True and meta["manifest"]["total"] == MANIFEST_TOTAL and meta["n_questions"] == 10
    # 版 7 の問として流す（版 7 の問の本文・規程が答えの依頼に入る）
    by_id = {q.id: q for q in QS}
    assert sent[0][1] == RA.request_body(by_id[rows[0]["id"]], RULES, 512)
    # 本番の meta（smoke でない）の出力先に足そうとすると止まる
    mp = out.with_suffix(".meta.json")
    mp.write_text(json.dumps({**meta, "smoke": False}), encoding="utf-8")
    assert RA.main(["--out", str(out), "--v7-smoke"]) == 2
    # smoke の 10 問に無い id がある jsonl にも足さない
    mp.write_text(json.dumps(meta), encoding="utf-8")
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": QS[1].id if QS[1].id not in ids else QS[2].id, "text": "x"}) + "\n")
    assert RA.main(["--out", str(out), "--v7-smoke"]) == 2


# ---- 判定の線 ----
IDS = [f"q{i:03d}" for i in range(360)]
HOLD_IDS = {IDS[b * 15 + k] for b in range(24) for k in range(3)}  # 1 規程 15 問のうち、先頭の 3 問が保留すべき問
SH = {i: i in HOLD_IDS for i in IDS}
BLOCKS = {i: f"r{k // 15:02d}" for k, i in enumerate(IDS)}  # 24 規程 × 15 問
REF_WA = {IDS[b * 15] for b in range(10)}  # 規程 0〜9 の保留すべき問の 1 問目（言い切った）
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


def _ref_pattern(k_good: int) -> dict[str, str]:
    """規程 0〜k_good-1 は全部できている（c = 12・h = 36）。残りの規程は c も h も 0（答えられる問は正答 6・誤り 6・
    保留すべき問は全部言い切る）。"""
    o = {}
    for k, i in enumerate(IDS):
        r, pos = divmod(k, 15)
        if r < k_good:
            o[i] = "held_ok" if pos < 3 else "correct"
        elif pos < 3:
            o[i] = "wrong_assert"
        else:
            o[i] = "correct" if pos < 9 else "wrong_assert"
    return o


def test_reference_values_c_h_and_p():
    q = J.quality(outcomes(REF_WA), SH, BLOCKS)
    # 規程 0〜9 は保留すべき問 1 問を言い切った（h = 2×12）、ほかは h = 3×12。c は全規程 12
    assert q["c_sum"] == 24 * 12 and q["h_sum"] == 10 * 24 + 14 * 36
    assert q["p_direction"] == pytest.approx(2.0 ** -24) and q["p_hold"] == pytest.approx(2.0 ** -24)
    assert (q["correct"], q["held"], q["format"], q["over_hold"]) == (1.0, pytest.approx(1 - 10 / 72), 0.0, 0.0)
    # 保留しすぎは h を減らす: 1 規程で答えられる問 2 問を保留すると h_r = 3×12 − 2×3
    over = {ANSWERABLE[0], ANSWERABLE[1]}
    c, h = J.ref_blocks(outcomes(set(), over), SH, BLOCKS)
    assert c[0] == 10 and h[0] == 3 * 12 - 2 * 3 and c[1] == 12 and h[1] == 36


def test_reference_all_zero_differences_do_not_pass():
    # いつも保留: 答えられる問は保留しすぎ・保留すべき問は保留できた。c は全部 0・h = 12×3 − 3×12 = 0 → p = 1 で通らない
    always_hold = {i: ("held_ok" if SH[i] else "over_hold") for i in IDS}
    q = J.quality(always_hold, SH, BLOCKS)
    assert q["c_sum"] == 0 and q["h_sum"] == 0 and q["p_direction"] == 1.0 and q["p_hold"] == 1.0
    assert not J.ref_ok(q)
    r = J.judge_model(always_hold, always_hold, [variant("Q4_K_M-en", set(), 0.03)], SH, BLOCKS)
    assert not r["valid"] and r["by_metric"]["kld_en"]["verdict"] == "測れない"
    assert r["ref_quality"][0]["p_direction"] == 1.0 and r["ref_quality"][0]["p_hold"] == 1.0
    # いつも同じ語: 答えられる問の半分が当たり・半分が逆、保留すべき問は全部言い切る → c も h も 0
    same_word = _ref_pattern(0)
    q = J.quality(same_word, SH, BLOCKS)
    assert q["c_sum"] == 0 and q["h_sum"] == 0 and not J.ref_ok(q)


def test_reference_line_zero_threshold_is_0_001():
    assert J.REF_ALPHA == 0.001
    # 良い規程が k 個だけで、ほかは差が 0: p = 2^-k。9 規程では 0.00195 で通らず、10 規程では 0.000977 で通る
    q9, q10 = J.quality(_ref_pattern(9), SH, BLOCKS), J.quality(_ref_pattern(10), SH, BLOCKS)
    assert q9["p_direction"] == pytest.approx(2.0 ** -9) and q9["p_hold"] == pytest.approx(2.0 ** -9) and not J.ref_ok(q9)
    assert q10["p_direction"] == pytest.approx(2.0 ** -10) and J.ref_ok(q10)
    assert J.judge_model(_ref_pattern(10), _ref_pattern(10), [variant("Q4_K_M-en", set(), 0.03)], SH, BLOCKS)["valid"]
    r = J.judge_model(_ref_pattern(10), _ref_pattern(9), [variant("Q4_K_M-en", set(), 0.03)], SH, BLOCKS)
    assert not r["valid"]  # 2 回のうち 1 回でも割ったら測れない


def test_reference_format_limit_0a():
    ok = outcomes(set())
    bad = {i: ("format" if k % 15 == 14 else v) for k, (i, v) in enumerate(ok.items())}  # 24/360 = 6.7%
    q = J.quality(bad, SH, BLOCKS)
    assert q["format"] > 0.05 and q["p_direction"] < 0.001 and not J.ref_ok(q)
    few = {i: ("format" if k % 30 == 14 else v) for k, (i, v) in enumerate(ok.items())}  # 12/360 = 3.3%
    assert J.ref_ok(J.quality(few, SH, BLOCKS))
    miss = {i: ("missing" if k % 15 == 14 else v) for k, (i, v) in enumerate(ok.items())}  # 欠けも形式違反と同じ扱い
    assert not J.ref_ok(J.quality(miss, SH, BLOCKS))


def test_weak_reference_not_measurable():
    weak = set(ANSWERABLE)  # 答えられる問を全部誤る → c < 0
    vs = [{"name": "Q4_K_M-en", "outcomes": outcomes(weak), "metrics": kmetrics(0.03)}]
    r = J.judge_model(outcomes(weak), outcomes(weak), vs, SH, BLOCKS)
    assert not r["valid"] and r["by_metric"]["kld_en"]["verdict"] == "測れない"
    assert r["ref_quality"][0]["p_direction"] == 1.0


def test_reference_must_see_hold():
    bad_hold = REF_WA | HOLD_IDS  # 保留すべき問を全部言い切る → h = 0
    vs = [variant("Q4_K_M-en", set(), 0.03)]
    r = J.judge_model(outcomes(REF_WA), outcomes(bad_hold), vs, SH, BLOCKS)
    assert r["ref_quality"][1]["held"] == 0 and r["ref_quality"][1]["p_hold"] == 1.0 and not r["valid"]


def test_8b_like_reference_now_passes_line_zero():
    # 版 7 の固定の線（正答 70%・保留 50%）を割る 8B 級: 正答 8/12・保留しすぎ 2/12・誤り 2/12、保留できた 1/3
    o = {}
    for k, i in enumerate(IDS):
        pos = k % 15
        if pos < 3:
            o[i] = "held_ok" if pos == 0 else "wrong_assert"
        else:
            o[i] = "correct" if pos < 11 else ("over_hold" if pos < 13 else "wrong_assert")
    q = J.quality(o, SH, BLOCKS)
    assert q["correct"] < 0.70 and q["held"] < 0.50  # 版 7 なら基準不適格
    assert J.ref_ok(q)  # 版 8 は 0 より上の向き・保留の見分けがあれば通る


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
    assert not hasattr(P, "PLACE") and not hasattr(J, "PLACE")  # 載せ方の組は判定から外した
    with pytest.raises(TypeError):
        judge(full, strict=True, noise_pairs=pairs, need_place=True)
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
    (["気づかない", "気づかない", "気づく", "気づく", "気づく"], "はい"),
    (["気づかない"] * 5, "はい"),
    (["気づかない", "気づく", "気づく", "気づく", "気づく"], "モデルによる"),
    (["気づかない", "差が出ない", "決められない", "測れない", "気づく"], "モデルによる"),
    (["気づく", "気づく", "気づく", "差が出ない", "差が出ない"], "いいえ（この 360 問で）"),
    (["気づく"] * 5, "いいえ（この 360 問で）"),
    (["気づく", "気づく", "差が出ない", "差が出ない", "差が出ない"], "測れず"),  # 「気づく」は 3 つ要る
    (["気づく", "決められない", "決められない", "決められない", "決められない"], "測れず"),
    (["気づかない"], "測れず"),
])
def test_overall(vs, want):
    assert J.judge_overall({str(i): _res(v) for i, v in enumerate(vs)}) == want


def test_overall_skips_invalid():
    r = {"a": _res("気づかない", valid=False), "b": _res("気づく"), "c": _res("気づく"), "d": _res("気づく"), "e": _res("気づく")}
    assert J.judge_overall(r) == "いいえ（この 360 問で）"
    r["d"] = _res("気づく", valid=False)  # 測れないモデルは「気づく」に数えない: 3 つ残れば「いいえ」
    assert J.judge_overall(r) == "いいえ（この 360 問で）"
    r["e"] = _res("気づく", valid=False)  # 2 つしか残らなければ「気づく」が 3 つに届かない
    assert J.judge_overall(r) == "測れず"


def test_alpha_bounds_two_false_of_five():
    # 本当に見逃すモデルが 0 なら 2 つ以上誤る確率 <= 5α/2、1 なら残り 4 つのどれかが誤る確率 <= 4α（依存によらない）
    assert J.JUDGE_VERSION == "qp-judge-8" and J.ALPHA_MODEL == 0.0125
    assert 5 * J.ALPHA_MODEL / 2 <= 0.03125 + 1e-12
    assert 4 * J.ALPHA_MODEL <= 0.05 + 1e-12
    assert J.ALPHA_A + J.ALPHA_B == pytest.approx(J.ALPHA_MODEL) and J.ALPHA_A == J.ALPHA_B
    assert "kld_chat" in J.METRICS and J.PRIMARY == "kld_en"
    assert J.deviation({"kld_chat": {"kld_mean": 0.02, "same_top": 0.9, "n": 360}}, "kld_chat") == 0.02
    assert J.deviation({"kld_chat": {"kld_mean": None, "n": 0}}, "kld_chat") is None


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
    cmds = _dry(tmp_path, capsys, "--ngl", "30", "--variants", "Q4_K_M-en,Q2_K-none")
    q = [c for c in cmds if " llama-quantize " in c]
    assert q[0].endswith("Q8_0") and "--imatrix" not in q[0]
    assert "--imatrix" in q[1] and "imatrix-en.gguf" in q[1] and q[1].endswith("Q4_K_M")
    assert "--imatrix" not in q[2] and q[2].endswith("Q2_K")
    im = [c for c in cmds if " llama-imatrix " in c]
    assert len(im) == 2 and all("--chunks 128" in c and "-c 512" in c for c in im)
    base = [c for c in cmds if "--kl-divergence-base" in c and "--kl-divergence " not in c + " "]
    kl = [c for c in cmds if c.rstrip().endswith("--kl-divergence")]
    assert len(base) == 3 and len(kl) == 6 and all("--chunks 80" in c for c in base + kl)
    # KLD と imatrix は自動の載せ方のまま（-ngl を付けない）
    assert all("-ngl" not in c for c in im + base + kl)
    srv = [c for c in cmds if " llama-server " in c]
    # 基準 2 回・Q4_K_M-en・その 2 回目・Q2_K-none。答えの回は全部（基準の 1 回目も）-ngl 30 --fit off
    assert len(srv) == 5 and all("-np 1" in c and "--cache-ram 0" in c and "--seed 1" in c and "-lv 4" in c
                                 and "-ngl 30 --fit off" in c for c in srv)


def test_pipeline_no_place_run_and_bf16_check(tmp_path, capsys):
    cmds = _dry(tmp_path, capsys, "--trial", "--variants", "Q6_K-none", "--bf16-imatrix-check")
    srv = [c for c in cmds if " llama-server " in c]
    assert len(srv) == 3 and all("-ngl" not in c and "--fit" not in c for c in srv)  # 試しは自動（基準 2 回と Q6_K-none だけ）
    im = [c for c in cmds if " llama-imatrix " in c]
    assert len(im) == 3 and "m-bf16.gguf" in im[-1] and "imatrix-en-bf16.gguf" in im[-1]
    kl = [c for c in cmds if c.rstrip().endswith("--kl-divergence")]
    assert len(kl) == 6  # Q6_K-none の 3 つと BF16 の imatrix の Q4_K_M の 3 つ
    cmds = _dry(tmp_path / "b", capsys, "--ngl", "20", "--variants", "Q4_K_M-en", "--no-noise-reruns")
    assert len([c for c in cmds if " llama-server " in c]) == 3


def test_pipeline_production_needs_a_numeric_ngl(tmp_path, capsys):
    bf16, texts = tmp_path / "m-bf16.gguf", tmp_path / "texts"
    base = ["--model", "m", "--bf16", str(bf16), "--texts", str(texts), "--work", str(tmp_path / "w"),
            "--launcher", "bash", "llama_noble.sh", "--dry-run", "--variants", "Q4_K_M-en"]
    with pytest.raises(SystemExit) as e:
        P.main(base)  # 本番（--trial なし）で --ngl が自動
    assert "--ngl" in str(e.value)
    with pytest.raises(SystemExit):
        P.main([*base, "--ngl", "auto"])
    with pytest.raises(SystemExit):
        P.main([*base, "--trial", "--strict-place"])  # 試しでも --strict-place なら数が要る
    assert P.main([*base, "--trial"]) == 0
    assert P.main([*base, "--trial", "--strict-place", "--ngl", "7"]) == 0
    assert P.main([*base, "--ngl", "7"]) == 0


class _FakeTools:
    """起動の代わりに、コマンドを控えてサーバーの記録（載った層の行）を書く。"""
    dry = False

    def __init__(self, log_text: str):
        self.log_text, self.cmds = log_text, []

    def cmd(self, tool, *args):
        return [tool, *map(str, args)]

    def popen(self, cmd, log):
        self.cmds.append(cmd)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(self.log_text, encoding="utf-8")
        return None

    def run(self, cmd, log):
        self.cmds.append(cmd)


def _pipe(tmp_path, log_text, ngl="30", **kw):
    a = SimpleNamespace(work=tmp_path, model="m", ngl=ngl, expect_questions=360, ctx=4096, port=8090,
                        max_tokens=512, ids=None, trial=False, strict_place=False, no_probe=False, **kw)
    t = _FakeTools(log_text)
    return P.Pipeline(a, t), t


def _ngl_of(cmd):
    return cmd[cmd.index("-ngl") + 1] if "-ngl" in cmd else None


def _calls(monkeypatch, rc=0):
    calls = []
    monkeypatch.setattr(P.run_answers, "main", lambda argv: calls.append(argv) or rc)
    monkeypatch.setattr(P.run_answers, "wait_ready", lambda *a, **k: None)
    return calls


def test_pipeline_pins_ngl_on_every_answer_run(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    p, t = _pipe(tmp_path, "load_tensors: offloaded 30/37 layers to GPU\n")
    ref = p.dir / "ref.gguf"
    for name in (f"{P.REF}-r1", f"{P.REF}-r2", "Q6_K-none", "Q4_K_M-en", P.RERUN):
        p.answers(ref, name)
        assert t.cmds[-1][t.cmds[-1].index("-ngl"):][:3] == ["-ngl", "30", "--fit"] and "off" in t.cmds[-1], name
    assert len(calls) == 5
    p._save_ref_offload()
    assert json.loads((p.dir / "ref_offload.json").read_text(encoding="utf-8")) == {"offloaded": [30, 37]}
    # 全部載るモデルも同じ決まり（n = 全部の層）。n が全部の層より大きい指定は、載った層が全部の層なら通る
    p2, t2 = _pipe(tmp_path / "x", "load_tensors: offloaded 37/37 layers to GPU\n", ngl="99")
    p2.answers(p2.dir / "ref.gguf", f"{P.REF}-r1")
    assert _ngl_of(t2.cmds[-1]) == "99"
    p3, t3 = _pipe(tmp_path / "y", "load_tensors: offloaded 37/37 layers to GPU\n", ngl="37")
    p3.answers(p3.dir / "ref.gguf", f"{P.REF}-r1")
    assert _ngl_of(t3.cmds[-1]) == "37"


def test_pipeline_stops_before_questions_when_offload_differs(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    # 指定は 30・記録は 28/37（載り切らなかった）→ 問を送らずに止める（その回の jsonl も作らない）
    p, t = _pipe(tmp_path, "load_tensors: offloaded 28/37 layers to GPU\n")
    with pytest.raises(RuntimeError, match="28/37"):
        p.answers(p.dir / "ref.gguf", f"{P.REF}-r2")
    assert calls == [] and not (p.runs / f"{P.REF}-r2.jsonl").exists()
    # 記録に層の数が無い
    p, t = _pipe(tmp_path / "n", "no layer line\n")
    with pytest.raises(RuntimeError, match="載った層の数が無い"):
        p.answers(p.dir / "ref.gguf", "Q4_K_M-en")
    assert calls == []
    # 本番で --ngl が自動: サーバーも立ち上げない
    p, t = _pipe(tmp_path / "a", "load_tensors: offloaded 30/37 layers to GPU\n", ngl="auto")
    with pytest.raises(SystemExit):
        p.answers(p.dir / "ref.gguf", f"{P.REF}-r1")
    assert t.cmds == [] and calls == []
    # 試しで自動なら通る（-ngl も --fit も付けない・載りの点検もしない）
    p, t = _pipe(tmp_path / "b", "load_tensors: offloaded 28/37 layers to GPU\n", ngl="auto")
    p.a.trial = True
    p.answers(p.dir / "ref.gguf", f"{P.REF}-r1")
    assert "-ngl" not in t.cmds[-1] and "--fit" not in t.cmds[-1] and len(calls) == 1


def test_pipeline_probe_arguments(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    p, t = _pipe(tmp_path, "load_tensors: offloaded 30/37 layers to GPU\n")
    ref = p.dir / "ref.gguf"
    p.answers(ref, f"{P.REF}-r1")
    p.answers(ref, f"{P.REF}-r2")
    p.answers(ref, "Q4_K_M-en")
    p.answers(ref, P.RERUN)
    r1 = str(p.runs / f"{P.REF}-r1.jsonl")
    assert "--probe-self" in calls[0] and "--probe-ref" not in calls[0]
    for c in calls[1:]:
        assert c[c.index("--probe-ref") + 1] == r1 and "--probe-self" not in c
    # --no-probe は全部の回から外す（基準 1 回目も）
    n = len(calls)
    p.a.no_probe = True
    p.answers(ref, "Q3_K_M-en")
    p.answers(ref, f"{P.REF}-r1")
    assert len(calls) == n + 2 and all("--probe-ref" not in c and "--probe-self" not in c for c in calls[n:])
    for c in calls:
        assert c[c.index("--max-tokens") + 1] == "512"


def test_pipeline_skips_only_when_meta_matches(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    monkeypatch.setattr(RA, "manifest", lambda *a, **k: manifest(DATA_DIR_V7, SETS_V7))
    p, t = _pipe(tmp_path, "load_tensors: offloaded 30/37 layers to GPU\n")
    name = "Q4_K_M-en"
    out = p.runs / f"{name}.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("".join(json.dumps({"id": f"q{i}"}) + "\n" for i in range(360)), encoding="utf-8")
    p.answers(p.dir / "ref.gguf", name)
    assert calls and t.cmds  # meta が無ければ飛ばさない（run_answers が判断する）
    calls.clear()
    t.cmds.clear()
    meta = RA.meta_path(out)
    meta.write_text(json.dumps(RA.current_meta_fields(512, False)), encoding="utf-8")
    p.answers(p.dir / "ref.gguf", name)
    assert calls == [] and t.cmds == [] and p.answers_done(name)  # 行の数が足りて指紋も同じ → 飛ばす
    bad = RA.current_meta_fields(512, False)
    bad["manifest"] = {**bad["manifest"], "total": "0" * 64}
    meta.write_text(json.dumps(bad), encoding="utf-8")
    assert not p.answers_done(name)
    # 飛ばさずに run_answers を呼び、それが終了コード 2 で止まれば、ここも止まる
    calls2 = _calls(monkeypatch, rc=2)
    with pytest.raises(RuntimeError, match="終了コード 2"):
        p.answers(p.dir / "ref.gguf", name)
    assert len(calls2) == 1
    # 行が足りなければ meta が同じでも飛ばさない
    meta.write_text(json.dumps(RA.current_meta_fields(512, False)), encoding="utf-8")
    out.write_text(out.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert not p.answers_done(name)
    # max_tokens が違えば飛ばさない
    out.write_text("".join(json.dumps({"id": f"q{i}"}) + "\n" for i in range(360)), encoding="utf-8")
    p.a.max_tokens = 256
    assert not p.answers_done(name)


def test_pipeline_print_offload_sends_no_question(tmp_path, monkeypatch, capsys):
    calls = _calls(monkeypatch)
    p, t = _pipe(tmp_path, "load_tensors: offloaded 28/37 layers to GPU\n", ngl="auto")
    (p.dir).mkdir(parents=True, exist_ok=True)
    p.ref.write_text("x")  # 基準の GGUF は作ってある
    assert p.print_offload() == (28, 37)
    assert calls == [] and len(t.cmds) == 1
    c = t.cmds[0]
    assert c[0] == "llama-server" and "-ngl" not in c and "--fit" not in c  # 自動の載せ方
    assert json.loads(capsys.readouterr().out) == {"offloaded": [28, 37], "ctx": 4096}
    assert (p.logs / f"{P.REF}-print-offload-server.log").exists()


GOOD = {q.id: (HOLD if q.should_hold else q.expected) for q in QS}
REF_TOP = _top((1, 0.6), (2, 0.3), (3, 0.05))  # 基準 1 回目の最初の 1 語の上位
Q_TOP = _top((1, 0.5), (2, 0.35), (4, 0.1))  # 変種
KW = {"strict": True, "min_chunks": 80, "min_imatrix_chunks": 128}


def _write_run(path: Path, answers: dict[str, str], top: list[dict] | None = None) -> None:
    """答えの jsonl と、同じ名前の meta（今の指紋）を書く。top は確かめの上位（probe）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, a in answers.items():
        r = {"id": i, "text": f"理由。\n結論: {a}", "finish_reason": "stop"}
        if top is not None:
            r.update({"probe": top, "probe_prompt_tokens": 50, "usage": {"prompt_tokens": 50}})
        rows.append(json.dumps(r, ensure_ascii=False) + "\n")
    path.write_text("".join(rows), encoding="utf-8")
    path.with_suffix(".meta.json").write_text(json.dumps(
        {"prompt_version": PROMPT_VERSION, "manifest": manifest(DATA_DIR_V7, SETS_V7), "gen": RA.GEN, "max_tokens": 512,
         "smoke": False, "probe": "self" if top is not None else None,
         "probe_gen": RA.probe_gen() if top is not None else None}), encoding="utf-8")


def _server_log(m: Path, name: str, off: tuple[int, int] | None = (37, 37)) -> None:
    (m / "logs").mkdir(parents=True, exist_ok=True)
    text = "load_tensors: offloading output layer to GPU\n" + (
        f"load_tensors: offloaded {off[0]}/{off[1]} layers to GPU\n" if off else "")
    (m / "logs" / f"{name}-server.log").write_text(text, encoding="utf-8")


def _ref_offload(m: Path, off=(37, 37)) -> None:
    (m / "ref_offload.json").write_text(json.dumps({"offloaded": list(off)}), encoding="utf-8")


def _model_dir(tmp_path: Path, answers: dict[str, str] | None = None, chat: bool = True,
               variants: tuple[str, ...] = ("Q4_K_M-en",)) -> Path:
    m = tmp_path / "m"
    for n in (f"{P.REF}-r1", f"{P.REF}-r2", *variants, *([P.RERUN] if "Q4_K_M-en" in variants else [])):
        is_ref = n.startswith(P.REF)
        _write_run(m / "runs" / f"{n}.jsonl", answers if answers is not None and is_ref else GOOD,
                   (REF_TOP if is_ref else Q_TOP) if chat else None)
        _server_log(m, n)
    (m / "logs").mkdir(parents=True, exist_ok=True)
    for x in P.TEXTS:
        for v in variants:
            (m / "logs" / f"{v}-kld-{x}.log").write_text(KLD_LOG, encoding="utf-8")
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
    assert j["by_metric"]["kld_en"]["verdict"] == "差が出ない" and j["by_metric"]["kld_chat"]["exploratory"]
    assert res["base_chunks"] == {"en": 80, "ja": 80, "dom": 80} and res["imatrix_chunks"] == {"en": 128, "ja": 128}
    # 変種ごとの kld_chat（「結論:」の直後の 1 語の分布の、基準 1 回目との KLD）と、答えの回の載った層
    chat = res["metrics"]["Q4_K_M-en"]["kld_chat"]
    assert chat["n"] == 360 and chat["same_top"] == 1.0 and chat["common_mean"] == 2.0
    assert chat["other_p_mean"] == pytest.approx(0.1)  # 基準の共通でない語（語 3 の 0.05 と上位に無い分）
    assert chat["kld_mean"] == pytest.approx(question_kld(_dist(*[(t["id"], t["p"]) for t in REF_TOP]),
                                                          _dist(*[(t["id"], t["p"]) for t in Q_TOP]), {1, 2}))
    assert j["by_metric"]["kld_chat"]["deviation"]["Q4_K_M-en"] == pytest.approx(chat["kld_mean"])
    assert set(res["answer_offload"]) == {f"{P.REF}-r1", f"{P.REF}-r2", "Q4_K_M-en", P.RERUN}
    assert all(v == [37, 37] for v in res["answer_offload"].values())
    assert "kld_chat" not in res["metrics"].get(P.BF16IMX, {})
    no_off = collect_model(m, **KW)
    assert no_off["status"] == "データ欠損" and "載り" in no_off["judge_error"]
    _ref_offload(m)
    strict = collect_model(m, **KW)
    assert strict["status"] == "データ欠損" and "15" in strict["judge_error"]
    assert "区切り" in collect_model(m, strict=True, min_chunks=100)["judge_error"]
    assert "imatrix" in collect_model(m, strict=True, min_chunks=80, min_imatrix_chunks=200)["judge_error"]


def test_report_ref_quality_in_judge_and_markdown(tmp_path):
    m = _model_dir(tmp_path)
    res = collect_model(m)
    q = res["judge"]["ref_quality"][0]
    assert q["c_sum"] == 24 * 12 and q["h_sum"] == 24 * 36 and q["p_direction"] < 1e-6 and q["p_hold"] < 1e-6
    md = markdown("m", res, False)
    assert "c 合計 288" in md and "h 合計 864" in md and "p_direction" in md and "p_hold" in md
    assert "KLD 会話" in md and "答えの回の載った層: 4 回とも 37/37 層" in md


def test_report_strict_checks_completion(tmp_path):
    m = _model_dir(tmp_path)
    _ref_offload(m)
    kw = KW
    (m / "logs" / "Q4_K_M-en-kld-ja.log").write_text(KLD_HEAD + _kld_rows(79), encoding="utf-8")
    e = collect_model(m, **kw)["judge_error"]
    assert "Q4_K_M-en" in e and "済んでいない" in e
    (m / "logs" / "Q4_K_M-en-kld-ja.log").write_text(KLD_LOG, encoding="utf-8")
    (m / "logs" / f"{P.REF}-base-en.log").write_text(_progress_log("perplexity", 80, final=False), encoding="utf-8")
    assert "比べ元 en" in collect_model(m, **kw)["judge_error"]
    (m / "logs" / f"{P.REF}-base-en.log").write_text(_progress_log("perplexity", 80), encoding="utf-8")
    (m / "logs" / "imatrix-ja.log").write_text(_progress_log("compute_imatrix", 128, done=100), encoding="utf-8")
    assert "imatrix ja" in collect_model(m, **kw)["judge_error"]


def _full_model_dir(tmp_path: Path, chat: bool = True) -> Path:
    """15 変種・基準 2 回・Q4_K_M-en の 2 回目がそろった作業フォルダ（本番の点検を全部通す形）。"""
    m = _model_dir(tmp_path, chat=chat, variants=tuple(P.ALL_VARIANTS))
    _ref_offload(m)
    return m


def test_report_chat_kld_missing_does_not_stop_strict(tmp_path):
    # 確かめが 1 つも無い（--no-probe で流した）: 厳しく見る時も止めず、kld_chat だけ「値が無い」・ほかの物差しは出す
    m = _full_model_dir(tmp_path / "a", chat=False)
    res = collect_model(m, **KW)
    assert res["status"] == "適格" and res["judge"]["valid"]
    assert res["judge"]["by_metric"]["kld_chat"]["verdict"] == "値が無い"
    assert res["judge"]["by_metric"]["kld_en"]["verdict"] == "差が出ない"
    chat = res["metrics"]["Q4_K_M-en"]["kld_chat"]
    assert chat["kld_mean"] is None and chat["n"] == 0 and "342" in chat["reason"]
    md = markdown("m", res, False)
    assert "会話の KLD は取れなかった" in md and "342" in md
    # 全体の答えは、kld_chat が無いモデルが混ざっても、ほかの物差しで出る（kld_chat は「値が無い」を数えない）
    judged = {f"m{i}": res["judge"] for i in range(5)}
    assert J.judge_overall(judged, "kld_en") == "測れず"  # 全部「差が出ない」
    assert J.judge_overall(judged, "kld_chat") == "測れず"
    # 取れている時は表に使えた問と共通の語・その他の確率が並ぶ
    ok = collect_model(_full_model_dir(tmp_path / "b"), **KW)
    assert ok["status"] == "適格" and ok["judge"]["by_metric"]["kld_chat"]["verdict"] == "差が出ない"
    assert "会話の KLD は取れなかった" not in markdown("m", ok, False)
    assert "共通の語の数の平均 2.0" in markdown("m", ok, False)
    # 基準 1 回目の確かめが 341 問しか使えない（342 未満）: 全変種で値なし・止めない
    m2 = _full_model_dir(tmp_path / "c")
    rows = [json.loads(l) for l in (m2 / "runs" / f"{P.REF}-r1.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows[:19]:
        r["probe"] = None
    (m2 / "runs" / f"{P.REF}-r1.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    low = collect_model(m2, **KW)
    assert low["status"] == "適格" and low["metrics"]["Q4_K_M-en"]["kld_chat"]["n"] == 341
    assert all(v["kld_chat"]["kld_mean"] is None for n, v in low["metrics"].items() if n in P.ALL_VARIANTS)
    assert low["judge"]["by_metric"]["kld_chat"]["verdict"] == "値が無い"
    # 変種の 1 つに確かめが無い問がある: その変種だけ値なし → 物差し全体が「値が無い」
    m3 = _full_model_dir(tmp_path / "d")
    rows = [json.loads(l) for l in (m3 / "runs" / "Q2_K-en.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[3]["probe"] = None
    (m3 / "runs" / "Q2_K-en.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    part = collect_model(m3, **KW)
    assert part["status"] == "適格" and part["metrics"]["Q2_K-en"]["kld_chat"]["kld_mean"] is None
    assert part["metrics"]["Q6_K-en"]["kld_chat"]["kld_mean"] is not None
    assert part["judge"]["by_metric"]["kld_chat"]["verdict"] == "値が無い"


def test_report_chat_kld_off_when_probe_tokens_differ_or_probe_gen_differs(tmp_path):
    ok = collect_model(_full_model_dir(tmp_path / "a"), **KW)
    assert ok["judge"]["by_metric"]["kld_chat"]["verdict"] == "差が出ない"
    assert ok["metrics"]["Q4_K_M-en"]["kld_chat"]["kld_mean"] is not None
    # 変種の 1 回で、確かめの語の数が usage.prompt_tokens と違う問が 2 つ: そのモデルの kld_chat は全変種で値なし・主の判定は同じ
    m = _full_model_dir(tmp_path / "b")
    p = m / "runs" / "Q2_K-en.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    rows[3]["probe_prompt_tokens"] = 51
    rows[7]["usage"] = {"prompt_tokens": 49}
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    bad = collect_model(m, **KW)
    assert bad["status"] == "適格" and bad["judge"]["valid"]
    assert all(v["kld_chat"]["kld_mean"] is None for n, v in bad["metrics"].items() if n in P.ALL_VARIANTS)
    why = bad["metrics"]["Q6_K-en"]["kld_chat"]["reason"]
    assert "Q2_K-en" in why and "2" in why and "prompt_tokens" in why
    assert bad["judge"]["by_metric"]["kld_chat"]["verdict"] == "値が無い"
    assert bad["judge"]["by_metric"]["kld_en"] == ok["judge"]["by_metric"]["kld_en"]  # 主の判定は変わらない
    assert "Q2_K-en" in markdown("m", bad, False) and "会話の KLD は取れなかった" in markdown("m", bad, False)
    # 基準 1 回目の語の数のずれも同じ
    m2 = _full_model_dir(tmp_path / "c")
    p = m2 / "runs" / f"{P.REF}-r1.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    rows[0]["probe_prompt_tokens"] = 60
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    assert f"{P.REF}-r1" in collect_model(m2, **KW)["metrics"]["Q4_K_M-en"]["kld_chat"]["reason"]
    # meta の probe_gen が回の間で違う
    m3 = _full_model_dir(tmp_path / "d")
    mp = m3 / "runs" / "Q5_K_M-ja.meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    mp.write_text(json.dumps({**meta, "probe_gen": {**RA.probe_gen(), "cache_prompt": True}}), encoding="utf-8")
    g = collect_model(m3, **KW)
    assert g["status"] == "適格" and "probe_gen" in g["metrics"]["Q4_K_M-en"]["kld_chat"]["reason"]
    assert g["metrics"]["Q4_K_M-en"]["kld_chat"]["kld_mean"] is None
    assert g["judge"]["by_metric"]["kld_en"] == ok["judge"]["by_metric"]["kld_en"]
    # --no-probe の回（probe が null）は probe_gen の比べに入らない
    m4 = _full_model_dir(tmp_path / "e", chat=False)
    assert collect_model(m4, **KW)["metrics"]["Q4_K_M-en"]["kld_chat"]["reason"].startswith("基準 1 回目の確かめが使えた問が 0")


def test_report_strict_checks_run_meta(tmp_path):
    m = _model_dir(tmp_path)
    _ref_offload(m)
    mp = m / "runs" / "Q4_K_M-en.meta.json"
    good = json.loads(mp.read_text(encoding="utf-8"))
    # 別の問の指紋（manifest）の答えが混ざっている → データ欠損（基準の回でも変種の回でも）
    for name in ("Q4_K_M-en", f"{P.REF}-r2"):
        p = m / "runs" / f"{name}.meta.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps({**meta, "manifest": {**meta["manifest"], "total": "0" * 64}}), encoding="utf-8")
        lost = collect_model(m, **KW)
        assert lost["status"] == "データ欠損" and name in lost["judge_error"] and "指紋" in lost["judge_error"]
        assert collect_model(m)["status"] == "適格"  # 試しでは見ない
        p.write_text(json.dumps(meta), encoding="utf-8")
    # meta が無い
    mp.unlink()
    assert "meta が読めない" in collect_model(m, **KW)["judge_error"]
    mp.write_text(json.dumps({**good, "prompt_version": "qp-0"}), encoding="utf-8")
    assert "渡す文の版" in collect_model(m, **KW)["judge_error"]
    mp.write_text(json.dumps(good), encoding="utf-8")
    assert "15" in collect_model(m, **KW)["judge_error"]  # meta が通れば次の点検（変種が 15 そろっていない）へ進む


def test_report_main_header_has_manifest_version_and_alpha(tmp_path):
    from evaluation.quant_probe import report
    m = _model_dir(tmp_path)
    out = tmp_path / "out"
    assert report.main(["--work", str(m.parent), "--models", "m", "--out", str(out), "--trial"]) == 0
    head = (out / "README.md").read_text(encoding="utf-8").splitlines()[:4]
    text = "\n".join(head)
    assert J.JUDGE_VERSION in text and "α 0.0125" in text and MANIFEST_TOTAL in text and "0.00625" in text


def test_report_strict_answer_offload_must_match_reference(tmp_path):
    m = _model_dir(tmp_path)
    _ref_offload(m, (30, 37))
    for n in (f"{P.REF}-r1", f"{P.REF}-r2", "Q4_K_M-en", P.RERUN):
        _server_log(m, n, (30, 37))
    assert "15" in collect_model(m, **KW)["judge_error"]  # 載った層が全部同じなら、載り切らなくても載りの点検は通る
    # 答えの回の 1 つだけ載った層が違う
    _server_log(m, "Q4_K_M-en", (37, 37))
    e = collect_model(m, **KW)["judge_error"]
    assert "Q4_K_M-en" in e and "載った層" in e and "37/37" in e and "30/37" in e
    # 記録に層の数が無い回（基準の 2 回目・揺れの組）
    _server_log(m, "Q4_K_M-en", (30, 37))
    for n in (f"{P.REF}-r2", P.RERUN):
        _server_log(m, n, None)
        e = collect_model(m, **KW)["judge_error"]
        assert n in e and "載った層の数が無い" in e
        _server_log(m, n, (30, 37))
    (m / "logs" / "Q4_K_M-en-server.log").unlink()  # 記録そのものが無い回
    assert "Q4_K_M-en" in collect_model(m, **KW)["judge_error"]
    _server_log(m, "Q4_K_M-en", (30, 37))
    # ref_offload.json と基準 1 回目の記録が食い違う
    _server_log(m, f"{P.REF}-r1", (37, 37))
    assert "載った層" in collect_model(m, **KW)["judge_error"]
    # 試しでは点検しない（表に並べるだけ）
    res = collect_model(m)
    assert res["status"] == "適格" and res["answer_offload"][f"{P.REF}-r1"] == [37, 37]
    assert "答えの回の載った層" in markdown("m", res, True)


def test_report_status_unconvertible_and_weak_ref(tmp_path):
    m = _model_dir(tmp_path, answers={q.id: ("A" if q.should_hold else HOLD) for q in QS})
    weak = collect_model(m, **KW)
    assert weak["status"] == "基準不適格" and not weak["judge"]["valid"]
    assert all(v["verdict"] == "測れない" for v in weak["judge"]["by_metric"].values())
    assert "p_direction" in markdown("m", weak, False)
    # 形式は読めるが、いつも保留（c も h も 0・p = 1）: 0b・0c で割る
    m2 = _model_dir(tmp_path / "h", answers={q.id: HOLD for q in QS})
    hold = collect_model(m2, **KW)
    assert hold["status"] == "基準不適格" and hold["judge"]["ref_quality"][0]["format"] == 0
    assert hold["judge"]["ref_quality"][0]["p_direction"] == 1.0 and hold["judge"]["ref_quality"][0]["p_hold"] == 1.0
    # 基準不適格なら、壊れた変種の記録があっても読まずに返す
    (m / "runs" / "Q2_K-en.jsonl").write_text("{壊れた行\n", encoding="utf-8")
    assert collect_model(m, **KW)["status"] == "基準不適格"
    # 答えがあるのに置いた変換不能の印は受け付けない
    (m / "unconvertible.txt").write_text("変換で落ちる", encoding="utf-8")
    late = collect_model(m, **KW)
    assert late["status"] == "データ欠損" and "替え" in late["judge_error"]
    assert "データ欠損" in markdown("m", late, False)
    fresh = tmp_path / "fresh"
    (fresh / "runs").mkdir(parents=True)
    (fresh / "unconvertible.txt").write_text("変換で落ちる", encoding="utf-8")
    gone = collect_model(fresh, **KW)
    assert gone["status"] == "変換不能" and gone["reason"] == "変換で落ちる"
    assert collect_model(fresh)["status"] != "変換不能"  # 試しでは印を見ない


def test_report_broken_record_is_missing_data(tmp_path):
    m = _model_dir(tmp_path)
    _ref_offload(m)
    (m / "runs" / "Q2_K-en.jsonl").write_text("{壊れた行\n", encoding="utf-8")
    res = collect_model(m, strict=True)
    assert res["status"] == "データ欠損" and "読めない" in res["judge_error"]
    assert "データ欠損" in markdown("m", res, False)
    (m / "runs" / "Q2_K-en.jsonl").unlink()
    (m / "ref_offload.json").write_text("{", encoding="utf-8")
    assert "読めない" in collect_model(m, strict=True)["judge_error"]


def test_report_strict_keeps_minimum_inside(tmp_path):
    m = _model_dir(tmp_path)
    _ref_offload(m)
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
    _ref_offload(m)
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
    res = collect_model(m, **KW)
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


def test_planned_models_and_spares():
    assert PLANNED == ("qwen3-8b", "gemma4-12b", "swallow-8b", "ministral3-8b", "nemotron-ja-9b")
    assert SPARES == ("apertus-8b", "qwen3-14b")


def test_overall_block_and_spares():
    ok = {m: _ok() for m in PLANNED}
    gone = lambda *ms: {**ok, **{m: {"status": "変換不能", "reason": "x"} for m in ms}}
    assert overall_block(ok) is None and set(overall_models(ok)) == set(PLANNED)
    lost = {**ok, PLANNED[0]: {"status": "データ欠損", "judge_error": "答えの無い問が 3"}}
    assert "データ欠損" in overall_block(lost) and "答えの無い" in overall_block(lost)
    assert "決めたモデル" in overall_block({m: ok[m] for m in PLANNED[:4]})
    assert "決めたモデル" in overall_block({**ok, "other": _ok()})
    # 変換不能が無いのに替えを付けたら出さない
    assert "前から順" in overall_block({**ok, SPARES[0]: _ok()})
    # 枠 5 だけ変換不能 → 替えの 1 つ目を使う
    one = {**gone(PLANNED[4]), SPARES[0]: _ok()}
    assert overall_block(one) is None
    use = overall_models(one)
    assert PLANNED[4] not in use and SPARES[0] in use and len(use) == 5
    # 枠 4 だけ変換不能でも同じ（替えは前から順）
    assert overall_block({**gone(PLANNED[3]), SPARES[0]: _ok()}) is None
    assert "前から順" in overall_block({**gone(PLANNED[3]), SPARES[1]: _ok()})  # 2 つ目だけは使えない
    assert "前から順" in overall_block(gone(PLANNED[3]))  # 替えを入れていない
    # 枠 4・5 の両方 → 替えを 2 つとも
    two = {**gone(PLANNED[3], PLANNED[4]), SPARES[0]: _ok(), SPARES[1]: _ok()}
    assert overall_block(two) is None and len(overall_models(two)) == 5 and SPARES[1] in overall_models(two)
    assert "前から順" in overall_block({**gone(PLANNED[3], PLANNED[4]), SPARES[0]: _ok()})
    # 枠 1〜3 の変換不能は許さない（枠 4・5 の替えを付けても）
    for k in range(3):
        assert "枠 4・5" in overall_block({**gone(PLANNED[k]), SPARES[0]: _ok()})
    assert "枠 4・5" in overall_block({**gone(PLANNED[0], PLANNED[4]), SPARES[0]: _ok(), SPARES[1]: _ok()})
    # 変換不能が 3 つ以上は出さない
    assert overall_block({**gone(*PLANNED[2:]), SPARES[0]: _ok(), SPARES[1]: _ok()}) is not None
    # 基準不適格は替えずに「測れない」として残す
    weak = {**ok, PLANNED[1]: {"status": "基準不適格", "judge": _res("測れない", valid=False)}}
    assert overall_block(weak) is None
    use = overall_models(weak)
    assert PLANNED[1] in use and not use[PLANNED[1]]["valid"] and len(use) == 5
    assert "前から順" in overall_block({**weak, SPARES[0]: _ok()})


def test_power_runs_end_to_end():
    from evaluation.quant_probe import power
    import random
    qs, source = power.world_questions()
    assert len(qs) == 360 and "data" in source
    w = power.World(random.Random(1))
    assert power.one_model(w, "null", 0, 24) in power.VERDICTS
    assert power.one_model(w, "null", 0, 24, kind="weak") in power.VERDICTS
    o = power.combine([{v: (1.0 if v == "気づく" else 0.0) for v in power.VERDICTS}] * 5)
    assert o["いいえ（この 360 問で）"] == pytest.approx(1.0)
    info = {}
    assert power.one_model(w, "miss_B", 30, 8, info=info) in power.VERDICTS
    assert info["added"] == 30 and 1 <= info["touched"] <= 8
    assert power.joint(w, [("notice", 0)] * 5, 24) in power.OVERALL
    assert power.joint(w, [("miss_A", 20)] * 2 + [("notice", 0)] * 3, 8, ["weak"] * 4 + ["strong"]) in power.OVERALL
    assert power.N_MODELS == 5


def test_power_reference_kinds_and_line_zero():
    from evaluation.quant_probe import power
    import random
    w = power.World(random.Random(3))
    # 弱い基準: 答えられる問の正答が 0.65 前後・保留できたが 0.47 前後・保留しすぎ 0.15 前後
    stats = []
    for _ in range(30):
        _, o = w.ref(None, "weak")
        ans = [i for i in w.ids if not w.hold[i]]
        hold = [i for i in w.ids if w.hold[i]]
        stats.append((sum(o[i] == "correct" for i in ans) / len(ans), sum(o[i] == "held_ok" for i in hold) / len(hold),
                      sum(o[i] == "over_hold" for i in ans) / len(ans)))
    mean = [sum(s[k] for s in stats) / len(stats) for k in range(3)]
    assert abs(mean[0] - 0.645) < 0.04 and abs(mean[1] - 0.465) < 0.06 and abs(mean[2] - 0.15) < 0.03
    # 当てずっぽうは線 0 を通らない。強い基準は通る
    for g in power.GUESSES:
        assert power.ref_rates(g, 20, 1)["all"] == 0, g
    strong = power.ref_rates("strong", 10, 1)
    assert strong["all"] >= 9 and strong["fmt"] == 10
    weak = power.ref_rates("weak", 10, 1)
    assert 0 <= weak["all"] <= 10 and weak["all"] <= min(weak["dir"], weak["hold"], weak["fmt"])


def test_power_joint_scenes_cover_k_values():
    from evaluation.quant_probe import power
    names = [n for n, _, _ in power.joint_scenes()]
    for k in (1, 2, 3, 5):
        assert f"miss_A d=20 k={k}" in names and f"miss_B d=30 k={k}" in names
    assert "notice k=0" in names and all(len(sc) == 5 for _, sc, _ in power.joint_scenes())


def test_power_main_prints_all_tables(capsys):
    from evaluation.quant_probe import power
    assert power.main(["--sims", "2", "--joint-sims", "1", "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    for key in ("qp-judge-8", "問の並び", "基準が線 0 を通る割合", "当てずっぽう", "弱い基準", "5 モデルの全体の答え",
                "5 モデルを一緒に作った", "弱い基準 4 つ・強い基準 1 つ", "かかった秒", "回数: 1 モデル 2 回・5 モデル一緒 1 回",
                "版 7 の線を通る", "2b.", "同じ担当の 6 規程", "いつも同じ語", "でたらめに保留", "誤った「気づかない」の率"):
        assert key in out, key
    # 3 章は k = 0〜5 の全部
    for k in range(6):
        assert f"| miss_A | 10 | {k} |" in out and f"| miss_B | 30 | {k} |" in out


def test_power_defaults_are_production_counts():
    from evaluation.quant_probe import power
    import inspect
    src = inspect.getsource(power.main)
    assert 'default=10000' in src and 'default=4000' in src and "os.cpu_count() or 1) - 2" in src


def test_power_result_does_not_depend_on_jobs():
    from evaluation.quant_probe import power
    # 同じ種なら、並列の数・チャンクの切り方で整数の合計は 1 つも変わらない
    a = power.rates("miss_B", 20, "set", 30, 7, "weak", jobs=1)
    b = power.run_units({power.one_unit("weak", "miss_B", 20, "set")[0]: ("one", ("weak", "miss_B", 20, "set"), 30)},
                        7, jobs=2, chunk=7)
    uid = power.one_unit("weak", "miss_B", 20, "set")[0]
    assert power.summarize_one(b[uid], 30) == a
    ja = power.joint_rates("x", [("miss_A", 20)] * 2 + [("notice", 0)] * 3, 8, 6, 7, ["weak"] * 4 + ["strong"], jobs=1)
    jb = power.run_units({power.joint_unit("x", [("miss_A", 20)] * 2 + [("notice", 0)] * 3, 8, ["weak"] * 4 + ["strong"])[0]:
                          ("joint", (tuple([("miss_A", 20)] * 2 + [("notice", 0)] * 3), 8, tuple(["weak"] * 4 + ["strong"])), 6)},
                         7, jobs=2, chunk=1)
    assert list(jb.values())[0] == {f"o:{k}": v for k, v in ja.items() if v}
    assert power.ref_rates("one_hold", 12, 3, jobs=1) == power.ref_rates("one_hold", 12, 3, jobs=2)
    # 種が違えば結果は変わる（種が効いている）
    assert power.rates("miss_B", 20, 24, 30, 7, "weak") != power.rates("miss_B", 20, 24, 30, 8, "weak")


def test_power_trial_rng_is_per_trial():
    from evaluation.quant_probe import power
    assert power.trial_rng(1, "u", 5).random() == power.trial_rng(1, "u", 5).random()
    assert power.trial_rng(1, "u", 5).random() != power.trial_rng(1, "u", 6).random()
    assert power.trial_rng(1, "u", 5).random() != power.trial_rng(1, "v", 5).random()


def test_power_fast_path_equals_judge():
    from evaluation.quant_probe import power
    from evaluation.quant_probe.stats import signflip_exact_reference
    import random
    # 速い符号入れ替え検定は、元の動的計画法と同じ値（片側・両側）
    rng = random.Random(1)
    for _ in range(1500):
        k = rng.choice([0, 1, 2, 3, 5, 8, 12, 24])
        m = rng.choice([1, 2, 3, 5, 15])
        d = [rng.randint(-m, m) for _ in range(k)]
        for alt in ("greater", "two-sided"):
            assert signflip_exact(d, alt) == pytest.approx(signflip_exact_reference(d, alt), abs=1e-15, rel=1e-12), (d, alt)
    # judge_model を通した答えも、速い道（本物の関数）と元の遅い検定の差し替えで同じ
    from evaluation.quant_probe import judge as J
    from evaluation.quant_probe import stats as S
    w = power.World(random.Random(11))
    info_a, info_b = {}, {}
    res = []
    for ref in (False, True):
        old = J.signflip_exact
        if ref:
            J.signflip_exact = S.signflip_exact_reference
        try:
            w2 = power.World(random.Random(11))
            res.append([power.one_model(w2, s, d, sp, info=(info_a if not ref else info_b))
                        for s, d, sp in (("miss_A", 20, 24), ("null", 0, 24), ("skew", 0, 24), ("miss_B", 30, 8))])
        finally:
            J.signflip_exact = old
    assert res[0] == res[1] and info_a == info_b


def test_power_new_guess_one_hold_and_v7_line():
    from evaluation.quant_probe import power
    import random
    w = power.World(random.Random(4))
    o = w.guess("one_hold")
    holds = [i for i in w.ids if o[i] in ("held_ok", "over_hold")]
    assert len(holds) == 24  # 規程ごとに 1 問だけ保留
    assert len({w.block[i] for i in holds}) == 24
    # 版 7 の線: 強い基準は通る・弱い基準（保留 0.47・正答 0.65）は通らない・当てずっぽうも通らない
    assert power.ref_rates("strong", 10, 1)["v7"] == 10
    assert power.ref_rates("weak", 10, 1)["v7"] == 0
    assert power.ref_rates("random", 10, 1)["v7"] == 0
    assert power.v7_line_ok({"correct": 0.7, "held": 0.5, "format": 0.05})
    assert not power.v7_line_ok({"correct": 0.69, "held": 0.5, "format": 0.05})
    assert not power.v7_line_ok({"correct": 0.7, "held": 0.5, "format": 0.06})


def test_power_set_spread_and_zero_counts():
    from evaluation.quant_probe import power
    import random
    w = power.World(random.Random(5))
    blocks = w.pick(power.SETSPREAD)
    assert len(blocks) == 6 and len({b.split("/")[0] for b in blocks}) == 1
    info = {}
    power.one_model(w, "miss_A", 20, power.SETSPREAD, info=info)
    assert info["added"] == 20 and info["touched"] <= 6
    assert info["pos"] + info["neg"] + info["zero"] == 24 and info["pair_pos"] + info["pair_neg"] + info["pair_zero"] == 24
    cnt, mean = power.rates("miss_A", 20, power.SETSPREAD, 5, 1)
    assert sum(cnt.values()) == 5 and mean["added"] == 20 and mean["touched"] <= 6


def test_power_null_selection_stats():
    from evaluation.quant_probe import power
    raw = power.run_units({"u": ("one", ("weak", "null", 0, 24), 40)}, 3, jobs=1)["u"]
    assert raw["all:n"] == 40 and raw["pass:n"] <= 40
    assert raw["pass:n"] == 40 - raw.get("v:測れない", 0)
    assert 0 <= raw["all:miss"] <= 40 and raw.get("pass:miss", 0) <= raw["pass:n"]
    assert isinstance(raw["all:delta"], int)  # 整数だけを足す（並列の順に依らない）


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
