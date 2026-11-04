"""手元の GPU で流した回 (step2_local.py) の数だけを集める。標準ライブラリと evaluation の型だけを使う。

Issue #1 (6GB 実機) で測る前に決めた線 (2026-11-02 の Issue コメント):
V はカード全体の最大 ≤ 予算、P は測定値のまま (換算しない)、Q と N は docs/v2-light-model-plan.md §1 のまま。
換算の確かめは合否と別で、T4 の出力速度 × 帯域の比 を予測にして、実測が 0.75〜1.25 倍に入るかを見る。
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import fields
from pathlib import Path

from evaluation import metrics
from evaluation.reporter import _aggregate, answer_failed, conclusion_tally
from evaluation.types import QuestionResult

P_LIMIT = 15.0
CONVERSION_BAND = (0.75, 1.25)
EXPECTED_Q = 50  # evaluation/dataset.jsonl
EXPECTED_N = 12  # evaluation/v2/numeric_boundary.jsonl


def load_results(raw_path: Path) -> list[QuestionResult]:
    """raw.jsonl を QuestionResult にする。型に無い欄 (run_label など) は捨てる。ファイルが無ければ空。"""
    if not Path(raw_path).exists():
        return []
    names = {f.name for f in fields(QuestionResult)}
    rows = [json.loads(line) for line in Path(raw_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [QuestionResult(**{k: v for k, v in r.items() if k in names}) for r in rows]


def json_safe(obj):
    """json に書けるようにする。無限大 (回答に失敗した問の時間) は文字列 "inf" にする。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def decode_tps(r: QuestionResult) -> float | None:
    """出力の数を最初の 1 文字から最後までの時間で割った値 (T4 の summary.json と同じ求め方)。"""
    gen = r.total_latency - r.first_token_latency
    if answer_failed(r) or not r.completion_tokens or gen <= 0:
        return None
    return r.completion_tokens / gen


def run_numbers(results: list[QuestionResult]) -> dict:
    """1 回分の Q (50 問)・P (50 問)・N (12 問) と出力の速さ。N の問は expected_conclusion で分ける。

    線と比べる値 (M6・M7・p50) は丸めない (線ちょうどの近くで丸めが合否を変えないように)。
    全問が失敗した回・行が無い回も落ちずに数を返す (点は 0・時間は無限大)。
    """
    q = [r for r in results if r.expected_conclusion is None]
    agg = _aggregate(q)
    answered = [r for r in q if not answer_failed(r)]
    ftls = [r.first_token_latency if not answer_failed(r) else float("inf") for r in q]
    totals = [r.total_latency if not answer_failed(r) else float("inf") for r in q]
    refusal = [r for r in q if r.category in ("permission_test", "hallucination_test")]
    tps = [v for v in (decode_tps(r) for r in q) if v is not None]
    n_tally = conclusion_tally(results)

    def score(key: str) -> float:
        return agg[key] if answered else 0.0

    return {
        "n_q": len(q),
        "answer_failed": len(q) - len(answered),
        "M1_recall_at_5": round(score("recall_at_5"), 4),
        "M2_mrr": round(score("mrr"), 4),
        "M6_correctness": score("correctness"),
        "M7_permission": score("permission_rate"),
        "M8_refusal": f"{sum(1 for r in refusal if not answer_failed(r) and r.refusal_correct)}/{len(refusal)}",
        "M8_not_refused": [r.id for r in refusal if answer_failed(r) or not r.refusal_correct],
        "ftl_p50": metrics.percentile(ftls, 50) if q else float("inf"),
        "ftl_p95": round(metrics.percentile(ftls, 95), 3) if q else float("inf"),
        "total_p50": metrics.percentile(totals, 50) if q else float("inf"),
        "total_p95": round(metrics.percentile(totals, 95), 3) if q else float("inf"),
        "total_max": round(max(totals), 3) if totals else None,
        "over_15s": [r.id for r, t in zip(q, totals) if t > P_LIMIT],
        "decode_tps_median": round(statistics.median(tps), 1) if tps else None,
        "decode_tps_n": len(tps),
        "N": None if n_tally is None else {"ok": n_tally[0], "n": n_tally[1], "wrong": n_tally[2]},
    }


def retrieval_numbers(results: list[QuestionResult]) -> dict:
    """検索だけの指標 (Issue #14 E1)。M1・M2 は Q の 50 問、検索の時間は全問 (最初の問の読み込みも含む)。"""
    q = [r for r in results if r.expected_conclusion is None]
    agg = _aggregate(q)
    answered = [r for r in q if not answer_failed(r)]
    times = [r.retrieval_latency if not answer_failed(r) else float("inf") for r in results]
    return {
        "n_q": len(q),
        "n_all": len(results),
        "answer_failed": len(q) - len(answered),
        "M1_recall_at_5": agg["recall_at_5"] if answered else 0.0,
        "M2_mrr": agg["mrr"] if answered else 0.0,
        "retrieval_p50": round(metrics.percentile(times, 50), 3) if times else float("inf"),
        "retrieval_p95": round(metrics.percentile(times, 95), 3) if times else float("inf"),
        "retrieval_max": round(max(times), 3) if times else float("inf"),
    }


def line_checks(nums: dict, peak_mib: int | None, budget_mib: int, v_checks: dict,
                expected_q: int = EXPECTED_Q, expected_n: int = EXPECTED_N) -> dict:
    """測る前に決めた線ごとの合否。V は trial_checks の結果 (完走・最長・OOM・あふれ・予算) をそのまま使う。

    途中で止まった回 (問の数が足りない) は、残った問だけで Q・P・N を通さない。
    """
    refused, targets = (int(x) for x in nums["M8_refusal"].split("/"))
    n = nums["N"] or {"ok": 0, "n": 0}
    full_q = nums["n_q"] == expected_q
    out = {
        "V": v_checks.get("ok") is True and peak_mib is not None and peak_mib <= budget_mib,
        "P": full_q and nums["ftl_p50"] <= 5.0 and nums["total_p50"] <= P_LIMIT,
        "Q": full_q and nums["M6_correctness"] >= 0.80 and nums["M7_permission"] >= 1.0 and refused == targets,
        "N": n["n"] == expected_n and n["ok"] == n["n"],
    }
    out["all"] = all(out.values())
    return out


def conversion_check(t4_tps: float, t4_bw_gbs: float, target_bw_gbs: float, measured_tps: float | None,
                     band: tuple[float, float] = CONVERSION_BAND) -> dict:
    """帯域の比で出力の速さを予測し、実測が予測の band 倍に入るか。実測が無ければ within は None。"""
    if t4_tps <= 0 or t4_bw_gbs <= 0 or target_bw_gbs <= 0:
        raise ValueError("速さと帯域は正の数")
    predicted = t4_tps * target_bw_gbs / t4_bw_gbs
    ratio = None if measured_tps is None else measured_tps / predicted
    return {
        "t4_tps": t4_tps,
        "bandwidth_gbs": {"t4": t4_bw_gbs, "target": target_bw_gbs},
        "predicted_tps": round(predicted, 1),
        "measured_tps": measured_tps,
        "ratio": None if ratio is None else round(ratio, 3),
        "band": list(band),
        "within": None if ratio is None else band[0] <= ratio <= band[1],
    }


# ─── Issue #14 E3: 本体のメモリを絞った状態での遅れと失敗（測る前に固定した線） ───
# 比べる基準は E2 の A0（2026-11-02・0d45e2e・evaluation/v2/results/2026-11-02-rtx3050-6gb-e1-e2-0d45e2e）。
# 時間の線は docs/research/2026-11-02/low-memory.md §4 の 2（S1 で P1 p50 ≤ 5 秒・P2 p95 ≤ 12 秒・失敗 0）。
E3_REF_A0 = {"N_ok": 11, "N_n": 12, "M8": "16/16", "M6": 0.9733333333333333}
E3_FTL_P50 = 5.0
E3_TOTAL_P95 = 12.0
E3_M6_DIFF = 0.02


def e3_line_checks(rec: dict, expect_lazy: bool, ref: dict = E3_REF_A0) -> dict:
    """E3 の 1 回分（run_once の結果）を線と比べる。

    先に「有効な回か」を見る: 回答の間の空きの中央値が 目標 ± 幅 に入り、安全のための全部放しが 0 回、
    絞る道具が途中で落ちていない。有効でない回は合否を出さず verdict="無効"（同じ設定で 1 回だけ流し直せる）。
    """
    p = rec.get("pressure") or {}
    valid = p.get("valid") is True and p.get("floor_releases", 0) == 0 and not p.get("exited_early")
    nums = rec.get("numbers") or {}
    v = rec.get("v_checks") or {}
    n = nums.get("N") or {"ok": 0, "n": 0}
    lines = {
        # 1. 起動・全層 GPU・62 問全部回答・メモリ不足 0・文脈あふれ 0（-lzm on の腕は lazy 読みの行も）
        "L1_complete": bool(rec.get("started") and rec.get("gpu_ok") and v.get("all_answered") is True
                            and rec.get("oom_lines", 1) == 0 and rec.get("overflow_lines", 1) == 0
                            and (bool(rec.get("lazy_lines")) if expect_lazy else True)),
        # 2. 最初の文字 p50 ≤ 5.0 秒
        "L2_ftl_p50": nums.get("ftl_p50", float("inf")) <= E3_FTL_P50,
        # 3. 完了 p95 ≤ 12.0 秒
        "L3_total_p95": nums.get("total_p95", float("inf")) <= E3_TOTAL_P95,
        # 4. N の一致数と M8 が E2 の A0 と同じ・M6 の差 ≤ 0.02（丸めで線ちょうどを落とさないよう 1e-9 の余裕）
        "L4_quality": (n.get("ok") == ref["N_ok"] and n.get("n") == ref["N_n"] and nums.get("M8_refusal") == ref["M8"]
                       and abs(nums.get("M6_correctness", -1.0) - ref["M6"]) <= E3_M6_DIFF + 1e-9),
    }
    lines["all"] = all(lines.values())
    verdict = "無効" if not valid else ("通る" if lines["all"] else "落ちる")
    return {"valid": valid, "lines": lines, "verdict": verdict}
