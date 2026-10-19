"""評価結果を Markdown レポートに整形する。

CRITERIA.md の合格基準と照合し ✓/✗ を付ける。
"""
from __future__ import annotations

from datetime import datetime, timezone

from evaluation import metrics

# CRITERIA.md §1 の合格基準
THRESHOLDS = {
    "recall_at_5": ("≥ 0.80", lambda v: v >= 0.80),
    "mrr":         ("≥ 0.60", lambda v: v >= 0.60),
    "context_relevance": ("≥ 0.70", lambda v: v >= 0.70),
    "faithfulness": ("≥ 0.90", lambda v: v >= 0.90),
    "answer_relevance": ("≥ 0.80", lambda v: v >= 0.80),
    "correctness": ("≥ 0.80", lambda v: v >= 0.80),
    "permission_rate": ("= 1.00", lambda v: v >= 1.0),
    "refusal_rate":    ("≥ 0.95", lambda v: v >= 0.95),
    "ftl_p50":         ("≤ 5.0s", lambda v: v <= 5.0),
    "ftl_p95":         ("≤ 12.0s", lambda v: v <= 12.0),
    "total_p50":       ("≤ 15.0s", lambda v: v <= 15.0),
}

# docs/v2-light-model-plan.md の P: T4 で測った時間を 8GB カード (RTX 4060) の時間に換算する。
# 文章を作る速さはメモリ帯域で決まるので、生成の部分 (最初の文字〜完了) だけに 320÷272 GB/s を掛ける。
# 検索と入力の処理 (質問〜最初の文字) は 4060 の方が遅くならないので、そのまま足す。
T4_TO_8GB_DECODE = 320 / 272
TARGET_TOTAL_LIMIT = 15.0


def predicted_8gb_total(r) -> float:
    """T4 で測った 1 問の完了時間を 8GB カードの完了時間に換算する。"""
    return r.first_token_latency + T4_TO_8GB_DECODE * (r.total_latency - r.first_token_latency)


def _aggregate(results: list) -> dict:
    """1 プロバイダの QuestionResult 配列を集計。"""
    if not results:
        return {}
    nonerror = [r for r in results if not r.error]
    if not nonerror:
        return {"all_errored": True, "error_count": len(results)}

    refusal_targets = [
        r for r in nonerror
        if getattr(r, "category", "") in ("permission_test", "hallucination_test")
    ]
    refusal_rate = (
        sum(1 for r in refusal_targets if r.refusal_correct) / len(refusal_targets)
        if refusal_targets else 1.0
    )

    ftls = [r.first_token_latency for r in nonerror]
    totals = [r.total_latency for r in nonerror]

    return {
        "n": len(nonerror),
        "errors": len(results) - len(nonerror),
        "recall_at_5":     metrics.mean([r.recall_at_5 for r in nonerror]),
        "mrr":             metrics.mean([r.mrr for r in nonerror]),
        "context_relevance": metrics.mean([r.context_relevance for r in nonerror]),
        "faithfulness":    metrics.mean([r.faithfulness for r in nonerror]),
        "answer_relevance": metrics.mean([r.answer_relevance for r in nonerror]),
        "correctness":     metrics.mean([r.correctness for r in nonerror]),
        "permission_rate": metrics.mean([1.0 if r.permission_ok else 0.0 for r in nonerror]),
        "refusal_rate":    refusal_rate,
        "ftl_p50":         metrics.percentile(ftls, 50),
        "ftl_p95":         metrics.percentile(ftls, 95),
        "total_p50":       metrics.percentile(totals, 50),
    }


def _check(metric: str, value: float) -> str:
    _, ok = THRESHOLDS[metric]
    return f"{value:.2f} {'✓' if ok(value) else '✗'}"


def _check_secs(metric: str, value: float) -> str:
    _, ok = THRESHOLDS[metric]
    return f"{value:.2f}s {'✓' if ok(value) else '✗'}"


def render_report(
    all_results: dict[str, list],
    judge_name: str | None,
    dataset_size: int,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append("# RAG 評価レポート")
    lines.append("")
    lines.append(f"- 実行日時: {ts}")
    lines.append(f"- 質問数: {dataset_size}")
    lines.append(f"- ジャッジ LLM: {judge_name or '(--no-judge)'}")
    lines.append(f"- 評価対象プロバイダ: {', '.join(all_results.keys()) or '(なし)'}")
    lines.append("")
    lines.append("詳細仕様は [`CRITERIA.md`](../../CRITERIA.md) を参照。")
    lines.append("")

    # ─── プロバイダ別サマリ表 ─────────────────────────────────
    lines.append("## サマリ (プロバイダ × メトリクス)")
    lines.append("")
    header = (
        "| Provider | Recall@5 | MRR | Context Rel. | Faithful. | Ans Rel. | "
        "Correct. | Permission | Refusal | FTL p50 | FTL p95 | Total p50 | 総合 |"
    )
    sep = "|---" * 13 + "|"
    lines.append(header)
    lines.append(sep)
    for name, results in all_results.items():
        agg = _aggregate(results)
        if not agg or agg.get("all_errored"):
            label = "NO DATA" if not agg else "ALL ERRORED"
            lines.append(f"| {name} | {label} | | | | | | | | | | | ✗ |")
            continue
        verdict = "✓" if all(
            THRESHOLDS[k][1](agg[k]) for k in (
                "recall_at_5", "mrr", "context_relevance", "faithfulness",
                "answer_relevance", "correctness",
                "permission_rate", "refusal_rate",
                "ftl_p50", "ftl_p95", "total_p50",
            )
        ) else "✗"
        lines.append(
            f"| {name} "
            f"| {_check('recall_at_5', agg['recall_at_5'])} "
            f"| {_check('mrr', agg['mrr'])} "
            f"| {_check('context_relevance', agg['context_relevance'])} "
            f"| {_check('faithfulness', agg['faithfulness'])} "
            f"| {_check('answer_relevance', agg['answer_relevance'])} "
            f"| {_check('correctness', agg['correctness'])} "
            f"| {_check('permission_rate', agg['permission_rate'])} "
            f"| {_check('refusal_rate', agg['refusal_rate'])} "
            f"| {_check_secs('ftl_p50', agg['ftl_p50'])} "
            f"| {_check_secs('ftl_p95', agg['ftl_p95'])} "
            f"| {_check_secs('total_p50', agg['total_p50'])} "
            f"| {verdict} |"
        )
    lines.append("")

    # ─── 8GB カードへの換算 (T4 で測ったときだけ意味がある) ───────────
    lines.append("## 8GB カードへの換算（T4 で測ったときだけ使う）")
    lines.append("")
    lines.append(
        f"8GB 換算 = 最初の文字まで + {T4_TO_8GB_DECODE:.2f} × (最初の文字〜完了)。"
        "線は換算後の中央値 ≤ 15 秒（docs/v2-light-model-plan.md の P）。"
    )
    lines.append("")
    lines.append("| Provider | 検索 p50 | 最初の文字 p50 | 完了 p50（測定値） | 完了 p50（8GB 換算） | 8GB 換算で 15 秒超 | 8GB 換算の最大 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, results in all_results.items():
        nonerror = [r for r in results if not r.error]
        if not nonerror:
            lines.append(f"| {name} | NO DATA | | | | | |")
            continue
        pred = [predicted_8gb_total(r) for r in nonerror]
        pred_p50 = metrics.percentile(pred, 50)
        over = sum(1 for v in pred if v > TARGET_TOTAL_LIMIT)
        lines.append(
            f"| {name} "
            f"| {metrics.percentile([r.retrieval_latency for r in nonerror], 50):.2f}s "
            f"| {metrics.percentile([r.first_token_latency for r in nonerror], 50):.2f}s "
            f"| {metrics.percentile([r.total_latency for r in nonerror], 50):.2f}s "
            f"| {pred_p50:.2f}s {'✓' if pred_p50 <= TARGET_TOTAL_LIMIT else '✗'} "
            f"| {over}/{len(pred)} "
            f"| {max(pred):.2f}s |"
        )
    lines.append("")

    # ─── 合格基準凡例 ─────────────────────────────────────────
    lines.append("## 合格基準")
    lines.append("")
    lines.append("| メトリクス | 基準 |")
    lines.append("|---|---|")
    for key, (label, _) in THRESHOLDS.items():
        lines.append(f"| {key} | {label} |")
    lines.append("")

    # ─── 質問別の失敗内訳 ─────────────────────────────────────
    for name, results in all_results.items():
        failed = [r for r in results if r.error or not r.permission_ok or not r.refusal_correct]
        if not failed:
            continue
        lines.append(f"## {name} の失敗ケース")
        lines.append("")
        lines.append("| ID | カテゴリ | 失敗理由 |")
        lines.append("|---|---|---|")
        for r in failed:
            reason = []
            if r.error:
                reason.append(r.error)
            if not r.permission_ok:
                reason.append("権限フィルタ違反")
            if not r.refusal_correct:
                reason.append("拒否すべきだが応答してしまった")
            lines.append(f"| {r.id} | {r.category} | {'; '.join(reason)} |")
        lines.append("")

    return "\n".join(lines) + "\n"
