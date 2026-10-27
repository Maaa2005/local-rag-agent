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


def conclusion_tally(results: list) -> tuple[int, int, list[str]] | None:
    """N の問 (expected_conclusion あり) の (結論が合った数, 問数, 合わなかった ID)。

    回答に失敗した問 (conclusion_ok が None) も合わなかった側に数え、分母から外さない。
    N の問が無ければ None。
    """
    rows = [r for r in results if r.expected_conclusion is not None]
    if not rows:
        return None
    missed = [r.id for r in rows if r.conclusion_ok is not True]
    return len(rows) - len(missed), len(rows), missed


# V: --max-model-len = 入力トークンの最大 + 回答の上限 を 512 の倍数に切り上げる (docs/v2-light-model-plan.md の V)。
# 入力トークン数はトークナイザで変わるので、候補ごとに測った値で決める
ANSWER_MAX_TOKENS = 1024  # settings.llm_max_tokens と同じ
MODEL_LEN_STEP = 512


def needed_max_model_len(results: list) -> tuple[int | None, str | None, int | None, list[str]]:
    """(入力トークンの最大, その問, 必要な max-model-len, トークン数が無い問) を返す。

    1 問でもトークン数が無ければ必要値は None (その問の入力が最大かもしれず、足りるか分からないため)。
    """
    known = [r for r in results if getattr(r, "prompt_tokens", None) is not None]
    missing = [r.id for r in results if getattr(r, "prompt_tokens", None) is None]
    if not known:
        return None, None, None, missing
    top = max(known, key=lambda r: r.prompt_tokens)
    needed = -(-(top.prompt_tokens + ANSWER_MAX_TOKENS) // MODEL_LEN_STEP) * MODEL_LEN_STEP
    return top.prompt_tokens, top.id, (None if missing else needed), missing


INF = float("inf")


def answer_failed(r) -> bool:
    """回答を作れなかった問か。採点役の失敗 (judge_failed) は回答の失敗に数えない。"""
    return bool(r.error) and not r.error.startswith("judge_failed")


def _aggregate(results: list) -> dict:
    """1 プロバイダの QuestionResult 配列を集計。

    回答に失敗した問は分母から外さない (docs/v2-light-model-plan.md の「失敗の数え方」)。
    点は 0、拒否すべき問なら拒否できなかった側、時間は無限大として数える。
    権限 (M7) だけは、回答できた問で見る (失敗した問は何も見せていない)。
    """
    if not results:
        return {}
    failed = [r for r in results if answer_failed(r)]
    answered = [r for r in results if not answer_failed(r)]
    if not answered:
        return {"all_errored": True, "error_count": len(results)}

    def scores(attr: str) -> list[float]:
        return [getattr(r, attr) for r in answered] + [0.0] * len(failed)

    refusal_targets = [
        r for r in results
        if getattr(r, "category", "") in ("permission_test", "hallucination_test")
    ]
    refusal_rate = (
        sum(1 for r in refusal_targets if not answer_failed(r) and r.refusal_correct)
        / len(refusal_targets)
        if refusal_targets else 1.0
    )

    ftls = [r.first_token_latency for r in answered] + [INF] * len(failed)
    totals = [r.total_latency for r in answered] + [INF] * len(failed)

    return {
        "n": len(results),
        "answer_failed": len(failed),
        "judge_failed": sum(1 for r in answered if r.error),
        "recall_at_5":     metrics.mean(scores("recall_at_5")),
        "mrr":             metrics.mean(scores("mrr")),
        "context_relevance": metrics.mean(scores("context_relevance")),
        "faithfulness":    metrics.mean(scores("faithfulness")),
        "answer_relevance": metrics.mean(scores("answer_relevance")),
        "correctness":     metrics.mean(scores("correctness")),
        "permission_rate": metrics.mean([1.0 if r.permission_ok else 0.0 for r in answered]),
        "refusal_rate":    refusal_rate,
        "ftl_p50":         metrics.percentile(ftls, 50),
        "ftl_p95":         metrics.percentile(ftls, 95),
        "total_p50":       metrics.percentile(totals, 50),
    }


def _secs(value: float) -> str:
    return "∞" if value == INF else f"{value:.2f}s"


def _check(metric: str, value: float) -> str:
    _, ok = THRESHOLDS[metric]
    return f"{value:.2f} {'✓' if ok(value) else '✗'}"


def _check_secs(metric: str, value: float) -> str:
    _, ok = THRESHOLDS[metric]
    return f"{_secs(value)} {'✓' if ok(value) else '✗'}"


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
    lines.append(
        "回答に失敗した問も分母に入れる（点は 0・時間は ∞）。採点役が失敗した問があれば総合は「未判定」。"
        "採点なし（--no-judge）の実行では Faithful. などが 0 になるので、総合は出さない。"
    )
    lines.append("")
    header = (
        "| Provider | 回答の失敗 | Recall@5 | MRR | Context Rel. | Faithful. | Ans Rel. | "
        "Correct. | Permission | Refusal | FTL p50 | FTL p95 | Total p50 | 総合 |"
    )
    sep = "|---" * 14 + "|"
    lines.append(header)
    lines.append(sep)
    for name, results in all_results.items():
        agg = _aggregate(results)
        if not agg or agg.get("all_errored"):
            label = "NO DATA" if not agg else "ALL ERRORED"
            lines.append(f"| {name} | {label} | | | | | | | | | | | | ✗ |")
            continue
        if judge_name is None:
            verdict = "採点なし"
        elif agg["judge_failed"]:
            verdict = f"未判定（採点役の失敗 {agg['judge_failed']} 問）"
        else:
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
            f"| {agg['answer_failed']}/{agg['n']} "
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

    # ─── P 速さ (8GB 換算は T4 で測ったときだけ意味がある) ─────────────
    lines.append("## P 速さ（docs/v2-light-model-plan.md の P）")
    lines.append("")
    lines.append(
        f"回答に失敗した問は ∞ として中央値に入れる。8GB 換算 = 最初の文字まで + {T4_TO_8GB_DECODE:.2f} × (最初の文字〜完了)。"
        "線は P1 ≤ 5 秒、完了の中央値 ≤ 15 秒（8GB は換算後・12GB は測定値）。15 秒超の数と最大は記録で、合否に使わない。"
    )
    lines.append("")
    lines.append(
        "| Provider | 回答の失敗 | 検索 p50 | 最初の文字 p50（P1） | 完了 p50（測定値・12GB の線） "
        "| 完了 p50（8GB 換算・8GB の線） | 8GB 換算で 15 秒超（記録） | 8GB 換算の最大（記録） |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, results in all_results.items():
        if not results:
            lines.append(f"| {name} | NO DATA | | | | | | |")
            continue
        answered = [r for r in results if not answer_failed(r)]
        n_failed = len(results) - len(answered)
        ftls = [r.first_token_latency for r in answered] + [INF] * n_failed
        totals = [r.total_latency for r in answered] + [INF] * n_failed
        pred = [predicted_8gb_total(r) for r in answered] + [INF] * n_failed
        ftl_p50 = metrics.percentile(ftls, 50)
        total_p50 = metrics.percentile(totals, 50)
        pred_p50 = metrics.percentile(pred, 50)
        over = sum(1 for v in pred if v > TARGET_TOTAL_LIMIT)
        retrieval = (
            _secs(metrics.percentile([r.retrieval_latency for r in answered], 50)) if answered else "—"
        )
        lines.append(
            f"| {name} "
            f"| {n_failed}/{len(results)} "
            f"| {retrieval} "
            f"| {_secs(ftl_p50)} {'✓' if ftl_p50 <= 5.0 else '✗'} "
            f"| {_secs(total_p50)} {'✓' if total_p50 <= TARGET_TOTAL_LIMIT else '✗'} "
            f"| {_secs(pred_p50)} {'✓' if pred_p50 <= TARGET_TOTAL_LIMIT else '✗'} "
            f"| {over}/{len(pred)} "
            f"| {_secs(max(pred))} |"
        )
    lines.append("")

    # ─── N 数値の境目 (N の問を流したときだけ出す) ──────────────────
    tallies = {name: conclusion_tally(rs) for name, rs in all_results.items()}
    if any(t is not None for t in tallies.values()):
        lines.append("## N 数値の境目（最後の「結論: X」行が正解と一致した数）")
        lines.append("")
        lines.append("線は全問一致（docs/v2-light-model-plan.md の N）。回答に失敗した問も不一致に数える。")
        lines.append("")
        lines.append("| Provider | 一致 | 合わなかった問 |")
        lines.append("|---|---|---|")
        for name, t in tallies.items():
            if t is None:
                continue
            ok, n, missed = t
            lines.append(f"| {name} | {ok}/{n} {'✓' if ok == n else '✗'} | {', '.join(missed) or '—'} |")
        lines.append("")

    # ─── V の max-model-len (トークン数を記録した実行があるときだけ出す) ───────
    if any(getattr(r, "prompt_tokens", None) is not None for rs in all_results.values() for r in rs):
        lines.append("## max-model-len の必要値（docs/v2-light-model-plan.md の V）")
        lines.append("")
        lines.append(
            f"必要値 = 入力トークンの最大 + 回答の上限 {ANSWER_MAX_TOKENS:,} を {MODEL_LEN_STEP} の倍数に切り上げ。"
            "トークナイザで入力の数が変わるので、候補ごとにこの値を使う。トークン数が無い問が 1 つでもあれば「不明」。"
        )
        lines.append("")
        lines.append("| Provider | 入力トークンの最大 | その問 | 必要な max-model-len | トークン数が無い問 |")
        lines.append("|---|---|---|---|---|")
        for name, results in all_results.items():
            top, top_id, needed, missing = needed_max_model_len(results)
            shown = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
            lines.append(
                f"| {name} "
                f"| {f'{top:,}' if top is not None else '—'} "
                f"| {top_id or '—'} "
                f"| {f'{needed:,}' if needed is not None else '不明'} "
                f"| {f'{len(missing)} 問（{shown}）' if missing else '—'} |"
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
