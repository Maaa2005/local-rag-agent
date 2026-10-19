"""RAG 評価ランナー (CLI)。

使用例:
    python -m evaluation.run_eval --provider vllm --judge vllm
    python -m evaluation.run_eval --provider vllm --no-judge
    python -m evaluation.run_eval --judge-from evaluation/results/<ts>/raw.jsonl

候補を比べるときは採点を 2 回に分ける: 候補ごとに --no-judge で回答を保存し、
採点役だけを vLLM に載せ直してから --judge-from で全候補を同じ採点役で採点する
(--judge だけだと候補自身が自分の答えを採点する)。

ジャッジもローカル LLM (vllm) を用いる。社内文書を含む評価を外部 API に
送らないため、外部プロバイダはジャッジにも使わない。

出力:
    evaluation/results/<UTC timestamp>/raw.jsonl  各質問のスコア
    evaluation/results/<UTC timestamp>/report.md  プロバイダ間比較

詳細仕様は evaluation/CRITERIA.md。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# backend を import パスへ追加 (このスクリプトは repo ルートから実行する想定)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.services.llm import stream_answer  # noqa: E402
from app.services.providers import (  # noqa: E402
    get_meta,
    get_provider,
    list_providers,
)
from app.services.retriever import retrieve  # noqa: E402

from evaluation import judges, metrics  # noqa: E402
from evaluation.timing import TimedAnswer, timed_answer  # noqa: E402
from evaluation.types import QuestionResult  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


# ─── ヘルパ ────────────────────────────────────────────────────
def load_dataset() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _collect_answer_and_chunks(question: str, level: int, provider_name: str) -> TimedAnswer:
    """retriever で取得 → stream_answer で生成し、レイテンシも計測 (検索込み・evaluation/timing.py)。"""
    def _stream(q: str, chunks: list[dict]):
        return stream_answer(q, chunks, provider_name=provider_name)

    return await timed_answer(retrieve, _stream, question, level)


async def evaluate_question(
    item: dict, provider_name: str, judge_provider, do_judge: bool
) -> QuestionResult:
    res = QuestionResult(
        id=item["id"],
        provider=provider_name,
        question=item["question"],
        category=item.get("category", ""),
        user_access_level=item["user_access_level"],
        expected_conclusion=item.get("expected_conclusion"),
    )
    try:
        t = await _collect_answer_and_chunks(
            item["question"], item["user_access_level"], provider_name
        )
        chunks, answer = t.chunks, t.answer
        sources = [c.get("source_file", "") for c in chunks]
        res.retrieved_sources = sources
        res.retrieved_chunks = [
            {"source_file": c.get("source_file", ""), "content": c.get("content", "")}
            for c in chunks
        ]
        res.expected_answer_keywords = item.get("expected_answer_keywords", [])
        res.answer = answer
        res.first_token_latency = t.first_token_latency
        res.total_latency = t.total_latency
        res.retrieval_latency = t.retrieval_latency

        # M1, M2, M6, M7, M8
        res.recall_at_5 = metrics.recall_at_k(sources, item.get("expected_sources", []), k=5)
        res.mrr = metrics.reciprocal_rank(sources, item.get("expected_sources", []))
        res.correctness = metrics.correctness_keyword_score(answer, item.get("expected_answer_keywords", []))
        res.permission_ok = metrics.permission_ok(chunks, item["user_access_level"])
        if item.get("should_refuse", False):
            res.refusal_correct = metrics.is_refusal(answer)
        else:
            res.refusal_correct = True  # 適用外
        # N: 回答に失敗した問は conclusion_ok が None のまま残り、レポートでは不一致に数える
        if res.expected_conclusion is not None:
            res.conclusion_ok = metrics.numeric_conclusion_ok(answer, res.expected_conclusion)

        # M3, M4, M5 (judge)
        if do_judge and judge_provider is not None:
            try:
                scores = await judges.judge(
                    judge_provider,
                    item["question"],
                    chunks,
                    answer,
                    item.get("expected_answer_keywords", []),
                )
                res.faithfulness = scores.faithfulness
                res.answer_relevance = scores.answer_relevance
                res.context_relevance = scores.context_relevance
            except Exception as exc:
                res.error = f"judge_failed: {exc}"
    except Exception as exc:
        res.error = f"run_failed: {exc}"
    return res


# ─── プロバイダ準備 ─────────────────────────────────────────────
async def _resolve_provider(name: str):
    # ローカル LLM 固定アプリ。外部送信プロバイダ・資格情報は持たない。
    meta = get_meta(name)
    if meta.requires_credentials:
        raise RuntimeError(f"Provider '{name}' は資格情報が必要だが本アプリは未対応")
    return await get_provider(name)


async def _available_providers() -> list[str]:
    return [meta.name for meta in list_providers()]


# ─── 1 プロバイダ評価 ──────────────────────────────────────────
async def evaluate_provider(
    provider_name: str, dataset: list[dict], judge_provider, do_judge: bool
) -> list[QuestionResult]:
    print(f"\n=== Evaluating provider: {provider_name} ===", flush=True)
    results: list[QuestionResult] = []
    for item in dataset:
        print(f"  [{item['id']}] {item['question'][:40]}…", flush=True)
        r = await evaluate_question(item, provider_name, judge_provider, do_judge)
        results.append(r)
        if r.error:
            print(f"    !! {r.error}", flush=True)
    return results


# ─── 保存済み回答の採点 (2 回目) ───────────────────────────────
async def _judge_from(saved: Path, judge_name: str, report: str | None) -> None:
    from evaluation.rejudge import judge_saved, load_saved_results

    groups = load_saved_results(saved)
    n = sum(len(rs) for rs in groups.values())
    print(f"Loaded {n} saved answers ({len(groups)} runs) from {saved}")
    judge_provider = await _resolve_provider(judge_name)
    await judge_saved(groups, judge_provider)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(__file__).resolve().parent / "results" / f"{ts}-judged"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as f_raw:
        for label, rs in groups.items():
            for r in rs:
                row = {**asdict(r), "run_label": label}
                f_raw.write(json.dumps(row, ensure_ascii=False) + "\n")

    from evaluation.reporter import render_report
    dataset_size = max(len(rs) for rs in groups.values()) if groups else 0
    md = render_report(
        groups,
        judge_name=f"{judge_name} ({settings.llm_model})",
        dataset_size=dataset_size,
    )
    report_path = Path(report) if report else (out_dir / "report.md")
    report_path.write_text(md, encoding="utf-8")
    print(f"\nSaved answers:  {saved}")
    print(f"Report written: {report_path}")
    print(f"Raw results:    {raw_path}")


# ─── メイン ─────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument("--provider", help="単一プロバイダ名 (例: vllm)")
    parser.add_argument("--all", action="store_true", help="利用可能な全プロバイダを評価")
    parser.add_argument("--judge", default="vllm", help="LLM-judge に使うプロバイダ (既定 vllm / ローカル固定)")
    parser.add_argument("--no-judge", action="store_true", help="LLM 採点をスキップし性能/構造系のみ計算")
    parser.add_argument("--report", help="report.md の出力先")
    parser.add_argument("--dataset", help="dataset.jsonl のパスを上書き")
    parser.add_argument(
        "--rerank",
        choices=["on", "off", "both"],
        default=None,
        help=(
            "Cross-Encoder リランクの有無を切り替えて評価する (既定: settings.rerank_enabled を尊重)。"
            " 'both' は rerank on/off の両方を実行しレポートに並記する"
        ),
    )
    parser.add_argument(
        "--judge-from",
        help=(
            "--no-judge で保存した raw.jsonl を、--judge の採点役だけで採点する"
            " (回答は作り直さない。全候補を同じ採点役で採点するため)"
        ),
    )
    args = parser.parse_args()

    if args.judge_from:
        if args.provider or args.all or args.no_judge or args.rerank:
            parser.error("--judge-from は --provider/--all/--no-judge/--rerank と併用しない")
        await _judge_from(Path(args.judge_from), args.judge, args.report)
        return

    if args.dataset:
        global DATASET
        DATASET = Path(args.dataset)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} questions from {DATASET}")

    # データベース接続を確立
    from app.database import init_db
    await init_db()

    # プロバイダ決定
    if args.all:
        providers_to_run = await _available_providers()
    elif args.provider:
        providers_to_run = [args.provider]
    else:
        parser.error("--provider または --all を指定してください")

    # ジャッジ
    judge_provider = None
    if not args.no_judge:
        try:
            judge_provider = await _resolve_provider(args.judge)
        except Exception as exc:
            print(f"[warn] judge '{args.judge}' を初期化できません: {exc}")
            print("       --no-judge で性能のみ評価に切り替えるか、judge の資格情報を設定してください")
            return

    # 出力先
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(__file__).resolve().parent / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else (out_dir / "report.md")
    raw_path = out_dir / "raw.jsonl"

    # rerank A/B: on/off どちらで評価するかを決める。
    # --rerank 未指定なら settings.rerank_enabled の現在値をそのまま尊重し、上書きしない。
    if args.rerank is None:
        rerank_modes: list[bool | None] = [None]
    elif args.rerank == "both":
        rerank_modes = [True, False]
    else:
        rerank_modes = [args.rerank == "on"]

    original_rerank_enabled = settings.rerank_enabled

    # 実行
    all_results: dict[str, list[QuestionResult]] = {}
    with raw_path.open("w", encoding="utf-8") as f_raw:
        try:
            for rerank_mode in rerank_modes:
                if rerank_mode is not None:
                    settings.rerank_enabled = rerank_mode
                suffix = ""
                if len(rerank_modes) > 1:
                    suffix = "+rerank_on" if rerank_mode else "+rerank_off"
                    print(f"\n--- rerank_enabled={settings.rerank_enabled} ---", flush=True)
                for name in providers_to_run:
                    try:
                        await _resolve_provider(name)
                    except Exception as exc:
                        print(f"[skip] {name}: {exc}")
                        continue
                    rs = await evaluate_provider(name, dataset, judge_provider, not args.no_judge)
                    all_results[f"{name}{suffix}"] = rs
                    for r in rs:
                        row = {**asdict(r), "run_label": f"{name}{suffix}"}
                        f_raw.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            settings.rerank_enabled = original_rerank_enabled

    # レポート生成
    from evaluation.reporter import render_report
    md = render_report(
        all_results,
        judge_name=args.judge if judge_provider else None,
        dataset_size=len(dataset),
    )
    report_path.write_text(md, encoding="utf-8")
    print(f"\nReport written: {report_path}")
    print(f"Raw results:    {raw_path}")


if __name__ == "__main__":
    asyncio.run(main())
