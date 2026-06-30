"""RAG 評価ランナー (CLI)。

使用例:
    python -m evaluation.run_eval --provider vllm --judge vllm
    python -m evaluation.run_eval --provider vllm --no-judge

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
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# backend を import パスへ追加 (このスクリプトは repo ルートから実行する想定)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.services.llm import stream_answer  # noqa: E402
from app.services.providers import (  # noqa: E402
    get_meta,
    get_provider,
    list_providers,
)
from app.services.retriever import retrieve  # noqa: E402

from evaluation import judges, metrics  # noqa: E402
from evaluation.types import QuestionResult  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


# ─── ヘルパ ────────────────────────────────────────────────────
def load_dataset() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _collect_answer_and_chunks(question: str, level: int, provider_name: str):
    """retriever で取得 → stream_answer で生成し、レイテンシも計測。"""
    chunks = await retrieve(question, level)
    sources = [c.get("source_file", "") for c in chunks]

    t0 = time.perf_counter()
    first_token: float | None = None
    buf: list[str] = []
    async for tok in stream_answer(question, chunks, provider_name=provider_name):
        if first_token is None:
            first_token = time.perf_counter() - t0
        buf.append(tok)
    total = time.perf_counter() - t0
    return chunks, sources, "".join(buf), (first_token or total), total


async def evaluate_question(
    item: dict, provider_name: str, judge_provider, do_judge: bool
) -> QuestionResult:
    res = QuestionResult(
        id=item["id"],
        provider=provider_name,
        question=item["question"],
        category=item.get("category", ""),
        user_access_level=item["user_access_level"],
    )
    try:
        chunks, sources, answer, ftl, total = await _collect_answer_and_chunks(
            item["question"], item["user_access_level"], provider_name
        )
        res.retrieved_sources = sources
        res.answer = answer
        res.first_token_latency = ftl
        res.total_latency = total

        # M1, M2, M6, M7, M8
        res.recall_at_5 = metrics.recall_at_k(sources, item.get("expected_sources", []), k=5)
        res.mrr = metrics.reciprocal_rank(sources, item.get("expected_sources", []))
        res.correctness = metrics.correctness_keyword_score(answer, item.get("expected_answer_keywords", []))
        res.permission_ok = metrics.permission_ok(chunks, item["user_access_level"])
        if item.get("should_refuse", False):
            res.refusal_correct = metrics.is_refusal(answer)
        else:
            res.refusal_correct = True  # 適用外

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


# ─── メイン ─────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument("--provider", help="単一プロバイダ名 (例: vllm)")
    parser.add_argument("--all", action="store_true", help="利用可能な全プロバイダを評価")
    parser.add_argument("--judge", default="vllm", help="LLM-judge に使うプロバイダ (既定 vllm / ローカル固定)")
    parser.add_argument("--no-judge", action="store_true", help="LLM 採点をスキップし性能/構造系のみ計算")
    parser.add_argument("--report", help="report.md の出力先")
    parser.add_argument("--dataset", help="dataset.jsonl のパスを上書き")
    args = parser.parse_args()

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

    # 実行
    all_results: dict[str, list[QuestionResult]] = {}
    with raw_path.open("w", encoding="utf-8") as f_raw:
        for name in providers_to_run:
            try:
                await _resolve_provider(name)
            except Exception as exc:
                print(f"[skip] {name}: {exc}")
                continue
            rs = await evaluate_provider(name, dataset, judge_provider, not args.no_judge)
            all_results[name] = rs
            for r in rs:
                f_raw.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

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
