"""RAG 評価ランナー (CLI)。

使用例:
    python -m evaluation.run_eval --provider vllm --judge vllm
    python -m evaluation.run_eval --provider vllm --no-judge
    python -m evaluation.run_eval --judge-from evaluation/results/<ts>/raw.jsonl
    python -m evaluation.run_eval --provider vllm --no-judge --rerank off \
        --dataset evaluation/dataset.jsonl --dataset evaluation/v2/numeric_boundary.jsonl \
        --ids Q010,N02 --meta gpu=T4 --out-dir /content/run1

候補を比べるときは採点を 2 回に分ける: 候補ごとに --no-judge で回答を保存し、
採点役だけを vLLM に載せ直してから --judge-from で全候補を同じ採点役で採点する
(--judge だけだと候補自身が自分の答えを採点する)。

ジャッジもローカル LLM (vllm) を用いる。社内文書を含む評価を外部 API に
送らないため、外部プロバイダはジャッジにも使わない。

出力:
    evaluation/results/<UTC timestamp>/raw.jsonl  各質問のスコア
    evaluation/results/<UTC timestamp>/report.md  プロバイダ間比較
    evaluation/results/<UTC timestamp>/run_meta.json  実行条件 (evaluation/run_meta.py)
raw.jsonl は 1 問ごとに書き足すので、途中で止まってもそこまでの回答は残る。
--out-dir を渡すと results/<ts> の代わりにそこへ書く (空でないディレクトリは拒否)。

詳細仕様は evaluation/CRITERIA.md。
"""
from __future__ import annotations

import argparse
import asyncio
import faulthandler
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

from evaluation import judges, metrics, run_meta  # noqa: E402
from evaluation.timing import TimedAnswer, timed_answer  # noqa: E402
from evaluation.types import QuestionResult  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


# ─── ヘルパ ────────────────────────────────────────────────────
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
        # 採点役は同じインスタンスを使い回すので、回答の直後 (採点で上書きされる前) に読む
        provider = await get_provider(provider_name)
        if hasattr(provider, "last_usage"):
            provider.last_usage = None
        t = await _collect_answer_and_chunks(
            item["question"], item["user_access_level"], provider_name
        )
        usage = getattr(provider, "last_usage", None)
        if usage is not None:
            res.prompt_tokens = getattr(usage, "prompt_tokens", None)
            res.completion_tokens = getattr(usage, "completion_tokens", None)
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
    provider_name: str, dataset: list[dict], judge_provider, do_judge: bool, on_result=None
) -> list[QuestionResult]:
    """on_result があれば 1 問ごとに呼ぶ (raw.jsonl へ書き足して、途中で落ちても残すため)。"""
    print(f"\n=== Evaluating provider: {provider_name} ===", flush=True)
    results: list[QuestionResult] = []
    for item in dataset:
        print(f"  [{item['id']}] {item['question'][:40]}…", flush=True)
        r = await evaluate_question(item, provider_name, judge_provider, do_judge)
        results.append(r)
        if on_result is not None:
            on_result(r)
        if r.error:
            print(f"    !! {r.error}", flush=True)
    return results


async def _index_status() -> dict:
    """検索が見る索引の数 (evaluation/colab/corpus_index.py の count_index)。数えられなければ止める。"""
    from evaluation.colab.corpus_index import count_index

    try:
        return await count_index()
    except Exception as exc:
        raise SystemExit(
            f"索引を数えられない (QDRANT_URL={settings.qdrant_url} / DATA_DIR={settings.data_dir}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _check_index(index: dict) -> None:
    """検索で通る文書が 1 件も無ければ止める。

    索引と評価で DATA_DIR・QDRANT_URL・QDRANT_COLLECTION が 1 つでもずれると、点はあっても
    documents 行で全部はじかれ、全問の根拠が消えてモデルの失敗に見える。
    削除済み文書の点が残る本番の索引もあるので、一部だけ合わないのは記録に残すだけで止めない。
    """
    if index["matched"] == 0:
        raise SystemExit(
            f"検索で通る文書が 0 件: {index} (DATA_DIR={settings.data_dir} / QDRANT_URL={settings.qdrant_url})。"
            " 索引を作った時の index.json の env と環境変数を揃える"
        )
    if index["matched"] < index["qdrant_documents"]:
        print(f"[warn] 点はあるが documents 行が無い文書がある: {index}", flush=True)


def _prepare_out_dir(out_dir: str | None, suffix: str = "") -> Path:
    """--out-dir があればそこ (空でなければ拒否)、無ければ results/<UTC ts><suffix>。"""
    if out_dir:
        path = Path(out_dir)
        if path.exists() and any(path.iterdir()):
            raise SystemExit(f"--out-dir {path} は空ではない (前の結果を上書きしないため拒否)")
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = Path(__file__).resolve().parent / "results" / f"{ts}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── 保存済み回答の採点 (2 回目) ───────────────────────────────
async def _judge_from(
    saved: Path, judge_name: str, report: str | None, out_dir_arg: str | None, extras: dict
) -> None:
    from evaluation.rejudge import judge_saved, load_saved_results

    groups = load_saved_results(saved)
    n = sum(len(rs) for rs in groups.values())
    print(f"Loaded {n} saved answers ({len(groups)} runs) from {saved}")
    judge_provider = await _resolve_provider(judge_name)

    out_dir = _prepare_out_dir(out_dir_arg, suffix="-judged")
    meta = run_meta.judged_meta(saved, judge_name, settings.llm_model, sys.argv, extras)
    meta["vllm"] = run_meta.fetch_vllm_models(settings.vllm_base_url)
    run_meta.write_run_meta(out_dir, meta)

    raw_path = out_dir / "raw.jsonl"
    try:
        await judge_saved(groups, judge_provider)
        counts: dict[str, int] = {}
        with raw_path.open("w", encoding="utf-8") as f_raw:
            for label, rs in groups.items():
                for r in rs:
                    row = {**asdict(r), "run_label": label}
                    f_raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[label] = len(rs)
    except BaseException as exc:
        run_meta.write_run_meta(
            out_dir, run_meta.finish_run_meta(meta, {}, error=f"{type(exc).__name__}: {exc}")
        )
        raise
    run_meta.write_run_meta(out_dir, run_meta.finish_run_meta(meta, counts))

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
    print(f"Run meta:       {out_dir / run_meta.META_NAME}")


# ─── メイン ─────────────────────────────────────────────────────
async def _main():
    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument("--provider", help="単一プロバイダ名 (例: vllm)")
    parser.add_argument("--all", action="store_true", help="利用可能な全プロバイダを評価")
    parser.add_argument("--judge", default="vllm", help="LLM-judge に使うプロバイダ (既定 vllm / ローカル固定)")
    parser.add_argument("--no-judge", action="store_true", help="LLM 採点をスキップし性能/構造系のみ計算")
    parser.add_argument("--report", help="report.md の出力先")
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "問の jsonl (既定 evaluation/dataset.jsonl)。繰り返すと順につなげる"
            " (例: --dataset evaluation/dataset.jsonl --dataset evaluation/v2/numeric_boundary.jsonl)。"
            " ファイルをまたいで同じ ID があれば止める"
        ),
    )
    parser.add_argument("--ids", help="流す問の ID をカンマ区切りで (例: Q010,Q039,N02)。無い ID は止める")
    parser.add_argument(
        "--meta",
        action="append",
        metavar="KEY=VALUE",
        help="run_meta.json の extras に残す実行条件 (例: --meta gpu=T4 --meta gpu_util=0.80)。繰り返し可",
    )
    parser.add_argument("--out-dir", help="結果の出力先 (既定 evaluation/results/<UTC ts>)。空でなければ止める")
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

    try:
        extras = run_meta.parse_meta_args(args.meta)
        ids = run_meta.parse_ids(args.ids)
    except ValueError as exc:
        parser.error(str(exc))

    if args.judge_from:
        if args.provider or args.all or args.no_judge or args.rerank or args.dataset or args.ids:
            parser.error(
                "--judge-from は --provider/--all/--no-judge/--rerank/--dataset/--ids と併用しない"
            )
        await _judge_from(Path(args.judge_from), args.judge, args.report, args.out_dir, extras)
        return

    dataset_paths = [Path(d) for d in args.dataset] if args.dataset else [DATASET]
    try:
        dataset, dataset_infos = run_meta.load_questions(dataset_paths, ids)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Loaded {len(dataset)} questions from {', '.join(str(p) for p in dataset_paths)}")

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

    # rerank A/B: on/off どちらで評価するかを決める。
    # --rerank 未指定なら settings.rerank_enabled の現在値をそのまま尊重し、上書きしない。
    if args.rerank is None:
        rerank_modes: list[bool | None] = [None]
    elif args.rerank == "both":
        rerank_modes = [True, False]
    else:
        rerank_modes = [args.rerank == "on"]

    # 索引の確かめ。出力先を作る前に止める
    index = await _index_status()
    _check_index(index)

    # 出力先。実行条件は最初の問を流す前に書く (途中で落ちても条件が残る)
    out_dir = _prepare_out_dir(args.out_dir)
    report_path = Path(args.report) if args.report else (out_dir / "report.md")
    raw_path = out_dir / "raw.jsonl"
    uses_vllm = "vllm" in providers_to_run or (judge_provider is not None and args.judge == "vllm")
    meta = run_meta.build_run_meta(
        argv=sys.argv,
        datasets=dataset_infos,
        ids=ids,
        n_questions=len(dataset),
        providers=providers_to_run,
        rerank_modes=rerank_modes,
        judge=args.judge if judge_provider else None,
        settings=settings,
        extras=extras,
        git=run_meta.git_info(ROOT),
        vllm=run_meta.fetch_vllm_models(settings.vllm_base_url) if uses_vllm else None,
    )
    meta["index"] = index
    run_meta.write_run_meta(out_dir, meta)

    original_rerank_enabled = settings.rerank_enabled

    # 実行
    all_results: dict[str, list[QuestionResult]] = {}
    counts: dict[str, int] = {}
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
                    label = f"{name}{suffix}"
                    counts[label] = 0

                    def _write(r: QuestionResult, label: str = label) -> None:
                        row = {**asdict(r), "run_label": label}
                        f_raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f_raw.flush()
                        counts[label] += 1

                    rs = await evaluate_provider(
                        name, dataset, judge_provider, not args.no_judge, on_result=_write
                    )
                    all_results[label] = rs
        except BaseException as exc:
            run_meta.write_run_meta(
                out_dir,
                run_meta.finish_run_meta(meta, counts, error=f"{type(exc).__name__}: {exc}"),
            )
            raise
        finally:
            settings.rerank_enabled = original_rerank_enabled
    run_meta.write_run_meta(out_dir, run_meta.finish_run_meta(meta, counts))

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
    print(f"Run meta:       {out_dir / run_meta.META_NAME}")


async def main():
    try:
        await _main()
    finally:
        # aiosqlite 0.20.0 (requirements の版) は接続を閉じないとワーカスレッドが残り、
        # main が終わってもプロセスが終わらない (2026-10-20 Colab の測定で 62 問を終えたまま止まった)
        from app.database import close_db
        from app.services.qdrant import close_client

        await close_client()
        await close_db()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        sys.stdout.flush()
        # それでも終了処理が止まったら、60 秒後に全スレッドのスタックを出して rc=1 で終わる (例外で抜けた時も)
        faulthandler.dump_traceback_later(60, exit=True)
