"""保存済みの回答を、採点役だけを載せた別の実行で採点する (2 回に分ける採点)。

1 回目: run_eval.py --no-judge で回答だけ作り raw.jsonl に保存する。
2 回目: run_eval.py --judge-from <raw.jsonl> で、その答えを採点役が採点する。

候補と採点役が同じ GPU を取り合わず、全候補を同じ採点役で採点できる。
このモジュールは検索・埋め込みを読み込まない (採点役の実行で GPU を使わせない)。
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from evaluation import judges
from evaluation.types import QuestionResult

_FIELDS = {f.name for f in fields(QuestionResult)}
_REQUIRED = ("retrieved_chunks", "expected_answer_keywords")


def load_saved_results(path: Path) -> dict[str, list[QuestionResult]]:
    """raw.jsonl を読み、run_label (無ければ provider) ごとにまとめる。

    採点に必要な retrieved_chunks / expected_answer_keywords が無い行
    (この形式より前の raw.jsonl) があれば ValueError。チャンク無しで採点すると
    faithfulness が 0 に寄り、候補の差ではない差が出るため。
    """
    groups: dict[str, list[QuestionResult]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in _REQUIRED if k not in row]
            if missing:
                raise ValueError(
                    f"{path}:{lineno} に {', '.join(missing)} が無い。"
                    " --no-judge で回答を取り直してから採点する"
                )
            label = row.get("run_label") or row["provider"]
            r = QuestionResult(**{k: v for k, v in row.items() if k in _FIELDS})
            groups.setdefault(label, []).append(r)
    return groups


async def judge_saved(groups: dict[str, list[QuestionResult]], judge_provider) -> None:
    """保存済みの結果をその場で採点する。回答で失敗した行は採点しない。"""
    for label, results in groups.items():
        print(f"\n=== Judging: {label} ===", flush=True)
        for r in results:
            if r.error:
                continue
            r.faithfulness = r.answer_relevance = r.context_relevance = 0.0
            try:
                scores = await judges.judge(
                    judge_provider,
                    r.question,
                    r.retrieved_chunks,
                    r.answer,
                    r.expected_answer_keywords,
                )
            except Exception as exc:
                r.error = f"judge_failed: {exc}"
                print(f"  [{r.id}] !! {r.error}", flush=True)
                continue
            r.faithfulness = scores.faithfulness
            r.answer_relevance = scores.answer_relevance
            r.context_relevance = scores.context_relevance
