"""保存済み回答の採点 (evaluation/rejudge.py) のテスト。"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from evaluation.rejudge import judge_saved, load_saved_results
from evaluation.types import QuestionResult


class _FakeJudge:
    """受け取ったプロンプトを記録し、決まった JSON を返す採点役。"""

    def __init__(self, reply: str = '{"faithfulness": 1.0, "answer_relevance": 0.5, "context_relevance": 0.5}'):
        self.reply = reply
        self.prompts: list[str] = []

    async def stream(self, messages):
        self.prompts.append(messages[-1]["content"])
        yield self.reply


class _BrokenJudge:
    async def stream(self, messages):
        raise RuntimeError("judge down")
        yield  # pragma: no cover


def _row(id_: str, label: str | None = "vllm+rerank_off", **overrides) -> dict:
    r = QuestionResult(
        id=id_,
        provider="vllm",
        question=f"{id_} の質問",
        category="general",
        user_access_level=1,
        answer="回答",
        retrieved_chunks=[{"source_file": "a.md", "content": f"{id_} 本文"}],
        expected_answer_keywords=["本文"],
    )
    row = {**asdict(r), **overrides}
    if label is not None:
        row["run_label"] = label
    return row


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_load_groups_by_run_label_and_falls_back_to_provider(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("Q1"), _row("Q2", label="vllm+rerank_on"), _row("Q3", label=None)])

    groups = load_saved_results(p)

    assert {k: [r.id for r in v] for k, v in groups.items()} == {
        "vllm+rerank_off": ["Q1"],
        "vllm+rerank_on": ["Q2"],
        "vllm": ["Q3"],
    }


def test_load_ignores_unknown_keys(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("Q1", extra_field="x")])

    groups = load_saved_results(p)

    assert groups["vllm+rerank_off"][0].id == "Q1"


@pytest.mark.parametrize("key", ["retrieved_chunks", "expected_answer_keywords"])
def test_load_rejects_rows_without_judge_material(tmp_path, key):
    row = _row("Q1")
    del row[key]
    p = tmp_path / "raw.jsonl"
    _write(p, [row])

    with pytest.raises(ValueError, match=key):
        load_saved_results(p)


def test_judge_saved_scores_with_saved_chunks(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("Q1")])
    groups = load_saved_results(p)
    judge = _FakeJudge()

    asyncio.run(judge_saved(groups, judge))

    r = groups["vllm+rerank_off"][0]
    assert (r.faithfulness, r.answer_relevance, r.context_relevance) == (1.0, 0.5, 0.5)
    assert r.error is None
    assert "Q1 本文" in judge.prompts[0]
    assert "本文" in judge.prompts[0].split("【期待キーワード (参考)】")[1]


def test_judge_saved_skips_rows_that_failed_to_answer(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("Q1", error="run_failed: boom"), _row("Q2")])
    groups = load_saved_results(p)
    judge = _FakeJudge()

    asyncio.run(judge_saved(groups, judge))

    q1, q2 = groups["vllm+rerank_off"]
    assert q1.error == "run_failed: boom"
    assert q1.faithfulness == 0.0
    assert len(judge.prompts) == 1
    assert q2.faithfulness == 1.0


def test_judge_saved_overwrites_previous_scores(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("Q1", faithfulness=0.2, answer_relevance=0.2, context_relevance=0.2)])
    groups = load_saved_results(p)

    asyncio.run(judge_saved(groups, _FakeJudge('{"faithfulness": 0.5}')))

    r = groups["vllm+rerank_off"][0]
    assert (r.faithfulness, r.answer_relevance, r.context_relevance) == (0.5, 0.0, 0.0)


def test_judge_saved_marks_judge_failure(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("Q1")])
    groups = load_saved_results(p)

    asyncio.run(judge_saved(groups, _BrokenJudge()))

    r = groups["vllm+rerank_off"][0]
    assert r.error == "judge_failed: judge down"
    assert r.faithfulness == 0.0


def test_load_keeps_numeric_conclusion(tmp_path):
    p = tmp_path / "raw.jsonl"
    _write(p, [_row("N01", expected_conclusion="必要", conclusion_ok=False)])

    r = load_saved_results(p)["vllm+rerank_off"][0]

    assert (r.expected_conclusion, r.conclusion_ok) == ("必要", False)
