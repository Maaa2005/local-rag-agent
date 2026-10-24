"""1 問の計測 (evaluation/timing.py) のテスト。偽の時計で検索込みの時間を確かめる。"""
from __future__ import annotations

import asyncio

from evaluation.timing import timed_answer


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _fakes(clock: _Clock, tokens: list[str]):
    async def retrieve(question, level):
        clock.now += 2.0  # 検索に 2 秒
        return [{"source_file": "a.md", "content": f"{question}/{level}"}]

    async def stream(question, chunks):
        clock.now += 1.0  # 入力の処理に 1 秒
        for tok in tokens:
            yield tok
            clock.now += 1.5  # 1 文字ごとに 1.5 秒

    return retrieve, stream


def test_latencies_include_retrieval():
    clock = _Clock()
    retrieve, stream = _fakes(clock, ["こん", "にちは"])

    t = asyncio.run(timed_answer(retrieve, stream, "Q", 2, clock=clock))

    assert t.retrieval_latency == 2.0
    assert t.first_token_latency == 3.0   # 検索 2 + 入力 1
    assert t.total_latency == 6.0         # 3 + 1.5 × 2
    assert t.answer == "こんにちは"
    assert t.chunks == [{"source_file": "a.md", "content": "Q/2"}]


def test_no_tokens_uses_total_as_first_token():
    clock = _Clock()
    retrieve, stream = _fakes(clock, [])

    t = asyncio.run(timed_answer(retrieve, stream, "Q", 1, clock=clock))

    assert t.answer == ""
    assert t.first_token_latency == t.total_latency == 3.0
