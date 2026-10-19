"""1 問ぶんの検索と生成を流し、時間を測る (run_eval から使う)。

run_eval 本体は重い依存を読み込むので、テストできるよう計測だけをここに分ける。
時計は検索の前に動かす。CRITERIA の P1/P3 は「質問送信〜」で、検索の時間も含むため。
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TimedAnswer:
    chunks: list[dict]
    answer: str
    retrieval_latency: float    # 検索だけ
    first_token_latency: float  # 質問〜最初の文字 (検索込み)
    total_latency: float        # 質問〜完了 (検索込み)


async def timed_answer(retrieve, stream, question: str, level: int, clock=time.perf_counter) -> TimedAnswer:
    """retrieve(question, level) で取得し、stream(question, chunks) の出力を集める。"""
    t0 = clock()
    chunks = await retrieve(question, level)
    retrieval = clock() - t0

    first_token: float | None = None
    buf: list[str] = []
    async for tok in stream(question, chunks):
        if first_token is None:
            first_token = clock() - t0
        buf.append(tok)
    total = clock() - t0
    return TimedAnswer(
        chunks=chunks,
        answer="".join(buf),
        retrieval_latency=retrieval,
        first_token_latency=first_token if first_token is not None else total,
        total_latency=total,
    )
