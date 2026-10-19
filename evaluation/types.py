"""evaluation パッケージ共有型。

run_eval 本体は重い依存 (sentence_transformers, qdrant_client) を引きずるので、
テストとレポーターからは型だけを参照できるよう分離する。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuestionResult:
    id: str
    provider: str
    question: str
    category: str
    user_access_level: int
    retrieved_sources: list[str] = field(default_factory=list)
    answer: str = ""
    # 時間は質問を受けた時点から測る (検索込み・evaluation/timing.py)
    first_token_latency: float = 0.0
    total_latency: float = 0.0
    retrieval_latency: float = 0.0  # 検索だけ。この欄より前の raw.jsonl では 0.0
    error: str | None = None
    # メトリクス
    recall_at_5: float = 0.0
    mrr: float = 0.0
    correctness: float = 0.0
    permission_ok: bool = True
    refusal_correct: bool = True
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    # N (数値の境目): 正解の結論と、最後の「結論: X」行が合ったか。N 以外の問では両方 None
    expected_conclusion: str | None = None
    conclusion_ok: bool | None = None
    # 後から別の実行で採点するための材料 (evaluation/rejudge.py が読む)
    retrieved_chunks: list[dict] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
