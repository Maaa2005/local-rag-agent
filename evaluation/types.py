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
    first_token_latency: float = 0.0
    total_latency: float = 0.0
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
