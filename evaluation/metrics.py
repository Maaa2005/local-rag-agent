"""LLM を使わない検索・運用系メトリクス。

純関数として実装。テストしやすい。
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable, Sequence


# ─── 検索メトリクス ──────────────────────────────────────────────
def recall_at_k(retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int = 5) -> float:
    """上位 k 件のチャンクのソースに期待ソースが 1 件以上含まれれば 1。"""
    if not expected_sources:
        return 1.0  # 期待ソースが空 (拒否系) はスキップ扱い
    top_k = set(retrieved_sources[:k])
    return 1.0 if any(e in top_k for e in expected_sources) else 0.0


def reciprocal_rank(retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int = 10) -> float:
    """期待ソースが最初に登場する順位の逆数 (見つからなければ 0)。"""
    if not expected_sources:
        return 1.0
    expected_set = set(expected_sources)
    for i, src in enumerate(retrieved_sources[:k], start=1):
        if src in expected_set:
            return 1.0 / i
    return 0.0


# ─── 運用メトリクス ──────────────────────────────────────────────
def permission_ok(retrieved_chunks: Iterable[dict], user_access_level: int) -> bool:
    """取得チャンクすべての access_level <= user_access_level なら True。"""
    return all(
        (chunk.get("access_level", 0)) <= user_access_level
        for chunk in retrieved_chunks
    )


_REFUSAL_PATTERNS = (
    "情報は現在の資料には含まれていません",
    "資料には含まれていません",
    "情報なし",
    "確認できません",
    "資料に記載がありません",
    "見つかりませんでした",
)


def is_refusal(answer: str) -> bool:
    """応答が「情報なし」系の定型句を含むか。"""
    return any(p in answer for p in _REFUSAL_PATTERNS)


def correctness_keyword_score(answer: str, expected_keywords: Sequence[str]) -> float:
    """期待キーワードのうち回答に含まれている比率。"""
    if not expected_keywords:
        return 1.0
    hits = sum(1 for kw in expected_keywords if kw in answer)
    return hits / len(expected_keywords)


# ─── 集計 ────────────────────────────────────────────────────────
def percentile(values: Sequence[float], p: float) -> float:
    """p ∈ [0,100] のパーセンタイル (線形補間)。空なら 0。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ─── 文分割 (Faithfulness 用) ─────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[。．\.!?！？])\s*")


def split_sentences(text: str) -> list[str]:
    """日本語/英語混在テキストの簡易文分割。"""
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])
