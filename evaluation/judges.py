"""LLM-as-a-Judge による生成品質採点。

ジャッジは既存の app.services.providers レジストリから取得した
LLMProvider を使う。JSON 構造化出力を要求し、パースに失敗したら 0 点扱い。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from evaluation.metrics import split_sentences


@dataclass
class JudgeScores:
    faithfulness: float            # M4
    answer_relevance: float        # M5
    context_relevance: float       # M3
    raw: dict                      # ジャッジ生 JSON


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """LLM 応答から最後の JSON オブジェクトを取り出す。失敗時 {}。"""
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ─── プロンプト ──────────────────────────────────────────────────
_PROMPT = """あなたは RAG システムの厳格な評価者です。以下を評価して **JSON のみ** で回答してください。

【質問】
{question}

【取得チャンク】
{chunks}

【RAG の回答】
{answer}

【期待キーワード (参考)】
{keywords}

以下の 3 項目について 0.0〜1.0 (0.5 刻み) で採点し、JSON 形式のみで返答:

- "context_relevance": 取得チャンクが質問にどれだけ関連しているか
- "answer_relevance":  回答が質問に直接答えているか
- "faithfulness":      回答中の主張がチャンク内のテキストから推論可能な比率

加えて回答中の主張ごとの根拠判定 ("sentence_grounded": 0/1 の配列) も出力。

例:
{{"context_relevance": 1.0, "answer_relevance": 0.5, "faithfulness": 0.66,
 "sentence_grounded": [1, 0, 1], "rationale": "根拠コメント"}}"""


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(なし)"
    return "\n\n".join(
        f"[{i+1}] {c.get('source_file', '?')}\n{c.get('content', '')[:600]}"
        for i, c in enumerate(chunks)
    )


async def judge(
    provider,                            # LLMProvider (app.services.providers)
    question: str,
    chunks: list[dict],
    answer: str,
    expected_keywords: list[str],
) -> JudgeScores:
    """LLM-judge を 1 回呼び出して 3 メトリクスを取得。"""
    prompt = _PROMPT.format(
        question=question,
        chunks=_format_chunks(chunks),
        answer=answer or "(空)",
        keywords=", ".join(expected_keywords) if expected_keywords else "(なし)",
    )
    messages = [
        {"role": "system", "content": "あなたは公平な RAG 評価者です。必ず JSON のみで答えてください。"},
        {"role": "user", "content": prompt},
    ]
    buf: list[str] = []
    async for tok in provider.stream(messages):
        buf.append(tok)
    raw = _extract_json("".join(buf))

    return JudgeScores(
        faithfulness=_clamp(raw.get("faithfulness", 0.0)),
        answer_relevance=_clamp(raw.get("answer_relevance", 0.0)),
        context_relevance=_clamp(raw.get("context_relevance", 0.0)),
        raw=raw,
    )


def _clamp(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


# ─── 文単位の Faithfulness 内訳 (オプション) ─────────────────────
def faithfulness_from_grounded_array(answer: str, grounded: list[int]) -> float:
    """sentence_grounded 配列が信頼できるときの厳密計算。"""
    sentences = split_sentences(answer)
    if not sentences or not grounded:
        return 0.0
    n = min(len(sentences), len(grounded))
    return sum(grounded[:n]) / n
