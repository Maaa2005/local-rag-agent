import asyncio
import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    # 既定は CPU (VRAM を LLM 専用にする)。GPU に載せる場合は rerank_device で切り替える
    return CrossEncoder(settings.rerank_model, device=settings.rerank_device)


def _rerank_sync(question: str, chunks: list[dict]) -> list[dict]:
    model = _load_model()
    pairs = [(question, chunk["content"]) for chunk in chunks]
    scores = model.predict(pairs)
    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = []
    for score, chunk in scored:
        new_chunk = dict(chunk)
        new_chunk["rerank_score"] = float(score)
        reranked.append(new_chunk)
    return reranked


async def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """Cross-Encoder で (question, content) ペアをスコアリングし降順ソートして返す。

    ロード失敗・推論例外時はログを出し、元の順序のまま chunks を返す
    (検索自体を失敗させないためのフォールバック)。
    """
    if not chunks:
        return chunks
    try:
        return await asyncio.to_thread(_rerank_sync, question, chunks)
    except Exception:
        logger.exception("Reranker inference failed; falling back to original order")
        return chunks


async def warmup() -> None:
    """起動時のバックグラウンドウォームアップ用 (main.py から rerank_enabled 時のみ呼ばれる)。"""
    await asyncio.to_thread(_load_model)
