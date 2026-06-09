import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    # CPU で動作させ VRAM を LLM 専用にする
    return SentenceTransformer(settings.embed_model, device="cpu")


def _embed_sync(texts: list[str], is_query: bool) -> list[list[float]]:
    model = _load_model()
    # multilingual-e5-large は prefix が必要
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


async def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_sync, texts, is_query)


async def embed_query(text: str) -> list[float]:
    vecs = await embed_texts([text], is_query=True)
    return vecs[0]
