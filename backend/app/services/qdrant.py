"""Qdrant AsyncClient のシングルトン管理。

indexer と retriever で個別にクライアントを作っていたものを集約し、
lifespan で確実に close する。
"""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.config import settings

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
