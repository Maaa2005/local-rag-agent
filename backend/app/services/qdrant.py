"""Qdrant AsyncClient のシングルトン管理。

indexer と retriever で個別にクライアントを作っていたものを集約し、
lifespan で確実に close する。

時間切れと送り直し（Issue #14）:
- settings.qdrant_timeout_s が 0 より大きいときだけ AsyncQdrantClient に timeout を渡す。
  0 なら今までどおり qdrant-client の既定（1.12.1 では httpx の既定 5 秒）。
- 読むだけの要求は read_with_retry で包む。時間切れのときだけ settings.qdrant_search_retries 回まで
  同じ要求を送り直す。それ以外の失敗（接続拒否・4xx/5xx・応答の形の誤り）は送り直さずそのまま上げる。
  検索は Qdrant の状態を変えないので、同じ要求をもう一度送っても結果の意味は変わらない。権限の絞り込み
  （access_level の filter）は送り直しでも同じ要求に入っており、その後の SQLite の確かめ直しも変わらない。
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, TypeVar

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_client: AsyncQdrantClient | None = None
# 送り直した回数（測るときに読む。プロセスの中の合計）
retry_count = 0


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        kw = {"timeout": settings.qdrant_timeout_s} if settings.qdrant_timeout_s > 0 else {}
        _client = AsyncQdrantClient(url=settings.qdrant_url, **kw)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def is_timeout(e: BaseException) -> bool:
    """qdrant-client の REST は httpx の例外を ResponseHandlingException に包んで上げる（1.12.1）。"""
    if isinstance(e, httpx.TimeoutException):
        return True
    return isinstance(e, ResponseHandlingException) and isinstance(getattr(e, "source", None), httpx.TimeoutException)


async def read_with_retry(call: Callable[[], Awaitable[T]], retries: int | None = None) -> T:
    """読むだけの要求 call() を、時間切れのときだけ retries 回（既定は設定値）まで送り直す。"""
    global retry_count
    n = settings.qdrant_search_retries if retries is None else retries
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as e:
            if attempt >= n or not is_timeout(e):
                raise
            attempt += 1
            retry_count += 1
            logger.warning("Qdrant の検索が時間切れ。送り直す (%d/%d)", attempt, n)
