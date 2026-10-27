"""v1 バックエンド（backend/app）の検索・生成を local_answer の差し替え口につなぐ。

- 検索: app.services.retriever.retrieve(question, user_access_level)。Qdrant のフィルタと SQLite 再検証は v1 のまま
- 生成: app.services.llm.stream_answer(question, chunks)。ローカル LLM（vLLM）固定で資格情報を取らない

app パッケージは qdrant-client などを要するため、呼ばれた時点で import する（import 失敗は例外 → held）。
使うときは backend/ を PYTHONPATH に入れる。

v1 は async で、Qdrant の AsyncQdrantClient などをモジュール内で使い回す（app/services/qdrant.py）。
呼び出しごとに asyncio.run で新しいループを作ると、2 回目以降は閉じたループに結び付いたクライアントで失敗する。
そのため専用スレッドの 1 本のループを検索・生成で共有し、そこへ投げて上限時間まで待つ。
上限を超えたら打ち切って例外にする（→ held。N4）。走っているループの中からも呼べる。
"""
from __future__ import annotations

import asyncio
import importlib
import ntpath
import posixpath
import threading
from typing import Any, Awaitable, Sequence, TypeVar

from orchestrator.local_answer import Chunk

RETRIEVER_MODULE = "app.services.retriever"
LLM_MODULE = "app.services.llm"
RETRIEVE_TIMEOUT_S = 30.0
GENERATE_TIMEOUT_S = 180.0  # 6GB 機のローカル LLM を想定した上限

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _shared_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="rag-backend-loop", daemon=True).start()
            _loop = loop
        return _loop


def run_on_shared_loop(coro: Awaitable[T], timeout_s: float) -> T:
    """共有ループで coro を実行し、timeout_s 秒まで待つ。超えたら取り消して TimeoutError。"""
    fut = asyncio.run_coroutine_threadsafe(coro, _shared_loop())
    try:
        return fut.result(timeout=timeout_s)
    except TimeoutError:
        fut.cancel()
        raise


def _basename(path: Any) -> str:
    if not isinstance(path, str) or not path:
        return "不明"
    return posixpath.basename(ntpath.basename(path))


def to_chunk(d: dict[str, Any]) -> Chunk:
    level = d.get("access_level")
    # bool は int の派生だがレベルとしては不正。不明扱いにして local_answer 側で捨てさせる
    if isinstance(level, bool) or not isinstance(level, int):
        level = None
    return Chunk(content=str(d.get("content") or ""), source=_basename(d.get("source_file")), access_level=level,
                 document_id=d.get("document_id"), chunk_index=d.get("chunk_index"))


class BackendRetriever:
    def __init__(self, module: str = RETRIEVER_MODULE, timeout_s: float = RETRIEVE_TIMEOUT_S) -> None:
        self._module = module
        self._timeout_s = timeout_s

    def retrieve(self, question: str, level: int) -> list[Chunk]:
        mod = importlib.import_module(self._module)
        rows = run_on_shared_loop(mod.retrieve(question, level), self._timeout_s)
        return [to_chunk(r) for r in rows]


class BackendGenerator:
    def __init__(self, module: str = LLM_MODULE, timeout_s: float = GENERATE_TIMEOUT_S) -> None:
        self._module = module
        self._timeout_s = timeout_s

    def answer(self, question: str, chunks: Sequence[Chunk]) -> str:
        mod = importlib.import_module(self._module)
        # v1 の _build_context が読む形に戻す。source は basename のみ（フルパスを LLM に渡さない）
        rows = [{"content": c.content, "source_file": c.source, "access_level": c.access_level,
                 "document_id": c.document_id, "chunk_index": c.chunk_index} for c in chunks]

        async def collect() -> str:
            return "".join([t async for t in mod.stream_answer(question, rows)])

        return run_on_shared_loop(collect(), self._timeout_s)
