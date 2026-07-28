"""Reranker のユニットテスト。

CrossEncoder の実モデルロードは重いため、_load_model をモックに差し替えて検証する。
検証観点: スコア降順ソート・元データの非破壊・空入力・ロード/推論失敗時のフォールバック。
"""
from __future__ import annotations

import asyncio

from app.services import reranker


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, pairs):
        assert len(pairs) == len(self._scores)
        return self._scores


def setup_function(_fn):
    reranker._load_model.cache_clear()


def teardown_function(_fn):
    reranker._load_model.cache_clear()


def _chunks(n: int) -> list[dict]:
    return [{"content": f"content-{i}", "source_file": f"file-{i}.txt", "score": 1.0} for i in range(n)]


def test_rerank_sorts_by_score_descending(monkeypatch):
    chunks = _chunks(3)
    # chunk 0 -> 低スコア, chunk 1 -> 最高スコア, chunk 2 -> 中間
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    monkeypatch.setattr(reranker, "_load_model", lambda: fake)

    result = asyncio.run(reranker.rerank("question", chunks))

    assert [c["content"] for c in result] == ["content-1", "content-2", "content-0"]
    assert result[0]["rerank_score"] == 0.9
    assert result[-1]["rerank_score"] == 0.1


def test_rerank_does_not_mutate_original_chunks(monkeypatch):
    chunks = _chunks(2)
    fake = _FakeCrossEncoder([0.2, 0.8])
    monkeypatch.setattr(reranker, "_load_model", lambda: fake)

    asyncio.run(reranker.rerank("question", chunks))

    assert "rerank_score" not in chunks[0]
    assert "rerank_score" not in chunks[1]


def test_rerank_empty_chunks_returns_empty():
    result = asyncio.run(reranker.rerank("question", []))
    assert result == []


def test_rerank_falls_back_on_load_failure(monkeypatch):
    def _raise():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(reranker, "_load_model", _raise)
    chunks = _chunks(2)

    result = asyncio.run(reranker.rerank("question", chunks))

    # フォールバック: 元の順序・内容のまま返る
    assert result == chunks


def test_rerank_falls_back_on_inference_failure(monkeypatch):
    class _BrokenCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(reranker, "_load_model", lambda: _BrokenCrossEncoder())
    chunks = _chunks(2)

    result = asyncio.run(reranker.rerank("question", chunks))

    assert result == chunks


class _FakePoint:
    def __init__(self, content: str, score: float):
        # document_id を持たせないと _revalidate_against_sqlite が fail-closed で
        # 全件弾いてしまう (項目高8)。実際の Qdrant ペイロードには常に含まれる。
        self.payload = {"content": content, "source_file": "f.txt", "document_id": content}
        self.score = score


class _FakeResponse:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    def __init__(self, points):
        self._points = points

    async def query_points(self, **kwargs):
        return _FakeResponse(self._points)


class _FakeRevalidateRow(dict):
    """sqlite3.Row 風に ["key"] アクセスができる dict。"""


class _FakeDb:
    """_revalidate_against_sqlite が問い合わせる documents を、渡された全 id が
    常に権限内・有効であるかのように装って返すスタブ。"""

    async def fetchall(self, sql, params=()):
        return [
            _FakeRevalidateRow(id=doc_id, access_level=1, status="done", unclassified=0)
            for doc_id in params
        ]


def _setup_retriever_stubs(monkeypatch, points):
    from app.services import retriever

    async def _fake_embed_query(question):
        return [0.0] * 4

    monkeypatch.setattr(retriever, "embed_query", _fake_embed_query)
    monkeypatch.setattr(retriever, "build_sparse", lambda q: ([], []))
    monkeypatch.setattr(retriever, "get_client", lambda: _FakeQdrantClient(points))
    monkeypatch.setattr(retriever, "db", _FakeDb())
    return retriever


def test_retrieve_calls_rerank_when_enabled(monkeypatch):
    """rerank_enabled=True のとき retriever.retrieve は rerank() を経由し、
    その結果を retrieval_top_k 件に絞って返す。
    """
    from app.config import settings

    points = [_FakePoint(f"c{i}", 1.0) for i in range(3)]
    retriever = _setup_retriever_stubs(monkeypatch, points)

    called_with = {}

    async def _spy_rerank(question, chunks):
        called_with["question"] = question
        called_with["chunks"] = chunks
        # 順序を反転して返し、retrieve がこの結果をそのまま使うことを確認する
        return list(reversed(chunks))

    monkeypatch.setattr(retriever, "rerank", _spy_rerank)
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "retrieval_top_k", 2)

    result = asyncio.run(retriever.retrieve("question", user_access_level=1))

    assert called_with["question"] == "question"
    assert len(called_with["chunks"]) == 3
    assert [c["content"] for c in result] == ["c2", "c1"]


def test_retrieve_skips_rerank_when_disabled(monkeypatch):
    """rerank_enabled=False のとき retriever.retrieve は rerank() を呼ばず
    一次検索の結果をそのまま (top_k 件) 返す。
    """
    from app.config import settings

    points = [_FakePoint(f"c{i}", 1.0) for i in range(3)]
    retriever = _setup_retriever_stubs(monkeypatch, points)

    called = False

    async def _spy_rerank(question, chunks):
        nonlocal called
        called = True
        return chunks

    monkeypatch.setattr(retriever, "rerank", _spy_rerank)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "retrieval_top_k", 2)

    result = asyncio.run(retriever.retrieve("question", user_access_level=1))

    assert called is False
    assert [c["content"] for c in result] == ["c0", "c1"]
