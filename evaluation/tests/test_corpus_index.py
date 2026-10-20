"""評価コーパスの索引 (evaluation/colab/corpus_index.py) のテスト。

Qdrant はメモリ上、SQLite は一時ファイル、埋め込みは偽物 (全部同じ向き) にして、本物のコーパスを載せる。
確かめること:
- 載せた数と数え直した数が合い、検索で通る文書が全部そろう
- DATA_DIR が別の索引を指すと「検索で通る文書 0 件」になる (run_eval はここで止める)
- 点が既にある集合・既にある app.db には載せない
- 載せた索引を v1 の検索が引け、根拠の場所が dataset の expected_sources と同じ形になる
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "evaluation" / "corpus"
N_DOCS = sum(1 for _ in CORPUS.glob("*/*.md"))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))  # run_eval と同じ。単独で流しても app を import できるように


@pytest.fixture
def ci(tmp_path, monkeypatch):
    # 埋め込みモデルの重い依存はここでは使わない。入っていなければ空の偽物で import だけ通す
    try:
        importlib.import_module("sentence_transformers")
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = object
        stub.CrossEncoder = object
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

    import app.database
    import app.services.indexer
    import app.services.qdrant
    import app.services.retriever
    from app.config import settings

    vec = [1.0] + [0.0] * (settings.vector_size - 1)

    async def fake_embed(texts, is_query=False):
        return [list(vec) for _ in texts]

    async def fake_embed_query(text):
        return list(vec)

    assert app.database.db._conn is None
    monkeypatch.setattr(app.database, "DB_PATH", tmp_path / "data" / "app.db")
    monkeypatch.setattr(app.services.qdrant, "_client", None)
    monkeypatch.setattr(settings, "qdrant_collection", "eval_test")
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(app.services.indexer, "embed_texts", fake_embed)
    monkeypatch.setattr(app.services.retriever, "embed_query", fake_embed_query)
    return importlib.import_module("evaluation.colab.corpus_index")


def _run(monkeypatch, body):
    """メモリ上の Qdrant を同じループで作り、終わったら閉じる。"""
    import app.services.qdrant
    from app.database import close_db
    from app.services.qdrant import close_client
    from qdrant_client import AsyncQdrantClient

    async def scenario():
        monkeypatch.setattr(app.services.qdrant, "_client", AsyncQdrantClient(location=":memory:"))
        try:
            return await body()
        finally:
            await close_client()
            await close_db()

    return asyncio.run(scenario())


def test_index_then_count_matches(ci, monkeypatch):
    async def body():
        chunks, doc_level, doc_name = await ci.index_corpus(CORPUS)
        return chunks, doc_level, doc_name, await ci.count_index()

    chunks, doc_level, doc_name, counted = _run(monkeypatch, body)
    assert len(doc_level) == N_DOCS == 13
    assert sorted(doc_level.values()) == [1] * 7 + [2] * 3 + [3] * 3
    assert "経費精算規程.md" in doc_name.values()
    assert counted == {
        "collection": "eval_test",
        "points": chunks,
        "documents": N_DOCS,
        "qdrant_documents": N_DOCS,
        "matched": N_DOCS,
    }
    assert chunks > N_DOCS


def test_db_from_another_index_matches_nothing(ci, monkeypatch, tmp_path):
    import app.database

    async def body():
        await ci.index_corpus(CORPUS)
        # 評価側の DATA_DIR が別の (空の) 場所を指した状態
        await app.database.close_db()
        monkeypatch.setattr(app.database, "DB_PATH", tmp_path / "other" / "app.db")
        await app.database.init_db()
        return await ci.count_index()

    counted = _run(monkeypatch, body)
    assert counted["points"] > 0
    assert counted["documents"] == 0
    assert counted["matched"] == 0


def test_missing_collection_counts_zero(ci, monkeypatch):
    async def body():
        from app.database import init_db

        await init_db()
        return await ci.count_index()

    assert _run(monkeypatch, body) == {
        "collection": "eval_test",
        "points": 0,
        "documents": 0,
        "qdrant_documents": 0,
        "matched": 0,
    }


def test_v1_retriever_finds_indexed_chunks_with_dataset_paths(ci, monkeypatch):
    from app.services.retriever import retrieve

    async def body():
        await ci.index_corpus(CORPUS)
        return await retrieve("経費精算の上限はいくらか", 1), await retrieve("役員報酬の決め方", 3)

    general, executive = _run(monkeypatch, body)
    assert general and {c["access_level"] for c in general} == {1}
    assert all(c["source_file"].startswith("/watched/general/") for c in general)
    assert executive and all(c["source_file"].startswith("/watched/") for c in executive)


def _cli_args(tmp_path):
    return Namespace(repo=ROOT, work=tmp_path / "work", qdrant_url="http://127.0.0.1:6333", collection="eval_test")


def _match_cli_settings(monkeypatch, a):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", str(a.work / "data"))
    monkeypatch.setattr(settings, "qdrant_url", a.qdrant_url)


def test_run_refuses_collection_with_points(ci, monkeypatch, tmp_path):
    import app.services.qdrant
    from qdrant_client import AsyncQdrantClient

    a = _cli_args(tmp_path)
    _match_cli_settings(monkeypatch, a)

    async def must_not_index(corpus):
        raise AssertionError("点がある集合に載せようとした")

    async def scenario():
        monkeypatch.setattr(app.services.qdrant, "_client", AsyncQdrantClient(location=":memory:"))
        await ci.index_corpus(CORPUS)
        monkeypatch.setattr(ci, "index_corpus", must_not_index)
        with pytest.raises(SystemExit, match="既に点がある"):
            await ci._run(a)
        # 止めた後も閉じている (finally を通った)
        assert app.services.qdrant._client is None

    asyncio.run(scenario())


def test_run_writes_counts_after_indexing(ci, monkeypatch, tmp_path):
    a = _cli_args(tmp_path)
    _match_cli_settings(monkeypatch, a)

    import app.services.qdrant
    from qdrant_client import AsyncQdrantClient

    async def scenario():
        monkeypatch.setattr(app.services.qdrant, "_client", AsyncQdrantClient(location=":memory:"))
        return await ci._run(a)

    info = asyncio.run(scenario())
    assert info["documents"] == N_DOCS
    assert info["chunks"] > N_DOCS
    assert info["env"] == ci.index_env(a.work, a.qdrant_url, "eval_test")


def test_main_refuses_existing_app_db(ci, monkeypatch, tmp_path):
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    (work / "data" / "app.db").write_bytes(b"")
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    with pytest.raises(SystemExit, match="app.db"):
        ci.main(["--repo", str(ROOT), "--work", str(work), "--collection", "eval_test"])
    assert "QDRANT_COLLECTION" not in os.environ
    assert not (work / ci.INDEX_NAME).exists()


def test_index_env_is_what_run_eval_reads():
    from evaluation.colab.corpus_index import index_env

    env = index_env(Path("/w"), "http://q:6333", "c1")
    assert env == {
        "DATA_DIR": "/w/data",
        "WATCHED_PATH": "/w/watched",
        "QDRANT_URL": "http://q:6333",
        "QDRANT_COLLECTION": "c1",
    }
