"""評価コーパス (evaluation/corpus) を v1 の索引 (SQLite の documents 行 + Qdrant) に載せる。

rag_connection_check.py と、run_eval の前準備 (docs/v2-light-model-plan.md の Step 0.5) で共有する。

v1 の検索は Qdrant で取った後に SQLite の documents 行で権限を確かめ直す。索引と評価で
DATA_DIR・QDRANT_URL・QDRANT_COLLECTION が 1 つでもずれると全問の根拠が消え、モデルの
失敗に見える。そこで CLI はこの 3 つを <work>/index.json に書き、評価側はそこから環境変数を作る。

CLI (評価とは別のプロセスで 1 回だけ流す。vLLM を立てる前に流すと GPU の記録に混ざらない):
    python evaluation/colab/corpus_index.py --repo . --work /content/eval_work \\
        --qdrant-url http://127.0.0.1:6333 --collection eval_run1

既に点がある集合・既にある app.db には載せない (前の索引と混ざると数え直せないため)。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

LEVELS = {"general": 1, "manager": 2, "executive": 3}
INDEX_NAME = "index.json"


def index_env(work: Path, qdrant_url: str, collection: str) -> dict[str, str]:
    """索引と評価で揃える環境変数。app を import する前に入れる (import 時に読まれる)。"""
    return {
        "DATA_DIR": str(Path(work) / "data"),
        "WATCHED_PATH": str(Path(work) / "watched"),
        "QDRANT_URL": qdrant_url,
        "QDRANT_COLLECTION": collection,
    }


async def index_corpus(corpus: Path) -> tuple[int, dict[str, int], dict[str, str]]:
    """corpus/{general,manager,executive}/*.md を載せ、(チャンク数, doc_id→レベル, doc_id→ファイル名) を返す。

    v1 と同じく documents 行 (検索後の SQLite 再検証の正) を入れてから Qdrant に載せる。
    source_path は /watched/<フォルダ>/<ファイル名> (dataset の expected_sources と同じ形)。
    AsyncQdrantClient は作ったループに縛られるので、検索と同じループで呼ぶこと。
    """
    from app.database import db, init_db
    from app.services.indexer import ensure_collection, index_document

    await init_db()
    await ensure_collection()
    doc_level: dict[str, int] = {}
    doc_name: dict[str, str] = {}
    n = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    for folder, level in LEVELS.items():
        for f in sorted((Path(corpus) / folder).glob("*.md")):
            text = f.read_text(encoding="utf-8")
            doc_id = str(uuid.uuid4())
            src = f"/watched/{folder}/{f.name}"
            await db.execute(
                "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, status, "
                "unclassified, index_version, created_at, updated_at) VALUES (?, ?, ?, ?, '.md', 'done', 0, 1, ?, ?)",
                (doc_id, src, hashlib.sha256(text.encode()).hexdigest(), level, now, now),
            )
            n += await index_document(doc_id, text, src, level, 1)
            doc_level[doc_id], doc_name[doc_id] = level, f.name
    return n, doc_level, doc_name


async def count_index() -> dict:
    """今の設定で検索が見る索引の数。

    points: 集合の点 (集合が無ければ 0)。documents: 検索で通る documents 行 (削除済み・未分類を除く)。
    qdrant_documents: 点が持つ document_id の種類。matched: そのうち documents 行がある文書の数。
    検索は点を取った後に documents 行で確かめ直すので、DATA_DIR と集合が別の索引どうしだと matched は 0 になる。
    """
    from app.config import settings
    from app.database import db
    from app.services.qdrant import get_client

    client = get_client()
    names = {c.name for c in (await client.get_collections()).collections}
    points = 0
    doc_ids: set[str] = set()
    if settings.qdrant_collection in names:
        points = (await client.count(settings.qdrant_collection, exact=True)).count
        offset = None
        while True:
            batch, offset = await client.scroll(
                settings.qdrant_collection, limit=1000, offset=offset,
                with_payload=["document_id"], with_vectors=False,
            )
            doc_ids.update(p.payload["document_id"] for p in batch if (p.payload or {}).get("document_id"))
            if offset is None:
                break
    rows = await db.fetchall(
        "SELECT id FROM documents WHERE status NOT IN ('deleted', 'purged') AND unclassified = 0"
    )
    valid = {r[0] for r in rows}
    return {
        "collection": settings.qdrant_collection,
        "points": points,
        "documents": len(valid),
        "qdrant_documents": len(doc_ids),
        "matched": len(doc_ids & valid),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="評価コーパスを v1 の索引に載せる (index.json に場所を残す)")
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path, help="DATA_DIR はこの下の data/")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--collection", required=True)
    return p.parse_args(argv)


async def _run(a: argparse.Namespace) -> dict:
    from app.config import settings
    from app.database import close_db
    from app.services.qdrant import close_client, get_client

    env = index_env(a.work, a.qdrant_url, a.collection)
    assert (settings.data_dir, settings.qdrant_url, settings.qdrant_collection) == (
        env["DATA_DIR"], env["QDRANT_URL"], env["QDRANT_COLLECTION"],
    ), "環境変数が効いていない"
    try:
        # ensure_collection は旧形式の集合を消して作り直すので、確かめはその前に行う
        names = [c.name for c in (await get_client().get_collections()).collections]
        if a.collection in names and (await get_client().count(a.collection, exact=True)).count > 0:
            raise SystemExit(f"集合 {a.collection} に既に点がある (前の索引と混ざるので拒否)。別の名前を使う")

        t0 = time.monotonic()
        chunks, doc_level, _ = await index_corpus(a.repo / "evaluation" / "corpus")
        seconds = round(time.monotonic() - t0, 1)
        counted = await count_index()
    finally:
        await close_client()
        await close_db()

    n_docs = len(doc_level)
    if (counted["points"], counted["documents"], counted["matched"]) != (chunks, n_docs, n_docs):
        raise SystemExit(f"載せた数と数え直した数が合わない: chunks={chunks} docs={n_docs} counted={counted}")
    return {
        "env": env,
        "documents": n_docs,
        "chunks": chunks,
        "seconds": seconds,
        "embed_model": settings.embed_model,
        "embed_device": settings.embed_device,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    # index.json の場所は評価側 (別の cwd) でも使うので絶対パスにする
    a.repo, a.work = a.repo.resolve(), a.work.resolve()
    db_path = a.work / "data" / "app.db"
    if db_path.exists():
        raise SystemExit(f"{db_path} が既にある (前の索引の documents 行と混ざるので拒否)。空の --work を使う")
    (a.work / "data").mkdir(parents=True, exist_ok=True)
    os.environ.update(index_env(a.work, a.qdrant_url, a.collection))
    sys.path[:0] = [str(a.repo / "backend"), str(a.repo)]

    info = asyncio.run(_run(a))
    (a.work / INDEX_NAME).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"索引: 文書 {info['documents']} 件 / チャンク {info['chunks']} 件 / {info['seconds']} 秒 → {a.work / INDEX_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
