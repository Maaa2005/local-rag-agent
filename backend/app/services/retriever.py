import ntpath
import posixpath

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from app.config import settings
from app.database import db
from app.services.embedder import embed_query
from app.services.qdrant import get_client
from app.services.reranker import rerank
from app.services.sparse import build_sparse


def basename_source(path: str | None) -> str:
    """フルパスからファイル名のみを取り出す (項目中7: フルパス開示の排除)。

    監視フォルダのフルパス (組織名・部署名等を含みうる) を API 境界の外へ
    出さないよう、表示・監査ログ用の source_file はここで一律 basename 化する。
    Windows 区切り (\\) / POSIX 区切り (/) のどちらでも対応する。
    """
    if not path:
        return path or ""
    return posixpath.basename(ntpath.basename(path))


def build_access_filter(user_access_level: int) -> Filter:
    """権限プレフィルタ: access_level <= user_access_level かつ is_active=True かつ
    unclassified ではない。

    権限外文書 (access_level > ユーザー権限)・無効化チャンク (is_active=False)・
    分類未完了で fail-open により既定 Lv1 になっている文書 (unclassified=True) を
    Qdrant 側で除外する。リリース基準「権限フィルタ正答率 100%」の中核。
    unclassified は Qdrant のペイロード欠落時 (旧データ) を False 扱いにするため
    must_not で明示的に True を除外する形にしている。
    """
    return Filter(
        must=[
            FieldCondition(key="access_level", range=Range(lte=user_access_level)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ],
        must_not=[
            FieldCondition(key="unclassified", match=MatchValue(value=True)),
        ],
    )


async def _revalidate_against_sqlite(chunks: list[dict], user_access_level: int) -> list[dict]:
    """Qdrant のヒット結果を SQLite (単一の真実源) で再検証する。

    権限変更時に Qdrant のペイロード更新が SQLite と非同期/競合するケースや、
    削除直後でまだ Qdrant の物理削除/無効化が反映されていないケースでも、
    ここで documents テーブルの最新状態と突き合わせて弾く (項目2, 3a)。
    """
    doc_ids = {c["document_id"] for c in chunks if c.get("document_id")}
    if not doc_ids:
        # document_id を1つも持たないヒットは SQLite で真偽を検証できないため、
        # fail-open (そのまま通す) ではなく fail-closed で全て弾く (項目高8)。
        return []

    placeholders = ",".join("?" * len(doc_ids))
    rows = await db.fetchall(
        f"SELECT id, access_level, status, unclassified FROM documents "
        f"WHERE id IN ({placeholders})",
        tuple(doc_ids),
    )
    valid_ids = {
        r["id"]
        for r in rows
        if r["status"] not in ("deleted", "purged")
        and r["access_level"] <= user_access_level
        and not r["unclassified"]
    }
    return [c for c in chunks if c.get("document_id") in valid_ids]


async def retrieve(question: str, user_access_level: int) -> list[dict]:
    """
    ハイブリッド検索 (dense + sparse) + RRF でチャンクを返す。
    access_level <= user_access_level かつ is_active=True のみ対象。
    rerank_enabled の場合は候補を rerank_candidates 件取得し、Cross-Encoder で
    リランクしてから retrieval_top_k 件へ絞り込む。
    """
    dense_vec = await embed_query(question)
    s_idx, s_val = build_sparse(question)

    access_filter = build_access_filter(user_access_level)

    # リランクする場合は候補数を広めに取り、リランク後に top_k へ絞る。
    fetch_limit = settings.rerank_candidates if settings.rerank_enabled else settings.retrieval_top_k
    prefetch_limit = max(fetch_limit, settings.retrieval_top_k) * 3

    prefetch_list: list[Prefetch] = [
        Prefetch(
            query=dense_vec,
            using="dense",
            limit=prefetch_limit,
            filter=access_filter,
        ),
    ]
    if s_idx:
        prefetch_list.append(
            Prefetch(
                query=SparseVector(indices=s_idx, values=s_val),
                using="sparse",
                limit=prefetch_limit,
                filter=access_filter,
            )
        )

    response = await get_client().query_points(
        collection_name=settings.qdrant_collection,
        prefetch=prefetch_list,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_limit,
        with_payload=True,
    )

    chunks = [
        {
            "content": h.payload["content"],
            "source_file": h.payload.get("source_file", ""),
            "score": h.score,
            "access_level": h.payload.get("access_level"),
            "document_id": h.payload.get("document_id"),
            "chunk_index": h.payload.get("chunk_index"),
        }
        for h in response.points
    ]

    # Qdrant 側フィルタと SQLite の状態がずれている可能性 (権限変更・削除の
    # 反映タイミングの競合) に備え、返す前に必ず SQLite で再検証する。
    chunks = await _revalidate_against_sqlite(chunks, user_access_level)

    if settings.rerank_enabled:
        chunks = await rerank(question, chunks)

    return chunks[: settings.retrieval_top_k]


async def get_chunk_content(document_id: str, chunk_index: int) -> str | None:
    """指定した document_id + chunk_index のチャンク本文を Qdrant から取得する。

    会話履歴の引用表示は本文を複製保存せず (項目5)、表示のたびに ID 参照から
    権限チェック付きで解決する (項目3b)。無効化済み (is_active=False) や
    存在しないチャンクは None を返す。
    """
    result = await get_client().scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index)),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    points, _ = result
    if not points:
        return None
    return points[0].payload.get("content")
