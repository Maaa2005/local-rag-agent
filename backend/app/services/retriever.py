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
from app.services.embedder import embed_query
from app.services.qdrant import get_client
from app.services.reranker import rerank
from app.services.sparse import build_sparse


def build_access_filter(user_access_level: int) -> Filter:
    """権限プレフィルタ: access_level <= user_access_level かつ is_active=True。

    権限外文書 (access_level > ユーザー権限) と無効化チャンク (is_active=False) を
    Qdrant 側で除外する。リリース基準「権限フィルタ正答率 100%」の中核。
    """
    return Filter(
        must=[
            FieldCondition(key="access_level", range=Range(lte=user_access_level)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
    )


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
        }
        for h in response.points
    ]

    if settings.rerank_enabled:
        chunks = await rerank(question, chunks)

    return chunks[: settings.retrieval_top_k]
