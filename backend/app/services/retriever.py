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
from app.services.sparse import build_sparse


async def retrieve(question: str, user_access_level: int) -> list[dict]:
    """
    ハイブリッド検索 (dense + sparse) + RRF でチャンクを返す。
    access_level <= user_access_level かつ is_active=True のみ対象。
    """
    dense_vec = await embed_query(question)
    s_idx, s_val = build_sparse(question)

    access_filter = Filter(
        must=[
            FieldCondition(key="access_level", range=Range(lte=user_access_level)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
    )

    prefetch_list: list[Prefetch] = [
        Prefetch(
            query=dense_vec,
            using="dense",
            limit=settings.retrieval_top_k * 3,
            filter=access_filter,
        ),
    ]
    if s_idx:
        prefetch_list.append(
            Prefetch(
                query=SparseVector(indices=s_idx, values=s_val),
                using="sparse",
                limit=settings.retrieval_top_k * 3,
                filter=access_filter,
            )
        )

    response = await get_client().query_points(
        collection_name=settings.qdrant_collection,
        prefetch=prefetch_list,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=settings.retrieval_top_k,
        with_payload=True,
    )

    return [
        {
            "content": h.payload["content"],
            "source_file": h.payload.get("source_file", ""),
            "score": h.score,
        }
        for h in response.points
    ]
