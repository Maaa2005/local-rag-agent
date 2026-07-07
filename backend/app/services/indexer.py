import re
import uuid
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.services.embedder import embed_texts
from app.services.qdrant import get_client
from app.services.sparse import build_sparse


async def ensure_collection() -> None:
    client = get_client()
    existing = [c.name for c in (await client.get_collections()).collections]

    if settings.qdrant_collection in existing:
        info = await client.get_collection(settings.qdrant_collection)
        vcfg = info.config.params.vectors
        if isinstance(vcfg, dict) and "dense" in vcfg:
            return  # 正しいフォーマット（dense + sparse 名前付きベクトル）
        # 旧フォーマット（無名ベクトル）→ 再作成
        await client.delete_collection(settings.qdrant_collection)

    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={"dense": VectorParams(size=settings.vector_size, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    for field, schema in [
        ("access_level", "integer"),
        ("document_id", "keyword"),
        ("is_active", "bool"),
    ]:
        await client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field,
            field_schema=schema,
        )


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """文末で区切るシンプルなチャンク分割。"""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            snip = text[max(end - 30, start): end + 30]
            for sep in ("。\n", "。", "\n\n", "\n", "．", ". "):
                idx = snip.rfind(sep)
                if idx != -1:
                    end = max(end - 30, start) + idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        # 文末調整で end が後退した結果 end - overlap が start 以下になり得るため、
        # 必ず 1 文字以上前進させて無限ループを防ぐ。
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


async def deactivate_document(document_id: str) -> None:
    """旧バージョンのチャンクをソフト削除して古い情報を防ぐ。"""
    await get_client().set_payload(
        collection_name=settings.qdrant_collection,
        payload={"is_active": False},
        points=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )


async def delete_document(document_id: str) -> None:
    """ドキュメントの全チャンクを Qdrant から物理削除する。"""
    await get_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )


async def index_document(
    document_id: str,
    text: str,
    source_file: str,
    access_level: int,
) -> int:
    # コレクションの存在チェックは lifespan で 1 度だけ。タスクごとには呼ばない。
    await deactivate_document(document_id)

    chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    dense_vecs = await embed_texts(chunks, is_query=False)

    points = []
    for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vecs)):
        s_idx, s_val = build_sparse(chunk)
        vec_dict: dict = {"dense": dense_vec}
        if s_idx:
            vec_dict["sparse"] = SparseVector(indices=s_idx, values=s_val)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec_dict,
                payload={
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk,
                    "source_file": source_file,
                    "access_level": access_level,
                    "is_active": True,
                },
            )
        )

    await get_client().upsert(collection_name=settings.qdrant_collection, points=points)

    # upsert 成功後にのみ旧チャンク (is_active=False) を物理削除する。
    # upsert 前に消すと、途中で失敗した場合に旧チャンクでの検索継続すらできなくなる。
    await get_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="is_active", match=MatchValue(value=False)),
            ]
        ),
    )
    return len(points)
