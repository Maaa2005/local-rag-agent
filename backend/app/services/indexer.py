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

    needs_create = True
    if settings.qdrant_collection in existing:
        info = await client.get_collection(settings.qdrant_collection)
        vcfg = info.config.params.vectors
        if isinstance(vcfg, dict) and "dense" in vcfg:
            needs_create = False  # 正しいフォーマット（dense + sparse 名前付きベクトル）
        else:
            # 旧フォーマット（無名ベクトル）→ 再作成
            await client.delete_collection(settings.qdrant_collection)

    if needs_create:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={"dense": VectorParams(size=settings.vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    # 新規作成時だけでなく、既存コレクションに対しても不足しているインデックスを
    # 補う (例: unclassified フィールドを後から追加した場合)。create_payload_index
    # は同名フィールドに対して冪等なため無条件に呼んでよい。
    for field, schema in [
        ("access_level", "integer"),
        ("document_id", "keyword"),
        ("is_active", "bool"),
        ("unclassified", "bool"),
        ("index_version", "integer"),
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
    index_version: int,
    unclassified: bool = False,
) -> int:
    """世代切替方式でチャンクを再インデックスする (項目8)。

    呼び出し元 (task_processor) は documents.index_version の現在値 + 1 を
    `index_version` として渡す。手順:

      1. この document_id の残骸 (is_active=False のポイント) を物理削除する。
         これは「前回この関数が upsert 後・activate 前に失敗して残した
         新世代の残骸」を次回実行の頭で掃除するためのもの。
      2. 新チャンクを is_active=False + index_version=index_version で upsert。
      3. upsert が全件成功した後にのみ、この世代のポイントを
         set_payload で is_active=True へ切り替える。
      4. 旧世代 (index_version が一致しない、または旧スキーマで未設定) の
         ポイントを物理削除する。

    途中失敗時の挙動 (検索結果を消さないことが目的):
      - 手順2 (embed/upsert) で失敗 → 旧世代は is_active=True のまま残るため
        検索は旧内容で継続できる。新世代の残骸は次回実行の手順1で掃除される。
      - 手順3/4 で失敗 → 同様に旧世代は is_active=True のまま。新世代は
        is_active=False のまま取り残されるが、次回実行の手順1で掃除される。

    手順3 (activate) と手順4 (旧世代削除) の間の数msの窓では新旧両方の
    チャンクが is_active=True で検索にヒットしうるが、設計判断として許容し
    dedup は行わない。
    """
    client = get_client()

    # 1. 前回実行が upsert 後・activate 前に失敗した場合の残骸を掃除する。
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="is_active", match=MatchValue(value=False)),
            ]
        ),
    )

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
                    # 新世代はまず inactive で書き込み、全件成功後に一括で
                    # activate する (途中失敗時に旧世代を検索し続けられるようにするため)。
                    "is_active": False,
                    "index_version": index_version,
                    # 未分類 (どの監視フォルダにも一致しない) 文書は検索対象外に
                    # するため、常に払い出しペイロードに反映する (項目1)。
                    "unclassified": bool(unclassified),
                },
            )
        )

    await client.upsert(collection_name=settings.qdrant_collection, points=points)

    # 3. 新世代を activate する。
    await client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"is_active": True},
        points=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="index_version", match=MatchValue(value=index_version)),
            ]
        ),
    )

    # 4. 旧世代 (index_version 不一致、または旧スキーマで未設定) を物理削除する。
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))],
            must_not=[FieldCondition(key="index_version", match=MatchValue(value=index_version))],
        ),
    )
    return len(points)
