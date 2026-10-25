# -*- coding: utf-8 -*-
"""
Qdrant 向量库操作：建 collection、批量写入、混合检索（dense + sparse）、删除。

关键设计：
  1. 命名向量（named vectors）：同一 payload 下挂两路向量
     - "dense"：BGE-M3 1024 维，cosine + int8 量化 + on_disk
     - "sparse"：BGE-M3 lexical_weights 转成的稀疏向量
  2. 四路融合在应用层做 RRF：
     Qdrant 原生 Prefetch 只支持 2 路，四路（exact/fuzzy/sparse/dense）需要
     应用层把四路结果拿回来自己算加权 Reciprocal Rank Fusion。
     本文件负责 dense + sparse 两路 hybrid_search，exact/fuzzy 走 SQLite 侧，
     融合逻辑属检索批次（第 5 批）。
  3. payload 设计：检索过滤所需字段全部 flat 放 payload
     （doc_id / chunk_id / chunk_type / section_path / doc_type / language /
      has_table / has_image / source_format），
     建 collection 时创建 payload index 以加速 filter。
  4. 批量 upsert：用 client.upsert 配合 PointStruct，单次不超过 profile.batch_size
     万条（Qdrant 推荐每批 <64K），这里默认 256 条提交一次。
"""

from __future__ import annotations

from contextlib import suppress
from functools import lru_cache
from typing import Any

from index import resilience
from ingestion.parsers.common import get_settings


# =================================================================== 客户端单例
@lru_cache(maxsize=1)
def get_client() -> Any:
    """
    Qdrant 客户端单例（同进程一个连接池）。
    优先使用 gRPC（吞吐比 REST 高 3-5x，100GB 入库必须走这个）。
    """
    from qdrant_client import QdrantClient

    qcfg = get_settings().get("qdrant") or {}
    host = str(qcfg.get("host") or "localhost")
    port = int(qcfg.get("port") or 6333)
    grpc_port = int(qcfg.get("grpc_port") or 6334)
    prefer_grpc = bool(qcfg.get("prefer_grpc", True))
    api_key = qcfg.get("api_key") or None
    https = bool(qcfg.get("https", False))
    timeout = int(qcfg.get("timeout") or 120)

    return QdrantClient(
        host=host,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=prefer_grpc,
        api_key=api_key,
        https=https,
        timeout=timeout,
    )


def _collection_name() -> str:
    qcfg = get_settings().get("qdrant") or {}
    return str(qcfg.get("collection") or "rag_zh_chunks")


# =================================================================== 建 Collection
def create_collection(name: str | None = None) -> None:
    """
    创建含 dense + sparse 命名向量的 collection。
    若已存在则跳过（幂等），不会删已有数据。
    参数全部来自 profile.qdrant：量化 / HNSW / on_disk 按档位切换。
    """
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        SparseIndexParams,
        SparseVectorConfig,
        SparseVectorParams,
        VectorParams,
    )

    client = get_client()
    coll = name or _collection_name()

    if client.collection_exists(coll):
        return

    active = get_settings().get("active") or {}
    qcfg = get_settings().get("qdrant") or {}
    dim = int(active.get("embedding_dim") or 1024)
    supports_sparse = bool(active.get("supports_sparse", True))

    hnsw_m = int(qcfg.get("hnsw_m") or active.get("hnsw_m") or 16)
    hnsw_ef = int(qcfg.get("hnsw_ef_construct") or active.get("hnsw_ef_construct") or 200)
    on_disk = bool(qcfg.get("on_disk", True))
    payload_on_disk = bool(qcfg.get("payload_on_disk", True))

    vectors_config = {
        "dense": VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=on_disk,
        ),
    }

    sparse_config: dict[str, Any] | None = None
    if supports_sparse:
        sparse_config = {
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=on_disk),
            ),
        }

    quantization_config = ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    )

    create_kwargs: dict[str, Any] = {
        "collection_name": coll,
        "vectors_config": vectors_config,
        "hnsw_config": HnswConfigDiff(
            m=hnsw_m,
            ef_construct=hnsw_ef,
            on_disk=on_disk,
        ),
        "quantization_config": quantization_config,
        "on_disk_payload": payload_on_disk,
    }
    if sparse_config:
        create_kwargs["sparse_vectors_config"] = SparseVectorConfig(**sparse_config)

    client.create_collection(**create_kwargs)

    # 为常用过滤字段创建 payload index，加速 WHERE 过滤
    _ensure_payload_indexes(client, coll)


def _ensure_payload_indexes(client: Any, coll: str) -> None:
    """给高频过滤字段建关键词索引（幂等，重复创建不报错）。"""
    from qdrant_client.models import PayloadSchemaType

    fields = [
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("chunk_id", PayloadSchemaType.KEYWORD),
        ("chunk_type", PayloadSchemaType.KEYWORD),
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("language", PayloadSchemaType.KEYWORD),
        ("source_format", PayloadSchemaType.KEYWORD),
        ("has_table", PayloadSchemaType.BOOL),
        ("has_image", PayloadSchemaType.BOOL),
        ("section_path", PayloadSchemaType.TEXT),
    ]
    for field_name, schema in fields:
        with suppress(Exception):
            client.create_payload_index(coll, field_name=field_name, field_schema=schema)


# =================================================================== 批量写入
def upsert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[dict[str, Any]],
    collection: str | None = None,
    batch_size: int = 256,
) -> int:
    """
    把 chunk + embedding 批量写入 Qdrant。
    chunks[i] 和 embeddings[i] 必须一一对应。
    返回成功写入的点数。
    embeddings[i] 格式：{"dense": [float], "sparse": {"indices": [...], "values": [...]} | None}
    """
    from qdrant_client.models import PointStruct, SparseVector

    client = get_client()
    coll = collection or _collection_name()
    written = 0

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_embs = embeddings[start : start + batch_size]
        points: list[PointStruct] = []

        for chunk, emb in zip(batch_chunks, batch_embs, strict=True):
            payload = _build_payload(chunk)
            vector: dict[str, Any] = {"dense": emb["dense"]}
            if emb.get("sparse"):
                vector["sparse"] = SparseVector(
                    indices=emb["sparse"]["indices"],
                    values=emb["sparse"]["values"],
                )
            # 用 chunk_id 做 point id 的 hash（Qdrant 接受 uint64 或 uuid 字符串）
            point_id = _point_id(chunk.get("chunk_id", ""))
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            client.upsert(collection_name=coll, points=points)
            written += len(points)

    return written


def _build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    """从 chunk 提取 Qdrant payload（只保留过滤需要的 flat 字段，content 也带着供展示）。"""
    meta = chunk.get("metadata") or {}
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "content": chunk.get("content", ""),
        "chunk_type": chunk.get("chunk_type", "text"),
        "section_path": chunk.get("section_path", ""),
        "doc_type": meta.get("doc_type", "unknown"),
        "topic": meta.get("topic", "unknown"),
        "language": meta.get("language", "zh"),
        "has_table": bool(meta.get("has_table", False)),
        "has_image": bool(meta.get("has_image", False)),
        "layout_style": meta.get("layout_style", ""),
        "source_format": meta.get("source_format", ""),
        "bbox": chunk.get("bbox"),
        "pages": chunk.get("pages"),
    }


def _point_id(chunk_id: str) -> str:
    """
    chunk_id 是 sha256 hex（64 字符），Qdrant 接受 uuid 字符串做 id。
    把 hex 转成标准 uuid 格式保证合法性。
    """
    if not chunk_id:
        import uuid
        return str(uuid.uuid4())
    # sha256 hex 的前 32 位拆成 uuid 格式
    clean = chunk_id.replace("-", "")
    if len(clean) >= 32:
        return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:32]}"
    import hashlib
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# =================================================================== 混合检索
def hybrid_search(
    query_dense: list[float],
    query_sparse: dict[str, list] | None,
    top_k: int = 50,
    filters: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """
    dense + sparse 两路各召回 top_k 条，应用层做加权 RRF 融合。
    这是 Qdrant hybrid 的替代方案：原生 FusionQuery(RRF) 不支持加权，
    且 Qdrant 的 Prefetch 最多两路，而完整四路还需要 exact/fuzzy 从 SQLite 侧拿，
    所以这里 dense 和 sparse 分别搜索后在应用层合并。

    weights: {"dense": 1.0, "sparse": 1.0}（默认等权）
    filters: {"doc_type": "contract", "language": "zh", "has_table": True} 等
    返回 [{chunk_id, doc_id, content, score, payload}, ...] 按融合分数降序。
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    client = get_client()
    coll = collection or _collection_name()
    k = 60  # RRF 平滑常数
    w_dense = (weights or {}).get("dense", 1.0)
    w_sparse = (weights or {}).get("sparse", 1.0)
    # 仅读路径：瞬时网络抖动（连接/超时/5xx）做有界退避重试；重试耗尽后上抛，
    # 由 retrieval.hybrid_search 的按路降级兑住。
    attempts, base_delay = resilience.read_retry_policy(get_settings())

    # 构造过滤条件
    qdrant_filter: Filter | None = None
    if filters:
        must_conditions = []
        for field_name, value in filters.items():
            if isinstance(value, (bool, str)):
                must_conditions.append(FieldCondition(key=field_name, match=MatchValue(value=value)))
            elif isinstance(value, list):
                must_conditions.append(FieldCondition(key=field_name, match=MatchAny(any=value)))
        if must_conditions:
            qdrant_filter = Filter(must=must_conditions)

    # dense 搜索
    dense_results: list[dict[str, Any]] = []
    if query_dense and w_dense > 0:
        dense_hits = resilience.retry_transient(
            lambda: client.search(
                collection_name=coll,
                query_vector=("dense", query_dense),
                limit=top_k,
                query_filter=qdrant_filter,
            ),
            attempts=attempts,
            base_delay=base_delay,
        )
        dense_results = [{"point_id": h.id, "score": h.score, "payload": h.payload} for h in dense_hits]

    # sparse 搜索
    sparse_results: list[dict[str, Any]] = []
    if query_sparse and w_sparse > 0:
        from qdrant_client.models import SparseVector as SV

        sparse_query = SV(indices=query_sparse["indices"], values=query_sparse["values"])
        sparse_hits = resilience.retry_transient(
            lambda: client.search(
                collection_name=coll,
                query_vector=("sparse", sparse_query),
                limit=top_k,
                query_filter=qdrant_filter,
            ),
            attempts=attempts,
            base_delay=base_delay,
        )
        sparse_results = [{"point_id": h.id, "score": h.score, "payload": h.payload} for h in sparse_hits]

    # 加权 RRF 融合
    fused = _weighted_rrf(dense_results, sparse_results, k, w_dense, w_sparse)

    # 组装输出
    output: list[dict[str, Any]] = []
    for point_id, rrf_score, source_payload in fused:
        output.append({
            "point_id": point_id,
            "chunk_id": source_payload.get("chunk_id", ""),
            "doc_id": source_payload.get("doc_id", ""),
            "content": source_payload.get("content", ""),
            "score": round(rrf_score, 6),
            "payload": source_payload,
        })
    return output


def _weighted_rrf(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    k: int,
    w_dense: float,
    w_sparse: float,
) -> list[tuple[Any, float, dict[str, Any]]]:
    """
    加权 Reciprocal Rank Fusion。
    score(d) = sum over lists: weight_i / (k + rank_i(d))
    同一条目在两个列表都出现时分数叠加（这是 RRF 的精髓：多路一致命中的优先）。
    """
    scores: dict[Any, float] = {}
    payloads: dict[Any, dict[str, Any]] = {}

    for rank, item in enumerate(dense, start=1):
        pid = item["point_id"]
        scores[pid] = scores.get(pid, 0) + w_dense / (k + rank)
        payloads[pid] = item["payload"]

    for rank, item in enumerate(sparse, start=1):
        pid = item["point_id"]
        scores[pid] = scores.get(pid, 0) + w_sparse / (k + rank)
        if pid not in payloads:
            payloads[pid] = item["payload"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(pid, score, payloads[pid]) for pid, score in ranked]


# =================================================================== 删除
def delete_by_doc_id(doc_id: str, collection: str | None = None) -> None:
    """按 doc_id 删除该文档的全部 chunk 向量（重新入库前清理旧数据）。"""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_client()
    coll = collection or _collection_name()
    client.delete(
        collection_name=coll,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )
