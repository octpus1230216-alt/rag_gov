# -*- coding: utf-8 -*-
"""
混合检索主入口：四路并行召回 + 加权 RRF 融合 + Rerank + Context 组装。

流程：
  Query → route() 拿权重
       → 四路并行召回（ThreadPoolExecutor）：
           exact:  keyword_store.fts_search
           fuzzy:  fuzzy_search.fuzzy_search
           sparse: vector_store.hybrid_search (sparse)
           dense:  vector_store.hybrid_search (dense)
       → 加权 RRF 融合（k=60）
       → Reranker (BGE-reranker-v2-m3, top20→top10)
       → ContextBuilder 组装最终结果

红线：
  - fuzzy 结果不直接返回，必须经过 reranker（fuzzy_output_allowed_bypass_rerank=false）
  - 四路必须并行，不串行
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FuturesTimeout
from typing import Any

from ingestion.parsers.common import get_settings
from retrieval.query_router import get_route_candidates, route

# 各路召回延迟/条数埋点。脱敏红线：只记 route / count / elapsed，绝不记 query 正文。
logger = logging.getLogger("rag_zh.retrieval.hybrid")


class HybridSearcher:
    """混合检索器：四路并行召回 + 加权 RRF + Rerank + Context。"""

    def __init__(self) -> None:
        rcfg = get_settings().get("retrieval") or {}
        self._rrf_k: int = int(rcfg.get("rrf_k") or 60)
        self._recall_top_k: int = int(rcfg.get("recall_top_k") or 50)
        rerank_cfg = rcfg.get("rerank") or {}
        self._rerank_enabled: bool = bool(rerank_cfg.get("enabled", True))
        self._rerank_candidates: int = int(rerank_cfg.get("candidates_in") or 20)
        self._rerank_output_top_k: int = int(rerank_cfg.get("output_top_k") or 10)
        self._rerank_threshold: float = float(rerank_cfg.get("score_threshold") or 0.15)
        self._route_candidates = get_route_candidates()
        # 整次在线召回的墙钟预算（秒）：超时未完成的各路降级为空，不把一次检索拖满
        # Qdrant 连接级 timeout（120s）。
        rcfg_for_timeout = get_settings().get("retrieval") or {}
        self._recall_timeout: float = float(rcfg_for_timeout.get("recall_timeout_seconds") or 3.0)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """
        主检索入口。

        参数：
          query: 用户查询
          top_k: 最终返回条数
          filters: payload 过滤条件（doc_type/language/has_table 等），透传给 Qdrant
          mode: "hybrid"(全部四路) | "semantic"(仅 dense) | "keyword"(仅 exact+sparse) | "fuzzy"(仅 fuzzy)

        返回：
          {
            "results": [{"chunk_id", "doc_id", "content", "rerank_score", "rrf_score", "source_routes"}],
            "context": {...},  # context_builder 输出
            "route_weights": {...},
            "query": query,
          }
        """
        weights = route(query)
        started = time.perf_counter()

        # 按 mode 决定启用哪些路
        active_routes = self._resolve_routes(mode, weights)

        # 并行召回（各路耗时/条数在 _parallel_recall 内逐路 DEBUG）
        route_results = self._parallel_recall(query, active_routes, filters)

        # 加权 RRF 融合
        fused = self._rrf_fusion(route_results, active_routes, self._rrf_k)

        # 取 top candidates 送 reranker
        candidates_for_rerank = fused[: self._rerank_candidates]

        if self._rerank_enabled and candidates_for_rerank:
            reranked = self._do_rerank(query, candidates_for_rerank)
        else:
            reranked = candidates_for_rerank[:top_k]

        # 过滤低于阈值的
        reranked = [r for r in reranked if r.get("rerank_score", r.get("rrf_score", 0)) >= self._rerank_threshold]

        # 截断到 top_k
        final = reranked[:top_k]

        # 组装 context
        from retrieval.context_builder import build_context

        context_result = build_context(final)

        logger.info(
            "search done mode=%s query_len=%d routes=%s recalled=%d fused=%d final=%d elapsed=%.3fs",
            mode,
            len(query or ""),
            sorted(active_routes),
            sum(len(v) for v in route_results.values()),
            len(fused),
            len(final),
            time.perf_counter() - started,
        )

        return {
            "results": final,
            "context": context_result,
            "route_weights": weights,
            "query": query,
        }

    def _resolve_routes(self, mode: str, weights: dict[str, float]) -> dict[str, float]:
        """按 mode 过滤有效路。"""
        if mode == "semantic":
            return {"dense": weights.get("dense", 1.0)}
        if mode == "keyword":
            return {
                "exact": weights.get("exact", 1.0),
                "sparse": weights.get("sparse", 1.0),
            }
        if mode == "fuzzy":
            return {"fuzzy": weights.get("fuzzy", 1.0)}
        # hybrid：全部四路
        return dict(weights)

    def _parallel_recall(
        self,
        query: str,
        active_routes: dict[str, float],
        filters: dict[str, Any] | None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        四路并行召回（ThreadPoolExecutor）。
        每路返回 [{"chunk_id": str, "score": float, ...}]。
        """
        results: dict[str, list[dict[str, Any]]] = {}
        max_workers = len(active_routes) or 1
        pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="recall")
        futures: dict[Any, str] = {}
        try:
            if "exact" in active_routes:
                top_k = self._route_candidates.get("exact", 50)
                futures[pool.submit(self._timed, "exact", self._recall_exact, query, top_k)] = "exact"

            if "fuzzy" in active_routes:
                top_k = self._route_candidates.get("fuzzy", 20)
                futures[pool.submit(self._timed, "fuzzy", self._recall_fuzzy, query, top_k)] = "fuzzy"

            if "sparse" in active_routes:
                top_k = self._route_candidates.get("sparse", 50)
                futures[
                    pool.submit(self._timed, "sparse", self._recall_sparse, query, top_k, filters)
                ] = "sparse"

            if "dense" in active_routes:
                top_k = self._route_candidates.get("dense", 50)
                futures[
                    pool.submit(self._timed, "dense", self._recall_dense, query, top_k, filters)
                ] = "dense"

            finished: set[str] = set()
            try:
                for future in as_completed(futures, timeout=self._recall_timeout):
                    route_name = futures[future]
                    finished.add(route_name)
                    try:
                        results[route_name] = future.result()
                    except Exception:  # 某路挂掉不能拖垮整个检索：记错后按空结果降级
                        logger.exception("recall route=%s failed, degrade to empty", route_name)
                        results[route_name] = []
            except _FuturesTimeout:
                # 超出整次召回墙钟预算：未完成的各路按空结果降级，不继续等（如 Qdrant 挂住）
                pass
            for future, route_name in futures.items():
                if route_name not in finished:
                    future.cancel()
                    logger.warning(
                        "recall route=%s exceeded budget %.1fs, degrade to empty",
                        route_name,
                        self._recall_timeout,
                    )
                    results[route_name] = []
        finally:
            # wait=False：预算已到就立即返回，孤儿线程由其自身 client timeout 收尾
            pool.shutdown(wait=False, cancel_futures=True)

        return results

    @staticmethod
    def _timed(
        route_name: str, fn: Callable[..., list[dict[str, Any]]], *args: Any
    ) -> list[dict[str, Any]]:
        """在工作线程内度量单路召回的真实耗时与条数（DEBUG，不记 query 正文）。"""
        started = time.perf_counter()
        try:
            items = fn(*args)
        except Exception:
            logger.debug(
                "recall route=%s raised elapsed=%.3fs", route_name, time.perf_counter() - started
            )
            raise
        logger.debug(
            "recall route=%s count=%d elapsed=%.3fs",
            route_name,
            len(items),
            time.perf_counter() - started,
        )
        return items

    # ---------------------------------------------------------------- 各路召回实现

    @staticmethod
    def _recall_exact(query: str, top_k: int) -> list[dict[str, Any]]:
        """FTS5 精确路。"""
        from index.keyword_store import fts_search

        return fts_search(query, top_k=top_k)

    @staticmethod
    def _recall_fuzzy(query: str, top_k: int) -> list[dict[str, Any]]:
        """模糊路（RapidFuzz + n-gram + 同义词）。"""
        from retrieval.fuzzy_search import fuzzy_search

        return fuzzy_search(query, top_k=top_k)

    @staticmethod
    def _recall_sparse(
        query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """BGE-M3 sparse 路。"""
        from embedding.text_embedder import encode_query
        from index.vector_store import hybrid_search as qdrant_search

        emb = encode_query(query)
        if not emb.get("sparse"):
            return []
        # 只搜 sparse，不搜 dense（weights 里 dense 设 0 让 hybrid_search 跳过）
        results = qdrant_search(
            query_dense=[],
            query_sparse=emb["sparse"],
            top_k=top_k,
            filters=filters,
            weights={"dense": 0.0, "sparse": 1.0},
        )
        return [{"chunk_id": r["chunk_id"], "score": r["score"], **r} for r in results]

    @staticmethod
    def _recall_dense(
        query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """BGE-M3 dense 路。"""
        from embedding.text_embedder import encode_query
        from index.vector_store import hybrid_search as qdrant_search

        emb = encode_query(query)
        if not emb.get("dense"):
            return []
        # 只搜 dense
        results = qdrant_search(
            query_dense=emb["dense"],
            query_sparse=None,
            top_k=top_k,
            filters=filters,
            weights={"dense": 1.0, "sparse": 0.0},
        )
        return [{"chunk_id": r["chunk_id"], "score": r["score"], **r} for r in results]

    # ---------------------------------------------------------------- RRF 融合

    def _rrf_fusion(
        self,
        route_results: dict[str, list[dict[str, Any]]],
        weights: dict[str, float],
        k: int,
    ) -> list[dict[str, Any]]:
        """
        加权 RRF：score(cid) = sum(weight_route / (k + rank_in_route))。
        融合后按 score 降序，附带来源路信息。
        """
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}
        route_hits: dict[str, list[str]] = {}

        for route_name, items in route_results.items():
            w = weights.get(route_name, 1.0)
            if w <= 0:
                continue
            for rank, item in enumerate(items, start=1):
                cid = item.get("chunk_id", "")
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0) + w / (k + rank)
                route_hits.setdefault(cid, []).append(route_name)
                # 保留信息最丰富的 metadata（vector_store 返回的 payload 最全）
                if cid not in metadata or "payload" in item:
                    metadata[cid] = item

        # 组装排序结果
        fused: list[dict[str, Any]] = []
        for cid in sorted(scores, key=lambda c: scores[c], reverse=True):
            meta = metadata.get(cid, {})
            payload = meta.get("payload", {})
            fused.append({
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", payload.get("doc_id", "")),
                "content": meta.get("content", payload.get("content", "")),
                "rrf_score": round(scores[cid], 6),
                "source_routes": route_hits.get(cid, []),
                "metadata": payload,
            })
        return fused

    # ---------------------------------------------------------------- Rerank

    def _do_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """调用 reranker 精排。"""
        from retrieval.reranker import rerank

        return rerank(query, candidates, top_k=self._rerank_output_top_k)


# =================================================================== 便捷函数
# 模块级单例
_SEARCHER: HybridSearcher | None = None


def get_searcher() -> HybridSearcher:
    """全局检索器单例。"""
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = HybridSearcher()
    return _SEARCHER


def search(
    query: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """模块级便捷入口（等价于 get_searcher().search(...)）。"""
    return get_searcher().search(query, top_k=top_k, filters=filters, mode=mode)


def rrf_fusion(
    result_lists: list[list[dict[str, Any]]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    独立 RRF 融合函数（供外部测试或自定义融合用）。
    result_lists[i] 是第 i 路的 [{"chunk_id": ...}]，weights[i] 是对应权重。
    返回 [(chunk_id, fused_score), ...] 降序。
    """
    scores: dict[str, float] = {}
    for items, w in zip(result_lists, weights, strict=True):
        if w <= 0:
            continue
        for rank, item in enumerate(items, start=1):
            cid = item.get("chunk_id", "")
            if cid:
                scores[cid] = scores.get(cid, 0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
