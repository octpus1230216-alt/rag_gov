# -*- coding: utf-8 -*-
"""
重排序：BAAI/bge-reranker-v2-m3（交叉编码器）。

用途：
  四路召回 + RRF 融合后取 top20 送进 reranker 做精排，输出 top10。
  交叉编码器把 (query, passage) 拼一起过 Transformer 出相关性分数，
  精度远高于一期召回的 bi-encoder / BM25，是最后一道质量门。

性能约束：
  - lite 档 CPU：top20→top10 目标 < 300ms（batch_size=4，max_length=512）；
  - standard/performance 档 GPU：batch_size 按 profile，max_length 可到 1024。

单例：@lru_cache(maxsize=1)，同进程模型只加载一次。
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from ingestion.parsers.common import get_settings


class Reranker:
    """BGE-reranker-v2-m3 封装。"""

    def __init__(self) -> None:
        active = get_settings().get("active") or {}
        rerank_cfg = active.get("reranker") or {}
        self._model_name: str = str(rerank_cfg.get("model") or "BAAI/bge-reranker-v2-m3")
        self._max_length: int = int(rerank_cfg.get("max_length") or 512)
        self._batch_size: int = int(rerank_cfg.get("batch_size") or 4)
        self._device: str = str(active.get("device") or "cpu")
        self._use_fp16: bool = bool(active.get("use_fp16", False))
        self._model: Any = None
        # 懒加载锁（双检）：与 TextEmbedder 同理，避免并发首次精排时重复构造模型。
        self._load_lock: threading.Lock = threading.Lock()

    def _load(self) -> None:
        """懒加载 cross-encoder 模型（双检锁，并发下只构造一次）。"""
        if self._model is not None:  # 快路径
            return
        with self._load_lock:  # 慢路径：串行化构造
            if self._model is not None:  # 二次检查
                return
            from FlagEmbedding import FlagReranker

            device = self._device
            # FlagReranker 的 device 参数用 "cpu"/"cuda" 格式
            if device in ("auto",):
                device = "cpu"
            use_fp16 = self._use_fp16 and device != "cpu"
            self._model = FlagReranker(
                self._model_name,
                use_fp16=use_fp16,
                devices=[device] if device != "cpu" else None,
                max_length=self._max_length,
            )

    def compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        批量计算 (query, passage) 对的相关性分数。
        FlagReranker.compute_score 支持多种格式，这里用 pair list。
        """
        if not pairs:
            return []
        self._load()
        scores = self._model.compute_score(pairs, normalize=True)
        # 单条时返回标量，多条返回 list
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        精排入口：给候选列表按 rerank score 降序排列，返回 top_k。
        candidates: [{"chunk_id", "content", "score", ...}]
        返回同结构 + "rerank_score" 字段。
        """
        if not candidates:
            return []

        pairs: list[tuple[str, str]] = []
        for c in candidates:
            content = str(c.get("content", ""))[:self._max_length * 2]  # 预截断避免超长
            pairs.append((query, content))

        scores = self.compute_scores(pairs)

        # 附加分数
        scored: list[dict[str, Any]] = []
        for c, s in zip(candidates, scores, strict=True):
            item = dict(c)
            item["rerank_score"] = round(s, 6)
            scored.append(item)

        # 降序排列
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


# =================================================================== 单例
@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """全局单例：同进程模型只加载一次。"""
    return Reranker()


# =================================================================== 便捷函数
def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    精排便捷入口（需求点名函数）。
    candidates: [{"chunk_id": str, "content": str, "score": float, ...}]
    返回: 同结构 + "rerank_score"，按 rerank_score 降序取 top_k。
    """
    return get_reranker().rerank(query, candidates, top_k=top_k)
