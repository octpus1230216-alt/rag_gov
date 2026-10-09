# -*- coding: utf-8 -*-
"""
文本向量化：BGE-M3（dense + sparse 同时产出）与降级 bge-small-zh-v1.5（仅 dense）。

设计要点：
  1. 单例模式（lru_cache）：模型权重体积大（~2GB），同进程只加载一次；
     multiprocessing.Pool spawn 出子进程时各进程独立持有，由 batch_size 控吞吐。
  2. profile 参数驱动：embedding_model / embedding_dim / supports_sparse /
     batch_size / use_fp16 / max_seq_length 全部从 settings.active 读取，
     切档只改配置不改代码。
  3. sparse 输出格式转换：BGE-M3 的 lexical_weights 是 {token_id: weight} 字典，
     Qdrant 要求 SparseVector(indices=[int], values=[float])，
     encode_documents 在出口处就完成转换，下游不再碰原始格式。
  4. dense 向量 L2 归一化：配合 cosine 距离时点积等于相似度，
     Qdrant 的 cosine 虽内部会归一化，但提前做可以减少网络传输浮点精度损失。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ingestion.parsers.common import get_settings


@dataclass
class EmbeddingResult:
    """单条文本的编码结果。"""
    dense: list[float]
    sparse: dict[str, list] | None  # {"indices": [int], "values": [float]} or None


class TextEmbedder:
    """
    BGE-M3 / bge-small-zh-v1.5 封装。
    外部只通过 get_embedder() 拿到实例，不直接 new（保证单例且配置一致）。
    """

    def __init__(self) -> None:
        active = get_settings().get("active") or {}
        self._model_name: str = str(active.get("embedding_model") or "BAAI/bge-m3")
        self._dim: int = int(active.get("embedding_dim") or 1024)
        self._supports_sparse: bool = bool(active.get("supports_sparse", True))
        self._batch_size: int = int(active.get("batch_size") or 16)
        self._use_fp16: bool = bool(active.get("use_fp16", False))
        self._max_seq_length: int = int(active.get("max_seq_length") or 1024)
        self._normalize: bool = bool(
            (get_settings().get("embedding") or {}).get("normalize_dense", True)
        )
        self._model: Any = None
        self._device: str = str(active.get("device") or "cpu")

    def _load_model(self) -> None:
        """懒加载：第一次 encode 时才把权重读进内存/显存。"""
        if self._model is not None:
            return
        if self._supports_sparse:
            # BGE-M3：同时支持 dense + sparse(lexical_weights) + colbert + multi-vector
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=self._use_fp16,
                device=self._device if self._device != "auto" else None,
                max_length=self._max_seq_length,
            )
        else:
            # 降级：bge-small-zh-v1.5 只有 dense，没有 sparse 能力
            from FlagEmbedding import FlagModel

            self._model = FlagModel(
                self._model_name,
                use_fp16=self._use_fp16,
                query_instruction_for_retrieval="",
                devices=[self._device] if self._device != "auto" else None,
                max_length=self._max_seq_length,
            )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return self._supports_sparse

    def encode_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[EmbeddingResult]:
        """
        批量编码文档文本 -> 每条一个 EmbeddingResult。
        BGE-M3 一次 forward 同时拿 dense 和 lexical_weights，省掉重复 encode。
        batch_size 默认从 profile 取，允许调用方覆盖（内存压力大时手动调小）。
        """
        if not texts:
            return []
        self._load_model()
        bs = batch_size or self._batch_size
        results: list[EmbeddingResult] = []

        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            if self._supports_sparse:
                output = self._model.encode(
                    batch,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                dense_vecs = output["sentence_dense_vector"]
                lexical_weights_list = output["lexical_weights"]
                for i, dv in enumerate(dense_vecs):
                    dense = self._finalize_dense(dv)
                    sparse = self._convert_sparse(lexical_weights_list[i])
                    results.append(EmbeddingResult(dense=dense, sparse=sparse))
            else:
                dense_vecs = self._model.encode(batch)
                for dv in dense_vecs:
                    dense = self._finalize_dense(dv)
                    results.append(EmbeddingResult(dense=dense, sparse=None))

        return results

    def encode_query(self, text: str) -> EmbeddingResult:
        """编码单条查询（与文档侧同一模型，BGE-M3 query/document 不需要不同指令）。"""
        out = self.encode_documents([text])
        return out[0]

    # ---------------------------------------------------------------- 内部工具

    def _finalize_dense(self, vec: Any) -> list[float]:
        """numpy array -> list[float]，可选 L2 归一化。"""
        import numpy as np

        arr = np.asarray(vec, dtype=np.float32)
        if self._normalize:
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
        return arr.tolist()

    @staticmethod
    def _convert_sparse(lexical_weights: dict[int, float]) -> dict[str, list]:
        """
        BGE-M3 lexical_weights: {token_id(int): weight(float)}
        转成 Qdrant SparseVector 需要的 {"indices": [...], "values": [...]} 格式。
        过滤掉权重为 0 的 token（BGE-M3 偶尔输出 0 权 token，无意义且浪费索引空间）。
        """
        indices: list[int] = []
        values: list[float] = []
        for token_id, weight in lexical_weights.items():
            w = float(weight)
            if w > 0:
                indices.append(int(token_id))
                values.append(w)
        return {"indices": indices, "values": values}


# =================================================================== 单例入口
@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedder:
    """全局单例：同进程内多次调用返回同一实例（模型只加载一次）。"""
    return TextEmbedder()


# =================================================================== 便捷函数
def encode_documents(
    texts: list[str], batch_size: int | None = None
) -> list[dict[str, Any]]:
    """
    外部调用入口（需求点名函数）：返回 list[{dense, sparse}]。
    sparse 为 {"indices": [...], "values": [...]} 或 None（降级模型）。
    """
    embedder = get_embedder()
    results = embedder.encode_documents(texts, batch_size)
    return [{"dense": r.dense, "sparse": r.sparse} for r in results]


def encode_query(text: str) -> dict[str, Any]:
    """编码单条 query，格式与 encode_documents 单条输出一致。"""
    embedder = get_embedder()
    r = embedder.encode_query(text)
    return {"dense": r.dense, "sparse": r.sparse}
