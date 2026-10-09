# -*- coding: utf-8 -*-
"""
MCP 工具定义：6 个对外检索/入库工具。

设计要点：
  - 每个 tool 的 docstring 会被 FastMCP 提取为「工具描述」直接喂给 LLM，
    因此 docstring 必须准确说明「什么时候该用这个工具、参数含义、返回结构」；
  - 所有 tool 捕获异常并返回结构化错误（{"error": ..., "detail": ...}），
    绝不抛裸异常给 MCP 客户端（会让 LLM 看到一堆 traceback 无法处理）；
  - 真正的检索/入库逻辑都委托给既有模块，本层只做参数校验 + 结果整形。

工具注册方式：server.py 创建 FastMCP 实例后调用 register_tools(mcp)。
"""

from __future__ import annotations

from typing import Any

# =================================================================== 内部辅助


def _fail(exc: Exception) -> dict[str, Any]:
    """统一的异常 → 结构化错误返回。"""
    return {"error": type(exc).__name__, "detail": str(exc)}


def _presigned_url(doc_id: str) -> str:
    """
    生成 MinIO 预签名下载 URL。
    对象命名规则（见 settings.minio.object_key_prefix）：raw/{doc_id}/...
    这里用 documents.path 记录的原始文件名定位对象。
    """
    from datetime import timedelta

    from minio import Minio

    from index import metadata_store
    from ingestion.parsers.common import get_settings

    minio_cfg = get_settings().get("minio") or {}
    endpoint = str(minio_cfg.get("endpoint") or "localhost:9000")
    access_key = str(minio_cfg.get("access_key") or "minioadmin")
    secret_key = str(minio_cfg.get("secret_key") or "minioadmin")
    bucket = str(minio_cfg.get("bucket") or "rag-zh-raw")
    prefix = str(minio_cfg.get("object_key_prefix") or "raw")
    secure = bool(minio_cfg.get("secure", False))
    expire = int(minio_cfg.get("presigned_expire_seconds") or 3600)

    doc = metadata_store.get_document(doc_id)
    if doc is None:
        raise ValueError(f"文档不存在: {doc_id}")
    source_path = str(doc.get("path") or "")
    filename = source_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if source_path else doc_id
    object_key = f"{prefix}/{doc_id}/{filename}"

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    url = client.presigned_get_object(bucket, object_key, expires=timedelta(seconds=expire))
    return str(url)


# =================================================================== 工具注册


def register_tools(mcp: Any) -> None:
    """把 6 个检索/入库工具挂到 FastMCP 实例上。"""

    @mcp.tool
    def search(
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        在中文知识库上做混合检索，返回最相关的文本片段（带引用信息）。

        何时用：用户想「查找/定位文档内容」时。这是最常用的检索入口。

        参数：
          query: 自然语言问题或关键词（支持中文、引号精确匹配、法条编号等）
          top_k: 返回条数，默认 5
          mode: 检索模式
                - "hybrid"（默认）：exact+模糊+sparse+dense 四路融合，最全面
                - "semantic"：只走 dense 向量，适合开放式语义问题
                - "keyword"：只走 FTS5 + sparse，适合含专有名词/编号的查询
                - "fuzzy"：只走模糊路，适合疑似拼写/表述不确定的关键词
          filters: 可选的元数据过滤，如 {"doc_type": "contract", "language": "zh", "has_table": true}

        返回：list，每条含
          {chunk_id, content, score, metadata, section_path, doc_id, bbox}
          其中 bbox 非空表示该片段来自 PDF 定位区域，可用于原文高亮。
        """
        try:
            from retrieval.hybrid_search import search as _search

            result = _search(query, top_k=top_k, filters=filters, mode=mode)
            return result.get("results", [])
        except Exception as exc:  # 工具层要吞掉所有异常转结构化错误
            return [_fail(exc)]

    @mcp.tool
    def ingest(path: str) -> dict[str, Any]:
        """
        导入一个文件或整个目录到知识库（跑完整入库流水线：解析→分类→切分→向量化→建索引）。

        何时用：用户新增了文档、要求「入库/更新/同步」某些文件时。

        参数：
          path: 文件或目录的绝对/相对路径。目录会递归处理。

        返回：
          {total, done, partial, error, n_chunks, elapsed, reports}
          来自 pipeline 的汇总；reports 内含每个文件的处理状态与失败原因。
          幂等：已入库且未变化的文件会被跳过（看 done 里的 fully_skipped）。
        """
        try:
            from ingestion.pipeline import run

            summary = run(path, from_stage="parse", to_stage="index")
            return {
                "total": summary["total"],
                "done": summary["done"],
                "partial": summary["partial"],
                "error": summary["error"],
                "n_chunks": summary["n_chunks"],
                "elapsed": summary["elapsed"],
                "reports": summary["reports"],
            }
        except Exception as exc:
            return _fail(exc)

    @mcp.tool
    def get_chunk(chunk_id: str) -> dict[str, Any]:
        """
        按 chunk_id 取回某个文本片段的完整内容（含元数据、bbox、所属文档）。

        何时用：检索结果里 content 被截断、或需要某片段的完整上下文 / 定位信息时。

        参数：
          chunk_id: 片段唯一标识（来自 search 结果的 chunk_id 字段）

        返回：该 chunk 的完整记录；不存在时返回 {"error": "NotFound", ...}。
        """
        try:
            from index import metadata_store

            chunk = metadata_store.get_chunk(chunk_id)
            if chunk is None:
                return {"error": "NotFound", "detail": f"chunk 不存在: {chunk_id}"}
            return chunk
        except Exception as exc:
            return _fail(exc)

    @mcp.tool
    def list_documents(filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        列出已入库的文档清单（可按类型 / 语言 / 来源格式过滤）。

        何时用：用户问「知识库里有哪些文档 / 某个类型的文件」，或需要 doc_id 做后续操作。

        参数：
          filters: 可选，形如 {"doc_type": "...", "language": "zh", "source_format": "pdf",
                   "limit": 100, "offset": 0}

        返回：list，每条是一个文档记录 {doc_id, path, source_format, doc_type, language, created_at}。
        """
        try:
            from index import metadata_store

            f = filters or {}
            return metadata_store.list_documents(
                doc_type=f.get("doc_type"),
                language=f.get("language"),
                source_format=f.get("source_format"),
                limit=int(f.get("limit") or 100),
                offset=int(f.get("offset") or 0),
            )
        except Exception as exc:
            return [_fail(exc)]

    @mcp.tool
    def get_document_url(doc_id: str) -> str:
        """
        获取某原始文档的 MinIO 预签名下载链接（有效期见配置，默认 1 小时）。

        何时用：用户想「打开/下载原文」，或需要把原始文件展示给用户时。

        参数：
          doc_id: 文档 ID（来自 search 结果或 list_documents）

        返回：预签名 URL 字符串；出错时返回以 "error:" 开头的字符串。
        """
        try:
            return _presigned_url(doc_id)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    @mcp.tool
    def classify_image(image_path: str) -> dict[str, Any]:
        """
        对一张图片做 zero-shot 场景分类（基于 Chinese-CLIP），返回类别标签与置信度。

        何时用：用户给了一张图，想知道「这是什么类型的内容」（图表/流程图/截图/照片等）。

        参数：
          image_path: 图片文件路径（png/jpg 等）。

        返回：{label, confidence, topk: {类别: 分数}, ...}；出错返回结构化错误。
        """
        try:
            from ingestion.classifiers.image_classifier import classify_image_path

            return classify_image_path(image_path)
        except Exception as exc:
            return _fail(exc)
