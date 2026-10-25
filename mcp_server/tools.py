# -*- coding: utf-8 -*-
"""
MCP 工具定义：7 个对外问答/检索/入库工具。

设计要点：
  - 每个 tool 的 docstring 会被 FastMCP 提取为「工具描述」直接喂给 LLM，
    因此 docstring 必须准确说明「什么时候该用这个工具、参数含义、返回结构」；
  - 所有 tool 捕获异常并返回结构化错误（{"error": ..., "detail": ...}），
    绝不抛裸异常给 MCP 客户端（会让 LLM 看到一堆 traceback 无法处理）；
  - 真正的检索/入库逻辑都委托给既有模块，本层只做参数校验 + 结果整形。

工具注册方式：server.py 创建 FastMCP 实例后调用 register_tools(mcp)。
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from config.logging_setup import request_context

logger = logging.getLogger("rag_zh.mcp.tools")

# =================================================================== 内部辅助


def _fail(exc: Exception) -> dict[str, Any]:
    """
    统一的异常 → 结构化错误返回。

    以 ERROR 级 + traceback 落日志（仍吞掉异常不冒给 MCP 客户端）：工具层契约是
    把异常转成 {"error", "detail"}，但服务端排障需要看到完整栈。request_id 由外层
    _traced 上下文注入，能把这条 ERROR 与同一次调用的 start/end 串起来。
    """
    logger.exception("tool error type=%s detail=%s", type(exc).__name__, exc)
    return {"error": type(exc).__name__, "detail": str(exc)}


@contextlib.contextmanager
def _traced(tool: str, **safe_fields: Any) -> Iterator[None]:
    """
    工具调用埋点：进入时开一个 request_id 上下文并记 start，退出时记 elapsed。

    脱敏红线：safe_fields 只放**非正文元数据**（top_k/mode/过滤键/查询长度/id/路径），
    绝不放 query 正文或检索/答案内容（可能含入库文档敏感信息）。
    """
    with request_context() as rid:
        started = time.perf_counter()
        fields = " ".join(f"{k}={v}" for k, v in safe_fields.items() if v is not None)
        logger.info("tool start rid=%s tool=%s %s", rid, tool, fields)
        try:
            yield
        finally:
            logger.info(
                "tool end rid=%s tool=%s elapsed=%.3fs", rid, tool, time.perf_counter() - started
            )


@lru_cache(maxsize=1)
def _get_minio_client() -> Any:
    """
    MinIO 客户端单例（lru_cache 缓存一个连接对象，避免每次 presign 重建）。

    注：`presigned_get_object` 是本地签名（不走网络），单例只为省去重复构造开销；
    lru_cache 本身线程安全，极端并发首调可能各构造一次（幂等、无害），不值得为上双检锁。
    """
    from minio import Minio

    from ingestion.parsers.common import get_settings

    minio_cfg = get_settings().get("minio") or {}
    return Minio(
        str(minio_cfg.get("endpoint") or "localhost:9000"),
        access_key=str(minio_cfg.get("access_key") or ""),
        secret_key=str(minio_cfg.get("secret_key") or ""),
        secure=bool(minio_cfg.get("secure", False)),
        region=minio_cfg.get("region") or None,
    )


def _presigned_url(doc_id: str) -> str:
    """
    生成 MinIO 预签名下载 URL。
    对象命名规则（见 settings.minio.object_key_prefix）：raw/{doc_id}/...
    这里用 documents.path 记录的原始文件名定位对象。
    """
    from datetime import timedelta

    from index import metadata_store
    from ingestion.parsers.common import get_settings
    from mcp_server import security

    settings = get_settings()
    minio_cfg = settings.get("minio") or {}
    # 不再兜底 minioadmin：留空即视为未配置，交由 validate 拒绝签发（去弱默认凭据）
    access_key = str(minio_cfg.get("access_key") or "")
    secret_key = str(minio_cfg.get("secret_key") or "")
    bucket = str(minio_cfg.get("bucket") or "rag-zh-raw")
    prefix = str(minio_cfg.get("object_key_prefix") or "raw")

    # 凭据校验早于任何构造/签名操作；有效期取配置与上限的 min（默认收紧到 <=300s）
    security.validate_minio_credentials(access_key, secret_key)
    expire = security.presigned_expiry(settings)

    doc = metadata_store.get_document(doc_id)
    if doc is None:
        raise ValueError(f"文档不存在: {doc_id}")
    source_path = str(doc.get("path") or "")
    filename = source_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if source_path else doc_id
    object_key = f"{prefix}/{doc_id}/{filename}"

    client = _get_minio_client()
    url = client.presigned_get_object(bucket, object_key, expires=timedelta(seconds=expire))
    return str(url)


# =================================================================== 工具注册


def register_tools(mcp: Any) -> None:
    """把 7 个问答/检索/入库工具挂到 FastMCP 实例上。"""

    @mcp.tool()
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
        with _traced(
            "search", query_len=len(query), top_k=top_k, mode=mode, has_filters=bool(filters)
        ):
            try:
                from retrieval.hybrid_search import search as _search

                result = _search(query, top_k=top_k, filters=filters, mode=mode)
                results = result.get("results", [])
                logger.info("search returned count=%d", len(results))
                return results
            except Exception as exc:  # 工具层要吞掉所有异常转结构化错误
                return [_fail(exc)]

    @mcp.tool()
    def ask(
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        端到端问答（RAG）：先混合检索，再由大模型生成一段带引用的中文答案。

        何时用：用户想要「成文的答案/解释/总结」时（如“这份合同的违约条款讲了什么”）。
        只想拿原始片段做定位/列举时改用 search，不必用 ask（那会多花一次大模型调用）。

        参数：
          query: 自然语言问题
          top_k: 参考的检索片段条数，默认 5
          mode: 检索模式（同 search：hybrid/semantic/keyword/fuzzy）
          filters: 元数据过滤，如 {"doc_type": "contract"}

        返回 dict：
          {answer, answerable, confidence, degraded, refusal, reason, sources, retrieved_count, elapsed}
          - answer 非空：模型基于 sources 生成的答案，正文用 [N] 标注引用（N 对应 sources 下标+1）；
          - answer=None 且 answerable=False：知识库资料不足，refusal 是拒答话术（未硬编）；
          - answer=None 且 degraded=True：大模型不可用（未配 LLM_API_KEY / 鉴权 / 超时），
            此时 context 与 sources 仍是检索到的证据，可退化为展示检索结果。
        """
        with _traced(
            "ask",
            query_len=len(query),
            top_k=top_k,
            mode=mode,
            has_filters=bool(filters),
        ):
            try:
                from generation.answer_generator import get_generator

                out = get_generator().generate(query, top_k=top_k, mode=mode, filters=filters)
                logger.info(
                    "ask done answerable=%s degraded=%s sources=%d",
                    out.get("answerable"),
                    out.get("degraded"),
                    len(out.get("sources") or []),
                )
                return out
            except Exception as exc:
                return _fail(exc)

    @mcp.tool()
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
        with _traced("ingest", path=path):
            from ingestion.parsers.common import get_settings
            from mcp_server import security

            # 安全边界：MCP 是不可信远程面，先把读取路径锁在允许的入库根内
            try:
                resolved = security.confined_read_path(
                    path, security.allowed_ingest_roots(get_settings())
                )
            except security.PathNotAllowed as exc:
                logger.warning("ingest blocked path=%s reason=%s", path, exc)
                return {"error": f"路径不在允许的入库根内: {path}", "code": 400}
            try:
                from ingestion.pipeline import run

                summary = run(str(resolved), from_stage="parse", to_stage="index")
                logger.info(
                    "ingest done total=%s done=%s error=%s n_chunks=%s",
                    summary["total"],
                    summary["done"],
                    summary["error"],
                    summary["n_chunks"],
                )
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

    @mcp.tool()
    def get_chunk(chunk_id: str) -> dict[str, Any]:
        """
        按 chunk_id 取回某个文本片段的完整内容（含元数据、bbox、所属文档）。

        何时用：检索结果里 content 被截断、或需要某片段的完整上下文 / 定位信息时。

        参数：
          chunk_id: 片段唯一标识（来自 search 结果的 chunk_id 字段）

        返回：该 chunk 的完整记录；不存在时返回 {"error": "NotFound", ...}。
        """
        with _traced("get_chunk", chunk_id=chunk_id):
            try:
                from index import metadata_store

                chunk = metadata_store.get_chunk(chunk_id)
                if chunk is None:
                    logger.warning("get_chunk not found chunk_id=%s", chunk_id)
                    return {"error": "NotFound", "detail": f"chunk 不存在: {chunk_id}"}
                return chunk
            except Exception as exc:
                return _fail(exc)

    @mcp.tool()
    def list_documents(filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        列出已入库的文档清单（可按类型 / 语言 / 来源格式过滤）。

        何时用：用户问「知识库里有哪些文档 / 某个类型的文件」，或需要 doc_id 做后续操作。

        参数：
          filters: 可选，形如 {"doc_type": "...", "language": "zh", "source_format": "pdf",
                   "limit": 100, "offset": 0}

        返回：list，每条是一个文档记录 {doc_id, path, source_format, doc_type, language, created_at}。
        """
        with _traced("list_documents", has_filters=bool(filters)):
            try:
                from index import metadata_store

                f = filters or {}
                docs = metadata_store.list_documents(
                    doc_type=f.get("doc_type"),
                    language=f.get("language"),
                    source_format=f.get("source_format"),
                    limit=int(f.get("limit") or 100),
                    offset=int(f.get("offset") or 0),
                )
                logger.info("list_documents returned count=%d", len(docs))
                return docs
            except Exception as exc:
                return [_fail(exc)]

    @mcp.tool()
    def get_document_url(doc_id: str) -> str:
        """
        获取某原始文档的 MinIO 预签名下载链接（有效期取配置与上限的 min，默认 <=300 秒）。

        何时用：用户想「打开/下载原文」，或需要把原始文件展示给用户时。

        参数：
          doc_id: 文档 ID（来自 search 结果或 list_documents）

        返回：预签名 URL 字符串；凭据不安全时返回 {"error", "code": 403}；
        其它错误返回以 "error:" 开头的字符串。
        """
        with _traced("get_document_url", doc_id=doc_id):
            from mcp_server import security

            try:
                return _presigned_url(doc_id)
            except security.InsecureConfigError as exc:
                logger.warning("get_document_url denied doc_id=%s reason=%s", doc_id, exc)
                return {
                    "error": "MinIO 凭据缺失或为弱默认值，已拒绝签发下载链接",
                    "code": 403,
                }
            except Exception as exc:
                logger.exception(
                    "get_document_url failed doc_id=%s type=%s", doc_id, type(exc).__name__
                )
                return f"error: {type(exc).__name__}: {exc}"

    @mcp.tool()
    def classify_image(image_path: str) -> dict[str, Any]:
        """
        对一张图片做 zero-shot 场景分类（基于 Chinese-CLIP），返回类别标签与置信度。

        何时用：用户给了一张图，想知道「这是什么类型的内容」（图表/流程图/截图/照片等）。

        参数：
          image_path: 图片文件路径（png/jpg 等）。

        返回：{label, confidence, topk: {类别: 分数}, ...}；出错返回结构化错误。
        """
        with _traced("classify_image", image_path=image_path):
            from ingestion.parsers.common import get_settings
            from mcp_server import security

            # 与 ingest 同理：图片读取也先过路径沙箱，越界直接拒绝，不触发读字节/CLIP
            try:
                resolved = security.confined_read_path(
                    image_path, security.allowed_ingest_roots(get_settings())
                )
            except security.PathNotAllowed as exc:
                logger.warning("classify_image blocked path=%s reason=%s", image_path, exc)
                return {"error": f"图片路径不在允许的读取根内: {image_path}", "code": 400}
            try:
                from ingestion.classifiers.image_classifier import classify_image_path

                return classify_image_path(str(resolved))
            except Exception as exc:
                return _fail(exc)
