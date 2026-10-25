# -*- coding: utf-8 -*-
"""
数据生命周期编排：把一个文档的删除 / 更新在**全部存储**中级联生效。

此前散落的 delete_* 函数（keyword_store / metadata_store / vector_store 各管一片、
互不调用）有两个硬伤：一是没人把它们串起来（全仓无调用点），二是各表清理口径不一
（fts 用子查询、ngram 完全没清）。本模块做唯一的编排入口，把顺序与完整性收口。

覆盖的存储面（一次删除要同时清干净）：
  1. SQLite：documents / chunks / fts_chunks_virt / ngram_index（委托 metadata_store）
  2. Qdrant：该 doc 的全部向量点（best-effort，外部库不可达不阻断本地清理）
  3. 磁盘产物：parsed / chunks / vectors(.jsonl) / vectors(.index) 以及 CONTENT 的 by-path 缓存

红线：
  - 外部库（Qdrant/MinIO）不可达时**降级为 warning 而非抛异常**：CI 与离线环境没有
    Qdrant，SQLite 与磁盘清理仍必须完成；只有显式 strict=True 才把向量删除失败冒出来。
  - 日志只记元数据（doc_id / 各层计数 / 耗时），绝不记 chunk 正文（脱敏，对齐统一日志约定）。
  - 幂等：重复删同一 doc_id 不报错（DELETE 命中 0 行、unlink(missing_ok=True) 都安全）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from config.logging_setup import get_logger, request_context
from index import metadata_store

logger = get_logger("rag_zh.lifecycle")


def _purge_artifacts(doc_id: str, root: str | None) -> list[str]:
    """
    删磁盘上属于该 doc 的阶段产物，返回被删文件的路径列表。

    懒导入 pipeline（避免 index <-> ingestion 模块级循环 import）。缺文件一律
    missing_ok 跳过；by-path 缓存以「路径指纹」为文件名、内容是 content doc_id，
    所以只能扫目录挑出指向本 doc_id 的那些（仅 CONTENT 策略会产生，扫不到也不报错）。
    """
    from ingestion.pipeline import artifact_path, stage_dir

    removed: list[str] = []
    for stage in ("parse", "chunk", "embed", "index"):
        path: Path = artifact_path(stage, doc_id, root)
        if path.exists():
            path.unlink(missing_ok=True)
            removed.append(path.name)

    by_path = stage_dir("parse", root) / "by-path"
    if by_path.is_dir():
        for cache in by_path.glob("*.id"):
            try:
                if cache.read_text(encoding="utf-8").strip() == doc_id:
                    cache.unlink(missing_ok=True)
                    removed.append(f"by-path/{cache.name}")
            except OSError:  # pragma: no cover - 缓存文件读失败就留着，不影响主流程
                continue
    return removed


def _purge_vectors(doc_id: str, warnings: list[str], strict: bool) -> bool:
    """
    删 Qdrant 中该 doc 的全部向量；best-effort。

    成功返回 True。外部库不可达 / 出错时：strict=True 直接抛出，否则记 warning 返回 False，
    让调用方在无 Qdrant 的环境里仍能把本地 SQLite + 磁盘清理走完。
    """
    from index import vector_store

    try:
        vector_store.delete_by_doc_id(doc_id)
        return True
    except Exception as exc:  # 外部存储异常种类多，这里就是要全兜住做降级
        logger.warning("vector purge degraded doc_id=%s error=%s", doc_id, str(exc)[:200])
        if strict:
            raise
        warnings.append(f"vectors: {type(exc).__name__}: {exc}")
        return False


def purge_index(doc_id: str, *, root: str | None = None, vectors: bool = True, strict: bool = False) -> dict[str, Any]:
    """
    只清「索引层」：SQLite(documents/chunks/fts/ngram) + Qdrant 向量，**保留**磁盘产物。

    给 do_index 重索引前调用：先把本文档旧索引行/向量整体删掉，再写新数据，杜绝
    chunk_id 变化遗留的孤儿。返回 {doc_id, sqlite, vectors}。
    """
    with request_context() as rid:
        started = time.perf_counter()
        warnings: list[str] = []
        logger.info("purge_index start rid=%s doc_id=%s vectors=%s", rid, doc_id, vectors)
        sqlite_counts = metadata_store.purge_document_sqlite(doc_id)
        vectors_done = _purge_vectors(doc_id, warnings, strict) if vectors else False
        elapsed = time.perf_counter() - started
        logger.info(
            "purge_index done rid=%s doc_id=%s sqlite=%s vectors=%s elapsed=%.3fs",
            rid,
            doc_id,
            sqlite_counts,
            vectors_done,
            elapsed,
        )
        return {"doc_id": doc_id, "sqlite": sqlite_counts, "vectors": vectors_done, "warnings": warnings}


def delete_document(
    doc_id: str,
    *,
    root: str | None = None,
    purge_artifacts: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """
    跨全部存储级联删除一个文档（SQLite + Qdrant + 磁盘产物）。

    参数：
      doc_id: 目标文档 id。
      root: 产物根目录（透传 pipeline 的 artifact_path / stage_dir），None 走配置默认。
      purge_artifacts: 是否一并删 parsed/chunks/vectors(.jsonl/.index)/by-path 缓存。
      strict: True 时向量删除失败即抛出；默认 False，降级为 warning，本地清理照常完成。

    返回 report：{doc_id, removed: {sqlite, vectors, artifacts}, warnings: [...]}。
    幂等：重复删同一 doc_id 不报错（各层都是「删不到就算过」）。
    """
    with request_context() as rid:
        started = time.perf_counter()
        warnings: list[str] = []
        logger.info(
            "delete start rid=%s doc_id=%s purge_artifacts=%s strict=%s",
            rid,
            doc_id,
            purge_artifacts,
            strict,
        )
        sqlite_counts = metadata_store.purge_document_sqlite(doc_id)
        vectors_done = _purge_vectors(doc_id, warnings, strict)
        artifacts = _purge_artifacts(doc_id, root) if purge_artifacts else []
        elapsed = time.perf_counter() - started
        logger.info(
            "delete done rid=%s doc_id=%s sqlite=%s vectors=%s artifacts=%d elapsed=%.3fs",
            rid,
            doc_id,
            sqlite_counts,
            vectors_done,
            len(artifacts),
            elapsed,
        )
        return {
            "doc_id": doc_id,
            "removed": {"sqlite": sqlite_counts, "vectors": vectors_done, "artifacts": artifacts},
            "warnings": warnings,
        }
