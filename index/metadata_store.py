# -*- coding: utf-8 -*-
"""
元数据存储：documents / chunks 表的 CRUD。

与 keyword_store 共享同一个 sqlite 文件（exact_db_path），
表由 keyword_store.init_tables() 统一创建，本模块只负责数据操作。

设计考量：
  - documents 表记录源文件级信息（路径、格式、哈希、入库时间），
    方便「这个文件入过库没 / 在哪个路径」类查询，不用扫 parsed/ 目录；
  - chunks 表存 chunk 原文 + metadata JSON，供检索展示与回溯；
    （与 keyword_store 写入的是同一张表，两边互补：keyword_store 写 fts，本模块写 doc 级查询）
  - list_documents 支持按 doc_type/language/source_format 过滤，
    用于管理后台的「查看已入库文档列表」。
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from ingestion.parsers.common import utc_now_iso


# =================================================================== Documents CRUD
def insert_document(doc: dict[str, Any]) -> None:
    """
    插入或更新文档记录。
    doc 必须含 doc_id，其余字段可选（缺省写空串）。
    """
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO documents(doc_id, path, source_format, doc_type, language, sha256, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            doc.get("doc_id", ""),
            doc.get("path", ""),
            doc.get("source_format", ""),
            doc.get("doc_type", "unknown"),
            doc.get("language", "zh"),
            doc.get("sha256", ""),
            doc.get("created_at") or utc_now_iso(),
        ),
    )
    conn.commit()


def get_document(doc_id: str) -> dict[str, Any] | None:
    """按 doc_id 查询单条文档，不存在返回 None。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_documents(
    doc_type: str | None = None,
    language: str | None = None,
    source_format: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    按条件过滤列出文档（全部参数为 None 时无 WHERE 子句，返回全量分页）。
    """
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    conditions: list[str] = []
    params: list[Any] = []

    if doc_type:
        conditions.append("doc_type = ?")
        params.append(doc_type)
    if language:
        conditions.append("language = ?")
        params.append(language)
    if source_format:
        conditions.append("source_format = ?")
        params.append(source_format)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def delete_document(doc_id: str) -> None:
    """删除文档记录及其全部 chunk（级联清理）。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM fts_chunks_virt WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)", (doc_id,))
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()


def count_documents() -> int:
    """已入库文档总数。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
    return int(row["n"]) if row else 0


# =================================================================== Chunks CRUD
def get_chunk(chunk_id: str) -> dict[str, Any] | None:
    """按 chunk_id 取单条 chunk（含原文 content、metadata、bbox 等）。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    # 反序列化 JSON 字段
    if result.get("bbox"):
        with suppress(json.JSONDecodeError, TypeError):
            result["bbox"] = json.loads(result["bbox"])
    if result.get("metadata"):
        with suppress(json.JSONDecodeError, TypeError):
            result["metadata"] = json.loads(result["metadata"])
    return result


def get_chunks_by_doc(doc_id: str) -> list[dict[str, Any]]:
    """取某个文档的全部 chunk（按写入顺序）。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        if d.get("bbox"):
            with suppress(json.JSONDecodeError, TypeError):
                d["bbox"] = json.loads(d["bbox"])
        if d.get("metadata"):
            with suppress(json.JSONDecodeError, TypeError):
                d["metadata"] = json.loads(d["metadata"])
        out.append(d)
    return out


def count_chunks() -> int:
    """已入库 chunk 总数。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
    return int(row["n"]) if row else 0


def count_chunks_by_doc(doc_id: str) -> int:
    """某文档的 chunk 数。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
    return int(row["n"]) if row else 0


# =================================================================== 批量入库（供 pipeline embed/index 阶段调用）
def register_parsed_document(
    parsed: dict[str, Any],
    source_path: str = "",
) -> None:
    """
    把一个 parsed 文档注册进 documents 表（embed/index 阶段入口调一次）。
    从 parsed dict 提取 doc_id、source_format、metadata 等字段。
    """
    doc_id = str(parsed.get("doc_id", ""))
    if not doc_id:
        return
    meta = parsed.get("metadata") or {}
    insert_document({
        "doc_id": doc_id,
        "path": source_path or str(parsed.get("source_file", "")),
        "source_format": str(parsed.get("source_format", "")),
        "doc_type": str(meta.get("doc_type", "unknown")),
        "language": str(meta.get("language", "zh")),
        "sha256": str(parsed.get("bytes_doc_id", doc_id)),
        "created_at": utc_now_iso(),
    })
