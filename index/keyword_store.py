# -*- coding: utf-8 -*-
"""
精确索引：SQLite FTS5 + jieba 预分词 + 同义词/停用词管理。

红线：FTS5 不用默认 tokenizer 处理中文（默认按空格/标点切，中文没有空格，
     整段正文会被当成一个 token，永远命中不了）。
解决：写入前用 jieba 分词、token 间补空格；查询时同样 jieba 切词后拼接匹配。

表结构（同一 sqlite 文件内）：
  documents     — 文档级元信息（doc_id 主键）
  chunks        — chunk 原文 + 结构化字段
  fts_chunks    — FTS5 虚拟表，content 列存 jieba 预分词结果
  ngram_index   — n-gram 倒排（由 ngram_index.py 管理写入，这里只建表）
  vocab         — 词表 df/idf（由 vocab_builder.py 写入）
  synonyms      — 同义词展开表（由本模块 load_synonyms 灌入）

连接管理：
  模块级单例 + WAL 模式（并发读写不阻塞），每个线程拿独立连接（threading.local）。
  SQLite 在 WAL + synchronous=NORMAL 下读者不阻塞写者，多进程入库天然安全。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from ingestion.parsers.common import get_settings

# 模块级：线程本地存储，保证每个线程有自己的 connection
_LOCAL = threading.local()
_DB_PATH: Path | None = None
_INITIALISED = False


def _db_path() -> Path:
    """取 sqlite.exact_db_path（相对于项目根）。"""
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH
    scfg = get_settings().get("sqlite") or {}
    raw = str(scfg.get("exact_db_path") or "storage/sqlite/exact_fts.sqlite")
    _DB_PATH = Path(raw)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    """获取当前线程的 SQLite 连接（不存在则创建）。"""
    conn = getattr(_LOCAL, "conn", None)
    if conn is not None:
        return conn
    path = _db_path()
    scfg = get_settings().get("sqlite") or {}
    journal = str(scfg.get("journal_mode") or "wal")
    sync = str(scfg.get("synchronous") or "normal")
    timeout = int(scfg.get("timeout_seconds") or 30)

    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.execute(f"PRAGMA journal_mode={journal}")
    conn.execute(f"PRAGMA synchronous={sync}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _LOCAL.conn = conn
    return conn


def close_connection() -> None:
    """关闭当前线程的连接（测试或 shutdown 时用）。"""
    conn = getattr(_LOCAL, "conn", None)
    if conn:
        conn.close()
        _LOCAL.conn = None


# =================================================================== 建表
def init_tables() -> None:
    """创建所有表（IF NOT EXISTS，幂等）。"""
    global _INITIALISED
    if _INITIALISED:
        return
    conn = get_connection()

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        path TEXT NOT NULL DEFAULT '',
        source_format TEXT NOT NULL DEFAULT '',
        doc_type TEXT NOT NULL DEFAULT 'unknown',
        language TEXT NOT NULL DEFAULT 'zh',
        sha256 TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        content TEXT NOT NULL,
        chunk_type TEXT NOT NULL DEFAULT 'text',
        section_path TEXT NOT NULL DEFAULT '',
        bbox TEXT,
        metadata TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    );

    CREATE TABLE IF NOT EXISTS fts_chunks (
        chunk_id UNINDEXED,
        content TEXT
    );

    CREATE TABLE IF NOT EXISTS ngram_index (
        gram TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (gram, chunk_id)
    );

    CREATE TABLE IF NOT EXISTS vocab (
        term TEXT PRIMARY KEY,
        df INTEGER NOT NULL DEFAULT 0,
        idf REAL NOT NULL DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS synonyms (
        term TEXT NOT NULL,
        synonym TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        PRIMARY KEY (term, synonym)
    );
    """)

    # FTS5 虚拟表（sqlite 编译时需要含 fts5 模块，Python 3.10+ 标准库都带了）
    # 用 content='' 的外部内容模式：真正的 chunk content 存在 chunks 表里
    # 这里 fts_chunks 单独存 jieba 预分词后的文本
    with suppress(sqlite3.OperationalError):
        conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_virt
        USING fts5(
            chunk_id UNINDEXED,
            content,
            tokenize='unicode61'
        );
        """)

    conn.commit()
    # 表建好后跑 schema 迁移（PRAGMA user_version 驱动，幂等）：把既有库升到 SCHEMA_VERSION
    from index.schema_migrations import run_migrations

    run_migrations(conn)
    _INITIALISED = True


# =================================================================== jieba 分词
def _jieba_tokens(text: str) -> list[str]:
    """jieba 精确模式分词，过滤单字和纯空白 token。"""
    import jieba

    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


def _tokenize_for_fts(text: str) -> str:
    """分词后用空格拼接（FTS5 unicode61 tokenizer 按空格/标点切 token）。"""
    return " ".join(_jieba_tokens(text))


# =================================================================== 写入
def _chunk_row(chunk: dict[str, Any]) -> tuple[Any, ...]:
    """
    把一个 chunk 整理成 chunks 表的 7 元组（chunk_id, doc_id, content, chunk_type,
    section_path, bbox, metadata）。单条写与批量写共用，保证 bbox/metadata 的
    JSON 序列化口径与列顺序完全一致。
    """
    bbox_json = json.dumps(chunk.get("bbox")) if chunk.get("bbox") else None
    meta_json = json.dumps(chunk.get("metadata") or {}, ensure_ascii=False)
    return (
        chunk.get("chunk_id", ""),
        chunk.get("doc_id", ""),
        chunk.get("content", ""),
        chunk.get("chunk_type", "text"),
        chunk.get("section_path", ""),
        bbox_json,
        meta_json,
    )


def insert_chunk(chunk: dict[str, Any]) -> None:
    """
    写入一个 chunk 到 chunks 表 + FTS5 虚拟表。
    同一 chunk_id 重复写入会被 INSERT OR REPLACE 覆盖（幂等）。
    """
    init_tables()
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, content, chunk_type, section_path, bbox, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        _chunk_row(chunk),
    )

    # FTS5 写 jieba 预分词结果
    tokenized = _tokenize_for_fts(chunk.get("content", ""))
    conn.execute(
        "INSERT OR REPLACE INTO fts_chunks_virt(chunk_id, content) VALUES (?, ?)",
        (chunk.get("chunk_id", ""), tokenized),
    )
    conn.commit()


def insert_chunks_batch(chunks: list[dict[str, Any]]) -> int:
    """
    批量写入 chunk（chunks + fts 两张表），返回成功条数。

    与逐条 insert_chunk 的区别：**整批只提交一次事务**。逐条写会在每个 chunk 上
    触发一次 WAL fsync（N 条 = N 次 commit），大批量入库时是主要瓶颈；这里用两次
    executemany + 一次 commit，中途异常整批 rollback，不留半批。
    幂等口径不变（INSERT OR REPLACE）。
    """
    if not chunks:
        return 0
    init_tables()
    conn = get_connection()
    chunk_rows = [_chunk_row(c) for c in chunks]
    fts_rows = [(c.get("chunk_id", ""), _tokenize_for_fts(c.get("content", ""))) for c in chunks]
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, content, chunk_type, section_path, bbox, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            chunk_rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO fts_chunks_virt(chunk_id, content) VALUES (?, ?)",
            fts_rows,
        )
        conn.commit()  # 整批一次提交
    except Exception:
        conn.rollback()  # 半途失败不留下半批
        raise
    return len(chunks)


def delete_chunks_by_doc(doc_id: str) -> None:
    """删除某文档的全部 chunk（重建索引前清理）。"""
    init_tables()
    conn = get_connection()
    # 先清 fts 虚拟表（FTS5 不支持按 UNINDEXED 列过滤删除，需查出 chunk_id）
    rows = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    for row in rows:
        conn.execute("DELETE FROM fts_chunks_virt WHERE chunk_id = ?", (row["chunk_id"],))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()


# =================================================================== FTS5 搜索
def fts_search(query: str, top_k: int = 50) -> list[dict[str, Any]]:
    """
    FTS5 精确检索：query jieba 分词 -> " ".join -> MATCH。
    返回 [{"chunk_id": ..., "content": ..., "score": bm25_score}, ...]
    bm25 分数越小相关性越高（SQLite FTS5 的特性），这里取反后统一为越大越好。
    """
    init_tables()
    conn = get_connection()
    tokens = _jieba_tokens(query)
    if not tokens:
        return []

    # 构造 FTS5 MATCH 表达式：各 token 用 OR 连接（任一命中即为候选）
    # 多词 AND 更严格但容易漏召回，这里用 OR 取宽召回，精排交给上层 RRF + Rerank
    match_expr = " OR ".join(f'"{t}"' for t in tokens)

    try:
        rows = conn.execute(
            """
            SELECT f.chunk_id, f.content, bm25(fts_chunks_virt) AS rank
            FROM fts_chunks_virt f
            WHERE fts_chunks_virt MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_expr, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        # FTS5 不可用时退化 LIKE（慢但不至于整个检索路挂掉）
        like_pattern = f"%{tokens[0]}%"
        rows = conn.execute(
            "SELECT chunk_id, content, 0 AS rank FROM fts_chunks_virt WHERE content LIKE ? LIMIT ?",
            (like_pattern, top_k),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        # bm25 是负数（越小越好），取反让分数越大越好（与其他路统一）
        raw_score = row["rank"] if row["rank"] is not None else 0
        results.append({
            "chunk_id": row["chunk_id"],
            "content": row["content"],
            "score": -float(raw_score),
        })
    return results


# =================================================================== 同义词 / 停用词
def load_synonyms() -> int:
    """
    从 config/synonyms.yaml 导入 synonyms 表。
    格式 {主词: [同义词...]}，加载后双向展开：A->[B,C] => 写入 (A,B),(A,C),(B,A),(C,A),(B,C),(C,B)...
    重复导入安全（INSERT OR REPLACE）。返回写入条目数。
    """
    init_tables()
    import yaml

    lcfg = get_settings().get("lexicon") or {}
    syn_path = Path(str(lcfg.get("synonyms_path") or "config/synonyms.yaml"))
    if not syn_path.exists():
        return 0

    data = yaml.safe_load(syn_path.read_text(encoding="utf-8")) or {}
    conn = get_connection()
    count = 0
    for main_term, syn_list in data.items():
        if not isinstance(syn_list, list):
            continue
        group = [str(main_term)] + [str(s) for s in syn_list]
        for a in group:
            for b in group:
                if a != b:
                    conn.execute(
                        "INSERT OR REPLACE INTO synonyms(term, synonym, weight) VALUES (?, ?, ?)",
                        (a, b, 1.0),
                    )
                    count += 1
    conn.commit()
    return count


def load_stopwords() -> set[str]:
    """从 config/stopwords_zh.txt 加载停用词集合（n-gram / vocab 构建时过滤用）。"""
    lcfg = get_settings().get("lexicon") or {}
    sw_path = Path(str(lcfg.get("stopwords_path") or "config/stopwords_zh.txt"))
    if not sw_path.exists():
        return set()
    lines = sw_path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}
