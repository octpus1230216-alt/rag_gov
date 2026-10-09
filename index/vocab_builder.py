# -*- coding: utf-8 -*-
"""
词表构建：从全量 chunk 中统计 df/idf，过滤噪声后写入 vocab 表。

用途：
  - 检索路由层判断 query 哪些 token 有区分度（df 太低 = 罕见词 = 高 idf = 强信号）；
  - 模糊检索扩展时排除无意义高频词；
  - sparse 权重调优（与 BGE-M3 的 lexical_weights 互补校验）。

分批提交：
  百万级 chunk 全量分词可能耗时 30+ 分钟，内存也会膨胀（全部 token 计数）。
  采用「每处理 N 条 chunk 就把当前计数 flush 到 DB 再清空」的策略：
  内存占用恒定（单批 chunk 量 × 平均 token 数），代价是多次写入同一词。

命令入口：python -m index.vocab_builder [--chunk-dir chunks/] [--batch 1000]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# 词频过滤参数（需求指定）
MIN_TERM_LEN = 2        # 单字一般无语义区分度（除非是"合同"等双字缩写）
MIN_DF = 2              # 只在 1 个 chunk 出现的词可能是 OCR 错误 / 罕见专有名词，不值得入词表
MAX_DF_RATIO = 0.30     # 出现在 >30% chunk 中的词 = 太通用（"的""是""在"等），对区分文档没帮助


def _load_stopwords() -> set[str]:
    from index.keyword_store import load_stopwords

    return load_stopwords()


def build_vocab(
    chunks: list[dict[str, Any]],
    batch_size: int = 1000,
) -> int:
    """
    从 chunk 列表构建词表并写入 vocab 表。
    支持分批提交：每处理 batch_size 条 chunk 就 commit 一次中间计数。
    返回最终写入的词条目数。

    步骤：
      1. jieba 分词每个 chunk（去停用词、去单字）
      2. 统计 df（出现该词的 chunk 数，同一 chunk 内多次出现只计一次）
      3. 过滤：len(term) >= MIN_TERM_LEN, MIN_DF <= df <= total * MAX_DF_RATIO
      4. 计算 idf = log(total / df)
      5. 写入 vocab 表（INSERT OR REPLACE，重复构建安全）
    """
    import jieba

    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    stopwords = _load_stopwords()

    # ---- 第一遍：统计 df ----
    df_map: dict[str, int] = {}
    total_chunks = 0

    for idx, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "")
        if not content:
            continue
        total_chunks += 1
        # jieba 精确模式分词
        tokens = jieba.lcut(content)
        # 本 chunk 内的不重复词集合（df 只统计出现在多少个 chunk 中）
        seen_in_chunk: set[str] = set()
        for t in tokens:
            t = t.strip()
            if len(t) < MIN_TERM_LEN:
                continue
            if t in stopwords:
                continue
            # 纯数字/纯标点跳过
            if all(c.isdigit() or c.isspace() or not c.isalnum() for c in t):
                continue
            seen_in_chunk.add(t)
        for term in seen_in_chunk:
            df_map[term] = df_map.get(term, 0) + 1

        # 分批提交中间进度（内存占用不降，但防止大事务超时）
        if idx % batch_size == 0:
            conn.commit()

    if total_chunks == 0:
        return 0

    # ---- 第二遍：过滤 + 计算 idf + 写入 ----
    max_df = int(total_chunks * MAX_DF_RATIO)
    if max_df < MIN_DF:
        max_df = MIN_DF  # 极小库（<10 chunk）时放宽上界

    written = 0
    batch_rows: list[tuple[str, int, float]] = []

    for term, df in df_map.items():
        if df < MIN_DF:
            continue
        if df > max_df:
            continue
        idf = math.log(total_chunks / df)
        batch_rows.append((term, df, idf))

        if len(batch_rows) >= batch_size:
            conn.executemany(
                "INSERT OR REPLACE INTO vocab(term, df, idf) VALUES (?, ?, ?)",
                batch_rows,
            )
            written += len(batch_rows)
            batch_rows.clear()
            conn.commit()

    # 剩余尾批
    if batch_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO vocab(term, df, idf) VALUES (?, ?, ?)",
            batch_rows,
        )
        written += len(batch_rows)

    conn.commit()
    return written


def get_vocab() -> list[str]:
    """从 vocab 表读取全部词条（检索路由/查询扩展时使用）。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    rows = conn.execute("SELECT term FROM vocab ORDER BY idf DESC").fetchall()
    return [row["term"] for row in rows]


def get_vocab_stats() -> dict[str, Any]:
    """词表统计信息（调试/监控用）。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS n, AVG(idf) AS avg_idf, AVG(df) AS avg_df FROM vocab"
    ).fetchone()
    return {
        "total_terms": row["n"] or 0,
        "avg_idf": round(float(row["avg_idf"] or 0), 4),
        "avg_df": round(float(row["avg_df"] or 0), 2),
    }


# =================================================================== 从磁盘 chunks/ 构建
def build_from_chunk_dir(chunk_dir: str | Path | None = None, batch_size: int = 1000) -> int:
    """
    从流水线产物目录 chunks/*.jsonl 逐文件流式读取，构建词表。
    不把全量 chunk 一次性 load 进内存（100GB 库可能有亿级 chunk）。
    每个文件单独一个 list（单文档 chunk 通常 <10K 条），处理完释放。
    """
    if chunk_dir is None:
        from ingestion.pipeline import stage_dir

        chunk_dir = stage_dir("chunk")
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.exists():
        print(f"[vocab] chunk 目录不存在: {chunk_dir}")
        return 0

    jsonl_files = sorted(chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("[vocab] 没有找到 chunks/*.jsonl")
        return 0

    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()

    # 先清空旧词表（全量重建语义）
    conn.execute("DELETE FROM vocab")
    conn.commit()

    # 第一遍：统计 df（按文件流式，不全 load）
    import jieba

    stopwords = _load_stopwords()
    df_map: dict[str, int] = {}
    total_chunks = 0

    for fpath in jsonl_files:
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("content", "")
                if not content:
                    continue
                total_chunks += 1
                tokens = jieba.lcut(content)
                seen: set[str] = set()
                for t in tokens:
                    t = t.strip()
                    if len(t) < MIN_TERM_LEN or t in stopwords:
                        continue
                    if all(c.isdigit() or c.isspace() or not c.isalnum() for c in t):
                        continue
                    seen.add(t)
                for term in seen:
                    df_map[term] = df_map.get(term, 0) + 1

        # 每处理一个文件打进度
        print(f"[vocab] 已处理 {fpath.name}，累计 {total_chunks} chunks", flush=True)

    if total_chunks == 0:
        return 0

    # 过滤 + 写入
    max_df = max(int(total_chunks * MAX_DF_RATIO), MIN_DF)
    written = 0
    batch_rows: list[tuple[str, int, float]] = []

    for term, df in df_map.items():
        if df < MIN_DF or df > max_df:
            continue
        idf = math.log(total_chunks / df)
        batch_rows.append((term, df, idf))
        if len(batch_rows) >= batch_size:
            conn.executemany("INSERT OR REPLACE INTO vocab(term, df, idf) VALUES (?, ?, ?)", batch_rows)
            written += len(batch_rows)
            batch_rows.clear()
            conn.commit()

    if batch_rows:
        conn.executemany("INSERT OR REPLACE INTO vocab(term, df, idf) VALUES (?, ?, ?)", batch_rows)
        written += len(batch_rows)

    conn.commit()
    print(f"[vocab] 完成：{total_chunks} chunks -> {written} 个词条")
    return written


# =================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        prog="python -m index.vocab_builder",
        description="从 chunks/*.jsonl 构建词表（df/idf）",
    )
    cli.add_argument("--chunk-dir", default=None, help="chunks/ 目录路径（默认从 settings 推断）")
    cli.add_argument("--batch", type=int, default=1000, help="分批提交大小")
    args = cli.parse_args(argv)

    n = build_from_chunk_dir(args.chunk_dir, args.batch)
    print(f"词表条目数: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
