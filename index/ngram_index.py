# -*- coding: utf-8 -*-
"""
n-gram 倒排索引：模糊检索路（fuzzy）的候选扩展 + 二次验证。

原理：
  中文以 2-gram 和 3-gram 为主要粒度（单字信息量不足，4-gram 以上太稀疏）。
  建库时：每个 chunk 去停用词去空格后取全部 n-gram 写入 ngram_index 表。
  查询时：query 做同样的 n-gram，取交集得到候选 chunk_id。

**红线：n-gram 交集是假阳性来源，二次验证必须做。**
  "付款" 的 2-gram {"付款"} 会命中包含 "付款" 子串的所有 chunk，
  但也可能命中 "付了款" / "代付款项" 等需要进一步判断的条目。
  二次验证 = 取 chunk 原文确认 query 的核心 token 确实出现，剔除假阳性。

与 keyword_store 共享同一 sqlite 连接（exact_db_path），
ngram_index 表由 keyword_store.init_tables() 创建，这里只负责写入和查询。
"""

from __future__ import annotations

from typing import Any

# 默认 n-gram 粒度
DEFAULT_N_MIN = 2
DEFAULT_N_MAX = 3

# 二次验证时 query 核心 token 的最小覆盖比例（低于此比例认为假阳性）
_VERIFY_THRESHOLD = 0.5


def _load_stopwords() -> set[str]:
    """从 keyword_store 获取停用词集合（统一来源，避免重复 IO）。"""
    from index.keyword_store import load_stopwords

    return load_stopwords()


def make_ngrams(text: str, n_min: int = DEFAULT_N_MIN, n_max: int = DEFAULT_N_MAX) -> set[str]:
    """
    生成 n-gram 集合。
    步骤：去空格 -> 去停用词（整词匹配移除）-> 滑窗取 n-gram。
    去空格是因为 "付 款" 和 "付款" 语义相同，但带空格的 2-gram 会漏匹配。
    """
    if not text:
        return set()

    stopwords = _load_stopwords()
    # 去空格
    cleaned = text.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    # 去停用词（逐词扫描，只移除完整匹配）
    if stopwords:
        for sw in sorted(stopwords, key=len, reverse=True):
            if sw and len(sw) >= 2:
                cleaned = cleaned.replace(sw, "")

    grams: set[str] = set()
    for n in range(n_min, n_max + 1):
        for i in range(len(cleaned) - n + 1):
            g = cleaned[i : i + n]
            if g:
                grams.add(g)
    return grams


# =================================================================== 建索引
def build_ngram_index(chunks: list[dict[str, Any]], batch_commit: int = 500) -> int:
    """
    批量写入 ngram_index 表。
    chunks: [{"chunk_id": ..., "content": ...}, ...]
    batch_commit: 每处理多少条 chunk 提交一次事务（百万 chunk 一次性提交内存炸掉）。
    返回写入的 (gram, chunk_id) 总行数。
    """
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    total_rows = 0

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", "")
        content = chunk.get("content", "")
        if not chunk_id or not content:
            continue
        grams = make_ngrams(content)
        if not grams:
            continue
        # 批量插入（INSERT OR IGNORE：同一 gram+chunk_id 重复写入不报错）
        conn.executemany(
            "INSERT OR IGNORE INTO ngram_index(gram, chunk_id) VALUES (?, ?)",
            [(g, chunk_id) for g in grams],
        )
        total_rows += len(grams)
        if idx % batch_commit == 0:
            conn.commit()

    conn.commit()
    return total_rows


# =================================================================== 查询
def ngram_search(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    n-gram 倒排检索 + 二次验证。
    步骤：
      1. query -> n-grams
      2. 查 ngram_index 取候选 chunk_id（按命中 gram 数降序）
      3. 二次验证：从 chunks 表取原文，确认 query 的核心字符/词确实出现
      4. 返回 [{chunk_id, score}]，score = 命中 gram 数 / query 总 gram 数（0~1 的覆盖度）

    **不能省二次验证**：n-gram 交集只保证"有共同片段"，不保证语义相关；
    例如 "合同期限" 和 "期限合同" 的 2-gram 集合几乎相同，但可能完全不同文档。
    """
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()

    query_grams = make_ngrams(query)
    if not query_grams:
        return []

    gram_list = list(query_grams)
    placeholders = ",".join("?" * len(gram_list))

    # 取候选（按命中 gram 数降序，扩展 3 倍候选量给二次验证留出过滤余量）
    fetch_limit = top_k * 3
    rows = conn.execute(
        f"""
        SELECT chunk_id, COUNT(*) AS hit_count
        FROM ngram_index
        WHERE gram IN ({placeholders})
        GROUP BY chunk_id
        ORDER BY hit_count DESC
        LIMIT ?
        """,
        (*gram_list, fetch_limit),
    ).fetchall()

    if not rows:
        return []

    # 二次验证：取 chunk 原文，检查 query 去空格后是否包含在 content 去空格后
    # 这是最严格的验证（子串匹配），覆盖率阈值用于放宽部分匹配的情况
    query_clean = query.replace(" ", "").replace("\t", "").replace("\n", "")
    n_total = len(gram_list)

    results: list[dict[str, Any]] = []
    for row in rows:
        chunk_id = row["chunk_id"]
        hit_count = row["hit_count"]
        coverage = hit_count / n_total if n_total > 0 else 0

        # 查原文做二次验证
        chunk_row = conn.execute(
            "SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if chunk_row is None:
            continue

        content_clean = (chunk_row["content"] or "").replace(" ", "").replace("\t", "").replace("\n", "")

        # 验证策略：query 完整子串出现 OR coverage 达到阈值
        is_valid = False
        if query_clean and query_clean in content_clean:
            is_valid = True
        elif coverage >= _VERIFY_THRESHOLD:
            # 逐 gram 验证（至少半数 gram 在 content 中出现）
            gram_hits = sum(1 for g in gram_list if g in content_clean)
            if gram_hits / n_total >= _VERIFY_THRESHOLD:
                is_valid = True

        if is_valid:
            results.append({"chunk_id": chunk_id, "score": round(coverage, 4)})

        if len(results) >= top_k:
            break

    return results


# =================================================================== CLI 调试入口
if __name__ == "__main__":
    import argparse
    import json

    cli = argparse.ArgumentParser(description="n-gram 索引调试工具")
    cli.add_argument("command", choices=["build", "search", "grams"])
    cli.add_argument("--input", help="search 时传入查询文本 / build 时传入 chunks jsonl 路径")
    cli.add_argument("--top-k", type=int, default=20)
    args = cli.parse_args()

    if args.command == "grams":
        text = args.input or ""
        grams = make_ngrams(text)
        print(json.dumps(sorted(grams), ensure_ascii=False, indent=2))
    elif args.command == "search":
        results = ngram_search(args.input or "", args.top_k)
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
    elif args.command == "build":
        from pathlib import Path as P

        path = P(args.input or "")
        chunks_data: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunks_data.append(json.loads(line))
        n = build_ngram_index(chunks_data)
        print(f"写入 {n} 条 ngram 索引")
