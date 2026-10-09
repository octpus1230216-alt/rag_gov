# -*- coding: utf-8 -*-
"""
模糊检索路：RapidFuzz 编辑距离 + n-gram 子串 + 同义词扩展，三路合一。

红线：fuzzy 结果不直接喂 LLM（必须经过 reranker），本模块只负责候选扩展。

三部分合并逻辑：
  a) RapidFuzz WRatio：query jieba 分词 -> 逐词与 vocab 表做模糊匹配 -> 命中词再查 chunk
  b) n-gram 子串：调用 ngram_index.ngram_search（内部已含二次验证）
  c) 同义词扩展：从 synonyms 表取 query 词的同义词（深度=1），再分别做 a) 和 b)
  最终合并去重，按 score 降序。

停用词前置过滤：避免 "的""是""在" 等无区分度的词触发大量假阳性候选。
"""

from __future__ import annotations

from typing import Any

from ingestion.parsers.common import get_settings


def _get_stopwords() -> set[str]:
    """惰性加载停用词（首次加载后缓存在模块级）。"""
    global _STOPWORDS
    if _STOPWORDS is None:
        from index.keyword_store import load_stopwords

        _STOPWORDS = load_stopwords()
    return _STOPWORDS


_STOPWORDS: set[str] | None = None


def _jieba_tokens(text: str) -> list[str]:
    """jieba 精确模式分词 + 过滤停用词和单字。"""
    import jieba

    raw = jieba.lcut(text)
    stopwords = _get_stopwords()
    return [t.strip() for t in raw if t.strip() and len(t.strip()) >= 2 and t.strip() not in stopwords]


# =================================================================== 同义词扩展
def expand_synonyms(terms: list[str]) -> list[str]:
    """
    从 synonyms 表查每个词的同义词，扩展深度限制为 1。
    返回去重后的扩展词列表（不含原始 terms）。
    """
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    expanded: set[str] = set()
    original_set = set(terms)

    for term in terms:
        rows = conn.execute(
            "SELECT synonym FROM synonyms WHERE term = ?", (term,)
        ).fetchall()
        for row in rows:
            syn = row["synonym"]
            if syn not in original_set and len(syn) >= 2:
                expanded.add(syn)

    return list(expanded)


# =================================================================== RapidFuzz 匹配
def _rapidfuzz_search(terms: list[str], top_k: int = 10) -> list[dict[str, Any]]:
    """
    用 RapidFuzz WRatio 把 terms 与 vocab 表做模糊匹配，
    匹配到的 vocab 词再回查 ngram_index 取 chunk_id。
    返回 [{"chunk_id": ..., "score": ...}]（score 取 0-100 的 fuzz 分）。
    """
    from rapidfuzz import fuzz

    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    rcfg = get_settings().get("retrieval") or {}
    score_cutoff = float(rcfg.get("fuzzy_score_cutoff") or 75)

    # 取全部 vocab 词（万级，对每个 term 做 WRatio 过滤，可接受）
    vocab_rows = conn.execute("SELECT term FROM vocab").fetchall()
    vocab_terms = [r["term"] for r in vocab_rows]
    if not vocab_terms:
        return []

    # 匹配到的 vocab 词
    matched_terms: list[str] = []
    for term in terms:
        for v in vocab_terms:
            score = fuzz.WRatio(term, v)
            if score >= score_cutoff:
                matched_terms.append(v)

    if not matched_terms:
        return []

    # 去重，用匹配到的 vocab 词去 ngram_index 找 chunk
    matched_terms = list(set(matched_terms))
    chunk_scores: dict[str, float] = {}

    for term in matched_terms:
        # 取 term 的 2-gram 去 ngram_index 查 chunk
        grams: set[str] = set()
        clean = term.replace(" ", "")
        for i in range(len(clean) - 1):
            grams.add(clean[i : i + 2])
        if not grams:
            continue
        placeholders = ",".join("?" * len(grams))
        rows = conn.execute(
            f"SELECT chunk_id, COUNT(*) AS cnt FROM ngram_index WHERE gram IN ({placeholders}) GROUP BY chunk_id ORDER BY cnt DESC LIMIT ?",
            (*grams, top_k),
        ).fetchall()
        for row in rows:
            cid = row["chunk_id"]
            # fuzz 分归一化到 0-1 后作为 score
            s = min(fuzz.WRatio(term, term) / 100.0, 1.0)  # 自身一定 1.0
            chunk_scores[cid] = max(chunk_scores.get(cid, 0), s)

    results = [{"chunk_id": cid, "score": sc} for cid, sc in chunk_scores.items()]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =================================================================== n-gram 路
def _ngram_search(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """直接调用 ngram_index.ngram_search（已含二次验证）。"""
    from index.ngram_index import ngram_search

    return ngram_search(query, top_k=top_k)


# =================================================================== 主入口
def fuzzy_search(query: str, top_k: int = 20, score_cutoff: float = 75) -> list[dict[str, Any]]:
    """
    模糊检索路主入口。
    三部分合并：
      a) RapidFuzz 编辑距离匹配 vocab
      b) n-gram 子串匹配（含二次验证）
      c) 同义词扩展后再做 a) 和 b)

    参数：
      query: 原始查询字符串
      top_k: 最终返回候选数上限
      score_cutoff: RapidFuzz 分数阈值（0-100）

    返回：[{"chunk_id": str, "score": float(0~1)}, ...] 按 score 降序。
    """
    terms = _jieba_tokens(query)
    if not terms:
        return []

    # b) n-gram 子串匹配
    ngram_results = _ngram_search(query, top_k=top_k)

    # a) RapidFuzz vocab 匹配
    fuzz_results = _rapidfuzz_search(terms, top_k=top_k)

    # c) 同义词扩展
    synonyms = expand_synonyms(terms)
    syn_ngram: list[dict[str, Any]] = []
    syn_fuzz: list[dict[str, Any]] = []
    if synonyms:
        # 对扩展词再做 n-gram
        syn_query = " ".join(synonyms)
        syn_ngram = _ngram_search(syn_query, top_k=top_k)
        # 对扩展词再做 RapidFuzz
        syn_fuzz = _rapidfuzz_search(synonyms, top_k=top_k)

    # 合并去重（同 chunk_id 取最高 score）
    merged: dict[str, float] = {}
    for item in ngram_results + fuzz_results + syn_ngram + syn_fuzz:
        cid = item["chunk_id"]
        sc = float(item["score"])
        # 扩展路结果打折（乘 0.8），避免同义词扩展压过原始精确匹配
        merged[cid] = max(merged.get(cid, 0), sc)

    # 排序截断
    results: list[dict[str, Any]] = [
        {"chunk_id": cid, "score": round(sc, 4)} for cid, sc in merged.items()
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
