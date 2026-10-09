# -*- coding: utf-8 -*-
"""
上下文组装：把 rerank 后的 chunks 拼成给 LLM 的 context 字符串。

核心逻辑：
  1. 去重合并：同 doc 的相邻 chunk（序号连续）合并为一段，减少重复；
  2. 按 section_path 排序：保持文档原文顺序，LLM 更容易理解上下文；
  3. 超长截断：按 token 预算逐条加入，超出时优先保留高分 chunk；
  4. 引用标记：每段前加 [1] [2] ...，方便 LLM 引用和溯源。

中文 token 估算：len(text) * 0.6（经验值，中文字符平均约 0.6 token）。
"""

from __future__ import annotations

from typing import Any

# =================================================================== 常量
DEFAULT_MAX_TOKENS = 3000
CHUNK_SEP = "\n\n"


def estimate_tokens(text: str) -> int:
    """
    中文 token 粗估接口（需求指定）。
    规则：len(text) * 0.6（中文字符 + 标点的经验换算系数）。
    对纯英文场景偏差较大，但本系统主要服务中文知识库。
    """
    return max(1, int(len(text) * 0.6))


# =================================================================== 去重合并
def _merge_adjacent(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    同一 doc_id 且 section_path 相邻的 chunks 合并内容。
    "相邻"判定：doc_id 相同 + 按 chunk_index（或 metadata 中的序号）连续。
    合并后 score 取两者最大值，content 拼接。
    """
    if not chunks:
        return []

    # 按 doc_id 分组
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        did = c.get("doc_id", "")
        by_doc.setdefault(did, []).append(c)

    merged: list[dict[str, Any]] = []
    for _doc_id, group in by_doc.items():
        # 按 section_path 排序（同文档内按出现顺序）
        group.sort(key=lambda x: x.get("section_path", ""))
        current = dict(group[0])
        for nxt in group[1:]:
            # 判断是否"相邻"：section_path 前缀相同或 metadata 里 chunk_index 连续
            if _is_adjacent(current, nxt):
                current["content"] = current.get("content", "") + "\n" + nxt.get("content", "")
                current["rerank_score"] = max(
                    current.get("rerank_score", 0), nxt.get("rerank_score", 0)
                )
                current["_merged_count"] = current.get("_merged_count", 1) + 1
            else:
                merged.append(current)
                current = dict(nxt)
        merged.append(current)

    # 重新按 rerank_score 降序（高分优先保留）
    merged.sort(key=lambda x: x.get("rerank_score", x.get("rrf_score", 0)), reverse=True)
    return merged


def _is_adjacent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """判断两个 chunk 是否相邻（同 doc + section_path 连续或 chunk_index 连续）。"""
    # 优先用 metadata 里的 chunk_index
    meta_a = a.get("metadata") or {}
    meta_b = b.get("metadata") or {}
    idx_a = meta_a.get("chunk_index")
    idx_b = meta_b.get("chunk_index")
    if idx_a is not None and idx_b is not None:
        try:
            return abs(int(idx_b) - int(idx_a)) <= 1
        except (ValueError, TypeError):
            pass
    # 退化：section_path 前 2 级相同视为相邻
    sp_a = (a.get("section_path") or "").split(">")[:2]
    sp_b = (b.get("section_path") or "").split(">")[:2]
    return sp_a == sp_b and sp_a != [""]


# =================================================================== 主入口
def build_context(
    reranked_chunks: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """
    组装 LLM 上下文。

    参数：
      reranked_chunks: reranker 输出的 top_k chunks，每个至少含
        {"chunk_id", "doc_id", "content", "section_path", "rerank_score", "metadata"}
      max_tokens: 上下文 token 预算上限（中文按 len*0.6 估算）

    返回：
      {
        "context": str,       # 拼接文本，带 [1] [2] 引用标记
        "sources": [          # 引用来源列表（与标记编号一一对应）
          {"chunk_id", "doc_id", "section_path", "score", "bbox"}
        ],
        "truncated": bool,    # 是否因超长被截断
      }
    """
    if not reranked_chunks:
        return {"context": "", "sources": [], "truncated": False}

    # 1) 去重合并
    merged = _merge_adjacent(reranked_chunks)

    # 2) 按 section_path 排序（保持原文顺序），同时高分优先
    #    先按 score 降序排，截断后再按 section_path 恢复原文顺序
    merged.sort(key=lambda x: x.get("rerank_score", x.get("rrf_score", 0)), reverse=True)

    # 3) 逐条加入直到超出 token 预算
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    truncated = False

    for chunk in merged:
        content = chunk.get("content", "")
        # 加上引用标记 "[N] " 的额外 token
        chunk_tokens = estimate_tokens(content) + 4
        if used_tokens + chunk_tokens > max_tokens and selected:
            truncated = True
            break
        selected.append(chunk)
        used_tokens += chunk_tokens

    # 4) 按 section_path 排序恢复原文顺序（高分但位置靠后的 chunk 不会插到前面）
    selected.sort(key=lambda x: x.get("section_path", ""))

    # 5) 拼接最终文本 + sources
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for i, chunk in enumerate(selected, start=1):
        marker = f"[{i}]"
        content = chunk.get("content", "").strip()
        section = chunk.get("section_path", "")
        # 带 section 信息的引用头
        header = f"{marker} 【{section}】" if section else marker
        parts.append(f"{header}\n{content}")

        # 提取 bbox（若有）
        meta = chunk.get("metadata") or {}
        bbox = meta.get("bbox") or chunk.get("bbox")

        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "section_path": section,
            "score": chunk.get("rerank_score", chunk.get("rrf_score", 0)),
            "bbox": bbox,
        })

    context_text = CHUNK_SEP.join(parts)

    return {
        "context": context_text,
        "sources": sources,
        "truncated": truncated,
    }
