# -*- coding: utf-8 -*-
"""上下文组装离线自检（由 _check_batch6.py 迁移）：相邻合并 / 非相邻保留 / 截断 / token 估算 / [N] 引用头。"""

from __future__ import annotations

from retrieval.context_builder import build_context, estimate_tokens


def test_estimate_tokens():
    assert estimate_tokens("一二三四五") == 3  # 5 * 0.6 = 3


def test_adjacent_merged():
    chunks = [
        {
            "chunk_id": "a1",
            "doc_id": "d1",
            "content": "第一段内容",
            "section_path": "文档>一",
            "rerank_score": 0.9,
            "metadata": {"chunk_index": 0},
        },
        {
            "chunk_id": "a2",
            "doc_id": "d1",
            "content": "第二段内容",
            "section_path": "文档>二",
            "rerank_score": 0.8,
            "metadata": {"chunk_index": 1},
        },
    ]
    r = build_context(chunks, max_tokens=3000)
    assert r["context"] and "[1]" in r["context"] and r["truncated"] is False
    # 同 doc + chunk_index 相邻(0,1) → 合并为 1 条 source
    assert len(r["sources"]) == 1, f"相邻应合并成1条，实际 {len(r['sources'])}"
    assert "第一段内容" in r["context"]


def test_non_adjacent_kept():
    chunks = [
        {
            "chunk_id": "b1",
            "doc_id": "d2",
            "content": "甲",
            "section_path": "x>a",
            "rerank_score": 0.9,
            "metadata": {"chunk_index": 0},
        },
        {
            "chunk_id": "b2",
            "doc_id": "d2",
            "content": "乙",
            "section_path": "y>b",
            "rerank_score": 0.8,
            "metadata": {"chunk_index": 9},
        },
    ]
    r = build_context(chunks, max_tokens=3000)
    assert len(r["sources"]) == 2 and r["sources"][0]["chunk_id"]


def test_truncated_flag():
    big = [
        {
            "chunk_id": f"c{i}",
            "doc_id": f"d{i}",
            "content": "字" * 2000,
            "section_path": f"s{i}",
            "rerank_score": 1.0 - i * 0.1,
        }
        for i in range(5)
    ]
    r = build_context(big, max_tokens=100)
    assert r["truncated"] is True


def test_reference_header_marker():
    chunks = [
        {
            "chunk_id": "a1",
            "doc_id": "d1",
            "content": "内容",
            "section_path": "文档>一",
            "rerank_score": 0.9,
            "metadata": {"chunk_index": 0},
        }
    ]
    r = build_context(chunks, max_tokens=3000)
    assert r["context"].startswith("[1] 【文档>一】")


def test_empty_input():
    r = build_context([], max_tokens=3000)
    assert r == {"context": "", "sources": [], "truncated": False}
