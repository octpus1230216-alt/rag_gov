# -*- coding: utf-8 -*-
"""第 6 批轻量自检：路由 / context 组装 / qa_set 覆盖 / schema 合法。不依赖模型与服务。"""
import json
from collections import Counter

from retrieval.context_builder import build_context, estimate_tokens
from retrieval.query_router import route

# ---- 路由断言
assert route("第三条 违约责任")["dense"] == 0.3
assert route("\u201c商业秘密\u201d")["fuzzy"] == 0.3  # 中文双引号
assert route("2024-05-01 会议")["exact"] == 1.0
assert route("¥1,000,000")["sparse"] == 0.8
assert route("HT-2024-001")["sparse"] == 1.0
assert route("请假")["fuzzy"] == 1.0  # 短查询
assert route("这个平台到底怎么样才能提高大家的整体协作效率呢？")["dense"] == 1.0  # 疑问句
assert set(route("随便查点东西")) == {"exact", "fuzzy", "sparse", "dense"}
print("route OK")

# ---- context 断言
chunks = [
    {"chunk_id": "a1", "doc_id": "d1", "content": "第一段内容", "section_path": "文档>一", "rerank_score": 0.9, "metadata": {"chunk_index": 0}},
    {"chunk_id": "a2", "doc_id": "d1", "content": "第二段内容", "section_path": "文档>二", "rerank_score": 0.8, "metadata": {"chunk_index": 1}},
]
# 同 doc + chunk_index 相邻(0,1) → 应合并为 1 条 source（去重合并设计意图）
r = build_context(chunks, max_tokens=3000)
assert r["context"] and "[1]" in r["context"] and r["truncated"] is False
assert len(r["sources"]) == 1, f"相邻应合并成1条，实际 {len(r['sources'])}"
assert "第一段内容" in r["sources"][0].get("content", "") or "第一段内容" in r["context"]
print("build_context merge OK, sources=", len(r["sources"]))

# 不相邻(chunk_index 0, 9) → 保留 2 条
chunks2 = [
    {"chunk_id": "b1", "doc_id": "d2", "content": "甲", "section_path": "x>a", "rerank_score": 0.9, "metadata": {"chunk_index": 0}},
    {"chunk_id": "b2", "doc_id": "d2", "content": "乙", "section_path": "y>b", "rerank_score": 0.8, "metadata": {"chunk_index": 9}},
]
r2 = build_context(chunks2, max_tokens=3000)
assert len(r2["sources"]) == 2 and r2["sources"][0]["chunk_id"]
print("build_context non-adjacent OK, sources=", len(r2["sources"]))
assert estimate_tokens("一二三四五") == 3  # 5 * 0.6 = 3

# ---- 截断断言
big = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "content": "字" * 2000, "section_path": f"s{i}", "rerank_score": 1.0 - i * 0.1} for i in range(5)]
rt = build_context(big, max_tokens=100)
assert rt["truncated"] is True
print("truncated OK, kept=", len(rt["sources"]))

# ---- qa_set 类型覆盖
with open("eval/qa_set.jsonl", encoding="utf-8") as fh:
    items = [json.loads(line) for line in fh if line.strip()]
c = Counter(i["type"] for i in items)
assert c["semantic"] == 5 and c["exact"] == 4 and c["fuzzy"] == 4 and c["table"] == 4 and c["image"] == 3, dict(c)
assert len(items) == 20
assert all(i.get("expected_keywords") or i.get("expected_chunk_ids") for i in items)
print("qa_set OK", dict(c))

# ---- schema 合法
for f in ["schemas/chunk.schema.json", "schemas/mcp_tools.schema.json"]:
    with open(f, encoding="utf-8") as fh:
        json.load(fh)
print("schema JSON OK")

# ---- tools 注册（不启动服务，只验证 FastMCP 能挂上 6 个工具）
try:
    from fastmcp import FastMCP

    from mcp_server.tools import register_tools

    mcp = FastMCP(name="rag-zh-test")
    register_tools(mcp)
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool_names = set(tool_manager._tools.keys()) if tool_manager else set()
    if not tool_names:  # 兼容不同 FastMCP 版本内部结构
        import asyncio

        tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {"search", "ingest", "get_chunk", "list_documents", "get_document_url", "classify_image"}
    assert expected <= tool_names, f"缺工具: {expected - tool_names}"
    print("mcp tools OK", sorted(tool_names))
except ImportError as exc:
    print("mcp tools SKIP (fastmcp 未安装):", exc)

print("ALL BATCH6 CHECKS PASSED")
