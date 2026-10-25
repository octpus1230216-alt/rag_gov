# -*- coding: utf-8 -*-
"""评测集 / schema / MCP 工具注册离线自检（由 _check_batch6.py 迁移）。"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest


# ============================================================ qa_set 类型覆盖
def test_qa_set_coverage(repo_root: Path):
    with open(repo_root / "eval" / "qa_set.jsonl", encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    c = Counter(i["type"] for i in items)
    assert c["semantic"] == 5
    assert c["exact"] == 4
    assert c["fuzzy"] == 4
    assert c["table"] == 4
    assert c["image"] == 3
    assert len(items) == 20
    # 每条都要有可判定的期望（关键词或 chunk id）
    assert all(i.get("expected_keywords") or i.get("expected_chunk_ids") for i in items)


# ============================================================ schema 合法
@pytest.mark.parametrize(
    "schema",
    ["schemas/chunk.schema.json", "schemas/mcp_tools.schema.json"],
)
def test_schema_is_valid_json(repo_root: Path, schema: str):
    with open(repo_root / schema, encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)


# ============================================================ MCP 工具注册（缺 fastmcp 则跳过）
def test_mcp_registers_seven_tools(repo_root: Path):
    pytest.importorskip("fastmcp", reason="未安装 fastmcp，跳过工具注册检查")
    from fastmcp import FastMCP

    from mcp_server.tools import register_tools

    mcp = FastMCP(name="rag-zh-test")
    register_tools(mcp)
    tool_manager = getattr(mcp, "_tool_manager", None)
    tool_names = set(tool_manager._tools.keys()) if tool_manager else set()
    if not tool_names:  # 兼容不同 FastMCP 版本内部结构
        tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "search",
        "ask",  # 本批生成层新增的第 7 个工具
        "ingest",
        "get_chunk",
        "list_documents",
        "get_document_url",
        "classify_image",
    }
    assert expected <= tool_names, f"缺工具: {expected - tool_names}"
