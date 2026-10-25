# -*- coding: utf-8 -*-
"""切分器离线自检（由 _check_batch3.py 迁移，仅纯逻辑部分）。

token 估算、配置解析、正文切分（跨页合并 / overlap 整句 / chunk_id 稳定 /
无内容丢失）、表格切分（小表整块 / 大表行组分片带表头 / 非结构化原样保留）。
流水线相关断言见 test_pipeline_stages.py / test_pipeline_docid.py。"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

from ingestion.chunkers import layout_chunker as lc
from ingestion.chunkers.layout_chunker import chunk_text_blocks, estimate_tokens, resolve_config
from ingestion.chunkers.table_chunker import chunk_table, is_markdown_table, row_to_kv
from tests.helpers import sample_docs


# ============================================================ token 估算
def test_token_estimator():
    assert 50 <= lc.estimate_tokens_zh("中" * 100) <= 70
    # 纯英文不被 ×0.6 高估
    assert lc.estimate_tokens_zh("internationalization organization") < 22 * 0.6 + 2
    assert lc.estimate_tokens_zh("   ") == 0

    func, backend = lc.choose_token_estimator("char")
    assert func is lc.estimate_tokens_zh and backend == "char_heuristic"
    try:
        lc.choose_token_estimator("jieba")
        jieba_ok = True
    except Exception:
        jieba_ok = False
    _auto_func, auto_backend = lc.choose_token_estimator(None)
    # auto 缺 jieba 时退回字符口径
    assert auto_backend == ("jieba" if jieba_ok else "char_heuristic")

    named = lc.set_token_estimator(lambda t: len(t))
    assert lc.estimate_tokens("四个字") == 3 and named == "<lambda>"
    lc.set_token_estimator(prefer="char")
    assert lc.token_estimator_name() == "char_heuristic"


def test_config_resolve():
    cfg = resolve_config(None)
    assert cfg.chunk_size == 384
    assert cfg.chunk_overlap == 64
    assert cfg.max_table_rows in (30, 8)
    assert len(set(cfg.separators)) == len(cfg.separators)  # separators 去重保序
    assert resolve_config({"chunk_size": 128}).chunk_size == 128  # 显式覆盖生效


# ============================================================ 正文切分
def test_text_chunking():
    blocks, metadata = sample_docs.synth_blocks(40)
    cfg = resolve_config(None)
    chunks = chunk_text_blocks(blocks, metadata, cfg, doc_id="d" * 64)
    texts = [c for c in chunks if c["chunk_type"] == "text"]

    assert len(texts) >= 3  # 长章节被分窗口
    assert all(estimate_tokens(c["content"]) <= cfg.chunk_size * 1.35 for c in texts)
    # 标题不单独成 chunk
    assert not any(c["content"].strip() in ("第3章 付款方式", "3.2 表格") for c in texts)
    # 表格块不走正文切分
    assert all("项目" not in c["content"] or "|" not in c["content"] for c in texts)
    assert texts[0]["section_path"] == "第3章 付款方式"

    # 跨页同 section 合并
    sec3 = [c for c in texts if c["section_path"] == "第3章 付款方式"]
    covered: set[int] = set()
    for c in sec3:
        covered.update(c.get("pages") or [])
    assert any(len(c.get("pages") or []) > 1 for c in sec3)
    assert covered >= {1, 2, 3}
    assert any(c["section_path"] == "第4章 结算" for c in texts)  # 不同 section 不混

    # 无内容丢失
    sentences = [b["content"] for b in blocks if b["type"] == "text"]
    lost = [s for s in sentences if not any(s in c["content"] for c in texts)]
    assert not lost, f"{len(lost)} 句丢失: {lost[:1]}"

    # overlap 必须是整句
    bad: list[str] = []
    overlapped = 0
    for section in dict.fromkeys(c["section_path"] for c in texts):
        group = [c for c in texts if c["section_path"] == section]
        for prev, nxt in pairwise(group):
            head = nxt["content"].split("。")[0]
            if nxt["content"][:12] in prev["content"]:
                overlapped += 1
            if len(head) > 1 and head not in prev["content"]:
                bad.append(head[:20])
    assert not bad
    assert overlapped > 0  # 相邻片确实带重叠内容（不是硬拼接口）

    # chunk_id 稳定
    again = chunk_text_blocks(blocks, metadata, cfg, doc_id="d" * 64)
    assert [c["chunk_id"] for c in again] == [c["chunk_id"] for c in texts]
    assert all(re.fullmatch(r"[0-9a-f]{64}", c["chunk_id"]) for c in texts)
    assert len({c["chunk_id"] for c in texts}) == len(texts)
    assert [c["chunk_index"] for c in texts] == list(range(len(texts)))
    assert {c["chunk_id"] for c in chunk_text_blocks(blocks, metadata, cfg, doc_id="e" * 64)} != {
        c["chunk_id"] for c in texts
    }


# ============================================================ 表格切分
def test_table_chunking_small():
    md_small = (
        "| 项目 | 金额（元） | 备注 |\n|---|---|---|\n"
        "| 预付款 | 30,000.00 | 签约后七日内支付 |\n| 尾款 | 70,000.50 | 验收合格后支付 |\n"
    )
    block = {
        "type": "table",
        "content": md_small,
        "page": 3,
        "section_path": "第3章 > 3.2",
        "header": ["项目", "金额（元）", "备注"],
    }
    out = chunk_table(block, {"doc_type": "contract"}, None, doc_id="f" * 64)
    assert len(out) == 1  # 小表整表一个 chunk
    assert "| 预付款 |" in out[0]["content"] and "---" in out[0]["content"]
    assert out[0]["section_path"] == "第3章 > 3.2 > 表格1"
    assert out[0]["columns"] == ["项目", "金额（元）", "备注"]
    assert out[0]["row_lines"][0] == "项目: 预付款 | 金额（元）: 30,000.00 | 备注: 签约后七日内支付"
    assert is_markdown_table(md_small) and not is_markdown_table("纯文本")
    assert row_to_kv(["a", "", "c"], ["X", "Y", "Z"]) == "X: a | Z: c"


def test_table_chunking_big():
    rows = [
        f"| {i} | 设备采购明细第{i}项 | {12000 + i * 37} | 含三年质保与现场安装培训服务费用已含在单价内 |"
        for i in range(1, 61)
    ]
    header_lines = ["| 序号 | 名称 | 单价 | 备注 |", "|---|---|---|---|"]
    big = "\n".join([*header_lines, *rows]) + "\n"
    assert len(big) > 2000
    big_block = {
        "type": "table",
        "content": big,
        "page": 5,
        "section_path": "附表",
        "header": ["序号", "名称", "单价", "备注"],
    }
    parts = chunk_table(big_block, {}, None, doc_id="g" * 64)
    assert len(parts) > 1  # 大表按行组切成多片
    assert all(p["content"].splitlines()[0] == "| 序号 | 名称 | 单价 | 备注 |" for p in parts)
    assert all(
        all(ln.startswith("序号: ") and "名称: " in ln for ln in p["row_lines"])
        for p in parts
        if p.get("row_lines")
    )
    assert all(
        p["part"] == i + 1 and p["n_parts"] == len(parts) and p["row_range"]
        for i, p in enumerate(parts)
    )
    rebuilt = [ln for p in parts for ln in p["content"].splitlines() if ln.startswith("|")]
    data_lines = [ln for ln in rebuilt if "序号" not in ln and not set(ln) <= set("|-: ")]
    uniq = list(dict.fromkeys(data_lines))
    assert len(uniq) == len(rows)  # 数据行不丢
    assert len(data_lines) == len(uniq)  # 数据行不重
    assert [int(p["row_range"].split("-")[0]) for p in parts] == sorted(
        int(p["row_range"].split("-")[0]) for p in parts
    )
    assert all(len(p["content"]) <= 2200 for p in parts)


def test_table_chunking_edge():
    # 非 markdown 表格原样保留，绝不静默丢内容
    loose = {
        "type": "table",
        "content": "表1：2024年度费用汇总（图片式表格无法结构化）",
        "page": 1,
    }
    out2 = chunk_table(loose, {}, None, doc_id="h" * 64)
    assert len(out2) == 1 and out2[0]["not_structured"] is True
    assert chunk_table({"type": "table", "content": "   "}, {}) == []

    # sheet 复用同一切分器
    md_small = (
        "| 项目 | 金额（元） | 备注 |\n|---|---|---|\n"
        "| 预付款 | 30,000.00 | 签约后七日内支付 |\n| 尾款 | 70,000.50 | 验收合格后支付 |\n"
    )
    sheet_block: dict[str, Any] = {
        "type": "sheet",
        "content": md_small,
        "sheet_name": "费用",
        "region_index": 2,
    }
    out3 = chunk_table(sheet_block, {}, None, chunk_type="sheet", table_index=2, doc_id="i" * 64)
    assert out3[0]["chunk_type"] == "sheet" and out3[0]["sheet_name"] == "费用"
