# -*- coding: utf-8 -*-
"""
表格切分：type=table / sheet 的块专用，**表格永远不当普通文本切**。

为什么要单独一个切分器：
  中文表格丢失列名后，「12,345.67」这种数值 token 在 dense/sparse 两路里
  都跟正文里的任何数字都长得一样 —— 检索回来既不知道是哪一列，
  也无法做精确匹配。所以本切分器始终保证：**每个 chunk 自带表头**。

三种产出形态（同一份内容，三种用法）：
  content     原始 markdown 表格（结构完整，展示给 LLM 时最省 token）
  row_lines   逐行「列名: 值 | 列名: 值」展开（给 FTS5 精确路用：列名重复出现在
              每一行，「毛利率 25.4%」这类查询才可能精确命中）
  columns     列名数组（给结构化过滤 / 后续做表格问答用）
两个字段都保留，检索批次按通道选用，不在这里做取舍。

超长表按行组分片：同时受 max_table_chars（需求口径 2000 字符）与
max_table_rows（settings.chunking.max_table_rows_per_chunk）约束，
每片都重新拼上全部表头行 —— 丢表头的分片等同于噪声。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ingestion.chunkers.layout_chunker import (
    ChunkConfig,
    estimate_tokens,
    new_chunk,
    resolve_config,
    token_estimator_name,
)
from ingestion.parsers.common import normalize_text, pick_header_row, table_to_markdown

# markdown 表格行：必须以竖线开头并以竖线结尾（允许行首缩进）
_MD_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# 分隔行：|---|:--:|--- 之类，对齐冒号与长度都可能出现
_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

# 列名最多取几级表头（三级以上多级表头极罕见，再多就把数据行当列名了）
MAX_HEADER_LEVELS = 3
# 多级表头各层级之间的连接符（"收入 / 营业收入" 这种可被人直接读懂）
LEVEL_JOIN = " / "
# 行内「列名: 值」的分隔，与 markdown 表格竖线一致，便于肉眼核对
KV_JOIN = " | "
KV_SEP = ": "


def is_markdown_table(text: str) -> bool:
    """内容是否是 markdown 表格（必须同时有表头行和 |---| 分隔行）。"""
    header, sep, _body = parse_markdown_table(text)
    return bool(header and sep)


def split_md_row(line: str) -> list[str]:
    """
    一行 markdown -> 单元格列表。
    先在转义竖线 \\| 处占位，切完再还原，否则「备注: a\\|b」会被切成两列，
    整行列错位（这是中文表格里最常见的撕裂原因：单元格里写了管道符）。
    """
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    guarded = raw.replace("\\|", "\x00")
    return [normalize_text(cell.replace("\x00", "|")) for cell in guarded.split("|")]


def parse_markdown_table(text: str) -> tuple[list[list[str]], bool, list[list[str]]]:
    """
    markdown -> (表头行, 是否有分隔行, 数据行)。
    连续的表格行才算一张表；中间夹的普通文本会被忽略（parser 产出的 table block
    理论上全是表格，容错只是为了人工塞进来的脏数据不炸整条流水线）。
    """
    header: list[list[str]] = []
    body: list[list[str]] = []
    has_sep = False
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        if _MD_SEP_RE.match(line):
            has_sep = True
            continue
        if not _MD_ROW_RE.match(line):
            continue
        cells = split_md_row(line)
        (body if has_sep else header).append(cells)
    return header, has_sep, body


def flatten_headers(header_rows: Sequence[Sequence[str]]) -> list[str]:
    """
    多级表头压平成列名：同一下标各层级去重后用 " / " 连接。
    [["收入","收入"],["营业收入","净利润"]] -> ["收入 / 营业收入", "收入 / 净利润"]
    空层级跳过（合并横幅那类整行同词的行不参与列名）。
    """
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    columns: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in header_rows[:MAX_HEADER_LEVELS]:
            if col >= len(row):
                continue
            cell = str(row[col] or "").strip()
            if cell and cell not in parts:
                parts.append(cell)
        columns.append(LEVEL_JOIN.join(parts))
    return columns


def resolve_columns(
    block: dict[str, Any],
    header_rows: Sequence[Sequence[str]],
    body_rows: Sequence[Sequence[str]],
) -> list[str]:
    """
    定列名，优先级：parser 已给的 header > 多级表头压平 > pick_header_row 兜底。
    parser 的 header 是解析阶段用合并单元格还原算出来的，比这里重新压平更准，
    所以能用就用；拿不到再从 markdown 自己推。
    长度不足数据行宽度的部分补「列N」占位（缺列名的行不能参与「列名: 值」拼接）。
    """
    given = [str(c or "").strip() for c in (block.get("header") or [])]
    columns = given if any(given) else flatten_headers(header_rows)
    if not columns:
        grid = [list(r) for r in header_rows] + [list(r) for r in body_rows[:3]]
        _idx, picked = pick_header_row(grid) if grid else (0, [])
        columns = [str(c or "").strip() for c in picked]
    width = max([len(c) for c in body_rows] + [len(columns), 1])
    out = list(columns[:width])
    while len(out) < width:
        out.append(f"列{len(out) + 1}")
    return [c if c else f"列{i + 1}" for i, c in enumerate(out)]


def row_to_kv(cells: Sequence[str], columns: Sequence[str]) -> str:
    """
    一行 -> "列1: 值1 | 列2: 值2"。
    空值跳过：拼出「备注: 」这种尾巴会让 sparse 检索里出现只有列名没有内容的噪声 token，
    也会白白吃掉 chunk 预算。
    """
    pairs: list[str] = []
    for i, value in enumerate(cells):
        text = str(value or "").strip()
        if not text:
            continue
        name = columns[i] if i < len(columns) and columns[i] else f"列{i + 1}"
        pairs.append(f"{name}{KV_SEP}{text}")
    return KV_JOIN.join(pairs)


def rows_to_markdown(header_rows: Sequence[Sequence[str]], body_rows: Sequence[Sequence[str]]) -> str:
    """把表头 + 若干数据行重新渲染成 markdown（每片都自带完整表头）。"""
    grid = [list(r) for r in header_rows] + [list(r) for r in body_rows]
    if not grid:
        return ""
    return table_to_markdown(grid, header_rows=len(header_rows))


def _row_groups(
    header_rows: Sequence[Sequence[str]],
    body_rows: Sequence[Sequence[str]],
    cfg: ChunkConfig,
) -> list[list[list[str]]]:
    """
    按「字符 + 行数」双约束把数据行分组。
    表头自身的开销要先从字符预算里扣掉（每片都要重复表头，预算不扣会让片数偏多）；
    单行就超预算时该组只装这一行 —— 行数再大也不能切进一行内部，
    一行是一条业务事实，切开就两边都读不懂。
    """
    header_chars = len(rows_to_markdown(header_rows, []))
    budget = max(cfg.max_table_chars - header_chars, 120)
    max_rows = max(cfg.max_table_rows, 1)

    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    used = 0
    for row in body_rows:
        cost = len(KV_JOIN.join(str(c or "") for c in row)) + 8  # 8 ≈ "| " + " |" 的排版开销
        if current and (used + cost > budget or len(current) >= max_rows):
            groups.append(current)
            current = []
            used = 0
        current.append(row)
        used += cost
    if current:
        groups.append(current)
    return groups


def chunk_table(
    block: dict[str, Any],
    metadata: dict[str, Any],
    cfg: ChunkConfig | dict[str, Any] | None = None,
    chunk_type: str = "table",
    table_index: int = 1,
    doc_id: str = "",
) -> list[dict[str, Any]]:
    """
    表格块 -> chunk 列表（需求点名的函数，签名 chunk_table(block, metadata) 的扩展版）。

    整表能塞下就一整块；超过 max_table_chars / max_table_rows 才按行组分片，
    每片都重复表头。参数 chunk_type 允许传 "sheet"（xlsx 的区域块沿用本切分器，
    但检索侧要能区分「文档里的表」和「整张电子表格」）。
    section_path 追加「表格N」，与统一格式示例「第3章 > 3.2 > 表格1」对齐。
    """
    if chunk_type not in ("table", "sheet"):
        raise ValueError(f"chunk_table 只处理 table/sheet，收到 {chunk_type}")
    config = resolve_config(cfg)
    content = str(block.get("content") or "").strip()
    if not content:
        return []

    base_section = str(block.get("section_path") or "")
    section_path = f"{base_section} > 表格{table_index}" if base_section else f"表格{table_index}"
    pages = [block["page"]] if block.get("page") else None
    bbox = block.get("bbox")

    header_rows, has_sep, body_rows = parse_markdown_table(content)
    if not has_sep or (not header_rows and not body_rows):
        # 不是合法 markdown 表格（图片式表格、纯文本表名）：原样成一块，绝不丢内容
        return [
            new_chunk(
                doc_id=doc_id,
                section_path=section_path,
                idx=0,
                content=content,
                chunk_type=chunk_type,
                metadata=metadata,
                bbox=bbox,
                pages=pages,
                token_estimate=estimate_tokens(content),
                n_rows=len(body_rows) or None,
                not_structured=True,
                token_backend=token_estimator_name(),
            )
        ]

    columns = resolve_columns(block, header_rows, body_rows)
    # header_rows 为空（表格没表头）时补一行列名，保证每片都带表头
    if not header_rows:
        header_rows = [columns]

    # 整表一块：字符数与行数都在预算内
    if len(content) <= config.max_table_chars and len(body_rows) <= config.max_table_rows:
        groups = [list(body_rows)]
    else:
        groups = _row_groups(header_rows, body_rows, config)

    out: list[dict[str, Any]] = []
    row_cursor = 0
    for part, group in enumerate(groups):
        markdown = rows_to_markdown(header_rows, group)
        if not markdown:
            row_cursor += len(group)
            continue
        # 「列名: 值」逐行展开（每片自带，检索侧精确路可直接用）
        kv_lines = [row_to_kv(row, columns) for row in group]
        kv_lines = [ln for ln in kv_lines if ln]
        flat = "\n".join(kv_lines)
        too_big = len(markdown) + len(flat) > config.max_table_row_lines_chars
        first_row = row_cursor + 1
        last_row = row_cursor + len(group)
        row_cursor = last_row
        out.append(
            new_chunk(
                doc_id=doc_id,
                section_path=section_path,
                idx=part,
                content=markdown,
                chunk_type=chunk_type,
                metadata=metadata,
                bbox=bbox,
                pages=pages,
                token_estimate=estimate_tokens(markdown + flat),
                header=columns,
                columns=columns,
                n_rows=len(group),
                n_cols=len(columns),
                row_lines=kv_lines if kv_lines and not too_big else None,
                row_text=flat if flat and not too_big else None,
                table_index=table_index,
                part=part + 1 if len(groups) > 1 else None,
                n_parts=len(groups) if len(groups) > 1 else None,
                row_range=f"{first_row}-{last_row}" if len(groups) > 1 else None,
                sheet_name=block.get("sheet_name"),
                table_title=block.get("table_title"),
                token_backend=token_estimator_name(),
            )
        )
    return out


def chunk_sheet_block(block: dict[str, Any], metadata: dict[str, Any], cfg: ChunkConfig | dict[str, Any] | None = None, doc_id: str = "") -> list[dict[str, Any]]:
    """xlsx 的 type=sheet 块：语义与表格一致，只是 chunk_type 要标成 sheet。"""
    return chunk_table(block, metadata, cfg, chunk_type="sheet", table_index=int(block.get("region_index") or 1), doc_id=doc_id)


if __name__ == "__main__":
    import argparse
    import json

    cli = argparse.ArgumentParser(description="把一个 table/sheet block 转成 chunk（调试用）")
    cli.add_argument("json_path", help="含单个 block 的 JSON 文件（parser 的 blocks[i]）")
    cli.add_argument("--type", default="table", choices=["table", "sheet"])
    args = cli.parse_args()

    block_data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    for piece in chunk_table(block_data, {}, chunk_type=args.type):
        print(json.dumps({k: v for k, v in piece.items() if k != "metadata"}, ensure_ascii=False, indent=2))
