# -*- coding: utf-8 -*-
"""
XLSX 解析（openpyxl）。

核心原则：**表格不是文本**。每张 sheet 产出的是「带列名的 markdown 区块」，
每个区块自带表头，这样在 chunking / 检索阶段任何一片都能独立被理解
（丢了列名的数字行，对 dense/sparse 检索都是纯噪声）。

处理要点：
  1. read_only=True + data_only=True：100GB 库里 xlsx 动辄几十万行，
     非只读模式会把整张表建成对象塞进内存。data_only 取公式的缓存计算值；
     若文件从未被 Excel 打开保存过，公式单元格缓存为空，这里会退化成读原始公式串。
  2. 空行空列全部剔除（xlsx 的 dimension 常常虚报到一个格式被改过的空单元格）。
  3. 合并单元格：openpyxl 只在左上角存值，其余为 None。
     这里按 merged_cells.ranges 把左上角的值「回填」到整个区域，
     否则「2024年」合并了 12 行后，第 2 行开始的时间维度就消失了。
  4. 每 100 个数据行切一个区块（对齐 settings.chunking.max_table_rows_per_chunk 的思路）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from ingestion.classifiers.doc_classifier import classify, merge_into_metadata
from ingestion.parsers.common import (
    compute_doc_id,
    detect_language,
    dump_json_for_cli,
    new_block,
    new_document,
    normalize_text,
    pick_header_row,
    table_to_markdown,
    utc_now_iso,
)

# 每个区块的数据行数（不含表头）
DEFAULT_ROWS_PER_BLOCK = 100
# 表头最多连续识别几行（多级表头很常见，但超过 3 行通常是数据本身）
MAX_HEADER_ROWS = 3
# 判定「这行是表头」的启发：短、无数字、非空单元格占比高
_HEADER_MAX_CELL_CHARS = 24


def _rows_per_block() -> int:
    """
    每个区块的数据行数。
    配置里 max_table_rows_per_chunk=30 是面向「文本 chunk 里的表格」的，
    xlsx 的 markdown 行很短，30 行一片会把一张几万行的表切成上千块，
    所以这里取 max(配置值, 100)：既尊重配置，又不至于碎到无法检索。
    """
    from ingestion.parsers.common import get_settings

    raw = (get_settings().get("chunking") or {}).get("max_table_rows_per_chunk")
    try:
        configured = int(raw) if raw else DEFAULT_ROWS_PER_BLOCK
    except (TypeError, ValueError):
        configured = DEFAULT_ROWS_PER_BLOCK
    return max(configured, DEFAULT_ROWS_PER_BLOCK)


def _cell_text(value: Any) -> str:
    """单元格值标准化为字符串：日期只到日、float 去掉多余的 .0、None 给空串。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if hasattr(value, "strftime"):
        # 日期时间：带时分秒的保留，纯日期只留到日，避免 markdown 里一屏 19 位数字
        try:
            if value.hour or value.minute:
                return value.strftime("%Y-%m-%d %H:%M")
            return value.strftime("%Y-%m-%d")
        except Exception:
            return str(value)
    return normalize_text(str(value)).replace("\n", " ")


def _merged_value_map(worksheet: Any) -> dict[tuple[int, int], str]:
    """
    合并单元格回填表：{(行, 列): 左上角值}。

    read_only 模式下 worksheet.merged_cells 依然可用（来自 worksheet XML），
    但注意 read_only 的 iter_rows 不会给出 MergedCell 对象，所以只能靠这张表补值。
    """
    lookup: dict[tuple[int, int], str] = {}
    ranges = getattr(getattr(worksheet, "merged_cells", None), "ranges", []) or []
    for rng in ranges:
        top_left = (rng.min_row, rng.min_col)
        # 左上角真实值要回读：read_only 下拿不到 cell 对象，只能从原始行缓存里取，
        # 因此这里先占位，等主循环扫到该行时再写回（见 hydrate 逻辑）
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                lookup[(r, c)] = top_left
    return lookup


def _detect_header_rows(rows: list[list[str]]) -> int:
    """
    识别表头区行数（支持多级表头 + 顶部合并横幅）。
    规则：从第一行起，只要该行「非空单元格占比 >= 60%」「所有单元格 <= 24 字」
    且「不含纯数字单元格」，就算表头；最多 3 行。
    特例：只有一个非空值的行是「合并出来的表题横幅」，它属于表头区但永远不是列名行，
    所以记一行并继续往下看，否则列名行会被当成数据行（本函数返回值会少算 1）。
    """
    count = 0
    for row in rows[: MAX_HEADER_ROWS + 1]:
        filled = [c for c in row if c]
        if not filled:
            break
        if len(filled) == 1 and len(row) > 1:
            count += 1
            continue
        if len(filled) / max(len(row), 1) < 0.6:
            break
        if any(len(c) > _HEADER_MAX_CELL_CHARS for c in filled):
            break
        if any(c.replace(",", "").replace(".", "").replace("-", "").isdigit() for c in filled):
            break
        count += 1
    return max(count, 1)


def _collapse_banner_rows(rows: list[list[str]], limit: int = MAX_HEADER_ROWS) -> list[list[str]]:
    """
    顶部「整行都是同一个词」的行，只保留第一个值。

    这是合并单元格回填的副作用：A1:D1 合并成大标题后，回填会把
    "2024年费用明细" 复制进 4 列，markdown 表头就变成四个一模一样的词，
    列名（月份/部门/金额）反而被挤到数据行里去了。
    只对前 limit 行做，数据区里合法重复的值不受影响。
    """
    for row in rows[:limit]:
        filled = [c for c in row if c]
        if len(filled) > 1 and len(set(filled)) == 1:
            first = next(i for i, c in enumerate(row) if c)
            for i in range(len(row)):
                if i != first:
                    row[i] = ""
    return rows


def parse_xlsx(path: str | Path) -> dict[str, Any]:
    """
    解析 xlsx/xlsm -> 统一中间 JSON。
    每张 sheet 按行区间产出多个 type=sheet 的 block，每块自带列名表头。
    """
    p = Path(path)
    data = p.read_bytes()
    doc_id = compute_doc_id(p, data)

    workbook = load_workbook(filename=str(p), read_only=True, data_only=True)
    blocks: list[dict[str, Any]] = []
    sheet_summaries: list[dict[str, Any]] = []
    rows_per_block = _rows_per_block()

    try:
        for worksheet in workbook.worksheets:
            # 隐藏 sheet（辅助计算表）与图表 sheet 不入库
            if getattr(worksheet, "sheet_state", "visible") != "visible":
                continue
            if worksheet.max_row is None or worksheet.max_column is None:
                continue

            raw_rows = _read_sheet_rows(worksheet)
            if not raw_rows:
                continue

            keep_cols = _significant_columns(raw_rows)
            trimmed = [[row[c] for c in keep_cols] for row in raw_rows]
            trimmed = [r for r in trimmed if any(c for c in r)]
            if not trimmed:
                continue

            trimmed = _collapse_banner_rows(trimmed)
            header_n = _detect_header_rows(trimmed)
            headers = trimmed[:header_n]
            body = trimmed[header_n:]
            # 表头区里可能混着合并出来的表题横幅，用 pick_header_row 选出真正的列名行
            col_idx, col_names = pick_header_row(headers)
            table_title = headers[0][0] if col_idx > 0 and headers[0] else None

            for start in range(0, len(body), rows_per_block):
                chunk = body[start : start + rows_per_block]
                # 每块都拼回完整表头（多级表头一并带上）
                grid = headers + chunk
                first_row = start + header_n + 1
                last_row = start + header_n + len(chunk)
                region = f"第{first_row}-{last_row}行"
                md = table_to_markdown(grid, header_rows=header_n)
                if not md:
                    continue
                blocks.append(
                    new_block(
                        "sheet",
                        md,
                        section_path=f"{worksheet.title} > {region}",
                        sheet_name=worksheet.title,
                        header=col_names,
                        columns=col_names,
                        table_title=table_title,
                        region=region,
                        n_rows=len(chunk),
                        n_cols=len(col_names),
                    )
                )

            sheet_summaries.append(
                {
                    "sheet_name": worksheet.title,
                    "n_rows": len(trimmed),
                    "n_cols": len(col_names),
                    "table_title": table_title,
                    "columns": col_names,
                }
            )
    finally:
        workbook.close()

    full_text = "\n".join(str(b.get("content", "")) for b in blocks)
    metadata = {
        "has_table": bool(blocks),
        "has_image": False,
        "language": detect_language(full_text),
        "layout_style": "sheet",
    }
    metadata = merge_into_metadata(metadata, classify(full_text, blocks))

    return new_document(
        doc_id=doc_id,
        source_format="xlsx",
        metadata=metadata,
        blocks=blocks,
        file_name=p.name,
        sheets=sheet_summaries,
        parsed_at=utc_now_iso(),
    )


def _read_sheet_rows(worksheet: Any) -> list[list[str]]:
    """
    流式读取一张 sheet，返回已做「合并单元格回填 + 类型标准化」的二维文本表。

    两遍逻辑：read_only 下 iter_rows 只能顺序过一次，
    所以先把原始行缓存成 values（已经是字符串化结果，体积可控），
    再按 merged ranges 回填。超大 sheet（>50 万行）由 ingestion 层分片，不在此处理。
    """
    merged = _merged_value_map(worksheet)
    raw: list[list[str]] = []
    top_left_cache: dict[tuple[int, int], str] = {}

    for r_idx, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
        cells = [_cell_text(v) for v in row]
        raw.append(cells)
        # 记录每个合并区域左上角的真实值（扫到那一行时才拿得到）
        for c_idx, text in enumerate(cells, start=1):
            key = (r_idx, c_idx)
            if text and merged.get(key) == key:
                top_left_cache[key] = text

    # 回填：非左上角位置若为空，就用所属合并区域左上角的值
    for r_idx in range(1, len(raw) + 1):
        row = raw[r_idx - 1]
        for c_idx in range(1, len(row) + 1):
            if row[c_idx - 1]:
                continue
            top_left = merged.get((r_idx, c_idx))
            if not top_left:
                continue
            value = top_left_cache.get(top_left)
            if value:
                row[c_idx - 1] = value
    return raw


def _significant_columns(rows: list[list[str]]) -> list[int]:
    """
    剔除「整列都空」与「只有装饰性内容」的列。
    返回保留列的下标（0-based）。同时限制最多 80 列，防止宽表一片撑爆 context。
    """
    if not rows:
        return []
    width = max(len(r) for r in rows)
    filled: dict[int, int] = {}
    for row in rows:
        for idx in range(len(row)):
            if row[idx]:
                filled[idx] = filled.get(idx, 0) + 1
    n = len(rows)
    if n >= 20:
        # 大表：有值行数 <2%（且至少 2 行）的列是批注/装饰列
        threshold = max(2, int(n * 0.02))
    else:
        # 小表不激进：只有一格有值的合法列（如「备注」列只填了一行）不该被丢掉
        threshold = 1
    keep = [idx for idx in range(width) if filled.get(idx, 0) >= threshold]
    if not keep:
        keep = list(range(min(width, 20)))
    if len(keep) > 80:
        keep = keep[:80]
    return keep


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="解析 xlsx，输出统一中间 JSON")
    cli.add_argument("path", help="xlsx 文件路径")
    args = cli.parse_args()
    print(dump_json_for_cli(parse_xlsx(args.path)))
