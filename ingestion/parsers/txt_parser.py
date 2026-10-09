# -*- coding: utf-8 -*-
"""
纯文本 / Markdown 解析。

难点只有两个：
  1. 编码：中文 Windows 导出的 txt 经常是 GBK，而 utf-16 文件不带 BOM 时
     根本没法靠猜（每个 ASCII 字符后面都跟一个 \x00）。所以顺序是
     BOM 判定 -> utf-8 严格解码 -> gbk 严格解码 -> charset-normalizer 统计判定 -> utf-16 兜底。
     任何一步用 except UnicodeDecodeError 严格失败才继续，不用 errors="ignore"，
     因为 ignore 会把坏字节静默吃掉，最后得到一份「看起来正常但丢了字」的语料。
  2. 分块：按空行切，遇到 Markdown 标题 / 「第X章」行就同时更新 section_path，
     保证纯文本也能带章节定位（否则检索结果只能说「在第 40 块」，人无法复核）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ingestion.classifiers.doc_classifier import classify, merge_into_metadata
from ingestion.parsers.common import (
    HeadingStack,
    compute_doc_id,
    detect_language,
    dump_json_for_cli,
    filter_blank_rows,
    new_block,
    new_document,
    normalize_text,
    pick_header_row,
    read_bytes,
    table_to_markdown,
    utc_now_iso,
)

# 解码尝试顺序：BOM 命中时会被短路
_FALLBACK_ENCODINGS = ("utf-8", "gbk", "gb18030", "big5", "utf-16", "utf-32")

# 标题行识别：markdown # / 中文「第X章」「X.Y」编号
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*#*$")
_CN_CHAPTER_RE = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十百]+\s*[章篇部卷]\s*(?P<title>.{0,40})$")
_NUM_SECTION_RE = re.compile(r"^\s*(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\s*[、.．]?\s*(?P<title>[^\s].{0,40})$")

# 表格行（markdown pipe 表）与分隔线
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

# 空行分段
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# 语言占比阈值（与全局口径一致）
ZH_RATIO = 0.30
EN_RATIO = 0.05


# =================================================================== 编码
def detect_encoding(data: bytes) -> tuple[str, str]:
    """
    探测字节流的文本编码。

    返回 (encoding, method)，method 取值：bom / strict / normalizer / fallback
    normalizer 用 charset-normalizer（batch1 依赖里已随 requests/transformers 传递进来，
    若环境缺失则自动跳过，不影响前三步）。
    """
    # 1) BOM 最可靠，直接短路
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return "utf-16", "bom"
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "bom"
    if data.startswith(b"\x00\xfe\xff") or data.startswith(b"\xff\xfe\x00\x00"):
        return "utf-32", "bom"

    # 2) 严格解码逐个尝试：只要有一个字符解不开就换下一个编码
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            data.decode(enc)
            return enc, "strict"
        except UnicodeDecodeError:
            continue

    # 3) 统计法判定（GBK 解得开但语义乱码的情况，交给它给个更可能的候选）
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best is not None and getattr(best, "encoding", None):
            return str(best.encoding), "normalizer"
    except Exception:
        pass

    # 4) 兜底：utf-16/32（无 BOM 时按字节对齐猜）
    if len(data) >= 4 and (data[:2] in (b"\x00\x00",) or data[2:4] == b"\x00\x00" or b"\x00" in data[:4]):
        return "utf-16-le", "fallback"
    return "utf-8", "fallback"


def decode_bytes(data: bytes) -> tuple[str, str, str]:
    """返回 (文本, 编码, 判定方式)。解码失败退化为 replace，但会把 encoding 标出来供人工复核。"""
    encoding, method = detect_encoding(data)
    try:
        return data.decode(encoding), encoding, method
    except (UnicodeDecodeError, LookupError):
        return data.decode("utf-8", errors="replace"), "utf-8(replace)", "fallback"


# =================================================================== 结构
def heading_level_of(line: str) -> tuple[int, str] | None:
    """
    判断这一行是不是标题，返回 (level, title)。
    markdown # 数量即层级；「第X章」=1，「第X节」=2，「x.y」编号按点号层数 +1。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    m = _MD_HEADING_RE.match(stripped)
    if m:
        return len(m.group(1)), normalize_text(m.group("title"))

    m = _CN_CHAPTER_RE.match(stripped)
    if m:
        chapter = stripped[:6]
        tail = normalize_text(m.group("title") or "")
        return 1, normalize_text(f"{chapter} {tail}").strip()

    m = _NUM_SECTION_RE.match(stripped)
    if m and m.group("num"):
        # 编号段落必须是短行且不以句号结尾，否则会把普通句子误判成标题
        if stripped.endswith(("。", "！", "？", ".", "!", "?")):
            return None
        depth = m.group("num").count(".") + 1
        title = normalize_text(f"{m.group('num')} {m.group('title')}")
        return depth, title
    return None


def _split_markdown_table(chunk_lines: list[str]) -> tuple[str, list[list[str]]]:
    """
    从一个片段里剥出 markdown 表格，返回（去掉表格后的文本, 表格网格）。

    认定条件很严：必须是「表头行 + |---| 分隔线」开头才算表格，
    否则正文里偶然出现的竖线（比如「男|女」）会被误当表格，毁掉这块的语义。
    分隔线本身不进 grid（它是格式不是内容）。
    """
    grid: list[list[str]] = []
    rest: list[str] = []
    i = 0
    n = len(chunk_lines)
    while i < n:
        line = chunk_lines[i]
        nxt = chunk_lines[i + 1] if i + 1 < n else ""
        is_table_start = bool(_TABLE_ROW_RE.match(line)) and bool(_TABLE_SEP_RE.match(nxt))
        if not is_table_start:
            rest.append(line)
            i += 1
            continue
        # 连续 pipe 行全部收下，跳过分隔线
        while i < n and _TABLE_ROW_RE.match(chunk_lines[i]):
            raw = chunk_lines[i].strip()
            if not _TABLE_SEP_RE.match(raw):
                cells = raw.strip("|").split("|")
                grid.append([c.strip() for c in cells])
            i += 1
    return "\n".join(rest).strip(), grid


def _looks_like_form(paragraphs: list[str]) -> bool:
    """
    form 版式：大量「键：值」短行（合同要素表、报销单文本版）。
    短行占比 > 50% 且含冒号行占比 > 22% 才判定，阈值与 doc_classifier 保持一致。
    """
    lines = [p.strip() for p in paragraphs if p.strip()]
    if len(lines) < 8:
        return False
    short = sum(1 for ln in lines if len(ln) <= 24)
    colon = sum(1 for ln in lines if re.match(r"^[\u4e00-\u9fa5A-Za-z0-9（）()·]{1,16}[：:]", ln))
    return short / len(lines) > 0.5 and colon / len(lines) > 0.22


# =================================================================== parser
def parse_txt(path: str | Path) -> dict[str, Any]:
    """
    解析 txt/md 文件 -> 统一中间 JSON。
    blocks 里 text 与 title 分开，表格单独成 table block（重新用 table_to_markdown 渲染，
    保证全库表格格式与 pdf/docx/xlsx 一致，而不是保留原文里那套缩进）。
    """
    p = Path(path)
    data = read_bytes(p)
    doc_id = compute_doc_id(p, data)
    text, encoding, enc_method = decode_bytes(data)

    paragraphs = [seg for seg in _PARAGRAPH_SPLIT_RE.split(text) if seg.strip()]
    stack = HeadingStack()
    blocks: list[dict[str, Any]] = []
    has_table = False

    for seg in paragraphs:
        seg = seg.rstrip()
        lines = seg.splitlines()
        residual, grid = _split_markdown_table(lines)
        if grid:
            clean = filter_blank_rows(grid)
            md = table_to_markdown(clean, header_rows=1)
            if md:
                has_table = True
                h_idx, header = pick_header_row(clean)
                blocks.append(
                    new_block(
                        "table",
                        md,
                        section_path=stack.path,
                        header=header,
                        header_row_index=h_idx,
                        n_rows=len(clean),
                        n_cols=max(len(r) for r in clean),
                    )
                )

        candidate = residual or seg
        first_line = candidate.strip().splitlines()[0] if candidate.strip() else ""
        heading = heading_level_of(first_line)

        if heading and len(candidate.strip().splitlines()) == 1:
            level, title = heading
            stack.push(level, title)
            blocks.append(new_block("title", title, section_path=stack.path))
            continue

        if heading:
            # 标题行后面还跟着正文：标题先入栈并单独成块，剩余部分继续做正文
            level, title = heading
            stack.push(level, title)
            blocks.append(new_block("title", title, section_path=stack.path))
            candidate = "\n".join(candidate.strip().splitlines()[1:]).strip()
            if not candidate:
                continue

        for piece in [b for b in candidate.split("\n\n") if b.strip()]:
            blocks.append(new_block("text", normalize_text(piece), section_path=stack.path))

    full_text = "\n".join(str(b.get("content", "")) for b in blocks)
    is_form = _looks_like_form(paragraphs)
    metadata = {
        "has_table": has_table,
        "has_image": False,
        "language": detect_language(full_text, ZH_RATIO, EN_RATIO),
        "layout_style": "form" if is_form else "prose",
    }
    metadata = merge_into_metadata(metadata, classify(full_text, blocks))
    # parser 的结构判定优先：form/prose 由短行比例决定，不被关键词分类反向改写
    metadata["layout_style"] = "form" if is_form else metadata.get("layout_style", "prose")

    return new_document(
        doc_id=doc_id,
        source_format="txt",
        metadata=metadata,
        blocks=blocks,
        file_name=p.name,
        encoding=encoding,
        encoding_method=enc_method,
        parsed_at=utc_now_iso(),
    )


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="解析 txt/md，输出统一中间 JSON")
    cli.add_argument("path", help="文本文件路径")
    args = cli.parse_args()
    print(dump_json_for_cli(parse_txt(args.path)))
