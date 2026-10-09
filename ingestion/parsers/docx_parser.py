# -*- coding: utf-8 -*-
"""
DOCX 解析（python-docx）。

三个必须处理对的点：
  1. 顺序：doc.paragraphs 和 doc.tables 是两个独立列表，直接分别遍历会打乱
     「表格出现在哪一章节下」的事实。这里按 body 的 XML 子元素顺序迭代，
     段落与表格严格按原文先后顺序产出 block。
  2. 标题层级：中文 Word 样式名是「标题 1」，英文是 "Heading 1"，
     两种都要认；另外大量文档根本没用内置标题样式，只用「加粗 + 大字号」，
     所以对没有 style level 的段落再用一次字号/加粗判定（复用 pdf 的标题打分函数）。
  3. 表格：合并单元格在 python-docx 里会把同一个值重复出现在多个 cell，
     必须去重（否则 markdown 里一行全是同一个词），表头行必须保留进 header 字段，
     供 chunking 批次在切分表格时给每片重复表头。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import docx
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from ingestion.classifiers.doc_classifier import classify, merge_into_metadata
from ingestion.parsers.common import (
    HeadingStack,
    compute_doc_id,
    detect_language,
    document_root_title,
    dump_json_for_cli,
    filter_blank_rows,
    new_block,
    new_document,
    normalize_text,
    pick_header_row,
    table_to_markdown,
    utc_now_iso,
)
from ingestion.parsers.image_parser import parse_image_bytes

# 「标题 1」「Heading 2」「标题二」都取出层级数字
_HEADING_STYLE_RE = re.compile(r"(?:heading|标题)\s*([1-9一二三四五六七八九])", re.IGNORECASE)
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# 无内置样式时的兜底标题判定：短行 + 加粗 + 字号明显大于正文
_TITLE_MAX_CHARS = 40
_TITLE_BODY_SIZE_RATIO = 1.3
# 整篇都拿不到任何字号信息时的正文基准（Word 默认模板 Normal = 11pt）
DEFAULT_BODY_SIZE = 11.0
# 句末标点：以这些结尾的是句子，不是标题
_SENTENCE_END = ("。", "！", "？", "；", ";", ".", "!", "?")

# 图片关系类型（python-docx 里内嵌图都是这个）
_BLIP_TAG = qn("a:blip")
_R_EMBED = qn("r:embed")

# 图注行：「图1」「图 2-1」「附表3」「Fig. 1」「Chart 2」开头 + 编号
_CAPTION_RE = re.compile(r"^(图|附图|附表|表\s*\d|Fig\.?|Chart\.?)\s*[\d\-.－—]+")


# =================================================================== 遍历
def iter_block_items(document: Any) -> Iterator[Any]:
    """按 body 子元素顺序产出 Paragraph / Table，保证正文与表格的原始先后关系。"""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield DocxParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, document)


def heading_level_of(paragraph: DocxParagraph, body_size: float | None) -> tuple[int, str] | None:
    """
    返回 (level, title)，非标题返回 None。

    优先级：内置标题样式 > 字号/加粗启发。
    style.name 里 "Heading 1"/"标题 1" 直接给层级；
    退化情况下（样式全是「正文」）用「加粗 + 字号 >= 正文均值×1.3 + 短行」判定为 2 级标题，
    因为这类文档的「大字号短行」几乎不会是小节标题，直接给 2 级最稳。
    """
    style_name = ""
    try:
        style_name = str(paragraph.style.name or "")
    except Exception:
        style_name = ""

    m = _HEADING_STYLE_RE.search(style_name)
    if m:
        token = m.group(1)
        level = _CN_NUM[token] if token in _CN_NUM else int(token)
        return level, normalize_text(paragraph.text)
    if style_name.lower() in ("title", "标题"):
        return 1, normalize_text(paragraph.text)

    text = normalize_text(paragraph.text)
    if not text or len(text) > _TITLE_MAX_CHARS or paragraph.style.name not in ("Normal", "正文"):
        # 只在最日常的正文样式上做退化判定，避免把列表/页眉误升成标题
        return None
    if body_size is None or body_size <= 0:
        return None
    # 句子不是标题（中文 Word 里「正文段落首行加粗加大」很常见）
    if text.endswith(_SENTENCE_END):
        return None

    # 有效字号：run 显式 > 段落样式 > 正文基准（继承正文基准的比值是 1，不会误判）
    style_size = _style_font_size(paragraph.style)
    sizes = [
        float(run.font.size.pt) if run.font.size else float(style_size or body_size)
        for run in paragraph.runs
        if (run.text or "").strip()
    ]
    bold = any(r.font.bold for r in paragraph.runs)
    max_size = max(sizes) if sizes else None
    if max_size and bold and max_size >= body_size * _TITLE_BODY_SIZE_RATIO:
        return 2, text
    return None


def estimate_body_font_size(document: Any) -> float | None:
    """
    估算正文基准字号：全文档「有效字号」按字符数加权取众数。

    解析链：run 显式字号 > 段落样式字号（含 basedOn 继承）> Normal 样式字号 > 默认 11pt。
    为什么不能只统计 run 显式字号：中文 Word 里绝大多数正文 run 不带 w:sz，
    字号是从样式继承来的。只数显式值会得到「只有标题带字号」的假象，
    基准被估成 18pt，于是 18pt 的加粗假标题再也不满足 >= 基准×1.3，整篇一个标题都识别不出来。
    """
    normal_size = _style_font_size(_get_style(document, "Normal"))
    votes: dict[float, int] = {}
    for paragraph in document.paragraphs:
        p_size = _style_font_size(paragraph.style) or normal_size
        for run in paragraph.runs:
            text = run.text or ""
            if not text.strip():
                continue
            size = float(run.font.size.pt) if run.font.size else float(p_size or DEFAULT_BODY_SIZE)
            key = round(size * 2) / 2  # 半磅粒度，避免 10.4/10.5 分裂
            votes[key] = votes.get(key, 0) + len(text)
    if not votes:
        return normal_size or DEFAULT_BODY_SIZE
    return max(votes, key=lambda k: votes[k])


def _get_style(document: Any, name: str) -> Any:
    """按名字取样式，取不到返回 None（中文模板里样式名可能是「正文」）。"""
    for candidate in (name, {"Normal": "正文"}.get(name, "")):
        if not candidate:
            continue
        try:
            return document.styles[candidate]
        except (KeyError, TypeError):
            continue
    return None


def _style_font_size(style: Any) -> float | None:
    """
    从样式对象取字号，沿 basedOn 继承链最多回溯 5 层。
    python-docx 的 style.font.size 只给该样式自己声明的值，
    Heading 2 通常不设字号（继承 Heading 1 / Normal），不回溯就拿不到。
    """
    node = style
    for _ in range(5):
        if node is None:
            return None
        try:
            font = node.font
        except (AttributeError, TypeError):
            return None
        if font is not None and font.size is not None:
            return float(font.size.pt)
        node = getattr(node, "base_style", None)
    return None


# =================================================================== 表格
def table_grid(table: DocxTable) -> list[list[str]]:
    """
    表格 -> 二维网格，并把合并单元格「拆回」它覆盖的列。

    python-docx 对横向合并的 cell 会返回同一个 _tc 元素多次，
    合并后的 cell.text 是把区域内各段用 \\n 拼起来的。两种情况分开处理：
      A. 「科目\\n2023年\\n2024年」跨 3 列 —— 每列各有一段，行数 == 跨度，
         直接拆回 3 列，列名完整保留（这是多级/横幅表头最常见的形态）；
      B. 一段文字跨多列（真合并横幅），或跨行合并 —— 值放首列，其余列留空，
         绝不重复填同一个值（重复会让 markdown 里出现「2024年 | 2024年 | 2024年」）。
    """
    grid: list[list[str]] = []
    for row in table.rows:
        # 先按 _tc 身份聚合成 [(marker, raw_text, span), ...]，保持列序
        groups: list[list[Any]] = []
        for cell in row.cells:
            marker = id(cell._tc)
            if groups and groups[-1][0] == marker:
                groups[-1][2] += 1
                continue
            groups.append([marker, cell.text or "", 1])

        cells: list[str] = []
        for _marker, raw, span in groups:
            parts = [normalize_text(p) for p in (raw or "").split("\n") if normalize_text(p)]
            if span > 1 and len(parts) == span:
                cells.extend(parts)                     # 情况 A：拆回各列
            else:
                cells.append(normalize_text((raw or "").replace("\n", " ")))
                cells.extend([""] * (span - 1))         # 情况 B：延续列留空
        grid.append(cells)
    return grid


# =================================================================== 图片
def paragraph_images(paragraph: DocxParagraph, document: Any) -> list[tuple[str, bytes]]:
    """
    抽出段落里内嵌的图片，返回 [(rId, bytes)]。
    只从 relationship 里取真字节，不用临时文件——落盘临时图会带来权限与清理问题。
    """
    results: list[tuple[str, bytes]] = []
    for blip in paragraph._p.iter(_BLIP_TAG):
        rid = blip.get(_R_EMBED)
        if not rid:
            continue
        try:
            part = document.part.related_parts[rid]
        except KeyError:
            continue
        blob = getattr(part, "blob", None)
        if blob:
            results.append((rid, blob))
    return results


def _image_format_of(part_name: str, content_type: str) -> str:
    """由 partname / MIME 推断归一化后的 source_format（统一只承认 jpg / png）。"""
    name = (part_name or "").lower()
    ctype = (content_type or "").lower()
    if ".png" in name or "png" in ctype:
        return "png"
    return "jpg"


# =================================================================== parser
def parse_docx(path: str | Path) -> dict[str, Any]:
    """解析 docx -> 统一中间 JSON。"""
    p = Path(path)
    data = p.read_bytes()
    doc_id = compute_doc_id(p, data)

    document = docx.Document(str(p))
    core = getattr(document, "core_properties", None)
    body_size = estimate_body_font_size(document)
    stack = HeadingStack()
    # 根路径：文档属性标题优先，没有就用文件名。
    # 手册类文档的「第1章」单独出现在检索结果里没有意义，必须带上是哪本手册。
    stack.seed_root(document_root_title(getattr(core, "title", None), p))
    blocks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"n_images": 0, "n_tables": 0, "model_errors": []}

    items = list(iter_block_items(document))
    for idx, item in enumerate(items):
        if isinstance(item, DocxParagraph):
            # 图注通常是图片段的「下一段」，所以要把下一个块也传进去
            nxt = items[idx + 1] if idx + 1 < len(items) else None
            next_text = nxt.text if isinstance(nxt, DocxParagraph) else ""
            _handle_paragraph(item, document, body_size, stack, blocks, doc_id, stats, next_text)
        elif isinstance(item, DocxTable):
            _handle_table(item, stack, blocks)
            stats["n_tables"] += 1

    full_text = "\n".join(str(b.get("content", "")) for b in blocks)
    # has_image 用「抽到几张图」判断，而不是「几个 image 块有正文」：
    # 一张纯照片 OCR 没出字仍然是「本文档含图」，元数据不能说没有
    has_table = stats["n_tables"] > 0
    has_image = stats["n_images"] > 0

    metadata = {
        "has_table": has_table,
        "has_image": has_image,
        "language": detect_language(full_text),
        "layout_style": "prose",
    }
    metadata = merge_into_metadata(metadata, classify(full_text, blocks))

    return new_document(
        doc_id=doc_id,
        source_format="docx",
        metadata=metadata,
        blocks=blocks,
        file_name=p.name,
        # 文档属性里往往已经有作者/日期，chunk payload 用不上，但检索结果展示很有用
        doc_title=getattr(core, "title", None) or None,
        doc_author=getattr(core, "author", None) or None,
        doc_created=getattr(core, "created", None).isoformat()
        if getattr(core, "created", None)
        else None,
        body_font_size=body_size,
        n_images=stats["n_images"],
        parsed_at=utc_now_iso(),
        model_errors=stats["model_errors"] or None,
    )


def _handle_paragraph(
    paragraph: DocxParagraph,
    document: Any,
    body_size: float | None,
    stack: HeadingStack,
    blocks: list[dict[str, Any]],
    doc_id: str,
    stats: dict[str, Any],
    next_text: str = "",
) -> None:
    """段落 -> title / text / image block。同一段落里既有文字又有图片时，两者都产出。"""
    heading = heading_level_of(paragraph, body_size)
    text = normalize_text(paragraph.text)

    if heading:
        level, title = heading
        if title:
            stack.push(level, title)
            blocks.append(
                new_block(
                    "title",
                    title,
                    section_path=stack.path,
                    font=_dominant_font(paragraph),
                    size=_dominant_size(paragraph),
                )
            )
    elif text:
        blocks.append(
            new_block(
                "text",
                text,
                section_path=stack.path,
                font=_dominant_font(paragraph),
                size=_dominant_size(paragraph),
            )
        )

    for rid, blob in paragraph_images(paragraph, document):
        related = document.part.related_parts.get(rid)
        fmt = _image_format_of(
            str(getattr(related, "partname", "")), str(getattr(related, "content_type", ""))
        )
        stats["n_images"] += 1
        caption = _paragraph_caption(text, next_text)
        image_doc = parse_image_bytes(
            blob,
            source_format=fmt,
            caption=caption,
            section_path=stack.path,
            doc_id=f"{doc_id}:img:{rid}",
        )
        if not image_doc or not image_doc["blocks"]:
            continue
        block = image_doc["blocks"][0]
        block["doc_id"] = image_doc["doc_id"]
        blocks.append(block)
        for err in image_doc.get("model_errors") or []:
            if err not in stats["model_errors"]:
                stats["model_errors"].append(err)


def _paragraph_caption(own_text: str, next_text: str) -> str:
    """
    图注：docx 惯例是图片所在段落本身，或紧随其后的段落以「图1 / 附表2 / Fig.1」开头。
    只有这两处才算图注——再远就可能是正文里毫不相干的编号句，
    错配的 caption 会被当成图片语义写进 OCR/CLIP 结果，比缺失更难查。
    """
    for candidate in (own_text, next_text):
        text = normalize_text(candidate or "")
        if text and _CAPTION_RE.match(text):
            return text
    return ""


def _handle_table(table: DocxTable, stack: HeadingStack, blocks: list[dict[str, Any]]) -> None:
    """表格 -> table block（markdown + 显式 header，供切分层复用）。"""
    grid = filter_blank_rows(table_grid(table))
    if not grid:
        return
    # 表头可能被合并横幅占位，用 pick_header_row 选出真正的列名行
    h_idx, header = pick_header_row(grid)
    blocks.append(
        new_block(
            "table",
            table_to_markdown(grid, header_rows=max(h_idx + 1, 1)),
            section_path=stack.path,
            header=header,
            header_row_index=h_idx,
            n_rows=len(grid),
            n_cols=max(len(r) for r in grid),
        )
    )


def _dominant_font(paragraph: DocxParagraph) -> str | None:
    """段落里字数最多的 run 的字体名（东亚字体优先，中文文档真正显示用的是它）。"""
    best: tuple[int, str | None] = (0, None)
    for run in paragraph.runs:
        weight = len(run.text or "")
        if weight <= best[0]:
            continue
        name = None
        rpr = run._element.rPr
        if rpr is not None and rpr.rFonts is not None:
            name = rpr.rFonts.get(qn("w:eastAsia")) or rpr.rFonts.get(qn("w:ascii"))
        best = (weight, name or run.font.name)
    return best[1]


def _dominant_size(paragraph: DocxParagraph) -> float | None:
    """段落里字数最多的 run 的字号（pt）。"""
    best: tuple[int, float | None] = (0, None)
    for run in paragraph.runs:
        weight = len(run.text or "")
        if run.font.size and weight > best[0]:
            best = (weight, float(run.font.size.pt))
    return best[1]


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="解析 docx，输出统一中间 JSON")
    cli.add_argument("path", help="docx 文件路径")
    args = cli.parse_args()
    print(dump_json_for_cli(parse_docx(args.path)))
