# -*- coding: utf-8 -*-
"""
PPTX 解析（python-pptx）。

策略：**一页一个 slide block**，content = 标题 + 正文要点 + 演讲者备注。
理由：PPT 的语义单元天然就是「一页」，按字数硬切会把「问题/原因/对策」三行拆散，
检索回来的片段没有结论。备注（notes）里往往才是讲者真正想表达的内容，必须一起进块。

表格里、图片里的文字不属于 slide 正文：
  - 表格另开 table block（markdown，带表头）
  - 图片另开 image block（走 image_parser：OCR + CLIP + zero-shot 分类）
  这样 chunk 层的 chunk_type 才能正确落成 text/table/image/slide 四类。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

from ingestion.classifiers.doc_classifier import classify, merge_into_metadata
from ingestion.parsers.common import (
    compute_doc_id,
    detect_language,
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

# EMU -> pt（1pt = 12700 EMU）。PDF/XLSX 的 bbox 单位不同，
# 这里统一用 pt，检索层只按「同一坐标系内比较」使用，不做跨格式对齐。
_EMU_PER_PT = 12700.0

# 图片类形状：PICTURE 是本地位图，LINKED_PICTURE/MEDIA 往往没有内嵌字节，
# 交给 _picture_blob 返回 b'' 后自然跳过（外链图不能凭空 OCR）
_PICTURE_SHAPE_TYPES = (
    MSO_SHAPE_TYPE.PICTURE,
    MSO_SHAPE_TYPE.LINKED_PICTURE,
    MSO_SHAPE_TYPE.MEDIA,
)

# 段落级别名符号（中文 PPT 里项目符号常被打进文本，需要剥掉）
_BULLET_CHARS = ("•", "·", "‣", "◦", "▪", "- ", "· ")


def _bbox_pt(shape: Any) -> list[float] | None:
    """shape 的位置 -> [x0,y0,x1,y1]（pt）。缺 left/top 的 shape（如占位符继承）返回 None。"""
    try:
        left, top = shape.left, shape.top
        width, height = shape.width, shape.height
        if None in (left, top, width, height):
            return None
        x0, y0 = Emu(left).pt, Emu(top).pt
        x1, y1 = x0 + Emu(width).pt, y0 + Emu(height).pt
        return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
    except (TypeError, ValueError, AttributeError):
        return None


def _clean_bullet(text: str) -> str:
    """
    去掉行首装饰性项目符号（可能嵌套两层，所以循环剥）。
    注意：不用裸 "-" 匹配，否则「-15%」的负号会被当符号删掉，直接改变数值语义。
    """
    stripped = (text or "").strip()
    changed = True
    while changed:
        changed = False
        for mark in _BULLET_CHARS:
            if not stripped.startswith(mark):
                continue
            tail = stripped[len(mark) :].strip()
            if mark == "- " and tail.startswith(("-", "－", "—", ".")):
                continue  # 形如 "- -15%" 才剥外层；"-15%" 不是符号
            stripped = tail
            changed = True
    return stripped


def _paragraph_lines(text_frame: Any) -> list[str]:
    """一个文本框 -> 若干非空行（保留段落顺序，供要点列表用）。"""
    lines: list[str] = []
    for paragraph in text_frame.paragraphs:
        raw = "".join(run.text or "" for run in paragraph.runs) or (paragraph.text or "")
        raw = _clean_bullet(raw)
        if raw:
            lines.append(raw)
    return lines


def _walk_shapes(shapes: Any) -> Iterable[Any]:
    """递归展开组合形状（GROUP）。PPT 里排版稍复杂的页面全是组合，不递归会漏内容。"""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _walk_shapes(shape.shapes)
        else:
            yield shape


def _shape_title(slide: Any) -> str:
    """
    取页标题：优先 placeholders（idx=0 或 type 含 TITLE），
    退化到本页最长的短文本框——很多「标题」其实是手工画的文本框。
    """
    try:
        if slide.shapes.title is not None and normalize_text(slide.shapes.title.text):
            return normalize_text(slide.shapes.title.text)
    except (AttributeError, ValueError):
        pass

    for shape in _walk_shapes(slide.shapes):
        if not shape.has_text_frame:
            continue
        text = normalize_text(shape.text_frame.text)
        if text and len(text) <= 40:
            return text
    return ""


def _table_grid(table: Any) -> list[list[str]]:
    """pptx 表格 -> 二维网格。"""
    grid: list[list[str]] = []
    for row in table.rows:
        grid.append([normalize_text(cell.text or "") for cell in row.cells])
    return grid


def parse_pptx(path: str | Path) -> dict[str, Any]:
    """解析 pptx -> 统一中间 JSON（每页一个 slide block，表格/图片另开块）。"""
    p = Path(path)
    data = p.read_bytes()
    doc_id = compute_doc_id(p, data)

    presentation = Presentation(str(p))
    blocks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"n_images": 0, "n_tables": 0, "model_errors": []}

    for index, slide in enumerate(presentation.slides, start=1):
        title = _shape_title(slide)
        section_path = f"第{index}页" + (f" > {title}" if title else "")

        body_lines: list[str] = [title] if title else []
        notes_text = ""
        # 本页的块单独收集：最后把 slide 块插到本页最前面，
        # 保证「先页面正文、后本页表格/图片」的顺序在全篇稳定（跨页 append 会乱序）
        slide_blocks: list[dict[str, Any]] = []

        for shape in _walk_shapes(slide.shapes):
            # ---- 表格：单独成块 ---------------------------------------------
            if getattr(shape, "has_table", False) and shape.table is not None:
                grid = filter_blank_rows(_table_grid(shape.table))
                if grid:
                    h_idx, header = pick_header_row(grid)
                    stats["n_tables"] += 1
                    slide_blocks.append(
                        new_block(
                            "table",
                            table_to_markdown(grid, header_rows=max(h_idx + 1, 1)),
                            page=index,
                            bbox=_bbox_pt(shape),
                            section_path=section_path,
                            header=header,
                            header_row_index=h_idx,
                            n_rows=len(grid),
                            n_cols=max(len(r) for r in grid),
                        )
                    )
                continue

            # ---- 图片：交给 image_parser ------------------------------------
            if shape.shape_type in _PICTURE_SHAPE_TYPES:
                blob = _picture_blob(shape)
                if not blob:
                    continue
                stats["n_images"] += 1
                caption = _nearby_caption(shape, slide.shapes)
                image_doc = parse_image_bytes(
                    blob,
                    source_format=_picture_format(shape),
                    caption=caption,
                    page=index,
                    bbox=_bbox_pt(shape),
                    section_path=section_path,
                    doc_id=f"{doc_id}:s{index}:img{_slug(shape)}",
                )
                if not image_doc or not image_doc["blocks"]:
                    continue
                block = image_doc["blocks"][0]
                block["doc_id"] = image_doc["doc_id"]
                slide_blocks.append(block)
                for err in image_doc.get("model_errors") or []:
                    if err not in stats["model_errors"]:
                        stats["model_errors"].append(err)
                continue

            # ---- 文本：正文要点（标题已单独收过，这里跳过占位符 0） -----------
            if shape.has_text_frame:
                if title and normalize_text(shape.text_frame.text) == title:
                    continue
                body_lines.extend(_paragraph_lines(shape.text_frame))

            # ---- 内嵌图表：数值抽不成表格，就退化成「图表标题 + 类别」文本 -----
            if getattr(shape, "has_chart", False) and shape.chart is not None:
                chart_text = _chart_text(shape.chart)
                if chart_text:
                    body_lines.append(chart_text)

        # ---- 备注 ----------------------------------------------------------
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes_text = normalize_text(slide.notes_slide.notes_text_frame.text)
        except (AttributeError, ValueError):
            notes_text = ""

        parts = ["\n".join(body_lines)]
        if notes_text:
            parts.append(f"备注：{notes_text}")
        content = "\n".join(x for x in parts if x)

        slide_blocks.insert(
            0,
            new_block(
                "slide",
                content if content else f"第{index}页（无文本内容）",
                page=index,
                section_path=section_path,
                slide_index=index,
                title=title,
                has_notes=bool(notes_text),
            ),
        )
        blocks.extend(slide_blocks)

    full_text = "\n".join(str(b.get("content", "")) for b in blocks)
    metadata = {
        # 与 has_image 一致地用「抽到几个」判断，不用「块有没有正文」：
        # 纯照片页 OCR 空，但这一页确实有图
        "has_table": stats["n_tables"] > 0,
        "has_image": stats["n_images"] > 0,
        "language": detect_language(full_text),
        "layout_style": "slides",
    }
    metadata = merge_into_metadata(metadata, classify(full_text, blocks))

    core = getattr(presentation, "core_properties", None)
    return new_document(
        doc_id=doc_id,
        source_format="pptx",
        metadata=metadata,
        blocks=blocks,
        file_name=p.name,
        n_slides=len(presentation.slides),
        doc_title=getattr(core, "title", None) or None,
        doc_author=getattr(core, "author", None) or None,
        n_images=stats["n_images"],
        parsed_at=utc_now_iso(),
        model_errors=stats["model_errors"] or None,
    )


def _picture_blob(shape: Any) -> bytes:
    """取图片字节。损坏/外链图片（link）返回 b''，由调用方跳过——外链图没有本地字节。"""
    try:
        image = shape.image
    except (AttributeError, ValueError, KeyError):
        return b""
    return getattr(image, "blob", b"") or b""


def _picture_format(shape: Any) -> str:
    """由 content_type 决定归一化 source_format（统一只承认 jpg / png）。"""
    try:
        ctype = str(shape.image.content_type or "").lower()
    except (AttributeError, ValueError, KeyError):
        return "jpg"
    return "png" if "png" in ctype else "jpg"


def _slug(shape: Any) -> str:
    """
    shape 的稳定标识，用来拼子图片 doc_id。
    同一页两张图如果共用一个 doc_id，入库时后者会覆盖前者（Qdrant 按 point id upsert）。
    """
    try:
        return str(shape.shape_id)
    except (AttributeError, TypeError):
        return "0"


def _nearby_caption(shape: Any, slide_shapes: Any) -> str:
    """
    图注猜测：PPT 里图注通常是与图片水平重叠、且紧贴图片下方的文本框。
    找不到就返回空串——宁可没有 caption，也不要拿毫不相关的文本当图注，
    错误 caption 会被 OCR/CLIP 当成图片语义，比缺失更难查。
    """
    box = _bbox_pt(shape)
    if not box:
        return ""
    best: tuple[float, str] = (1e9, "")
    for other in _walk_shapes(slide_shapes):
        if other is shape or not getattr(other, "has_text_frame", False):
            continue
        ob = _bbox_pt(other)
        if not ob:
            continue
        horizontal_overlap = min(box[2], ob[2]) - max(box[0], ob[0])
        below_gap = ob[1] - box[3]
        if horizontal_overlap <= 0 or below_gap < -5 or below_gap > 60:
            continue
        text = normalize_text(other.text_frame.text)
        if not text or len(text) > 60:
            continue
        distance = abs(below_gap)
        if distance < best[0]:
            best = (distance, text)
    return best[1]


def _chart_text(chart: Any) -> str:
    """
    PPT 内嵌图表：把系列名 + 类别 + 数值抽成一行文本。
    图表本质是表格，这里只保留结构化摘要，具体数值精度交给 OCR 图片路径补充。
    """
    try:
        cats = [str(c) for c in chart.plots[0].categories]
        series_names = [s.name for s in chart.series]
        head = "图表：" + "、".join(n for n in series_names if n)
        tail = "类别：" + "、".join(cats[:20])
        return normalize_text(f"{head}；{tail}")
    except (IndexError, AttributeError, TypeError):
        return ""


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="解析 pptx，输出统一中间 JSON")
    cli.add_argument("path", help="pptx 文件路径")
    args = cli.parse_args()
    print(dump_json_for_cli(parse_pptx(args.path)))
