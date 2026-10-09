# -*- coding: utf-8 -*-
"""
PDF 解析：PyMuPDF（文本块 + bbox + 字体字号）+ pdfplumber（表格结构）+ 双路分流。

三条硬规矩：
  1. 表格走 pdfplumber.extract_tables() 转 markdown，绝不混进正文文本块。
     PyMuPDF 的 text dict 会把表格里的每个单元格当成独立 text block 吐出来，
     直接入库会得到一堆「2024」「1,234.00」这种无上下文碎片，
     所以先取表格矩形，再把中心落在矩形内的文本块丢弃（见 _drop_blocks_in_tables）。
  2. 扫描件分流：page.get_text() 字符数 < 阈值 且 页面有图 -> 整页渲染成位图走 PaddleOCR；
     否则走文本层。判定与执行分离（is_scanned_page / ocr_page），方便单测和调阈值。
  3. 中文标题判定独立成 score_title / is_title 两个纯函数（不依赖 fitz，可直接单测）。
     判定口径是三条信号的「或」：字号 > 正文基准×1.3，或 字体在黑体/雅黑集合，
     或 居中 + 短文本（<30 字）。
     中文 PDF 普遍存在「假加粗」：导出时字体名仍是 SimSun，视觉上却描边加粗，
     所以字号证据最可靠（权重 0.6），字体只作辅助（0.25），居中短文本再作弱证据（0.15）；
     权重只用于层级提示与排序，不再用来卡「是否标题」的总分阈值。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

from ingestion.classifiers.doc_classifier import classify, merge_into_metadata
from ingestion.parsers.common import (
    TITLE_FONT_HINTS,
    HeadingStack,
    bbox_inside,
    bbox_to_list,
    compute_doc_id,
    detect_language,
    document_root_title,
    dump_json_for_cli,
    horizontal_center_ratio,
    new_block,
    new_document,
    normalize_text,
    parsing_option,
    pick_header_row,
    table_to_markdown,
    utc_now_iso,
)
from ingestion.parsers.image_parser import min_side_ok, parse_image_bytes

# --------------------------------------------------------------- 阈值（可测）
SCANNED_PAGE_TEXT_THRESHOLD = 50     # 文本层字符数低于此值 + 有图 -> 判扫描件
TITLE_LEN_LIMIT = 30                 # 「居中/黑体 + 短文本」这类证据的长度上限（字）
MIN_TITLE_CHARS = 2
MAX_TITLE_CHARS = 60
BODY_SIZE_BINS = 0.5                 # 字号众数统计的粒度（半磅）
NOISE_BLOCK_MAX_CHARS = 12           # 页脚/页码一类碎片的上限
PAGE_MARGIN_RATIO = 0.06             # 距页面上下边 <6% 视为页眉页脚带
# 标题三条「或」信号的阈值
TITLE_SIZE_RATIO = 1.30              # 信号一：字号 > 正文基准 × 1.3
TITLE_FONT_SIZE_RATIO = 1.05         # 信号二：黑体/雅黑 时字号至少要不低于正文
TITLE_CENTER_SIZE_RATIO = 0.95       # 信号三：居中短文本 时允许的最小字号比
CENTER_TOLERANCE = 0.08              # 块中心偏离页面中线 <8% 视为居中
# 打分权重：字号 > 字体 > 居中长。中文假加粗使字体不可靠，所以字号权重最高。
# 权重只影响 score（用于排序与层级提示），不再决定「是不是标题」——见 score_title。
TITLE_WEIGHTS = {"size": 0.6, "font": 0.25, "center_short": 0.15}
TITLE_SCORE_THRESHOLD = 0.50         # 遗留：仅 is_title(threshold=...) 做 A/B 对比时用

# 页码/目录点线这类噪声
_PAGE_NUM_RE = re.compile(r"^[\s\-—\u3000]*\d{1,4}[\s\-—\u3000]*$")
_DOTTED_TOC_RE = re.compile(r"[.…·]{4,}")


@dataclass
class SpanStat:
    """一个文本块聚合并后的证据：文本、主字体、主字号、bbox、页码。"""

    text: str
    font: str
    size: float
    bbox: list[float]
    page: int
    lines: list[str] = field(default_factory=list)


# =============================================================== 标题判定
def size_ratio(size: float, body_size: float | None) -> float:
    """字号相对正文基准的倍数（body_size 未知时按 1.0 处理）。"""
    if not body_size or body_size <= 0:
        return 1.0
    return size / body_size


def is_title_font(font: str | None) -> bool:
    """字体名是否落在黑体/雅黑/宋体粗体这类「标题常用字库」里。"""
    if not font:
        return False
    lowered = font.lower()
    return any(hint in lowered for hint in TITLE_FONT_HINTS)


def score_title(
    text: str,
    size: float,
    font: str | None,
    bbox: list[float],
    page_width: float,
    body_size: float | None,
) -> dict[str, Any]:
    """
    标题打分（纯函数，独立可测）。

    **判定口径是三条信号的「或」，不是加权和过阈值**（对应需求：
    字号 > 正文均值×1.3，或 字体在黑体/雅黑集合，或 居中 + 短文本）。
    加权版会把「14pt SimHei 短标题」这种只有两条弱证据的情况判漏，
    所以这里 signals 决定是不是标题，score 只用来排序和给层级。

    返回：
      signals : {"size": bool, "font": bool, "center_short": bool}
      parts   : 三条信号的连续值（0~1），写进 block 当证据，便于调阈值时回看
      score   : 加权和（字号 0.6 / 字体 0.25 / 居中短文本 0.15）
      hit     : 任一信号命中 -> 判为标题
      level_hint : 字号档位给的层级（1 章 / 2 节 / 3 小节）
      rejected   : 被硬性排除的原因（"" 表示没被排除）
    """
    stripped = normalize_text(text)
    length = len(stripped)
    verdict: dict[str, Any] = {
        "score": 0.0,
        "ratio": round(size_ratio(size, body_size), 3),
        "signals": {"size": False, "font": False, "center_short": False},
        "parts": {"size": 0.0, "font": 0.0, "center_short": 0.0},
        "hit": False,
        "level_hint": 3,
        "rejected": "",
    }
    if length < MIN_TITLE_CHARS or length > MAX_TITLE_CHARS:
        verdict["rejected"] = "length"
        return verdict
    # 以句末标点结尾的一定是句子，不是标题
    if stripped.endswith(("。", "！", "？", ".", "!", "?", "；", ";")):
        verdict["rejected"] = "sentence_end"
        return verdict

    ratio = size_ratio(size, body_size)
    single_line = "\n" not in stripped

    # 信号一：字号。需求口径「> 正文均值×1.3」，用众数基准替代均值（见 estimate_body_size）
    size_ok = body_size is not None and body_size > 0 and ratio > TITLE_SIZE_RATIO
    # 信号二：字体。黑体/雅黑 且 字号不低于正文（正文里也常有黑体小标签，
    # 靠 ratio >= 1.05 + 单行 + 短文本 三条约束挡住）
    font_ok = (
        is_title_font(font)
        and single_line
        and length < TITLE_LEN_LIMIT
        and ratio >= TITLE_FONT_SIZE_RATIO
    )
    # 信号三：居中 + 短文本。同样要求字号不明显小于正文，否则页脚居中的页码也会命中
    center_ratio = horizontal_center_ratio(bbox, page_width)
    center_ok = (
        center_ratio < CENTER_TOLERANCE
        and single_line
        and length < TITLE_LEN_LIMIT
        and ratio >= TITLE_CENTER_SIZE_RATIO
    )

    verdict["signals"] = {"size": size_ok, "font": font_ok, "center_short": center_ok}
    verdict["parts"] = {
        "size": round(max(0.0, min(1.0, (ratio - 1.1) / 0.5)), 3),
        "font": 1.0 if is_title_font(font) else 0.0,
        "center_short": 1.0 if (center_ratio < CENTER_TOLERANCE and length < TITLE_LEN_LIMIT) else 0.0,
    }
    verdict["score"] = round(
        TITLE_WEIGHTS["size"] * verdict["parts"]["size"]
        + TITLE_WEIGHTS["font"] * verdict["parts"]["font"]
        + TITLE_WEIGHTS["center_short"] * verdict["parts"]["center_short"],
        4,
    )
    verdict["hit"] = bool(size_ok or font_ok or center_ok)

    # 层级：字号是唯一稳定的层级证据；只有字体/居中证据时统一给最深层
    if ratio >= 1.8:
        verdict["level_hint"] = 1
    elif ratio >= 1.45:
        verdict["level_hint"] = 2
    elif ratio >= TITLE_SIZE_RATIO:
        verdict["level_hint"] = 3
    else:
        verdict["level_hint"] = 3
    return verdict


def title_signals_hit(verdict: dict[str, Any]) -> bool:
    """读打分结果的「或」判定结果（parse_pdf 里复用同一个 verdict，避免二次打分）。"""
    return bool(verdict.get("hit"))


def is_title(
    text: str,
    size: float,
    font: str | None,
    bbox: list[float],
    page_width: float,
    body_size: float | None,
    threshold: float | None = None,
) -> bool:
    """
    标题判定入口（对外承诺的独立可测函数）。

    threshold 是历史遗留参数：早期版本想用加权和阈值判定，实测会漏掉
    「黑体 + 1.27 倍字号」的中文小标题，现在改为三条信号取或；
    传入 threshold 时退化为「加权和 >= threshold」，只为做 A/B 对比保留。
    """
    verdict = score_title(text, size, font, bbox, page_width, body_size)
    if threshold is None:
        return verdict["hit"]
    return verdict["score"] >= threshold


# =============================================================== 扫描件分流
def page_text_length(page: fitz.Page) -> int:
    """页面文本层字符数（去掉空白）。"""
    return len((page.get_text("text") or "").strip())


def is_scanned_page(page: fitz.Page, threshold: int = SCANNED_PAGE_TEXT_THRESHOLD) -> bool:
    """
    扫描件判定：文本层几乎为空（<50 字）但页面有图像 -> 需要 OCR。
    只看字符数不看块数，是因为扫描件的「文本层」常常是 OCR 残留的水印一行。
    """
    has_image = bool(page.get_images(full=True)) or bool(page.get_image_info())
    return page_text_length(page) < threshold and has_image


def ocr_page(page: fitz.Page, dpi: int | None = None) -> tuple[str, list[dict[str, Any]], str | None]:
    """
    整页渲染成位图后走 OCR。

    返回 (拼平文本, 逐行结果, 错误)。
    dpi 默认取配置 parsing.ocr.dpi（200）；低于 150 中文小字识别率会明显下降。
    """
    dpi = int(dpi or parsing_option("ocr.dpi", 200))
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image_bytes = pix.tobytes("png")
    from ingestion.parsers.image_parser import ocr_image_bytes

    text, lines, error = ocr_image_bytes(image_bytes)
    # OCR 行的 bbox 是像素坐标，换算回 PDF 点坐标，保持与文本块同一坐标系
    scale = 72.0 / dpi
    for line in lines:
        if line.get("bbox"):
            line["bbox"] = [round(v * scale, 1) for v in line["bbox"]]
    return text, lines, error


# =============================================================== 文本块聚合
def collect_page_blocks(page: fitz.Page, page_index: int) -> list[SpanStat]:
    """
    PyMuPDF text dict -> SpanStat 列表（一个 block 一条，字体/字号按字符数加权取主值）。

    不做跨块合并：版面信息留给 section_path 与检索层，这里只保证「块 = 版面段落」。
    """
    raw = page.get_text("dict")
    stats: list[SpanStat] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        font_votes: dict[str, int] = {}
        size_votes: dict[float, int] = {}
        for line in block.get("lines", []):
            chunks = [span.get("text", "") for span in line.get("spans", [])]
            line_text = "".join(chunks)
            if not line_text.strip():
                continue
            lines.append(line_text.strip())
            for span in line.get("spans", []):
                weight = max(len(span.get("text", "")), 1)
                fname = span.get("font", "") or ""
                font_votes[fname] = font_votes.get(fname, 0) + weight
                size = round(float(span.get("size", 0.0)) / BODY_SIZE_BINS) * BODY_SIZE_BINS
                size_votes[size] = size_votes.get(size, 0) + weight
        text = "\n".join(lines)
        if not text.strip():
            continue
        font = max(font_votes, key=lambda k: font_votes[k]) if font_votes else ""
        size = max(size_votes, key=lambda k: size_votes[k]) if size_votes else 0.0
        stats.append(
            SpanStat(
                text=text,
                font=font,
                size=round(size, 2),
                bbox=bbox_to_list(block.get("bbox")) or [0, 0, 0, 0],
                page=page_index,
                lines=lines,
            )
        )
    return stats


def estimate_body_size(all_stats: list[SpanStat]) -> float | None:
    """
    全文正文基准字号 = 按字符数加权的众数字号。
    用众数而非均值：几个 28pt 大标题就能把均值抬上去，导致正文全被判成标题。
    """
    votes: dict[float, int] = {}
    for stat in all_stats:
        votes[stat.size] = votes.get(stat.size, 0) + len(stat.text)
    if not votes:
        return None
    return max(votes, key=lambda k: votes[k])


def is_noise_block(stat: SpanStat, page_height: float, body_size: float | None) -> bool:
    """
    页码 / 页眉页脚 / 目录点线 噪声过滤。
    条件：位于上下边带内 + (文本极短 或 纯数字 或 带点线) + 字号 <= 正文。
    只在这几项同时成立时才丢，宁可留噪声也不要丢掉正文里的短行。
    """
    text = stat.text.strip()
    if not text:
        return True
    top = stat.bbox[1]
    bottom = stat.bbox[3]
    margin = page_height * PAGE_MARGIN_RATIO
    in_edge_band = top <= margin or bottom >= page_height - margin
    if not in_edge_band:
        return False
    short = len(text) <= NOISE_BLOCK_MAX_CHARS or _PAGE_NUM_RE.match(text)
    dotted = bool(_DOTTED_TOC_RE.search(text)) and len(text) <= 60
    small = body_size is None or stat.size <= body_size + 0.5
    return bool((short or dotted) and small)


def drop_blocks_in_tables(stats: list[SpanStat], table_rects: list[list[float]]) -> list[SpanStat]:
    """丢掉中心落在表格矩形内的文本块（表格内容已由 pdfplumber 结构化抽出，避免重复）。"""
    if not table_rects:
        return stats
    kept: list[SpanStat] = []
    for stat in stats:
        if any(bbox_inside(stat.bbox, rect) for rect in table_rects):
            continue
        kept.append(stat)
    return kept


# =============================================================== 表格 / 图片
def extract_tables(plumber_page: Any) -> list[dict[str, Any]]:
    """
    pdfplumber 抽表格，返回 [{"markdown","bbox","header","grid"}]。
    策略由配置 parsing.pdf.table_strategy 决定（lines / text / explicit），
    lines 对带框线中文表格最稳；无边框表格需改 text 才抽得出。
    """
    strategy = parsing_option("pdf.table_strategy", "lines")
    out: list[dict[str, Any]] = []
    try:
        found = plumber_page.find_tables(
            table_settings={"vertical_strategy": strategy, "horizontal_strategy": strategy}
        )
    except Exception:
        found = []
    for table in found:
        try:
            grid = table.extract()
            rect = list(table.bbox)
        except Exception:
            continue
        rows = [list(r) for r in (grid or []) if r and any(c for c in r)]
        if len(rows) < 2:
            continue  # 只有一行的多是装饰线
        # 表头行可能被「跨列合并的表题」占据（pdfplumber 会给出 ["表题", None, None]），
        # 直接取 rows[0] 就当表头会丢列名，交给 pick_header_row 选
        h_idx, header = pick_header_row(rows)
        out.append(
            {
                "markdown": table_to_markdown(rows, header_rows=max(h_idx + 1, 1)),
                "bbox": [round(v, 1) for v in rect],
                "header": header,
                "header_row_index": h_idx,
                "n_rows": len(rows),
                "n_cols": max(len(r) for r in rows),
            }
        )
    return out


def extract_images(doc: fitz.Document, page: fitz.Page, page_index: int) -> list[dict[str, Any]]:
    """
    页面内嵌图片 -> [{bytes, bbox, xref, ext}]。
    用 get_image_info 拿渲染位置（bbox），比 get_images 的 xref 列表更准，
    同一张图被缩放/平铺时也能各自拿到自己的矩形。
    """
    results: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()
    for info in page.get_image_info(xrefs=True):
        xref = int(info.get("xref") or 0)
        if xref <= 0 or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue
        if not base or not base.get("image"):
            continue
        results.append(
            {
                "xref": xref,
                "bytes": base["image"],
                "ext": (base.get("ext") or "jpg").lower(),
                "bbox": bbox_to_list(info.get("bbox")) or [0, 0, 0, 0],
                "page": page_index,
            }
        )
    return results


# =============================================================== 主流程
def parse_pdf(path: str | Path) -> dict[str, Any]:
    """
    解析 PDF -> 统一中间 JSON。

    两遍扫描：第一遍收全文文本块以估计正文基准字号（标题判定必须有基准），
    第二遍才做标题/正文/表格/图片分流。代价是多一次 get_text，
    但拿不到基准字号时中文 PDF 的标题几乎全漏，值得。
    """
    p = Path(path)
    data = p.read_bytes()
    doc_id = compute_doc_id(p, data)

    doc = fitz.open(stream=data, filetype="pdf")
    model_errors: list[str] = []
    scanned_pages: list[int] = []
    page_stats: list[list[SpanStat]] = []

    try:
        # ---- 第一遍：全文文本块 + 基准字号 ---------------------------------
        for page_index in range(doc.page_count):
            page_stats.append(collect_page_blocks(doc[page_index], page_index + 1))
        all_stats = [s for page in page_stats for s in page]
        body_size = estimate_body_size(all_stats)

        # ---- 第二遍：分流 + 结构化 -----------------------------------------
        with pdfplumber.open(str(p)) as plumber_doc:
            # 单一标题栈按页序推进：章节标题通常只出现一次，后续页正文要继续沿用它，
            # 所以绝不能「每页一个栈」；根用文档属性标题（没有就退化成文件名），
            # 这样第 20 页的一个碎片块也能带上「哪份文件」的信息
            stack = HeadingStack()
            stack.seed_root(document_root_title((doc.metadata or {}).get("title"), p))
            blocks: list[dict[str, Any]] = []

            for page_index in range(doc.page_count):
                page = doc[page_index]
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
                plumb_page = (
                    plumber_doc.pages[page_index] if page_index < len(plumber_doc.pages) else None
                )

                # 表格先抽，拿到矩形后才能剔除区域内的重复文本
                tables = extract_tables(plumb_page) if plumb_page else []
                table_rects = [t["bbox"] for t in tables]

                # ---- 扫描件分支 --------------------------------------------
                if is_scanned_page(page):
                    scanned_pages.append(page_index + 1)
                    text, lines, error = ocr_page(page)
                    if error:
                        model_errors.append(f"page{page_index + 1}: {error}")
                    if text:
                        blocks.append(
                            new_block(
                                "text",
                                text,
                                page=page_index + 1,
                                bbox=[0, 0, page_width, page_height],
                                section_path=stack.path,
                                from_ocr=True,
                                ocr_lines=lines,
                            )
                        )
                    blocks.extend(_table_blocks(tables, page_index + 1, stack.path))
                    # 扫描件整页就是一张大图：整页 OCR 已经覆盖了图里的内容，
                    # 再为「页内图片」开一个 image block 会得到 90% 重复的文本，
                    # 只有整页 OCR 没出字（模型不可用 / 纯图片页）才补图片块
                    if not text:
                        blocks.extend(
                            _image_blocks(
                                doc, page, page_index, data, doc_id, stack.path, model_errors
                            )
                        )
                    continue

                # ---- 文本层分支 ---------------------------------------------
                stats = page_stats[page_index]
                stats = drop_blocks_in_tables(stats, table_rects)
                stats = [
                    s for s in stats if not is_noise_block(s, page_height, body_size)
                ]
                for stat in stats:
                    verdict = score_title(
                        stat.text, stat.size, stat.font, stat.bbox, page_width, body_size
                    )
                    if title_signals_hit(verdict):
                        title = normalize_text(stat.text.replace("\n", " "))
                        stack.push(verdict["level_hint"], title)
                        blocks.append(
                            new_block(
                                "title",
                                title,
                                page=page_index + 1,
                                bbox=stat.bbox,
                                font=stat.font,
                                size=stat.size,
                                section_path=stack.path,
                                title_score=verdict["score"],
                                title_signals=verdict["signals"],
                                title_level=verdict["level_hint"],
                            )
                        )
                        continue
                    # 正文里可能夹着「本章小结」这类短行，交给 chunk 层，这里不再拆
                    blocks.append(
                        new_block(
                            "text",
                            stat.text,
                            page=page_index + 1,
                            bbox=stat.bbox,
                            font=stat.font,
                            size=stat.size,
                            section_path=stack.path,
                        )
                    )

                blocks.extend(_table_blocks(tables, page_index + 1, stack.path))
                blocks.extend(
                    _image_blocks(doc, page, page_index, data, doc_id, stack.path, model_errors)
                )

        # 栈已在页序推进中实时写入 section_path，这里只需拼接全文本做分类
        full_text = "\n".join(str(b.get("content", "")) for b in blocks)

        metadata = {
            "has_table": any(b["type"] == "table" for b in blocks),
            "has_image": any(b["type"] == "image" for b in blocks),
            "language": detect_language(full_text),
            "layout_style": "prose",
        }
        metadata = merge_into_metadata(metadata, classify(full_text, blocks))

        toc = doc.get_toc(simple=True)
        return new_document(
            doc_id=doc_id,
            source_format="pdf",
            metadata=metadata,
            blocks=blocks,
            file_name=p.name,
            n_pages=doc.page_count,
            n_images=sum(1 for b in blocks if b["type"] == "image"),
            body_font_size=body_size,
            pdf_title=(doc.metadata or {}).get("title") or None,
            # 书签里通常已有权威章节名，检索/评估阶段可直接拿来对齐 section_path
            # 上限 200 条：超长书签多是网页打印件，写进中间 JSON 只会膨胀
            outline=list(toc)[:200],
            scanned_pages=scanned_pages,
            parsed_at=utc_now_iso(),
            model_errors=model_errors or None,
        )
    finally:
        doc.close()


def _table_blocks(tables: list[dict[str, Any]], page_no: int, section_path: str) -> list[dict[str, Any]]:
    """表格 dict -> table block。"""
    return [
        new_block(
            "table",
            t["markdown"],
            page=page_no,
            bbox=t["bbox"],
            section_path=section_path,
            header=t["header"],
            header_row_index=t.get("header_row_index", 0),
            n_rows=t["n_rows"],
            n_cols=t["n_cols"],
        )
        for t in tables
        if t.get("markdown")
    ]


def _image_blocks(
    doc: fitz.Document,
    page: fitz.Page,
    page_index: int,
    data: bytes,
    doc_id: str,
    section_path: str,
    model_errors: list[str],
) -> list[dict[str, Any]]:
    """页面图片 -> image block（OCR + CLIP 由 image_parser 统一处理）。"""
    out: list[dict[str, Any]] = []
    for item in extract_images(doc, page, page_index):
        if not min_side_ok(item["bytes"]):
            continue
        image_doc = parse_image_bytes(
            item["bytes"],
            source_format="png" if item["ext"] == "png" else "jpg",
            caption="",
            page=page_index + 1,
            bbox=item["bbox"],
            section_path=section_path,
            doc_id=f"{doc_id}:p{page_index + 1}:x{item['xref']}",
        )
        if not image_doc or not image_doc["blocks"]:
            continue
        block = image_doc["blocks"][0]
        block["doc_id"] = image_doc["doc_id"]
        out.append(block)
        for err in image_doc.get("model_errors") or []:
            if err not in model_errors:
                model_errors.append(err)
    return out


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="解析 pdf，输出统一中间 JSON")
    cli.add_argument("path", help="PDF 文件路径")
    args = cli.parse_args()
    print(dump_json_for_cli(parse_pdf(args.path)))
