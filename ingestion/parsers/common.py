# -*- coding: utf-8 -*-
"""
解析层公共工具（被本批 6 个 parser 共用）。

为什么单独一个文件：表格转 markdown、语言判定、doc_id 计算、标题栈这几件事
在 pdf/docx/pptx/xlsx 四个 parser 里完全一样，复制 4 份必然走样。

约定：
  - 所有 parser 返回的 dict 必须调用 new_document() / new_block() 构造，
    保证统一 schema（缺字段直接报错，不静默补 None）
  - 本模块不做任何模型加载，纯函数 + 轻量状态对象
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 统一枚举
SOURCE_FORMATS = ("pdf", "docx", "pptx", "xlsx", "txt", "jpg", "png")
BLOCK_TYPES = ("text", "title", "table", "image", "slide", "sheet")
DOC_TYPES = ("contract", "report", "manual", "invoice", "unknown")
TOPICS = ("finance", "legal", "technical", "unknown")
LANGUAGES = ("zh", "en", "mixed")
LAYOUT_STYLES = ("form", "prose", "slides", "sheet", "image")

# 中日韩统一表意文字（含扩展 A）+ 兼容区
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# 全角标点 / 空白归一化时保留，英文单词边界用
_WHITESPACE_RE = re.compile(r"[ \t\u3000]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# 中文标题判定里视作「标题字体」的字体名片段（小写比较，兼容 PDF 里的各种写法）。
# 注意：**绝对不要把 宋体/STSong/SimSun 放进来**。标题判定用的是「或」逻辑，
# 而宋体是中文正文的默认字体，一旦入表，整篇正文的短行都会被判成标题。
# 这里只收「正文不会用、标题才用」的黑体 / 雅黑类字库。
TITLE_FONT_HINTS = (
    "simhei",
    "heiti",
    "hei",
    "microsoftyahei",
    "yahei",
    "stheiti",
    "pingfang",
    "sourcehansans",
    "黑体",
    "雅黑",
    "苹方",
    "思源黑体",
)


# =================================================================== 基础工具
def sha256_hex(data: bytes) -> str:
    """字节内容 sha256，作为 doc_id。"""
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: str | Path) -> bytes:
    """整文件读入字节（100GB 库里的单文档通常 <50MB，可整读；超大 PDF 后续批次改流式哈希）。"""
    return Path(path).read_bytes()


def compute_doc_id(path: str | Path, data: bytes | None = None) -> str:
    """
    doc_id = sha256(file_bytes)。
    注意：内容寻址，同一份文件重复入库天然幂等（Qdrant 侧 chunk_id 也由它派生）。
    """
    return sha256_hex(data if data is not None else read_bytes(path))


def normalize_text(text: str) -> str:
    """
    文本清洗：压平行内多余空白、去 CRLF、压缩 3 个以上连续换行。
    注意：含中文的文本不做 NFKC —— NFKC 会把全角逗号/句号（U+FF0C/U+3002）
    转成半角，破坏中文标点，进而影响 jieba 分词与 FTS5 精确命中。
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not has_cjk(text):
        text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def has_cjk(text: str) -> bool:
    """是否含中文字符。"""
    return bool(_CJK_RE.search(text or ""))


def cjk_ratio(text: str) -> float:
    """
    中文字符占「有效字符」的比例。
    分母排除空白，否则大量缩进/空行会把比例压到 0，导致英文误判。
    """
    if not text:
        return 0.0
    effective = [ch for ch in text if not ch.isspace()]
    if not effective:
        return 0.0
    cjk = sum(1 for ch in effective if _CJK_RE.match(ch))
    return cjk / len(effective)


def detect_language(text: str, zh_threshold: float = 0.30, en_threshold: float = 0.05) -> str:
    """
    language 判定（本批统一口径）：
      中文占比 > 30%          -> zh
      中文占比 < 5%           -> en
      5% <= 中文占比 <= 30%   -> mixed
    空文本按 en 处理（无中文证据）。
    """
    ratio = cjk_ratio(text)
    if ratio > zh_threshold:
        return "zh"
    if ratio < en_threshold:
        return "en"
    return "mixed"


# =================================================================== 表格处理
def _cell_to_str(value: Any) -> str:
    """单元格标准化：None -> 空串；float 整数化；换行 -> <br>（markdown 表格不能换行）。"""
    if value is None:
        return ""
    if isinstance(value, float) and float(value).is_integer():
        value = int(value)
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("|", "\\|")          # 转义竖线，防止撕裂表格
    text = text.replace("\n", "<br>")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    # 单元格里的连续空白行也压掉
    return _MULTI_BLANK_RE.sub("<br>", text)


def is_blank_row(row: Sequence[Any]) -> bool:
    """整行是否为空（None / 空串 / 纯空白 / 只有 0 之外的无意义字符）。"""
    return all(_cell_to_str(v) == "" for v in row)


def filter_blank_rows(rows: Iterable[Sequence[Any]]) -> list[list[Any]]:
    """过滤空行并统一成 list[list]。"""
    return [list(r) for r in rows if not is_blank_row(r)]


def table_to_markdown(
    rows: Iterable[Sequence[Any]],
    header_rows: int = 1,
    repeat_header: bool = True,
    max_rows: int | None = None,
) -> str:
    """
    表格 -> markdown（表格绝不能当普通文本切，必须保结构）。

    参数：
      header_rows : 前 N 行当表头（生成 |---| 分隔线的位置）
      repeat_header: 超长表格被 max_rows 切成多段时，是否每段重复表头。
                     默认 True —— 表头缺失的表格分片对检索是有害的（列名丢了，
                     数字就变成了无意义 token），检索批次不要关掉它。
      max_rows    : 单块最大行数（含表头），超出则切成多段 markdown
    返回：markdown 字符串；空表返回 ""。
    """
    grid = [list(r) for r in rows if r is not None]
    if not grid:
        return ""
    # 列数对齐（pdfplumber 常见末列缺失）
    width = max(len(r) for r in grid)
    grid = [list(r) + [""] * (width - len(r)) for r in grid]
    grid = [[_cell_to_str(c) for c in row] for row in grid]

    data_rows = [r for r in grid if not all(c == "" for c in r)]
    if not data_rows:
        return ""

    header_rows = max(0, min(header_rows, len(data_rows) - 1))
    headers = data_rows[:header_rows] or [data_rows[0]]
    body = data_rows[header_rows:] if header_rows else data_rows[1:]
    if not headers:
        headers = [[""] * width]

    sep_cols = len(headers[0])
    header_block = ["| " + " | ".join(headers[0]) + " |", "|" + "---|" * sep_cols]
    for extra in headers[1:]:
        header_block.insert(1, "| " + " | ".join(extra) + " |")

    def _render(chunk: list[list[str]]) -> list[str]:
        lines = list(header_block)
        lines += ["| " + " | ".join(row) + " |" for row in chunk]
        return lines

    if max_rows is None or len(body) <= max_rows:
        return "\n".join(_render(body))

    # 超长表格分段：默认每段重复表头，保证任一分片都能独立被检索到
    per = max(1, max_rows - len(headers))
    segments: list[list[str]] = []
    for start in range(0, len(body), per):
        chunk = body[start : start + per]
        if repeat_header or start == 0:
            segments.extend(_render(chunk))
        else:
            segments.extend(["| " + " | ".join(row) + " |" for row in chunk])
        segments.append("")  # 段间空行
    return "\n".join(segments).strip()


def pick_header_row(grid: Sequence[Sequence[str]], max_scan: int = 3) -> tuple[int, list[str]]:
    """
    从网格里挑出「真正的列名行」，返回 (行下标, 表头)。

    为什么需要：中文表格第一行经常是一条合并出来的表题横幅
    （"2024年度费用明细表" 横跨全部列），拿它当表头会得到
    ["2024年度费用明细表", "", "", ""] —— 列名全丢，
    这张表在 sparse/dense 检索里就退化成一堆无上下文数字。
    规则：从上往下找第一行「非空单元格数 >= 列数一半 且 >= 2」的行；
    都不满足就退回第 0 行（保持原行为，不做更激进的重排）。
    """
    if not grid:
        return 0, []
    width = max(len(r) for r in grid) or 1
    for idx, row in enumerate(list(grid)[:max_scan]):
        non_empty = sum(1 for cell in row if str(cell or "").strip())
        if non_empty >= max(2, math.ceil(width / 2)):
            return idx, [normalize_text(str(c or "")) for c in row]
    return 0, [normalize_text(str(c or "")) for c in grid[0]]


# =================================================================== 几何工具
def bbox_to_list(bbox: Any) -> list[float] | None:
    """把 fitz.Rect / pdfplumber tuple / 任意 4 元序列统一成 [x0,y0,x1,y1]。"""
    if bbox is None:
        return None
    try:
        if hasattr(bbox, "x0"):
            return [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
        seq = list(bbox)
        if len(seq) == 4:
            x0, y0, x1, y1 = (float(v) for v in seq)
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    except (TypeError, ValueError):
        return None
    return None


def bbox_inside(inner: Sequence[float], outer: Sequence[float], tol: float = 2.0) -> bool:
    """inner 的中心点是否落在 outer 内（带容差）。用于「表格区域内的文本块不重复入库」。"""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return (outer[0] - tol) <= cx <= (outer[2] + tol) and (outer[1] - tol) <= cy <= (outer[3] + tol)


def horizontal_center_ratio(bbox: Sequence[float], page_width: float) -> float:
    """块中心偏离页面中线的比例，0 表示完全居中（居中文本大概率是标题）。"""
    if page_width <= 0:
        return 1.0
    cx = (bbox[0] + bbox[2]) / 2
    return abs(cx - page_width / 2) / page_width


# =================================================================== 标题栈
class HeadingStack:
    """
    维护「章节标题 -> section_path」的层级栈。

    level = 0 是文档根（由 parser 用文档属性标题或文件名播种），
    1 为章、2 为节，数字越大越深。pdf_parser 用字号档位映射 level，
    docx_parser 直接用 Heading N 的 N。
    """

    def __init__(self, max_depth: int = 5) -> None:
        self.max_depth = max_depth
        self._stack: list[tuple[int, str]] = []

    def seed_root(self, title: str) -> None:
        """播种文档根；空标题（很多 PDF 没有 metadata title）什么都不做。"""
        self.push(0, title)

    def push(self, level: int, title: str) -> None:
        """
        level 越小越高层：0 是「文档根」（文件名或文档属性标题），1 是章，2 是节。
        允许 0 级是为了给栈播种一个根，跨章内容永远带上文档名，
        检索结果展示时不会只剩一句「第 3 章」。
        """
        level = max(0, int(level))
        title = (title or "").strip()
        if not title:
            return
        # 与父级同名的标题不重复入栈（文档属性标题和正文第一个大标题往往一模一样，
        # 都拼进去会得到「产品手册 > 产品手册 > 第1章」这种脏路径）
        if self._stack and self._stack[-1][1] == title:
            return
        # 弹出所有 >= level 的旧标题（父级保留）
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))
        while len(self._stack) > self.max_depth:
            self._stack.pop(0)

    @property
    def path(self) -> str:
        """形如 "第3章 > 3.2 付款条款"；无标题时返回空串。"""
        return " > ".join(t for _, t in self._stack)


def document_root_title(meta_title: Any, path: str | Path) -> str:
    """
    文档根标题：优先文档属性里的标题，退化成文件名去扩展名。

    中文实务里大量扫描件/导出件的 core properties title 是空的或是制作软件名
    （"WPS 文字""Microsoft Word"），这种字符串当章节前缀毫无检索价值，
    所以做一次黑名单过滤。
    """
    candidate = normalize_text(str(meta_title or ""))
    junk = ("microsoft word", "wps", "python-docx", "powerpoint", "excel", "未命名", "document")
    if candidate and not any(j in candidate.lower() for j in junk) and len(candidate) <= 60:
        return candidate
    return Path(path).stem


# =================================================================== schema 构造
def empty_metadata() -> dict[str, Any]:
    """统一 Metadata Schema 的默认值，parser 只覆盖自己能确定的字段。"""
    return {
        "doc_type": "unknown",
        "topic": "unknown",
        "has_table": False,
        "has_image": False,
        "language": "unknown",
        "layout_style": "prose",
        "source_format": "unknown",
    }


def new_block(
    block_type: str,
    content: str,
    page: int | None = None,
    bbox: Sequence[float] | None = None,
    font: str | None = None,
    size: float | None = None,
    section_path: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """
    构造统一 block。必带 type/content，可选字段为空时不写入，避免 JSON 膨胀。
    extra 用于放解析期的附加证据（如 clip_vector、ocr_text、header），
    这些字段是「追加」的，不影响统一 schema 的必读字段。
    """
    if block_type not in BLOCK_TYPES:
        raise ValueError(f"非法 block type: {block_type}，可选 {BLOCK_TYPES}")
    block: dict[str, Any] = {"type": block_type, "content": normalize_text(content)}
    if page is not None:
        block["page"] = int(page)
    if bbox is not None:
        block["bbox"] = bbox_to_list(bbox) or list(map(float, bbox))
    if font:
        block["font"] = str(font)
    if size is not None:
        block["size"] = round(float(size), 2)
    if section_path:
        block["section_path"] = section_path
    block.update({k: v for k, v in extra.items() if v is not None})
    return block


def new_document(
    doc_id: str,
    source_format: str,
    metadata: dict[str, Any],
    blocks: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """构造 parser 统一返回值，并做最小 schema 校验。"""
    if source_format not in SOURCE_FORMATS:
        raise ValueError(f"非法 source_format: {source_format}，可选 {SOURCE_FORMATS}")
    merged = empty_metadata()
    merged.update(metadata or {})
    merged["source_format"] = source_format
    merged["language"] = merged.get("language") if merged.get("language") in LANGUAGES else "mixed"
    merged["doc_type"] = merged.get("doc_type") if merged.get("doc_type") in DOC_TYPES else "unknown"
    merged["topic"] = merged.get("topic") if merged.get("topic") in TOPICS else "unknown"
    merged["layout_style"] = (
        merged.get("layout_style") if merged.get("layout_style") in LAYOUT_STYLES else "prose"
    )
    doc: dict[str, Any] = {
        "doc_id": doc_id,
        "source_format": source_format,
        "metadata": merged,
        "blocks": blocks,
    }
    doc.update(extra)
    return doc


# =================================================================== 配置读取
_SETTINGS_CACHE: dict[str, Any] | None = None


def get_settings() -> dict[str, Any]:
    """
    懒加载 batch1 的 settings；读不到就返回最小默认值。
    parser 不能因为配置缺失而崩溃（例如只想知道 OCR dpi 和 chunk 无关的参数）。
    """
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE
    try:
        from config.autodetect import load_settings

        _SETTINGS_CACHE = load_settings()
    except Exception as exc:  # pragma: no cover
        print(f"[parsers] 配置加载失败，使用默认值: {exc}")
        _SETTINGS_CACHE = {
            "profile": "lite",
            "active": {},
            "parsing": {},
            "chunking": {},
            "retrieval": {},
            "lexicon": {},
        }
    return _SETTINGS_CACHE


def parsing_option(key: str, default: Any) -> Any:
    """取 parsing.<key>（支持 'ocr.dpi' 这种点号路径）。"""
    node: Any = get_settings().get("parsing") or {}
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def dump_json_for_cli(result: dict[str, Any]) -> str:
    """CLI 打印用：截断超长向量，避免把 512 维浮点刷屏。"""
    import json

    def _drop_vector(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if (k.endswith("_vector") and isinstance(v, (list, tuple))) or (k == "clip_vector" and isinstance(v, (list, tuple))):
                    out[k] = f"<dim={len(v)}>"
                else:
                    out[k] = _drop_vector(v)
            return out
        if isinstance(obj, list):
            return [_drop_vector(v) for v in obj]
        return obj

    return json.dumps(_drop_vector(result), ensure_ascii=False, indent=2)


def utc_now_iso() -> str:
    """解析时间戳，写进中间 JSON 便于增量重建时判断新旧。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
