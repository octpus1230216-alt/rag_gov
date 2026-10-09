# -*- coding: utf-8 -*-
"""
版面切分：把 parser 的 blocks 切成统一 chunk。

三条硬规矩：
  1. **表格绝不套用文本参数**。type=table / sheet 一律走 table_chunker，
     文本窗口切分只会把「列名 -> 数值」的对应关系切碎，检索回来全是无意义数字。
  2. **标题不单独成 chunk**，它折进 section_path（parser 已把它拼在块的
     section_path 末段）。单独成块会得到一堆「第3章」这种无内容命中，
     挤占 top-k 却回答不了任何问题。
  3. **同 section 的相邻正文跨页合并**。中文 PDF 一章跨十几页很常见，
     按页切会让「付款方式」这段话在页边界被劈成两半，dense 召回两头都读不懂。

切分粒度用「中文 token」而不是字符数：cfg.chunk_size=384 指 token。
token 估算独立成 estimate_tokens（可整体替换，见 set_token_estimator），
将来接真实 BGE-M3 tokenizer 时只换这一个函数，切分逻辑不动。

overlap 的取舍：**以完整语义单元（整句）为单位**，不做字符串级截断。
按字符切重叠区会把「123,456.78」切成两半，两侧都变成无法匹配的噪声 token；
代价是实际重叠 token 略小于配置值（可接受，且完全确定可复现）。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.parsers.common import get_settings

# ---------------------------------------------------------------- 默认切分参数
DEFAULT_CHUNK_SIZE = 384           # 目标 chunk 大小（中文 token）
DEFAULT_CHUNK_OVERLAP = 64         # 相邻 chunk 重叠（token）
DEFAULT_MIN_CHUNK_SIZE = 48        # 低于此值并入前一块，不单独成 chunk
DEFAULT_MAX_TABLE_CHARS = 2000     # 表格超过此字符数才按行组切（需求口径）
DEFAULT_MAX_TABLE_ROWS = 30        # 对应 settings.chunking.max_table_rows_per_chunk
SLIDE_SPLIT_FACTOR = 2.0           # slide 超过 chunk_size×2 才允许破坏「一页一片」
HARD_SIZE_FACTOR = 1.2             # min_chunk 回并时允许超出的上限倍数
MAX_TABLE_ROW_LINES_CHARS = 20000  # 超过此长度的表不再附带逐行 kv 展开（防 JSONL 膨胀）

# token 估算口径（见 estimate_tokens_zh 的 docstring）
CJK_TOKEN_FACTOR = 0.6             # 每个中文字符约 0.6 token
LATIN_CHARS_PER_TOKEN = 4.0        # 英文/数字约 4 字符一个 subword token

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# 连续非中文片段（英文单词、数字、半角标点混排），单独按长度估 token
_LATIN_RUN_RE = re.compile(r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\s]+")
# 句子边界：重叠与硬切后回缩时用，保证不以标点/数字中间开口
_BOUNDARY_CHARS = "。！？；!?;\n"

DEFAULT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ")

CHUNK_TYPES = ("text", "table", "image", "slide", "sheet")


@dataclass(frozen=True)
class ChunkConfig:
    """切分参数。frozen 保证同一批 chunk 的切分口径在进程内不会漂移。"""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE
    respect_sentence_boundary: bool = True
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    # 表格：整表优先，同时受下面两条约束才按行组分片（每片都重复表头）
    max_table_chars: int = DEFAULT_MAX_TABLE_CHARS
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS
    max_table_row_lines_chars: int = MAX_TABLE_ROW_LINES_CHARS


def resolve_config(cfg: ChunkConfig | dict[str, Any] | None = None) -> ChunkConfig:
    """
    合并配置：默认值 <- settings.chunking <- 显式传入（后者覆盖前者）。

    取不到配置时安静退回默认值：切分层不能因为 settings.yaml 缺失就跑不动。
    separators 会去重并保持顺序（模板里「。」重复写了两次）。
    """
    merged: dict[str, Any] = {}
    try:
        node = get_settings().get("chunking") or {}
        if isinstance(node, dict):
            merged.update(node)
    except Exception:
        pass
    if isinstance(cfg, ChunkConfig):
        return cfg
    if isinstance(cfg, dict):
        merged.update({k: v for k, v in cfg.items() if v is not None})

    separators = merged.get("separators") or DEFAULT_SEPARATORS
    unique_seps = tuple(dict.fromkeys(str(s) for s in separators if str(s)))

    def _int(key: str, default: int) -> int:
        try:
            value = int(merged.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return ChunkConfig(
        chunk_size=_int("chunk_size", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_int("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        min_chunk_size=_int("min_chunk_size", DEFAULT_MIN_CHUNK_SIZE),
        respect_sentence_boundary=bool(merged.get("respect_sentence_boundary", True)),
        separators=unique_seps,
        max_table_chars=_int("max_table_chars", DEFAULT_MAX_TABLE_CHARS),
        max_table_rows=_int("max_table_rows_per_chunk", DEFAULT_MAX_TABLE_ROWS),
        max_table_row_lines_chars=_int("max_table_row_lines_chars", MAX_TABLE_ROW_LINES_CHARS),
    )


# =================================================================== token 估算
def estimate_tokens_zh(text: str) -> int:
    """
    中文优先的 token 估算（需求给的口径是 len(text)×0.6，这里是它的细化版）。

    为什么拆开算而不是直接 len×0.6：
      - 中文字符 ≈ 0.6 token/字，与需求一致；
      - 纯英文/数字按 ×0.6 会明显高估（"internationalization" 22 字符 → 估 13 token，
        真实约 6 个 subword），高估会把 chunk 切得过碎，上下文反而断裂；
      - 所以非中文片段按「4 字符 ≈ 1 token」估，空白与半角标点按 0.5 折。
    它是「够用即可」的估算器，不是真 tokenizer：chunk_size 用它控窗口，
    入库时的实际截断以 embedding 阶段的模型 tokenizer 为准。
    """
    if not text or not text.strip():
        return 0
    cjk = len(_CJK_RE.findall(text))
    latin_chars = sum(len(m) for m in _LATIN_RUN_RE.findall(text))
    other = max(len(text) - cjk - latin_chars, 0)
    tokens = cjk * CJK_TOKEN_FACTOR + latin_chars / LATIN_CHARS_PER_TOKEN + other * 0.5
    return max(1, math.ceil(tokens))


def estimate_tokens_jieba(text: str) -> int:
    """
    jieba 分词计数版估算器：中文按「词」估比按「字」估更接近真实 token 数。
    长英文词仍按字符补（jieba 把一个英文串算 1 个词，模型侧可能是多个 subword）。
    未安装 jieba 时抛 ImportError，由 choose_token_estimator() 捕获后退回启发式。
    """
    import jieba  # 延迟导入：没装也能跑，见 choose_token_estimator

    if not text or not text.strip():
        return 0
    words = [w for w in jieba.cut(text) if w.strip()]
    latin_chars = sum(len(m) for m in _LATIN_RUN_RE.findall(text))
    base = len(words) * 0.9 + latin_chars / LATIN_CHARS_PER_TOKEN
    return max(1, math.ceil(base))


_TOKEN_ESTIMATOR = estimate_tokens_zh
_TOKEN_BACKEND = "char_heuristic"


def choose_token_estimator(prefer: str | None = None) -> tuple[Any, str]:
    """
    选择 token 估算器：prefer ∈ {"jieba", "char", None(自动)}。
    自动优先 jieba，导入失败退回字符启发式。返回 (函数, 后端名)。
    """
    if prefer in ("char", "char_heuristic", "heuristic"):
        return estimate_tokens_zh, "char_heuristic"
    if prefer == "jieba":
        estimate_tokens_jieba("测试一下")  # 显式要求时装不上就直接抛，别静默降级
        return estimate_tokens_jieba, "jieba"
    if prefer in (None, "", "auto"):
        try:
            import jieba  # noqa: F401  仅探测可用性

            return estimate_tokens_jieba, "jieba"
        except Exception:
            return estimate_tokens_zh, "char_heuristic"
    raise ValueError(f"未知的 token 估算后端: {prefer}")


def set_token_estimator(func: Any | None = None, prefer: str | None = None) -> str:
    """
    替换全局 token 估算器（后续批次接真实 tokenizer 的唯一入口）。
    func 传 None 时按 prefer 自动选择。返回生效的后端名。
    注意：换估算器会改变片数，chunk_id 随之变化，必须整库重建，不能只补新增文档。
    """
    global _TOKEN_ESTIMATOR, _TOKEN_BACKEND
    if func is not None:
        _TOKEN_ESTIMATOR = func
        _TOKEN_BACKEND = getattr(func, "__name__", "custom")
    else:
        _TOKEN_ESTIMATOR, _TOKEN_BACKEND = choose_token_estimator(prefer)
    return _TOKEN_BACKEND


def token_estimator_name() -> str:
    """当前生效的 token 估算后端名（便于回溯这批 chunk 是哪套口径切出来的）。"""
    return _TOKEN_BACKEND


def estimate_tokens(text: str) -> int:
    """统一入口：所有切分逻辑只调它，内部实现可整体替换。"""
    return _TOKEN_ESTIMATOR(text or "")


# =================================================================== chunk 构造
def make_chunk_id(doc_id: str, section_path: str, idx: int) -> str:
    """
    chunk_id = sha256(doc_id + section_path + idx)。
    用 \\x1f 分隔而不是裸拼接：裸拼会让 ("ab","c") 与 ("a","bc") 撞 id。
    idx 是文档内顺序号 —— 同 blocks + 同 cfg + 同估算器 ⇒ 同输出，重跑幂等。
    """
    key = f"{doc_id}\x1f{section_path}\x1f{idx}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_chunk(
    doc_id: str,
    section_path: str,
    idx: int,
    content: str,
    chunk_type: str,
    metadata: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """
    构造统一 chunk。六个必带字段：chunk_id / doc_id / content / chunk_type /
    section_path / metadata；bbox 与 extra 证据字段为空时不落键，避免 JSONL 膨胀。
    chunk_id 这里先按传入 idx 算一次，最终由 assign_chunk_ids 按全序重算覆盖。
    """
    if chunk_type not in CHUNK_TYPES:
        raise ValueError(f"非法 chunk_type: {chunk_type}，可选 {CHUNK_TYPES}")
    chunk: dict[str, Any] = {
        "chunk_id": make_chunk_id(doc_id, section_path, idx),
        "doc_id": doc_id,
        "content": content,
        "chunk_type": chunk_type,
        "section_path": section_path,
        "metadata": dict(metadata or {}),
    }
    bbox = extra.pop("bbox", None)
    if bbox:
        chunk["bbox"] = [round(float(v), 2) for v in bbox]
    chunk.update({k: v for k, v in extra.items() if v is not None})
    return chunk


def assign_chunk_ids(chunks: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    """
    按最终顺序补 chunk_index 与 chunk_id。
    必须等全部切完再统一编号：切分过程中总片数未知，提前编号会在重跑时错位。
    """
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["doc_id"] = doc_id
        chunk["chunk_id"] = make_chunk_id(doc_id, chunk["section_path"], idx)
    return chunks


# =================================================================== 语义单元
@dataclass
class Unit:
    """一个不可再切的语义单元（整句/整行），带溯源：来自哪一页、哪个 bbox。"""

    text: str
    page: int | None = None
    bbox: Sequence[float] | None = None


def split_keep_delim(text: str, sep: str) -> list[str]:
    """
    按 sep 切分并把分隔符**粘回左侧**片段。
    中文句号绝不能丢：丢了「付款期限为30天。」变成「付款期限为30天」，
    FTS5 精确检索与句子边界判断都会退化；纯空白分隔则直接按空白分。
    切不开（没命中 sep）就原样返回单元素列表。
    """
    if not text:
        return []
    if sep.strip() == "":
        pieces = [p for p in re.split(r"\s+", text) if p]
        return pieces or [text]
    parts = text.split(sep)
    if len(parts) == 1:
        return [text]
    out: list[str] = []
    for i, part in enumerate(parts):
        piece = part if i == len(parts) - 1 else part + sep
        if piece:
            out.append(piece)
    return out


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """
    兜底硬切：片段里没有任何分隔符（超长英文串、无标点 OCR 文本）时按字符窗口切。
    步长由「本片实际 字符/token 比」反推，而不是写死 ×0.6，
    这样换 token 估算器时硬切粒度自动跟着变。
    """
    tokens = estimate_tokens(text)
    if tokens <= max_tokens:
        return [text]
    chars_per_token = max(len(text) / tokens, 1.0)
    step = max(1, int(max_tokens * chars_per_token * 0.95))
    return [text[start : start + step] for start in range(0, len(text), step)]


def split_into_units(text: str, max_tokens: int, cfg: ChunkConfig) -> list[str]:
    """
    把文本切成「每片 token <= max_tokens」的最小语义单元。
    按 separators 优先级逐级下切（段落 -> 行 -> 句 -> 分句 -> 逗号 -> 空格）；
    respect_sentence_boundary=False 时直接硬切（纯吞吐场景，一般不开）。
    保证：返回的每一片都不超 max_tokens，因此窗口推进一定前进（无死循环）。
    """
    stripped = (text or "").strip()
    if not stripped:
        return []
    if not cfg.respect_sentence_boundary:
        return _hard_split(stripped, max_tokens)

    pieces = [stripped]
    for sep in cfg.separators:
        nxt: list[str] = []
        for piece in pieces:
            if estimate_tokens(piece) <= max_tokens:
                nxt.append(piece)
                continue
            subs = split_keep_delim(piece, sep)
            nxt.extend(subs if len(subs) > 1 else [piece])
        pieces = nxt
        if all(estimate_tokens(p) <= max_tokens for p in pieces):
            break  # 已达标，不必再往更细的分隔符走

    units: list[str] = []
    for piece in pieces:
        if estimate_tokens(piece) <= max_tokens:
            units.append(piece)
        else:
            units.extend(_hard_split(piece, max_tokens))
    return [u for u in units if u.strip()]


# =================================================================== 窗口装填
def _overlap_units(units: Sequence[Unit], start: int, end: int, cfg: ChunkConfig) -> int:
    """
    计算下一片的回退起点：从上一片尾部往前累计，直到 token 触及 overlap 预算。
    以整句为单位（不做字符截断），重叠预算被单条长句吃掉时就让位给更少句子，
    极端情况重叠为 0 —— 宁可少重叠，也不要切碎数字/英文串。
    """
    budget = cfg.chunk_overlap
    used = 0
    head = end
    for i in range(end - 1, start, -1):
        t = estimate_tokens(units[i].text)
        if used + t > budget:
            break
        used += t
        head = i
    return head


def _pack_windows(units: Sequence[Unit], cfg: ChunkConfig) -> list[tuple[int, int]]:
    """
    贪心装窗，返回 [(起点, 终点)]（左闭右开，下标指向 units）。
    每个窗口至少含一个单元；下一窗口从上一窗口尾部整句处重新开始（overlap）。
    start 严格递增，保证一定能收敛。
    """
    windows: list[tuple[int, int]] = []
    n = len(units)
    i = 0
    while i < n:
        j = i
        tokens = 0
        while j < n:
            u_tok = estimate_tokens(units[j].text)
            # 当前已有内容且再加会超窗 -> 收口；空窗时至少装一个单元
            if j > i and tokens + u_tok > cfg.chunk_size:
                break
            tokens += u_tok
            j += 1
        windows.append((i, j))
        # head 恒 > i（_overlap_units 不会回退到窗口起点本身），再兜一层保证推进
        head = _overlap_units(units, i, j, cfg)
        i = max(head, i + 1)
    return windows


def _merge_small_windows(units: Sequence[Unit], windows: list[tuple[int, int]], cfg: ChunkConfig) -> list[tuple[int, int]]:
    """
    过短片段回并到前一片（min_chunk_size），前提是合并后不超过 chunk_size×HARD_SIZE_FACTOR。
    合并区间取并集端点（重叠窗口下 prev_start..cur_end 是连续区间，内容不丢不重）。
    首片即使过短也保留：内容一条都不能丢，是切分层唯一的绝对底线。
    """
    if len(windows) < 2:
        return windows
    limit = cfg.chunk_size * HARD_SIZE_FACTOR
    out: list[tuple[int, int]] = []
    for start, end in windows:
        tokens = sum(estimate_tokens(units[i].text) for i in range(start, end))
        if out and tokens < cfg.min_chunk_size:
            prev_start, prev_end = out[-1]
            combined = sum(estimate_tokens(units[i].text) for i in range(prev_start, end))
            if combined <= limit:
                out[-1] = (prev_start, end)
                continue
        out.append((start, end))
    return out


def _group_text_runs(
    blocks: Sequence[dict[str, Any]],
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    """
    相邻且 section_path 相同的 text 块合成一组（这就是「跨页同 section 合并」）。
    返回 (section_path, 首个块下标, 块列表)。
    首个块下标用作最终排序键：一组文本可能跨过中间的表格/图片，
    用首次出现位置排序既不丢顺序也保证重跑结果一致。
    注意：中间夹的非 text 块不打断分组（表格另成 chunk，正文语义该连起来读）。
    """
    groups: list[tuple[str, int, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    section = ""
    start_index = 0
    for idx, block in enumerate(blocks):
        if block.get("type") != "text":
            continue
        path = str(block.get("section_path") or "")
        if current and path != section:
            groups.append((section, start_index, current))
            current = []
        if not current:
            section = path
            start_index = idx
        current.append(block)
    if current:
        groups.append((section, start_index, current))
    return groups


def merge_bbox(rects: Iterable[Sequence[float]]) -> list[float] | None:
    """多个 bbox 求并集矩形；无有效输入返回 None（bbox 是可选字段，不硬造）。"""
    x0 = y0 = x1 = y1 = None
    for rect in rects:
        if not rect or len(rect) != 4:
            continue
        a, b, c, d = (float(v) for v in rect)
        x0 = a if x0 is None else min(x0, a)
        y0 = b if y0 is None else min(y0, b)
        x1 = c if x1 is None else max(x1, c)
        y1 = d if y1 is None else max(y1, d)
    if None in (x0, y0, x1, y1):
        return None
    return [x0, y0, x1, y1]


def _chunk_text_group(
    run: Sequence[dict[str, Any]],
    section: str,
    metadata: dict[str, Any],
    cfg: ChunkConfig,
    doc_id: str,
) -> list[dict[str, Any]]:
    """
    一组同 section 正文 -> 若干 text chunk。
    跨块拼接时给后续块补换行，避免上一段末尾和下一段开头黏成一句（中文没有空格，
    黏连后 jieba/FTS5 都会在接缝处切出错误 token）。
    """
    units: list[Unit] = []
    for block in run:
        text = str(block.get("content") or "").strip()
        if not text:
            continue
        page = block.get("page")
        bbox = block.get("bbox")
        piece_units = split_into_units(text, cfg.chunk_size, cfg)
        for j, unit in enumerate(piece_units):
            if j == 0 and units:
                unit = "\n" + unit  # 块间接缝
            units.append(Unit(text=unit, page=page, bbox=bbox))
    if not units:
        return []

    windows = _merge_small_windows(units, _pack_windows(units, cfg), cfg)
    out: list[dict[str, Any]] = []
    for start, end in windows:
        content = "".join(units[i].text for i in range(start, end)).strip()
        if not content:
            continue
        pages = sorted({units[i].page for i in range(start, end) if units[i].page})
        out.append(
            new_chunk(
                doc_id=doc_id,
                section_path=section,
                idx=len(out),
                content=content,
                chunk_type="text",
                metadata=metadata,
                bbox=merge_bbox(units[i].bbox for i in range(start, end)),
                pages=pages or None,
                token_estimate=estimate_tokens(content),
                n_units=end - start,
                token_backend=_TOKEN_BACKEND,
            )
        )
    return out


def chunk_text_blocks(
    blocks: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    cfg: ChunkConfig | dict[str, Any] | None = None,
    doc_id: str = "",
) -> list[dict[str, Any]]:
    """
    正文（type=text）切分主入口（需求点名的函数）。
    按标题层级聚合 -> 跨页同 section 合并 -> token 窗口 + 整句 overlap。
    type=title 的块不产出 chunk；其它类型由 chunk_blocks 调度到各自切分器。
    """
    config = resolve_config(cfg)
    results: list[dict[str, Any]] = []
    for section, _start, run in _group_text_runs(blocks):
        results.extend(_chunk_text_group(run, section, metadata, config, doc_id))
    return assign_chunk_ids(results, doc_id) if doc_id else results


# =================================================================== slide / image
def chunk_slide_block(
    block: dict[str, Any],
    metadata: dict[str, Any],
    cfg: ChunkConfig | dict[str, Any] | None = None,
    doc_id: str = "",
) -> list[dict[str, Any]]:
    """
    slide 块：一页一 chunk —— PPT 的语义单元就是整页，切开会「只有论据没有结论」。
    仅当页面内容超过 chunk_size×SLIDE_SPLIT_FACTOR（塞了整篇讲稿的畸形页）才切开，
    并给每片标 overflow=True，检索侧可据此降权。
    """
    config = resolve_config(cfg)
    content = str(block.get("content") or "").strip()
    if not content:
        return []
    pages = [block["page"]] if block.get("page") else None
    base_extra = {
        "pages": pages,
        "token_estimate": estimate_tokens(content),
        "slide_index": block.get("slide_index"),
        "title": block.get("title") or None,
        "token_backend": _TOKEN_BACKEND,
    }
    if estimate_tokens(content) <= config.chunk_size * SLIDE_SPLIT_FACTOR:
        return [
            new_chunk(
                doc_id=doc_id,
                section_path=str(block.get("section_path") or ""),
                idx=0,
                content=content,
                chunk_type="slide",
                metadata=metadata,
                bbox=block.get("bbox"),
                **base_extra,
            )
        ]

    units = [Unit(u, block.get("page"), block.get("bbox")) for u in split_into_units(content, config.chunk_size, config)]
    windows = _merge_small_windows(units, _pack_windows(units, config), config)
    out: list[dict[str, Any]] = []
    for part, (start, end) in enumerate(windows):
        text = "".join(units[i].text for i in range(start, end)).strip()
        if not text:
            continue
        extra = dict(base_extra)
        extra["token_estimate"] = estimate_tokens(text)
        extra["overflow"] = True
        extra["part"] = part + 1
        out.append(
            new_chunk(
                doc_id=doc_id,
                section_path=str(block.get("section_path") or ""),
                idx=part,
                content=text,
                chunk_type="slide",
                metadata=metadata,
                bbox=block.get("bbox"),
                **extra,
            )
        )
    return out


def chunk_image_block(
    block: dict[str, Any],
    metadata: dict[str, Any],
    doc_id: str = "",
) -> list[dict[str, Any]]:
    """
    image 块：一图一 chunk，永不切开（一张图对应一条 CLIP 向量，切开就变两条向量
    指向同一张图，dense 召回会自己和自己竞争）。
    clip_vector 原样透传给 embedding 阶段 —— 本批只做切分，不编码也不存储。
    """
    content = str(block.get("content") or "").strip()
    if not content:
        return []
    return [
        new_chunk(
            doc_id=doc_id,
            section_path=str(block.get("section_path") or ""),
            idx=0,
            content=content,
            chunk_type="image",
            metadata=metadata,
            bbox=block.get("bbox"),
            pages=[block["page"]] if block.get("page") else None,
            token_estimate=estimate_tokens(content),
            ocr_text=block.get("ocr_text") or None,
            clip_vector=block.get("clip_vector"),
            classification=block.get("classification"),
            image_doc_id=block.get("doc_id"),
            content_placeholder=bool(block.get("content_placeholder")) or None,
            token_backend=_TOKEN_BACKEND,
        )
    ]


# =================================================================== 总调度
def chunk_blocks(
    blocks: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    cfg: ChunkConfig | dict[str, Any] | None = None,
    doc_id: str = "",
) -> list[dict[str, Any]]:
    """
    总调度：按 block type 分流，最后按版面首次出现位置统一排序并编号 chunk_id。
      text  -> 同 section 聚合 + token 窗口（含跨页合并）
      table -> table_chunker.chunk_table（整表优先，超限按行组分片）
      sheet -> table_chunker.chunk_table(chunk_type="sheet")
      slide -> chunk_slide_block（一页一片）
      image -> chunk_image_block（一片一图，透传 clip_vector）
      title -> 不产出 chunk，只作为 section_path 的一部分
    """
    # 函数级导入：layout 依赖 table 一次，反向也依赖一次，模块级导入会成环
    from ingestion.chunkers.table_chunker import chunk_table

    config = resolve_config(cfg)
    plan: list[tuple[int, list[dict[str, Any]]]] = []
    table_seq = 0
    for idx, block in enumerate(blocks):
        btype = block.get("type")
        if btype in ("table", "sheet"):
            table_seq += 1
            plan.append(
                (idx, chunk_table(block, metadata, cfg=config, chunk_type=btype, table_index=table_seq, doc_id=doc_id))
            )
        elif btype == "slide":
            plan.append((idx, chunk_slide_block(block, metadata, config, doc_id)))
        elif btype == "image":
            plan.append((idx, chunk_image_block(block, metadata, doc_id)))

    for section, start_index, run in _group_text_runs(blocks):
        sub = _chunk_text_group(run, section, metadata, config, doc_id)
        if sub:
            plan.append((start_index, sub))

    plan.sort(key=lambda kv: kv[0])
    ordered: list[dict[str, Any]] = []
    for _key, chunks in plan:
        ordered.extend(chunks)
    return assign_chunk_ids(ordered, doc_id)


def chunk_document(parsed: dict[str, Any], cfg: ChunkConfig | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    喂 parser 的完整中间 JSON，产出全部 chunk（pipeline 的 chunk 阶段入口）。
    doc_id / metadata 全部继承；metadata 里补 source_format，检索侧可直接按格式过滤。
    """
    doc_id = str(parsed.get("doc_id") or "")
    metadata = dict(parsed.get("metadata") or {})
    blocks = list(parsed.get("blocks") or [])
    if not blocks:
        return []
    chunks = chunk_blocks(blocks, metadata, cfg, doc_id)
    fmt = parsed.get("source_format")
    if fmt:
        for chunk in chunks:
            chunk["metadata"]["source_format"] = fmt
    return chunks


def chunk_file(path: str | Path, cfg: ChunkConfig | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """独立入口：读一个 parsed/*.json 直接切（调试与批量补切用）。"""
    return chunk_document(json.loads(Path(path).read_text(encoding="utf-8")), cfg)


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="把 parsed/*.json 切成 chunk（本地调试用）")
    cli.add_argument("path", help="parser 输出的中间 JSON 路径")
    cli.add_argument(
        "--token-backend", default="auto", choices=["auto", "jieba", "char"], help="token 估算口径"
    )
    cli.add_argument("--limit", type=int, default=20, help="最多打印几条 chunk")
    args = cli.parse_args()
    set_token_estimator(prefer=args.token_backend)
    pieces = chunk_file(args.path)
    print(f"token 后端={token_estimator_name()} chunk 数={len(pieces)}")
    for c in pieces[: args.limit]:
        print(
            f"  [{c['chunk_type']:<5}] tok={c['token_estimate']:>4} idx={c['chunk_index']:>3} "
            f"{c['section_path']} :: {c['content'][:40]!r}"
        )
