# -*- coding: utf-8 -*-
"""
入库流水线骨架：parse -> classify -> chunk -> embed -> index。

五个阶段均已实现：
    parse / classify / chunk   产中间 JSON（parsed/chunks）
    embed                      读 chunks 产向量文件 vectors/{doc_id}.jsonl（BGE-M3 dense+sparse）
    index                      读向量文件推入 Qdrant + SQLite(FTS5/ngram) + 元数据表，落 .index 标记

设计约定：
  1. 幂等以「产物文件是否存在」判定，doc_id 先算再查产物，所以重跑同一份文件
     在 parse 阶段就会被跳过，不必真的解析一遍。换内容 = 换 doc_id = 重跑；
     改代码想强制重跑用 --force。
  2. 每阶段产物落盘（parsed/{doc_id}.json、chunks/{doc_id}.jsonl、
     vectors/{doc_id}.jsonl、vectors/{doc_id}.index），断点续跑靠 --from-stage/--to-stage 表达。
     写盘一律「临时文件 + os.replace」：进程被 kill 时不能留下半截 JSON，
     否则下次跑会把半截文件当成「已完成」而永久跳过。
  3. 失败定位到具体文件：单文件异常被捕获后写进 report（含 traceback 与失败阶段），
     并追加 logs/pipeline_errors.jsonl，绝不因为一个坏 PDF 停掉整批。
  4. 多进程用 multiprocessing.Pool（Windows 是 spawn，子进程重新 import，
     所以 worker 必须是模块级函数、参数必须是可 pickle 的普通 dict）。
  5. doc_id 只从本文件的 compute_doc_id() 出（见下一节），parser 内部那个
     sha256(字节) 只是 BYTES 口径的实现细节，落盘前一律被回填覆盖。

===========================================================================
doc_id 三种策略（DocIdStrategy），以及为什么默认是 BYTES
===========================================================================
  BYTES       sha256(file_bytes)。这是需求定的契约，也是唯一「不需要先解析
              就能算出 id」的免费策略：同一份原始文件永久稳定，重跑、断点续跑、
              按 id 查产物都不依赖解析器状态，且天然把「字节完全相同」的重复
              文件（企业库里同名附件到处复制，量非常大）合并成一条。
              已知代价：OOXML（docx/xlsx/pptx 本质是 zip）存盘会写本地时间戳，
              同一份内容两次导出的字节不同 -> doc_id 不同；同一份原始文件重跑
              稳定，但「重新导出」会被当成新文档。这是接受默认策略换来的代价，
              需要用 CONTENT 消除。

  CONTENT     对解析后的 blocks 做规范化哈希（只取 type + content，丢
              bbox/font/size/page，图片走图像内容指纹），跨导出稳定，
              顺带把「同一文档的两个不同副本」合并。代价三条都要认：
                a) 必须先 parse 再算 id -> parse 阶段无法用 id 预判跳过，
                   所以额外用 parsed/by-path/ 缓存（键含 size+mtime_ns，
                   文件一变自动失效）把重跑成本降到「读一个 id 小文件」；
                b) id 依赖解析器逻辑：解析器一升级（标题判定、表格还原改了）
                   所有 CONTENT id 全变，所以指纹里编了 PARSER_VERSION，
                   改解析逻辑必须手动 bump 并整库重建，别指望它自动稳定；
                   （分段用 0x1f/0x1e 控制字符做分隔，避免内容里的竖线冒犯边界）
                c) 没有真实模型时图片指纹会退化（见 image_fingerprint）。
              仅在「同一份文档被反复重新导出」的场景才值得开，默认不启用。

  PATH_MTIME  sha256(绝对路径 + mtime_ns + size)。最快（连文件内容都不读），
              适合本地一次性建索引 / 数据清洗实验。但身份挂在路径上：
              文件改名、换目录、换机器、拷贝给别人，id 全变，同一份内容会以
              多个文档重复入库。生产库禁用。

  共同红线：**同一套库必须固定一种策略**。中途换策略会让同一文档以两个
  doc_id 分别入库（chunk 也翻倍），--force 也救不回来，只能清库重建；
  所以产物目录里没有「混策略」的自愈逻辑，只有 report/日志里的
  doc_id_strategy 字段供事后排查。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from ingestion.chunkers.layout_chunker import (
    chunk_document,
    resolve_config,
    set_token_estimator,
    token_estimator_name,
)
from ingestion.classifiers.doc_classifier import classify
from ingestion.parsers.common import get_settings, read_bytes, sha256_hex, utc_now_iso
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.pptx_parser import parse_pptx
from ingestion.parsers.txt_parser import parse_txt
from ingestion.parsers.xlsx_parser import parse_xlsx

# 统一日志：阶段里程碑走 INFO，坏文件走 ERROR（含 traceback）。
# 红线：绝不用 print 到 stdout —— MCP stdio 下 stdout 是 JSON-RPC 通道；日志经
# config.logging_setup 统一挂到 stderr + 文件。tqdm 进度条本身默认写 stderr，保留。
logger = logging.getLogger("rag_zh.pipeline")

# ---------------------------------------------------------------- 阶段表
STAGES: tuple[str, ...] = ("parse", "classify", "chunk", "embed", "index")
# 五阶段全部实现：parse/classify/chunk 落中间产物，embed 落向量文件，index 推外部库。
# embed/index 依赖重模型与外部服务（FlagEmbedding / qdrant_client），函数内惰性 import，
# 保证只跑到 chunk 的场景（如离线自检）不因缺依赖而 import 失败。
IMPLEMENTED: dict[str, bool] = {
    "parse": True,
    "classify": True,
    "chunk": True,
    "embed": True,
    "index": True,
}
# 阶段 -> 产物目录配置键（settings.runtime）
STAGE_DIR_KEY: dict[str, str] = {
    "parse": "parsed_dir",
    "classify": "parsed_dir",     # classify 就地回写 parsed/{doc_id}.json
    "chunk": "chunks_dir",
    "embed": "vectors_dir",
    "index": "vectors_dir",
}
STAGE_DIR_DEFAULT: dict[str, str] = {
    "parse": "parsed",
    "classify": "parsed",
    "chunk": "chunks",
    "embed": "vectors",
    "index": "vectors",
}

# 扩展名 -> parser。图片与文本之外的格式在第 2 批都有独立 parser，这里只做映射。
EXT_DISPATCH: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".txt": "txt",
    ".md": "txt",
    ".markdown": "txt",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
}

STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_PENDING = "pending"
STATUS_ERROR = "error"


def stage_index(stage: str) -> int:
    """阶段序号，用于区间比较。"""
    try:
        return STAGES.index(stage)
    except ValueError as exc:
        raise ValueError(f"未知阶段: {stage}，可选 {STAGES}") from exc


def stage_range(from_stage: str, to_stage: str) -> list[str]:
    """取 [from_stage, to_stage] 闭区间内的阶段名（顺序错则报清晰错误）。"""
    lo, hi = stage_index(from_stage), stage_index(to_stage)
    if lo > hi:
        raise ValueError(f"阶段区间非法：--from {from_stage} 在 --to {to_stage} 之后")
    return list(STAGES[lo : hi + 1])


# =================================================================== 路径
def output_root(root: str | os.PathLike[str] | None = None) -> Path:
    """产物根目录：显式参数 > settings.runtime.output_root > 当前工作目录。"""
    if root:
        return Path(root)
    node = get_settings().get("runtime") or {}
    value = node.get("output_root") if isinstance(node, dict) else None
    return Path(str(value)) if value else Path.cwd()


def stage_dir(stage: str, root: str | os.PathLike[str] | None = None, create: bool = True) -> Path:
    """
    某阶段的产物目录。create=True 时 mkdir（幂等），写产物的路径都要它；
    只查询「产物在不在」的场合（未实现阶段的路径推测）传 create=False，
    免得探一次错误阶段就凭空建出一个空目录。
    """
    if stage not in STAGE_DIR_KEY:
        raise ValueError(f"未知阶段: {stage}")
    node = get_settings().get("runtime") or {}
    key = STAGE_DIR_KEY[stage]
    # 配置缺失或类型不对（有人把 runtime 写成列表）都退回默认目录名，不让切分跑不起来
    name = str(node.get(key) or STAGE_DIR_DEFAULT[stage]) if isinstance(node, dict) else STAGE_DIR_DEFAULT[stage]
    path = output_root(root) / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(stage: str, doc_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    """
    阶段产物路径：parse/classify -> parsed/{doc_id}.json，chunk -> chunks/{doc_id}.jsonl，
    embed -> vectors/{doc_id}.jsonl（chunk + dense/sparse 逐行），
    index -> vectors/{doc_id}.index（成功推入 Qdrant+SQLite 后的幂等标记，内容仅状态）。
    embed/index 用 JSONL 而非 parquet：向量文件不引入 pandas/pyarrow 依赖，
    与 chunk 阶段同构，index 读它时流式逐行、不把整文档向量一次性 load 进内存。
    未实现的阶段只拼路径不建目录（幂等判定与错误定位都要能问它）。
    """
    create = IMPLEMENTED.get(stage, False)
    if stage in ("parse", "classify"):
        return stage_dir("parse", root, create) / f"{doc_id}.json"
    if stage == "chunk":
        return stage_dir("chunk", root, create) / f"{doc_id}.jsonl"
    if stage == "embed":
        return stage_dir("embed", root, create) / f"{doc_id}.jsonl"
    return stage_dir("index", root, create) / f"{doc_id}.index"


def log_dir(root: str | os.PathLike[str] | None = None) -> Path:
    """日志目录（错误明细写这里，用于「哪些文件失败、失败在哪一步」）。"""
    node = get_settings().get("runtime") or {}
    name = str(node.get("log_dir") or "logs") if isinstance(node, dict) else "logs"
    path = output_root(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    """临时文件 + os.replace 原子落盘，避免半截产物被当成已完成。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """追加一行 JSON（错误日志/运行日志），失败不影响主流程。"""
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# =================================================================== doc_id 策略
class DocIdStrategy(str, Enum):
    """
    doc_id 生成策略，取舍见文件头注释。str 枚举 -> CLI 可以直接收字符串，
    写进产物 JSON 也天然序列化（report/parsed 里的 doc_id_strategy 就是它的 value）。
    """

    BYTES = "bytes"              # sha256(file_bytes)，默认，需求契约
    CONTENT = "content"          # 解析后 blocks 的规范化哈希，跨导出稳定
    PATH_MTIME = "path_mtime"    # sha256(绝对路径+size+mtime_ns)，最快，仅本地

    @property
    def needs_parse(self) -> bool:
        """CONTENT 必须先解析才能算 id；另两种可以先查产物再决定要不要解析。"""
        return self is DocIdStrategy.CONTENT


# 解析逻辑版本号：改标题判定/表格还原/清洗规则等任何会影响 block 的 type 或
# content 的代码时，必须手动 bump 这里（CONTENT 策略下旧 id 会全部失效，整库重建）
PARSER_VERSION = "3.0.0"
DEFAULT_ID_STRATEGY = DocIdStrategy.BYTES

# CONTENT 指纹的分隔符：0x1f 分字段、0x1e 分块。用控制字符是因为正文里
# 出现「|」「:」太频繁，拿它们当分隔符会让 ("a|b","c") 与 ("a","b|c") 撞指纹。
FIELD_SEP = "\x1f"
BLOCK_SEP = "\x1e"

# 只有这些 source_format 表示「整篇文档就是一张图片」，才能拿输入字节算 pHash
IMAGE_FORMATS = ("jpg", "jpeg", "png")
# pHash 口径（改动 = 所有图片 CONTENT 指纹变化 = 整库重建，所以钉死在这里）
PHASH_RESAMPLE = 32              # 灰度重采样边长（DCT 输入）
PHASH_BLOCK = 8                  # 取左上 8x8 低频 -> 64 bit


def normalize_strategy(value: str | DocIdStrategy | None) -> DocIdStrategy:
    """把 CLI 字符串 / 枚举 / None 归一成 DocIdStrategy；非法值给出可选清单。"""
    if value is None:
        return DEFAULT_ID_STRATEGY
    if isinstance(value, DocIdStrategy):
        return value
    text = str(value).strip().lower().replace("-", "_")
    try:
        return DocIdStrategy(text)
    except ValueError as exc:
        raise ValueError(f"未知的 doc_id 策略: {value!r}，可选 {[s.value for s in DocIdStrategy]}") from exc


def path_identity(path: str | Path) -> str:
    """
    绝对路径 + size + mtime_ns 的可读指纹（PATH_MTIME 的内容，也是 by-path 缓存的键）。
    Windows 文件系统大小写不敏感，所以 Windows 上对路径做 casefold ——
    否则 "D:\\Docs\\A.docx" 和 "d:\\docs\\a.docx" 会算出两个 id，同一份文件被重复入库。
    """
    p = Path(path)
    st = p.stat()
    raw = str(p.resolve())
    if os.name == "nt":
        raw = raw.casefold()
    return f"{raw}{FIELD_SEP}{st.st_size}{FIELD_SEP}{st.st_mtime_ns}"


def _normalize_block_text(text: Any) -> str:
    """
    block 正文规范化：统一换行 -> 去每行尾随空格 -> 掐头尾空白。
    为什么必须去尾随空格：OOXML 重导出时行尾常常多/少一个空格（拼写检查、
    自动校正留下的），留着它「同一份内容两次导出」的指纹就不等了，
    CONTENT 策略的意义当场失效。
    """
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


@lru_cache(maxsize=2)
def _dct_basis(n: int) -> Any:
    """DCT-II 基矩阵（用 numpy 现算，不引 scipy）。按 n 缓存，浮点口径每次一致。"""
    import numpy as np

    idx = np.arange(n, dtype=float)
    return np.cos(np.pi * (2.0 * idx[:, None] + 1.0) * idx[None, :] / (2.0 * n))


def phash_from_bytes(data: bytes) -> str | None:
    """
    感知哈希 pHash：灰度 -> PHASH_RESAMPLE 方图 -> DCT -> 左上 PHASH_BLOCK 低频
    （丢掉 DC，它只反映整体亮度）-> 与中位数比大小 -> 64 bit。
    对缩放、重压缩、亮度微调稳定，所以「同一张图重新导出」指纹不变。
    读不出图（损坏/非图片）返回 None，交给上层退到下一路证据，绝不瞎猜。
    """
    if not data:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            gray = img.convert("L").resize((PHASH_RESAMPLE, PHASH_RESAMPLE), Image.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    except Exception:
        return None
    basis = _dct_basis(PHASH_RESAMPLE)
    coeffs = basis @ pixels @ basis.T
    low = coeffs[:PHASH_BLOCK, :PHASH_BLOCK].reshape(-1)[1:]
    median = float(np.median(low))
    bits = 0
    for value in low:
        bits = (bits << 1) | (1 if float(value) > median else 0)
    return f"{bits:016x}"


def _quantize_vector(vector: Sequence[Any]) -> str | None:
    """
    CLIP 向量粗量化：每维只留「符号位 + 4 档量级」。
    不做小数取整 —— 模型推理的浮点噪声就落在低位上，round 到 3 位照样漂；
    粗桶对噪声免疫，代价是语义极近的图会被判同一张（去重场景里这是想要的行为）。
    """
    signs: list[str] = []
    levels: list[str] = []
    try:
        for raw in vector:
            value = float(raw)          # 非数值当场抛错，由下面的 except 兜住
            signs.append("1" if value > 0 else "0")
            mag = abs(value)
            levels.append("0" if mag < 0.1 else "1" if mag < 0.3 else "2" if mag < 0.6 else "3")
    except (TypeError, ValueError):
        return None
    if not signs:
        return None
    return sha256_hex(("".join(signs) + "".join(levels)).encode())


def image_fingerprint(
    block: dict[str, Any],
    data: bytes | None = None,
    source_format: str = "",
) -> tuple[str, str]:
    """
    图片块的「图像内容」指纹，返回 (指纹, 用了哪一路)。
    **绝不用文件名/路径/位置编号**：第 2 批给图片块写的 block["doc_id"] 形如
    "<父文档bytes_id>:p3:x12"，那是父文档字节哈希的派生物，拿它当内容指纹
    会让 CONTENT 策略「跨导出稳定」当场破功，所以这里刻意不读那个字段。

    优先级（越靠前越可靠）：
      1. block["phash"] —— 后续去重批次会显式落一个，有就直接用，口径统一
      2. 整篇就是一张图片时，对输入字节算 pHash（不依赖模型，本地就能跑）
      3. clip_vector 粗量化 —— 图片在 PDF/DOCX 内部时拿不到原始字节，
         向量化已经做完，量化比重新解码稳
      4. 退化：用图注 + OCR 文本兜底（标 method=text_fallback）
         —— 模型没装、字节也拿不到时只剩这些证据。它仍是内容派生而非文件名，
         但不是感知哈希：同一张图换个 OCR 模型结果就变，用 CONTENT 策略前要认这条。
    """
    given = str(block.get("phash") or "").strip()
    if given:
        return given, "phash"
    if source_format in IMAGE_FORMATS and data:
        fp = phash_from_bytes(data)
        if fp:
            return fp, "phash_input_bytes"
    vector = block.get("clip_vector")
    if isinstance(vector, (list, tuple)) and vector:
        quantized = _quantize_vector(vector)
        if quantized:
            return quantized, "clip_quantized"
    text = _normalize_block_text(block.get("content"))
    return (sha256_hex(text.encode("utf-8")) if text else "no-image-evidence"), "text_fallback"


def content_fingerprint(parsed: dict[str, Any], data: bytes | None = None) -> tuple[str, dict[str, Any]]:
    """
    解析结果 -> 规范化内容指纹（CONTENT 策略的实现）。
    只取每个 block 的 type + content，丢掉 bbox/font/size/page/section_path ——
    这些在重导出时必然抖（字体子集重排、行盒位置变化），留着就等于白做。
    块的**顺序**保留（来自阅读顺序，是稳定信息），所以调序会被识别成不同文档。
    图片块换成图像内容指纹（见 image_fingerprint）。
    指纹头编了 PARSER_VERSION：解析逻辑变更会改变 type/content，必须手动 bump。
    返回 (指纹, 元信息)，元信息写进 parsed 便于回溯「图片那几块靠什么算的」。
    """
    h = hashlib.sha256()
    h.update(f"rag-zh|content|v{PARSER_VERSION}\n".encode())
    fmt = str(parsed.get("source_format") or "")
    methods: dict[str, int] = {}
    for block in parsed.get("blocks") or []:
        btype = str(block.get("type") or "")
        if btype == "image":
            fp, method = image_fingerprint(block, data, fmt)
            body = f"{method}{FIELD_SEP}{fp}"
            methods[method] = methods.get(method, 0) + 1
        else:
            body = _normalize_block_text(block.get("content"))
        h.update(btype.encode("utf-8"))
        h.update(FIELD_SEP.encode("utf-8"))
        h.update(body.encode("utf-8"))
        h.update(BLOCK_SEP.encode("utf-8"))
    return h.hexdigest(), {"parser_version": PARSER_VERSION, "image_fp": methods}


def compute_doc_id(
    path: str | Path,
    strategy: str | DocIdStrategy = DEFAULT_ID_STRATEGY,
    *,
    data: bytes | None = None,
    parsed: dict[str, Any] | None = None,
) -> str:
    """
    doc_id 的唯一出口（三种策略全部收敛到这一个函数，别处不许自己拼哈希）。

    BYTES / PATH_MTIME 只依赖文件系统，先算 id 再查产物，parse 可以被整个跳过；
    CONTENT 依赖解析结果，必须传 parsed=（没传就直接报错，绝不静默退化成 BYTES
    —— 那会让调用方以为开了跨导出稳定，实际拿到的是字节 id）。
    """
    strat = normalize_strategy(strategy)
    p = Path(path)
    if strat is DocIdStrategy.BYTES:
        return sha256_hex(data if data is not None else read_bytes(p))
    if strat is DocIdStrategy.PATH_MTIME:
        return sha256_hex(path_identity(p).encode("utf-8"))
    if parsed is None:
        raise ValueError("CONTENT 策略必须先 parse 再算 doc_id：compute_doc_id(..., parsed=) 缺 parsed")
    fp, _meta = content_fingerprint(parsed, data)
    return fp


def by_path_cache_file(path: str | Path, root: str | os.PathLike[str] | None = None) -> Path:
    """
    CONTENT 策略的「路径 -> content doc_id」缓存文件。
    键里编了 size + mtime_ns，所以文件被改动/改名/换目录后自然查不中，
    不需要任何失效逻辑；命中时 parsed/{content_id}.json 仍要存在才算数
    （产物被清了就去重新解析，缓存在这里只是省一次解析，不是身份来源）。
    """
    return stage_dir("parse", root) / "by-path" / f"{sha256_hex(path_identity(path).encode('utf-8'))}.id"


def read_cached_content_id(path: str | Path, root: str | os.PathLike[str] | None = None) -> str | None:
    """查缓存；没缓存文件或文件本身读不到都返回 None（当作未命中，不抛）。"""
    try:
        cache = by_path_cache_file(path, root)
    except OSError:
        return None
    if not cache.exists():
        return None
    value = cache.read_text(encoding="utf-8").strip()
    return value or None


def write_cached_content_id(path: str | Path, doc_id: str, root: str | os.PathLike[str] | None = None) -> None:
    """写缓存（原子）：CONTENT 解析完成后，同一份未变动的文件再跑就不用重新解析。"""
    try:
        cache = by_path_cache_file(path, root)
    except OSError:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(cache, doc_id + "\n")


def peek_doc_id(
    path: str | Path,
    strategy: str | DocIdStrategy = DEFAULT_ID_STRATEGY,
    root: str | os.PathLike[str] | None = None,
) -> str | None:
    """
    不解析就能拿到的 doc_id；CONTENT 策略只能查 by-path 缓存。
    返回 None 表示「现在还算不出 id」，由调用方决定是去解析还是报缺上游产物。
    """
    strat = normalize_strategy(strategy)
    if strat is DocIdStrategy.CONTENT:
        return read_cached_content_id(path, root)
    return compute_doc_id(path, strat)


# =================================================================== 阶段实现
def parser_for(path: str | Path) -> Any:
    """按扩展名挑 parser。图片走 image_parser（延迟导入，没装 paddle 也能列目录）。"""
    suffix = Path(path).suffix.lower()
    kind = EXT_DISPATCH.get(suffix)
    if kind is None:
        raise ValueError(f"不支持的文件类型: {suffix or '(无扩展名)'}")
    if kind == "pdf":
        return parse_pdf
    if kind == "docx":
        return parse_docx
    if kind == "pptx":
        return parse_pptx
    if kind == "xlsx":
        return parse_xlsx
    if kind == "txt":
        return parse_txt
    if kind == "image":
        from ingestion.parsers.image_parser import parse_image

        return parse_image
    raise ValueError(f"未登记的解析器: {kind}")


def do_parse(
    file_path: str | Path,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
    strategy: str | DocIdStrategy = DEFAULT_ID_STRATEGY,
) -> tuple[dict[str, Any], str]:
    """
    parse 阶段：定 doc_id -> 查产物 -> 必要时解析并写 parsed/{doc_id}.json。
    返回 (中间 JSON, 状态)，状态 ∈ done/skipped。

    两种走法（差别全在「id 什么时候能算出来」）：
      BYTES / PATH_MTIME：先算 id 再查产物，产物在就整个阶段跳过，一次解析都不做；
      CONTENT：id 来自解析结果，只能先解析。为了不每次重跑都把 100GB 重新解析一遍，
      用 by-path 缓存（键含 size+mtime_ns）把命中路径变成「读一个小文件 + 读产物」。

    无论哪种策略，落盘前都把 doc_id 回填进 parsed：parser 内部算的是 bytes 口径，
    不回填的话 classify/chunk 会拿着错 id 写产物。原始 bytes id 留在 bytes_doc_id
    里便于回溯（图片块的 "<父id>:p3:x12" 那种位置编号就是由它派生的）。
    """
    p = Path(file_path)
    strat = normalize_strategy(strategy)
    data = read_bytes(p)

    if strat is DocIdStrategy.CONTENT:
        cached = read_cached_content_id(p, root)
        if cached and not force:
            target = artifact_path("parse", cached, root)
            if target.exists():
                return json.loads(target.read_text(encoding="utf-8")), STATUS_SKIPPED
        parsed = parser_for(p)(p)
        doc_id = compute_doc_id(p, strat, data=data, parsed=parsed)
        write_cached_content_id(p, doc_id, root)
    else:
        doc_id = compute_doc_id(p, strat, data=data)
        target = artifact_path("parse", doc_id, root)
        if target.exists() and not force:
            # 复用已落盘结果，不再解析（幂等的主战场）
            return json.loads(target.read_text(encoding="utf-8")), STATUS_SKIPPED
        parsed = parser_for(p)(p)

    parsed.setdefault("source_file", str(p))
    parsed["bytes_doc_id"] = str(parsed.get("doc_id") or sha256_hex(data))
    parsed["doc_id"] = doc_id
    parsed["doc_id_strategy"] = strat.value
    if strat is DocIdStrategy.CONTENT:
        # 记下图片块是靠哪一路证据算的（phash / clip / 退化文本），排查跨导出稳定性时用
        _fp, meta = content_fingerprint(parsed, data)
        parsed["content_fp"] = meta
    parsed["parsed_at"] = utc_now_iso()
    _atomic_write_text(artifact_path("parse", doc_id, root), json.dumps(parsed, ensure_ascii=False, indent=1))
    return parsed, STATUS_DONE


def do_classify(
    parsed: dict[str, Any],
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> str:
    """
    classify 阶段：用当前规则表重算 doc_type/topic/language/layout_style 并回写。

    为什么要独立成阶段（parser 里已经分类过一次）：规则表一定会改
    （新业务类型、新关键词），改完只想重跑分类不想重新解析 100GB 文档，
    这个阶段就是那个入口 —— --from-stage classify 直接读 parsed/*.json 重算。
    force=True 时覆盖旧判定（规则改了必须能生效），否则只补 unknown 字段。
    """
    metadata = parsed.get("metadata") or {}
    if metadata.get("classified_at") and not force:
        return STATUS_SKIPPED
    blocks = list(parsed.get("blocks") or [])
    full_text = "\n".join(str(b.get("content", "")) for b in blocks)
    result = classify(full_text, blocks)

    for key in ("doc_type", "topic", "language", "layout_style"):
        if force or metadata.get(key) in (None, "", "unknown"):
            metadata[key] = result[key]
    # 结构事实只累加不撤销：parser 数出来的「有几张图」比关键词可靠
    metadata["has_table"] = bool(metadata.get("has_table")) or bool(result["has_table"])
    metadata["has_image"] = bool(metadata.get("has_image")) or bool(result["has_image"])
    metadata["doc_type_scores"] = result.get("doc_type_scores", {})
    metadata["topic_scores"] = result.get("topic_scores", {})
    metadata["classified_at"] = utc_now_iso()
    parsed["metadata"] = metadata

    target = artifact_path("parse", str(parsed.get("doc_id") or ""), root)
    _atomic_write_text(target, json.dumps(parsed, ensure_ascii=False, indent=1))
    return STATUS_DONE


def do_chunk(
    parsed: dict[str, Any],
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """
    chunk 阶段：中间 JSON -> chunk 列表 -> chunks/{doc_id}.jsonl（一行一个 chunk）。
    返回 (chunk 数, 状态)。JSONL 而不是 JSON 数组：100GB 库里单文档可能有上万 chunk，
    流式读入索引阶段不用先把整个数组 load 进内存。
    """
    doc_id = str(parsed.get("doc_id") or "")
    target = artifact_path("chunk", doc_id, root)
    if target.exists() and target.stat().st_size > 0 and not force:
        # 数行数要用 with 接管句柄：Windows 下裸 open 不关会让后续覆盖写同一个文件失败
        with target.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip()), STATUS_SKIPPED
    chunks = chunk_document(parsed, resolve_config(cfg))
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
    _atomic_write_text(target, (lines + "\n") if lines else "")
    return len(chunks), STATUS_DONE


def do_embed(
    doc_id: str,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """
    embed 阶段：chunks/{doc_id}.jsonl -> 逐条编码 dense+sparse -> vectors/{doc_id}.jsonl。
    一行 = 一个 chunk 的全部原字段 + "dense"(list[float]) + "sparse"({indices,values}|None)。
    返回 (向量条数, 状态)。

    重模型惰性 import：只有真跑 embed 才加载 BGE-M3；离线只到 chunk 的自检不受影响，
    缺 FlagEmbedding 依赖时也不会让 pipeline 模块 import 失败。
    降级模型（bge-small-zh）只产 dense，sparse 字段写 None，index/检索侧按 None 兜底。
    """
    if not doc_id:
        raise RuntimeError("embed 需要 doc_id（CONTENT 策略请先跑 parse 定位 id）")
    chunks_file = artifact_path("chunk", doc_id, root)
    if not chunks_file.exists():
        raise RuntimeError(f"缺少上游产物 {chunks_file}；请先跑 --to-stage chunk")
    target = artifact_path("embed", doc_id, root)
    if target.exists() and target.stat().st_size > 0 and not force:
        with target.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip()), STATUS_SKIPPED

    chunks = list(iter_chunks(doc_id, root))
    if not chunks:
        # 空文档（没切出任何 chunk）：写空文件标记已完成，index 阶段读到空自然跳过
        _atomic_write_text(target, "")
        return 0, STATUS_DONE

    from embedding.text_embedder import encode_documents

    texts = [str(c.get("content", "")) for c in chunks]
    batch_size = (cfg or {}).get("embed_batch_size")
    embeddings = encode_documents(texts, batch_size=batch_size)

    lines: list[str] = []
    for chunk, emb in zip(chunks, embeddings, strict=True):
        record = dict(chunk)
        record["dense"] = emb["dense"]
        record["sparse"] = emb["sparse"]
        lines.append(json.dumps(record, ensure_ascii=False))
    _atomic_write_text(target, ("\n".join(lines) + "\n") if lines else "")
    return len(lines), STATUS_DONE


def do_index(
    doc_id: str,
    parsed: dict[str, Any],
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> str:
    """
    index 阶段：vectors/{doc_id}.jsonl -> Qdrant(dense+sparse) + SQLite(FTS5/ngram) + 元数据表。
    外部写入本身幂等（Qdrant point_id 由 chunk_id 决定、SQLite INSERT OR REPLACE），
    另用 vectors/{doc_id}.index 标记文件做「本文档已索引」判定，支持 --from-stage index 续跑。
    返回状态字符串。
    """
    if not doc_id:
        raise RuntimeError("index 需要 doc_id")
    vec_file = artifact_path("embed", doc_id, root)
    marker = artifact_path("index", doc_id, root)
    if marker.exists() and not force:
        return STATUS_SKIPPED
    if not vec_file.exists():
        raise RuntimeError(f"缺少上游产物 {vec_file}；请先跑 --to-stage embed")

    # 流式读向量文件：每行拆回 chunk（去掉 dense/sparse）与 embedding 两份对齐列表
    chunks: list[dict[str, Any]] = []
    embeddings: list[dict[str, Any]] = []
    with vec_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            embeddings.append({"dense": rec.pop("dense", []), "sparse": rec.pop("sparse", None)})
            chunks.append(rec)

    from index import keyword_store, lifecycle, metadata_store, ngram_index
    from index.vector_store import upsert_chunks

    keyword_store.init_tables()
    # 0) 先整体清掉本文档在索引层的旧数据（SQLite 四表 + Qdrant），再写新数据：
    #    重新切分会改变 chunk_id，只 upsert 会让旧 chunk 变成永久孤儿。Qdrant 不可达
    #    时 purge_index 内部降级为 warning（strict=False），本地先清后写照常完成。
    lifecycle.purge_index(doc_id, root=root, vectors=True, strict=False)
    # 1) 元数据登记（documents 表），供 list_documents / get_document_url 定位原文
    metadata_store.register_parsed_document(parsed, source_path=str(parsed.get("source_file", "")))
    if chunks:
        # 2) FTS5 关键词路（jieba 预分词写入，幂等覆盖）
        keyword_store.insert_chunks_batch(chunks)
        # 3) n-gram 模糊倒排
        ngram_index.build_ngram_index(chunks)
        # 4) 向量库：dense + sparse 挂同一 payload
        upsert_chunks(chunks, embeddings)

    _atomic_write_text(
        marker,
        json.dumps(
            {"doc_id": doc_id, "n_chunks": len(chunks), "indexed_at": utc_now_iso()},
            ensure_ascii=False,
        ),
    )
    return STATUS_DONE


# =================================================================== 单文件驱动
@dataclass
class FileTask:
    """
    一个文件的工作项。必须是普通 dataclass 且字段全部可 pickle，
    否则 Windows/Linux 的 spawn/fork 子进程拿不到（Pool 会静默失败）。
    """

    file: str
    from_stage: str = "parse"
    to_stage: str = "chunk"
    root: str = ""
    force: bool = False
    cfg: dict[str, Any] = field(default_factory=dict)
    id_strategy: str = DEFAULT_ID_STRATEGY.value


def process_file(task: FileTask) -> dict[str, Any]:
    """
    单文件跑完 [from_stage, to_stage] 区间。
    **任何异常都在此收口**，转成带 file/stage/traceback 的 report 返回，
    这样批量入库时一个坏文件不会拖垮整批，且日志里能直接看到是哪个路径炸的。
    """
    started = time.time()
    report: dict[str, Any] = {
        "file": task.file,
        "doc_id": None,
        "doc_id_strategy": normalize_strategy(task.id_strategy).value,
        "status": STATUS_DONE,
        "stages": {},
        "n_chunks": 0,
        "error": None,
        "failed_stage": None,
        "elapsed": 0.0,
    }
    root = task.root or None
    try:
        doc_id = peek_doc_id(Path(task.file), task.id_strategy, root)
        report["doc_id"] = doc_id
        stages = stage_range(task.from_stage, task.to_stage)
        first = stage_index(stages[0])

        # 前置依赖检查：从中间阶段起跑时，上游产物必须已存在，否则明确提示怎么补。
        # CONTENT 策略下 doc_id 可能还算不出来（没解析过、缓存也没有）—— 这时不能
        # 瞎猜 id，直接点名：这个策略必须先把 parse 跑一遍留下缓存。
        if first > stage_index("parse"):
            if doc_id is None:
                raise RuntimeError(
                    f"{report['doc_id_strategy']} 策略需要已有 parsed 产物才能定位 doc_id；"
                    f"请先对 {task.file} 跑 --from-stage parse"
                )
            if not artifact_path("parse", doc_id, root).exists():
                raise RuntimeError(
                    f"缺少上游产物 {artifact_path('parse', doc_id, root)}；请先跑 --from-stage parse"
                )

        parsed: dict[str, Any] | None = None
        for stage in stages:
            if not IMPLEMENTED.get(stage, False):
                report["stages"][stage] = STATUS_PENDING
                report["status"] = "partial"
                continue
            if stage == "parse":
                parsed, st = do_parse(task.file, root, task.force, task.id_strategy)
                report["stages"][stage] = st
                # CONTENT 策略的 id 是这一趟才算出来的，回填给 report（幂等判定要它）
                report["doc_id"] = str(parsed.get("doc_id") or doc_id)
            elif stage == "classify":
                if parsed is None:
                    parsed, _ = do_parse(task.file, root, force=False, strategy=task.id_strategy)
                report["stages"][stage] = do_classify(parsed, root, task.force)
            elif stage == "chunk":
                if parsed is None:
                    parsed, _ = do_parse(task.file, root, force=False, strategy=task.id_strategy)
                n_chunks, st = do_chunk(parsed, root, task.force, task.cfg)
                report["stages"][stage] = st
                report["n_chunks"] = n_chunks
            elif stage == "embed":
                if parsed is None:
                    parsed, _ = do_parse(task.file, root, force=False, strategy=task.id_strategy)
                real_doc_id = str(parsed.get("doc_id") or doc_id)
                n_vec, st = do_embed(real_doc_id, root, task.force, task.cfg)
                report["stages"][stage] = st
                report["n_vectors"] = n_vec
            elif stage == "index":
                if parsed is None:
                    parsed, _ = do_parse(task.file, root, force=False, strategy=task.id_strategy)
                real_doc_id = str(parsed.get("doc_id") or doc_id)
                report["stages"][stage] = do_index(real_doc_id, parsed, root, task.force)
        # 已完成阶段里若有 error 不可能走到这里（异常已冒到 except）
    except Exception as exc:
        report["status"] = STATUS_ERROR
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_stage"] = _guess_failed_stage(task, exc)
        report["traceback"] = traceback.format_exc()
    report["elapsed"] = round(time.time() - started, 3)
    return report


def _guess_failed_stage(task: FileTask, exc: BaseException) -> str:
    """
    猜失败发生在哪个阶段，便于按阶段筛错误。
    判据：产物已存在说明该阶段过了；缺上游产物 / 缺 CONTENT 缓存的 RuntimeError
    都归到 parse（这两种情况确实都是 parse 这一步的活没干）。
    """
    if isinstance(exc, RuntimeError) and ("缺少上游产物" in str(exc) or "需要已有 parsed 产物" in str(exc)):
        return "parse"
    try:
        doc_id = peek_doc_id(Path(task.file), task.id_strategy, task.root or None)
    except Exception:
        return "parse"
    if doc_id is None:
        return "parse"
    for stage in stage_range(task.from_stage, task.to_stage):
        if not artifact_path(stage, doc_id, task.root or None).exists():
            return stage
    return task.to_stage


# =================================================================== 进度条
def _iter_progress(iterable: Iterable[Any], total: int, desc: str) -> Iterable[Any]:
    """
    tqdm 进度条；未安装 tqdm 时退化为「每 50 个文件打一行」的纯文本进度。
    退化不是偷懒：pipeline 要在只装了基础包的机器上也能把 100 个文件跑完，
    进度显示不出来总比整条命令 ImportError 强。
    """
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="file", dynamic_ncols=True)
    except ImportError:
        pass

    def _plain() -> Iterable[Any]:
        for count, item in enumerate(iterable, start=1):
            if count == 1 or count % 50 == 0 or count == total:
                logger.info("%s: %d/%d", desc, count, total)
            yield item

    return _plain()


# =================================================================== 文件收集
def collect_files(input_path: str | Path, recursive: bool = True) -> list[Path]:
    """
    收集待处理文件：传目录就按扩展名白名单扫描（默认递归），传文件就直接返回它。
    跳过隐藏目录（.venv/.git）与产物目录本身（parsed/chunks/vectors/logs），
    判定看相对路径的**每一段**而不是只看父目录名 —— 否则
    docs/parsed/sub/x.pdf 这种嵌套结构会把已产出的文件再喂回去重跑。
    """
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"输入路径不存在：{p}")
    skip_dirs = artifact_dir_names()
    found: list[Path] = []
    walker = p.rglob("*") if recursive else p.glob("*")
    for item in walker:
        if not item.is_file():
            continue
        rel_parts = item.relative_to(p).parts[:-1]
        if any(part.startswith(".") or part in skip_dirs for part in rel_parts):
            continue
        if item.suffix.lower() in EXT_DISPATCH:
            found.append(item)
    return sorted(found)


def artifact_dir_names() -> set[str]:
    """本流水线自己的产物目录名（含 settings 里改过的名字），扫描输入时要按整段排除。"""
    names = set(STAGE_DIR_DEFAULT.values()) | {"logs"}
    node = get_settings().get("runtime") or {}
    if isinstance(node, dict):
        for key in ("parsed_dir", "chunks_dir", "vectors_dir", "log_dir"):
            value = str(node.get(key) or "").strip()
            if value:
                names.add(Path(value).name)
    return names


# =================================================================== 主入口
def run(
    input_path: str | Path,
    from_stage: str = "parse",
    to_stage: str = "index",
    id_strategy: str | DocIdStrategy = DEFAULT_ID_STRATEGY,
    workers: int | None = None,
    force: bool = False,
    root: str | os.PathLike[str] | None = None,
    recursive: bool = True,
    cfg: dict[str, Any] | None = None,
    fail_fast: bool | None = None,
    token_backend: str | None = None,
) -> dict[str, Any]:
    """
    流水线主入口：把 input_path 下的文件跑到 [from_stage, to_stage]。

    id_strategy ∈ {"bytes"(默认), "content", "path_mtime"}，取舍见文件头注释；
    **同一套库必须固定一种策略**，中途换策略 = 同一文档两个 doc_id 重复入库。

    并行用 multiprocessing.Pool（chunk_workers 来自 settings.active，缺省按 CPU 核数）；
    只有一个文件或 workers<=1 时走串行 —— 省掉 spawn 开销，更重要的是
    串行时异常栈直接进终端，单文件调试不用先去翻错误日志。
    返回汇总 dict（total/done/skipped/partial/error + 每文件 report），
    调用方（含自检脚本）据此断言，不靠人肉看输出。
    """
    strat = normalize_strategy(id_strategy)   # 非法策略在扫目录之前就炸掉，别跑一半才报错
    files = collect_files(input_path, recursive)
    summary: dict[str, Any] = {
        "input": str(input_path),
        "stages": stage_range(from_stage, to_stage),
        "id_strategy": strat.value,
        "total": len(files),
        "done": 0,
        "error": 0,
        "partial": 0,
        "fully_skipped": 0,
        "n_chunks": 0,
        "n_vectors": 0,
        "skipped_stages": [s for s in stage_range(from_stage, to_stage) if not IMPLEMENTED.get(s, False)],
        "token_backend": token_backend or token_estimator_name(),
        "reports": [],
    }
    if not files:
        logger.warning("没有可处理的文件：%s", input_path)
        return summary

    if token_backend:
        summary["token_backend"] = set_token_estimator(prefer=token_backend)

    resolved_root = output_root(root)
    # 只为「本批已实现」的阶段建目录：给 pending 阶段建出空 vectors/ 会让人误以为向量已经产出过
    for stage in summary["stages"]:
        if IMPLEMENTED.get(stage, False):
            stage_dir(stage, root)
    err_log = log_dir(root) / "pipeline_errors.jsonl"

    if fail_fast is None:
        node = get_settings().get("runtime") or {}
        fail_fast = bool(node.get("fail_fast", False)) if isinstance(node, dict) else False

    tasks = [
        FileTask(
            file=str(f),
            from_stage=from_stage,
            to_stage=to_stage,
            root=str(resolved_root),
            force=force,
            cfg=dict(cfg or {}),
            id_strategy=strat.value,
        )
        for f in files
    ]

    n_workers = _resolve_workers(workers, len(tasks))
    logger.info(
        "pipeline start stages=%s total=%d workers=%d id_strategy=%s token_backend=%s",
        "->".join(summary["stages"]),
        len(tasks),
        n_workers,
        summary["id_strategy"],
        summary["token_backend"],
    )
    started = time.time()
    if n_workers <= 1 or len(tasks) == 1:
        for report in _iter_progress((process_file(t) for t in tasks), len(tasks), "入库"):
            _fold_report(report, summary, err_log, fail_fast)
    else:
        from multiprocessing import Pool

        with Pool(processes=n_workers) as pool:
            for report in _iter_progress(
                pool.imap_unordered(process_file, tasks, chunksize=1), len(tasks), "入库"
            ):
                _fold_report(report, summary, err_log, fail_fast)

    summary["workers"] = n_workers
    summary["elapsed"] = round(time.time() - started, 2)
    logger.info(
        "pipeline done done=%d partial=%d error=%d n_chunks=%d n_vectors=%d elapsed=%.2fs",
        summary["done"],
        summary["partial"],
        summary["error"],
        summary["n_chunks"],
        summary["n_vectors"],
        summary["elapsed"],
    )
    _print_summary(summary)
    return summary


def _resolve_workers(workers: int | None, n_tasks: int) -> int:
    """
    决定进程数：显式参数 > settings.active.chunk_workers > CPU 核数。
    上限收紧到 min(配置, 任务数)，避免 4 个文件起 16 个进程的无谓开销。
    """
    if workers is not None:
        value = int(workers)
    else:
        active = get_settings().get("active") or {}
        value = int(active.get("chunk_workers") or 0) if isinstance(active, dict) else 0
    if value <= 0:
        value = min(os.cpu_count() or 1, 8)
    return max(1, min(value, n_tasks))


def _fold_report(report: dict[str, Any], summary: dict[str, Any], err_log: Path, fail_fast: bool) -> None:
    """把单文件结果并进汇总；错误写错误日志并按 fail_fast 决定是否中断整批。"""
    summary["reports"].append(report)
    status = report["status"]
    if status == STATUS_DONE:
        summary["done"] += 1
        # 全阶段 skipped = 这个文件本次一个字没干，是幂等性的直接证据（第二跑应 100% 落在这里）
        if report["stages"] and all(v == STATUS_SKIPPED for v in report["stages"].values()):
            summary["fully_skipped"] += 1
    elif status == "partial":
        summary["partial"] += 1
    elif status == STATUS_ERROR:
        summary["error"] += 1
        _append_jsonl(err_log, report)
        # 坏文件点名 ERROR（哪个路径、挂在哪阶段），单文件失败不拖垮整批
        logger.error(
            "file failed file=%s doc_id=%s stage=%s error=%s",
            report.get("file"),
            report.get("doc_id"),
            report.get("failed_stage"),
            report.get("error"),
        )
        if fail_fast:
            raise RuntimeError(f"文件处理失败（fail_fast）: {report['file']} @ {report['failed_stage']} -> {report['error']}")
    summary["n_chunks"] += int(report.get("n_chunks") or 0)
    summary["n_vectors"] += int(report.get("n_vectors") or 0)


def _print_summary(summary: dict[str, Any]) -> None:
    """收尾日志：总数/成功/失败/跳过，失败文件带完整路径，pending 阶段点名说明。

    名保留 _print_summary 但实际走 logger（stderr）：MCP stdio 下不能往 stdout 刷字。
    """
    logger.info(
        "汇总 输入=%s 阶段=%s 进程数=%s doc_id策略=%s token估算=%s",
        summary["input"],
        "->".join(summary["stages"]),
        summary.get("workers"),
        summary.get("id_strategy"),
        summary["token_backend"],
    )
    logger.info(
        "  文件 %d | 完成 %d（其中全部跳过 %d）| 部分完成 %d | 失败 %d "
        "| chunk 总数 %d | 向量总数 %d | 耗时 %ss",
        summary["total"],
        summary["done"],
        summary["fully_skipped"],
        summary["partial"],
        summary["error"],
        summary["n_chunks"],
        summary.get("n_vectors", 0),
        summary["elapsed"],
    )
    if summary["skipped_stages"]:
        logger.warning("以下阶段未实现已跳过：%s", "、".join(summary["skipped_stages"]))
    failed = [r for r in summary["reports"] if r["status"] == STATUS_ERROR]
    for report in failed[:20]:
        logger.error(
            "  [失败] %s @ %s -> %s", report["file"], report["failed_stage"], report["error"]
        )
    if len(failed) > 20:
        logger.error("  [失败] 其余 %d 个见日志", len(failed) - 20)


def rebuild_chunks(
    input_path: str | Path,
    root: str | os.PathLike[str] | None = None,
    workers: int | None = None,
    force: bool = True,
    id_strategy: str | DocIdStrategy = DEFAULT_ID_STRATEGY,
) -> dict[str, Any]:
    """
    便捷入口：只重切 chunk（复用已有 parsed/*.json）。
    切分参数调优时用，不用重新解析整库（parse 是整条链路最慢的一段）。
    注意 CONTENT 策略下「复用 parsed」是靠 by-path 缓存实现的，缓存没了就得先跑 parse。
    """
    return run(input_path, "chunk", "chunk", id_strategy, workers=workers, force=force, root=root)


def iter_chunks(doc_id: str, root: str | os.PathLike[str] | None = None) -> Iterable[dict[str, Any]]:
    """流式读某个文档的 chunk（第 4 批 embedding / 索引阶段读的就是这个文件）。"""
    path = artifact_path("chunk", doc_id, root)
    if not path.exists():
        return []

    def _read() -> Iterable[dict[str, Any]]:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return _read()


# =================================================================== CLI
def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="python -m ingestion.pipeline",
        description="文档入库流水线：parse -> classify -> chunk（embed/index 属第 4 批）",
    )
    cli.add_argument("--input", required=True, help="文件或目录路径")
    # --from / --to 是需求里给出的命令写法，--from-stage / --to-stage 是参数说明里的全称，
    # 两个都接受（dest 统一），避免用户按任一文档来敲都会报错
    cli.add_argument(
        "--from", "--from-stage", dest="from_stage", default="parse", choices=list(STAGES), help="起始阶段"
    )
    cli.add_argument(
        "--to", "--to-stage", dest="to_stage", default="index", choices=list(STAGES), help="结束阶段"
    )
    cli.add_argument("--workers", type=int, default=None, help="并行进程数（默认取 profile.chunk_workers）")
    cli.add_argument("--force", action="store_true", help="忽略已存在产物，强制重跑各阶段")
    # 同一套库必须固定一种策略；换了策略要清库重建，所以这里只在启动时提示，不做自愈
    cli.add_argument(
        "--id-strategy",
        default=DEFAULT_ID_STRATEGY.value,
        choices=[s.value for s in DocIdStrategy],
        help=(
            "doc_id 生成策略：bytes=sha256(文件字节)（默认，需求契约，重导出会变 id）；"
            "content=解析内容规范化哈希（跨导出稳定，要先 parse）；"
            "path_mtime=绝对路径+size+mtime_ns（最快，移动即失效，仅本地）"
        ),
    )
    cli.add_argument("--out-root", default=None, help="产物根目录（默认 settings.runtime.output_root 或当前目录）")
    cli.add_argument("--no-recursive", action="store_true", help="目录输入时不递归")
    cli.add_argument(
        "--token-backend", default=None, choices=["auto", "jieba", "char"], help="token 估算口径（默认自动）"
    )
    cli.add_argument("--chunk-size", type=int, default=None, help="覆盖 chunking.chunk_size")
    cli.add_argument("--chunk-overlap", type=int, default=None, help="覆盖 chunking.chunk_overlap")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    # CLI 独立跑时自行初始化日志（走 stderr）；被 MCP ingest 调用时 server 已初始化，重复调用幂等。
    from config.logging_setup import setup_logging

    setup_logging()
    args = build_cli().parse_args(argv)
    cfg: dict[str, Any] = {}
    if args.chunk_size:
        cfg["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        cfg["chunk_overlap"] = args.chunk_overlap
    try:
        summary = run(
            args.input,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            id_strategy=args.id_strategy,
            workers=args.workers,
            force=args.force,
            root=args.out_root,
            recursive=not args.no_recursive,
            cfg=cfg,
            token_backend=args.token_backend,
        )
    except Exception as exc:
        logger.exception("致命错误: %s: %s", type(exc).__name__, exc)
        return 2
    # 有文件失败就返回 1，方便 shell/CI 里判断这批有没有漏入库的文件
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
