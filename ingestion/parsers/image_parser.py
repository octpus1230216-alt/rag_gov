# -*- coding: utf-8 -*-
"""
图片解析：PaddleOCR 抽文字 + Chinese-CLIP 出向量 + zero-shot 图片分类。

设计要点：
  1. 模型一律「懒加载 + 进程内单例」。import 本模块不加载任何权重，
     这样 multiprocessing.Pool 的每个子进程只在自己真正遇到图片时才付加载成本。
  2. 权重加载失败时绝不伪造向量：返回 None 并把错误写进 result["model_errors"]，
     让入库批次能把这些文档标成「待补 embedding」而不是静默入库成垃圾。
  3. 本模块同时被 pdf/docx/pptx parser 复用（它们抽出内嵌图片字节后直接调
     parse_image_bytes），保证图片 block 的字段口径全库一致。

依赖说明：
  - paddleocr / open_clip / torch / PIL 全部在函数内部 import
  - Chinese-CLIP 权重地址由配置驱动（embedding.image_clip_pretrained），
    默认走 HF hub 的 OFASys 官方仓库；离线机器改成本地 .pt 路径即可
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 图片 zero-shot 分类的类别表 / 阈值 / 实现统一放在 classifiers.image_classifier，
# 这里只做导入复用，避免同一套 CLIP 提示词在两处各写一遍而走样
from ingestion.classifiers.image_classifier import classify_image_bytes
from ingestion.parsers.common import (
    detect_language,
    dump_json_for_cli,
    get_settings,
    new_block,
    new_document,
    normalize_text,
    parsing_option,
    sha256_hex,
    utc_now_iso,
)

_OCR_SINGLETON: Any = None
_OCR_FAILED: str | None = None
_CLIP_SINGLETON: dict[str, Any] | None = None
_CLIP_FAILED: str | None = None


# =================================================================== OCR
def get_ocr() -> Any:
    """
    PaddleOCR 单例（ch_PP-OCRv4 / lang=ch）。
    返回 None 表示不可用，调用方必须能容忍「OCR 文本为空」。
    """
    global _OCR_SINGLETON, _OCR_FAILED
    if _OCR_SINGLETON is not None or _OCR_FAILED is not None:
        return _OCR_SINGLETON
    try:
        from paddleocr import PaddleOCR

        _OCR_SINGLETON = PaddleOCR(
            lang=parsing_option("ocr.lang", "ch"),
            ocr_version=parsing_option("ocr.model", "ch_PP-OCRv4"),
            use_angle_cls=parsing_option("ocr.use_angle_cls", True),
            show_log=False,
            # det+rec 都要：扫描件只有先检测出文本框，才能给每个框 bbox
            det=parsing_option("ocr.det", True),
        )
    except Exception as exc:  # 未安装 / 权重缺失 / so 加载失败
        _OCR_FAILED = f"{type(exc).__name__}: {exc}"
        _OCR_SINGLETON = None
    return _OCR_SINGLETON


def ocr_image_bytes(data: bytes) -> tuple[str, list[dict[str, Any]], str | None]:
    """
    对图片字节做 OCR。

    返回 (拼平的文本, [{text, bbox, conf}], 错误信息)。
    bbox 是像素坐标（原图尺度），入库后用于「点选跳转」，不与 PDF 的 pt 坐标混用。
    """
    ocr = get_ocr()
    if ocr is None:
        return "", [], _OCR_FAILED or "PaddleOCR 不可用"
    try:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            arr = np.array(im.convert("RGB"))
        raw = ocr.ocr(arr, cls=True)
    except Exception as exc:
        return "", [], f"OCR 执行失败: {type(exc).__name__}: {exc}"

    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    # paddleocr 2.x 返回 [[ [四点框, (文本, 置信度)], ... ]]；3.x 结构略有差异，两种都兜
    pages = raw or []
    for page in pages if isinstance(pages, list) else []:
        items = page if isinstance(page, list) else [page]
        for item in items:
            try:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    poly, rec = item
                elif isinstance(item, dict):
                    poly, rec = item.get("box"), (item.get("text"), item.get("confidence", 1.0))
                else:
                    continue
                text, conf = (rec[0], float(rec[1])) if isinstance(rec, (list, tuple)) else (str(rec), 1.0)
            except Exception:
                continue
            if not text:
                continue
            texts.append(str(text))
            lines.append(
                {
                    "text": normalize_text(str(text)),
                    "bbox": _poly_to_bbox(poly),
                    "conf": round(conf, 3),
                }
            )
    return normalize_text("\n".join(texts)), lines, None


def _poly_to_bbox(poly: Any) -> list[float] | None:
    """四点框 / 矩形 -> [x0,y0,x1,y1]（像素）。"""
    try:
        if poly is None:
            return None
        if isinstance(poly, (int, float)):
            return None
        xs = []
        ys = []
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    except (TypeError, ValueError):
        return None
    return None


# =================================================================== CLIP
def get_clip() -> dict[str, Any] | None:
    """
    Chinese-CLIP 单例。返回 {"model","preprocess","tokenizer","device"}，失败返回 None。

    配置项（settings.embedding）：
      image_clip_model_name : open_clip 结构名，默认 ViT-B-16
      image_clip_pretrained : 权重来源。默认 hf-hub:OFA-Sys/chinese-clip-vit-base-patch16；
                              离线机器填本地 .pt 路径
    """
    global _CLIP_SINGLETON, _CLIP_FAILED
    if _CLIP_SINGLETON is not None or _CLIP_FAILED is not None:
        return _CLIP_SINGLETON
    try:
        import open_clip
        import torch
        from PIL import Image
        from torchvision import transforms

        name = _embedding_option("image_clip_model_name", "ViT-B-16")
        pretrained = _embedding_option(
            "image_clip_pretrained", "hf-hub:OFA-Sys/chinese-clip-vit-base-patch16"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(name)
        img_size = int(getattr(model.visual, "image_size", 224) or 224)
        preprocess = transforms.Compose(
            [
                transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )
        _CLIP_SINGLETON = {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "device": device,
            "torch": torch,
            "Image": Image,
        }
    except Exception as exc:
        _CLIP_FAILED = f"{type(exc).__name__}: {exc}"
        _CLIP_SINGLETON = None
    return _CLIP_SINGLETON


def _embedding_option(key: str, default: Any) -> Any:
    """取 settings.embedding.<key>（CLIP 参数放在 embedding 段而非 parsing 段）。"""
    node = get_settings().get("embedding") or {}
    return node.get(key, default)


def encode_image_bytes(data: bytes) -> tuple[list[float] | None, str | None]:
    """
    图片 -> CLIP dense 向量（已 L2 归一化，配合 cosine）。
    返回 (向量, 错误信息)。模型不可用时向量为 None，由入库批次决定跳过还是标记待补。
    """
    clip = get_clip()
    if clip is None:
        return None, _CLIP_FAILED or "Chinese-CLIP 不可用"
    try:
        torch = clip["torch"]
        with clip["Image"].open(io.BytesIO(data)) as im:
            tensor = clip["preprocess"](im.convert("RGB")).unsqueeze(0).to(clip["device"])
        with torch.no_grad():
            feat = clip["model"].encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return [float(x) for x in feat.squeeze(0).tolist()], None
    except Exception as exc:
        return None, f"CLIP 编码失败: {type(exc).__name__}: {exc}"


def clip_load_error() -> str | None:
    """CLIP 加载失败原因（供 image_classifier 在降级路径里回填 error）。"""
    return _CLIP_FAILED


# classify_image_bytes 由 ingestion.classifiers.image_classifier 提供（模块顶部已导入）


# =================================================================== parser
def image_size(data: bytes) -> tuple[int, int] | None:
    """读图片宽高（像素）；损坏/非图片字节返回 None。"""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return (int(im.size[0]), int(im.size[1]))
    except Exception:
        return None


def min_side_ok(data: bytes, min_side: int | None = None) -> bool:
    """
    尺寸过滤：边长过小的图标、logo、装饰线不入库（100GB 库里这类碎片占图片总量一半以上）。
    读不到尺寸（损坏图）时保守放行，让 OCR 去判定。
    """
    min_side = min_side if min_side is not None else int(parsing_option("image_min_side_px", 120))
    size = image_size(data)
    if size is None:
        return True
    return min(size) >= min_side


def parse_image_bytes(
    data: bytes,
    source_format: str = "jpg",
    caption: str = "",
    page: int | None = None,
    bbox: Sequence[float] | None = None,
    section_path: str = "",
    doc_id: str | None = None,
    min_side: int | None = None,
) -> dict[str, Any] | None:
    """
    图片字节 -> 统一中间 JSON（单个 image block）。

    content = caption + OCR 文本（caption 在前，因为它通常是人工/正文给出的语义摘要，
    权重高于 OCR 出来的碎片字）；两者都为空时写诚实占位串，绝不产出空块。
    附带 ocr_lines / clip_vector / classification 供后续批次使用。
    太小被过滤掉时返回 None。
    """
    if not data:
        return None
    if not min_side_ok(data, min_side):
        return None

    ocr_text, ocr_lines, ocr_error = ocr_image_bytes(data)
    vector, encode_error = encode_image_bytes(data)
    classification = classify_image_bytes(data)

    parts = [p for p in (normalize_text(caption), ocr_text) if p]
    content = "\n".join(parts)
    errors = [e for e in (ocr_error, encode_error, classification.get("error")) if e]

    # 既没图注也没 OCR 结果（模型未加载 / 纯照片 / 空白扫描件）时给一个诚实占位：
    # 空 content 的块一旦入库，dense 命中后无法展示、FTS5 索引不到，
    # 比一条标注清楚「这里没有可读文字」的占位块更糟。
    placeholder = False
    if not content:
        w, h = image_size(data) or (0, 0)
        content = f"[图片 {w}x{h}px：无图注，OCR 未识别到文字]"
        placeholder = True

    block = new_block(
        "image",
        content,
        page=page,
        bbox=bbox,
        section_path=section_path,
        ocr_text=ocr_text,
        ocr_lines=ocr_lines,
        clip_vector=vector,
        classification=classification,
        content_placeholder=placeholder or None,
    )
    return new_document(
        doc_id=doc_id or compute_doc_id_from_bytes(data),
        source_format=source_format if source_format in ("jpg", "png") else "jpg",
        metadata={
            "doc_type": "unknown",
            "topic": "unknown",
            "has_table": False,
            "has_image": True,
            "language": _language_from_ocr(ocr_text),
            "layout_style": "image",
        },
        blocks=[block],
        parsed_at=utc_now_iso(),
        model_errors=errors or None,
    )


def compute_doc_id_from_bytes(data: bytes) -> str:
    """图片单独成文档时的 doc_id（内容哈希），与 parser 层的 sha256(file_bytes) 口径一致。"""
    return sha256_hex(data)


def _language_from_ocr(ocr_text: str) -> str:
    """OCR 文本里中文占比决定 language；无文字时给 mixed（没有足够证据判单语）。"""
    if not ocr_text:
        return "mixed"
    return detect_language(ocr_text)


def parse_image(path: str | Path, caption: str = "") -> dict[str, Any]:
    """
    独立入口：喂一张图片路径，输出统一 dict。
    被过滤（尺寸过小）时返回一个 blocks 为空的文档，而不是抛异常。
    """
    p = Path(path)
    data = p.read_bytes()
    fmt = "png" if p.suffix.lower() == ".png" else "jpg"
    result = parse_image_bytes(data, source_format=fmt, caption=caption or p.stem)
    if result is None:
        result = new_document(
            doc_id=sha256_hex(data),
            source_format=fmt,
            metadata={"has_image": True, "layout_style": "image", "language": "mixed"},
            blocks=[],
            parsed_at=utc_now_iso(),
            skipped_reason="图片尺寸小于 image_min_side_px 阈值，判定为装饰图",
        )
    return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="解析单张图片，输出统一中间 JSON")
    parser.add_argument("path", help="图片路径（jpg/png）")
    parser.add_argument("--caption", default="", help="外部提供的图注，拼进 content 头部")
    args = parser.parse_args()
    print(dump_json_for_cli(parse_image(args.path, args.caption)))
    sys.exit(0)
