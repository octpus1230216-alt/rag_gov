# -*- coding: utf-8 -*-
"""
图片向量化：Chinese-CLIP（OFA-Sys/chinese-clip-vit-base-patch16）。

用途：
  - encode_image：对图片文件编码为 512 维 dense 向量，写入 Qdrant 图片 collection；
  - encode_text：对文本编码到同一向量空间，实现「图文对齐」——
    用文字 query 可以直接搜图片向量（跨模态检索）。

设计要点：
  1. 单例（lru_cache）：CLIP 模型权重约 600MB，加载一次就够；
  2. 图片预处理：用 open_clip 的 transform（resize + center_crop + normalize），
     与训练时一致，手动 resize 会破坏对齐效果；
  3. 输出已 L2 归一化：配合 cosine 时点积就是相似度分数，无需下游再处理；
  4. device / dtype 从 settings.active 读取，GPU 可用时自动走 fp16 加速。
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

from ingestion.parsers.common import get_settings

# CLIP 模型标识（升级需改这里 + 重建图片向量库）
CLIP_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
CLIP_IMAGE_DIM = 512


class ImageEmbedder:
    """Chinese-CLIP 封装：图片/文本 -> 同空间向量。"""

    def __init__(self) -> None:
        active = get_settings().get("active") or {}
        self._device: str = str(active.get("device") or "cpu")
        self._use_fp16: bool = bool(active.get("use_fp16", False))
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """懒加载模型权重。"""
        if self._model is not None:
            return
        import open_clip
        import torch

        # 实际加载时如果 GPU 不可用会自动退化到 CPU
        device = self._device
        if device in ("auto", "cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained="cn_clip",
            device=device,
        )
        self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self._model.eval()

        # 混合精度：GPU + fp16 时节省显存，CPU 上强制关
        if self._use_fp16 and device != "cpu":
            self._model = self._model.half()
        else:
            self._use_fp16 = False

    def encode_image(self, path: str | Path | bytes) -> list[float]:
        """
        图片文件路径或字节 -> 512 维 L2 归一化向量。
        支持 jpg/png/webp（Pillow 能打开的格式均可）。
        """
        self._load()
        import torch
        from PIL import Image

        if isinstance(path, (str, Path)):
            img = Image.open(path).convert("RGB")
        elif isinstance(path, bytes):
            img = Image.open(io.BytesIO(path)).convert("RGB")
        else:
            raise TypeError(f"encode_image 需要路径或 bytes，收到 {type(path)}")

        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        dtype = torch.float16 if self._use_fp16 else torch.float32
        tensor = tensor.to(dtype=dtype)

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            # L2 归一化：CLIP 原文用 normalize 后的特征做 cosine
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).float().cpu().tolist()

    def encode_text(self, text: str) -> list[float]:
        """
        文本 -> 与图片同一向量空间的 512 维向量（图文对齐检索用）。
        中文 CLIP 原生支持中文 tokenize，不需要额外翻译。
        """
        self._load()
        import torch

        tokens = self._tokenizer([text]).to(self._device)
        if self._use_fp16:
            tokens = tokens.half()

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).float().cpu().tolist()

    def encode_text_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本编码（入库时 caption/OCR 文本对齐用）。"""
        if not texts:
            return []
        self._load()
        import torch

        tokens = self._tokenizer(texts).to(self._device)
        if self._use_fp16:
            tokens = tokens.half()

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.float().cpu().tolist()


# =================================================================== 单例入口
@lru_cache(maxsize=1)
def get_image_embedder() -> ImageEmbedder:
    """全局单例：同进程内模型只加载一次。"""
    return ImageEmbedder()


# =================================================================== 便捷函数
def encode_image(path: str | Path | bytes) -> list[float]:
    """编码单张图片 -> 512 维向量。"""
    return get_image_embedder().encode_image(path)


def encode_text(text: str) -> list[float]:
    """编码文本到图片向量空间（跨模态检索）。"""
    return get_image_embedder().encode_text(text)
