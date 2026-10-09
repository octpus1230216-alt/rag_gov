# -*- coding: utf-8 -*-
"""
图片类型 zero-shot 分类（Chinese-CLIP 文本塔 vs 图片塔余弦相似度）。

类别：chart / photo / diagram / signature / form，置信度 < 0.4 标 unknown。

实现要点：
  1. 模型加载完全复用 image_parser.get_clip() 的进程内单例，
     本模块不会额外加载第二份权重（100GB 库里图片极多，多加载一份就多点内存）。
  2. 每个类别用多条中文 prompt 求平均文本向量：单条 prompt 的用词偏差
     会让类别间相似度排序不稳定，多 prompt 平均是 zero-shot 的标准做法。
  3. softmax 温度用 open_clip 的 logit_scale；类别只有 5 个，
     不做温度修正会让置信度普遍偏低，全部落到 unknown。
  4. 模型不可用时返回 label=unknown + error 字段，绝不返回猜测结果。

分类结果的用途（下一批之后）：
  - chart/form 的图片走 OCR 文本检索为主，CLIP 向量为辅；
  - photo 的图片只有 CLIP 向量有意义（OCR 往往为空）；
  - signature 默认不入库内容，只记「存在签名」这一事实（合规考虑）。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

# 置信度阈值：低于它宁可 unknown
MIN_CLASS_CONFIDENCE = 0.4

# 类别顺序即输出向量维度顺序，改动会让已入库的分类结果失效，谨慎调整
IMAGE_CATEGORIES: tuple[str, ...] = ("chart", "photo", "diagram", "signature", "form")

CATEGORY_LABELS_ZH: dict[str, str] = {
    "chart": "图表",
    "photo": "照片",
    "diagram": "示意图",
    "signature": "签名",
    "form": "表单",
    "unknown": "未分类",
}

# 每类多条中文提示句（Chinese-CLIP 的文本塔是中文语料训练的，英文 prompt 效果明显更差）
CATEGORY_PROMPTS: dict[str, tuple[str, ...]] = {
    "chart": (
        "一张统计图表",
        "一张柱状图或折线图",
        "一张带坐标轴和数字的数据图",
        "一张财务报表截图",
    ),
    "photo": (
        "一张实景照片",
        "一张人物或场景的彩色照片",
        "一张产品实物照片",
        "一张会议现场拍摄的照片",
    ),
    "diagram": (
        "一张流程示意图",
        "一张组织架构框图",
        "一张系统原理图",
        "一张带箭头和方框的示意图",
    ),
    "signature": (
        "一个手写签名",
        "一张签字页面",
        "手写笔迹",
        "合同末尾的签名与日期",
    ),
    "form": (
        "一张填写好的表单",
        "一张表格扫描件",
        "一张发票或单据",
        "一张带勾选框和填空的申请表",
    ),
}


def _clip_singleton() -> dict[str, Any] | None:
    """
    取 image_parser 里的 CLIP 单例。
    放在函数内 import 是为了避开 image_parser <-> image_classifier 的模块级循环导入。
    """
    from ingestion.parsers.image_parser import get_clip

    return get_clip()


def category_prompts() -> tuple[list[str], list[str]]:
    """展开成 (prompt 列表, 每个 prompt 对应类别)，保持与 IMAGE_CATEGORIES 顺序一致。"""
    flat: list[str] = []
    owner: list[str] = []
    for cat in IMAGE_CATEGORIES:
        prompts = CATEGORY_PROMPTS[cat]
        flat.extend(prompts)
        owner.extend([cat] * len(prompts))
    return flat, owner


def classify_image_bytes(data: bytes, k: int | None = None) -> dict[str, Any]:
    """
    对图片字节做 zero-shot 分类。

    参数 k：保留的 topk 数量（默认只取最优）。返回：
      {"label", "confidence", "label_zh", "top": [(label, prob)...], "error"?}
    置信度 < MIN_CLASS_CONFIDENCE 时 label 变为 unknown，但 top 仍保留原判定，
    便于离线调阈值时看清楚「差一点就命中」的样本有多少。
    """
    clip = _clip_singleton()
    if clip is None:
        from ingestion.parsers.image_parser import clip_load_error

        return {
            "label": "unknown",
            "label_zh": CATEGORY_LABELS_ZH["unknown"],
            "confidence": 0.0,
            "top": [],
            "error": clip_load_error() or "Chinese-CLIP 不可用",
        }
    try:
        import torch

        prompts, owner = category_prompts()
        with clip["Image"].open(io.BytesIO(data)) as im:
            tensor = clip["preprocess"](im.convert("RGB")).unsqueeze(0).to(clip["device"])
        tokens = clip["tokenizer"](prompts).to(clip["device"])
        with torch.no_grad():
            img_feat = clip["model"].encode_image(tensor)
            txt_feat = clip["model"].encode_text(tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # 逐类聚合：同类多 prompt 的相似度取均值（比 max 更稳，不会被某条宽泛 prompt 主导）
        sims = []
        for cat in IMAGE_CATEGORIES:
            idx = [i for i, c in enumerate(owner) if c == cat]
            per_prompt = (img_feat @ txt_feat[idx].T).squeeze(0)
            sims.append(float(per_prompt.mean()))
        sims_t = torch.tensor(sims)
        # 温度对齐 open_clip 的训练尺度，否则 softmax 在 5 类上过于平坦
        logit_scale = float(getattr(clip["model"], "logit_scale", torch.tensor(4.6)).item())
        probs = torch.softmax(sims_t * min(max(logit_scale, 1.0), 12.0), dim=-1).tolist()

        ranked = sorted(
            zip(IMAGE_CATEGORIES, probs, strict=False), key=lambda kv: kv[1], reverse=True
        )[: (k or 1)]
        best_label, best_prob = ranked[0]
        label = best_label if best_prob >= MIN_CLASS_CONFIDENCE else "unknown"
        return {
            "label": label,
            "label_zh": CATEGORY_LABELS_ZH[label],
            "confidence": round(best_prob, 4),
            "predicted": best_label,
            "top": [(name, round(p, 4)) for name, p in ranked],
        }
    except Exception as exc:
        return {
            "label": "unknown",
            "label_zh": CATEGORY_LABELS_ZH["unknown"],
            "confidence": 0.0,
            "top": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def classify_image_path(path: str | Path) -> dict[str, Any]:
    """独立入口：喂图片路径，返回 {label, confidence, ...}。"""
    p = Path(path)
    return classify_image_bytes(p.read_bytes())


def is_vector_like(result: dict[str, Any]) -> bool:
    """
    供入库批次判断：这张图的内容是否值得进图片向量 collection。
    signature 只记事实、不进向量库（避免把签名当语义内容检索）。
    """
    return result.get("label") not in (None, "signature")


if __name__ == "__main__":
    import argparse
    import json

    cli = argparse.ArgumentParser(description="Chinese-CLIP zero-shot 图片分类")
    cli.add_argument("path", help="图片路径")
    cli.add_argument("--topk", type=int, default=3, help="输出前 k 个类别")
    args = cli.parse_args()
    out = classify_image_path(args.path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
