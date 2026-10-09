# -*- coding: utf-8 -*-
"""
查询路由：根据 query 特征决定四路召回的权重分配。

路由原则：
  - 精确特征（引号、编号、法条、日期、金额）→ 提高 exact/sparse 权重，压低 dense；
  - 模糊特征（短查询）→ 提高 fuzzy 权重；
  - 语义特征（疑问句/长文本）→ 提高 dense 权重；
  - 路由结果只影响 RRF 融合权重，不决定「跑不跑某路」—— 四路都跑，只是权重不同。

多条规则命中时按优先级取第一条（越具体的模式越靠前），不做权重叠加
（叠加会导致精确路被推太高、语义路消失，违反「四路都跑」原则）。
"""

from __future__ import annotations

import re

from ingestion.parsers.common import get_settings

# =================================================================== 正则模式

# 引号（中文双引号""、中文单引号「」『』、英文双引号""）
_RE_QUOTE = re.compile(r'["\u201c\u300c\u300e].*?["\u201d\u300d\u300f]|"[^"]{2,}"')

# 法条/章节编号（第X条/章/节/款/项）
_RE_ARTICLE = re.compile(r"第[一二三四五六七八九十百千\d]+[条章节点项]")

# 日期（YYYY-MM-DD / YYYY年MM月DD日 / YYYY/MM/DD）
_RE_DATE = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}[日号]"
)

# 金额（¥/$ + 数字，或 万/亿 后缀）
_RE_AMOUNT = re.compile(
    r"[¥$]\s*[\d,.]+|[\d,.]+\s*[万亿]?\s*(?:元|美元|人民币|块)"
)

# 大写英文+数字编号（ABC-123 / XLS-2024-001）
_RE_CODE = re.compile(r"[A-Z]{2,}[-_]?\d{2,}")

# 疑问句（? 或 ？ 结尾，允许尾部空白）
_RE_QUESTION = re.compile(r"[?？]\s*$")


# =================================================================== 路由函数
def route(query: str) -> dict[str, float]:
    """
    输入 query 字符串，返回四路权重字典。
    keys 固定为 {"exact", "fuzzy", "sparse", "dense"}，value ∈ (0, 1.0]。

    优先级（从高到低）：
      1. 引号
      2. 法条编号
      3. 日期
      4. 金额
      5. 大写编号
      6. 短查询（<4 字）
      7. 疑问句 或 长文本（>30 字）
      8. 默认
    """
    stripped = query.strip()
    # 去引号后的有效长度
    content_len = len(_RE_QUOTE.sub("", stripped).strip()) or len(stripped)

    # 1) 引号 → 用户要精确匹配这段文字
    if _RE_QUOTE.search(stripped):
        return {"exact": 1.0, "fuzzy": 0.3, "sparse": 0.5, "dense": 0.5}

    # 2) 法条/章节编号 → 必须 exact 命中编号 token
    if _RE_ARTICLE.search(stripped):
        return {"exact": 1.0, "fuzzy": 0.5, "sparse": 1.0, "dense": 0.3}

    # 3) 日期 → 精确匹配数字序列
    if _RE_DATE.search(stripped):
        return {"exact": 1.0, "fuzzy": 0.5, "sparse": 0.8, "dense": 0.3}

    # 4) 金额 → 精确匹配数字+单位
    if _RE_AMOUNT.search(stripped):
        return {"exact": 1.0, "fuzzy": 0.5, "sparse": 0.8, "dense": 0.3}

    # 5) 大写英文+数字编号 → 合同号/项目编号
    if _RE_CODE.search(stripped):
        return {"exact": 1.0, "fuzzy": 0.5, "sparse": 1.0, "dense": 0.3}

    # 6) 短查询（<4 字）→ 关键词太少，语义路信心不足，靠 fuzzy 扩展
    if content_len < 4:
        return {"exact": 1.0, "fuzzy": 1.0, "sparse": 0.8, "dense": 0.5}

    # 7) 疑问句或长文本 → 自然语言提问，语义理解优先
    if _RE_QUESTION.search(stripped) or len(stripped) > 30:
        return {"exact": 0.3, "fuzzy": 0.5, "sparse": 0.8, "dense": 1.0}

    # 8) 默认
    return {"exact": 1.0, "fuzzy": 0.8, "sparse": 1.0, "dense": 1.0}


def get_route_candidates() -> dict[str, int]:
    """从 settings.retrieval.route_candidates 取各路召回数量。"""
    rcfg = get_settings().get("retrieval") or {}
    candidates = rcfg.get("route_candidates") or {}
    return {
        "exact": int(candidates.get("exact") or 50),
        "fuzzy": int(candidates.get("fuzzy") or 20),
        "sparse": int(candidates.get("sparse") or 50),
        "dense": int(candidates.get("dense") or 50),
    }
