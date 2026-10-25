# -*- coding: utf-8 -*-
"""
可答性判断：在调用 LLM 之前先判「检索到的资料够不够回答问题」。

为什么需要这一层（防幻觉的关键闸门）：
  检索永远返回 top-k，哪怕库里根本没有答案——如果不设闸，LLM 会拿一堆不相干
  片段硬编出看似合理的答案。这里用检索置信度做「宁可拒答也不瞎答」的判断，
  不可答时直接返回拒答话术、根本不费 token 去调 LLM。

默认 rule 策略（零额外成本）：
  - context 为空 / 无来源                -> 不可答（empty）
  - 有 rerank 分数且最高分 < 阈值         -> 不可答（low_confidence）
  - 未开 rerank（只有 rrf 名次分）         -> 无法用同一尺度判，放宽为「有资料即可能答」
预留 llm_judge 策略（更准但要额外调用，本批只留入口不实现，避免默认多花钱）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("rag_zh.generation.answerability")

DEFAULT_MIN_SCORE = 0.30
DEFAULT_REFUSAL = "根据知识库现有内容无法可靠回答该问题。"


def _max_score(sources: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[float, bool]:
    """
    取所有来源/结果里的最高置信分，返回 (最高分, 是否有 rerank 分)。
    优先 rerank_score（交叉编码器 0-1 归一分），退化到 rrf_score（名次融合分，尺度不同）。
    """
    best = 0.0
    has_rerank = False
    for item in list(sources) + list(results):
        rs = item.get("rerank_score")
        if rs is not None:
            has_rerank = True
            best = max(best, float(rs))
            continue
        sc = item.get("score")
        if sc is not None:
            best = max(best, float(sc))
    return best, has_rerank


def assess(
    search_result: dict[str, Any], cfg: dict[str, Any] | None = None
) -> tuple[bool, float, str]:
    """
    判断这次检索能否可靠作答。

    参数：
      search_result: retrieval.hybrid_search.search() 的返回
                     （含 {"results": [...], "context": {"context","sources","truncated"}}）
      cfg:           settings.generation 段（读其中的 answerability 子段）

    返回：(answerable, confidence, reason)
      reason 供上层选择拒答话术 / 日志排查，不直接展示给终端用户。
    """
    cfg = cfg or {}
    acfg = cfg.get("answerability") or {}
    min_score = float(acfg.get("min_rerank_score") or DEFAULT_MIN_SCORE)
    require_non_empty = bool(acfg.get("require_non_empty_context", True))
    strategy = str(acfg.get("strategy") or "rule").lower()

    context = search_result.get("context") or {}
    context_text = str(context.get("context") or "")
    sources = context.get("sources") or []
    results = search_result.get("results") or []

    confidence, has_rerank = _max_score(sources, results)

    if strategy == "llm_judge":
        # 预留：后续批次用一次轻量 LLM 调用判「资料是否覆盖问题」，更准但有成本。
        # 本批不实现，明确退回 rule，保证行为可预期、不留半吊子分支。
        logger.debug("llm_judge 策略未实现，本批退回 rule 策略")

    # 1) 无资料直接拒
    if require_non_empty and (not context_text.strip() or not sources):
        return False, 0.0, "empty_context"

    # 2) 有 rerank 分时才用阈值卡（rrf 名次分尺度不同，不能套用同一阈值，否则全被误杀）
    if has_rerank and confidence < min_score:
        return False, round(confidence, 6), "low_confidence"

    return True, round(confidence, 6), "ok"


def refusal_message(cfg: dict[str, Any] | None = None) -> str:
    """取拒答话术（可配置，默认给一句稳妥的中文拒答）。"""
    acfg = (cfg or {}).get("answerability") or {}
    return str(acfg.get("refusal_message") or DEFAULT_REFUSAL)
