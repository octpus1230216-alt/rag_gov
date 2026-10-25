# -*- coding: utf-8 -*-
"""
问答编排入口：Query → 检索 → 可答性判断 → 生成带引用答案（RAG 的“G”）。

链路（复用既有检索层，不改其行为）：
  1. retrieval.hybrid_search.search() 拿到 {"results","context",...}
  2. answerability.assess() 先过闸门：不可答 -> 返回拒答，不调 LLM（省 token + 防幻觉）
  3. prompts.build_messages() 组提示 -> llm_client.chat() 生成
  4. 任何 LLM 故障（未配置/鉴权/不可用）-> 降级：返回检索 context + sources，
     工具对调用方仍然有用（至少能看到证据），而不是整个报错。

统一返回 dict（给 MCP / 后续评测都用这个结构）：
  {
    "answer": str | None,       # 生成答案；拒答/降级时为 None
    "answerable": bool,         # 可答性闸门结论
    "confidence": float,        # 检索最高置信度
    "degraded": bool,           # 是否因 LLM 故障退回仅检索
    "refusal": str | None,      # 拒答话术（answerable=False 时）
    "reason": str,              # ok / empty_context / low_confidence / llm_* / error
    "sources": [...],           # 与 context [N] 编号同序的引用来源
    "retrieved_count": int,     # 检索命中的 chunk 数
    "elapsed": float,
  }

惰性 import：search / llm_client 都只在 generate() 内 import，
保证只 import 本模块（如离线自检降级路径）时不因缺 openai 依赖而失败。
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from generation import answerability, prompts
from ingestion.parsers.common import get_settings

logger = logging.getLogger("rag_zh.generation.answer")


class AnswerGenerator:
    """端到端问答编排：检索 + 可答性 + 生成 + 降级。"""

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = get_settings().get("generation") or {}

    def generate(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        filters: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        生成答案主入口。

        参数：
          query:   用户问题
          top_k:   最终参考的片段条数
          mode:    检索模式（透传给 hybrid_search：hybrid/semantic/keyword/fuzzy）
          filters: 元数据过滤（透传给检索，如 {"doc_type": "contract"}）
          stream:  预留流式开关（本批未实现流式，置 True 也只走非流式，不报错）
        """
        if stream:
            # 流式输出接口缝：后续批次实现 token 流，本批显式降级为一次性返回，行为可预期
            logger.debug("stream=True 本批未实现流式，退回一次性非流式返回")

        started = time.time()
        base: dict[str, Any] = {
            "answer": None,
            "answerable": False,
            "confidence": 0.0,
            "degraded": False,
            "refusal": None,
            "reason": "",
            "sources": [],
            "retrieved_count": 0,
            "elapsed": 0.0,
        }

        # 1) 检索（失败不吞：让上层 MCP 工具转结构化错误，这里不假装成功）
        from retrieval.hybrid_search import search as _search

        search_result = _search(query, top_k=top_k, filters=filters, mode=mode)
        context = search_result.get("context") or {}
        sources = context.get("sources") or []
        base["sources"] = sources
        base["retrieved_count"] = len(search_result.get("results") or [])

        # 2) 可答性闸门
        answerable, confidence, reason = answerability.assess(search_result, self._cfg)
        base["answerable"] = answerable
        base["confidence"] = confidence
        base["reason"] = reason
        if not answerable:
            base["refusal"] = answerability.refusal_message(self._cfg)
            base["elapsed"] = round(time.time() - started, 3)
            return base

        # 3) 生成（LLM 故障 -> 降级，把检索证据原样交回）
        messages = prompts.build_messages(
            query, str(context.get("context") or ""), sources, self._cfg
        )
        try:
            from generation.llm_client import get_llm_client

            answer = get_llm_client().chat(messages)
        except Exception as exc:  # LLMNotConfigured/LLMAuthError/LLMUnavailable 及任何意外
            base["degraded"] = True
            base["reason"] = f"llm_{type(exc).__name__}"
            base["context"] = context.get("context") or ""
            base["elapsed"] = round(time.time() - started, 3)
            logger.warning("LLM 生成失败，退回仅检索模式: %s: %s", type(exc).__name__, exc)
            return base

        if not answer:
            # 模型返回空文本：视为答不出，降级交回证据，绝不给调用方一个空“答案”当成结论
            base["degraded"] = True
            base["reason"] = "llm_empty_response"
            base["context"] = context.get("context") or ""
            base["elapsed"] = round(time.time() - started, 3)
            return base

        base["answer"] = answer
        base["elapsed"] = round(time.time() - started, 3)
        return base


# =================================================================== 单例入口
@lru_cache(maxsize=1)
def get_generator() -> AnswerGenerator:
    """全局单例：同进程复用（读一次配置，无重状态）。"""
    return AnswerGenerator()


# =================================================================== 便捷函数
def answer(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """模块级便捷入口（等价于 get_generator().generate(...)）。"""
    return get_generator().generate(query, top_k=top_k, mode=mode, filters=filters)
