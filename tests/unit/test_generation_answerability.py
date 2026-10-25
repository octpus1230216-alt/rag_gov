# -*- coding: utf-8 -*-
"""可答性闸门离线自检（由 _check_generation.py 迁移）：三分支 + 阈值/策略 + 拒答话术。"""

from __future__ import annotations

from generation import answerability


def _result(context_text, sources, results=None):
    return {"context": {"context": context_text, "sources": sources}, "results": results or []}


def test_empty_context_not_answerable():
    ans, conf, reason = answerability.assess(_result("", []), {})
    assert ans is False and reason == "empty_context" and conf == 0.0


def test_low_rerank_score_not_answerable():
    sr = _result("x", [{"rerank_score": 0.1}], [{"rerank_score": 0.1}])
    ans, conf, reason = answerability.assess(sr, {})
    assert ans is False and reason == "low_confidence"


def test_high_rerank_score_answerable():
    sr = _result("x", [{"rerank_score": 0.9}], [{"rerank_score": 0.9}])
    ans, conf, _reason = answerability.assess(sr, {})
    assert ans is True and conf == 0.9


def test_only_rrf_score_relaxed():
    # 未开 rerank（只有 score，非 rerank_score）→ 尺度不同不套阈值，有资料即可能答
    sr = _result("x", [{"score": 0.05}])
    ans, _conf, reason = answerability.assess(sr, {})
    assert ans is True and reason == "ok"


def test_custom_threshold_via_cfg():
    sr = _result("x", [{"rerank_score": 0.5}], [{"rerank_score": 0.5}])
    cfg = {"answerability": {"min_rerank_score": 0.6}}
    ans, _conf, reason = answerability.assess(sr, cfg)
    assert ans is False and reason == "low_confidence"


def test_llm_judge_strategy_falls_back_to_rule():
    sr = _result("x", [{"rerank_score": 0.9}], [{"rerank_score": 0.9}])
    cfg = {"answerability": {"strategy": "llm_judge"}}
    ans, _conf, _reason = answerability.assess(sr, cfg)
    # 本批未实现 llm_judge，明确退回 rule 行为
    assert ans is True


def test_refusal_message_default_and_override():
    assert answerability.refusal_message({}) == answerability.DEFAULT_REFUSAL
    cfg = {"answerability": {"refusal_message": "抱歉，资料不足。"}}
    assert answerability.refusal_message(cfg) == "抱歉，资料不足。"


def test_default_min_score():
    assert answerability.DEFAULT_MIN_SCORE == 0.30
