# -*- coding: utf-8 -*-
"""问答编排离线自检（由 _check_generation.py 迁移）：降级 / 拒答 两条路径。

monkeypatch retrieval.hybrid_search.search，不触网、不起服务：
  - 高分检索 + LLM 不可用        -> degraded=True，原样带回 sources 证据
  - 空检索                       -> 拒答（answer=None / answerable=False / refusal 有话术）
"""

from __future__ import annotations

from typing import Any

import pytest

from generation.answer_generator import get_generator


@pytest.fixture
def generator():
    return get_generator()


def test_degraded_returns_evidence(generator, monkeypatch):
    fake: dict[str, Any] = {
        "results": [{"chunk_id": "c1", "rerank_score": 0.9}],
        "context": {
            "context": "[1] 答案资料",
            "sources": [{"rerank_score": 0.9, "chunk_id": "c1"}],
        },
    }
    monkeypatch.setattr("retrieval.hybrid_search.search", lambda *a, **k: fake)
    r = generator.generate("违约条款", top_k=3)
    # openai 未装 / 未配 key -> 降级
    assert r["degraded"] is True and r["answer"] is None
    assert r["reason"].startswith("llm_")
    assert len(r["sources"]) == 1  # 降级仍带回检索证据


def test_refusal_on_empty_retrieval(generator, monkeypatch):
    monkeypatch.setattr(
        "retrieval.hybrid_search.search",
        lambda *a, **k: {"results": [], "context": {"context": "", "sources": []}},
    )
    r = generator.generate("库里没有的话题", top_k=3)
    assert r["answer"] is None
    assert r["answerable"] is False
    assert r["reason"] == "empty_context"
    assert bool(r["refusal"])  # 拒答有话术
    assert r["degraded"] is False  # 拒答在调 LLM 之前，不算降级
