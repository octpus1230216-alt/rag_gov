# -*- coding: utf-8 -*-
"""LLM 客户端离线自检（由 _check_generation.py 迁移）：类型化异常契约 + 单例。

CI 未装 openai / 未配 api_key，两条路径都应归一成 LLMNotConfigured，
绝不向调用方漏 ImportError，也不该真的发起网络请求。"""

from __future__ import annotations

import pytest

from generation.llm_client import (
    LLMClient,
    LLMError,
    LLMNotConfigured,
    get_llm_client,
)


def test_exception_hierarchy():
    assert issubclass(LLMNotConfigured, LLMError)


def test_chat_raises_typed_error_when_unconfigured():
    client = LLMClient()
    if client.is_configured:  # 极少数本地环境真配了 key：跳过，避免打网络
        pytest.skip("检测到已配置 LLM_API_KEY，跳过离线不发起请求的前提")
    with pytest.raises(LLMNotConfigured):
        client.chat([{"role": "user", "content": "hi"}])


def test_get_llm_client_is_singleton():
    assert get_llm_client() is get_llm_client()
