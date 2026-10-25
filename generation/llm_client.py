# -*- coding: utf-8 -*-
"""
LLM 客户端：OpenAI 兼容接口（/v1/chat/completions）薄封装。

为什么用 OpenAI 兼容接口一个客户端打通：
  OpenAI / DeepSeek / 通义 / vLLM / Ollama(openai 兼容模式) 都暴露同一套
  /v1/chat/completions，只需切 base_url + api_key + model 即可换后端，
  不必为每家供应商写适配器。凭证一律走环境变量（settings 里 ${LLM_API_KEY}），
  本仓库不落任何真实密钥。

设计约定（对齐 embedding.text_embedder / retrieval.reranker）：
  1. 惰性单例 get_llm_client()：client 构建有网络对象开销，同进程复用；
  2. 重依赖（openai / tenacity）在方法内 import：没装依赖时本模块仍可被 import，
     只在实际调用时才报错，离线自检（走降级路径）不受影响；
  3. 类型化异常：把千奇百怪的 openai 异常归一成 3 类，供上层精确降级——
       LLMNotConfigured  缺 api_key / 缺依赖（根本没法调）
       LLMAuthError      401/403（重试也没用，凭证错）
       LLMUnavailable    连接失败/超时/限流/5xx 重试耗尽（临时不可用）

日志红线：只记 model / token 数 / 耗时 / 状态，**绝不打印 prompt 与答案正文**
  （里面可能含入库文档的敏感内容）。
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from ingestion.parsers.common import get_settings

logger = logging.getLogger("rag_zh.generation.llm")


# =================================================================== 类型化异常
class LLMError(Exception):
    """生成层 LLM 调用异常基类。"""


class LLMNotConfigured(LLMError):
    """未配置 api_key 或未安装 openai 依赖——根本无法调用。"""


class LLMAuthError(LLMError):
    """鉴权失败（401/403）——重试无意义，需人工修凭证。"""


class LLMUnavailable(LLMError):
    """临时不可用：连接失败 / 超时 / 限流 / 5xx，且重试已耗尽。"""


# =================================================================== 客户端
class LLMClient:
    """OpenAI 兼容 chat/completions 封装。外部只通过 get_llm_client() 拿实例。"""

    def __init__(self) -> None:
        gcfg = get_settings().get("generation") or {}
        self._provider: str = str(gcfg.get("provider") or "openai_compatible")
        self._base_url: str = str(gcfg.get("base_url") or "https://api.openai.com/v1")
        self._api_key: str = str(gcfg.get("api_key") or "").strip()
        self._model: str = str(gcfg.get("model") or "gpt-4o-mini")
        self._temperature: float = float(gcfg.get("temperature") or 0.2)
        self._max_tokens: int = int(gcfg.get("max_tokens") or 1024)
        self._timeout: float = float(gcfg.get("timeout_seconds") or 30)
        self._max_retries: int = int(gcfg.get("max_retries") or 2)
        self._client: Any = None

    # ---------------------------------------------------------------- 配置与建连
    @property
    def is_configured(self) -> bool:
        """是否具备调用条件（有 api_key）。server 启动预热用，不实际发请求。"""
        return bool(self._api_key)

    def _get_client(self) -> Any:
        """惰性构建 openai client；openai 内置重试关掉，改由 tenacity 统一控制，避免双重重试。"""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMNotConfigured("未配置 LLM_API_KEY：生成层无法调用大模型，将退回仅检索模式")
        try:
            from openai import OpenAI
        except ImportError as exc:  # 依赖缺失也归为「没法调」
            raise LLMNotConfigured(f"未安装 openai 依赖：pip install openai（{exc}）") from exc
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,
        )
        return self._client

    # ---------------------------------------------------------------- 对外主接口
    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        """
        发起一次对话补全，返回 assistant 文本。
        overrides 可覆盖 temperature / max_tokens。
        任何底层错误都归一成 LLMNotConfigured / LLMAuthError / LLMUnavailable，
        由调用方（AnswerGenerator）决定降级策略。
        """
        try:
            import openai
        except ImportError as exc:  # 缺依赖同样归为「没法调」，不向调用方漏 ImportError
            raise LLMNotConfigured(f"未安装 openai 依赖：pip install openai（{exc}）") from exc

        client = self._get_client()  # 可能抛 LLMNotConfigured（缺 api_key）
        temperature = float(overrides.get("temperature", self._temperature))
        max_tokens = int(overrides.get("max_tokens", self._max_tokens))

        started = time.time()
        try:
            resp = self._call_with_retry(client, messages, temperature, max_tokens)
        except openai.AuthenticationError as exc:
            raise LLMAuthError(f"LLM 鉴权失败（不重试）: {exc}") from exc
        except openai.PermissionDeniedError as exc:
            raise LLMAuthError(f"LLM 无权访问（不重试）: {exc}") from exc
        except openai.APIError as exc:
            # 覆盖 APITimeoutError / APIConnectionError / InternalServerError / 重试耗尽的限流
            raise LLMUnavailable(
                f"LLM 暂时不可用（重试后仍失败）: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:  # openai 之外的意外错误也归为不可用，交由上层降级
            raise LLMUnavailable(f"LLM 调用出现未预期错误: {type(exc).__name__}: {exc}") from exc

        text = self._extract_text(resp)
        self._log_success(resp, started)
        return text

    # ---------------------------------------------------------------- 内部工具
    def _call_with_retry(
        self, client: Any, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> Any:
        """tenacity 指数退避：仅对可重试错误（限流/超时/连接/5xx）重试，其余立即冒泡。"""
        import openai
        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        def _retryable(exc: BaseException) -> bool:
            return self._is_retryable(exc, openai)

        @retry(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )
        def _do() -> Any:
            return client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return _do()

    @staticmethod
    def _is_retryable(exc: BaseException, openai_mod: Any) -> bool:
        """
        只重试「换个时间大概率能成」的错误：限流(429)、超时、连接错误、5xx。
        4xx（鉴权/请求错误/404）不重试——重试也是白搭，且会拖慢降级。
        """
        if isinstance(
            exc,
            (openai_mod.RateLimitError, openai_mod.APITimeoutError, openai_mod.APIConnectionError),
        ):
            return True
        status = getattr(exc, "status_code", None)
        return bool(isinstance(status, int) and status >= 500)

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """从 chat.completions 响应里取第一条 choice 的文本；结构异常返回空串（上层按答不出处理）。"""
        try:
            choices = resp.choices
            if not choices:
                return ""
            content = choices[0].message.content
            return (content or "").strip()
        except (AttributeError, IndexError):
            return ""

    def _log_success(self, resp: Any, started: float) -> None:
        """成功日志：只记元数据，不记正文。"""
        usage = getattr(resp, "usage", None)
        logger.info(
            "llm chat ok model=%s elapsed=%.2fs prompt_tokens=%s completion_tokens=%s",
            self._model,
            round(time.time() - started, 2),
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
        )

    @property
    def model(self) -> str:
        return self._model


# =================================================================== 单例入口
@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """全局单例：同进程复用同一 client（连接池/配置一致）。"""
    return LLMClient()
