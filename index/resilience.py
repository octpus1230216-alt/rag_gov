# -*- coding: utf-8 -*-
"""
外部存储读路径的容错小工具（纯函数，离线可测，无网络/无重依赖）。

只做一件事：把「瞬时网络故障」与「确定性错误」区分开——前者可退避重试，后者立即失败上抛。
写入路径（upsert/delete）不走这里（离线批处理，重试有幂等/重数据风险，本批不自动重试）。

设计取舍：
  - 不引 tenacity，手写一个有界指数退避循环：参数来自配置、sleep 可注入（测试不真等），
    逻辑一眼可审，且不把重试语义散落到每个调用点。
  - 判定保守：只认连接/超时/5xx/gRPC UNAVAILABLE 这类"再试一次大概率会好"的错误；
    4xx、鉴权、参数错这些"重试也不会好"的绝不浪费退避时间。
  - 重试耗尽后原样抛最后一次异常 —— 由上层（retrieval.hybrid_search）按路降级为空结果，
    本模块不吞异常，避免"看起来成功实际空"。

日志红线：只记异常类型/尝试序号/退避秒数，绝不记 query 正文或返回内容。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from config.logging_setup import get_logger

logger = get_logger("rag_zh.resilience")

# 类名里出现这些关键词即视作瞬时错误（覆盖 httpx/urllib3/grpc 的各种 Timeout/Connection 类）
_TRANSIENT_NAME_HINTS = (
    "Timeout",
    "Connect",
    "Unavailable",
    "DeadlineExceeded",
    "NameResolution",
    "Reset",
    "Aborted",
    "Temporary",
)

# gRPC 状态码里属于"瞬时、可重试"的几个
_TRANSIENT_GRPC_CODES = frozenset({"UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED"})

# 默认读重试策略（可被 settings.resilience.qdrant.* 覆盖）
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 0.2


def is_transient_error(exc: BaseException) -> bool:
    """判断异常是否属于"可重试的瞬时故障"。保守：拿不准就判 False，不轻易重试。"""
    # 内置网络异常族（ConnectionError/TimeoutError 均为 OSError 子类，且覆盖
    # ConnectionReset/Aborted/Refused）
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    name = type(exc).__name__
    if any(hint in name for hint in _TRANSIENT_NAME_HINTS):
        return True

    # Qdrant REST UnexpectedResponse：连接失败(status_code=None) 或 5xx 视作瞬时；4xx 不重试
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
    except Exception:  # pragma: no cover - qdrant 未安装时跳过该分支
        UnexpectedResponse = None  # type: ignore[assignment]
    if UnexpectedResponse is not None and isinstance(exc, UnexpectedResponse):
        status = getattr(exc, "status_code", None)
        return status is None or (isinstance(status, int) and status >= 500)

    # gRPC RpcError：看 code() 返回的状态名（UNAVAILABLE / DEADLINE_EXCEEDED 等）
    code_fn = getattr(exc, "code", None)
    if callable(code_fn):
        try:
            code_name = str(code_fn()).rsplit(".", 1)[-1]
        except Exception:
            code_name = ""
        return code_name in _TRANSIENT_GRPC_CODES

    return False


def read_retry_policy(settings: dict[str, Any]) -> tuple[int, float]:
    """从 settings.resilience.qdrant 读 (attempts, base_delay)，缺省用默认值。attempts>=1。"""
    resilience = settings.get("resilience") or {}
    qdrant = resilience.get("qdrant") if isinstance(resilience, dict) else None
    qdrant = qdrant if isinstance(qdrant, dict) else {}
    # 用显式 None 判定而非 `or`：配置写 0 时 `0 or 默认` 会误落回默认值，绕过下方下限钳制。
    raw_attempts = qdrant.get("retry_attempts")
    raw_base_delay = qdrant.get("retry_base_delay_seconds")
    attempts = DEFAULT_RETRY_ATTEMPTS if raw_attempts is None else int(raw_attempts)
    base_delay = DEFAULT_RETRY_BASE_DELAY if raw_base_delay is None else float(raw_base_delay)
    return max(1, attempts), max(0.0, base_delay)


def retry_transient(
    fn: Callable[[], Any],
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """
    对瞬时错误做有界指数退避重试：第 n 次失败后退避 ``base_delay * 2**(n-1)`` 秒再试。

    - 非瞬时错误立即上抛（不重试）；
    - attempts<=1 即不重试；
    - 全部尝试都瞬时失败时，抛最后一次异常（由上层决定如何降级）。
    """
    attempts = max(1, int(attempts))
    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            last_exc = exc
            if i < attempts - 1:
                delay = base_delay * (2**i)
                logger.warning(
                    "transient error type=%s attempt=%d/%d retry in %.2fs",
                    type(exc).__name__,
                    i + 1,
                    attempts,
                    delay,
                )
                sleep(delay)
    assert last_exc is not None  # 循环内仅在有异常时才走到末尾
    raise last_exc
