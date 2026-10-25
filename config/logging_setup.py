# -*- coding: utf-8 -*-
"""
统一日志初始化 + 请求上下文（request_id）。

三条设计红线：
  1. **stdio 下 stdout 是 JSON-RPC 唯一通道**：MCP 以 stdio 传输时，stdout 被协议帧
     独占，任何日志写进 stdout 都会被客户端当成一帧解析而炸流。故本模块的流式 handler
     **一律挂 sys.stderr**，且各调用点禁止用 print 到 stdout（见 mcp_server / pipeline）。
  2. **request_id 走 contextvar 而非 threadlocal**：检索四路召回用 ThreadPoolExecutor
     并行，但同属一次请求上下文；MCP 工具入口 set、出口 reset，按请求天然隔离，
     同步 / 线程 / asyncio(SSE) 三种形态下都不会串号。
  3. **幂等**：重复调 setup_logging() 不叠加 handler（否则一行日志被打 N 遍）；
     只接管名为 "rag_zh" 的 logger，各模块 getLogger("rag_zh.xxx") 冒泡汇聚到此。

日志内容红线（对齐 generation.llm_client）：只记元数据（工具名 / request_id / 耗时 /
条数 / 错误类型），**绝不记 query 正文与检索/答案内容**（可能含入库文档的敏感信息）。

典型用法：
    from config.logging_setup import setup_logging, request_context
    setup_logging()                          # 进程启动时一次
    with request_context():                  # 每个工具调用包一层
        ...                                  # 期间日志自动带 [request_id]
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.handlers
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# 本模块统一管理、并被所有第一方模块冒泡汇聚的根 logger 名
ROOT_LOGGER_NAME = "rag_zh"

# request_id 承载体：contextvar，默认占位 "-"（未进入请求上下文的日志也有稳定字段，
# 格式化 %(request_id)s 不会 KeyError）
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "rag_zh_request_id", default="-"
)

# 打在 handler 上的标记：标识「本 handler 由 setup_logging 挂的」，幂等重建时据此回收
_OWNED_FLAG = "_rag_zh_owned"

# 进程级初始化标记：setup_logging 幂等判定用（logger 属性 monkeypatch 不干净）
_INITIALIZED = False


# =================================================================== request_id
def get_request_id() -> str:
    """取当前上下文的 request_id；未在请求上下文内时返回 '-'。"""
    return _request_id.get()


def _new_request_id() -> str:
    """生成一个短 request_id（uuid4 前 12 位，够定位单次调用又不刷屏）。"""
    return uuid.uuid4().hex[:12]


@contextlib.contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """
    进入一次请求上下文：set request_id，退出时复原（哪怕中途抛异常）。

    request_id 省略时自动生成一个短 id；传入沿用调用方已有的 id（便于跨服务串联）。
    with 块内所有 logger 输出都会带上该 request_id（经 RequestIdFilter 注入）。
    """
    rid = request_id or _new_request_id()
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """把当前 contextvar 里的 request_id 注入到每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


# =================================================================== 配置解析
def _resolve_level(cfg: dict[str, Any], settings: dict[str, Any]) -> int:
    """
    确定日志级别：logging.level > runtime.log_level > INFO。
    非法值（拼错 / 数字越界）不炸进程，兜底回 INFO。
    """
    runtime = settings.get("runtime") or {}
    raw = cfg.get("level") or (runtime.get("log_level") if isinstance(runtime, dict) else None) or "INFO"
    candidate = logging.getLevelName(str(raw).strip().upper())
    return candidate if isinstance(candidate, int) else logging.INFO


def _resolve_log_file(cfg: dict[str, Any], settings: dict[str, Any]) -> Path | None:
    """
    确定轮转日志文件路径。to_file=false 返回 None（不建文件 handler）。
    目录取 logging.directory（默认 logs，兼容 runtime.log_dir），文件名取 logging.file_name。
    """
    if not bool(cfg.get("to_file", True)):
        return None
    runtime = settings.get("runtime") or {}
    fallback_dir = runtime.get("log_dir") if isinstance(runtime, dict) else None
    directory = str(cfg.get("directory") or fallback_dir or "logs")
    file_name = str(cfg.get("file_name") or "rag_zh.log")
    return Path(directory) / file_name


# =================================================================== 初始化入口
def setup_logging(force: bool = False) -> logging.Logger:
    """
    配置并返回 "rag_zh" 根 logger。进程启动时调用一次即可（幂等）。

    - 流式 handler 永远挂 stderr（stdio 安全），级别由配置决定；
    - 可选轮转文件 handler（logging.to_file），落 logs/rag_zh.log；
    - force=True 时先清掉本模块此前挂的 handler 再重建（测试 / 改档用）。

    配置来自 get_settings()（缺失时容错：全走内置默认，绝不因读配置失败而抛异常）。
    """
    global _INITIALIZED
    try:
        from ingestion.parsers.common import get_settings

        settings = get_settings()
    except Exception:  # pragma: no cover - 配置层最坏兜底，日志初始化不能反过来炸进程
        settings = {}
    cfg = settings.get("logging") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    # 幂等：已配置且非 force → 只调级别，不重复挂 handler
    if _INITIALIZED and not force:
        logger.setLevel(_resolve_level(cfg, settings))
        return logger

    # 清掉本模块此前挂过的 handler（保留外部可能挂到 root 的，不动别的 logger）
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_FLAG, False):
            logger.removeHandler(handler)
            handler.close()

    level = _resolve_level(cfg, settings)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    id_filter = RequestIdFilter()

    # ---- 流式：stderr（stdio 红线，绝不 stdout）----
    if bool(cfg.get("to_stderr", True)):
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setLevel(level)
        stream.addFilter(id_filter)
        stream.setFormatter(fmt)
        setattr(stream, _OWNED_FLAG, True)
        logger.addHandler(stream)

    # ---- 轮转文件：logs/rag_zh.log（可选）----
    log_file = _resolve_log_file(cfg, settings)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=int(cfg.get("max_bytes") or 10_485_760),
                backupCount=int(cfg.get("backup_count") or 5),
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.addFilter(id_filter)
            file_handler.setFormatter(fmt)
            setattr(file_handler, _OWNED_FLAG, True)
            logger.addHandler(file_handler)
        except OSError:  # pragma: no cover - 磁盘/权限问题时降级为仅 stderr，不阻断启动
            pass

    logger.setLevel(level)
    # 不冒泡到 root：避免与第三方库 / uvicorn 的 root handler 重复输出
    logger.propagate = False
    _INITIALIZED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    取子 logger 并确保已初始化（各模块用 ``get_logger(__name__)`` 或
    ``logging.getLogger("rag_zh.xxx")`` 皆可；本函数只为调用点少写点样板）。
    name 若不是 "rag_zh" 前缀，会归一到 rag_zh.<name> 以纳入统一 handler。
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    if not _INITIALIZED:
        setup_logging()
    return logger
