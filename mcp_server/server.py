# -*- coding: utf-8 -*-
"""
FastMCP 服务入口：创建 "rag-zh" server，注册 6 个检索/入库工具。

两种 transport：
  - 默认            → stdio（供 Claude Desktop 等本地客户端直接拉起）
  - --sse           → sse（供 Web / 远程客户端，默认只绑 127.0.0.1:8765；
                     绑非本机地址（如 0.0.0.0）必须配置 MCP_AUTH_TOKEN，否则拒绝启动）

启动时：
  1. 初始化统一日志（走 stderr，绝不污染 stdio 的 JSON-RPC 通道）；
  2. 记录当前 profile 与将加载的模型（方便确认硬件档位对不对）；
  3. 预热单例：Qdrant client、TextEmbedder、Reranker —— 让首次检索不用等模型加载；
  4. 注册工具并 run()。

预热失败不阻断启动（比如 Qdrant 没起来），只打警告 —— 工具调用时各自再兜底报错。

日志红线：本模块绝不用 print。stdio 下 stdout 被 JSON-RPC 独占，任何 print 到 stdout
  都会被客户端当成一帧协议数据而炸流；诊断信息一律走 logger（stderr + 文件）。
"""

from __future__ import annotations

import argparse
import logging
import os

from fastmcp import FastMCP

from config.logging_setup import setup_logging
from ingestion.parsers.common import get_settings
from mcp_server.tools import register_tools

logger = logging.getLogger("rag_zh.mcp.server")

# 只允许这些地址在无 token 时绑定（本机回环）；其余对外地址必须有 token 才起。
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback_host(host: str) -> bool:
    """127.0.0.1 / localhost / ::1 / 任何 127.* 视为本机；其余（含 0.0.0.0）视为对外。"""
    h = host.strip().lower()
    return h in _LOOPBACK_HOSTS or h.startswith("127.")


def _resolve_auth_token() -> str:
    """token 优先环境变量 MCP_AUTH_TOKEN，退化 settings.mcp.auth_token；空串表示未配置。"""
    token = os.environ.get("MCP_AUTH_TOKEN")
    if token and token.strip():
        return token.strip()
    mcp_cfg = (get_settings().get("mcp") or {})
    return str(mcp_cfg.get("auth_token") or "").strip()


def _attach_bearer_middleware(mcp: object, token: str) -> None:
    """
    best-effort 逐请求 bearer 校验：仅当 FastMCP 提供中间件钩子且已配置 token 时挂上。

    本仓不硬依赖某个 FastMCP 版本的具体签名：导入/注册失败只记日志，
    启动层「非本机必须 token」的门禁仍照常生效（边缘/反代可补位透传校验）。
    """
    if not token:
        return
    try:
        from fastmcp.server.middleware import Middleware
    except Exception:  # pragma: no cover - 版本无中间件基类时降级
        logger.info("未启用逐请求 bearer 校验（FastMCP 版本无中间件钩子）；请在反向代理层校验 token。")
        return
    if not hasattr(mcp, "add_middleware"):
        logger.info("未启用逐请求 bearer 校验（FastMCP 实例无 add_middleware）；请确认版本或在反代校验。")
        return

    class _Bearer(Middleware):  # type: ignore[misc,valid-type]
        async def on_request(self, context, call_next):  # pragma: no cover - 依赖运行时请求上下文
            headers: dict[str, str] = {}
            try:
                request = getattr(getattr(context, "request_context", None), "request", None)
                if request is not None:
                    headers = {k.lower(): v for k, v in dict(request.headers).items()}
            except Exception:
                headers = {}
            if headers.get("authorization", "") != f"Bearer {token}":
                raise PermissionError("missing or invalid bearer token")
            return await call_next(context)

    try:
        mcp.add_middleware(_Bearer())  # type: ignore[attr-defined]
        logger.info("已启用逐请求 bearer 校验")
    except Exception as exc:  # pragma: no cover - 签名/版本差异不阻断启动
        logger.info("逐请求 bearer 校验注册失败（降级为启动门禁）: %s", exc)


def _log_startup_banner() -> None:
    """记录 profile 与已加载 / 将加载的模型信息。"""
    settings = get_settings()
    active = settings.get("active") or {}
    profile = settings.get("profile", "unknown")
    logger.info("=" * 60)
    logger.info(
        "profile = %s  device = %s", profile, active.get("device", "?")
    )
    logger.info(
        "embedding = %s (dim=%s, sparse=%s)",
        active.get("embedding_model", "?"),
        active.get("embedding_dim", "?"),
        active.get("supports_sparse", "?"),
    )
    reranker = active.get("reranker") or {}
    logger.info(
        "reranker  = %s (enabled=%s)",
        reranker.get("model", "?"),
        reranker.get("enabled", "?"),
    )
    image_model = (settings.get("embedding") or {}).get("image_embedding_model", "?")
    logger.info("image     = %s", image_model)
    logger.info("=" * 60)


def _warm_up() -> None:
    """预热重量级单例；任一失败只警告，不影响服务起来。"""
    checks = [
        ("Qdrant client", lambda: __import__("index.vector_store", fromlist=["get_client"]).get_client()),
        ("TextEmbedder", lambda: __import__("embedding.text_embedder", fromlist=["get_embedder"]).get_embedder()),
        ("Reranker", lambda: __import__("retrieval.reranker", fromlist=["get_reranker"]).get_reranker()),
    ]
    for name, fn in checks:
        try:
            fn()
            logger.info("%s 就绪", name)
        except Exception as exc:
            logger.warning("%s 初始化失败（调用相关工具时会重试）: %s", name, exc)
    # 生成层只探配置（不实际请求 LLM，避免启动就打网络/计费）
    try:
        llm = __import__("generation.llm_client", fromlist=["get_llm_client"]).get_llm_client()
        if llm.is_configured:
            logger.info("LLM 就绪（model=%s），ask 可生成带引用答案", llm.model)
        else:
            logger.warning("未配置 LLM_API_KEY，ask 将降级为仅返回检索证据")
    except Exception as exc:
        logger.warning("生成层初始化探测失败（ask 会走降级）: %s", exc)


def build_server() -> FastMCP:
    """创建并配置 FastMCP 实例。"""
    settings = get_settings()
    mcp_cfg = settings.get("mcp") or {}
    server_name = str(mcp_cfg.get("server_name") or "rag-zh")
    instructions = str(
        mcp_cfg.get("instructions")
        or "中文文档三模检索（语义/精确/模糊），返回带引用的 top-k 片段；ask 工具可生成带引用的答案"
    )
    mcp = FastMCP(name=server_name, instructions=instructions)
    register_tools(mcp)
    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mcp_server.server", description="rag-zh MCP 服务")
    parser.add_argument("--sse", action="store_true", help="以 SSE transport 启动（默认 stdio）")
    parser.add_argument(
        "--host",
        default=None,
        help="SSE 监听地址（默认 127.0.0.1；绑非本机地址必须配置 MCP_AUTH_TOKEN）",
    )
    parser.add_argument("--port", type=int, default=None, help="SSE 端口（默认 8765）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动预热（调试用）")
    args = parser.parse_args(argv)

    # 日志必须早于一切诊断输出初始化：stdio 下 stdout 是协议通道，日志只能走 stderr。
    setup_logging()
    _log_startup_banner()

    # 绑定地址与 token 在预热前先定：对外地址无 token 直接拒绝，不浪费去加载 2GB 模型。
    host = (args.host or "127.0.0.1") if args.sse else None
    port = args.port or 8765
    token = _resolve_auth_token() if args.sse else ""
    if args.sse and not _is_loopback_host(host) and not token:
        logger.error(
            "SSE 绑定 %s 属对外地址，但未配置 MCP_AUTH_TOKEN（或 settings.mcp.auth_token）；"
            "拒绝启动。本机使用请绑 127.0.0.1，对外暴露请设置 token 或经反向代理。",
            host,
        )
        return 2

    if not args.no_warmup:
        _warm_up()

    mcp = build_server()

    if args.sse:
        assert host is not None
        _attach_bearer_middleware(mcp, token)
        suffix = "（token 门禁启用）" if token else "（仅本机回环，无 token）"
        logger.info("以 SSE 启动: http://%s:%s %s", host, port, suffix)
        mcp.run(transport="sse", host=host, port=port)
    else:
        logger.info("以 stdio 启动（等待客户端连接）")
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
