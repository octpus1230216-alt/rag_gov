# -*- coding: utf-8 -*-
"""
FastMCP 服务入口：创建 "rag-zh" server，注册 6 个检索/入库工具。

两种 transport：
  - 默认            → stdio（供 Claude Desktop 等本地客户端直接拉起）
  - --sse           → sse（供 Web / 远程客户端，host=0.0.0.0 port=8765）

启动时：
  1. 打印当前 profile 与将加载的模型（方便确认硬件档位对不对）；
  2. 预热单例：Qdrant client、TextEmbedder、Reranker —— 让首次检索不用等模型加载；
  3. 注册工具并 run()。

预热失败不阻断启动（比如 Qdrant 没起来），只打警告 —— 工具调用时各自再兜底报错。
"""

from __future__ import annotations

import argparse

from fastmcp import FastMCP

from ingestion.parsers.common import get_settings
from mcp_server.tools import register_tools


def _print_startup_banner() -> None:
    """打印 profile 与已加载 / 将加载的模型信息。"""
    settings = get_settings()
    active = settings.get("active") or {}
    profile = settings.get("profile", "unknown")
    print("=" * 60, flush=True)
    print(f"[rag-zh MCP] profile = {profile}  device = {active.get('device', '?')}", flush=True)
    print(
        f"[rag-zh MCP] embedding = {active.get('embedding_model', '?')} "
        f"(dim={active.get('embedding_dim', '?')}, sparse={active.get('supports_sparse', '?')})",
        flush=True,
    )
    reranker = active.get("reranker") or {}
    print(
        f"[rag-zh MCP] reranker  = {reranker.get('model', '?')} "
        f"(enabled={reranker.get('enabled', '?')})",
        flush=True,
    )
    image_model = (settings.get("embedding") or {}).get("image_embedding_model", "?")
    print(f"[rag-zh MCP] image     = {image_model}", flush=True)
    print("=" * 60, flush=True)


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
            print(f"[rag-zh MCP] {name} 就绪", flush=True)
        except Exception as exc:
            print(f"[rag-zh MCP] [警告] {name} 初始化失败（调用相关工具时会重试）: {exc}", flush=True)


def build_server() -> FastMCP:
    """创建并配置 FastMCP 实例。"""
    settings = get_settings()
    mcp_cfg = settings.get("mcp") or {}
    server_name = str(mcp_cfg.get("server_name") or "rag-zh")
    instructions = str(
        mcp_cfg.get("instructions")
        or "中文文档三模检索（语义/精确/模糊），返回带引用的 top-k 片段"
    )
    mcp = FastMCP(name=server_name, instructions=instructions)
    register_tools(mcp)
    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mcp_server.server", description="rag-zh MCP 服务")
    parser.add_argument("--sse", action="store_true", help="以 SSE transport 启动（默认 stdio）")
    parser.add_argument("--host", default=None, help="SSE 监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=None, help="SSE 端口（默认 8765）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动预热（调试用）")
    args = parser.parse_args(argv)

    _print_startup_banner()
    if not args.no_warmup:
        _warm_up()

    mcp = build_server()

    if args.sse:
        host = args.host or "0.0.0.0"
        port = args.port or 8765
        print(f"[rag-zh MCP] 以 SSE 启动: http://{host}:{port}", flush=True)
        mcp.run(transport="sse", host=host, port=port)
    else:
        print("[rag-zh MCP] 以 stdio 启动（等待客户端连接）", flush=True)
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
