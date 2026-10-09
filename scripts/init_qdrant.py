# -*- coding: utf-8 -*-
"""
初始化 Qdrant collection（dense + sparse 命名向量）。

调用 index.vector_store.create_collection()，幂等：已存在则跳过，不删已有数据。
可独立运行：python scripts/init_qdrant.py [collection_name]
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """建 collection；参数为可选的 collection 名（默认 settings.qdrant.collection）。"""
    args = list(argv if argv is not None else sys.argv[1:])
    name = args[0] if args else None

    from index.vector_store import create_collection, get_client

    # 先探连接，连不上直接给清晰错误（别等到 create 才报一堆 grpc traceback）
    try:
        client = get_client()
        collections = client.get_collections().collections
        existing = {c.name for c in collections}
    except Exception as exc:
        print(f"[init_qdrant] 无法连接 Qdrant: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  请确认 docker compose up -d qdrant 已启动", file=sys.stderr)
        return 1

    target = name or "docs"
    if target in existing:
        print(f"[init_qdrant] collection '{target}' 已存在，跳过（幂等）")
        return 0

    create_collection(target)
    print(f"[init_qdrant] ✓ collection '{target}' 创建完成（dense + sparse 命名向量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
