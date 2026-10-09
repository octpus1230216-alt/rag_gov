# -*- coding: utf-8 -*-
"""
定期重建 vocab 表（建议每周一次 / 大批入库后手动触发）。

为什么需要重建：
  vocab 表存的是「词 -> df/idf」，入库时只增量追加当前文档的词。
  随着文档增多，某些词的 df 会变化（原本稀有的词变常见），idf 需要重算；
  被删除文档遗留的 dead term 也需要清理。定期全量重算保证 idf 准确。

与 index.vocab_builder 的区别：
  vocab_builder 从 chunks/*.jsonl 产物目录读；本脚本从 **chunks 表**（SQLite）
  读全量 content，适合「产物目录已清理、但库里数据还在」的例行维护。

用法：
  python scripts/rebuild_vocab.py
  python scripts/rebuild_vocab.py --batch-size 2000
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from index.vocab_builder import build_vocab


def _iter_all_chunks(batch_size: int) -> Any:
    """
    从 chunks 表分批读取 content，产出 [{"chunk_id", "content"}]。
    用 LIMIT/OFFSET 游标避免一次性 fetchall 撑爆内存。
    """
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    offset = 0
    while offset < total:
        rows = conn.execute(
            "SELECT chunk_id, content FROM chunks LIMIT ? OFFSET ?", (batch_size, offset)
        ).fetchall()
        if not rows:
            break
        yield from [{"chunk_id": r["chunk_id"], "content": r["content"]} for r in rows]
        offset += batch_size


def rebuild(batch_size: int) -> int:
    """全量重算 vocab，返回写入的词数。"""
    from index.keyword_store import get_connection, init_tables

    init_tables()
    conn = get_connection()
    # 重建前先清空旧词表（dead term 一并清掉）
    conn.execute("DELETE FROM vocab")
    conn.commit()

    # 分批喂给 build_vocab（它内部按 chunk 统计 df 再整体写表，需要完整列表）
    started = time.time()
    all_chunks: list[dict[str, Any]] = []
    for batch in _iter_all_chunks(batch_size):
        all_chunks.append(batch)
    n_terms = build_vocab(all_chunks, batch_size=batch_size)
    elapsed = time.time() - started
    print(f"[rebuild_vocab] 读取 {len(all_chunks)} 条 chunk，耗时 {elapsed:.1f}s")
    return n_terms


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(prog="python scripts/rebuild_vocab.py", description="定期重建 vocab 表")
    cli.add_argument("--batch-size", type=int, default=1000, help="每批读取/提交的 chunk 数")
    args = cli.parse_args(argv)

    n_terms = rebuild(args.batch_size)

    # 打印重建后的规模
    try:
        from index.vocab_builder import get_vocab_stats

        stats = get_vocab_stats()
        print(f"[rebuild_vocab] ✓ 重建完成，vocab 规模：{stats}")
    except Exception:
        print(f"[rebuild_vocab] ✓ 重建完成，共写入 {n_terms} 个词")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
