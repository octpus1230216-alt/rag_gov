# -*- coding: utf-8 -*-
"""
建索引脚本：调用 ingestion.pipeline.run() 把文件跑到 embed/index 阶段。

增量语义：
  pipeline 默认 force=False，已入库且产物未变的文件会在各阶段被 skip（幂等），
  因此重复执行本脚本天然只处理「新增/变化」的文档，无需自己维护已入库清单。
  本脚本额外在跑完后对比 documents 表数量，打印本次新增了多少 doc。

用法：
  python scripts/build_index.py --input samples/
  python scripts/build_index.py --input docs/ --from-stage chunk --to-stage index
  python scripts/build_index.py --input new_file.pdf --id-strategy bytes
"""

from __future__ import annotations

import argparse

from ingestion.pipeline import STAGES, run


def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="python scripts/build_index.py", description="建索引（增量入库）")
    cli.add_argument("--input", required=True, help="文件或目录路径")
    cli.add_argument("--from-stage", dest="from_stage", default="parse", choices=list(STAGES), help="起始阶段")
    cli.add_argument("--to-stage", dest="to_stage", default="index", choices=list(STAGES), help="结束阶段")
    cli.add_argument("--id-strategy", dest="id_strategy", default="bytes", help="doc_id 生成策略")
    cli.add_argument("--workers", type=int, default=None, help="并行进程数")
    cli.add_argument("--force", action="store_true", help="强制重跑（关闭增量跳过）")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)

    # 跑之前先记一下已入库文档数，用于算增量
    try:
        from index import metadata_store

        before = metadata_store.count_documents()
    except Exception:  # 库还没建时 count 会失败，视为 0
        before = 0

    summary = run(
        args.input,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        id_strategy=args.id_strategy,
        workers=args.workers,
        force=args.force,
    )

    try:
        from index import metadata_store

        after = metadata_store.count_documents()
    except Exception:
        after = before

    print("-" * 60)
    print(
        f"[build_index] 处理文件 {summary['total']}（完成 {summary['done']} / "
        f"失败 {summary['error']}）| 产出 chunk {summary['n_chunks']} | "
        f"库内文档数 {before} -> {after}（本次新增 {max(0, after - before)}）"
    )
    if summary.get("skipped_stages"):
        print(f"[build_index] 提示：以下阶段尚未接入 pipeline（占位跳过）：{'、'.join(summary['skipped_stages'])}")
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
