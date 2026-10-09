# -*- coding: utf-8 -*-
"""
检索评测：读 eval/qa_set.jsonl，对每条 query 跑 HybridSearcher.search()，
计算 Recall@5 / Recall@10 / MRR，并按类型（semantic/exact/fuzzy/table/image）分组，
输出 markdown 表格。

gold 匹配两种模式（每条 query 自动择一）：
  1. 严格 id：expected_chunk_ids 非空 → 命中 = chunk_id ∈ expected_chunk_ids；
  2. 关键词：expected_chunk_ids 为空但有 expected_keywords → 命中 = 结果 content
     同时包含全部关键词（入库前没有真实 chunk_id 时用它，评测开箱即跑）。

指标定义（binary relevance）：
  Recall@k：top-k 内命中至少 1 条记 1.0，否则 0.0，对全部 query 取均值；
  MRR      ：第一条命中的排名取倒数（1/rank），无命中记 0，取均值。

用法：
  python eval/run_eval.py
  python eval/run_eval.py --qa-set eval/qa_set.jsonl --top-k 10 --mode hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TYPE_ORDER = ["semantic", "exact", "fuzzy", "table", "image"]


def load_qa_set(path: str | Path) -> list[dict[str, Any]]:
    """读 jsonl，跳过空行与注释行。"""
    items: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def is_hit(chunk: dict[str, Any], gold_ids: set[str], keywords: list[str]) -> bool:
    """判断某条检索结果是否命中 gold。"""
    if gold_ids:
        return chunk.get("chunk_id") in gold_ids
    if keywords:
        content = str(chunk.get("content", ""))
        return all(kw in content for kw in keywords)
    return False


def first_hit_rank(results: list[dict[str, Any]], gold_ids: set[str], keywords: list[str], cutoff: int) -> int:
    """返回 top-cutoff 内第一条命中的排名（1-based），无命中返回 0。"""
    for rank, chunk in enumerate(results[:cutoff], start=1):
        if is_hit(chunk, gold_ids, keywords):
            return rank
    return 0


def evaluate(
    qa_items: list[dict[str, Any]],
    top_k: int,
    mode: str,
) -> dict[str, Any]:
    """对每条 query 跑检索并累计指标，返回按类型分组的统计。"""
    from retrieval.hybrid_search import get_searcher

    searcher = get_searcher()
    # 按类型分组：每组记 [recall@5 命中数, recall@10 命中数, mrr 累加, 总数]
    stats: dict[str, dict[str, float]] = {
        t: {"n": 0, "r5": 0.0, "r10": 0.0, "mrr": 0.0} for t in TYPE_ORDER
    }
    stats.setdefault("other", {"n": 0, "r5": 0.0, "r10": 0.0, "mrr": 0.0})
    details: list[dict[str, Any]] = []

    for item in qa_items:
        query = item["query"]
        qtype = item.get("type", "other")
        bucket = stats.setdefault(qtype, {"n": 0, "r5": 0.0, "r10": 0.0, "mrr": 0.0})
        gold_ids = set(item.get("expected_chunk_ids") or [])
        keywords = list(item.get("expected_keywords") or [])

        result = searcher.search(query, top_k=max(top_k, 10), mode=mode)
        results = result.get("results", [])

        rank5 = first_hit_rank(results, gold_ids, keywords, 5)
        rank10 = first_hit_rank(results, gold_ids, keywords, 10)
        rank_full = first_hit_rank(results, gold_ids, keywords, top_k)

        bucket["n"] += 1
        bucket["r5"] += 1.0 if rank5 else 0.0
        bucket["r10"] += 1.0 if rank10 else 0.0
        bucket["mrr"] += (1.0 / rank_full) if rank_full else 0.0

        details.append({
            "query": query,
            "type": qtype,
            "hit@5": bool(rank5),
            "hit@10": bool(rank10),
            "first_rank": rank_full or None,
            "top_result": results[0]["chunk_id"] if results else None,
        })

    return {"stats": stats, "details": details}


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def render_markdown(stats: dict[str, dict[str, float]], top_k: int, mode: str) -> str:
    """把分组统计渲染成 markdown 表格。"""
    lines = [
        f"# 检索评测报告（mode={mode}, top_k={top_k}）",
        "",
        "| 类型 | 样本数 | Recall@5 | Recall@10 | MRR |",
        "|------|-------:|---------:|----------:|----:|",
    ]
    total = {"n": 0, "r5": 0.0, "r10": 0.0, "mrr": 0.0}
    ordered = [t for t in TYPE_ORDER if t in stats] + [t for t in stats if t not in TYPE_ORDER]
    for t in ordered:
        s = stats[t]
        if s["n"] == 0:
            continue
        n = int(s["n"])
        r5 = s["r5"] / n
        r10 = s["r10"] / n
        mrr = s["mrr"] / n
        lines.append(f"| {t} | {n} | {_fmt(r5)} | {_fmt(r10)} | {_fmt(mrr)} |")
        total["n"] += n
        total["r5"] += s["r5"]
        total["r10"] += s["r10"]
        total["mrr"] += s["mrr"]

    if total["n"]:
        tn = total["n"]
        lines.append(
            f"| **总体** | {tn} | {_fmt(total['r5'] / tn)} | "
            f"{_fmt(total['r10'] / tn)} | {_fmt(total['mrr'] / tn)} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(prog="python eval/run_eval.py", description="检索质量评测")
    cli.add_argument("--qa-set", default="eval/qa_set.jsonl", help="评测集 jsonl 路径")
    cli.add_argument("--top-k", type=int, default=10, help="评测考虑的 top-k")
    cli.add_argument("--mode", default="hybrid", help="检索模式：hybrid/semantic/keyword/fuzzy")
    cli.add_argument("--out", default=None, help="把 markdown 报告写到文件（默认只打印 stdout）")
    args = cli.parse_args(argv)

    qa_items = load_qa_set(args.qa_set)
    if not qa_items:
        print(f"[run_eval] 评测集为空: {args.qa_set}")
        return 1
    print(f"[run_eval] 载入 {len(qa_items)} 条 query，开始检索（mode={args.mode}）...")

    report = evaluate(qa_items, top_k=args.top_k, mode=args.mode)
    markdown = render_markdown(report["stats"], args.top_k, args.mode)
    print(markdown)

    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"\n[run_eval] 报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
