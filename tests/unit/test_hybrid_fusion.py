# -*- coding: utf-8 -*-
"""加权 RRF 融合纯函数离线自检（新增，不起服务、不碰模型）。

rrf_fusion(result_lists, weights, k) -> [(chunk_id, score)] 降序：
  score(cid) = Σ_i weights[i] / (k + rank_i(cid))，w<=0 的整路跳过。"""

from __future__ import annotations

from retrieval.hybrid_search import rrf_fusion


def _ids(scored):
    return [cid for cid, _ in scored]


def test_single_route_rank_order_preserved():
    route_a = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    out = rrf_fusion([route_a], [1.0], k=60)
    assert _ids(out) == ["a", "b", "c"]  # 名次越靠前分越高
    # 手算验证：w / (k + rank)
    scores = dict(out)
    assert scores["a"] == 1.0 / 61
    assert scores["b"] == 1.0 / 62
    assert scores["c"] == 1.0 / 63


def test_weighted_two_routes():
    route_a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    route_b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
    out = rrf_fusion([route_a, route_b], [1.0, 0.5], k=60)
    scores = dict(out)
    # x 只在 a 路 rank1；y 在 a rank2 + b rank1；z 只在 b rank2
    assert scores["x"] == 1.0 / 61
    assert scores["y"] == 1.0 / 62 + 0.5 / 61
    assert scores["z"] == 0.5 / 62
    assert _ids(out) == ["y", "x", "z"]  # 两路都命中的 y 排第一


def test_zero_or_negative_weight_skips_route():
    route_a = [{"chunk_id": "a"}]
    route_b = [{"chunk_id": "b"}]
    # weights[1] <= 0 → 第二路整体不参与
    out = rrf_fusion([route_a, route_b], [1.0, 0.0], k=60)
    assert _ids(out) == ["a"]
    assert "b" not in dict(out)


def test_higher_k_flattens_rank_gap():
    route_a = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    low_k = dict(rrf_fusion([route_a], [1.0], k=1))
    high_k = dict(rrf_fusion([route_a], [1.0], k=100))
    # k 越大，相邻名次差距越小
    assert (low_k["a"] - low_k["b"]) > (high_k["a"] - high_k["b"])


def test_missing_chunk_id_ignored():
    out = rrf_fusion([[{"chunk_id": ""}, {"chunk_id": "ok"}]], [1.0], k=60)
    assert _ids(out) == ["ok"]
