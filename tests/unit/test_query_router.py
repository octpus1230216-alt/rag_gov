# -*- coding: utf-8 -*-
"""查询路由离线自检（由 _check_batch6.py 迁移）。route() 是纯正则函数，无需分词/模型。"""

from __future__ import annotations

from retrieval.query_router import route


def test_article_query_lowers_dense():
    assert route("第三条 违约责任")["dense"] == 0.3


def test_quoted_query_lowers_fuzzy():
    assert route("“商业秘密”")["fuzzy"] == 0.3  # 中文双引号


def test_date_query_exact():
    assert route("2024-05-01 会议")["exact"] == 1.0


def test_amount_query_sparse():
    assert route("¥1,000,000")["sparse"] == 0.8


def test_code_query_sparse():
    assert route("HT-2024-001")["sparse"] == 1.0


def test_short_query_fuzzy():
    assert route("请假")["fuzzy"] == 1.0  # 短查询


def test_question_query_dense():
    # 疑问句 → 语义优先
    assert route("这个平台到底怎么样才能提高大家的整体协作效率呢？")["dense"] == 1.0


def test_route_keys_and_range():
    weights = route("随便查点东西")
    assert set(weights) == {"exact", "fuzzy", "sparse", "dense"}
    assert all(0 < v <= 1.0 for v in weights.values())
