# -*- coding: utf-8 -*-
"""端到端集成样例（需 docker compose 起 Qdrant/MinIO + 安装 .[models]）。

默认被 pyproject 的 ``-m "not integration and not slow"``  deselect，CI 不跑。
本地按需执行：``pytest -m integration``。

这一批先建 1-2 个样例锚定链路（ingest 全量到 index → search → ask），任一前置
（重模型 / Qdrant / MinIO）不可用即 skip，不产生红灯。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def services_available() -> None:
    """前置守卫：缺重模型或缺外部服务则跳过整条端到端链路。"""
    pytest.importorskip("torch", reason="未安装 .[models] 重依赖，跳过端到端集成测试")
    pytest.importorskip("FlagEmbedding", reason="缺少 FlagEmbedding，跳过向量入库")

    from qdrant_client import QdrantClient  # 核心依赖，CI 已装

    from ingestion.parsers.common import get_settings

    qcfg = get_settings().get("qdrant") or {}
    host = str(qcfg.get("host") or "localhost")
    port = int(qcfg.get("port") or 6333)
    try:
        client = QdrantClient(host=host, port=port, timeout=3)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant 不可达（{host}:{port}）：{exc}")


def test_e2e_ingest_then_search(services_available, tmp_path: Path):
    """跑通 parse→chunk→embed→index，再 search 命中刚入库的样例正文。"""
    from ingestion.pipeline import run
    from tests.helpers import sample_docs

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    # 生成样例正文（含「采购合同 / 违约责任」等可断词）供流水线扫描入库
    sample_docs.make_txt(in_dir)
    out = tmp_path / "out"
    out.mkdir()

    summary = run(in_dir, "parse", "index", workers=1, root=out)
    assert summary["error"] == 0

    from retrieval.hybrid_search import search

    res = search("违约责任", top_k=5)
    assert res.get("results"), "端到端检索应至少命中一条"
    assert any("违约" in (r.get("content") or "") for r in res["results"])
