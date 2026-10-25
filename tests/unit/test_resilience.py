# -*- coding: utf-8 -*-
"""
⑥-C 外部存储容错单测（纯离线，不打网络 / 不加载模型 / 不起 Qdrant）。

覆盖：
  - index/resilience：is_transient_error 分类、read_retry_policy、retry_transient 退避语义
  - retrieval/hybrid_search：整次召回超预算时未完成的各路降级为空（bounded degradation）
  - index/vector_store：读路径瞬时错误退避重试贯通后成功
  - mcp_server/tools：MinIO 客户端单例（多次 presign 只构造一次）
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from index import resilience


# ------------------------------------------------------------------ is_transient_error
def test_is_transient_builtin_connection() -> None:
    assert resilience.is_transient_error(ConnectionError("reset"))
    assert resilience.is_transient_error(TimeoutError("slow"))


def test_is_transient_by_name() -> None:
    class ConnectLost(Exception):  # 名字含 "Connect"
        pass

    class PlainError(Exception):
        pass

    assert resilience.is_transient_error(ConnectLost())
    assert not resilience.is_transient_error(PlainError())


def test_is_transient_qdrant_status() -> None:
    UnexpectedResponse = pytest.importorskip(
        "qdrant_client.http.exceptions", reason="未安装 qdrant_client"
    ).UnexpectedResponse
    from httpx import Headers

    srv_503 = UnexpectedResponse(503, "unavailable", b"", Headers())
    client_400 = UnexpectedResponse(400, "bad request", b"", Headers())
    assert resilience.is_transient_error(srv_503)
    assert not resilience.is_transient_error(client_400)


def test_is_transient_grpc_unavailable() -> None:
    class _Code:
        def __str__(self) -> str:
            return "StatusCode.UNAVAILABLE"

    class RpcError(Exception):
        def code(self) -> Any:
            return _Code()

    assert resilience.is_transient_error(RpcError())


# ------------------------------------------------------------------ read_retry_policy
def test_read_retry_policy_defaults() -> None:
    assert resilience.read_retry_policy({}) == (3, 0.2)


def test_read_retry_policy_override() -> None:
    s = {"resilience": {"qdrant": {"retry_attempts": 5, "retry_base_delay_seconds": 0.5}}}
    assert resilience.read_retry_policy(s) == (5, 0.5)


def test_read_retry_policy_min_one() -> None:
    attempts, _ = resilience.read_retry_policy({"resilience": {"qdrant": {"retry_attempts": 0}}})
    assert attempts == 1


# ------------------------------------------------------------------ retry_transient
def test_retry_backs_off_then_succeeds() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    out = resilience.retry_transient(fn, attempts=3, base_delay=0.2, sleep=slept.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert slept == [0.2, 0.4]  # base*2**0, base*2**1


def test_retry_non_transient_raises_immediately() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def fn() -> None:
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        resilience.retry_transient(fn, attempts=3, base_delay=0.2, sleep=slept.append)
    assert calls["n"] == 1
    assert slept == []


def test_retry_exhausts_and_raises() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def fn() -> None:
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        resilience.retry_transient(fn, attempts=3, base_delay=0.1, sleep=slept.append)
    assert calls["n"] == 3
    assert slept == [0.1, 0.2]


def test_retry_single_attempt_no_retry() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        resilience.retry_transient(fn, attempts=1, base_delay=0.1, sleep=lambda d: None)
    assert calls["n"] == 1


# ------------------------------------------------------------------ 召回超时预算降级
def test_parallel_recall_degrades_on_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from retrieval.hybrid_search import HybridSearcher

    searcher = HybridSearcher()
    monkeypatch.setattr(searcher, "_recall_timeout", 0.2)

    wake = threading.Event()

    def slow_dense(_q: str, _k: int, _f: Any) -> list[dict[str, Any]]:
        wake.wait(5)  # 阻塞超过预算；测试结束由 finally 唤醒，不留孤儿线程
        return [{"chunk_id": "d"}]

    def fast_exact(_q: str, _k: int) -> list[dict[str, Any]]:
        return [{"chunk_id": "e"}]

    monkeypatch.setattr(searcher, "_recall_dense", slow_dense)
    monkeypatch.setattr(searcher, "_recall_exact", fast_exact)

    started = time.perf_counter()
    try:
        res = searcher._parallel_recall("q", {"exact": 1.0, "dense": 1.0}, None)
    finally:
        wake.set()
    elapsed = time.perf_counter() - started

    assert res["exact"] == [{"chunk_id": "e"}]
    assert res["dense"] == []
    assert elapsed < 2.0  # 远小于 slow_dense 的 5s：预算生效、未被拖住


# ------------------------------------------------------------------ vector_store 读重试贯通
def test_vector_store_retries_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    from index import vector_store as vs

    hit = type(
        "H",
        (),
        {"id": "p1", "score": 0.9, "payload": {"chunk_id": "c1", "doc_id": "d1", "content": "x"}},
    )()

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, **_kw: Any) -> list[Any]:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("reset")
            return [hit]

    client = FlakyClient()
    monkeypatch.setattr(vs, "get_client", lambda: client)
    monkeypatch.setattr(vs, "_collection_name", lambda: "coll")
    monkeypatch.setattr(
        vs,
        "get_settings",
        lambda: {"resilience": {"qdrant": {"retry_attempts": 3, "retry_base_delay_seconds": 0.2}}},
    )
    # 退避别真等
    monkeypatch.setattr(resilience.time, "sleep", lambda _d: None)

    results = vs.hybrid_search(query_dense=[0.1, 0.2], query_sparse=None, top_k=10)
    assert results and results[0]["chunk_id"] == "c1"
    assert client.calls == 2  # 首次瞬时失败 + 重试成功


# ------------------------------------------------------------------ MinIO 客户端单例
def test_minio_client_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import minio

    from index import metadata_store as ms
    from ingestion.parsers import common
    from mcp_server import tools

    count = {"n": 0}

    class FakeMinio:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            count["n"] += 1

        def presigned_get_object(self, bucket: str, key: str, expires: Any = None) -> str:
            return f"http://minio/{bucket}/{key}"

    tools._get_minio_client.cache_clear()
    try:
        monkeypatch.setattr(minio, "Minio", FakeMinio)
        monkeypatch.setattr(
            common,
            "get_settings",
            lambda: {
                "minio": {
                    "access_key": "ak_strong",
                    "secret_key": "sk_strong",
                    "endpoint": "e:9000",
                    "bucket": "b",
                    "object_key_prefix": "raw",
                    "region": "r",
                },
                "security": {"max_presigned_seconds": 300},
            },
        )
        monkeypatch.setattr(ms, "get_document", lambda doc_id: {"path": "raw/d1/f.pdf"})

        u1 = tools._presigned_url("d1")
        u2 = tools._presigned_url("d1")
        assert u1 == u2 == "http://minio/b/raw/d1/f.pdf"
        assert count["n"] == 1  # 两次调用只构造一个客户端
    finally:
        tools._get_minio_client.cache_clear()


# ------------------------------------------------------------------ retry 工厂签名哨兵
def test_retry_transient_passes_no_arg_callable() -> None:
    # retry_transient 期望无参 callable；确认 lambda 捕获入参的调用方式在真实调用点成立
    got: dict[str, int] = {}

    def work() -> int:
        got["v"] = 42
        return got["v"]

    result = resilience.retry_transient(work, attempts=2, base_delay=0.0, sleep=lambda d: None)
    assert result == 42
    assert got["v"] == 42
