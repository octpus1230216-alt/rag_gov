# -*- coding: utf-8 -*-
"""并发正确性 + SQLite 事务批写的离线自检。

A：模型懒加载双检锁 —— 用假 FlagEmbedding 模块（构造计数 + 轻微 sleep 放大竞态窗口），
   多线程并发触发首次加载，断言只构造一次（旧实现无锁会 N 次）。
B：insert_chunks_batch 单事务 —— 计数代理连接断言整批只 commit 一次；再验数据完整、
   幂等覆盖，以及中途异常整批回滚（不留半批）。
全程不加载真模型、不连外部服务。"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest


# ============================================================ SQLite 隔离夹具
@pytest.fixture
def sqlite_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from index import keyword_store

    db = tmp_path / "conc.sqlite"
    monkeypatch.setattr(keyword_store, "_DB_PATH", db, raising=False)
    keyword_store._INITIALISED = False
    keyword_store.close_connection()
    yield db
    keyword_store.close_connection()
    keyword_store._INITIALISED = False


# ============================================================ 假 FlagEmbedding（并发加载计数）
class _FakeModelBase:
    """记录构造次数并轻微 sleep，放大无锁时的竞态窗口。"""

    constructions = 0
    _lock = threading.Lock()

    def __init__(self, *args, **kwargs) -> None:
        with _FakeModelBase._lock:
            _FakeModelBase.constructions += 1
        threading.Event().wait(0.02)


def _install_fake_flagembedding(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeModelBase.constructions = 0

    class _BGEM3FlagModel(_FakeModelBase):
        pass

    class _FlagReranker(_FakeModelBase):
        pass

    fake = types.ModuleType("FlagEmbedding")
    fake.BGEM3FlagModel = _BGEM3FlagModel  # type: ignore[attr-defined]
    fake.FlagModel = _BGEM3FlagModel  # type: ignore[attr-defined]
    fake.FlagReranker = _FlagReranker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake)


def _hammer(fn, n_threads: int = 6) -> None:
    """让 n_threads 个线程在同一屏障起跑，并发执行 fn（制造加载竞争）。"""
    barrier = threading.Barrier(n_threads)

    def _worker() -> None:
        barrier.wait()
        fn()

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ============================================================ A 双检锁
def test_embedder_loads_model_once_under_concurrency(monkeypatch: pytest.MonkeyPatch):
    _install_fake_flagembedding(monkeypatch)
    from embedding.text_embedder import TextEmbedder

    emb = TextEmbedder()
    emb._supports_sparse = True
    assert emb._model is None

    _hammer(emb._load_model)

    assert _FakeModelBase.constructions == 1  # 关键：并发下只构造一份
    assert emb._model is not None


def test_reranker_loads_model_once_under_concurrency(monkeypatch: pytest.MonkeyPatch):
    _install_fake_flagembedding(monkeypatch)
    from retrieval.reranker import Reranker

    rr = Reranker()
    assert rr._model is None

    _hammer(rr._load)

    assert _FakeModelBase.constructions == 1
    assert rr._model is not None


# ============================================================ B 单事务提交
class _CountingConn:
    """转发到真实连接，但对 commit() 计数，用于断言整批只提交一次。"""

    def __init__(self, conn) -> None:
        self._conn = conn
        self.commits = 0

    def commit(self):
        self.commits += 1
        return self._conn.commit()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _mk_chunk(doc_id: str, cid: str) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": doc_id,
        "content": f"合同 付款 条款 {cid}",
        "chunk_type": "text",
        "section_path": "第1章",
        "metadata": {"doc_type": "contract"},
    }


def test_insert_chunks_batch_single_commit(sqlite_db: Path, monkeypatch: pytest.MonkeyPatch):
    from index import keyword_store, metadata_store

    # 先建表 + 登记文档（满足 chunks -> documents 外键），并在打桩前完成，避免计入提交数
    keyword_store.init_tables()
    metadata_store.insert_document({"doc_id": "dBatch", "path": "dBatch.pdf"})
    conn = keyword_store.get_connection()
    keyword_store._INITIALISED = True  # 让 insert_chunks_batch 内 init_tables 直接短路

    counter = _CountingConn(conn)
    monkeypatch.setattr(keyword_store, "get_connection", lambda: counter)

    n = keyword_store.insert_chunks_batch([_mk_chunk("dBatch", f"c{i}") for i in range(5)])

    assert n == 5
    assert counter.commits == 1  # 关键：整批一次提交（旧实现是 5 次）


def test_insert_chunks_batch_data_and_idempotent(sqlite_db: Path):
    from index import keyword_store, metadata_store

    keyword_store.init_tables()
    metadata_store.insert_document({"doc_id": "dData", "path": "dData.pdf"})
    conn = keyword_store.get_connection()

    chunks = [_mk_chunk("dData", "x1"), _mk_chunk("dData", "x2")]
    assert keyword_store.insert_chunks_batch(chunks) == 2
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id='dData'").fetchone()[0] == 2
    cid = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='dData'")}
    assert cid == {"x1", "x2"}
    # fts 也写入（按 chunk_id 数）
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM fts_chunks_virt WHERE chunk_id IN ('x1','x2')"
        ).fetchone()[0]
        == 2
    )

    # 幂等：重跑同批不新增（INSERT OR REPLACE）
    keyword_store.insert_chunks_batch(chunks)
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id='dData'").fetchone()[0] == 2


def test_insert_chunks_batch_rolls_back_on_failure(
    sqlite_db: Path, monkeypatch: pytest.MonkeyPatch
):
    """中途异常：整批回滚，不留半批。"""
    from index import keyword_store, metadata_store

    keyword_store.init_tables()
    metadata_store.insert_document({"doc_id": "dRb", "path": "dRb.pdf"})
    conn = keyword_store.get_connection()
    keyword_store._INITIALISED = True

    class _FailOnSecondExecutemany:
        def __init__(self, c):
            self._c = c
            self.exec_calls = 0
            self.rollbacks = 0

        def executemany(self, sql, rows):
            self.exec_calls += 1
            if self.exec_calls == 2:
                raise RuntimeError("boom")  # 模拟 fts 批量写失败
            return self._c.executemany(sql, rows)

        def rollback(self):
            self.rollbacks += 1
            return self._c.rollback()

        def commit(self):
            return self._c.commit()

        def __getattr__(self, name):
            return getattr(self._c, name)

    failer = _FailOnSecondExecutemany(conn)
    monkeypatch.setattr(keyword_store, "get_connection", lambda: failer)

    with pytest.raises(RuntimeError, match="boom"):
        keyword_store.insert_chunks_batch([_mk_chunk("dRb", "r1"), _mk_chunk("dRb", "r2")])

    assert failer.rollbacks == 1
    # 第一批 chunks executemany 已成功但因未提交且回滚 -> 库里没有
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id='dRb'").fetchone()[0] == 0
