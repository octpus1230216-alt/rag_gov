# -*- coding: utf-8 -*-
"""数据生命周期离线自检：级联删除、FTS/ngram 孤儿回归、更新去孤儿、Schema 迁移。

全部走纯离线 SQLite（tmp_path 隔离，无 Qdrant/MinIO）：向量删除用 monkeypatch
打桩，验证「外部库不可达时本地清理仍完成」的降级契约。锁死 P0 修掉的删除顺序 bug
（旧实现先删 chunks 再用子查询清 fts，导致 fts/ngram 永久残留）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# ============================================================ SQLite 隔离夹具
@pytest.fixture
def sqlite_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 keyword_store 指向临时库并重置模块态，用例结束自动回收连接。"""
    from index import keyword_store

    db = tmp_path / "lifecycle.sqlite"
    monkeypatch.setattr(keyword_store, "_DB_PATH", db, raising=False)
    keyword_store._INITIALISED = False
    keyword_store.close_connection()
    yield db
    keyword_store.close_connection()
    keyword_store._INITIALISED = False


def _count(conn: Any, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])


def _seed_doc(doc_id: str, chunk_ids: list[str]) -> None:
    """登记一个文档 + 一批 chunk（写 chunks/fts）+ ngram，模拟已入库。"""
    from index import keyword_store, metadata_store, ngram_index

    keyword_store.init_tables()
    metadata_store.insert_document(
        {"doc_id": doc_id, "path": f"{doc_id}.pdf", "source_format": "pdf"}
    )
    chunks = [
        {
            "chunk_id": cid,
            "doc_id": doc_id,
            "content": f"合同 付款 条款 违约 责任 {cid}",
            "chunk_type": "text",
            "section_path": "第1章",
            "metadata": {"doc_type": "contract"},
        }
        for cid in chunk_ids
    ]
    keyword_store.insert_chunks_batch(chunks)
    ngram_index.build_ngram_index(chunks)


def _make_artifacts(doc_id: str, root: Path) -> list[Path]:
    """在产物根目录下造出该 doc 的四类阶段产物文件，返回路径列表。"""
    from ingestion.pipeline import artifact_path

    made: list[Path] = []
    for stage in ("parse", "chunk", "embed", "index"):
        path = artifact_path(stage, doc_id, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"doc_id": doc_id}) + "\n", encoding="utf-8")
        made.append(path)
    # CONTENT 策略的 by-path 缓存（内容是 content doc_id，删除时按内容反查）
    from ingestion.pipeline import stage_dir

    cache_dir = stage_dir("parse", root) / "by-path"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "abc123.id"
    cache.write_text(doc_id + "\n", encoding="utf-8")
    made.append(cache)
    return made


# ============================================================ P0.2 FTS/ngram 孤儿回归
def test_purge_document_sqlite_cleans_all(sqlite_db: Path):
    """删除后 fts/ngram/chunks/documents 必须全部归零（旧实现 fts 子查询恒空的 bug）。"""
    from index import keyword_store, metadata_store

    _seed_doc("docA", ["c1", "c2", "c3"])
    conn = keyword_store.get_connection()
    assert _count(conn, "fts_chunks_virt") == 3 and _count(conn, "ngram_index") > 0

    removed = metadata_store.purge_document_sqlite("docA")

    assert removed["chunks"] == 3
    assert removed["fts"] == 3  # 关键回归：fts 真被删了（旧实现这里是 0）
    assert removed["ngram"] > 0
    assert _count(conn, "chunks") == 0
    assert _count(conn, "fts_chunks_virt") == 0
    assert _count(conn, "ngram_index") == 0
    assert metadata_store.get_document("docA") is None


def test_delete_document_is_idempotent(sqlite_db: Path):
    """重复删同一 doc_id 不报错（各层都是删不到就算过）。"""
    from index import metadata_store

    _seed_doc("docIdem", ["x1"])
    metadata_store.purge_document_sqlite("docIdem")
    again = metadata_store.purge_document_sqlite("docIdem")  # 二次删
    assert again == {"chunks": 0, "fts": 0, "ngram": 0}


# ============================================================ P0.3 级联删除（含磁盘产物）
def test_delete_document_cascade_sqlite_and_artifacts(sqlite_db: Path, tmp_path: Path, monkeypatch):
    from index import keyword_store, lifecycle, vector_store

    # 打桩向量删除为 no-op（CI 无 Qdrant），只验本地级联
    monkeypatch.setattr(vector_store, "delete_by_doc_id", lambda doc_id, collection=None: None)

    root = tmp_path / "out"
    _seed_doc("docC", ["m1", "m2"])
    artifacts = _make_artifacts("docC", root)
    assert all(p.exists() for p in artifacts)

    report = lifecycle.delete_document("docC", root=root)

    conn = keyword_store.get_connection()
    assert _count(conn, "chunks") == 0 and _count(conn, "fts_chunks_virt") == 0
    assert _count(conn, "ngram_index") == 0
    assert report["removed"]["vectors"] is True
    assert not any(p.exists() for p in artifacts)  # 磁盘产物（含 by-path 缓存）全清
    assert report["warnings"] == []

    # 幂等：再删一次不抛
    lifecycle.delete_document("docC", root=root)


# ============================================================ P0.3 外部库降级契约
def test_vector_downgrade_warns_not_aborts(sqlite_db: Path, monkeypatch):
    """Qdrant 不可达：默认降级为 warning 且 SQLite 仍被清；strict=True 才冒异常。"""
    from index import keyword_store, lifecycle, vector_store

    def _boom(doc_id: str, collection: str | None = None) -> None:
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(vector_store, "delete_by_doc_id", _boom)

    _seed_doc("docDown", ["d1"])
    report = lifecycle.delete_document("docDown", purge_artifacts=False)
    conn = keyword_store.get_connection()
    assert _count(conn, "chunks") == 0  # 本地清理照常完成
    assert report["removed"]["vectors"] is False
    assert report["warnings"] and "qdrant down" in report["warnings"][0]

    # strict：外部失败必须冒出来
    _seed_doc("docStrict", ["s1"])
    with pytest.raises(ConnectionError, match="qdrant down"):
        lifecycle.delete_document("docStrict", purge_artifacts=False, strict=True)


# ============================================================ P1 更新去孤儿（purge_index）
def test_purge_index_removes_old_chunks_on_reindex(sqlite_db: Path, monkeypatch):
    """重索引前 purge_index 整体替换：旧 chunk_id 必须从 chunks/fts/ngram 消失。"""
    from index import keyword_store, lifecycle, vector_store

    monkeypatch.setattr(vector_store, "delete_by_doc_id", lambda doc_id, collection=None: None)

    _seed_doc("docUp", ["old1", "old2"])
    conn = keyword_store.get_connection()
    assert _count(conn, "chunks") == 2

    # 模拟重新切分后 chunk_id 变化：先清旧索引，再写新批次
    lifecycle.purge_index("docUp", vectors=True)
    assert _count(conn, "chunks") == 0
    assert _count(conn, "ngram_index") == 0

    _seed_doc("docUp", ["new1"])
    ids = {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='docUp'")}
    assert ids == {"new1"}  # 只剩新数据，无 old* 孤儿


# ============================================================ P2 Schema 迁移
def test_schema_migration_up_and_idempotent(sqlite_db: Path):
    from index import keyword_store
    from index.schema_migrations import SCHEMA_VERSION, get_version, run_migrations

    keyword_store.init_tables()
    conn = keyword_store.get_connection()

    assert get_version(conn) == SCHEMA_VERSION
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    assert "updated_at" in cols
    idx = {r[1] for r in conn.execute("PRAGMA index_list(chunks)").fetchall()}
    assert "idx_chunks_doc" in idx

    # 重复跑幂等：已最新，不再执行迁移
    assert run_migrations(conn) == 0


def test_schema_migration_refers_higher_version(sqlite_db: Path):
    """库版本高于代码版本（拿旧代码开新库）必须抛清晰错误，拒绝静默降级。"""
    from index import keyword_store
    from index.schema_migrations import get_version, run_migrations, set_version

    keyword_store.init_tables()
    conn = keyword_store.get_connection()
    set_version(conn, get_version(conn) + 5)
    with pytest.raises(RuntimeError, match="高于当前代码"):
        run_migrations(conn)


def test_schema_migrate_to_downgrades(sqlite_db: Path):
    """migrate_to 回退到 baseline：加速索引被删掉。"""
    from index import keyword_store
    from index.schema_migrations import BASELINE_VERSION, get_version, migrate_to

    keyword_store.init_tables()
    conn = keyword_store.get_connection()

    n = migrate_to(conn, BASELINE_VERSION)
    assert n == 1 and get_version(conn) == BASELINE_VERSION
    idx = {r[1] for r in conn.execute("PRAGMA index_list(chunks)").fetchall()}
    assert "idx_chunks_doc" not in idx
