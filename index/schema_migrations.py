# -*- coding: utf-8 -*-
"""
Schema 版本化与迁移（PRAGMA user_version 驱动）。

背景：[keyword_store.init_tables] 一直只用 `CREATE TABLE IF NOT EXISTS` 建表，
没有任何版本追踪 —— 一旦表结构要变（加列 / 加索引），既有库不会被更新，新代码
却按新结构读写，跨版本静默错位。本模块引入 SQLite 官方的 `PRAGMA user_version`
做库内版本戳，配一份显式的迁移表，把「建表」与「升级」两件事分开。

设计红线：
  1. **升级幂等**：每个 up() 自带守卫（列存在检查 / CREATE INDEX IF NOT EXISTS），
     重复跑或跑一次已升级的库都不报错，也不重复改结构。
  2. **不猜 baseline**：user_version==0 代表「版本化机制之前建的库或全新库」——
     此时表已由 init_tables 建好，直接把 baseline 视为 BASELINE_VERSION 盖章，
     然后照常往上跑到 SCHEMA_VERSION（不重跑建表 DDL）。
  3. **拒绝静默降级**：库版本 > 代码版本（拿旧代码开新库）抛清晰 RuntimeError，
     提示「请用更新后的代码或从备份恢复」，绝不假装能自动回退。真正的回退只由
     显式 migrate_to(target) 触发（运维动作，不在常规启动路径上）。

典型用法（由 keyword_store.init_tables 末尾调用）：
    from index.schema_migrations import run_migrations
    run_migrations(conn)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

# baseline：本机制引入时已存在的表结构（documents/chunks/fts/ngram/vocab/synonyms）。
BASELINE_VERSION = 1
# 目标版本 = baseline + 本文件 MIGRATIONS 里的增量项数。加一条迁移就 +1。
SCHEMA_VERSION = 1 + 1  # v2：documents.updated_at + 生命周期加速索引


def get_version(conn: sqlite3.Connection) -> int:
    """读当前库版本（PRAGMA user_version，未盖章时为 0）。"""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def set_version(conn: sqlite3.Connection, version: int) -> None:
    """写库版本戳。user_version 是整数 pragma，不支持参数占位符，只能格式化写入。"""
    conn.execute(f"PRAGMA user_version={int(version)}")
    conn.commit()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """列出现有列名（ALTER ADD COLUMN 前查重，SQLite 无 IF NOT EXISTS 语法）。"""
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# =================================================================== 迁移定义
@dataclass(frozen=True)
class Migration:
    """一条版本迁移：升到 version 调 up，退到 version-1 调 down。"""

    version: int
    up: Callable[[sqlite3.Connection], None]
    down: Callable[[sqlite3.Connection], None]


def _v2_up(conn: sqlite3.Connection) -> None:
    """v1 -> v2：为数据生命周期加速 —— 文档更新时间列 + doc_id/chunk_id 反查索引。"""
    if "updated_at" not in _table_columns(conn, "documents"):
        conn.execute("ALTER TABLE documents ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
    # 生命周期删除/查询大量按 doc_id、chunk_id 过滤，无索引时全表扫
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ngram_chunk ON ngram_index(chunk_id)")
    conn.commit()


def _v2_down(conn: sqlite3.Connection) -> None:
    """
    v2 -> v1：回退索引。
    documents.updated_at 列**不删**：SQLite 的 DROP COLUMN 直到 3.35 才支持且对
    带索引/主键的表有限制，回退里动列风险远大于收益；保留一个空列是无害的，
    真正的向下兼容靠代码读写用 IF EXISTS / 列存在检查兜住。这里记录该已知限制。
    """
    conn.execute("DROP INDEX IF EXISTS idx_chunks_doc")
    conn.execute("DROP INDEX IF EXISTS idx_ngram_chunk")
    conn.commit()


# 版本 -> 迁移。加迁移时：追加一条并把 SCHEMA_VERSION 同步 +1。
MIGRATIONS: dict[int, Migration] = {
    2: Migration(version=2, up=_v2_up, down=_v2_down),
}


# =================================================================== 迁移执行
def run_migrations(conn: sqlite3.Connection) -> int:
    """
    把库升级到 SCHEMA_VERSION（幂等，供 init_tables 末尾调用）。

    返回本次实际应用的迁移条数（0 表示已是最新）。库版本高于代码版本时抛错，
    拒绝用旧代码打开新库。
    """
    current = get_version(conn)
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"数据库 schema 版本({current})高于当前代码({SCHEMA_VERSION})；"
            f"请升级到匹配的代码版本，或从备份恢复，勿用旧代码强行打开。"
        )
    # user_version==0：版本化前的库（表已由 CREATE IF NOT EXISTS 建好）—— 视为 baseline
    if current == 0:
        set_version(conn, BASELINE_VERSION)
        current = BASELINE_VERSION

    applied = 0
    for version in range(current + 1, SCHEMA_VERSION + 1):
        migration = MIGRATIONS[version]
        migration.up(conn)
        set_version(conn, version)
        applied += 1
    return applied


def migrate_to(conn: sqlite3.Connection, target: int) -> int:
    """
    显式把库迁移到 target 版本（可升可降，运维动作，不在常规启动路径）。

    升级走 run_migrations 的逐条 up；降级逐条调 down（从当前版本往下）。
    target 必须在 [BASELINE_VERSION, SCHEMA_VERSION] 内，否则 ValueError。
    返回实际执行的迁移条数。
    """
    if target < BASELINE_VERSION or target > SCHEMA_VERSION:
        raise ValueError(f"目标版本 {target} 越界，允许区间 [{BASELINE_VERSION}, {SCHEMA_VERSION}]")

    current = get_version(conn)
    if current == 0:  # 未盖章：视为 baseline 再比较
        set_version(conn, BASELINE_VERSION)
        current = BASELINE_VERSION

    count = 0
    if target > current:
        for version in range(current + 1, target + 1):
            MIGRATIONS[version].up(conn)
            set_version(conn, version)
            count += 1
    elif target < current:
        for version in range(current, target, -1):
            MIGRATIONS[version].down(conn)
            set_version(conn, version - 1)
            count += 1
    return count
