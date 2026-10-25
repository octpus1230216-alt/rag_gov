# -*- coding: utf-8 -*-
"""可观测性改造单测（caplog + capsys）：
1. _fail 以 ERROR 级 + traceback 落日志且返回结构化错误；
2. request_id 经 contextvar + RequestIdFilter 注入记录，出上下文复原；
3. _traced 埋点带 rid、只记元数据不泄漏 query 正文；
4. setup_logging 走 stderr 不污染 stdout、幂等不重复挂 handler；
5. logging 配置解析（autodetect 段 + _resolve_level/_resolve_log_file）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import config.logging_setup as ls
from mcp_server.tools import _fail, _traced


@pytest.fixture(autouse=True)
def _isolate_rag_zh_logger():
    """每个用例前后清干净 rag_zh 日志状态，避免用例间相互污染（handler/propagate/初始化标记）。"""
    root = logging.getLogger(ls.ROOT_LOGGER_NAME)
    saved = (list(root.handlers), root.propagate, root.level)
    saved_init = ls._INITIALIZED
    for h in root.handlers:
        if getattr(h, "_rag_zh_owned", False):
            h.close()
    root.handlers.clear()
    root.propagate = True
    ls._INITIALIZED = False
    yield
    for h in root.handlers:
        if getattr(h, "_rag_zh_owned", False):
            h.close()
    root.handlers.clear()
    root.handlers.extend(saved[0])
    root.propagate = saved[1]
    root.level = saved[2]
    ls._INITIALIZED = saved_init


# ------------------------------------------------------------------ _fail ERROR
def test_fail_logs_error_with_traceback(caplog):
    from mcp_server import tools

    with caplog.at_level(logging.ERROR, logger="rag_zh.mcp.tools"):
        try:
            raise ValueError("boom-detail")
        except ValueError as exc:
            out = _fail(exc)

    assert out == {"error": "ValueError", "detail": "boom-detail"}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "期望 _fail 落一条 ERROR"
    assert "boom-detail" in errors[0].getMessage()
    # logger.exception 应带上当前异常栈
    assert errors[0].exc_info is not None
    assert "Traceback" in caplog.text
    assert tools is not None  # 引用模块确保 import 生效


# ------------------------------------------------------------------ request_id
def test_request_id_filter_injects_record():
    filt = ls.RequestIdFilter()
    with ls.request_context("rid-123"):
        rec = logging.LogRecord("n", logging.INFO, "p", 1, "msg", None, None)
        assert filt.filter(rec) is True
        assert rec.request_id == "rid-123"
    # 出上下文后回到默认占位
    assert ls.get_request_id() == "-"


def test_request_context_nested_restores():
    with ls.request_context("outer"):
        assert ls.get_request_id() == "outer"
        with ls.request_context("inner"):
            assert ls.get_request_id() == "inner"
        assert ls.get_request_id() == "outer"
    assert ls.get_request_id() == "-"


def test_request_context_generates_id_when_omitted():
    with ls.request_context() as rid:
        assert rid and rid != "-"
        assert ls.get_request_id() == rid


# ------------------------------------------------------------------ _traced 埋点
def test_traced_logs_start_end_with_rid_and_no_query_body(caplog):
    secret_query = "涉密查询正文绝不该出现在日志里"
    with (
        caplog.at_level(logging.INFO, logger="rag_zh.mcp.tools"),
        _traced("search", query_len=len(secret_query), top_k=5, mode="hybrid"),
    ):
        logging.getLogger("rag_zh.mcp.tools").info("inside body")
        rid_inside = ls.get_request_id()

    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("tool start rid=") and "tool=search" in m for m in msgs)
    assert any("tool end rid=" in m and "elapsed=" in m for m in msgs)
    # 埋点期间的日志也带上同一个 rid（非默认占位）
    assert rid_inside and rid_inside != "-"
    assert f"query_len={len(secret_query)}" in " ".join(msgs)
    # 脱敏红线：只记长度，绝不记正文
    assert secret_query not in caplog.text


# ------------------------------------------------------------------ setup_logging
def test_setup_logging_writes_stderr_not_stdout(capsys, monkeypatch):
    monkeypatch.setattr(
        "ingestion.parsers.common.get_settings",
        lambda: {"logging": {"level": "INFO", "to_stderr": True, "to_file": False}},
    )
    ls.setup_logging()
    logging.getLogger("rag_zh.probe").info("stdout-leak-check")

    captured = capsys.readouterr()
    assert "stdout-leak-check" not in captured.out, "日志绝不能泄漏到 stdout"
    assert "stdout-leak-check" in captured.err, "日志应走 stderr"


def test_setup_logging_idempotent(monkeypatch):
    monkeypatch.setattr(
        "ingestion.parsers.common.get_settings",
        lambda: {"logging": {"to_stderr": True, "to_file": False}},
    )
    logger = ls.setup_logging()
    n = len(logger.handlers)
    assert n >= 1
    ls.setup_logging()
    assert len(logger.handlers) == n, "重复调用不应叠加 handler"
    ls.setup_logging(force=True)
    assert len(logger.handlers) == n, "force 重建后数量不变"


def test_setup_logging_creates_rotating_file(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        "ingestion.parsers.common.get_settings",
        lambda: {
            "logging": {
                "level": "INFO",
                "to_stderr": False,
                "to_file": True,
                "directory": str(log_dir),
            }
        },
    )
    logger = ls.setup_logging()
    logging.getLogger("rag_zh.filecheck").info("written-to-file")
    for h in logger.handlers:
        h.flush()
    assert (log_dir / "rag_zh.log").exists()
    assert "written-to-file" in (log_dir / "rag_zh.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------ get_logger
def test_get_logger_prefixes_and_initializes(monkeypatch):
    monkeypatch.setattr(
        "ingestion.parsers.common.get_settings",
        lambda: {"logging": {"to_stderr": True, "to_file": False}},
    )
    lg = ls.get_logger("rag_zh.custom")
    assert lg.name == "rag_zh.custom"
    lg2 = ls.get_logger("bare")
    assert lg2.name == "rag_zh.bare"


# ------------------------------------------------------------------ 配置解析
def test_resolve_level_priority_and_fallback():
    assert ls._resolve_level({"level": "DEBUG"}, {}) == logging.DEBUG
    # logging.level 缺省回落 runtime.log_level
    assert ls._resolve_level({}, {"runtime": {"log_level": "WARNING"}}) == logging.WARNING
    # 非法值兜底 INFO
    assert ls._resolve_level({"level": "bogus"}, {}) == logging.INFO
    assert ls._resolve_level({}, {}) == logging.INFO


def test_resolve_log_file():
    assert ls._resolve_log_file({"to_file": False}, {}) is None
    p = ls._resolve_log_file({"to_file": True, "directory": "logs", "file_name": "x.log"}, {})
    assert p is not None and str(p) == os.path.join("logs", "x.log")
    # 兼容旧 runtime.log_dir
    p2 = ls._resolve_log_file({"to_file": True}, {"runtime": {"log_dir": "old_logs"}})
    assert p2 is not None and str(p2) == os.path.join("old_logs", "rag_zh.log")


def test_logging_section_parsed_from_example(repo_root: Path):
    from config.autodetect import load_settings

    st = load_settings(repo_root / "config" / "settings.example.yaml")
    lg = st["logging"]
    assert lg["to_stderr"] is True
    assert lg["to_file"] is True
    # ${RAG_ZH_LOG_LEVEL:-INFO} 展开后为合法级别字符串
    assert str(lg["level"]).strip().upper() in {"DEBUG", "INFO", "WARNING", "ERROR"}


def test_logging_setdefault_when_template_missing(tmp_path: Path):
    from config.autodetect import load_settings

    minimal = tmp_path / "no_logging.yaml"
    minimal.write_text("profile: lite\nprofiles:\n  lite: {}\n", encoding="utf-8")
    st = load_settings(minimal)
    assert "logging" in st
    assert st["logging"]["to_stderr"] is True
