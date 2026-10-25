# -*- coding: utf-8 -*-
"""
⑧ 安全批次单测（纯离线，不打网络 / 不加载模型 / 不起外部服务）。

覆盖四类边界：
  - allowed_ingest_roots：环境变量 > settings.security.ingest_roots > 默认 data 的优先级
  - confined_read_path：沙箱内通过；../ 穿越、沙箱外绝对路径、沙箱内软链逃逸均被拒
  - presigned_expiry：取 min(配置, 上限)，默认压到 300s
  - validate_minio_credentials：空 / 弱默认口令被拒，强口令通过
  - tools.ingest / classify_image：越界路径返回 code=400 且下游 pipeline/CLIP 未被调用
  - tools.get_document_url：弱凭据返回 code=403 且 Minio 客户端未构造；强凭据正常签 URL
  - server.main：SSE 绑非本机无 token 拒绝启动；本机回环 / 有 token 放行
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mcp_server import security


# ------------------------------------------------------------------ 取工具底层函数
@pytest.fixture
def tools() -> dict[str, Callable[..., Any]]:
    """注册到临时 FastMCP 后取出工具的原始可调用对象（避开版本内部差异）。"""
    pytest.importorskip("fastmcp", reason="未安装 fastmcp，跳过工具级用例")
    from fastmcp import FastMCP

    from mcp_server.tools import register_tools

    m = FastMCP(name="sec-test")
    register_tools(m)
    tm = m._tool_manager
    return {name: tm._tools[name].fn for name in ("ingest", "classify_image", "get_document_url")}


# ------------------------------------------------------------------ allowed_ingest_roots
def test_allowed_roots_env_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("RAG_ZH_ALLOWED_INGEST_ROOTS", f"{a}{os.pathsep}{b}")
    roots = security.allowed_ingest_roots({"security": {"ingest_roots": ["ignored"]}})
    assert roots == [a.resolve(), b.resolve()]


def test_allowed_roots_settings_second(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RAG_ZH_ALLOWED_INGEST_ROOTS", raising=False)
    monkeypatch.setenv("RAG_ZH_DATA_DIR", str(tmp_path / "unused"))
    cfg_root = tmp_path / "cfgroot"
    roots = security.allowed_ingest_roots({"security": {"ingest_roots": [str(cfg_root)]}})
    assert roots == [cfg_root.resolve()]


def test_allowed_roots_default_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RAG_ZH_ALLOWED_INGEST_ROOTS", raising=False)
    monkeypatch.setenv("RAG_ZH_DATA_DIR", str(tmp_path / "mydata"))
    assert security.allowed_ingest_roots({}) == [(tmp_path / "mydata").resolve()]


# ------------------------------------------------------------------ confined_read_path
def test_confined_allows_inside(tmp_path: Path) -> None:
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    assert security.confined_read_path(f, [tmp_path]) == f.resolve()


def test_confined_blocks_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    with pytest.raises(security.PathNotAllowed):
        security.confined_read_path(root / ".." / "secret.txt", [root])


def test_confined_blocks_absolute_outside(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "etc" / "passwd"
    outside.parent.mkdir()
    outside.write_text("x")
    with pytest.raises(security.PathNotAllowed):
        security.confined_read_path(str(outside), [root])


def test_confined_blocks_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("s")
    link = root / "link.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows 无管理员权限时
        pytest.skip("当前系统/权限不支持创建符号链接")
    with pytest.raises(security.PathNotAllowed):
        security.confined_read_path(link, [root])


# ------------------------------------------------------------------ presigned_expiry
def test_presigned_expiry_capped_to_max() -> None:
    s = {"minio": {"presigned_expire_seconds": 3600}, "security": {"max_presigned_seconds": 300}}
    assert security.presigned_expiry(s) == 300


def test_presigned_expiry_below_cap() -> None:
    s = {"minio": {"presigned_expire_seconds": 120}, "security": {"max_presigned_seconds": 300}}
    assert security.presigned_expiry(s) == 120


def test_presigned_expiry_defaults() -> None:
    assert security.presigned_expiry({}) == 300  # min(3600, 300)


# ------------------------------------------------------------------ validate_minio_credentials
def test_validate_empty_creds_rejected() -> None:
    with pytest.raises(security.InsecureConfigError):
        security.validate_minio_credentials("", "secret123")
    with pytest.raises(security.InsecureConfigError):
        security.validate_minio_credentials("key", "")


def test_validate_weak_creds_rejected() -> None:
    with pytest.raises(security.InsecureConfigError):
        security.validate_minio_credentials("minioadmin", "minioadmin")


def test_validate_strong_creds_ok() -> None:
    security.validate_minio_credentials("ak_strong", "sk_strong_xyz")  # 不抛即通过


# ------------------------------------------------------------------ tools.ingest 沙箱
def test_ingest_blocks_outside_root(
    tools: dict[str, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("RAG_ZH_ALLOWED_INGEST_ROOTS", str(root))
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"x")

    import ingestion.pipeline as pipeline

    called = {"v": False}

    def _spy(*_a: Any, **_k: Any) -> dict[str, Any]:
        called["v"] = True
        return {}

    monkeypatch.setattr(pipeline, "run", _spy)
    out = tools["ingest"](str(outside))
    assert called["v"] is False, "越界路径不应触达 pipeline"
    assert isinstance(out, dict) and out.get("code") == 400 and "error" in out


def test_classify_image_blocks_outside(
    tools: dict[str, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("RAG_ZH_ALLOWED_INGEST_ROOTS", str(root))
    evil = tmp_path / "evil.png"
    evil.write_bytes(b"x")

    import ingestion.classifiers.image_classifier as ic

    called = {"v": False}

    def _spy(_p: str) -> dict[str, Any]:
        called["v"] = True
        return {"label": "chart"}

    monkeypatch.setattr(ic, "classify_image_path", _spy)
    out = tools["classify_image"](str(evil))
    assert called["v"] is False, "越界图片不应触达分类器"
    assert isinstance(out, dict) and out.get("code") == 400


def test_classify_image_allows_inside(
    tools: dict[str, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    img = root / "a.png"
    img.write_bytes(b"real")
    monkeypatch.setenv("RAG_ZH_ALLOWED_INGEST_ROOTS", str(root))

    import ingestion.classifiers.image_classifier as ic

    seen: dict[str, str] = {}

    def _spy(p: str) -> dict[str, Any]:
        seen["p"] = p
        return {"label": "chart"}

    monkeypatch.setattr(ic, "classify_image_path", _spy)
    out = tools["classify_image"](str(img))
    assert seen["p"] == str(img.resolve())
    assert out["label"] == "chart"


# ------------------------------------------------------------------ tools.get_document_url
def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any]) -> None:
    from ingestion.parsers import common

    monkeypatch.setattr(common, "get_settings", lambda: settings)


def test_get_document_url_denies_weak_creds(
    tools: dict[str, Callable[..., Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_settings(
        monkeypatch,
        {"minio": {"access_key": "minioadmin", "secret_key": "minioadmin"}, "security": {}},
    )
    import minio

    def _boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("凭据不安全时不应构造 Minio 客户端")

    monkeypatch.setattr(minio, "Minio", _boom)
    out = tools["get_document_url"]("doc123")
    assert isinstance(out, dict) and out.get("code") == 403


def test_get_document_url_signs_with_strong_creds(
    tools: dict[str, Callable[..., Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_settings(
        monkeypatch,
        {
            "minio": {
                "access_key": "ak_strong",
                "secret_key": "sk_strong",
                "endpoint": "localhost:9000",
                "bucket": "b",
                "object_key_prefix": "raw",
                "presigned_expire_seconds": 3600,
            },
            "security": {"max_presigned_seconds": 300},
        },
    )
    import minio

    import index.metadata_store as ms

    monkeypatch.setattr(ms, "get_document", lambda doc_id: {"path": "raw/doc1/f.pdf"})

    class _Client:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def presigned_get_object(self, bucket: str, key: str, expires: Any = None) -> str:
            return f"http://minio/{bucket}/{key}?exp={expires.total_seconds()}"

    monkeypatch.setattr(minio, "Minio", _Client)
    out = tools["get_document_url"]("doc1")
    assert isinstance(out, str) and out.startswith("http")
    # 配置 3600 被上限压到 300
    assert "exp=300" in out


# ------------------------------------------------------------------ server.main 绑定门禁
def _patch_server_light(monkeypatch: pytest.MonkeyPatch, run_flag: dict[str, bool]) -> None:
    import mcp_server.server as srv

    monkeypatch.setattr(srv, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(srv, "_log_startup_banner", lambda: None)
    monkeypatch.setattr(srv, "_warm_up", lambda: None)

    class _Dummy:
        def run(self, *_a: Any, **_k: Any) -> None:
            run_flag["v"] = True

    monkeypatch.setattr(srv, "build_server", lambda: _Dummy())


def test_sse_refuses_external_bind_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_server.server as srv

    flag: dict[str, bool] = {"v": False}
    _patch_server_light(monkeypatch, flag)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(srv, "get_settings", lambda: {"mcp": {}})
    rc = srv.main(["--sse", "--host", "0.0.0.0", "--no-warmup"])
    assert rc != 0
    assert flag["v"] is False, "被拒时不应调用 mcp.run"


def test_sse_allows_loopback_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_server.server as srv

    flag: dict[str, bool] = {"v": False}
    _patch_server_light(monkeypatch, flag)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(srv, "get_settings", lambda: {"mcp": {}})
    rc = srv.main(["--sse", "--host", "127.0.0.1", "--no-warmup"])
    assert rc == 0
    assert flag["v"] is True


def test_sse_allows_external_bind_with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_server.server as srv

    flag: dict[str, bool] = {"v": False}
    _patch_server_light(monkeypatch, flag)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(srv, "get_settings", lambda: {"mcp": {}})
    rc = srv.main(["--sse", "--host", "0.0.0.0", "--no-warmup"])
    assert rc == 0
    assert flag["v"] is True
