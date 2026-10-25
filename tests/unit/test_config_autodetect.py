# -*- coding: utf-8 -*-
"""config 层离线自检（由 _check_batch1.py 迁移）：三档画像、RRF 权重守恒、
sparse 不可用重分配、RAG_ZH_PROFILE 覆盖、${ENV:-default} 解析、autodetect CLI。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from config.autodetect import (
    _expand_env_in_value,
    detect_hardware,
    load_settings,
)

EXAMPLE = "config/settings.example.yaml"


@pytest.fixture
def example_path(repo_root: Path) -> Path:
    return repo_root / EXAMPLE


def test_example_yaml_three_profiles(example_path: Path):
    s = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    need_keys = {
        "embedding_model",
        "embedding_dim",
        "device",
        "batch_size",
        "reranker",
        "qdrant",
    }
    for p in ("lite", "standard", "performance"):
        assert need_keys <= set(s["profiles"][p]), p
    q = s["profiles"]
    assert q["lite"]["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert q["lite"]["batch_size"] == 4 and q["lite"]["qdrant"]["hnsw_m"] == 8
    assert q["standard"]["embedding_model"] == "BAAI/bge-m3"
    assert q["standard"]["batch_size"] == 16 and q["standard"]["qdrant"]["hnsw_m"] == 16
    assert q["performance"]["embedding_model"] == "BAAI/bge-m3"
    assert q["performance"]["batch_size"] == 128 and q["performance"]["qdrant"]["hnsw_m"] == 32
    assert q["performance"]["qdrant"]["on_disk"] is False

    r = s["retrieval"]
    assert r["rrf_k"] == 60
    assert r["rrf_weights"] == {"exact": 1.0, "fuzzy": 0.6, "sparse": 1.0, "dense": 1.0}
    assert r["fuzzy_score_cutoff"] == 75 and r["fuzzy_top_k"] == 10 and r["ngram_top_k"] == 20
    assert s["chunking"]["chunk_size"] == 384 and s["chunking"]["chunk_overlap"] == 64


def test_load_settings_weight_conservation(example_path: Path, monkeypatch):
    monkeypatch.delenv("RAG_ZH_PROFILE", raising=False)
    st = load_settings(example_path)
    assert set(st) >= {"profile", "profile_auto", "settings_source", "hardware", "active"}
    weights = st["retrieval"]["rrf_weights"]
    # 权重总量必须守恒（原 4 路之和 = 3.6），sparse 不可用时只是重新分配
    assert abs(sum(weights.values()) - 3.6) < 1e-2
    if st["retrieval"]["sparse_available"] is False:
        assert "sparse" not in weights


@pytest.mark.parametrize(
    "prof,expect_on_disk",
    [("performance", False), ("standard", True)],
)
def test_env_profile_override(example_path: Path, monkeypatch, prof, expect_on_disk):
    monkeypatch.setenv("RAG_ZH_PROFILE", prof)
    st = load_settings(example_path)
    assert st["profile"] == prof
    assert st["active"]["qdrant"]["on_disk"] is expect_on_disk
    weights = st["retrieval"]["rrf_weights"]
    # 显式指定档：四路权重原样保留（sum=3.6，fuzzy=0.6）
    assert abs(sum(weights.values()) - 3.6) < 1e-6
    assert weights.get("fuzzy") == 0.6


def test_sparse_unavailable_redistributes(example_path: Path, monkeypatch):
    # lite 档 supports_sparse=false → sparse 路应被移除且权重重新分配（总量守恒）
    monkeypatch.setenv("RAG_ZH_PROFILE", "lite")
    st = load_settings(example_path)
    assert st["retrieval"]["sparse_available"] is False
    weights = st["retrieval"]["rrf_weights"]
    assert "sparse" not in weights
    assert abs(sum(weights.values()) - 3.6) < 1e-2


def test_env_placeholder_expansion(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_UNITTEST", "gpt-4o")
    assert _expand_env_in_value("${LLM_MODEL_UNITTEST:-gpt-4o-mini}") == "gpt-4o"
    # 未设置环境变量 → 取默认值
    assert _expand_env_in_value("${NOT_SET_VAR_XYZ:-fallback}") == "fallback"
    # 空默认值 → 空串
    assert _expand_env_in_value("${NOT_SET_VAR_XYZ:-}") == ""
    # 递归处理 dict / list
    nested = _expand_env_in_value({"a": ["${NOT_SET_VAR_XYZ:-z}"], "b": {"c": 1}})
    assert nested == {"a": ["z"], "b": {"c": 1}}


def test_detect_hardware_shape():
    hw = detect_hardware()
    assert isinstance(hw, dict)


@pytest.mark.parametrize("args", (["--help"], ["-q"]))
def test_autodetect_cli_runs(repo_root: Path, args):
    # 子进程跑入口脚本（--help 会让 argparse SystemExit，与 batch1 原意一致）
    proc = subprocess.run(
        [sys.executable, str(repo_root / "config" / "autodetect.py"), *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
