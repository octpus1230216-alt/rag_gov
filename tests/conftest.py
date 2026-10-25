# -*- coding: utf-8 -*-
"""
pytest 共享夹具。

设计目标：CI 只跑不依赖重模型与外部服务的纯离线单测。默认夹具都基于
``tmp_path``（用例结束自动清理），守卫夹具在缺依赖时 ``skip`` 而非 fail。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 保证从任意工作目录运行 pytest 都能 import 仓库根（配合 pyproject pythonpath=["."]）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """样例文档工厂的落地目录（解析器用例用）。"""
    d = tmp_path / "samples_in"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_out_root(tmp_path: Path) -> Path:
    """流水线产物根（parsed/chunks/logs 落在其下，用例结束自动清理）。"""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def ingest_dir(tmp_path: Path) -> Path:
    """流水线输入目录（batch3 全套样例文件在此生成）。"""
    d = tmp_path / "in"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------ 守卫夹具：缺依赖即 skip
@pytest.fixture
def needs_paddle():
    """需要 PaddleOCR 的用例（扫描件/图片 OCR 分支）缺依赖时跳过。"""
    pytest.importorskip("paddleocr", reason="未安装 paddleocr（重模型 extras），跳过 OCR 相关用例")


@pytest.fixture
def needs_models():
    """需要向量化/CLIP 模型（FlagEmbedding/torch）的用例缺依赖时跳过。"""
    pytest.importorskip(
        "torch", reason="未安装 torch/FlagEmbedding（重模型 extras），跳过向量化用例"
    )
