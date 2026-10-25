# -*- coding: utf-8 -*-
""".gitignore 判定自检（由 _check_batch1.py 第 7 段迁移并修复其 bug）。

原脚本用 `git check-ignore -q --stdin`，`-q` 抑制了 stdout，导致解析恒为空、
所有判定误报 FAIL。这里改用「逐路径 + 退出码」判定：
  exit 0 → 被忽略；exit 1 → 未被忽略；exit 128 → 非 git 仓库（跳过）。
只让样例文档进仓库，真实数据/索引/模型/密钥必须被忽略。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# path -> 期望「是否被忽略」
PROBES = {
    "samples/demo.pdf": False,  # 样例文档必须可提交
    "data/secret.pdf": True,
    "storage/qdrant/x": True,
    "config/settings.yaml": True,
    "config/settings.example.yaml": False,
    "config/stopwords_zh.txt": False,
    "models/bge-m3.bin": True,
    "chunks/part-0.parquet": True,
}


def _is_ignored(repo_root: Path, path: str) -> bool | None:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 128:
        return None  # 非 git 仓库
    return proc.returncode == 0


@pytest.mark.parametrize("path,should_ignore", sorted(PROBES.items()))
def test_gitignore_verdicts(repo_root: Path, path: str, should_ignore: bool):
    actual = _is_ignored(repo_root, path)
    if actual is None:
        pytest.skip("未初始化 git 仓库，跳过 .gitignore 验证")
    assert actual is should_ignore, f"{path} 期望忽略={should_ignore} 实际忽略={actual}"
