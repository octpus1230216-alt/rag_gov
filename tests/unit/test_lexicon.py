# -*- coding: utf-8 -*-
"""词表离线自检（由 _check_batch1.py 迁移）：synonyms 组数/值类型、stopwords 去重量与必备虚词。"""

from __future__ import annotations

import collections
from pathlib import Path

import yaml

# 停用词里必须出现的常见虚词（缺失说明词表被误删）
_MUST_HAVE = (
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
    "或",
    "有",
    "被",
    "把",
    "让",
    "使",
    "对",
    "为",
    "以",
    "从",
    "到",
    "由",
    "等",
    "这",
    "那",
    "一个",
    "我们",
    "他们",
    "可以",
    "如果",
    "因为",
    "所以",
)


def test_synonyms(repo_root: Path):
    syn = yaml.safe_load((repo_root / "config" / "synonyms.yaml").read_text(encoding="utf-8"))
    groups = {k: v for k, v in syn.items() if isinstance(v, list)}
    assert len(groups) >= 50, f"同义词组数不足 50：{len(groups)}"
    bad = {k: v for k, v in groups.items() if not all(isinstance(x, str) and x for x in v)}
    assert not bad, f"同义词值非字符串：{list(bad)[:3]}"


def test_stopwords(repo_root: Path):
    raw = (repo_root / "config" / "stopwords_zh.txt").read_text(encoding="utf-8").splitlines()
    words = [w.strip() for w in raw if w.strip() and not w.strip().startswith("#")]
    uniq = set(words)
    assert len(uniq) >= 200, f"停用词去重后不足 200：{len(uniq)}"
    # 词表本身不应有重复行
    dups = [w for w, c in collections.Counter(words).items() if c > 1]
    assert not dups, f"停用词存在重复：{dups[:5]}"
    missing = [w for w in _MUST_HAVE if w not in uniq]
    assert not missing, f"停用词缺失必备项：{missing}"
