# -*- coding: utf-8 -*-
"""doc_id 三策略离线自检（由 _check_batch3.py test_doc_id_strategy 迁移）。

BYTES / PATH_MTIME / CONTENT 策略、content_fingerprint 跨导出稳定、
image_fingerprint(pHash/clip)、by-path 缓存、CONTENT/PATH_MTIME 流水线集成，
全程只跑 parse→chunk（离线，不碰模型）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ingestion.pipeline import (
    DEFAULT_ID_STRATEGY,
    DocIdStrategy,
    _normalize_block_text,
    compute_doc_id,
    content_fingerprint,
    image_fingerprint,
    normalize_strategy,
    peek_doc_id,
    read_cached_content_id,
    run,
    write_cached_content_id,
)


# ============================================================ normalize_strategy
def test_normalize_strategy():
    assert normalize_strategy("bytes") is DocIdStrategy.BYTES
    assert normalize_strategy("path-mtime") is DocIdStrategy.PATH_MTIME  # 兼容连字符
    assert normalize_strategy(None) is DEFAULT_ID_STRATEGY
    with pytest.raises(ValueError, match="bytes"):
        normalize_strategy("xxx_bad")


# ============================================================ BYTES
def test_bytes_strategy(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world 123")
    id1 = compute_doc_id(f, "bytes")
    id2 = compute_doc_id(f, DocIdStrategy.BYTES)
    assert id1 == id2
    assert id1 == hashlib.sha256(f.read_bytes()).hexdigest()
    f2 = tmp_path / "test2.txt"
    f2.write_bytes(b"hello world 124")
    assert compute_doc_id(f2, "bytes") != id1  # 不同字节 → 不同 id


# ============================================================ PATH_MTIME
def test_path_mtime_strategy(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"content")
    pid1 = compute_doc_id(f, "path_mtime")
    pid2 = compute_doc_id(f, DocIdStrategy.PATH_MTIME)
    assert pid1 == pid2
    f3 = tmp_path / "renamed.txt"
    f.rename(f3)
    pid3 = compute_doc_id(f3, "path_mtime")
    assert pid3 != pid1  # 改名后 id 变（路径是身份）


# ============================================================ _normalize_block_text
def test_normalize_block_text():
    assert _normalize_block_text("  hello  \r\nworld  \r\n") == "hello\nworld"
    assert _normalize_block_text(None) == ""
    assert _normalize_block_text("42") == "42"  # 纯数字不变形


# ============================================================ CONTENT 指纹
_PARSED_A = {
    "source_format": "docx",
    "blocks": [
        {"type": "text", "content": "第一章 概述", "page": 1, "bbox": [0, 0, 100, 50]},
        {"type": "text", "content": "这是正文内容。  \r\n  第二行", "page": 2, "bbox": [10, 20]},
    ],
}
# 同一内容、不同 bbox/font/page → CONTENT 指纹必须等（跨导出稳定）
_PARSED_B = {
    "source_format": "docx",
    "blocks": [
        {
            "type": "text",
            "content": "第一章 概述",
            "page": 99,
            "bbox": [999, 999, 999, 999],
            "font": "宋体",
        },
        {"type": "text", "content": "这是正文内容。\n  第二行", "page": 55},
    ],
}


def test_content_fingerprint_stable():
    fp_a, _ = content_fingerprint(_PARSED_A)
    fp_b, _ = content_fingerprint(_PARSED_B)
    assert fp_a == fp_b  # bbox/page/font 变化不影响指纹
    parsed_c = {"source_format": "docx", "blocks": [{"type": "text", "content": "完全不同内容"}]}
    fp_c, _ = content_fingerprint(parsed_c)
    assert fp_a != fp_c  # 不同正文 → 不同指纹


def test_content_compute_doc_id_requires_parsed(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"xxx")
    with pytest.raises(ValueError, match="必须先 parse"):
        compute_doc_id(f, "content", data=b"xxx")
    cid = compute_doc_id(f, "content", data=b"ignored", parsed=_PARSED_A)
    fp_a, _ = content_fingerprint(_PARSED_A)
    assert cid == fp_a  # 返回指纹等于 content_fingerprint


# ============================================================ image_fingerprint
def test_image_fingerprint():
    # 含 block["doc_id"] 但 image_fingerprint 绝不使用它
    img_block_with_docid = {
        "type": "image",
        "content": "[图片 200x100px]",
        "doc_id": "parent_sha:p3:x12",
        "clip_vector": None,
        "phash": "",
    }
    _, method = image_fingerprint(img_block_with_docid, data=None, source_format="docx")
    assert method != "doc_id"
    img_clip = {
        "type": "image",
        "content": "ignored",
        "clip_vector": [0.5, -0.2, 0.05, -0.7, 0.4, -0.15, 0.8, -0.3],
        "phash": "",
    }
    _, m2 = image_fingerprint(img_clip, data=None, source_format="docx")
    assert m2 == "clip_quantized"
    img_ph = {"type": "image", "content": "x", "phash": "abcdef0123456789"}
    fp_ph, m_ph = image_fingerprint(img_ph)
    assert fp_ph == "abcdef0123456789" and m_ph == "phash"  # 显式 phash 字段优先


# ============================================================ by-path 缓存
def test_by_path_cache(tmp_path: Path):
    cache_root = tmp_path / "out"
    cache_root.mkdir()
    test_file = tmp_path / "cached.txt"
    test_file.write_bytes(b"cached content for test")
    content_id = "abc123deadbeef"
    write_cached_content_id(test_file, content_id, cache_root)
    assert read_cached_content_id(test_file, cache_root) == content_id
    assert peek_doc_id(test_file, "content", cache_root) == content_id  # 命中缓存
    new_file = tmp_path / "new_no_cache.txt"
    new_file.write_bytes(b"new file no cache yet")
    assert peek_doc_id(new_file, "content", cache_root) is None  # 无缓存返回 None


# ============================================================ CONTENT / PATH_MTIME 流水线集成
def test_content_strategy_pipeline(tmp_path: Path):
    pipe_in = tmp_path / "pipe_in"
    pipe_in.mkdir()
    # 放一个 txt 文件进去（不用 Office，避免 zip 时间戳干扰）
    (pipe_in / "doc1.txt").write_text("第一章 测试\n\n这是正文。\n第二行内容。", encoding="utf-8")
    pipe_out = tmp_path / "pipe_out"
    pipe_out.mkdir()

    s1 = run(pipe_in, "parse", "chunk", id_strategy="content", workers=1, root=pipe_out)
    assert s1["error"] == 0
    assert s1["id_strategy"] == "content"
    parsed_files = list((pipe_out / "parsed").glob("*.json"))
    assert len(parsed_files) >= 1
    content_parsed = json.loads(parsed_files[0].read_text(encoding="utf-8"))
    bytes_id = hashlib.sha256((pipe_in / "doc1.txt").read_bytes()).hexdigest()
    assert content_parsed["doc_id"] != bytes_id  # doc_id 是 content 指纹不是 bytes 哈希
    assert content_parsed.get("doc_id_strategy") == "content"
    assert content_parsed.get("bytes_doc_id") == bytes_id  # 保留原始字节哈希

    # 第二跑 CONTENT：by-path 缓存命中 → 全部 skipped
    s2 = run(pipe_in, "parse", "chunk", id_strategy="content", workers=1, root=pipe_out)
    assert s2["fully_skipped"] >= 1


def test_path_mtime_strategy_pipeline(tmp_path: Path):
    pipe_in = tmp_path / "pipe_in"
    pipe_in.mkdir()
    (pipe_in / "doc1.txt").write_text("第一章 测试\n\n这是正文。\n第二行内容。", encoding="utf-8")
    pipe_out = tmp_path / "pipe_out_pm"
    pipe_out.mkdir()
    s3 = run(pipe_in, "parse", "chunk", id_strategy="path_mtime", workers=1, root=pipe_out)
    assert s3["error"] == 0
    assert s3["id_strategy"] == "path_mtime"
    parsed_pm = list((pipe_out / "parsed").glob("*.json"))
    assert len(parsed_pm) >= 1
    content_pm = json.loads(parsed_pm[0].read_text(encoding="utf-8"))
    assert content_pm.get("doc_id_strategy") == "path_mtime"
