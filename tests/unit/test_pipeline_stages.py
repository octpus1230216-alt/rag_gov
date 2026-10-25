# -*- coding: utf-8 -*-
"""流水线离线自检（由 _check_batch3.py 迁移，只到 chunk 阶段）。

统一 chunk schema、无内容丢失、幂等跳过、断点续跑复用 parsed、坏文件定位到
具体路径 + traceback、串行与多进程结果一致、CLI 返回码。全程 parse→chunk，
embed/index 依赖重模型不在本批离线跑（只校验阶段注册与区间）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ingestion.pipeline import (
    IMPLEMENTED,
    STAGES,
    _iter_progress,
    artifact_path,
    collect_files,
    compute_doc_id,
    iter_chunks,
    main,
    run,
    stage_range,
)
from tests.helpers import sample_docs


@pytest.fixture
def ingest_env(tmp_path: Path) -> Path:
    """只生成输入文件（含坏 PDF），不跑流水线。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    sample_docs.build_ingest_dir(in_dir)
    return in_dir


@pytest.fixture
def ingested(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """生成输入文件并跑一遍 parse→chunk，返回 (in_dir, out_root, summary)。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    sample_docs.build_ingest_dir(in_dir)
    out = tmp_path / "out"
    out.mkdir()
    summary = run(in_dir, "parse", "chunk", workers=2, root=out, force=True)
    return in_dir, out, summary


# ============================================================ 统一 chunk schema
def test_chunk_schema(ingested):
    _in_dir, out, summary = ingested
    assert summary["error"] == 1  # 除刻意造的坏文件外无失败

    required = {"chunk_id", "doc_id", "content", "chunk_type", "section_path", "metadata"}
    allowed_types = {"text", "table", "image", "slide", "sheet"}
    seen_types: set[str] = set()
    problems: list[str] = []
    meta_keys = {"doc_type", "topic", "language", "has_table", "has_image", "layout_style"}
    for path in sorted((out / "chunks").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if not required.issubset(c):
                problems.append(f"{path.name} 缺字段 {required - set(c)}")
            if c["chunk_type"] not in allowed_types:
                problems.append(f"{path.name} 非法 chunk_type {c['chunk_type']}")
            if not isinstance(c["section_path"], str) or not isinstance(c["metadata"], dict):
                problems.append(f"{path.name} section_path/metadata 类型错")
            if not meta_keys.issubset(c["metadata"]):
                problems.append(f"{path.name} metadata 未继承 {meta_keys - set(c['metadata'])}")
            if "bbox" in c and len(c["bbox"]) != 4:
                problems.append(f"{path.name} bbox 长度 {len(c['bbox'])}")
            if not c["content"].strip():
                problems.append(f"{path.name} 空内容 chunk")
            seen_types.add(c["chunk_type"])
    assert not problems, str(problems[:3])
    assert "text" in seen_types
    assert "table" in seen_types
    assert "slide" in seen_types
    assert len(list((out / "parsed").glob("*.json"))) >= 5
    assert len(list((out / "chunks").glob("*.jsonl"))) >= 5
    assert not (out / "vectors").exists()  # 未实现阶段不伪造 vectors/


# ============================================================ 真实文档切分质量
def test_real_docs_quality(ingested):
    in_dir, out, _summary = ingested

    def chunks_of(path: Path) -> list[dict[str, Any]]:
        p = out / "chunks" / f"{compute_doc_id(path)}.jsonl"
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    pdf_chunks = [c for c in chunks_of(in_dir / "sample.pdf") if c["chunk_type"] == "text"]
    multi = [c for c in pdf_chunks if len(c.get("pages") or []) > 1]
    assert multi or len(pdf_chunks) == 1  # 真实 PDF：跨页同 section 合并成片

    txt_chunks = chunks_of(in_dir / "sample.txt")
    tables = [c for c in txt_chunks if c["chunk_type"] == "table"]
    assert tables  # txt 表格走表格切分器
    assert all("|---" not in c["content"] for c in txt_chunks if c["chunk_type"] == "text")

    xlsx_chunks = chunks_of(in_dir / "sample.xlsx")
    assert any(c["chunk_type"] == "sheet" for c in xlsx_chunks)

    pptx_chunks = [c for c in chunks_of(in_dir / "sample.pptx") if c["chunk_type"] == "slide"]
    assert len(pptx_chunks) == 2  # pptx 一页一 chunk

    docx_chunks = chunks_of(in_dir / "sample.docx")
    small = [c for c in docx_chunks if c["chunk_type"] == "table"]
    assert len(small) == 1 and small[0].get("n_parts") in (None,)  # 小表整表成块不切碎


# ============================================================ 幂等 / 断点续跑 / 坏文件
def test_pipeline_idempotent(ingested):
    in_dir, out, _summary = ingested
    docx_path = in_dir / "sample.docx"
    first_ids = [c["chunk_id"] for c in iter_chunks(compute_doc_id(docx_path), out)]
    assert len(first_ids) > 0

    second = run(in_dir, "parse", "chunk", workers=2, root=out)
    assert second["error"] == 1  # 除坏文件外零失败
    ok_reports = [r for r in second["reports"] if r["status"] == "done"]
    # 所有阶段 skipped（幂等生效）
    assert all(all(v == "skipped" for v in r["stages"].values()) for r in ok_reports)
    assert second["fully_skipped"] == len(ok_reports)
    disk_chunks = sum(
        1
        for f in (out / "chunks").glob("*.jsonl")
        for ln in f.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    )
    assert second["n_chunks"] == disk_chunks  # 幂等不重复追加
    assert len(list((out / "chunks").glob("*.jsonl"))) == 5
    again_ids = [c["chunk_id"] for c in iter_chunks(compute_doc_id(docx_path), out)]
    assert again_ids == first_ids

    # 断点续跑：从 chunk 阶段起跑，复用 parsed，不重新解析
    txt_id = compute_doc_id(in_dir / "sample.txt")
    txt_ids_before = [c["chunk_id"] for c in iter_chunks(txt_id, out)]
    parsed_before = artifact_path("parse", txt_id, out).stat().st_mtime_ns
    res = run(in_dir, "chunk", "chunk", workers=1, root=out, force=True)
    parsed_after = artifact_path("parse", txt_id, out).stat().st_mtime_ns
    assert parsed_before == parsed_after  # 复用 parsed 产物（时间戳未变）
    assert [c["chunk_id"] for c in iter_chunks(txt_id, out)] == txt_ids_before  # chunk_id 稳定
    assert res["total"] >= 5 and res["error"] == 1

    # 阶段注册与区间（embed/index 依赖重模型，离线不实际跑）
    assert all(IMPLEMENTED[s] for s in STAGES)
    assert stage_range("parse", "index") == list(STAGES)
    with pytest.raises(ValueError, match="阶段区间非法"):
        run(in_dir / "sample.txt", "chunk", "parse", workers=1, root=out)

    # 缺上游产物：新文件从 chunk 起跑必须点名怎么补
    fresh = in_dir / "late.txt"
    fresh.write_text("# 新文档\n" + sample_docs.LONG_SENT + "\n", encoding="utf-8")
    rep = run(fresh, "chunk", "chunk", workers=1, root=out)
    err = rep["reports"][0]
    assert "请先跑 --from-stage parse" in (err["error"] or "")
    assert err["file"] == str(fresh) and err["failed_stage"] == "parse"  # 定位到具体文件路径

    # 坏文件：定位 + 错误日志 + traceback
    broken = in_dir / "broken.pdf"
    rep2 = run(broken, "parse", "chunk", workers=1, root=out)
    err2 = rep2["reports"][0]
    assert rep2["error"] == 1 and rep2["total"] == 1  # 单文件失败不炸整批
    assert err2["failed_stage"] == "parse"
    assert bool(err2.get("traceback")) and err2["file"].endswith("broken.pdf")
    err_log = out / "logs" / "pipeline_errors.jsonl"
    assert err_log.exists() and broken.name in err_log.read_text(encoding="utf-8")


def test_serial_vs_multiprocess(ingested):
    in_dir, out, _summary = ingested
    docx_path = in_dir / "sample.docx"
    docx_id = compute_doc_id(docx_path)
    serial = run(docx_path, "parse", "chunk", workers=1, root=out / "serial", force=True)
    par = run(docx_path, "parse", "chunk", workers=3, root=out / "par", force=True)
    a = [
        json.loads(ln)["chunk_id"]
        for ln in (out / "serial" / "chunks" / f"{docx_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    b = [
        json.loads(ln)["chunk_id"]
        for ln in (out / "par" / "chunks" / f"{docx_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert a == b and len(a) > 0  # 串行与多进程 chunk_id 一致
    assert serial["n_chunks"] == par["n_chunks"]


def test_cli_return_codes(ingested):
    in_dir, out, _summary = ingested
    code = main(
        [
            "--input",
            str(in_dir / "sample.docx"),
            "--from",
            "parse",
            "--to",
            "chunk",
            "--out-root",
            str(out / "cli"),
            "--workers",
            "1",
        ]
    )
    assert code == 0  # --from/--to 可跑通
    code2 = main(
        [
            "--input",
            str(in_dir / "sample.docx"),
            "--from-stage",
            "parse",
            "--to-stage",
            "chunk",
            "--out-root",
            str(out / "cli2"),
            "--token-backend",
            "char",
        ]
    )
    assert code2 == 0  # 别名可用
    code3 = main(
        ["--input", str(in_dir / "broken.pdf"), "--from", "parse", "--out-root", str(out / "cli3")]
    )
    assert code3 == 1  # 有失败文件返回码 1
    code4 = main(["--input", str(in_dir / "不存在.txt"), "--out-root", str(out / "cli4")])
    assert code4 == 2  # 路径不存在返回码 2


# ============================================================ 扫描跳过产物 / 进度降级
def test_scan_skips_artifacts(ingest_env: Path):
    in_dir = ingest_env
    files = collect_files(in_dir, recursive=True)
    assert all(f.suffix.lower() in (".txt", ".docx", ".pptx", ".xlsx", ".pdf") for f in files)
    (in_dir / "parsed").mkdir(exist_ok=True)
    (in_dir / "parsed" / "x.json").write_text("{}", encoding="utf-8")
    (in_dir / "nested").mkdir(exist_ok=True)
    (in_dir / "nested" / "chunks").mkdir(exist_ok=True)
    (in_dir / "nested" / "chunks" / "y.jsonl").write_text("", encoding="utf-8")
    (in_dir / "nested" / "real.txt").write_text("# 真文件\n内容\n", encoding="utf-8")
    again = collect_files(in_dir, recursive=True)
    # 产物目录被跳过（含嵌套一层）
    assert not any("chunks" in f.parts or "parsed" in f.parts for f in again)
    assert any(f.name == "real.txt" for f in again)  # 嵌套目录里的真文件仍被收集


def test_progress_fallback():
    got = list(_iter_progress(iter(range(7)), 7, "测试"))
    assert got == list(range(7))  # 无 tqdm 时进度条降级不丢数据
