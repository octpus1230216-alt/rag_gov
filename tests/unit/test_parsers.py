# -*- coding: utf-8 -*-
"""解析器离线自检（由 _check_batch2.py 迁移）。

用真实生成的 docx/pptx/xlsx/pdf/txt/png 跑全部 parser，校验统一中间 JSON 的
schema、标题判定、表格 markdown、section_path、编码探测。OCR/CLIP 缺失的
扫描件/图片分支走「降级路径」断言，因此全程离线，无需 paddle。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion.classifiers.doc_classifier import classify
from ingestion.parsers import common
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.image_parser import parse_image
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.pptx_parser import parse_pptx
from ingestion.parsers.txt_parser import parse_txt
from ingestion.parsers.xlsx_parser import parse_xlsx
from tests.helpers import sample_docs

_PDF_BODY = (
    "第一条 合同双方本着平等自愿、诚实信用的原则，经友好协商，就本次采购事项签订本合同。"
    "第二条 乙方应当按照约定的质量标准和交付期限完成供货义务，逾期交付的，每日按合同总价的千分之三支付违约金。"
    "第三条 因不可抗力导致合同无法履行的，双方均不承担违约责任，但应当及时通知对方并提供证明。"
)

REQUIRED_META = {
    "doc_type",
    "topic",
    "has_table",
    "has_image",
    "language",
    "layout_style",
    "source_format",
}
ALLOWED_TYPES = {"text", "title", "table", "image", "slide", "sheet"}


def validate_schema(result: dict[str, Any], label: str) -> None:
    assert {"doc_id", "source_format", "metadata", "blocks"} <= set(result), list(result)[:6]
    assert len(result.get("doc_id", "")) == 64, result.get("doc_id", "")
    meta = result.get("metadata", {})
    assert set(meta) >= REQUIRED_META, sorted(REQUIRED_META - set(meta))
    assert meta.get("doc_type") in {"contract", "report", "manual", "invoice", "unknown"}
    assert meta.get("topic") in {"finance", "legal", "technical", "unknown"}
    assert meta.get("language") in {"zh", "en", "mixed"}
    assert meta.get("layout_style") in {"form", "prose", "slides", "sheet", "image"}
    bad = [b["type"] for b in result.get("blocks", []) if b["type"] not in ALLOWED_TYPES]
    assert not bad, bad[:3]
    empty = [i for i, b in enumerate(result.get("blocks", [])) if not b.get("content", "").strip()]
    assert not empty, empty[:3]


# ============================================================ 标题判定纯函数
def test_title_detection():
    from ingestion.parsers.pdf_parser import is_title, score_title

    body = 11.0
    assert is_title("第一章 总则", 24, "STSong-Light", [150, 60, 445, 90], 595, body)
    assert is_title("故障排除", 14, "SimHei", [60, 100, 200, 120], 595, body)
    # 假加粗（字体名不变，仅靠字号）
    assert is_title("付款条款", 18, "STSong-Light", [60, 100, 200, 125], 595, body)
    assert not is_title(_PDF_BODY[:50] + "。", 11, "STSong-Light", [60, 140, 535, 240], 595, body)
    assert not is_title("本合同项下", 11, "STSong-Light", [60, 140, 180, 155], 595, body)
    assert not is_title("附表说明", 12, "Arial", [60, 300, 200, 315], 595, body)
    # 宋体正文小字（宋体必须在字体黑名单外）
    assert not is_title("第一章 总则说明", 11, "SimSun", [60, 300, 200, 315], 595, body)

    # 三条信号是「或」关系，各自可单独命中——层级由字号档位给出
    v1 = score_title("第一章 总则", 24, "STSong-Light", [150, 60, 445, 90], 595, body)
    assert v1["signals"]["size"] and not v1["signals"]["font"]
    assert v1["level_hint"] == 1
    v2 = score_title("故障排除", 14, "SimHei", [60, 100, 200, 120], 595, body)
    # 仅字体命中（字号 1.27 倍不够 1.3）
    assert v2["signals"]["font"] and not v2["signals"]["size"] and v2["hit"]
    v3 = score_title("附录 A", 11, "Arial", [260, 100, 335, 120], 595, body)
    assert v3["signals"]["center_short"] and not v3["signals"]["size"]
    assert v3["level_hint"] == 3  # 无字号证据给 3 级
    assert v1["score"] > v2["score"]  # 保留证据分供排序


# ============================================================ 分类器
def test_doc_classifier():
    c_contract = classify(
        "合同编号：HT-01 甲方：A 公司 乙方：B 公司 签订日期 2024年1月1日 违约责任 违约金 本合同一式贰式",
        [],
    )
    assert c_contract["doc_type"] == "contract"
    c_invoice = classify(
        "发票代码：044021900111 发票号码：12345678 开票日期：2024年5月8日 价税合计 ￥1,234.00 "
        "税额 142.43 纳税人识别号 91440300MA5XXXXX8K",
        [],
    )
    assert c_invoice["doc_type"] == "invoice"
    assert c_invoice["layout_style"] == "form"
    c_report = classify(
        "2023年度报告：报告期内公司实现营业收入12.3亿元，同比增长18.6%，净利润率12.4%，毛利率提升，董事会审议通过。",
        [],
    )
    assert c_report["doc_type"] == "report"
    assert c_report["topic"] == "finance"
    c_manual = classify(
        "第3章 安装步骤 步骤1：连接电源线。注意事项：请勿潮湿环境使用。故障排除：指示灯不亮检查适配器。"
        "技术参数：额定电压220V。本产品保修一年。",
        [],
    )
    assert c_manual["doc_type"] == "manual"
    assert classify("", [])["doc_type"] == "unknown"


# ============================================================ 表格 markdown / 表头 / 标题栈
def test_table_to_markdown():
    md = common.table_to_markdown([["月", "额"], ["1", "10"], ["2", None], ["", ""]], header_rows=1)
    assert md.startswith("| 月 | 额 |\n|---|---|"), repr(md)
    # 空行被剔除
    assert "2" in md and md.count("|---|") == 1
    long = common.table_to_markdown(
        [["h1", "h2"]] + [[f"r{i}", str(i)] for i in range(5)], header_rows=1, max_rows=3
    )
    assert long.count("| h1 | h2 |") == 3  # 超长分段重复表头
    assert "\\|" in common.table_to_markdown([["a|b", "c"], ["d", "e"]], 1)  # 竖线转义


def test_pick_header_row():
    # 跨列合并的表题横幅不能被当列名行
    h_idx, header = common.pick_header_row(
        [
            ["2024年费用明细", "", "", ""],
            ["月份", "部门", "费用类型", "金额"],
            ["1月", "研发", "差旅", "100"],
        ]
    )
    assert h_idx == 1 and header == ["月份", "部门", "费用类型", "金额"]
    h_idx2, header2 = common.pick_header_row(
        [["科目", "2023年", "2024年"], ["营业收入", "1000", "1180"]]
    )
    assert h_idx2 == 0 and header2 == ["科目", "2023年", "2024年"]


def test_heading_stack():
    st = common.HeadingStack()
    st.seed_root("产品使用手册")
    st.push(1, "产品使用手册")  # 与根同名的正文大标题不该重复入栈
    st.push(1, "第1章 安装")
    st.push(2, "1.1 硬件连接")
    assert st.path == "产品使用手册 > 第1章 安装 > 1.1 硬件连接"
    st.push(1, "第2章 使用")
    assert st.path == "产品使用手册 > 第2章 使用"  # 同级替换


# ============================================================ txt
def test_parse_txt(sample_dir: Path):
    txt_res = parse_txt(sample_docs.make_txt(sample_dir))
    validate_schema(txt_res, "txt")
    assert any(b["type"] == "table" for b in txt_res["blocks"])
    assert any(b["type"] == "title" for b in txt_res["blocks"])
    assert any("采购合同" in b.get("section_path", "") for b in txt_res["blocks"])
    assert txt_res["metadata"]["language"] == "zh"
    assert txt_res["metadata"]["doc_type"] == "contract"


def test_parse_txt_gbk(sample_dir: Path):
    gbk = parse_txt(sample_docs.make_txt_gbk(sample_dir))
    assert gbk.get("encoding", "").startswith("gb"), gbk.get("encoding")
    assert "营收" in json.dumps(gbk["blocks"], ensure_ascii=False)


# ============================================================ docx
def test_parse_docx(sample_dir: Path):
    dx = parse_docx(sample_docs.make_docx(sample_dir))
    validate_schema(dx, "docx")
    types = [b["type"] for b in dx["blocks"]]
    assert "title" in types and "table" in types and "image" in types
    tbl = next(b for b in dx["blocks"] if b["type"] == "table")
    assert tbl["content"].count("2024年") == 1  # 合并单元格不重复表头列
    assert tbl.get("header") == ["科目", "2023年", "2024年"]  # 合并表头拆回各列
    assert any("产品使用手册 > 第1章 安装" in b.get("section_path", "") for b in dx["blocks"])
    fake = next((b for b in dx["blocks"] if b["content"] == "故障排除"), None)
    assert fake is not None and fake["type"] == "title"  # 无样式加粗大字被识别为标题
    assert dx.get("body_font_size") == 11.0
    img_blk = next(b for b in dx["blocks"] if b["type"] == "image")
    assert "图1 营收趋势" in img_blk["content"]  # 内嵌图取到下一段图注
    assert bool(img_blk["content"].strip())  # OCR 缺失时用占位而非空串
    assert dx.get("n_images") == 1
    assert dx["metadata"]["language"] == "zh"


# ============================================================ pptx
def test_parse_pptx(sample_dir: Path):
    px = parse_pptx(sample_docs.make_pptx(sample_dir))
    validate_schema(px, "pptx")
    slides = [b for b in px["blocks"] if b["type"] == "slide"]
    assert len(slides) == 2  # 每页一个 slide block
    assert "2024年营收分析" in slides[0]["content"] and "同比增长 18%" in slides[0]["content"]
    assert "备注：" in slides[0]["content"]
    assert "-15% 的汇兑损失需关注" in slides[0]["content"]  # 负号不被当项目符号剥掉
    assert any(b["type"] == "table" for b in px["blocks"])
    assert any(b["type"] == "image" for b in px["blocks"])
    assert slides[1]["section_path"].startswith("第2页")
    assert px["metadata"]["layout_style"] == "slides"


# ============================================================ xlsx
def test_parse_xlsx(sample_dir: Path):
    xl = parse_xlsx(sample_docs.make_xlsx(sample_dir))
    validate_schema(xl, "xlsx")
    sheet_blocks = [b for b in xl["blocks"] if b["type"] == "sheet"]
    assert len(sheet_blocks) >= 2  # 每区域一块
    assert all(b["n_rows"] <= 100 for b in sheet_blocks)  # 每 100 行一切
    assert all("|| " not in b["content"] for b in sheet_blocks)  # 空行剔除
    assert all("备注列整列为空" not in b["content"] for b in sheet_blocks)  # 噪声列剔除
    assert sheet_blocks[0]["columns"] == ["月份", "部门", "费用类型", "金额"]
    assert sheet_blocks[0]["table_title"] == "2024年费用明细"
    assert all(b["sheet_name"] != "隐藏表" for b in sheet_blocks)  # 隐藏 sheet 不入库
    assert all("2024年费用明细" in b["content"] for b in sheet_blocks)  # 合并大标题回填
    assert sheet_blocks[0]["section_path"].startswith("月度明细 > 第")
    assert xl["metadata"]["has_table"] and xl["metadata"]["layout_style"] == "sheet"


# ============================================================ pdf
def test_parse_pdf(sample_dir: Path):
    pdf_path = sample_docs.make_pdf(sample_dir)
    pr = parse_pdf(pdf_path)
    validate_schema(pr, "pdf")
    types = [b["type"] for b in pr["blocks"]]
    assert {"title", "text", "table"} <= set(types)
    title_blk = next((b for b in pr["blocks"] if b["type"] == "title"), None)
    assert title_blk is not None and "第一章" in title_blk["content"]
    assert all(k in pr["blocks"][1] for k in ("font", "size", "bbox"))
    assert pr.get("body_font_size") in (10.5, 11.0)
    tblk = next((b for b in pr["blocks"] if b["type"] == "table"), None)
    assert tblk is not None and tblk["content"].startswith("| 科目 |")
    assert tblk and tblk["bbox"][0] < tblk["bbox"][2]
    inside = [
        b
        for b in pr["blocks"]
        if b["type"] == "text" and b.get("page") == 2 and "营业收入" in b["content"]
    ]
    assert not inside  # 表格区域内的文本块不重复入库
    assert pr.get("scanned_pages") == [3]  # 扫描件分支命中第 3 页
    assert bool(pr.get("model_errors"))  # 模型缺失走降级并记录原因
    p2_paths = [b.get("section_path", "") for b in pr["blocks"] if b.get("page") == 2]
    assert all("第一章 总则" in p for p in p2_paths if p)  # 章节跨页延续（单一全局标题栈）
    assert all(p.startswith("sample") for p in p2_paths if p)  # section_path 带文档根
    assert len(pr.get("outline", [])) == 2
    assert pr["metadata"]["doc_type"] in ("contract", "report")


def test_pdf_page_helpers(sample_dir: Path):
    from ingestion.parsers.pdf_parser import (
        collect_page_blocks,
        estimate_body_size,
        is_scanned_page,
    )

    fitz = sample_docs._fitz()
    doc = fitz.open(str(sample_docs.make_pdf(sample_dir)))
    stats = collect_page_blocks(doc[0], 1)
    bs = estimate_body_size(stats)
    assert 10.5 <= bs <= 11.5
    assert not is_scanned_page(doc[0])
    assert is_scanned_page(doc[2])
    doc.close()


# ============================================================ image
def test_parse_image(sample_dir: Path):
    ip = sample_dir / "scanned.png"
    sample_docs.make_png(ip)
    im_res = parse_image(ip)
    validate_schema(im_res, "image")
    assert im_res["metadata"]["layout_style"] == "image"
    assert im_res["metadata"]["has_image"]
    blk = im_res["blocks"][0]
    assert blk.get("clip_vector") in (None, []) or isinstance(blk.get("clip_vector"), list)
    # 向量缺失时不写 clip_vector 字段（不伪造）
    assert "clip_vector" not in blk or blk["clip_vector"] is None
    assert blk["classification"]["label"] == "unknown"
    assert ip.stem in blk["content"]  # 独立入口 caption 默认取文件名


def test_image_classifier(sample_dir: Path):
    from ingestion.classifiers.image_classifier import (
        IMAGE_CATEGORIES,
        MIN_CLASS_CONFIDENCE,
        classify_image_bytes,
        is_vector_like,
    )

    assert set(IMAGE_CATEGORIES) == {"chart", "photo", "diagram", "signature", "form"}
    assert MIN_CLASS_CONFIDENCE == 0.4
    ip = sample_dir / "scanned.png"
    sample_docs.make_png(ip)
    cls = classify_image_bytes(ip.read_bytes())
    # 模型缺失走 unknown 且带 error
    assert cls["label"] == "unknown" and bool(cls.get("error"))
    assert not is_vector_like({"label": "signature", "confidence": 0.91})
    assert is_vector_like({"label": "chart", "confidence": 0.82})
