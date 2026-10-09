# -*- coding: utf-8 -*-
"""
第 2 批自检：用真实生成的 docx/pptx/xlsx/pdf/txt 跑全部 parser，
校验统一中间 JSON 的 schema、标题判定、扫描件分流、表格 markdown、section_path。

运行：python _check_batch2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TMP = ROOT / "_tmp_batch2"
TMP.mkdir(exist_ok=True)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"[OK]   {name}")
    else:
        FAILS.append(f"{name} :: {detail}")
        print(f"[FAIL] {name} :: {detail}")


def make_png(path: Path, size: int = 300, text: str = "财务报表 2024") -> None:
    """画一张带中文的图，用于内嵌图片与独立图片入口。"""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size // 2), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([4, 4, size - 5, size // 2 - 5], outline="black", width=2)
    d.line([20, 90, 80, 40, 140, 70, 240, 20], fill="red", width=3)
    im.save(path, "PNG")


# ============================================================ 造样例文件
def make_txt() -> Path:
    p = TMP / "sample.txt"
    content = (
        "# 采购合同\n"
        "甲方：星辰科技有限公司\n"
        "乙方：海纳供应链有限公司\n\n"
        "## 第一条 标的物\n"
        "本合同项下采购内容如下表所示，交付地点为苏州工业园区。\n\n"
        "| 序号 | 名称 | 单价 | 数量 |\n"
        "|---|---|---|---|\n"
        "| 1 | 服务器 | 12000 | 3 |\n"
        "| 2 | 交换机 | 3500 | 10 |\n\n"
        "## 第二条 违约责任\n"
        "违约方应向守约方支付合同总价百分之十的违约金，并赔偿由此造成的全部损失。\n"
    )
    p.write_text(content, encoding="utf-8")
    return p


def make_txt_gbk() -> Path:
    p = TMP / "sample_gbk.txt"
    p.write_bytes("2024年度财务分析报告\n营收 1.2 亿元，同比增长 18%，成本下降 3%。\n".encode("gbk"))
    return p


def make_docx() -> Path:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    p = TMP / "sample.docx"
    d = docx.Document()
    # 文档属性标题 = section_path 的根（parser 会用它播种 level 0）
    d.core_properties.title = "产品使用手册"
    d.add_heading("产品使用手册", level=1)
    d.add_heading("第1章 安装", level=1)
    d.add_heading("1.1 硬件连接", level=2)
    d.add_paragraph("请将电源线插入插座，注意本产品不得在潮湿环境使用。")
    # 无内置样式的「假标题」：加粗 + 大字号 + 正文样式
    bold_para = d.add_paragraph("故障排除")
    run = bold_para.runs[0]
    run.font.bold = True
    run.font.size = Pt(18)
    bold_para.style = d.styles["Normal"]
    d.add_paragraph("指示灯不亮时请检查适配器是否插好。")

    table = d.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    table.cell(0, 0).text = "科目"
    table.cell(0, 1).text = "2023年"
    table.cell(0, 2).text = "2024年"
    table.cell(1, 0).text = "营业收入"
    table.cell(1, 1).text = "1000"
    table.cell(1, 2).text = "1180"
    table.cell(2, 0).text = "净利润"
    table.cell(2, 1).text = "120"
    table.cell(2, 2).text = "145"
    # 合并单元格：把第 1 行整行合并成标题
    table.cell(0, 0).merge(table.cell(0, 2))

    img = TMP / "inline.png"
    make_png(img)
    d.add_picture(str(img), width=Pt(120))
    d.add_paragraph("图1 营收趋势")
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.save(str(p))
    return p


def make_pptx() -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    layout = prs.slide_layouts[5]  # Title Only
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = "2024年营收分析"

    tb = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(2))
    tf = tb.text_frame
    tf.text = "• 营收同比增长 18%"
    tf.add_paragraph().text = "• 毛利率提升 2.1 个百分点"
    tf.add_paragraph().text = "-15% 的汇兑损失需关注"

    tbl = slide.shapes.add_table(3, 3, Inches(1), Inches(4), Inches(6), Inches(1.5)).table
    for i, row in enumerate([["季度", "营收", "同比"], ["Q1", "300", "+12%"], ["Q2", "320", "+22%"]]):
        for j, val in enumerate(row):
            tbl.cell(i, j).text = val
    slide.notes_slide.notes_text_frame.text = "本页强调二季度增速回升，成本端压力仍在。"

    img = TMP / "slide_img.png"
    make_png(img)
    s2 = prs.slides.add_slide(layout)
    s2.shapes.title.text = "示意图页"
    s2.shapes.add_picture(str(img), Inches(1), Inches(2), height=Pt(120))
    cap = s2.shapes.add_textbox(Inches(1), Inches(3.6), Inches(4), Inches(0.4))
    cap.text_frame.text = "图2 组织架构"
    path = TMP / "sample.pptx"
    prs.save(str(path))
    return path


def make_xlsx() -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "月度明细"
    ws["A1"] = "2024年费用明细"
    ws.merge_cells("A1:D1")  # 合并大标题，测试回填
    ws.append(["月份", "部门", "费用类型", "金额"])
    for m in range(1, 13):
        for dept in ("研发", "销售", "行政"):
            # 3 个月 × 3 部门 × 3 费用类型 = 108 数据行，跨过 100 行分区阈值
            for kind in ("差旅费", "培训费", "招待费"):
                ws.append([f"2024-{m:02d}", dept, kind, 1000 + m * 7])
    ws.append([None, None, None, None])       # 空行，应被剔除
    ws.append(["", "", "", ""])
    ws["F1"] = "备注列整列为空"               # 只有 1 行有值 -> 噪声列，应被剔除
    other = wb.create_sheet("隐藏表")
    other["A1"] = "不该出现"
    other.sheet_state = "hidden"
    path = TMP / "sample.xlsx"
    wb.save(str(path))
    return path


def make_pdf() -> Path:
    """
    造一个 3 页 PDF：
      p1 文本层 + 大字号居中标题（测标题判定）
      p2 带框线表格（测 pdfplumber 抽取 + 文本块去重）
      p3 只有一张图无文本层（测扫描件分流 + OCR 分支的降级路径）
    """
    import fitz

    path = TMP / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # 标题：黑体大字号居中
    page.insert_textbox(
        fitz.Rect(150, 60, 445, 100),
        "第一章 总则",
        fontname="china-s",
        fontsize=24,
        align=fitz.TEXT_ALIGN_CENTER,
    )
    page.insert_textbox(fitz.Rect(60, 140, 535, 300), _PDF_BODY, fontname="china-s", fontsize=11)

    # 表格页：先画框线再写字，lines 策略依赖真实线条
    tp = doc.new_page(width=595, height=842)
    tp.insert_textbox(fitz.Rect(60, 40, 535, 70), "附表：年度营收", fontname="china-s", fontsize=16)
    xs = [60, 200, 340, 480]
    ys = [120, 150, 180, 210]
    for x in xs:
        tp.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]), width=0.8)
    for y in ys:
        tp.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y), width=0.8)
    rows = [["科目", "2023", "2024"], ["营业收入", "1000", "1180"], ["净利润", "120", "145"]]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            tp.insert_text(
                fitz.Point(xs[c] + 6, ys[r] + 20), val, fontname="china-s", fontsize=10
            )

    # 扫描页：只贴一张图，不放文本层
    sp = doc.new_page(width=595, height=842)
    img = TMP / "scanned.png"
    make_png(img)
    sp.insert_image(fitz.Rect(50, 50, 545, 700), filename=str(img))
    doc.set_toc([[1, "第一章 总则", 1], [1, "附表：年度营收", 2]])
    doc.save(str(path))
    doc.close()
    return path


_PDF_BODY = (
    "第一条 合同双方本着平等自愿、诚实信用的原则，经友好协商，就本次采购事项签订本合同。"
    "第二条 乙方应当按照约定的质量标准和交付期限完成供货义务，逾期交付的，每日按合同总价的千分之三支付违约金。"
    "第三条 因不可抗力导致合同无法履行的，双方均不承担违约责任，但应当及时通知对方并提供证明。"
)


# ============================================================ 统一 schema 校验
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


def validate_schema(result: dict, label: str) -> None:
    check(f"{label}: 顶层键", {"doc_id", "source_format", "metadata", "blocks"} <= set(result), str(list(result)[:6]))
    check(f"{label}: doc_id 是 sha256", len(result.get("doc_id", "")) == 64, result.get("doc_id", ""))
    meta = result.get("metadata", {})
    check(f"{label}: metadata 键齐全", REQUIRED_META <= set(meta), str(sorted(REQUIRED_META - set(meta))))
    check(
        f"{label}: metadata 取值合法",
        meta.get("doc_type") in {"contract", "report", "manual", "invoice", "unknown"}
        and meta.get("topic") in {"finance", "legal", "technical", "unknown"}
        and meta.get("language") in {"zh", "en", "mixed"}
        and meta.get("layout_style") in {"form", "prose", "slides", "sheet", "image"},
        str(meta),
    )
    bad = [b["type"] for b in result.get("blocks", []) if b["type"] not in ALLOWED_TYPES]
    check(f"{label}: block 类型合法", not bad, str(bad[:3]))
    empty = [i for i, b in enumerate(result.get("blocks", [])) if not b.get("content", "").strip()]
    check(f"{label}: block 无空内容", not empty, str(empty[:3]))


# ============================================================ 逐项跑
def main() -> int:
    from ingestion.classifiers.doc_classifier import classify
    from ingestion.parsers import common
    from ingestion.parsers.docx_parser import parse_docx
    from ingestion.parsers.image_parser import parse_image
    from ingestion.parsers.pdf_parser import (
        estimate_body_size,
        collect_page_blocks,
        is_scanned_page,
        is_title,
        parse_pdf,
        score_title,
    )
    from ingestion.parsers.pptx_parser import parse_pptx
    from ingestion.parsers.txt_parser import parse_txt
    from ingestion.parsers.xlsx_parser import parse_xlsx

    # ---------- 标题判定纯函数（本批要求单独可测）----------
    body = 11.0
    check("标题: 大字号居中短行", is_title("第一章 总则", 24, "STSong-Light", [150, 60, 445, 90], 595, body))
    check("标题: 黑体中字", is_title("故障排除", 14, "SimHei", [60, 100, 200, 120], 595, body))
    check("标题: 假加粗（字体名不变，仅靠字号）", is_title("付款条款", 18, "STSong-Light", [60, 100, 200, 125], 595, body))
    check("正文不该判成标题: 长句", not is_title(_PDF_BODY[:50] + "。", 11, "STSong-Light", [60, 140, 535, 240], 595, body))
    check("正文不该判成标题: 同字号短句", not is_title("本合同项下", 11, "STSong-Light", [60, 140, 180, 155], 595, body))
    check("正文不该判成标题: 左对齐无字体证据", not is_title("附表说明", 12, "Arial", [60, 300, 200, 315], 595, body))
    check("正文不该判成标题: 宋体正文小字（宋体必须在字体黑名单外）",
          not is_title("第一章 总则说明", 11, "SimSun", [60, 300, 200, 315], 595, body))
    # 三条信号是「或」关系，且各自可单独命中——层级由字号档位给出
    v1 = score_title("第一章 总则", 24, "STSong-Light", [150, 60, 445, 90], 595, body)
    check("标题信号: 仅字号命中", v1["signals"]["size"] and not v1["signals"]["font"], str(v1["signals"]))
    check("标题层级: ratio>=1.8 给 1 级", v1["level_hint"] == 1, str(v1["level_hint"]))
    v2 = score_title("故障排除", 14, "SimHei", [60, 100, 200, 120], 595, body)
    check("标题信号: 仅字体命中（字号 1.27 倍不够 1.3）",
          v2["signals"]["font"] and not v2["signals"]["size"] and v2["hit"], str(v2))
    v3 = score_title("附录 A", 11, "Arial", [260, 100, 335, 120], 595, body)
    check("标题信号: 仅居中命中", v3["signals"]["center_short"] and not v3["signals"]["size"], str(v3["signals"]))
    check("标题层级: 无字号证据给 3 级", v3["level_hint"] == 3, str(v3["level_hint"]))
    check("标题打分保留证据分供排序", v1["score"] > v2["score"], f"{v1['score']} vs {v2['score']}")

    # ---------- 分类器 ----------
    c_contract = classify("合同编号：HT-01 甲方：A 公司 乙方：B 公司 签订日期 2024年1月1日 违约责任 违约金 本合同一式贰式", [])
    check("分类: contract", c_contract["doc_type"] == "contract", json.dumps(c_contract["doc_type_scores"], ensure_ascii=False))
    c_invoice = classify("发票代码：044021900111 发票号码：12345678 开票日期：2024年5月8日 价税合计 ￥1,234.00 税额 142.43 纳税人识别号 91440300MA5XXXXX8K", [])
    check("分类: invoice", c_invoice["doc_type"] == "invoice", json.dumps(c_invoice["doc_type_scores"], ensure_ascii=False))
    check("分类: invoice -> layout form", c_invoice["layout_style"] == "form", c_invoice["layout_style"])
    c_report = classify("2023年度报告：报告期内公司实现营业收入12.3亿元，同比增长18.6%，净利润率12.4%，毛利率提升，董事会审议通过。", [])
    check("分类: report", c_report["doc_type"] == "report", json.dumps(c_report["doc_type_scores"], ensure_ascii=False))
    check("分类: report -> topic finance", c_report["topic"] == "finance", c_report["topic"])
    c_manual = classify("第3章 安装步骤 步骤1：连接电源线。注意事项：请勿潮湿环境使用。故障排除：指示灯不亮检查适配器。技术参数：额定电压220V。本产品保修一年。", [])
    check("分类: manual", c_manual["doc_type"] == "manual", json.dumps(c_manual["doc_type_scores"], ensure_ascii=False))
    check("分类: 空文本 -> unknown", classify("", [])["doc_type"] == "unknown")

    # ---------- 表格 markdown ----------
    md = common.table_to_markdown([["月", "额"], ["1", "10"], ["2", None], ["", ""]], header_rows=1)
    check("表格: 表头 + 分隔线", md.startswith("| 月 | 额 |\n|---|---|"), repr(md))
    check("表格: 空行被剔除", "2" in md and md.count("|---|") == 1, repr(md))
    long = common.table_to_markdown([["h1", "h2"]] + [[f"r{i}", str(i)] for i in range(5)], header_rows=1, max_rows=3)
    check("表格: 超长分段重复表头", long.count("| h1 | h2 |") == 3, repr(long[:120]))
    check("表格: 竖线转义不撕裂", "\\|" in common.table_to_markdown([["a|b", "c"], ["d", "e"]], 1))
    # 表头选择：跨列合并的表题横幅不能被当列名行
    h_idx, header = common.pick_header_row(
        [["2024年费用明细", "", "", ""], ["月份", "部门", "费用类型", "金额"], ["1月", "研发", "差旅", "100"]]
    )
    check("表头选择: 跳过合并横幅", h_idx == 1 and header == ["月份", "部门", "费用类型", "金额"], str((h_idx, header)))
    h_idx2, header2 = common.pick_header_row([["科目", "2023年", "2024年"], ["营业收入", "1000", "1180"]])
    check("表头选择: 正常列名行不改动", h_idx2 == 0 and header2 == ["科目", "2023年", "2024年"], str((h_idx2, header2)))
    st = common.HeadingStack()
    st.seed_root("产品使用手册")
    st.push(1, "产品使用手册")   # 与根同名的正文大标题不该重复入栈
    st.push(1, "第1章 安装")
    st.push(2, "1.1 硬件连接")
    check("标题栈: 根 + 去重 + 层级", st.path == "产品使用手册 > 第1章 安装 > 1.1 硬件连接", st.path)
    st.push(1, "第2章 使用")
    check("标题栈: 同级替换", st.path == "产品使用手册 > 第2章 使用", st.path)

    # ---------- txt ----------
    txt_res = parse_txt(make_txt())
    validate_schema(txt_res, "txt")
    check("txt: markdown 表格成 table block", any(b["type"] == "table" for b in txt_res["blocks"]))
    check("txt: 标题产出 title block", any(b["type"] == "title" for b in txt_res["blocks"]))
    check("txt: section_path 拼接", any("采购合同" in b.get("section_path", "") for b in txt_res["blocks"]),
          str([b.get("section_path") for b in txt_res["blocks"][:4]]))
    check("txt: language=zh", txt_res["metadata"]["language"] == "zh", txt_res["metadata"]["language"])
    check("txt: doc_type=contract", txt_res["metadata"]["doc_type"] == "contract", txt_res["metadata"]["doc_type"])
    gbk = parse_txt(make_txt_gbk())
    check("txt: GBK 自动探测", gbk.get("encoding", "").startswith("gb"), str(gbk.get("encoding")))
    check("txt: GBK 内容不乱码", "营收" in json.dumps(gbk["blocks"], ensure_ascii=False))

    # ---------- docx ----------
    dx = parse_docx(make_docx())
    validate_schema(dx, "docx")
    types = [b["type"] for b in dx["blocks"]]
    check("docx: 有 title", "title" in types, str(types))
    check("docx: 有 table", "table" in types, str(types))
    check("docx: 有 image", "image" in types, str(types))
    tbl = next(b for b in dx["blocks"] if b["type"] == "table")
    check("docx: 合并单元格不重复表头列", tbl["content"].count("2024年") == 1, tbl["content"])
    check(
        "docx: 合并表头被拆回各列（列名完整保留）",
        tbl.get("header") == ["科目", "2023年", "2024年"],
        str(tbl.get("header")),
    )
    check("docx: 章节路径按 heading 层级", any("产品使用手册 > 第1章 安装" in b.get("section_path", "") for b in dx["blocks"]),
          str([b.get("section_path") for b in dx["blocks"][:6]]))
    fake = next((b for b in dx["blocks"] if b["content"] == "故障排除"), None)
    check("docx: 无样式的加粗大字被识别为标题", fake is not None and fake["type"] == "title", str(fake))
    check("docx: 正文基准字号解析链生效（11pt）", dx.get("body_font_size") == 11.0, str(dx.get("body_font_size")))
    img_blk = next(b for b in dx["blocks"] if b["type"] == "image")
    check("docx: 内嵌图取到下一段图注", "图1 营收趋势" in img_blk["content"], img_blk["content"])
    check("docx: OCR 缺失时 image 块用占位而非空串", bool(img_blk["content"].strip()), repr(img_blk["content"]))
    check("docx: 记录图片张数", dx.get("n_images") == 1, str(dx.get("n_images")))
    check("docx: language=zh", dx["metadata"]["language"] == "zh")

    # ---------- pptx ----------
    px = parse_pptx(make_pptx())
    validate_schema(px, "pptx")
    slides = [b for b in px["blocks"] if b["type"] == "slide"]
    check("pptx: 每页一个 slide block", len(slides) == 2, str(len(slides)))
    check("pptx: content 含标题+要点", "2024年营收分析" in slides[0]["content"] and "同比增长 18%" in slides[0]["content"])
    check("pptx: content 含备注", "备注：" in slides[0]["content"])
    check("pptx: 负号不被当项目符号剥掉", "-15% 的汇兑损失需关注" in slides[0]["content"], slides[0]["content"])
    check("pptx: 表格单独成块", any(b["type"] == "table" for b in px["blocks"]))
    check("pptx: 图片单独成块", any(b["type"] == "image" for b in px["blocks"]))
    check("pptx: section_path 用页码", slides[1]["section_path"].startswith("第2页"), slides[1]["section_path"])
    check("pptx: layout_style=slides", px["metadata"]["layout_style"] == "slides")

    # ---------- xlsx ----------
    xl = parse_xlsx(make_xlsx())
    validate_schema(xl, "xlsx")
    sheet_blocks = [b for b in xl["blocks"] if b["type"] == "sheet"]
    check("xlsx: 每区域一块", len(sheet_blocks) >= 2, str(len(sheet_blocks)))
    check("xlsx: 每 100 行一切", all(b["n_rows"] <= 100 for b in sheet_blocks), str([b["n_rows"] for b in sheet_blocks]))
    check("xlsx: 每块都带列名", all(b["content"].splitlines()[0].strip("| ").startswith("2024年费用明细") or
                                    "月份" in b["content"].splitlines()[0] for b in sheet_blocks),
          sheet_blocks[0]["content"].splitlines()[0])
    check("xlsx: 空行被剔除", all("|| " not in b["content"] for b in sheet_blocks))
    check("xlsx: 整列噪声列被剔除", all("备注列整列为空" not in b["content"] for b in sheet_blocks),
          sheet_blocks[0]["content"])
    check("xlsx: columns 是列名行而不是合并横幅",
          sheet_blocks[0]["columns"] == ["月份", "部门", "费用类型", "金额"], str(sheet_blocks[0]["columns"]))
    check("xlsx: 合并横幅单独记为 table_title",
          sheet_blocks[0]["table_title"] == "2024年费用明细", str(sheet_blocks[0]["table_title"]))
    check("xlsx: 隐藏 sheet 不入库", all(b["sheet_name"] != "隐藏表" for b in sheet_blocks))
    check("xlsx: 合并大标题回填到每行区域", all("2024年费用明细" in b["content"] for b in sheet_blocks))
    check("xlsx: section_path = sheet > 区域", sheet_blocks[0]["section_path"].startswith("月度明细 > 第"),
          sheet_blocks[0]["section_path"])
    check("xlsx: has_table=True & layout=sheet",
          xl["metadata"]["has_table"] and xl["metadata"]["layout_style"] == "sheet")

    # ---------- pdf ----------
    import fitz

    pdf_path = make_pdf()
    pr = parse_pdf(pdf_path)
    validate_schema(pr, "pdf")
    types = [b["type"] for b in pr["blocks"]]
    check("pdf: 产出 title/text/table/image 四类", {"title", "text", "table"} <= set(types), str(types))
    title_blk = next((b for b in pr["blocks"] if b["type"] == "title"), None)
    check("pdf: 大字号居中被判标题", title_blk is not None and "第一章" in title_blk["content"], str(title_blk))
    check("pdf: 文本块带 font/size/bbox", all(k in pr["blocks"][1] for k in ("font", "size", "bbox")),
          str(sorted(pr["blocks"][1])))
    check("pdf: 基准字号估计合理", pr.get("body_font_size") in (10.5, 11.0), str(pr.get("body_font_size")))
    tblk = next((b for b in pr["blocks"] if b["type"] == "table"), None)
    check("pdf: 表格转 markdown", tblk is not None and tblk["content"].startswith("| 科目 |"), str(tblk)[:120])
    check("pdf: 表格 bbox 有值", tblk and tblk["bbox"][0] < tblk["bbox"][2])
    inside = [
        b for b in pr["blocks"]
        if b["type"] == "text" and b.get("page") == 2 and "营业收入" in b["content"]
    ]
    check("pdf: 表格区域内的文本块不重复入库", not inside, str(inside)[:120])
    check("pdf: 扫描件分支命中第 3 页", pr.get("scanned_pages") == [3], str(pr.get("scanned_pages")))
    check("pdf: 模型缺失走降级并记录原因", bool(pr.get("model_errors")), str(pr.get("model_errors")))
    p2_paths = [b.get("section_path", "") for b in pr["blocks"] if b.get("page") == 2]
    check("pdf: 章节跨页延续（单一全局标题栈）",
          all("第一章 总则" in p for p in p2_paths if p), str(p2_paths))
    check("pdf: section_path 带文档根",
          all(p.startswith("sample") for p in p2_paths if p), str(p2_paths))
    check("pdf: outline 被带出", len(pr.get("outline", [])) == 2, str(pr.get("outline")))
    check("pdf: doc_type 命中 contract/report",
          pr["metadata"]["doc_type"] in ("contract", "report"), pr["metadata"]["doc_type"])

    # 标题判定的边界：fitz 抓到的真实 span 走一遍
    doc = fitz.open(str(pdf_path))
    stats = collect_page_blocks(doc[0], 1)
    bs = estimate_body_size(stats)
    check("pdf: 第一页正文基准字号 ~11", 10.5 <= bs <= 11.5, str(bs))
    check("pdf: is_scanned_page 第1页 False / 第3页 True",
          not is_scanned_page(doc[0]) and is_scanned_page(doc[2]))
    doc.close()

    # ---------- image ----------
    from ingestion.classifiers.image_classifier import (
        IMAGE_CATEGORIES,
        MIN_CLASS_CONFIDENCE,
        classify_image_bytes,
        is_vector_like,
    )

    ip = TMP / "scanned.png"
    im_res = parse_image(ip)
    validate_schema(im_res, "image")
    check("image: layout_style=image", im_res["metadata"]["layout_style"] == "image")
    check("image: has_image=True", im_res["metadata"]["has_image"])
    check("image: 模型缺失不抛异常", im_res["blocks"][0].get("clip_vector") in (None, []) or
          isinstance(im_res["blocks"][0].get("clip_vector"), list))
    check("image: 向量缺失时不写 clip_vector 字段（不伪造）",
          "clip_vector" not in im_res["blocks"][0] or im_res["blocks"][0]["clip_vector"] is None,
          str(im_res["blocks"][0].get("clip_vector"))[:60])
    check("image: 分类结果始终带 label", im_res["blocks"][0]["classification"]["label"] == "unknown",
          json.dumps(im_res["blocks"][0]["classification"], ensure_ascii=False)[:120])
    check("image: 独立入口 caption 默认取文件名", ip.stem in im_res["blocks"][0]["content"],
          im_res["blocks"][0]["content"])
    # ---------- 图片 zero-shot 分类 ----------
    check("图片分类: 五个类别齐全", set(IMAGE_CATEGORIES) == {"chart", "photo", "diagram", "signature", "form"},
          str(IMAGE_CATEGORIES))
    check("图片分类: 阈值 0.4", MIN_CLASS_CONFIDENCE == 0.4, str(MIN_CLASS_CONFIDENCE))
    cls = classify_image_bytes(ip.read_bytes())
    check("图片分类: 模型缺失走 unknown 且带 error",
          cls["label"] == "unknown" and bool(cls.get("error")), json.dumps(cls, ensure_ascii=False)[:120])
    check("图片分类: signature 不进向量库", not is_vector_like({"label": "signature", "confidence": 0.91}))
    check("图片分类: chart 进向量库", is_vector_like({"label": "chart", "confidence": 0.82}))

    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED {len(FAILS)} 项：")
        for f in FAILS:
            print("  -", f)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
