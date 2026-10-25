# -*- coding: utf-8 -*-
"""
样例文档工厂：由 _check_batch2 / _check_batch3 迁移而来，供 pytest 夹具复用。

约定：所有 make_* 工厂接收一个已存在的目标目录（通常是 tmp_path 下的子目录），
在其中生成真实可解析的 docx/pptx/xlsx/pdf/txt/png，返回落地文件路径。

- batch2 系列：面向解析器断言（标题判定、表格 markdown、section_path、编码探测…）
- chunking 系列（*_chunk 前缀）：面向切分器/流水线断言（长正文、大表、跨页同 section…）
- synth_blocks：直接构造统一中间 blocks，测切分器纯逻辑，不落文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# batch3 用的长正文（无标点整句，压 overlap 整句切分与 token 窗口）
LONG_SENT = (
    "本合同项下甲方的付款期限为发票到账后的三十个自然日内逾期按日万分之五"
    "计算违约金直至全部款项结清为止。"
)

# batch2 的 PDF 正文（第一条/第二条/第三条，供 contract 分类）
_PDF_BODY = (
    "第一条 合同双方本着平等自愿、诚实信用的原则，经友好协商，就本次采购事项签订本合同。"
    "第二条 乙方应当按照约定的质量标准和交付期限完成供货义务，逾期交付的，每日按合同总价的千分之三支付违约金。"
    "第三条 因不可抗力导致合同无法履行的，双方均不承担违约责任，但应当及时通知对方并提供证明。"
)


def _fitz():
    """兼容 pymupdf 新旧导入名（1.24 起 pymupdf 为正名，fitz 仍可用）。"""
    try:
        import pymupdf

        return pymupdf
    except ImportError:  # pragma: no cover - 旧版本回退
        import fitz

        return fitz


# ============================================================ 图片
def make_png(path: Path, size: int = 300, text: str = "财务报表 2024") -> None:
    """画一张带中文的图，用于内嵌图片与独立图片入口。"""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size // 2), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([4, 4, size - 5, size // 2 - 5], outline="black", width=2)
    d.line([20, 90, 80, 40, 140, 70, 240, 20], fill="red", width=3)
    im.save(path, "PNG")


# ============================================================ batch2：解析器样例
def make_txt(directory: Path) -> Path:
    p = directory / "sample.txt"
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


def make_txt_gbk(directory: Path) -> Path:
    p = directory / "sample_gbk.txt"
    p.write_bytes(
        "2024年度财务分析报告\n营收 1.2 亿元，同比增长 18%，成本下降 3%。\n".encode("gbk")
    )
    return p


def make_docx(directory: Path) -> Path:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    p = directory / "sample.docx"
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

    img = directory / "inline.png"
    make_png(img)
    d.add_picture(str(img), width=Pt(120))
    d.add_paragraph("图1 营收趋势")
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.save(str(p))
    return p


def make_pptx(directory: Path) -> Path:
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
    for i, row in enumerate(
        [["季度", "营收", "同比"], ["Q1", "300", "+12%"], ["Q2", "320", "+22%"]]
    ):
        for j, val in enumerate(row):
            tbl.cell(i, j).text = val
    slide.notes_slide.notes_text_frame.text = "本页强调二季度增速回升，成本端压力仍在。"

    img = directory / "slide_img.png"
    make_png(img)
    s2 = prs.slides.add_slide(layout)
    s2.shapes.title.text = "示意图页"
    s2.shapes.add_picture(str(img), Inches(1), Inches(2), height=Pt(120))
    cap = s2.shapes.add_textbox(Inches(1), Inches(3.6), Inches(4), Inches(0.4))
    cap.text_frame.text = "图2 组织架构"
    path = directory / "sample.pptx"
    prs.save(str(path))
    return path


def make_xlsx(directory: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "月度明细"
    ws["A1"] = "2024年费用明细"
    ws.merge_cells("A1:D1")  # 合并大标题，测试回填
    ws.append(["月份", "部门", "费用类型", "金额"])
    for m in range(1, 13):
        for dept in ("研发", "销售", "行政"):
            # 12 月 × 3 部门 × 3 费用类型 = 108 数据行，跨过 100 行分区阈值
            for kind in ("差旅费", "培训费", "招待费"):
                ws.append([f"2024-{m:02d}", dept, kind, 1000 + m * 7])
    ws.append([None, None, None, None])  # 空行，应被剔除
    ws.append(["", "", "", ""])
    ws["F1"] = "备注列整列为空"  # 只有 1 行有值 -> 噪声列，应被剔除
    other = wb.create_sheet("隐藏表")
    other["A1"] = "不该出现"
    other.sheet_state = "hidden"
    path = directory / "sample.xlsx"
    wb.save(str(path))
    return path


def make_pdf(directory: Path) -> Path:
    """
    造一个 3 页 PDF：
      p1 文本层 + 大字号居中标题（测标题判定）
      p2 带框线表格（测 pdfplumber 抽取 + 文本块去重）
      p3 只有一张图无文本层（测扫描件分流 + OCR 分支的降级路径）
    """
    fitz = _fitz()

    path = directory / "sample.pdf"
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
            tp.insert_text(fitz.Point(xs[c] + 6, ys[r] + 20), val, fontname="china-s", fontsize=10)

    # 扫描页：只贴一张图，不放文本层
    sp = doc.new_page(width=595, height=842)
    img = directory / "scanned.png"
    make_png(img)
    sp.insert_image(fitz.Rect(50, 50, 545, 700), filename=str(img))
    doc.set_toc([[1, "第一章 总则", 1], [1, "附表：年度营收", 2]])
    doc.save(str(path))
    doc.close()
    return path


# ============================================================ batch3：切分/流水线样例
def make_chunk_txt(directory: Path) -> Path:
    """带长正文 + markdown 大表的 txt（同时压到切分与表格两条路）。"""
    p = directory / "sample.txt"
    body = "".join(f"{LONG_SENT}第{i}条补充说明内容。\n" for i in range(1, 31))
    rows = "\n".join(
        f"| {i} | 设备采购明细第{i}项 | {12000 + i * 37} | 含三年质保与现场安装培训服务费用已含在单价内 |"
        for i in range(1, 41)
    )
    content = (
        "# 采购合同\n"
        "甲方：星辰科技有限公司\n"
        "乙方：海纳供应链有限公司\n\n"
        "## 第一条 标的物\n"
        f"{body}\n"
        "## 第二条 费用明细\n"
        "| 序号 | 名称 | 单价 | 备注 |\n"
        "|---|---|---|---|\n"
        f"{rows}\n\n"
        "## 第三条 违约责任\n"
        "违约方应向守约方支付合同总价百分之十的违约金，并赔偿由此造成的全部损失。\n"
    )
    p.write_text(content, encoding="utf-8")
    return p


def make_chunk_docx(directory: Path) -> Path:
    import docx

    p = directory / "sample.docx"
    d = docx.Document()
    d.core_properties.title = "产品使用手册"
    d.add_heading("产品使用手册", level=1)
    d.add_heading("第3章 系统配置", level=1)
    for i in range(1, 13):
        d.add_paragraph(f"第3章正文第{i}句：{LONG_SENT}")
    d.add_heading("3.2 参数表", level=2)
    table = d.add_table(rows=3, cols=3)
    table.rows[0].cells[0].text = "参数名"
    table.rows[0].cells[1].text = "默认值"
    table.rows[0].cells[2].text = "说明"
    table.rows[1].cells[0].text = "chunk_size"
    table.rows[1].cells[1].text = "384"
    table.rows[1].cells[2].text = "切分窗口大小"
    table.rows[2].cells[0].text = "chunk_overlap"
    table.rows[2].cells[1].text = "64"
    table.rows[2].cells[2].text = "相邻窗口重叠量"
    d.add_paragraph("表格之后的正文：上述参数需要与硬件档位一起调整。")
    d.save(p)
    return p


def make_chunk_pptx(directory: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    p = directory / "sample.pptx"
    prs = Presentation()
    for i in (1, 2):
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(3))
        para = box.text_frame.paragraphs[0]
        para.text = f"第{i}页：季度营收同比增长百分之十八，成本下降百分之三。"
        para.font.size = Pt(24)
    prs.save(p)
    return p


def make_chunk_xlsx(directory: Path) -> Path:
    from openpyxl import Workbook

    p = directory / "sample.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "费用明细"
    ws.append(["部门", "科目", "金额", "备注说明"])
    for i in range(1, 46):
        ws.append([f"研发部{i % 5}", "差旅费", 1000 + i * 13, "含往返交通与住宿补贴按公司标准执行"])
    wb.create_sheet("预算")
    wb.save(p)
    return p


def make_chunk_pdf(directory: Path) -> Path:
    """两页同一章节的正文，用来验「跨页同 section 合并」在真实 PDF 上生效。"""
    fitz = _fitz()

    p = directory / "sample.pdf"
    doc = fitz.open()
    for page_no in (1, 2):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 60, 540, 720),
            f"第{page_no}页 {LONG_SENT}这是跨页延续的第{page_no * 2}段文字内容用于验证合并。",
            fontname="china-ss",
            fontsize=12,
        )
    doc.set_metadata({"title": "财务分析报告"})
    doc.save(p)
    doc.close()
    return p


def make_broken_pdf(directory: Path) -> Path:
    p = directory / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not a real pdf body\n%%EOF-truncated")
    return p


def build_ingest_dir(directory: Path) -> list[Path]:
    """造齐 batch3 流水线用的整套输入文件（含坏 PDF）。"""
    return [
        make_chunk_txt(directory),
        make_chunk_docx(directory),
        make_chunk_pptx(directory),
        make_chunk_xlsx(directory),
        make_chunk_pdf(directory),
        make_broken_pdf(directory),
    ]


# ============================================================ 合成 blocks（batch3 切分纯逻辑）
def synth_blocks(n_sentences: int = 40) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """同 section 跨 3 页的正文 + 中间夹一张表 + 尾部另一个 section。"""
    blocks: list[dict[str, Any]] = [
        {"type": "title", "content": "第3章 付款方式", "page": 1, "section_path": "第3章 付款方式"},
    ]
    for i in range(n_sentences):
        blocks.append(
            {
                "type": "text",
                "content": f"{LONG_SENT}第{i}项补充约定。",
                "page": 1 + i // 14,
                "section_path": "第3章 付款方式",
                "bbox": [50.0, 60.0 + i, 540.0, 80.0 + i],
            }
        )
    blocks.append(
        {
            "type": "title",
            "content": "3.2 表格",
            "page": 3,
            "section_path": "第3章 付款方式 > 3.2 表格",
        }
    )
    blocks.append(
        {
            "type": "table",
            "content": (
                "| 项目 | 金额 | 说明 |\n|---|---|---|\n"
                "| 预付款 | 30000 | 签约后七日内支付 |\n| 尾款 | 70000 | 验收合格后支付 |\n"
            ),
            "page": 3,
            "section_path": "第3章 付款方式 > 3.2 表格",
            "header": ["项目", "金额", "说明"],
            "bbox": [50.0, 300.0, 540.0, 420.0],
        }
    )
    blocks.append(
        {
            "type": "text",
            "content": "独立章节的一句话：所有款项均以银行转账方式结算。",
            "page": 4,
            "section_path": "第4章 结算",
        }
    )
    metadata = {
        "doc_type": "contract",
        "topic": "finance",
        "has_table": True,
        "has_image": False,
        "language": "zh",
        "layout_style": "mixed",
        "title": "采购合同",
    }
    return blocks, metadata
