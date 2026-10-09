# -*- coding: utf-8 -*-
"""
文档类型 / 主题 / 语言 / 版式 分类（纯规则 + 关键词，不引入分类模型）。

为什么不上模型：
  100GB 库里文档分布未知且会漂移，规则表可解释、可单条增删、零推理成本；
  错分的文档在检索阶段还能靠 rerank 的 metadata 一致性惩罚兜住，
  而训练 doc_type 分类器需要先有一批标注，本批次拿不到。

打分口径（关键，避免长文档必然胜出）：
  score = Σ(命中权重) / sqrt(文本千字数) + 强锚点加分
  强锚点是「几乎唯一指向某类型」的正则（如发票号码、甲方/乙方成对出现）。

依赖：只用标准库 + 本项目的 common（语言判定）。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from ingestion.parsers.common import detect_language

# =================================================================== 阈值
# 命中密度阈值：分数低于它一律回落 unknown，宁可不知道也别瞎猜
DOC_TYPE_MIN_SCORE = 1.0
TOPIC_MIN_SCORE = 0.8
# 版式判定：form 的「短行 + 冒号键值对」比例
FORM_SHORT_LINE_RATIO = 0.5
FORM_COLON_LINE_RATIO = 0.22
FORM_MIN_LINES = 8

# =================================================================== 词表
# 每条 (关键词, 权重)。权重体现区分度：越专属越高。
DOC_TYPE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "contract": [
        ("甲方", 3.0), ("乙方", 3.0), ("委托方", 2.0), ("受托方", 2.0),
        ("签订", 2.5), ("签署", 1.5), ("本合同", 3.0), ("协议书", 2.5),
        ("条款", 2.0), ("违约责任", 3.5), ("违约金", 3.5), ("免责", 1.5),
        ("不可抗力", 3.0), ("争议解决", 2.5), ("仲裁", 2.0), ("盖章", 1.5),
        ("补充协议", 2.5), ("履约", 1.5), ("订于", 1.5), ("甲乙双方", 3.0),
        ("合同编号", 3.0), ("生效日期", 1.5), ("协议期限", 2.0),
    ],
    "report": [
        ("年度", 1.5), ("季度", 1.5), ("半年报", 2.5), ("同比", 2.5),
        ("环比", 2.5), ("营收", 2.5), ("营业收入", 2.0), ("净利润", 2.5),
        ("毛利率", 2.5), ("现金流", 2.0), ("资产负债表", 2.5), ("利润表", 2.5),
        ("报告", 1.0), ("摘要", 1.0), ("结论与建议", 2.0), ("董事会", 1.5),
        ("股东会", 2.0), ("审计", 1.5), ("财务分析", 2.5), ("综述", 1.5),
        ("报告期内", 3.0), ("营业收入构成", 3.0), ("风险提示", 2.0),
    ],
    "manual": [
        ("使用说明", 3.0), ("操作手册", 3.5), ("用户手册", 3.5), ("安装步骤", 3.0),
        ("注意事项", 2.0), ("故障排除", 3.0), ("技术参数", 2.5), ("配置项", 2.0),
        ("目录", 1.0), ("版本", 0.8), ("第1章", 1.5), ("第一章", 2.0),
        ("本节", 1.2), ("如图", 1.2), ("见表", 1.2), ("示例", 1.0),
        ("快速入门", 2.5), ("接口说明", 2.0), ("快捷键", 2.5), ("维护保养", 2.0),
        ("本产品", 1.5), ("操作步骤", 2.5),
    ],
    "invoice": [
        ("发票", 2.0), ("发票代码", 4.0), ("发票号码", 4.0), ("价税合计", 4.0),
        ("税额", 3.0), ("税率", 2.5), ("开票日期", 3.5), ("纳税人识别号", 4.0),
        ("增值税", 2.5), ("普通发票", 3.5), ("专用发票", 3.0), ("购销双方", 3.0),
        ("收款人", 1.5), ("复核", 1.5), ("开票人", 2.5), ("机器编号", 3.0),
        ("发票联", 3.5), ("抵扣联", 3.5), ("金额合计", 2.5), ("校验码", 3.5),
    ],
}

# 强锚点：命中即直接加分（几乎不会出现在别的类型里）
DOC_TYPE_ANCHORS: dict[str, list[tuple[str, float]]] = {
    "contract": [
        (r"(甲方|乙方|委托方|受托方)[：:]", 2.0),
        (r"本合同.{0,20}(一式|贰式|两式)各.{0,6}份", 3.0),
        (r"签订(日期|时间)", 2.0),
        (r"第[一二三四五六七八九十]+条", 1.5),
    ],
    "report": [
        (r"(20\d{2}|XXXX)年度.{0,10}(报告|总结|分析)", 2.5),
        (r"(同比|环比)(增长|下降)\s*[\d.]+", 2.5),
        (r"净利润率|每股收益|ROE|EBITDA", 2.0),
    ],
    "manual": [
        (r"第\s*\d+\s*章", 2.0),
        (r"\d+\.\d+\s+[\u4e00-\u9fa5]{2,12}\s*$", 1.0),
        (r"(步骤\s*\d+|Step\s*\d+)", 1.5),
    ],
    "invoice": [
        (r"发票代码[:：]?\s*\d{8,12}", 4.0),
        (r"发票号码[:：]?\s*\d{8}", 4.0),
        (r"￥\s*[\d,]+\.\d{2}", 2.0),
        (r"(91|12)\d{6}[0-9A-Z]{9,10}", 3.0),  # 统一社会信用代码
    ],
}

TOPIC_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "finance": [
        ("财务", 2.5), ("会计", 2.5), ("成本", 2.0), ("利润", 2.0), ("营收", 2.5),
        ("预算", 2.0), ("税", 1.5), ("资产", 1.5), ("负债", 1.5), ("现金流", 2.5),
        ("投资", 1.5), ("融资", 2.0), ("报销", 2.5), ("对账", 2.5), ("审计", 2.0),
        ("银行", 1.5), ("汇率", 2.0), ("应收", 2.5), ("应付", 2.5), ("折旧", 2.5),
        ("毛利", 2.5), ("净利", 2.5), ("金额", 1.0), ("价格", 1.0), ("结算", 1.5),
    ],
    "legal": [
        ("法律", 2.5), ("合同", 2.5), ("协议", 2.0), ("条款", 2.0), ("违约", 2.5),
        ("赔偿", 1.5), ("诉讼", 2.5), ("仲裁", 2.0), ("合规", 2.0), ("授权", 1.5),
        ("知识产权", 2.5), ("商标", 2.0), ("专利", 2.0), ("保密", 1.5),
        ("法律责任", 2.5), ("监管", 1.5), ("备案", 1.5), ("公证", 2.5),
        ("解除", 1.0), ("争议", 1.5), ("委托", 1.0),
    ],
    "technical": [
        ("系统", 1.5), ("接口", 2.0), ("数据库", 2.5), ("算法", 2.5), ("部署", 2.0),
        ("服务", 1.0), ("架构", 2.5), ("代码", 2.5), ("版本", 1.5), ("配置", 1.5),
        ("API", 2.5), ("SDK", 2.5), ("服务器", 2.0), ("网络", 1.5), ("协议栈", 2.5),
        ("性能", 1.5), ("测试", 1.0), ("缺陷", 1.5), ("参数", 1.0), ("设备", 1.0),
        ("安装", 1.0), ("驱动", 2.5), ("日志", 1.5), ("监控", 1.5),
    ],
}

# =================================================================== 正则
_COLON_LINE_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9（）()·、\.]{1,16}[：:]{1}")


# =================================================================== 工具
def _keyword_score(text: str, rules: Iterable[tuple[str, float]]) -> tuple[float, dict[str, int]]:
    """
    关键词命中打分。
    用「出现次数开方」做饱和，防止某个词在长表格里重复 500 次把分数刷爆。
    返回 (得分, 命中明细)。
    """
    score = 0.0
    hits: dict[str, int] = {}
    for word, weight in rules:
        cnt = text.count(word)
        if cnt:
            hits[word] = cnt
            score += weight * math.sqrt(cnt)
    return score, hits


def _anchor_score(text: str, patterns: Iterable[tuple[str, float]]) -> tuple[float, list[str]]:
    """强锚点正则打分（命中即加分，不做饱和）。"""
    score = 0.0
    matched: list[str] = []
    for pattern, weight in patterns:
        if re.search(pattern, text, flags=re.MULTILINE):
            matched.append(pattern)
            score += weight
    return score, matched


def _normalize_by_length(raw: float, text: str) -> float:
    """
    长度归一化：除以 sqrt(千字数)。
    1 万字的年报和 200 字的收据放在同一尺度上比较，否则长文必胜。
    """
    kchars = max(len(text) / 1000.0, 1.0)
    return raw / math.sqrt(kchars)


# =================================================================== 语言
def language_of(text: str, blocks: list[dict[str, Any]] | None = None) -> str:
    """language 判定统一入口（zh / en / mixed），blocks 参与时取拼接文本。"""
    joined = text or _blocks_text(blocks or [])
    return detect_language(joined)


def _blocks_text(blocks: list[dict[str, Any]]) -> str:
    """把 blocks 的 content 拼成一个用于分类的纯文本。"""
    return "\n".join(str(b.get("content", "")) for b in blocks if b.get("content"))


# =================================================================== 版式
def infer_layout_style(
    text: str,
    blocks: list[dict[str, Any]] | None = None,
    doc_type: str = "unknown",
) -> str:
    """
    layout_style 推断：form | prose | slides | sheet | image

    优先级：sheet/slide/image 由 block 类型直接决定（结构证据最强）；
    form 用「短行占比 + 冒号键值行占比」判定；否则 prose。
    发票默认 form（版式天然是字段表）。
    """
    blocks = blocks or []
    types = [b.get("type") for b in blocks]
    total = max(len(types), 1)

    def _ratio(t: str) -> float:
        return types.count(t) / total

    if _ratio("sheet") >= 0.5:
        return "sheet"
    if _ratio("slide") >= 0.5:
        return "slides"
    if types and _ratio("image") >= 0.6:
        return "image"
    if doc_type == "invoice":
        return "form"

    lines = [ln.strip() for ln in (text or _blocks_text(blocks)).splitlines() if ln.strip()]
    if len(lines) >= FORM_MIN_LINES:
        short_lines = sum(1 for ln in lines if len(ln) <= 24)
        colon_lines = sum(1 for ln in lines if _COLON_LINE_RE.match(ln))
        if short_lines / len(lines) >= FORM_SHORT_LINE_RATIO and (
            colon_lines / len(lines) >= FORM_COLON_LINE_RATIO
        ):
            return "form"
    return "prose"


# =================================================================== 主分类
def classify_doc_type(text: str, blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    doc_type 分类：contract | report | manual | invoice | unknown

    返回 {"doc_type", "scores", "evidence"}，evidence 保留命中词与锚点，
    方便后续抽检时人肉复核（不透明分类器在文档场景是负债）。
    """
    body = text or _blocks_text(blocks or [])
    if not body.strip():
        return {"doc_type": "unknown", "scores": {}, "evidence": {}}

    scores: dict[str, float] = {}
    evidence: dict[str, Any] = {}
    for doc_type, rules in DOC_TYPE_KEYWORDS.items():
        kw, hits = _keyword_score(body, rules)
        an, matched = _anchor_score(body, DOC_TYPE_ANCHORS.get(doc_type, []))
        raw = _normalize_by_length(kw, body) + an
        scores[doc_type] = round(raw, 3)
        evidence[doc_type] = {"keywords": hits, "anchors": matched}

    # 决胜顺序：invoice > contract > manual > report
    # 理由：发票/合同的锚点几乎互斥且更具体；报告与手册的关键词最容易互相污染，放最后
    priority = ["invoice", "contract", "manual", "report"]
    best = max(priority, key=lambda k: (scores.get(k, 0.0), -priority.index(k)))
    if scores.get(best, 0.0) < DOC_TYPE_MIN_SCORE:
        return {"doc_type": "unknown", "scores": scores, "evidence": {}}
    return {"doc_type": best, "scores": scores, "evidence": evidence}


def classify_topic(text: str, blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """topic 分类：finance | legal | technical | unknown（三个主题不互斥，取最高分）。"""
    body = text or _blocks_text(blocks or [])
    if not body.strip():
        return {"topic": "unknown", "scores": {}}
    scores: dict[str, float] = {}
    for topic, rules in TOPIC_KEYWORDS.items():
        kw, _ = _keyword_score(body, rules)
        scores[topic] = round(_normalize_by_length(kw, body), 3)
    best = max(scores, key=lambda k: scores[k])
    if scores[best] < TOPIC_MIN_SCORE:
        return {"topic": "unknown", "scores": scores}
    return {"topic": best, "scores": scores}


def has_table_from_blocks(blocks: list[dict[str, Any]]) -> bool:
    """
    has_table 判定：显式 table/sheet block 存在，或文本块里出现 markdown 表格分隔线。
    后者很重要——XLSX/DOCX 转出来的 markdown 里 `|---|` 就是表格结构证据。
    """
    for block in blocks:
        if block.get("type") in ("table", "sheet"):
            return True
        content = str(block.get("content", ""))
        if "|---|" in content or re.search(r"^\|.*\|$", content, flags=re.MULTILINE):
            return True
    return False


def has_image_from_blocks(blocks: list[dict[str, Any]]) -> bool:
    """has_image 判定：存在非空 image block（被尺寸过滤掉的装饰图不算）。"""
    return any(b.get("type") == "image" and b.get("content") for b in blocks)


def classify(text: str, blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    主入口：喂「全文 + blocks」，返回可直接并进 metadata 的字典。

    输出键与统一 Metadata Schema 完全对齐：
      doc_type / topic / has_table / has_image / language / layout_style
    额外返回 *_scores 与 classify_evidence，供离线抽检与阈值调优。
    """
    blocks = blocks or []
    body = text or _blocks_text(blocks)

    dtype = classify_doc_type(body, blocks)
    topic = classify_topic(body, blocks)
    language = language_of(body, blocks)
    layout = infer_layout_style(body, blocks, doc_type=dtype["doc_type"])

    return {
        "doc_type": dtype["doc_type"],
        "topic": topic["topic"],
        "has_table": has_table_from_blocks(blocks),
        "has_image": has_image_from_blocks(blocks),
        "language": language,
        "layout_style": layout,
        "doc_type_scores": dtype.get("scores", {}),
        "topic_scores": topic.get("scores", {}),
        "classify_evidence": dtype.get("evidence", {}),
    }


def merge_into_metadata(metadata: dict[str, Any], classified: dict[str, Any]) -> dict[str, Any]:
    """
    把分类结果并进 parser 的 metadata。
    parser 已经确定的结构事实（has_table / layout_style=sheet|slides|image）不被覆盖，
    只有 unknown / 缺失字段才用分类结果填——结构证据永远强于关键词证据。
    """
    structure_wins = {"has_table", "has_image"}
    for key, value in classified.items():
        if key.endswith("_scores") or key == "classify_evidence":
            continue
        if key in structure_wins:
            metadata[key] = bool(value) or bool(metadata.get(key))
            continue
        if key == "layout_style" and metadata.get("layout_style") in ("sheet", "slides", "image"):
            continue
        if metadata.get(key) in (None, "", "unknown"):
            metadata[key] = value
    return metadata


if __name__ == "__main__":
    import argparse
    import json

    cli = argparse.ArgumentParser(description="对一段文本 / 一个文件做规则分类")
    cli.add_argument("target", nargs="?", help="文本内容或文件路径（--file 时按路径读）")
    cli.add_argument("--file", action="store_true", help="target 是文件路径，读取其后分类")
    args = cli.parse_args()

    if args.target:
        if args.file:
            from pathlib import Path

            raw_text = Path(args.target).read_text(encoding="utf-8", errors="ignore")
        else:
            raw_text = args.target
        print(json.dumps(classify(raw_text, []), ensure_ascii=False, indent=2))
    else:
        # 无参数时跑一组自检样例，确认四条规则分支都活着
        samples = {
            "合同": "合同编号：HT-2024-017 甲方：某科技有限公司 乙方：某供应链有限公司 "
                    "经双方友好协商，就采购事项签订本合同。第一条 违约责任：违约方应支付违约金。",
            "发票": "发票代码：044021900111 发票号码：12345678 开票日期：2024年5月8日 "
                    "价税合计 ￥1,234.00 税额 ￥142.43 纳税人识别号：91440300MA5XXXXX8K",
            "报告": "2023年度报告 报告期内，公司实现营业收入 12.3 亿元，同比增长 18.6%，"
                    "净利润率 12.4%，毛利率同比提升 2.1 个百分点。董事会决议通过本利润分配方案。",
            "手册": "第3章 安装步骤 步骤1：连接电源线。注意事项：请勿在潮湿环境使用。"
                    "故障排除：指示灯不亮请检查适配器。技术参数：额定电压 220V。本产品保修一年。",
        }
        for name, txt in samples.items():
            res = classify(txt, [])
            print(f"{name:>3} -> doc_type={res['doc_type']:<8} topic={res['topic']:<9} "
                  f"layout={res['layout_style']:<6} lang={res['language']}")
