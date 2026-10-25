# -*- coding: utf-8 -*-
"""提示组装离线自检（由 _check_generation.py 迁移）：system+user、注入包裹、外部模板覆盖。"""

from __future__ import annotations

from pathlib import Path

from generation import prompts


def test_build_messages_basic():
    msgs = prompts.build_messages(
        "违约条款", "[1] 【第3章】乙方应支付违约金", [{"chunk_id": "c1"}], {}
    )
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    assert "违约条款" in msgs[1]["content"] and "参考资料" in msgs[1]["content"]
    # 默认无外部模板时 system 用内置指令
    assert msgs[0]["content"] == prompts.DEFAULT_SYSTEM_TEMPLATE


def test_injection_guard_wraps_context_as_data():
    msgs = prompts.build_messages("q", "正文内容", [], {})
    user = msgs[1]["content"]
    assert prompts.DATA_OPEN in user and prompts.DATA_CLOSE in user


def test_injection_guard_can_be_disabled():
    cfg = {"prompt": {"wrap_context_as_data": False}}
    msgs = prompts.build_messages("q", "正文内容", [], cfg)
    assert prompts.DATA_OPEN not in msgs[1]["content"]


def test_external_system_template_override(tmp_path: Path):
    tpl = tmp_path / "sys.txt"
    tpl.write_text("自定义系统指令：只回答财务问题。", encoding="utf-8")
    cfg = {"prompt": {"system_template_path": str(tpl)}}
    msgs = prompts.build_messages("q", "c", [], cfg)
    assert msgs[0]["content"] == "自定义系统指令：只回答财务问题。"


def test_missing_template_falls_back_to_default(tmp_path: Path):
    cfg = {"prompt": {"system_template_path": str(tmp_path / "不存在的模板.txt")}}
    msgs = prompts.build_messages("q", "c", [], cfg)
    # 模板读不到不崩掉，退回内置默认
    assert msgs[0]["content"] == prompts.DEFAULT_SYSTEM_TEMPLATE
