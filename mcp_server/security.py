# -*- coding: utf-8 -*-
"""
MCP 对外面的安全边界（纯函数，无网络 / 无模型加载，可离线单测）。

只服务「远程不可信面」——即 MCP 工具入参；本地 CLI / scripts 的 --input 视为
受信操作者，不走这里（否则运维日常会被路径沙箱卡死）。

四类边界：
  1. confined_read_path：把 ingest / classify_image 的读取路径锁在允许的入库根内，
     拒绝 ../ 越界与「沙箱内软链指向沙箱外」的逃逸（resolve 已规范化软链）。
  2. presigned_expiry：预签名下载 URL 的有效期取配置值与上限的 min，默认收紧到
     不超过 security.max_presigned_seconds（缺省 300s），不再默认 1 小时。
  3. validate_minio_credentials：拒绝空凭据与已知弱默认口令（minioadmin 等），
     从代码层堵住「配置没改就带着默认口令上线签发票据」。
  4. 供 server.py 用的启动绑定门禁在 server 侧实现（这里只提供 token 读取无关的纯逻辑）。

日志红线：被拒事件只记元数据（路径 / 原因 / doc_id），绝不落文件内容或凭据。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from config.logging_setup import get_logger

logger = get_logger("rag_zh.security")

# 已知弱默认口令（小写比较）：命中即拒绝，避免带默认凭据上线
_WEAK_SECRETS = frozenset({"minioadmin", "admin", "password", "root", "123456"})


class PathNotAllowed(RuntimeError):
    """读取路径不在允许的入库根内（含 ../ 越界、软链逃逸、沙箱外绝对路径）。"""


class InsecureConfigError(RuntimeError):
    """安全相关配置不安全（如 MinIO 凭据缺失或为弱默认值），拒绝继续。"""


# ---------------------------------------------------------------- 路径沙箱
def _split_env_roots(raw: str) -> list[str]:
    """
    拆分环境变量里的多根列表。

    分隔符取 {os.pathsep, ';'}：POSIX 下 os.pathsep 是 ':'，Windows 下是 ';'。
    **不额外按 ':' 切**——否则 Windows 的 'C:\\data' 会被Drive 字母截断。
    """
    separators = {os.pathsep, ";"}
    pattern = "|".join(re.escape(sep) for sep in separators if sep)
    return [part for part in re.split(pattern, raw) if part and part.strip()]


def allowed_ingest_roots(settings: dict[str, Any]) -> list[Path]:
    """
    解析允许的入库根，优先级：
      环境变量 RAG_ZH_ALLOWED_INGEST_ROOTS
      > settings.security.ingest_roots（list）
      > 默认 [RAG_ZH_DATA_DIR 或 'data']
    每个根都 expanduser().resolve()，返回绝对路径列表。
    """
    raw_env = os.environ.get("RAG_ZH_ALLOWED_INGEST_ROOTS")
    if raw_env and raw_env.strip():
        candidates = _split_env_roots(raw_env)
    else:
        security = settings.get("security") or {}
        cfg_roots = security.get("ingest_roots") if isinstance(security, dict) else None
        if cfg_roots:
            candidates = [str(r) for r in cfg_roots if str(r).strip()]
        else:
            candidates = [os.environ.get("RAG_ZH_DATA_DIR") or "data"]
    return [Path(c).expanduser().resolve() for c in candidates]


def confined_read_path(raw: str | Path, base_dirs: list[Path]) -> Path:
    """
    把 raw 解析成绝对路径，并确认它落在某个 base_dir 内（含等于 base_dir 本身）。

    resolve() 会一并规范化 '..' 与软链，所以「沙箱内软链指向沙箱外」也会在
    这一步暴露成沙箱外的真实路径而被拒。越界抛 PathNotAllowed。
    """
    resolved = Path(raw).expanduser().resolve()
    for base in base_dirs:
        base_abs = Path(base).expanduser().resolve()
        if resolved == base_abs or resolved.is_relative_to(base_abs):
            return resolved
    logger.warning("blocked path outside ingest roots: %s", resolved)
    raise PathNotAllowed(f"路径不在允许的入库根内: {raw}")


# ---------------------------------------------------------------- 下载票据
def presigned_expiry(settings: dict[str, Any]) -> int:
    """
    预签名 URL 有效期 = min(minio.presigned_expire_seconds, security.max_presigned_seconds)。

    即便运维把 minio 段调回 3600，也会被 security 段的上限（缺省 300）压住。
    下限保护到 1 秒，避免配置成 0/负数导致签出即失效。
    """
    minio_cfg = settings.get("minio") or {}
    security = settings.get("security") or {}
    configured = int(minio_cfg.get("presigned_expire_seconds") or 3600)
    cap = int(security.get("max_presigned_seconds") or 300)
    return max(1, min(configured, cap))


def validate_minio_credentials(access_key: str, secret_key: str) -> None:
    """空凭据或命中弱默认口令黑名单 → InsecureConfigError；否则通过。"""
    if not access_key or not secret_key:
        raise InsecureConfigError("MinIO access_key / secret_key 缺失，拒绝签发下载链接")
    if access_key.strip().lower() in _WEAK_SECRETS or secret_key.strip().lower() in _WEAK_SECRETS:
        raise InsecureConfigError("MinIO 凭据为弱默认值，请改用环境变量注入强口令")
