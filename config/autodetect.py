# -*- coding: utf-8 -*-
"""
rag-zh 硬件自动探测与配置加载。

设计目标：机器配置未知，必须能自动选定 lite / standard / performance 档，
并把该档参数合并进最终 settings dict，供上层（解析 / 入库 / 检索）统一读取。

可直接独立运行（不依赖项目其他模块，torch 缺失也不报错）：

    python config/autodetect.py            # 打印探测结果 + 合并后的配置摘要
    python config/autodetect.py --json     # 输出 JSON，供脚本消费
    python config/autodetect.py --write    # 探测并写出 config/settings.yaml（若不存在）
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import psutil
import yaml

# ------------------------------------------------------------------ 路径常量
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"           # 真实配置（gitignore）
SETTINGS_EXAMPLE = CONFIG_DIR / "settings.example.yaml"  # 模板（进仓库）

VALID_PROFILES = ("lite", "standard", "performance")

# ${VAR:-default} 占位符解析
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?}")


# ================================================================== 工具函数
def _to_gb(n_bytes: float) -> float:
    """字节转 GiB，保留一位小数。"""
    return round(float(n_bytes) / (1024**3), 1)


def _total_memory_gb() -> float:
    """用 psutil 读物理内存总量（GiB）。异常时返回 0.0，交给规则降级为 lite。"""
    try:
        return _to_gb(psutil.virtual_memory().total)
    except Exception as exc:  # pragma: no cover - 极端环境兜底
        print(f"[autodetect] 内存探测失败，按 0GB 处理: {exc}", file=sys.stderr)
        return 0.0


def _has_gpu() -> bool:
    """
    探测 GPU。优先 torch.cuda.is_available()；
    torch 未安装时退化为 nvidia-smi 是否存在（不引入新依赖，仅 shutil.which）。
    任何异常都吞掉并返回 False —— 本模块必须能独立跑通。
    """
    try:
        import torch  # 延迟导入：无 torch 时也能探测内存/磁盘

        if torch.cuda.is_available():
            return True
        # torch 装了但 cuda 不可用（CPU 版 wheel），继续走下面的兜底
    except Exception:
        pass

    return shutil.which("nvidia-smi") is not None


def _gpu_name() -> str | None:
    """取一块 GPU 名称用于日志展示，失败返回 None。"""
    try:
        import torch  # 延迟导入：无 torch 时退化成 nvidia-smi

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first = out.stdout.strip().splitlines()
        return first[0] if out.returncode == 0 and first else None
    except Exception:
        return None


def _read_rotational_flags() -> list[int]:
    """
    Linux：读 /sys/block/*/queue/rotational。
    0 = 非旋转（SSD/NVMe），1 = 机械盘。
    跳过 ram/loop/sr/zram 等虚拟设备，避免把内存盘当磁盘。
    """
    flags: list[int] = []
    for path in glob.glob("/sys/block/*/queue/rotational"):
        dev = Path(path).parts[2] if len(Path(path).parts) > 2 else ""
        if dev.startswith(("ram", "loop", "sr", "zram", "dm-")):
            continue
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
            flags.append(int(raw))
        except (OSError, ValueError):
            continue
    return flags


def _detect_windows_media_types() -> list[str]:
    """
    Windows：Get-PhysicalDisk 返回 MediaType（SSD / HDD / Unspecified）。
    查询较慢（约 1~3s），整体超时 8s，失败返回空列表。
    """
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-PhysicalDisk | Select-Object -ExpandProperty MediaType",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if out.returncode != 0:
            return []
        return [x.strip().upper() for x in out.stdout.splitlines() if x.strip()]
    except Exception:
        return []


@lru_cache(maxsize=1)
def detect_disk_type() -> tuple[str, bool]:
    """
    返回 (disk_type, confident)。

    disk_type: 'ssd' | 'hdd' | 'unknown'
    confident: 是否拿到了确证数据（Linux sysfs 或 Windows MediaType）

    unknown 时上层规则会「按 SSD 处理」并打印提示——因为无法在 Windows/macOS
    上无特权读到 rotational 位，宁可假设较快的盘，但会把提示告知用户可手工覆盖。
    """
    if sys.platform.startswith("linux"):
        flags = _read_rotational_flags()
        if flags:
            return ("ssd", True) if min(flags) == 0 else ("hdd", True)
        return ("unknown", False)

    if os.name == "nt":
        types = _detect_windows_media_types()
        if types:
            if any("SSD" in t for t in types):
                return ("ssd", True)
            if any("HDD" in t for t in types):
                return ("hdd", True)
        return ("unknown", False)

    return ("unknown", False)


# ================================================================== 核心逻辑
def detect_hardware() -> dict[str, Any]:
    """汇总一次硬件画像，供 detect_profile() 与日志使用。"""
    disk_type, confident = detect_disk_type()
    return {
        "total_memory_gb": _total_memory_gb(),
        "available_memory_gb": _to_gb(psutil.virtual_memory().available),
        "cpu_count": psutil.cpu_count(logical=True) or 1,
        "has_gpu": _has_gpu(),
        "gpu_name": _gpu_name(),
        "disk_type": disk_type,
        "disk_detect_confident": confident,
        "platform": sys.platform,
    }


def detect_profile() -> str:
    """
    按全局约束的三档规则返回 profile 名。

    1. GPU 且内存 >= 32GB                            -> performance
    2. 内存 >= 16GB 且 SSD（磁盘类型未知时按 SSD）    -> standard
    3. 其余（含探测失败）                             -> lite

    注：CPU 核数不足 4 核时强制降级为 lite，因为 multiprocessing.Pool
    与 Qdrant 并发写入会互相抢 CPU，standard 档反而更慢。
    """
    hw = detect_hardware()
    mem = hw["total_memory_gb"]
    is_ssd = hw["disk_type"] in ("ssd", "unknown")  # unknown 从宽处理

    if hw["has_gpu"] and mem >= 32:
        return "performance"
    if mem >= 16 and is_ssd and hw["cpu_count"] >= 4:
        return "standard"
    return "lite"


def explain_detection() -> str:
    """生成人类可读的探测说明，解释为什么落到这一档。"""
    hw = detect_hardware()
    profile = detect_profile()
    gpu_desc = f"有 GPU（{hw['gpu_name'] or '未知型号'}）" if hw["has_gpu"] else "无 GPU"
    disk_desc = hw["disk_type"] if hw["disk_detect_confident"] else f"{hw['disk_type']}（未确证，按 ssd 处理）"
    lines = [
        "===== rag-zh 硬件探测 =====",
        f"  物理内存   : {hw['total_memory_gb']} GB（可用 {hw['available_memory_gb']} GB）",
        f"  CPU 逻辑核 : {hw['cpu_count']}",
        f"  GPU        : {gpu_desc}",
        f"  磁盘类型   : {disk_desc}",
        f"  运行平台   : {hw['platform']}",
        f"  判定档位   : {profile}",
        "  规则        : GPU+32G -> performance; 16G+SSD -> standard; 否则 lite",
        f"  手工覆盖   : 编辑 {SETTINGS_FILE} 把 profile 改为 {VALID_PROFILES} 之一",
    ]
    return "\n".join(lines)


# ------------------------------------------------------- settings 读取与合并
def _expand_env_in_value(value: Any) -> Any:
    """递归解析字符串里的 ${VAR:-default} 占位符（环境变量优先，否则用默认值）。"""
    if isinstance(value, str):
        def _sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group("name"), m.group("default") or "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env_in_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_in_value(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    """安全加载 YAML，文件不存在返回空 dict。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是 mapping: {path}")
    return data


def _resolve_settings_path(settings_path: str | os.PathLike[str] | None) -> Path:
    """
    确定读哪个 YAML：
      显式传入 > config/settings.yaml > config/settings.example.yaml
    这样仓库刚 clone 下来、还没复制 settings.yaml 时也能跑通。
    """
    if settings_path:
        return Path(settings_path).expanduser().resolve()
    if SETTINGS_FILE.is_file():
        return SETTINGS_FILE
    return SETTINGS_EXAMPLE


def _pick_profile(raw: dict[str, Any]) -> tuple[str, bool]:
    """
    返回 (最终 profile, 是否为自动探测结果)。
    优先级：环境变量 RAG_ZH_PROFILE > settings.profile(auto 则探测) > lite。
    """
    env_profile = os.environ.get("RAG_ZH_PROFILE", "").strip().lower()
    if env_profile and env_profile != "auto":
        if env_profile not in VALID_PROFILES:
            raise ValueError(f"RAG_ZH_PROFILE 非法值: {env_profile}，可选 {VALID_PROFILES}")
        return env_profile, True

    declared = str(raw.get("profile", "auto")).strip().lower()
    if declared in VALID_PROFILES:
        return declared, False
    return detect_profile(), True


def _resolve_device(profile_cfg: dict[str, Any]) -> str:
    """profile 里 device=auto 时，按实际 GPU 情况填 cuda / cpu。"""
    device = str(profile_cfg.get("device", "auto")).lower()
    if device != "auto":
        return device
    return "cuda" if _has_gpu() else "cpu"


def load_settings(settings_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """
    加载并合并配置。

    返回的 dict 结构：
      {
        "profile": "standard",
        "profile_auto": true,          # 是否由 autodetect 决定
        "hardware": {...},             # 探测原始数据，便于排查
        "active": {embedding_model, embedding_dim, device, batch_size,
                   qdrant{...}, reranker{...}, supports_sparse, ...},
        "chunking": {...}, "retrieval": {...}, "embedding": {...},
        "qdrant": {...连接信息...}, "minio": {...}, "sqlite": {...},
        "parsing": {...}, "lexicon": {...}, "mcp": {...},
        "profiles": {...三档全量，供切换时不重读文件...}
      }

    关键：上层代码只读 active，不需要写 if profile == 'lite' 之类的分支。
    """
    path = _resolve_settings_path(settings_path)
    raw = _expand_env_in_value(_load_yaml(path))

    if not raw:
        # 兜底：连模板都读不到时给出一份最小可用配置，保证独立运行不报错
        raw = {"profile": "auto", "profiles": {p: {} for p in VALID_PROFILES}}

    profiles: dict[str, Any] = raw.get("profiles") or {}
    profile, auto = _pick_profile(raw)
    profile_cfg = copy.deepcopy(profiles.get(profile) or {})

    profile_cfg["device"] = _resolve_device(profile_cfg)
    # CPU 上强开 fp16 会直接报错，这里做一次静默修正
    if profile_cfg.get("device") == "cpu" and profile_cfg.get("use_fp16"):
        profile_cfg["use_fp16"] = False
        if profile_cfg.get("dtype") == "fp16":
            profile_cfg["dtype"] = "fp32"
    profile_cfg["profile"] = profile

    settings: dict[str, Any] = {
        "profile": profile,
        "profile_auto": auto,
        "settings_source": str(path),
        "hardware": detect_hardware(),
        "active": profile_cfg,
    }

    # 除 profiles 之外的所有顶层段原样带上（chunking/retrieval/qdrant/minio/...）
    for key, value in raw.items():
        if key in ("profile", "profiles"):
            continue
        settings[key] = value

    # 缺省段补齐，避免后续批次因模板删改而 KeyError
    settings.setdefault("chunking", {})
    settings.setdefault("retrieval", {})
    settings.setdefault("embedding", {})
    settings.setdefault("qdrant", {})
    settings.setdefault("minio", {})
    settings.setdefault("sqlite", {})
    settings.setdefault("parsing", {})
    settings.setdefault("lexicon", {"synonyms_path": "config/synonyms.yaml",
                                    "stopwords_path": "config/stopwords_zh.txt"})
    settings.setdefault("mcp", {})

    # RRF 权重与四路开关：sparse 不可用时归一化重分配（不补零向量）
    retrieval = settings["retrieval"]
    weights = dict(retrieval.get("rrf_weights") or {})
    weights = {k: float(v) for k, v in weights.items()}
    if weights and not profile_cfg.get("supports_sparse", True):
        sparse_w = weights.pop("sparse", 0.0)
        rest = sum(weights.values()) or 1.0
        # 把 sparse 的权重按剩余权重比例分给 exact/dense/fuzzy
        weights = {k: round(v + sparse_w * (v / rest), 4) for k, v in weights.items()}
        retrieval["sparse_available"] = False
    else:
        retrieval["sparse_available"] = bool(profile_cfg.get("supports_sparse", True))
    retrieval["rrf_weights"] = weights
    return settings


def write_settings_from_example(target: str | os.PathLike[str] | None = None) -> Path:
    """
    把 settings.example.yaml 复制为 settings.yaml，并把 profile 写成探测结果。
    仅用于首次初始化（--write），已存在则不覆盖，防止误改用户配置。
    """
    dst = Path(target) if target else SETTINGS_FILE
    if dst.exists():
        return dst
    text = SETTINGS_EXAMPLE.read_text(encoding="utf-8")
    text = text.replace("profile: auto", f"profile: {detect_profile()}", 1)
    dst.write_text(text, encoding="utf-8")
    return dst


# ==================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    """命令行入口：打印探测结果 / 导出 JSON / 初始化 settings.yaml。"""
    parser = argparse.ArgumentParser(description="rag-zh profile 自动探测")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出硬件与最终 profile")
    parser.add_argument("--write", action="store_true", help="按探测结果生成 config/settings.yaml")
    parser.add_argument("--settings", default=None, help="指定要读取的 yaml 路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="只输出 profile 名，便于 shell 取值")
    args = parser.parse_args(argv)

    if args.write:
        path = write_settings_from_example()
        print(f"[autodetect] 已写出配置文件: {path}")

    if args.quiet:
        print(detect_profile())
        return 0

    if args.json:
        settings = load_settings(args.settings)
        payload = {
            "hardware": settings["hardware"],
            "profile": settings["profile"],
            "profile_auto": settings["profile_auto"],
            "settings_source": settings["settings_source"],
            "active": settings["active"],
            "rrf_weights": settings["retrieval"].get("rrf_weights"),
            "sparse_available": settings["retrieval"].get("sparse_available"),
        }
        # numpy/torch 等对象不可 JSON 序列化时用 str 兜底，保证 CLI 不炸
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    print(explain_detection())
    settings = load_settings(args.settings)
    active = settings["active"]
    print("\n===== 合并后的生效参数 =====")
    print(f"  配置来源      : {settings['settings_source']}")
    print(f"  embedding     : {active.get('embedding_model')} (dim={active.get('embedding_dim')})")
    print(f"  sparse 路     : {'可用' if active.get('supports_sparse') else '不可用（权重已重分配）'}")
    print(f"  device/batch  : {active.get('device')} / {active.get('batch_size')}")
    print(f"  reranker      : {active.get('reranker', {}).get('model')}")
    q = active.get("qdrant", {})
    print(f"  qdrant 索引   : m={q.get('hnsw_m')} efc={q.get('hnsw_ef_construct')} "
          f"on_disk={q.get('on_disk')} quant={q.get('quantization')}")
    print(f"  进程数        : chunk={active.get('chunk_workers')} index={active.get('index_workers')}")
    print(f"  rrf           : k={settings['retrieval'].get('rrf_k')} "
          f"weights={settings['retrieval'].get('rrf_weights')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
