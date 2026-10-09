# -*- coding: utf-8 -*-
"""第 1 批骨架自检脚本（临时，验证配置可解析、autodetect 可独立运行）。"""
import collections
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

ok = True

# 1) settings.example.yaml 可解析 + 三档参数齐全
s = yaml.safe_load((ROOT / "config" / "settings.example.yaml").read_text(encoding="utf-8"))
need_keys = {
    "lite": ("embedding_model", "embedding_dim", "device", "batch_size", "reranker", "qdrant"),
    "standard": ("embedding_model", "embedding_dim", "device", "batch_size", "reranker", "qdrant"),
    "performance": ("embedding_model", "embedding_dim", "device", "batch_size", "reranker", "qdrant"),
}
for p, keys in need_keys.items():
    missing = [k for k in keys if k not in s["profiles"][p]]
    if missing:
        ok = False
        print(f"[FAIL] profile={p} 缺字段 {missing}")
q = s["profiles"]
assert q["lite"]["embedding_model"] == "BAAI/bge-small-zh-v1.5"
assert q["lite"]["batch_size"] == 4 and q["lite"]["qdrant"]["hnsw_m"] == 8
assert q["standard"]["embedding_model"] == "BAAI/bge-m3"
assert q["standard"]["batch_size"] == 16 and q["standard"]["qdrant"]["hnsw_m"] == 16
assert q["performance"]["embedding_model"] == "BAAI/bge-m3"
assert q["performance"]["batch_size"] == 128 and q["performance"]["qdrant"]["hnsw_m"] == 32
assert q["performance"]["qdrant"]["on_disk"] is False
r = s["retrieval"]
assert r["rrf_k"] == 60 and r["rrf_weights"] == {"exact": 1.0, "fuzzy": 0.6, "sparse": 1.0, "dense": 1.0}
assert r["fuzzy_score_cutoff"] == 75 and r["fuzzy_top_k"] == 10 and r["ngram_top_k"] == 20
assert s["chunking"]["chunk_size"] == 384 and s["chunking"]["chunk_overlap"] == 64
print("[OK] settings.example.yaml 三档 + 检索参数校验通过")

# 2) synonyms.yaml
syn = yaml.safe_load((ROOT / "config" / "synonyms.yaml").read_text(encoding="utf-8"))
groups = {k: v for k, v in syn.items() if isinstance(v, list)}
print(f"[INFO] 同义词组数 = {len(groups)}, 总词条数 = {len(groups) + sum(len(v) for v in groups.values())}")
bad = {k: v for k, v in groups.items() if not all(isinstance(x, str) and x for x in v)}
if bad:
    ok = False
    print(f"[FAIL] 同义词值非字符串: {list(bad)[:3]}")
if len(groups) < 50:
    ok = False
    print(f"[FAIL] 同义词组数不足 50: {len(groups)}")

# 3) stopwords 唯一性与数量
raw = (ROOT / "config" / "stopwords_zh.txt").read_text(encoding="utf-8").splitlines()
words = [w.strip() for w in raw if w.strip() and not w.strip().startswith("#")]
dups = [w for w, c in collections.Counter(words).items() if c > 1]
print(f"[INFO] 停用词行数={len(words)} 去重后={len(set(words))} 重复={dups}")
if len(set(words)) < 200:
    ok = False
    print(f"[FAIL] 停用词去重后不足 200: {len(set(words))}")
for must in ("的", "了", "是", "在", "和", "与", "及", "或", "有", "被", "把", "让", "使", "对",
             "为", "以", "从", "到", "由", "等", "这", "那", "一个", "我们", "他们", "可以",
             "如果", "因为", "所以"):
    if must not in set(words):
        ok = False
        print(f"[FAIL] 停用词缺失必备项: {must}")

# 4) autodetect 独立运行（三种入口）
for args in (["--help"], ["-q"]):
    p = subprocess.run([sys.executable, str(ROOT / "config" / "autodetect.py"), *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        ok = False
        print(f"[FAIL] autodetect {' '.join(args)} 退出码={p.returncode}\n{p.stderr[-800:]}")
    else:
        print(f"[OK] autodetect {' '.join(args)} -> {p.stdout.strip()[:120]}")

# 5) 从 example 加载（验证 auto 覆盖 + sparse 权重归一化）
sys.path.insert(0, str(ROOT))
from config.autodetect import detect_hardware, load_settings  # noqa: E402

print("[INFO] hardware =", detect_hardware())
st = load_settings(ROOT / "config" / "settings.example.yaml")
print(f"[INFO] 探测 profile={st['profile']} auto={st['profile_auto']} source={st['settings_source']}")
print(f"[INFO] active.embedding={st['active'].get('embedding_model')} dim={st['active'].get('embedding_dim')} "
      f"device={st['active'].get('device')} batch={st['active'].get('batch_size')}")
print(f"[INFO] rrf_weights={st['retrieval']['rrf_weights']} sparse_available={st['retrieval']['sparse_available']}")
total = sum(st["retrieval"]["rrf_weights"].values())
# 权重总量必须守恒（原 4 路之和 = 3.6），sparse 不可用时只是重新分配
if abs(total - 3.6) > 1e-2:
    ok = False
    print(f"[FAIL] RRF 权重未守恒: {total}（应为 3.6）")
if st["retrieval"]["sparse_available"] is False and "sparse" in st["retrieval"]["rrf_weights"]:
    ok = False
    print("[FAIL] sparse 不可用但权重里仍有 sparse 路")

# 6) 显式指定 profile（环境变量覆盖 auto）
import os

for prof, expect in (("performance", False), ("standard", True)):
    os.environ["RAG_ZH_PROFILE"] = prof
    st2 = load_settings(ROOT / "config" / "settings.example.yaml")
    if st2["profile"] != prof:
        ok = False
        print(f"[FAIL] RAG_ZH_PROFILE={prof} 未生效，实际 {st2['profile']}")
    if st2["active"]["qdrant"]["on_disk"] is not expect:
        ok = False
        print(f"[FAIL] {prof} on_disk 应为 {expect}")
    if abs(sum(st2["retrieval"]["rrf_weights"].values()) - 3.6) > 1e-6:
        ok = False
        print(f"[FAIL] {prof} 四路权重应原样保留 3.6，实际 {st2['retrieval']['rrf_weights']}")
    if st2["retrieval"]["rrf_weights"].get("fuzzy") != 0.6:
        ok = False
        print(f"[FAIL] {prof} fuzzy 权重被误改: {st2['retrieval']['rrf_weights']}")
del os.environ["RAG_ZH_PROFILE"]
print("[OK] 显式 profile 覆盖 + 四路权重保留校验通过")

# 7) gitignore：样例文档必须能进仓库，真实数据必须被忽略
if (ROOT / "samples").is_dir() or True:
    (ROOT / "samples").mkdir(exist_ok=True)
    (ROOT / "samples" / "demo.pdf").write_bytes(b"%PDF-1.4 demo")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "secret.pdf").write_bytes(b"%PDF-1.4 secret")
    probes = {
        "samples/demo.pdf": False,   # 期望：不被忽略（可提交）
        "data/secret.pdf": True,
        "storage/qdrant/x": True,
        "config/settings.yaml": True,
        "config/settings.example.yaml": False,
        "config/stopwords_zh.txt": False,
        "models/bge-m3.bin": True,
        "chunks/part-0.parquet": True,
    }
    r = subprocess.run(["git", "check-ignore", "-q", "--stdin"], capture_output=True, text=True,
                       input="\n".join(probes), cwd=str(ROOT))
    if r.returncode == 128:  # 还没 git init，跳过
        print("[SKIP] 未初始化 git 仓库，跳过 .gitignore 验证")
    else:
        ignored = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
        for path, should_ignore in probes.items():
            actually = path in ignored
            if actually != should_ignore:
                ok = False
                print(f"[FAIL] gitignore 判定异常: {path} 期望忽略={should_ignore} 实际忽略={actually}")
        if ok:
            print("[OK] .gitignore 判定全部正确（samples 放行，数据/索引/模型拦截）")

print("\nRESULT:", "ALL PASS" if ok else "HAS FAILURE")
sys.exit(0 if ok else 1)

