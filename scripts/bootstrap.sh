#!/usr/bin/env bash
# =============================================================================
# rag-zh 一键初始化脚本
#
# 用法：
#   bash scripts/bootstrap.sh              # 完整流程（下载模型 + 起服务 + 建索引）
#   SKIP_MODELS=1 bash scripts/bootstrap.sh   # 跳过模型下载（已下过）
#   SKIP_DOCKER=1 bash scripts/bootstrap.sh   # 跳过 docker（Qdrant/MinIO 已在跑）
#
# 前置：
#   - Python 3.10+，已激活虚拟环境
#   - Docker + Docker Compose（用于拉起 Qdrant/MinIO）
#   - 网络可达 huggingface.co（下载模型权重）
#
# 失败即停（set -e）；每步 echo 进度。
# =============================================================================
set -euo pipefail

echo "=============================================="
echo "[bootstrap] rag-zh 环境初始化"
echo "=============================================="

# ---------------------------------------------------------------------------
echo ""
echo "[1/7] 安装项目依赖（含 dev）..."
echo "----------------------------------------------"
pip install -e ".[dev]"
echo "[1/7] ✓ 依赖安装完成"

# ---------------------------------------------------------------------------
if [[ "${SKIP_MODELS:-0}" != "1" ]]; then
  echo ""
  echo "[2/7] 下载 BGE-M3 embedding 模型（dense+sparse）..."
  echo "----------------------------------------------"
  huggingface-cli download BAAI/bge-m3
  echo "[2/7] ✓ BGE-M3 下载完成"

  echo ""
  echo "[3/7] 下载 BGE-reranker-v2-m3 重排模型..."
  echo "----------------------------------------------"
  huggingface-cli download BAAI/bge-reranker-v2-m3
  echo "[3/7] ✓ reranker 下载完成"
else
  echo "[2-3/7] 跳过模型下载（SKIP_MODELS=1）"
fi

# ---------------------------------------------------------------------------
if [[ "${SKIP_DOCKER:-0}" != "1" ]]; then
  echo ""
  echo "[4/7] 启动 Qdrant + MinIO（docker compose）..."
  echo "----------------------------------------------"
  docker compose up -d qdrant minio
  echo "等待 Qdrant 就绪..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/collections > /dev/null 2>&1; then
      echo "  Qdrant 已就绪"
      break
    fi
    sleep 1
  done
  echo "[4/7] ✓ 容器启动完成"
else
  echo "[4/7] 跳过 Docker（SKIP_DOCKER=1）"
fi

# ---------------------------------------------------------------------------
echo ""
echo "[5/7] 初始化 Qdrant collection（dense + sparse 命名向量）..."
echo "----------------------------------------------"
python scripts/init_qdrant.py
echo "[5/7] ✓ Collection 就绪"

# ---------------------------------------------------------------------------
echo ""
echo "[6/7] 入库 samples/ 目录（完整流水线 parse→classify→chunk→embed→index）..."
echo "----------------------------------------------"
python scripts/build_index.py --input samples/
echo "[6/7] ✓ 入库完成"

# ---------------------------------------------------------------------------
echo ""
echo "[7/7] 构建词表（vocab + ngram + synonyms）..."
echo "----------------------------------------------"
python -m index.vocab_builder
echo "[7/7] ✓ 词表构建完成"

# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "[bootstrap] 全部完成！后续可："
echo "  - 启动 MCP（stdio）：python -m mcp_server.server"
echo "  - 启动 MCP（sse） ：python -m mcp_server.server --sse"
echo "  - 运行评测        ：python eval/run_eval.py"
echo "=============================================="
