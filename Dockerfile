# =============================================================================
# rag-zh —— 生产镜像（lite 档：仅核心运行时依赖）
#
# 设计要点（对应 ⑦ 部署工程化批次）：
#   - python:3.11-slim 基础镜像；多阶段构建，最终层不含编译产物 / pip 缓存。
#   - 非 root 运行（专用 app 用户，UID/GID 10001）。
#   - 只读配置：/app/config 由宿主 bind-mount 且挂 :ro（见 docker-compose.yml），
#     镜像内不把配置设为可写；日志文件目录不可写时 logging_setup 自动降级为 stderr。
#   - HEALTHCHECK：TCP 存活探针（默认探 8000，可用 MCP_PORT 覆写）。
#
# 依赖范围（本批取「仅轻量基础镜像」）：
#   只 `pip install .`（core 运行时依赖：fastmcp / qdrant-client / minio / 解析 /
#   jieba / rapidfuzz 等），**不装 models extras**（torch / FlagEmbedding / paddleocr
#   这类 GB 级重依赖）。缺重库时 dense/sparse 向量化与 OCR 工具在调用时按既有
#   「降级不崩溃」策略报错，服务仍可起来（warmup 为 best-effort）。
#   需要完整向量化/OCR 时，另建 models 目标（本批不做，见计划「不做」清单）。
#
# 构建 / 运行：
#   docker build -t rag-zh/mcp-server:latest .
#   docker run -d -e MCP_AUTH_TOKEN=<token> -p 127.0.0.1:8000:8000 rag-zh/mcp-server:latest
#   （SSE 绑 0.0.0.0 属对外地址，server 启动门禁要求 MCP_AUTH_TOKEN，否则退出 2）
# =============================================================================

# ---------------------------------------------------------------- 构建阶段
FROM python:3.11-slim AS builder

# 干净的 venv 前缀，便于整目录拷贝进运行阶段（shebang 路径两阶段保持一致）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"

WORKDIR /build
# 仅拷贝构建所需元数据 + 源码目录（大数据/产物目录已由 .dockerignore 排除在上下文外）
COPY . .
# 只装 core 依赖（无 extras）：[build-system] 会按需拉取 setuptools/wheel 到隔离环境
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------- 运行阶段
FROM python:3.11-slim

# compose 传入的档位占位；lite 阶段不消费（留作未来 models 目标的 hook）
ARG RAG_ZH_DEVICE=cpu

# 非 root 专用系统用户（--system + 显式 UID/GID 10001，便于宿主 bind-mount 授权对齐）
RUN groupadd --system app --gid 10001 \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

# 从构建阶段整份拷入已安装好依赖的 venv（应用代码作为命名空间包已随 `pip install .` 装入）
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin" \
    PYTHONUNBUFFERED=1

WORKDIR /app
# 可写运行目录（bind-mount 落点：向量 / 原文 / 中间产物 / 日志），属 app 用户
RUN mkdir -p /app/data /app/storage && chown -R app:app /app/data /app/storage
# /app/config 由 compose 以 :ro 挂载提供；镜像内不预置、不属 app（落实只读配置）

# 健康探针：确认 SSE 端口在监听（无 /health 路由，故做 TCP 存活检查）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, socket; s = socket.create_connection(('127.0.0.1', int(os.environ.get('MCP_PORT', '8000'))), 2); s.close()"

USER app
EXPOSE 8000

# 默认 SSE 启动；对外绑定所需的 token 由运行环境（compose 的 MCP_AUTH_TOKEN:?）强制注入
CMD ["python", "-m", "mcp_server.server", "--sse", "--host", "0.0.0.0", "--port", "8000"]
