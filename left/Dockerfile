FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ELAN_HOME=/root/.elan \
    PATH="/root/.elan/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 静默安装 elan 与稳定版 Lean 4 编译器
# 注：原文安装 URL 在导出时被截断，此处补全为官方 elan 安装脚本地址
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf \
    | sh -s -- -y --default-toolchain stable \
    && lean --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000 8501
