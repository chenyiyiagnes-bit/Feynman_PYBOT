from pathlib import Path

PROJECT_NAME = "feynman-qbot-v2"

# 一键安装器：从当前源码树打包出一个可独立运行的干净项目目录。
# 原方案将全部源码内嵌于 FILES 字典；此处等价地以「相对路径 -> 读源文件内容」构建，
# 避免二次转义，同时保持与原文 setup_feynman_qbot_v2.py 相同的解包行为与启动提示。

SOURCE_ROOT = Path(__file__).resolve().parent

FILES = {
    "requirements.txt": """fastapi>=0.115.0
uvicorn>=0.32.0
httpx>=0.27.2
pydantic>=2.9.0
streamlit>=1.39.0
plotly>=5.24.1
pandas>=2.2.3
numpy>=2.1.2
python-dotenv>=1.0.1
jieba>=0.42.1
""",
    ".env.example": """# 填入 DeepSeek API Key；留空则自动启用内置高精度物理/形式化离线仿真内核
DEEPSEEK_API_KEY=

# 模型分层配置（并发探索与 Map 审计用 deepseek-chat，Reduce 聚类与裁决同样用 deepseek-chat）
WORKER_MODEL=deepseek-chat
AUDITOR_MODEL=deepseek-chat

# DeepSeek 兼容接口地址（默认官方；如走代理/中转可覆盖）
DEEPSEEK_BASE_URL=https://api.deepseek.com

# /solve 结果缓存：秒数（0 关闭）与落盘路径
QBOT_CACHE_TTL_SEC=3600
# QBOT_CACHE_PATH=data/solve_cache.json

# LLM 调用预算：进程累计上限 / 单次请求上限（留空或负数 = 不限）
# QBOT_MAX_LLM_CALLS=200
# QBOT_MAX_LLM_CALLS_PER_SOLVE=12

# 需要 Mathlib 战术时，指向一个 lake 项目目录，Lean 验证改走 `lake env lean`
# QBOT_LEAN_PROJECT=/path/to/lake/project
""",
}

# 需打包进项目的源码文件（相对路径），从当前复刻目录逐字读取
PACKAGED_SOURCES = [
    "README.md",
    "data/feynman_style_vault.json",
    "docs/PARITY.md",
    "Dockerfile",
    "docker-compose.yml",
    "shared/__init__.py",
    "shared/protocol.py",
    "shared/phase_mapper.py",
    "shared/deepseek_client.py",
    "shared/llm_ledger.py",
    "shared/solve_cache.py",
    "workers/__init__.py",
    "workers/feynman_workers.py",
    "evaluator/__init__.py",
    "evaluator/mapreduce_auditor.py",
    "skeptic/__init__.py",
    "skeptic/oracle.py",
    "skeptic/lean_verifier.py",
    "skeptic/lemma_wiki.py",
    "plugins/__init__.py",
    "plugins/grover_engine.py",
    "plugins/quantum_tokenizer.py",
    "plugins/movable_type_engine.py",
    "plugins/feynman_vault.py",
    "plugins/feynman_typesetter.py",
    "bus/__init__.py",
    "bus/beam_evolver.py",
    "bus/main.py",
    "frontend/dashboard.py",
    "tests/__init__.py",
    "tests/test_majority_illusion.py",
    "tests/test_second_notebook.py",
    "tests/test_solve_cache.py",
    "tests/test_lean_sandbox_real.py",
    "tests/test_adaptive_fixes.py",
    "tests/test_quantum_movable_type.py",
    "tests/test_feynman_typesetter.py",
    "tests/benchmark_paradoxes.py",
    "tests/benchmark_quantum_gates.py",
    "tests/test_wiki_relay.py",
    "tests/test_smart_contract_audit.py",
]

for rel in PACKAGED_SOURCES:
    src = SOURCE_ROOT / rel
    if src.is_file():
        FILES[rel] = src.read_text(encoding="utf-8")


def main():
    root = Path(PROJECT_NAME)
    for rel_path, content in FILES.items():
        full_path = root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"✅ Feynman-QBot v2.0 完整项目已生成至：{root.resolve()}")
    print("\n👉 快速启动方式 A（本地 Python 环境）：")
    print(f" cd {PROJECT_NAME} && pip install -r requirements.txt && cp .env.example .env")
    print(" 终端1：uvicorn bus.main:app --reload")
    print(" 终端2：streamlit run frontend/dashboard.py")
    print("\n👉 快速启动方式 B（Docker 一键拉起含 Lean 4 编译器）：")
    print(f" cd {PROJECT_NAME} && cp .env.example .env && docker compose up --build -d")


if __name__ == "__main__":
    main()
