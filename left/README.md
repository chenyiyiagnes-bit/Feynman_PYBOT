# Feynman-QBot v2.0 — 分布式量子干涉与形式化科研智能体

## 快速启动（本地模式）
1. `pip install -r requirements.txt`
2. `cp .env.example .env`（可选填入 `DEEPSEEK_API_KEY`，不填亦可直接离线仿真）
3. 终端 1 启动总线：`uvicorn bus.main:app --reload`
4. 终端 2 启动大屏：`streamlit run frontend/dashboard.py`

## 快速启动（Docker 一键含 Lean 4 沙盒）
`docker compose up --build -d`
- 监控大屏：http://localhost:8501
- 总线接口：http://localhost:8000/docs

## 接口
- `POST /solve`：双向波函数对撞 + Map-Reduce 盲审；同一问题默认命中结果缓存（`cache_hit` / `cache_age_sec`），`force_refresh: true` 可强制重算；响应含本轮 `llm_calls` 与累计 `llm_budget`。
- `POST /solve_beam`：驻相剪枝的多轮束搜索。
- `POST /grover`：Grover 搜索插件（含目标态与演化步）。
- `POST /movable_type/quantize`：把一段中文语料铸成三槽字库（主语/动作/归宿字粒，含振幅、相位、标签），在线可借 LLM 抽词，离线用纯规则分词；结果按语料哈希缓存，响应含 `cache_hit` 与 `llm_calls`。
- `POST /movable_type/print`：纯本地相位排版，毫秒级。按 `bank_id` + 观测者心境偏振角 + `nonce` 确定性采样一条坍缩印句；套话字粒取负振幅，任何心境下都进不了 Top-1。
- `GET /movable_type/banks`：已缓存的字库 `bank_id` 列表。
- `GET /feynman/vault`：费曼心智字库概况（来源语料、四槽字粒数、实测槽位偏移、句法骨架、共现对数）；未铸字时返回 503 并提示运行 `python plugins/feynman_vault.py`。
- `GET /feynman/operators`：五个心智算子的预设权重与中文标签，前端据此画滑杆与雷达图。
- `POST /feynman/print`：四槽位（起手式×实体×动作×可观测后果）语体相位排版。字粒全部来自本地费曼语料并带出处；`style_deg` 0–120° 调语体（讲义/对话/演讲）；术语堆叠否决、跨域纠缠加成、语体相干门、语料句法骨架；`operators` 五个 0–2 权重（微观拆解/跨界类比/极端边界/时间反演/反直觉，默认全 0 = 基线）；响应含 `purity`（纯净度）、`killed`（被术语枪毙的最高权重路径）、`skeleton`、逐路径 `operator_scores`/`badges`；话题下无可用字粒时返回留白字粒而非编句子；同 `bank_id+style_deg+nonce` 确定性复现。
- `GET /stats`：运行模式、LLM 记账、缓存条目、Lean 工具链状态。
- `GET /lean/status`：Lean 4 探测（路径、版本、lake 项目模式、安装提示）。
- `POST /cache/clear`、`POST /counters/reset`：清缓存 / 重置调用计数。

## 缓存与预算（`.env`）
- `QBOT_CACHE_TTL_SEC`：缓存有效期秒数，`0` 关闭；`QBOT_CACHE_PATH` 默认 `data/solve_cache.json`。
- `QBOT_MAX_LLM_CALLS`：进程累计 LLM 调用上限；`QBOT_MAX_LLM_CALLS_PER_SOLVE`：单次请求上限。留空或负数 = 不限；超限自动降级到离线仿真并在响应中如实报告。
- `QBOT_LEAN_PROJECT`：指向 lake 项目目录时，Lean 验证改走 `lake env lean`（可用 Mathlib 战术）。

## Lean 4 接入
未安装时验证器返回 180° 并在 `critical_flaw` 中给出 elan 引导命令，不伪造通过：

```sh
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable
lean --version
```

## 运行自动化测试套件
- 极端多数派幻觉压力测试：`PYTHONPATH=. python tests/test_majority_illusion.py`
- 第二笔记本触发回归：`PYTHONPATH=. python tests/test_second_notebook.py`
- 结果缓存与调用预算：`PYTHONPATH=. python tests/test_solve_cache.py`
- 适应性改造（补救排序/确定性归并/离线诚实降级）：`PYTHONPATH=. python tests/test_adaptive_fixes.py`
- 真实 Lean 沙盒：`PYTHONPATH=. python tests/test_lean_sandbox_real.py`
- 三层悖论基准与三项指标（每层 4 题）：`PYTHONPATH=. python tests/benchmark_paradoxes.py`（`QBOT_BENCHMARK_OFFLINE=1` 强制离线）
- 量子电路等价性 Lean 验证：`PYTHONPATH=. python tests/benchmark_quantum_gates.py`
- 智能合约漏洞攻防与免疫：`PYTHONPATH=. python tests/test_smart_contract_audit.py`
- 共享引理 Wiki 接力实验：`PYTHONPATH=. python tests/test_wiki_relay.py`
- 量子活字印刷机（确定性/心境敏感性/套话否决/API 契约）：`PYTHONPATH=. python tests/test_quantum_movable_type.py`
- 费曼心智活字（出处可查/确定性/语体敏感/术语闸门/留白/延迟）：`PYTHONPATH=. python tests/test_feynman_typesetter.py`

## 费曼心智字库（本地语料铸字）
字粒来自本机费曼语料（默认 `~/Desktop/费曼物理_文字版/corpus/cpt_zh.jsonl` 中费曼署名作品），
与 `~/Projects/feynman-open/knowledge/` 的品味判据；讣告、悼念、身后评价类段落一律剔除——
字粒要能当“他本人会说的话”用。铸字一次约一分钟，产物 `data/feynman_style_vault.json`：

- 每颗粒子带 `source｜section` 出处、频次、语体相位与振幅；
- `slot_offsets` 是逐句量出的槽位相位偏移（不再手填 −12°/+20°，实测结果接近 0°）；
- `skeletons` 是从语料挖出的 7 类句式框，每条带真实出处与命中频次；
- `cooccurrence` 是字粒在同句 ±5 字窗口的共现表，供“反直觉”算子判断“陌生但有据”；
- 装了 `jieba` 时用词性（`n*` / `v*`）净化实体与动作槽，未装则只用后缀规则自动降级。

```sh
pip install jieba        # 可选：词性净化更准；不装也能铸字
python plugins/feynman_vault.py
```

路径可用 `QBOT_FEYNMAN_CORPUS` / `QBOT_FEYNMAN_KNOWLEDGE` / `QBOT_FEYNMAN_VAULT` 覆盖。

语料行格式自动识别两种：旧格式 `{"source","section","text"}`，以及审计库的会话格式
`{"messages":[系统,用户,助手],"meta":{...}}`。会话格式只取 assistant 正文作费曼的话
（按 `\n` 拼接多轮），来源取 `meta.work_name`、段落取 `meta.span_id`，且只收
`meta.lang == "zh"` 的行；噪声、去世词、长度与拉丁比例过滤两种格式相同。

模块级实现与原文的对照、以及所有偏离原因，见 `docs/PARITY.md`。
