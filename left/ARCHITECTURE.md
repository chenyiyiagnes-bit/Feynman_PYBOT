# 架构（自动生成）

由 `scripts/gen_architecture.py` 从运行中的对象现场读取生成，勿手改。
数据产物清单见 `data/README.md`；产品默认配置见 `bus/main.py` 的 `_product_defaults()`。

## 进程与职责

| 模块 | 职责 |
|---|---|
| `bus/main.py` | FastAPI 总线：路由、产品默认档、会话内存、进化写回、指标 |
| `plugins/feynman_typesetter.py` | 活字排版：槽位/相位/骨架组装，不生成新文字 |
| `plugins/paragraph_weaver.py` | 段落编织：一次 print 取池 + 三级重排（锚点→类比→收束） |
| `plugins/feynman_vault.py` | 字库加载与坏库回退 |
| `plugins/topology_builder.py` | 拓扑流形：团复形、洞检测、类比对、请求级重排 |
| `plugins/isomorphism_p21.py` | 同构基底（默认关）：句级 PPMI-SVD 字粒向量、OOV 投影、相位带宽对齐、相容乘子、诚实诊断 |
| `plugins/evolution_loop.py` | 反馈 → 扩展库边权更新（带跨任务锁、钳位） |
| `plugins/domain_expander.py` | 扩展库读写、EXT_LOCK、vault_id 溯源标记 |
| `shared/metrics.py` | 请求空白率 / 延迟滚动窗口 |
| `shared/solve_cache.py` | 解题缓存（TTL） |
| `shared/llm_ledger.py` | LLM 调用记账（默认关闭，见 .env 开关） |

## 路由（引自 app.routes）

| 路由 | 处理函数 | 说明 |
|---|---|---|
| `POST /feynman/digest` | `feynman_digest` | 只读感知：读一段文字，用现有段落编织器回一段读后感；默认不写任何东西。 |
| `POST /feynman/extend` | `feynman_extend` | 旁路爬虫入口。默认关闭；没开开关时直接 403，不偷偷联网。 |
| `POST /feynman/feedback` | `feynman_feedback` | 显式反馈：👍 +0.30 / 👎 −0.45，钳位 [0.15, 1.5]，跌破即剪枝。 |
| `GET /feynman/journal` | `feynman_journal` | 成长年轮：只读导出当前学到的搭配、骨架偏好、原型与会话痕迹。 |
| `GET /feynman/operators` | `feynman_operators` | 五个心智算子的预设与标签；前端据此画滑杆与雷达图。 |
| `POST /feynman/paragraph` | `feynman_paragraph` | 一段三句：现象锚点 → 92° 类比 → 15° 收束。只排版一次，重排三次。 |
| `POST /feynman/print` | `feynman_print` | 四槽位相位排版。纯本地、确定性；证据不足时返回留白字粒而不是编句子。 |
| `POST /feynman/session/reset` | `feynman_session_reset` | 真正清掉一个会话：量子记忆 + 隐式信号上一轮快照都清空。 |
| `GET /feynman/topology` | `feynman_topology` | 拓扑层状态：开关关闭或文件缺失时如实返回，不假装启用。 |
| `GET /feynman/vault` | `feynman_vault_status` | 字库概况：来源语料、段落/句子数、四槽字粒数、实测槽位偏移与骨架。 |
| `POST /cache/clear` | `clear_cache` |  |
| `POST /counters/reset` | `reset_counters` |  |
| `POST /grover` | `grover_search_endpoint` |  |
| `GET /lean/status` | `lean_status` |  |
| `GET /movable_type/banks` | `movable_type_banks` |  |
| `POST /movable_type/print` | `movable_type_print` | 固定字库 + 观测者心境 → 波函数坍缩印句（纯本地计算，毫秒级）。 |
| `POST /movable_type/quantize` | `movable_type_quantize` | 语料 → 三槽位量子字粒库（按语料哈希缓存，重复铸字不重算）。 |
| `POST /solve` | `solve_paradox` |  |
| `POST /solve_beam` | `solve_with_beam_pruning` |  |
| `GET /stats` | `stats` |  |

## 当前产品字库（Vault.stats()）

- bank_id：`feynman-e61eead9a209`
- 语料：data/p8_corpus_v2.jsonl；节 25165；句 153721
- 槽位字粒数：hook 60，entity 66，action 60，consequence 472
- 槽位相位偏置（度）：hook -3.9，entity -1.9，action 0.1，consequence 2.1
- 骨架 7 条；共现对 1970；纠缠对 80；术语 19

| 骨架 | 相位（度） | 频次 |
|---|---|---|
| declarative | 45.0 | 0 |
| conditional | 36.5 | 7348 |
| temporal | 47.6 | 2116 |
| causal | 34.7 | 13059 |
| contrast | 35.6 | 8068 |
| retro | 41.9 | 338 |
| analogy | 57.0 | 1593 |

## 当前拓扑图（engine.stats()）

- topo_hash：`48cdc9f6`
- 词 657；已建复形 343；三角按证据分级 tier2 720 / tier3 694（分级表可含未进复形列表的三角）；同节 11692；同段 154
- 边 1970，支撑≥2 的边 1314；类比对 172，alpha 0.85
- 引擎开关：请求级 `use_topology` 默认开（产品档），插件默认 `FEYNMAN_ENABLE_TOPOLOGY=0`（新进程缺省 0）

## 扩展库（学习产物）

- 文件：`data/domain_extensions.json`
- 内容哈希：`ec297a31`；实体原型 0 条；共现边 0 条
- 溯源标记 vault_id：`feynman-e61eead9a209`；当前字库 `feynman-e61eead9a209` → 一致

## 脚本（scripts/）

| 脚本 | 说明 |
|---|---|
| `ab_p21_blind.py` | P21 段落盲评材料：40 组"基线 vs 开启同构"段落，A/B 随机摆位 + 机械指标。 |
| `ab_tension_blind.py` | P16 收尾：把 40 组"基线 vs 张力弧"段落做成盲评材料，等人打分。 |
| `ab_tension_proxy.py` | P16 张力弧的机械代理评估：不替人打分，只量可自动核对的指标。 |
| `ablate_adaptive_beam_p15.py` | 按语体相位动态束宽的消融：同一批问题，逐档 beam_width 跑段落。 |
| `ablate_entangled_pairs_p15.py` | 纠缠对消融：同一字库/问题/语体，只换 entangled_pairs 表，看输出变化。 |
| `ablate_topology_p15.py` | P15 拓扑层消融：同一字库、同一问题，拓扑开/关只差一个乘数。 |
| `analyze_ratings.py` | 分析评分工作表：每个配置的平均分、自助法 95% 置信区间、与 baseline 的对比。 |
| `audit_vault.py` | 字库体检：四槽位计数、相位/层级/来源分布、逐字溯源与结构引用完整性。 |
| `build_entangled_pairs.py` | 为现有字库重算「可打分的」纠缠对：只取 实体×后果 的语料共现边。 |
| `build_isomorphism_p21.py` | P21 同构基底（离线）：句级共现 → PPMI → 截断 SVD → 固定符号的实数同构向量。 |
| `build_left_context.py` | P4/P5 语料层：为每个后果从句收集左上下文证据。 |
| `build_p10_inventory.py` | P10 后果库存：P9 库存 + 语料挖出的实体专属从句（已过质量门禁）。 |
| `build_p15_inventory.py` | P15 后果库存：P10 库存（391）+ 人工策展的非费曼物理书从句。 |
| `build_p8_corpus.py` | 把 su01 集成语料库（英文+中文对齐）转成铸字器吃的语料格式。 |
| `build_p9_inventory.py` | 生成 P9 后果槽库存：策展清单 + 语料实证溯源。 |
| `build_phase_bandwidth_p21.py` | P21 相位带宽：每个字粒在语料里实际出现时的语体相位分布（圆均值 + 集中度）。 |
| `build_user_overlay.py` | 把人工确认过的「笔记概念 → 主库字粒」映射编译成 overlay 文件。 |
| `build_vault_p10.py` | 构建 P10 词表与拓扑图：后果槽用 P9+挖掘补库（419 条）。 |
| `build_vault_p15.py` | 构建 P15 词表与拓扑图：P10 库存 + 非费曼物理书补库（468 条）。 |
| `build_vault_p9.py` | 构建 P9 词表与拓扑图：后果槽用策展库存，其余槽与 p8b 同源。 |
| `calibrate_p21.py` | P21 相容乘子标定：在开发集问句上量 score 分布与池内分位乘子的行为。 |
| `calibrate_topology.py` | P4 恒定参数标定：在 p8b 上用固定查询集扫 SIMPLEX_BOOST / HOLE_FACTOR / |
| `census_entity_consequences.py` | 实体专用后果缺口普查。 |
| `eval_fusion.py` | P0 评测台：固定查询集 × 融合/拓扑/类比/条件级联配置，与全基线逐项对比。 |
| `eval_isomorphism_p21.py` | P21 表征质量评估：留出句上的"共现回填"任务，不需要 LLM 判读。 |
| `eval_p14_specific.py` | P14 机械效应评估：specific_pref 开关对前 48 个查询的实际影响。 |
| `eval_p15_gap_entities.py` | P15 空档实体检索评估：p10 vs p15 在 12 个补库实体上的前 3 条后果。 |
| `gen_architecture.py` | 生成 ARCHITECTURE.md：路由、字库/拓扑现状、扩展库绑定、模块/脚本/测试清单。 |
| `gen_data_readme.py` | 生成 data/README.md：把数据目录里的产物按用途分类，附说明与大小。 |
| `make_p15_default_golden.py` | 生成产品默认档的黄金夹具：P15 字库 + 四检索开关 + 拓扑层。 |
| `make_p15_rating_worksheet.py` | P15 决策盲评表：只比两档——p10 检索 vs p15+专属从句优先。 |
| `make_rating_worksheet.py` | 生成出句评分工作表：固定抽样、配置溯源、去重、确定性乱序。 |
| `make_topology_rating_worksheet.py` | 拓扑消融盲评表：只用 ablate_topology_p15.py 的 60 种问法（每种两句）。 |
| `mine_entity_consequences.py` | 离线知识库扩展管线·第一段：实体专属后果取矿并落候选。 |
| `mine_gap_books.py` | P15 空档实体·跨书候选挖掘：扫桌面中文物理书 + p8 语料，落候选清单。 |
| `run_all_tests.py` | 一次跑完全部回归：逐个测试文件起独立进程，互不污染环境变量。 |
| `score_ab_p21.py` | P21 盲评揭盲与汇总：读评分表 × 钥匙，算两版均分、配对差与胜负。 |
| `switch_vault.py` | 切换产品字库/拓扑图：只改 .env 里两个已知键，其余行逐字节保留。 |

## 回归（tests/）

共 49 个文件。统一入口：

```sh
PYTHONPATH=. .venv/bin/python -B scripts/run_all_tests.py
```

| 文件 | 用例数 | 说明 |
|---|---|---|
| `test_abstract_morpheme_gate.py` | 5 | 实体槽残余抽象词门禁：词级黑名单 + 窄语素后缀。 |
| `test_adaptive_fixes.py` | 8 | 适应性改造回归：相对排序补救、确定性归并、离线诚实降级。 |
| `test_dialogue_modes.py` | 7 | P20 五条对话优化：默认全关，开启只加字段，正文逐字节不变。 |
| `test_discord_gateway.py` | 12 | P18 Discord 网关：离线用假总线测，不需要 bot token，也不装 discord.py。 |
| `test_domain_crawler.py` | 7 | 旁路扩展器回归（P6.3）：默认关闭、只写扩展库、死亡标记拦截、原型覆盖生效。 |
| `test_domain_expander.py` | 8 | 扩展库 ext_hash 与原子写入回归（P6.5）。 |
| `test_entangled_pairs.py` | 4 | 纠缠对建设回归：只从 E→C 共现边生成、(实体, 后果) 键可命中、覆盖可实验挂载。 |
| `test_entity_clause_miner.py` | 21 | 实体专属后果挖掘器：从句级门禁的回归测试。 |
| `test_evolution_loop.py` | 11 | 受控进化闭环回归（P6.6）：显式反馈、隐式延续、钳位、剪枝、GC 与隔离。 |
| `test_ext_vault_binding.py` | 4 | 扩展库与字库的溯源绑定：学出来的边必须标明是在哪个字库上学的。 |
| `test_feynman_abstract.py` | 6 | P6.2 抽象实体原型路由回归。 |
| `test_feynman_entities_memory.py` | 17 | 用户实体注入 + 会话记忆回归（P5.2 / P5.3）。 |
| `test_feynman_typesetter.py` | 21 | 费曼心智活字回归：出处可查、确定性、语体敏感、术语闸门、留白、骨架、算子。 |
| `test_frontend_contract.py` | 3 | 前端契约回归（P6.1）：大屏读到的每个字段都必须真实存在于后端响应里。 |
| `test_gap_clause_miner.py` | 16 | P15 空档实体·跨书从句挖掘器。 |
| `test_growth_journal.py` | 6 | 成长年轮与真实会话重置。 |
| `test_lean_sandbox_real.py` | 0 | 真实 Lean 4 沙盒回归。 |
| `test_majority_illusion.py` | 0 |  |
| `test_p10_decode_retrieval.py` | 6 | P10 回归：解码期检索约束——后果从句按实体的 E→C 语义边从语料里取， |
| `test_p10b_retrieval_gates.py` | 6 | P10b：检索池质量门禁（主语错配 / 方向矛盾 / 残句）。 |
| `test_p10c_retrieval_via_action.py` | 4 | P10c：没有 E→C 边的实体，用 E→A→C 两跳检索后果。 |
| `test_p12_topic_lock.py` | 5 | P12：问句实体锁定（topic lock）。 |
| `test_p14_specific_pref.py` | 6 | P14：检索时 specific（tier=2）从句优先于通用原子。 |
| `test_p15_curated_rank.py` | 5 | P15 策展从句排序：specific_pref 下 tier2 → curated → 次数。 |
| `test_p15_default_golden.py` | 2 | 产品默认档黄金回归：P15 字库 + 四检索开关 + 拓扑层，25 问。 |
| `test_p15_edge_supplement.py` | 6 | P15 语料外书籍补边：build_vault(edge_supplement=...) 只加不减。 |
| `test_p15_paragraph_defaults.py` | 4 | 段落模式与前缀同一套产品默认档（P15 + 专属从句优先）。 |
| `test_p15_product_defaults.py` | 4 | 产品默认档：P15 + 专属从句检索优先。 |
| `test_p21_isomorphism.py` | 12 | P21 同构空间：产物结构、乘子边界、默认关逐字节一致、段落实体锁。 |
| `test_p4p5_informativeness.py` | 5 | P4/P5：后果从句的信息量偏好（左上下文一致性）。 |
| `test_p7_fusion_topology.py` | 12 | P7 回归：自由能算子融合、拓扑层（PPMI/SVD/三角形）、类比传输、段落织构。 |
| `test_p8_conditioned.py` | 10 | P8 回归：铸字净化、条件级联（实体→动作→后果）、段落接力。 |
| `test_p8_query_verb.py` | 7 | P8 第二轮修复回归： |
| `test_p9_calibration.py` | 12 | P9 回归：算子分数连续化、Boltzmann 温度、拓扑证据分级与池内归一。 |
| `test_p9_consequence_inventory.py` | 0 | P9 后果槽扩表回归：build_vault 的 slot_limits 只改指定槽配额，别的不动。 |
| `test_paragraph_http_e2e.py` | 5 | 段落接口端到端回归：走真实 HTTP 栈（TestClient），锁住三件此前没人测的事。 |
| `test_paragraph_variety.py` | 5 | 反生硬回归：句子不能三句同一个起手式，也不能答非所问。 |
| `test_quantum_movable_type.py` | 10 | 量子活字印刷机回归：铸字质量、确定性、心境敏感性、套话否决、API 契约。 |
| `test_request_metrics.py` | 4 | 滚动指标回归：/stats 暴露空白率与延迟，单位逻辑独立可测。 |
| `test_second_notebook.py` | 2 |  |
| `test_sensory_ingest.py` | 6 | P19 只读感知接入：默认只读，写入要过两道闸，匹配不依赖分词模型。 |
| `test_session_store_bound.py` | 4 | 会话库有界 + 进化库并发写不丢更新。 |
| `test_smart_contract_audit.py` | 0 |  |
| `test_solve_cache.py` | 8 | 结果缓存 + 调用预算回归（全程离线/假 Key，不产生真实网络请求）。 |
| `test_switch_vault.py` | 4 | switch_vault 回归：只动两个已知键，其它行逐字节保留，秘密值不打印。 |
| `test_tension_arc.py` | 6 | P16 张力弧：只重排候选并在响应里加证据字段，不改任何文本。 |
| `test_user_overlay.py` | 6 | 用户 overlay 字库：只许把笔记别名映射到主库既有字粒。 |
| `test_vault_audit.py` | 6 | 字库体检脚本回归：合成小字库覆盖可回源/外部书/无源、结构错误与 --strict。 |
| `test_wiki_relay.py` | 0 |  |

## 约定

- 不生成新文字：所有输出字粒必须能在字库中逐字溯源；编辑过的标 `normalized_from`。
- 历史字库/拓扑版本只增不改；换产品指针用 `scripts/switch_vault.py`。
- 训练语料中英一一对应，英文只取费曼原话。
- 默认档由服务端补齐，客户端显式传值优先（pydantic `model_fields_set`）。
