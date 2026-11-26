# HANDOFF — 状态、约束、待办

任何新窗口、新协作者接手时先读这一页，再读 [`AGENTS.md`](AGENTS.md)。
这里记录的是**代码里看不出来的东西**：为什么这样设计、哪些决定已经定了、下一步按什么顺序做。
代码结构、提交历史请自己看，不在这里重复。

最后更新：2026-11-26。

---

## 1. 心智模型（一句话）

硬骨架只有一份，人设是包，一个包一种语言。

```
personas/feynman-zh/     费曼 · 中文
personas/feynman-en/     费曼 · 英文
personas/laozhuang-zh/   老庄 · 中文
personas/laozhuang-en/   老庄 · 英文
```

- 一个包 = `persona.json` + `knowledge/*.jsonl`，`language` 固定为 `zh` 或 `en`。
- 运行时语言不匹配直接报错（`PersonaError`），不做静默回退。**换语言就换包。**
- 两包之间**不共享任何软数据**：知识卡、联想边、学习覆盖、语料全部按包隔离。
- 骨架（`right/` `app/` `bridge/` `left/`）里不允许出现指向某个人设的默认路径；
  默认值用 `Path(__file__).resolve().parents[N]` 或环境变量。

## 2. 现在做到哪了

已推 `dev`（`9f4aa1d persona: one hard skeleton, single-language packages`），CI 全绿。

- `FeynmanKB` 按包装载单一语言；`source_prefix` 用包 id（`feynman-zh:flp-i-01`）。
- 英文卡从引擎目录搬进 `personas/feynman-en/knowledge`（293 张）；
  中文包 302 张。旧的「双语叠加」路径只留给直接调引擎的测试，包内不再触发。
- `laozhuang-zh` 有 `persona.json`，**没有卡片**。
- `laozhuang-en` 有 `persona.json`，英文语气是**占位**，也没有卡片。
- 英文联想边本地生成过 89 条（不入库，`data/assoc/` 被忽略）。

2026-11-26 这一轮做完 A1–A5、B1–B4、C1–C3、D4（提交 `03b642e`…`5296c9c`）：

- **A1–A3** 语言规则搬进 `right/omega_mind/lang/`，`modern`/`classical` 两套 profile；
  **A4** `tools/eval_segmentation.py`；**A5** 干净确认集 15/30（诚实基线，不粉饰）。
- **B1** `tools/build_lexicon.py`：派生 `terms.json`/`edges.jsonl`（不入库、可复现）；
  **B2** `tools/code_cards.py`：token id + 字节偏移，每个 token 可回溯；覆盖率
  zh 22.9% / en 46.9%（OOV 是事实，不是缺陷）；**B3** 召回扩展默认关，打开后
  golden 47/47、陷阱 5/5、确认集 11→12，但**只有 2 字邻居能命中**（索引是 bigram）；
  **B4** `bridge/slot_gate.py`：用户槽词必须在所选卡片的编码 token 里，否则拒答。
- **C1** `tools/promote_terms.py`：候选＝高 npmi 跨卡相邻词对，晋升须在人工切分集上
  F1 提升 ≥ `MIN_F1_GAIN`（0.0005，暂定）且不丢 exact；决定写入
  `data/lexicon/promotions.jsonl`，重建保留晋升、丢弃不重试（`--retry` 除外）。
- **C2** `right/omega_mind/corrections.py`：切错/答错记录带出处、词表版本与时间，
  可统计（按 kind/source/重复问题）与回放；日志在 `PYBOT_STATE_DIR`（默认 `data/state/`，已忽略）。
- **C3** `tools/lexicon_versions.py`：每次构建整批写入 `data/lexicon/versions/<内容哈希>/`，
  顶层文件是激活版本的副本；`activate` 是整文件替换，不是补丁。编码与词表版本不一致时
  闸门自动失效（`terms_digest` 对不上）。
- **D4** `tools/build_assoc.py`：由卡片重建联想边，只产出 `origin="cooccur"` 候选
  （永不朗读、没有 mechanism），每卡扇出封顶，重建逐字节一致。

验证方式：

```sh
python3 -m unittest discover -s tests                          # 合并层
cd right && python3 -m unittest discover -s omega_mind/tests   # 右脑
cd .. && python3 -m unittest tests.test_repo_hygiene            # 提交前必跑
```

## 3. 已定、不要再重新讨论的设计结论

这几条是几轮讨论后定下的，接手时直接沿用；要推翻必须先说服仓库所有者。

1. **频率 ≠ 核心。** 高频词是功能词，不是人格。词的重要性用 tf-idf / 意外度，不用原始频次。
2. **共现 ≠ 语义。** 邻居要 PMI 加度数归一化，否则「的」连接一切。
3. **槽填充不能自由生成。** 槽里只能放卡片里已有的东西；否则就是换了个地方编造。
4. **卡片是唯一事实源。** 一张卡 = 一条可核对断言 + 出处（quote/span）+ 边界或反例。
5. **token 编码是派生索引，不是事实源。** 编码可重建、可版本化，卡片是原文。
6. **编码的真正用处是机检闸门**：发出的每个 token 都必须能在所选卡片字段的编码序列里找到。
7. **切词迁移是不对称的。** 功能词、形态、词缀可以跨域迁移；领域词表和语域不能。
   文言语料就是反例——**不能**拿现代汉语的切词规则去切老庄。
8. **自进化必须有外部锚点。** 没有 golden 检索差异或人工确认，就不许晋升新词条。
9. **语言规则归骨架的「语言核心」层，不归引擎代码。** 引擎是语言无关的；每个语言一份
   版本化数据；每个包选一份。

## 4. 任务清单

未完成的工作，按依赖分组。每项都写了验收标准——**验收标准写不出来的，说明还不是任务。**

### A. 语言核心与切词

| # | 任务 | 验收标准 | 依赖 |
|---|---|---|---|
| A1 | 抽 `lang/zh/core.json`、`lang/en/core.json` | 把 `feynman_kb.py` 里的 `_tokens`、`_tokens_en`、`_STOP_BIGRAMS`、`_expand_zh`、`_expand_en`、`_bridge_zh`、`_stem_en` 与权重常量**原样**搬成数据文件；两套测试全绿、检索结果逐条不变 | — |
| A2 | 三处分词统一 | 右脑 zh/en 与左脑词级分词走同一接口；两套测试全绿 | A1 |
| A3 | `modern` / `classical` profile | `lang/zh` 下两套规则，包用 `"segmentation": "classical"` 选择；老庄包用 classical 能切出文言单字词 | A1、A4 |
| A4 | `tools/eval_segmentation.py` | 输入两份切分结果，输出 golden 检索命中差异 + 人工切分集的准确率；可当 CI 的一步 | A1 |
| A5 | 干净确认集 | 新确认集不来自任何已训练语料；登记出处与构建日期 | — |

### B. 词粒库与编码

| # | 任务 | 验收标准 | 依赖 |
|---|---|---|---|
| B1 | `tools/build_lexicon.py` | 产出 `personas/<id>/data/lexicon/{terms.json,edges.jsonl}`（派生、不入库）；terms 带类别、df-idf、出处；edges 带 PMI 与关系类型；重建结果可复现 | A1 |
| B2 | `cards.coded.jsonl` | 每个卡片字段编成 token id + 字节偏移；生成时断言「每个 token 都能回溯到原文片段」 | B1 |
| B3 | 词粒库驱动召回扩展 | 默认关；打开后 golden 检索命中率不低于基线，否则不合并 | B1、A4 |
| B4 | 槽填充接机检闸门 | 左脑输出中每个实词 token 都能在所选卡片字段的编码序列里找到；找不到就拒答或降级，不得自由生成 | B2、左脑桥 |

### C. 自进化

| # | 任务 | 验收标准 | 依赖 |
|---|---|---|---|
| C1 | 词条晋升流程 | **已完成** `68ca31c`：候选＝高 npmi 跨卡相邻词对；晋升须在人工切分集上 F1 提升 ≥ `MIN_F1_GAIN`（0.0005，暂定，可用 `--retry` 重评）且不丢 exact；决定记入 `data/lexicon/promotions.jsonl`，重建保留晋升 | B1、A4 |
| C2 | 人工修正日志 | **已完成** `386e182`：`omega_mind/corrections.py` 记录切错/答错（含出处、词表版本、时间），可统计、可回放 | B1 |
| C3 | 词表版本化 | **已完成** `d8f7dd3`：整批写入 `versions/<内容哈希>/`，`tools/lexicon_versions.py` 整文件回滚；编码过期时闸门自动失效 | B1 |

### D. 内容与包

| # | 任务 | 验收标准 | 依赖 |
|---|---|---|---|
| D1 | 合并 `persona/laozhuang` | **已完成**：不 merge 旧分支（其 content 只有 `TASK.md` 是新东西，`persona.json`/`README` 反而是旧版），TASK 按新结构重写进 `personas/laozhuang-zh/`；旧分支留档 | — |
| D2 | `laozhuang-en` 语气重写 | **已决定：占位先留**，中文卡定稿后按原文对应表达重写，不靠中文回译 | D1 |
| D3 | 文言语料随包发布 | **已决定：不随包发布**，与费曼包一致——仓库只放卡片，语料自备；已写进 README/TASK | 所有者确认 |
| D4 | `tools/build_assoc.py` | **已完成** `5296c9c`：由卡片重建 `origin="cooccur"` 候选边，扇出封顶，重建逐字节一致 | B1（可复用 PMI） |
| D5 | 通知协作者目录改名 | Gemini / Astra 的脚本不再引用 `personas/feynman`、`personas/laozhuang` | — |

### E. 更远

| # | 任务 | 说明 |
|---|---|---|
| E1 | 人格 = 阈值与偏好 | 让语料决定「什么算答得上、遇边界补一句还是闭嘴、偏爱机制还是反例」，而不是落在词汇表上 |
| E2 | `knowledge.corpus` 接左脑 vault | 目前没有消费方，缺一个 `tools/build_vault.py` |

## 5. 已拍板的三件事（2026-11-26，所有者）

1. **老庄分支**：不 merge 旧分支，内容按新结构迁入 `personas/laozhuang-zh/`，旧分支留档。
   同一批不重复折腾：以后所有协作者只按 `COORDINATION.md` 的流程往 `dev` 提 PR。
2. **英文老庄语气**：占位先留；中文卡定稿后按原文对应表达重写。
3. **文言语料**：**不随仓库发布**，与费曼包一致。这条是定规：任何语料原文都不入库，
   不只文言——不要再就「某个语料要不要随包发」重新提问。

C1 的提升阈值目前取 `MIN_F1_GAIN = 0.0005`（暂定）：A4/A5 的基线只够证明切分集不回归，
不足以校准具体数字；有更多人工切分样本后应重新标定，代码里只承认这一个常量。

## 6. 已知的坑

- **改名会静默打断别人。** `personas/feynman` → `feynman-zh` 这次让 29 个测试失败、
  也让协作者脚本失效。以后改路径先用 `git grep` 扫全仓。
- **结构化卡片没有 `text` 字段。** `physics_edge` / `teachable_unit` / `method_card`
  的正文在 `phenomenon`、`force_invariant`、`pattern` 等字段里。任何「按 `text` 判断」
  的代码都会静默丢卡——`_already_english` 就踩过，丢了 21 张。
- **`data/assoc/` 与 `data/lexicon/` 都不入库。** 新克隆只有卡片，没有边。不要假设它们存在。
- **挖矿产物和人工边分文件。** `assoc_edges.jsonl` 是人工确认边（带 mechanism），
  `tools/build_assoc.py` 只写 `assoc_candidates.jsonl`，且拒绝覆盖含非 `cooccur` 行的文件。
  2026-11-26 曾用早期版本把 148 条人工边覆盖成 685 条候选，从 `Feynman_PYBOT-assoc`
  worktree 副本恢复；教训：派生工具不要在别人的产物路径上覆盖写。
- **未知词串会整段粘连。** FMM 把不认识的 CJK 连续段合成一个 token（≤6 字整段保留），
  所以「相邻词对」永远由已知词构成；想促销生词，先得让它能被切开。按字符偏移回指时
  一律用 `segment_spans`，别用 `str.find`——粘连跨空白会让偏移对不上（B2 踩过）。
- **召回扩展的邻居只能加分，不能进分母。** 把邻居算进覆盖分母会稀释原问题，
  确认集曾因此 11→9；`subject` 只含原查询 token（B3）。
- **重建词表后必须重跑 `tools/code_cards.py`。** 闸门用 `terms_digest` 对账，
  编码与词表不一致时自动失效（不拒答、也不假装验证过）。
- **每次构建都会快照。** `data/lexicon/versions/` 会随重建累积；回滚用
  `tools/lexicon_versions.py activate`，旧版本可以手工删，不删也不影响正确性。
- **确认集已污染。** 旧的 `heldout_voice` 等集合已进过训练语料，不能再用作评估证据（见 A5）。
- **训练语料红线**：中英必须一一对应；英文只能用费曼原话，禁止中文回译。
- **`.gitignore` 是最后一道，不是第一道。** 提交前仍然要 `git status` 亲眼看。

## 7. 分工

见 [`AGENTS.md`](AGENTS.md) 第 7 节与 [`COORDINATION.md`](COORDINATION.md)。
一句话：**审核与发布分离，合并闸门只有一个。**
