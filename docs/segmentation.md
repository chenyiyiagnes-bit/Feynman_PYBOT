# 切词与检索回归：边界只有一个出处，好坏由每个包自己的题集说话

中文没有空格，先切词才能建索引、算相似度、排版。切法一变，同一个问题就是
另一组 token，检索结果会悄悄挪位——所以这件事必须只有一套规则，而且换规则
之前要有证据。

## 1. 词边界只有一处真源

| 谁 | 在哪 | 干什么 |
|---|---|---|
| 右脑检索 | `right/omega_mind/lang/` | 规则表按语言（`zh`/`en`）和 profile（`default`/`classical`）选；`LanguageCore.segment()` 决定词边界 |
| 左脑排版 / 挖矿 / 字库 | `left/plugins/word_segmentation.py` | 不重新实现，只包一层 `omega_mind.lang`，左右脑拿到的是同一组边界 |
| 词性标注 | `jieba`（可选依赖） | 只给词打 n/v 标签，**不**决定边界；jieba 没装时边界照样不变 |

`persona.json` 的 `language_profile` 决定这个包用哪套规则（老庄用 `classical`），
其余用 `default`。新增语言＝在 `right/omega_mind/lang/<lang>/` 加数据表，不是改代码。

**`left/quantum_tokenizer.py`（QGlyph 2–4 字短语候选）不是分词器。** 它是旧的
短语候选挖掘启发式：产出「这个词组可能是个术语」的候选，喂给词表构建与晋升
流程。不要把它当成第二套切点，也不要让检索走它。

## 2. 检索的回归网属于包，不属于骨架

每个包可以在 `personas/<id>/evals/golden.json` 自带一张题集：

```json
{
  "schema": 1,
  "language": "zh",
  "golden":   [{"query": "熵是什么", "acceptable": ["th-004"]}],
  "negatives": ["今天天氣怎麼樣"]
}
```

- `golden`：必须命中 `acceptable` 里任一卡的问句；
- `negatives`：**必须**落到 `no_match` 的陷阱（能答出来就是泄漏）。

跑起来的 mind 只审自己包那张网（`reflect.golden_ok` → `golden_queries.for_package`）；
两包合并的 `GOLDEN`/`NEGATIVES` 只留给测试与评测工具。骨架里没有一个卡 id——
没有 `evals/golden.json` 的包就是没有网：`golden_ok` 直接放行（没有可审的题，
不能借别人的题——那些卡不在这个索引里，条条都会"失败"）。
新增包应该补一张自己的网；空网只表示"暂未审"，不表示"审过没问题"。

出厂包的网各自维护：`feynman-zh` 77 题 / `feynman-en` 69 题 /
`laozhuang-zh` 16 题 / `laozhuang-en` 13 题。加题前先在真实 KB 上跑一遍
`answer()`，确认它现在真的命中再写进去；题集不追新卡，只钉住已经可靠的行为。

## 3. 评测一个切法：`tools/eval_segmentation.py`

```sh
python3 tools/eval_segmentation.py                    # 全部方案，只报告不失败
python3 tools/eval_segmentation.py --scheme bigram --strict
python3 tools/eval_segmentation.py --json out.json    # 完整报告（含逐条 misses/leaks）
```

每个方案在**每个包自己的**卡片和题集上打分，两件事各算各的：

- **检索**：golden 命中数、陷阱拒答数（分包 + 合计）；
- **人工切分参照**：`tests/data/segmentation_reference.json` 上的边界
  precision/recall/F1 与整句完全正确的条数。

方案三个：

| 方案 | 实现 | 现状 |
|---|---|---|
| `bigram` | 出厂基线，`feynman_kb._tokens` 的字符二元组 | **在用的索引方案** |
| `words` | `LanguageCore.segment()` 的词级切分 | 人工切分集 F1 0.825（bigram 0.389），但 golden 165/175（例：`什么是电介质` 命中 opt-009 而非 opt-020）——**未晋升** |
| `jieba` | jieba 精确模式 | 只是对照；没装就报告 `unavailable`，不假装跑过 |

`--strict` 的判据：出厂的 `bigram` 必须仍全中自己包的网（命中 < 总数、陷阱被答出
都算回归），且任何候选方案不得比基线少命中或少拒答。CI 只用
`--scheme bigram --strict`——它盯的是「语料/索引改动有没有弄坏现在的检索」，
候选方案的比较是本地实验，不该让主干长期红着。

**它不证明什么：** golden 命中≠答案好。题集只保证「该拿的卡还拿得到」，
答案质量看 `tests/data` 里的确认集与人工评审（见 `docs/cards.md`）。

## 4. 词表图（lexicon）：按包选择，默认关

`tools/build_lexicon.py` 从卡片派生 `terms.json` / `edges.jsonl`（派生数据，
不入库），检索时用共现邻居做**有上限、不能单独开门**的召回扩展。
它由 `persona.json` 的 `recall_expansion` 决定：

- 关（默认，所有出厂包）：`lexicon_dir` 为 `None`，有没有构建过词表图回答都一样；
- 开：`app/runtime.build()` 和语言切换时的 `FeynmanKB.for_language()` 都会指向
  包自己的 `data/lexicon/`；文件不存在时照样为空操作。

## 5. 学到的卡放在状态目录，不放进包

学到的新卡与别名（`kb_learned`）对所有入口只有一个位置：
`$PYBOT_STATE_DIR`（默认 `~/.pybot`）`/<包 id>/kb_learned`（`omega_mind.paths`）。
应用、CLI、评测脚本都走 `learned_dir_for()`，谁教过的卡另一处也看得到。
包目录只放随仓库分发的数据；检出放在只读路径或多人共享时，学习也不会丢。

开之前必须在真实卡片上过 golden 网（`tests/test_recall_expansion.py::RealCorpus`），
宁可不开，也不让「这台机器碰巧构建过词表」改变同一个包的答案。
