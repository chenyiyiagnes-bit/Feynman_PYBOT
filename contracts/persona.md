# 人设格式（persona.json）

一个「PYBOT 心智」= 一个 26 方块的评价器 + 7 维情感基线 + 一套说话外壳 + 自带知识卡。
全部写在一个 JSON 里，不写代码就能换人。

## 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema` | int | 目前为 1 |
| `id` / `name` | str | 人设标识与显示名 |
| `description` | str | 可选，一句话介绍（导出工具会带上） |
| `language` | `"zh"` \| `"en"` | 本包的语言。一个包只有一种语言 |
| `baselines` | object | 7 维基线：dopamine/cortisol/valence/arousal/affinity/defensiveness/energy；valence ∈ [-1,1]，其余 [0,1] |
| `taus` | object | 每维的指数衰减时间常数（秒） |
| `voice` | object | 只放本包语言的 `prefixes_<lang>/endings_<lang>`，以及答不上来时的台词 `abstain_<lang>`、追问走不下去时的 `followup_gap_<lang>`、把握不大时的 `hedge_<lang>`；不写则用引擎的中性默认句 |
| `switches` | object | `correction_as_curiosity` / `jargon_aversion` / `politeness` |
| `disable_cubies` | array | 关掉的方块名（如费曼关掉 `corner_shame`） |
| `override_triggers` | object | 方块名 → 额外正则列表（覆盖默认触发器） |
| `extra_effects` | object | 方块名 → 额外情感增量 |
| `cubies` | array | 26 个方块，必须恰好 26 个 |
| `knowledge` | object | `kb_dirs`（相对本文件）、`corpus`（自备语料路径，可为 null）、`corpus_note`（可选，说明语料来源与许可） |
| `license` | object | 人设内容与代码各自的许可证 |

## 骨架一个，包很多，一个包一种语言

硬骨架只有一份（`right/`、`app/`、`bridge/`、`left/`）。人设是一个个独立的包，目录名即包名：

```
personas/feynman-zh/     费曼 · 中文
personas/feynman-en/     费曼 · 英文
personas/laozhuang-zh/   老庄 · 中文
personas/laozhuang-en/   老庄 · 英文
```

- 每个包的 `persona.json` 里 `language` 固定为 `zh` 或 `en`，`voice` 只放该语言的
  `prefixes_<lang>` / `endings_<lang>`；另一种语言可以不写。
- 知识卡也只放该语言。英文包不是「中文包 + 翻译叠加」，而是自带英文卡。
- 运行时不允许在中文包上请求英文回复（反之亦然），会直接报错：换语言就换包。
- 想做双语角色，就是建两个包；它们共享骨架，不共享任何软数据（见下表）。

## 数据归属：一个目录就是一个心智

硬骨架（`right/omega_mind`、`app/`、`bridge/`）只有一份；两个人设共用引擎，但
**不共用任何软数据**。运行时按人设解析：

| 数据 | 位置 | 进仓库 |
|---|---|---|
| 情感参数、外壳、方块覆盖 | `personas/<id>/persona.json` | 是 |
| 知识卡 | `personas/<id>/knowledge/*.jsonl` | 是 |
| 语料（喂左脑排版器的原文） | `knowledge.corpus` 指向的文件 | 否（版权） |
| 联想边（卡片间的判断） | `personas/<id>/data/assoc/assoc_edges.jsonl` | 否，`.gitignore` 已覆盖 |
| 学习覆盖（教给它的新卡片） | `$PYBOT_STATE_DIR/<id>/kb_learned/`，默认 `~/.pybot/<id>/kb_learned/`（旧目录 `~/.feynman_pybot` 仍存在且新目录未建时沿用旧目录） | 否（运行时状态） |

引擎里没有指向某个人设的默认路径：换人设就换整个目录，不存在「老庄继承了费曼的联想」。

## 骨架与血肉：一条可以检查的分界线

判断一样东西该写进引擎（骨架）还是写进包（血肉），只问一句：**换掉人设之后，
它还应不应该成立？**

- **骨架**：对话循环、检索算法、风格测量、加载规则、协议、评测工具、默认阈值。
  它们对所有包一样，错了是整体出错，能用规则断言。
- **血肉**：卡片正文、`voice` 台词、方块取值、`baselines`/`taus`、别名与出处、
  语料。它们只属于一个包，错了只坏一个包，好坏靠金标集和盲评判断。
- **衍生血肉**：lexicon、联想边、风格带数值、账本。由卡片算出来，可重建，
  不入库；入骨架的只是重建程序。

三条推论：

1. 凡是能从血肉算出来的数字，不要写进文档当事实——文档写命令，数字让工具现算。
2. 只有骨架能保证的性质（英文前缀的空格、跨语言拒答、出处形状），不要指望某个
   包的数据碰巧写对；反过来，某个包的说话习惯也不要硬编码进引擎。
3. 同一份事实只有一个真源。卡片出处统一走 `omega_mind/provenance.py`：新卡写
   `support`，旧的顶层 `source` 在读取时归一化，不再各工具各认一套。

## 方块（cubie）

```json
{
  "name": "core_logic",
  "role": "core",
  "face": "L",
  "valence": 0.0,
  "triggers": [["为什么|原理|逻辑|分析|推导|证明", 0.6]],
  "effects": {"arousal": -0.03, "energy": -0.01},
  "note": "慢思考"
}
```

- `role` ∈ `core|edge|corner`；数量必须 6/12/8。
- `face` ∈ `F|B|L|R|U|D`，同面的方块共享一个面收益。
- `triggers` 是 `[正则, 增益]` 列表；命中后强度 × 增益，3 轮 0.6ⁿ 习惯化。
- `effects` 是命中后注入的 7 维增量。

## 最小可用人设

复制 `personas/feynman-zh/persona.json`，改四个地方即可变成新角色：

1. `baselines` / `taus` —— 性格的温度和恢复速度
2. `voice` —— 口头禅
3. `cubies[].triggers` / `effects` —— 什么事会让他起反应
4. `knowledge.kb_dirs` —— 换掉知识卡目录

## 语料边界

仓库不附带任何有版权的语料。`knowledge.corpus` 留空时排版器只用知识卡；
使用者放自己的副本并填路径后才会启用。

《道德经》《庄子》原文属公有领域，可以整本随人设发布；
费曼著作、现代译本等受版权保护的内容一律不进仓库。
