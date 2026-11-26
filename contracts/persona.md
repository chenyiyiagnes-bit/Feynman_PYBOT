# 人设格式（persona.json）

一个「PYBOT 心智」= 一个 26 方块的评价器 + 7 维情感基线 + 一套说话外壳 + 自带知识卡。
全部写在一个 JSON 里，不写代码就能换人。

## 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema` | int | 目前为 1 |
| `id` / `name` | str | 人设标识与显示名 |
| `description` | str | 可选，一句话介绍（导出工具会带上） |
| `language` | `"zh"` \| `"en"` | 默认语言；回复内不混语言 |
| `baselines` | object | 7 维基线：dopamine/cortisol/valence/arousal/affinity/defensiveness/energy；valence ∈ [-1,1]，其余 [0,1] |
| `taus` | object | 每维的指数衰减时间常数（秒） |
| `voice` | object | `prefixes_zh/endings_zh/prefixes_en/endings_en` |
| `switches` | object | `correction_as_curiosity` / `jargon_aversion` / `politeness` |
| `disable_cubies` | array | 关掉的方块名（如费曼关掉 `corner_shame`） |
| `override_triggers` | object | 方块名 → 额外正则列表（覆盖默认触发器） |
| `extra_effects` | object | 方块名 → 额外情感增量 |
| `cubies` | array | 26 个方块，必须恰好 26 个 |
| `knowledge` | object | `kb_dirs`（相对本文件）、`corpus`（自备语料路径，可为 null）、`corpus_note`（可选，说明语料来源与许可） |
| `license` | object | 人设内容与代码各自的许可证 |

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

复制 `personas/feynman/persona.json`，改四个地方即可变成新角色：

1. `baselines` / `taus` —— 性格的温度和恢复速度
2. `voice` —— 口头禅
3. `cubies[].triggers` / `effects` —— 什么事会让他起反应
4. `knowledge.kb_dirs` —— 换掉知识卡目录

## 语料边界

仓库不附带任何有版权的语料。`knowledge.corpus` 留空时排版器只用知识卡；
使用者放自己的副本并填路径后才会启用。

《道德经》《庄子》原文属公有领域，可以整本随人设发布；
费曼著作、现代译本等受版权保护的内容一律不进仓库。
