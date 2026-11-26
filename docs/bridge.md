# 左脑桥

## 知识包回答

用 `python3 scripts/chat.py --bridge --no-state` 体验。也可设置 `OMEGA_LEFT_BRIDGE=1`。
右脑先确定卡片证据或已审核联想，再把原句和可以断行的位置传给左脑。左脑按回答篇幅和当前
对话动作选择停顿，右脑核验后从原句生成显示文本。中文和英文共用这条路径，无需 vault、
网络、派生索引或额外依赖。

`base_reply` 和 `source` 保留原值，`retrieval.evidence_layout` 记录实际采用的字符偏移。
引号和括号内部保留原样。含公式标记、已有换行、原样复述以及缺证据的回合沿用原显示流程。
默认关闭。
协议见 [接口说明](../contracts/engine-left-bridge.md)。

## 独立原型的 vault 排版

独立原型可把情绪状态和魔方偏好映射为排版参数，从本机 vault 组织文字。

```
用户消息
  → 右脑：appraisal → 7 维情绪 → Policy → StyleDirective（控制语气壳）
  → CubeBias(depth, verbosity)
      ↓  bridge.map_affect()
  左脑：ParagraphRequest(style_deg, 8 个算子权重, 开关)
      ↓  weave_paragraph()
  vault 素材组成的段落
```

## 映射规则（`bridge/mapping.py`）

所有维度先转成 **相对人设基线的偏差**，所以同一套公式对费曼（高多巴胺、
低皮质醇）和老庄（低唤醒、高愉悦度）都成立，不需要按人设调参。

| 情绪偏差 | 影响的左脑参数 |
|---|---|
| 好奇（dopamine 高于基线） | `cross` ↑、`anti_intuition` ↑、`style_deg` ↑ |
| 压力（cortisol 高于基线） | `micro` ↑、`jargon_strip` ↑、`style_deg` ↓、`topic_lock` 开 |
| 投入（arousal 高于基线） | `boundary` ↑、`reversal` ↑、`conditioned_chain` 开 |
| 亲近（affinity 高于基线） | `style_deg` ↑ |
| 戒备（defensiveness 高于基线） | `cargo_cult` ↑、`dissonance_resolution` 开 |
| 疲劳（energy 低于基线） | `least_action` ↑、`boundary` ↓、`conditioned_chain` 关 |
| 魔方 depth > 0 且有好奇 | `use_isomorphism` 开 |
| 人设 `switches.jargon_aversion` | `jargon_strip` +0.5 |

关键约束：

- **单调**：好奇升高绝不会让 `cross` 降低（有测试逐对断言）。
- **有界**：输出一律夹到左脑自己的字段范围（`style_deg` 0–120，算子 0–2）。
  `tests/test_bridge_mapping.py::TestLeftBrainAgreement` 会对照左脑的
  pydantic 模型，防止哪边改字段了而这边没跟上。
- **确定性**：纯函数，无随机数、无时钟。同一状态永远同一参数。

## 开关与失败契约（`bridge/responder.py`）

```
OMEGA_LEFT_BRIDGE=1      # 默认不开
QBOT_LEFT_ROOT=...       # 左脑代码位置，默认 <repo>/left
QBOT_FEYNMAN_VAULT=...   # vault 路径，默认 left/data/feynman_style_vault.json
```

`LeftResponder.reply()` 的契约只有一句话：**拿不准就返回 None，让上层走原路**。
vault 缺失、左脑依赖导入失败、排版器抛异常、排出来是空串——全部返回 None
并把原因写进 `last_error`，绝不向上抛。

## 已知限制（2026-11-26 实测）

用本机 `feynman_style_vault.json` 端到端跑通后确认：**排版器能成句，但不负责事实**。

- 它从 vault 的骨架 + 实体 + 动作里拼句子。问"为什么天空是蓝的"，它给得出
  费曼腔的段落，但内容是"频率聚集，就像天空"这类从语料里抽来的搭配，不是答案。
- 这不是 bug：左脑是排版器，不是问答器。`weave_paragraph` 没有"给我一段源文本
  重新措辞"的输入口，它只能从 vault 里组合。
- 所以默认答案路径必须仍是 KB 卡片；排版器只适合**人设自建了领域 vault**、
  且接受"风格拼句"的场景。仓库不附带任何 vault：本机这份 vault 由书稿语料
  派生（实体/骨架/句子都带原书来源），属于排除在仓库外的派生物。

## 为什么不做成右脑直接调用左脑

右脑要能单独跑（401 个测试不依赖 pydantic、不依赖 `left/` 目录）。
把左脑藏在 responder 后面、懒加载、失败即退，是让两套依赖保持解耦的
唯一简单办法。`bridge/` 本身也只依赖标准库。
