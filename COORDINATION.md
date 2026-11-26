# 协作约定

两个窗口同时在同一个仓库上工作。文件所有权按目录划分，避免互相踩。

## 分支

| 谁 | 分支 | worktree |
|---|---|---|
| 主负责人（合并层 / 发布） | `main` | `Feynman_PYBOT/` |
| 右脑联想优化 | `assoc/*` | `Feynman_PYBOT-assoc/` |

`main` 只由主负责人合并。任何人不得直接推 `main`。

## 文件所有权

| 路径 | 所有者 |
|---|---|
| `bridge/` `contracts/` `tools/` `tests/` `docs/` `config/` | 主负责人 |
| `right/omega_mind/assoc.py` `spreading.py` `feynman_kb.py` `learner.py` | 联想优化窗口 |
| `right/omega_mind/engine.py` `policy.py` `renderer.py` `config.py` | 主负责人；改动须先写进 `contracts/` 并通知 |

## 冻结的接口

以下签名与返回结构不得单方面修改：

- `MindEngine.turn(text, base_reply=..., user=...)`
- `left_brain.plan_for(state, persona, activations, cube_bias) -> LeftBrainPolicy`
- `policy.Policy.compute(state, activations, language) -> StyleDirective`
- `TurnResult` 字段集合（只增不删）

需要改，先在 `contracts/` 写变更说明，主负责人确认后再动。

## 提交前

- `python3 -m unittest discover -s right/omega_mind/tests` 全绿
- 不提交任何 `*.state.json`、`snapshots/`、`data/kb_learned/`、`data/assoc/`
- 不提交语料
