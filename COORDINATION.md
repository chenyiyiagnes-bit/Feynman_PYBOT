# 协作约定

两个窗口同时在同一个仓库上工作。文件所有权按目录划分，避免互相踩。

## 分支

仓库：https://github.com/chenyiyiagnes-bit/Feynman_PYBOT（private，发布时转 public）

| 分支 | 用途 | 谁能推 |
|---|---|---|
| `main` | 稳定 / 发布 | 主负责人 |
| `dev` | 集成分支，阶段性成果都往这里合 | 两边都可推 |
| `assoc/*` | 右脑联想优化的特性分支 | 联想优化窗口 |

流程：`assoc/*` → PR 到 `dev` → 主负责人review后合入 → 里程碑时 `dev` → `main`。
任何人不得直接推 `main`。

## 语料红线

只上传代码、测试、人设 JSON、文档。以下一律不进仓库：

- 任何语料原文件（`*.jsonl` 的原文库、`p8_corpus_*`、`cpt_zh.jsonl`）
- 由语料派生的大产物（字库 `feynman_style_vault*.json`、拓扑 `*.npz`）
- 运行时状态（`*.state.json`、`snapshots/`、`data/kb_learned/`、`data/assoc/`）

`.gitignore` 已覆盖这些路径；提交前用 `git status` 再扫一眼。

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

## 上传权：只有一个

本机所有会话共用同一套 GitHub 凭据，因此**不需要每个 BOT 自己有账号**。

| 角色 | 怎么交付 | 谁上传 |
|---|---|---|
| 联想优化窗口 | 在 `Feynman_PYBOT-assoc` worktree 里本地提交到 `assoc/association` | 它可自行 `git push`；不确定就只提交，主负责人推 |
| Astra / 老庄开发 | 成品放进 `incoming/laozhuang/`（保持完整目录结构） | 主负责人检查后提交到 `persona/laozhuang` |
| Gemini / 审核 | 报告放进 `incoming/gemini/`，文件名 `YYYY-MM-DD-gemini.md` | 主负责人提交到 `review/gemini` |

主负责人每次上传前检查：只动该分支对应目录、无语料、无状态文件、`tools/validate_persona.py` 通过。
