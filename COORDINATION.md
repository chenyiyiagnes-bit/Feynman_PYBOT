# 协作约定

两个窗口同时在同一个仓库上工作。文件所有权按目录划分，避免互相踩。

## 分支

仓库：https://github.com/chenyiyiagnes-bit/Feynman_PYBOT（已转 public）

| 分支 | 用途 | 谁能推 |
|---|---|---|
| `main` | 稳定 / 发布 | 主负责人 |
| `dev` | 集成分支，阶段性成果都往这里合 | 主负责人（代传其他人的分支） |
| `assoc/association` | 右脑联想优化的特性分支 | 联想优化窗口（Claude） |
| `persona/laozhuang` | 老庄 PYBOT 开发 | Astra 团队 |
| `review/gemini` | 审核报告 | Gemini |

流程：特性分支 → PR（或由主负责人 cherry-pick）到 `dev` → 主负责人 review 后合入 →
里程碑时 `dev` → `main`。任何人不得直接推 `main`。

提交作者必须用能关联 GitHub 账号的邮箱（如 `227941007+chenyiyiagnes-bit@users.noreply.github.com`），
`yiyichen@YiyideMacStudio.lan` 这类本机邮箱不会关联到账号。

## 安全与隐私红线（所有 agent 必读）

仓库根的 [`AGENTS.md`](AGENTS.md) 是全部 agent 的硬性规则页，动手前先读。要点：

- **凭据一律不进仓库**：token、key、密码只放 `.env`（已忽略），示例只写 `.env.example`。
  发现凭据已进历史，通知主负责人**轮换**——删文件不能让凭据失效。
  先例：2026-11-26 `left/.en` 的 `DISCORD_BOT_TOKEN` 随 `814bc85` 泄漏到公开仓库，已删、已轮换。
- **不写死本机绝对路径**（macOS 用户主目录下的绝对路径之类），用相对路径或环境变量覆盖。
- **提交前跑** `python3 -m unittest tests.test_repo_hygiene`，它会在 git 索引上查这三类问题。
- 对外发布、发消息、付费、删除、登录等动作，即使是被分派的活也要先经主负责人授权。

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
