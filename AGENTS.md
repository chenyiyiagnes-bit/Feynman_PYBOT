# AGENTS.md — 所有 agent 的硬性规则

本仓库是 **public**。任何进入 git 历史的东西都视为已公开、已泄漏、已留档。
无论你是 Claude、GPT、Gemini 还是别的 agent、哪个窗口、哪条分支，动手前先读完这页。
规则冲突时以本页为准。协作分工见 `COORDINATION.md`。

## 1. 凭据：一律不进仓库

- **绝不**提交 token、key、密码、cookie、session、Webhook URL、bot 凭据，任何形式、任何分支。
- 真实配置只放 `.env`（已忽略）。要给人看格式，写到 `.env.example`，值用占位符
  （如 `DISCORD_BOT_TOKEN=your-token-here`）。
- 不要把凭据写进 commit message、测试 fixture、文档示例、日志、报错截图。
- 发现疑似凭据文件（`.env`、`.en`、`*.probe_write` 等）已在索引里：不要只是删文件，
  先读它、确认内容、`git rm --cached`，再通知主负责人**轮换**凭据。删除文件不能让凭据失效。

**先例（2026-11-26）：** `left/.en` 里的 `DISCORD_BOT_TOKEN` 随 `814bc85` 进了公开的
`main`/`dev`。文件已删（`7664f47`、`203bab6`）、`.gitignore` 已补、token 已轮换；
git 历史尚未重写。这就是不遵守本页的代价。

## 2. 不要写死本机绝对路径

- 代码、文档、默认值里不要出现 用户主目录下的绝对路径（`/home/…`、macOS 的 `~/Desktop/…` 之类）。
- 默认路径一律相对仓库根或 `Path(__file__).resolve().parents[N]`；可用环境变量覆盖
  （例：`QBOT_FEYNMAN_CORPUS`、`PYBOT_STATE_DIR`）。
- 本机能跑、CI 挂过：硬编码的本机用户目录默认值曾让右侧测试在 GitHub 上连续失败。

## 3. 语料红线：只上传代码、测试、人设 JSON、文档

以下**一律不进仓库**（`.gitignore` 已覆盖，提交前仍须自查）：

- 任何语料原文件：`*.jsonl` 原文库、`p8_corpus_*`、`cpt_zh.jsonl`
- 语料派生大产物：字库 `feynman_style_vault*.json`、拓扑 `*.npz`
- 运行时状态：`*.state.json`、`snapshots/`、`data/kb_learned/`、`data/assoc/`
- 他人的私人资料、聊天记录、私人训练数据

## 4. 提交前必须过的检查

```sh
python3 -m unittest tests.test_repo_hygiene   # 凭据文件名、个人路径、红线路径是否被忽略
python3 -m unittest discover -s tests         # 合并层测试
cd right && python3 -m unittest discover -s omega_mind/tests && cd ..   # 右脑测试
git status                                    # 再扫一眼：有没有不该进的东西
```

任何一项红，不要推。汇报时如实说哪项红、为什么。

## 5. 分支与发布

- `main` 只有主负责人能推；特性分支 → `dev` → 里程碑时 `dev` → `main`。
- 提交作者用能关联 GitHub 的邮箱（`227941007+chenyiyiagnes-bit@users.noreply.github.com`），
  不要用本机邮箱。
- 不要 force push 公共分支，除非主负责人明确批准（历史重写是例外操作）。

## 6. 对外的动作要授权

发布、发帖、发消息到外部服务、付费、删除远端数据、登录别人的账号——即使是在完成
被分派的任务，也要先得到主负责人明确同意。

## 7. 分工与权限边界

| 角色 | 负责 | 权限边界 |
|---|---|---|
| 主负责人（开发窗口） | 设计判断、核心代码、审所有交回的 diff、合并 | **唯一**能推 `dev` / `main` 的一方 |
| 审核（Gemini） | 从 `dev` 拉 `review/gemini`；CI 失败初筛；写 PR 描述、release notes、文档 diff；评审意见 | 只读 + 开 PR；可推 `review/gemini`，**不得**推 `dev` / `main` |
| 批量实现（DSH headless） | 边界清楚、验收标准写得死的活：批量改文件、跑测试修红、按规格实现、大范围整理 | 只交 diff / PR，不碰远端 |
| 老庄开发（Astra） | `persona/laozhuang` 分支 | 只推该分支 |

原则：

- **审核与发布分离。** 谁审谁不推。让同一个人既当独立检查又当发布口，评审会退化成走形式。
- **对外的闸门只有一个。** 合并进 `dev` / `main`、打 tag、改仓库设置、force push，一律经主负责人；
  涉及对外发言、发布、付费、删除、登录的，主负责人还须再向仓库所有者取得明确同意。
- **凭据不扩散。** 审核和准备性工作不需要写权限，就不给写权限。GitHub token 只存在于
  主负责人的本机环境，不写进任何配置文件，不交给其他 agent，不进 commit message。
- **交回的活要验过才算完成。** 收到 DSH / Gemini / Astra 的产物，主负责人读 diff、跑测试；
  没跑通就如实说，不把派出去的活说成已完成。

分支与上传细节见 `COORDINATION.md`。
