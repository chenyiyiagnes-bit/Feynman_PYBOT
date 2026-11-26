# Feynman_PYBOT

[![tests](https://github.com/chenyiyiagnes-bit/Feynman_PYBOT/actions/workflows/tests.yml/badge.svg?branch=dev)](https://github.com/chenyiyiagnes-bit/Feynman_PYBOT/actions/workflows/tests.yml)

一个不依赖大模型、可以「一键换人设」的角色心智框架：右脑管情绪与姿态，左脑管措辞，中间用一条可审计的接口连起来。

- **右脑**：26 个方块（事件评价器）→ 7 维情感状态 → 语气/篇幅/检索策略 → 真 3×3×3 魔方记录对话历史，带疲劳、冷却、睡眠复位。
- **左脑**：活字排版。只从词库里选字、排序、留白，从不自己造句子。
- **人设**：一个 `persona.json` 定义性格、口头禅、触发器和知识卡目录，不写代码就能换角色。

> 状态：可运行的开发版（`dev` 分支，测试 + CI 全绿）；正式发布版将同时带费曼与老庄两个示例人设，届时并入 `main`。

## 目录

```
personas/<id>/persona.json   人设定义（格式见 contracts/persona.md）
personas/<id>/knowledge/     该人设的知识卡
app/                         装配层：人设 → 可运行的 MindEngine
bridge/                      右脑 → 左脑的映射层
contracts/                   接口与人设格式契约
right/                       右脑引擎（omega_mind）
left/                        左脑排版器（代码，语料自备）
scripts/chat.py              终端聊天入口
tools/                       导出/校验工具
tests/                       合并层测试
```

## 快速上手

```bash
# 费曼，中文，读 personas/feynman/knowledge 的卡片回答
python3 scripts/chat.py

# 换人设
python3 scripts/chat.py --persona laozhuang

# 英文模式 / 看每句答案来自哪张卡
python3 scripts/chat.py --lang en
python3 scripts/chat.py --debug

# 列出仓库里的人设
python3 scripts/chat.py --list

# 装好先自检：Python 版本、人设能否加载、卡片数、桥状态
python3 tools/doctor.py
```

不需要 API key、不需要联网。对话状态默认保存在 `~/.feynman_pybot/<id>.mind_state.json`，
不想保存就加 `--no-state`。左脑排版器默认关闭（`--bridge` 或 `OMEGA_LEFT_BRIDGE=1` 才开，
需要自备 vault，见 `docs/bridge.md`）。

## 做一个自己的角色

```bash
# 中立人设（26 个方块、平基线、空知识库）
python3 tools/new_persona.py mybot --name 我的角色

# 或从已调好的人设改个名（保留基线、口头禅、触发器）
python3 tools/new_persona.py mybot --from feynman

# 往 personas/mybot/knowledge/ 放 jsonl 卡片，然后
python3 tools/validate_persona.py personas/mybot/persona.json
python3 scripts/chat.py --persona mybot
```

手头的笔记可以直接转成卡片（按标题或空行切分，不改写原文）：

```bash
python3 tools/import_cards.py 我的笔记.md -o personas/mybot/knowledge/notes.jsonl --tags 我的主题
```

知识卡格式见 `personas/mybot/knowledge/README.md`：一行一张，`title`/`text`/`tags`
就够；检索靠标题和标签，所以标题要写别人真会搜的词；命不中就不答，不会编。

## 两套示例人设

| 人设 | 状态 | 语料 |
|---|---|---|
| 费曼 | 可聊（310 张卡，中英双语） | 随仓库附带自建知识卡；原书语料需自备 |
| 老庄 | 人设骨架就位，知识卡共建中 | 《道德经》《庄子》原文属公有领域，可整本附带 |

## 语料与版权

本仓库**不附带任何受版权保护的语料**。知识卡为项目自建整理，按 CC-BY-4.0 发布；
引擎只提供加载器，语料路径由使用者自行配置。详见 `contracts/persona.md`。

## 许可

代码 MIT；人设文本 CC-BY-4.0。第三方语料不在许可范围内。
