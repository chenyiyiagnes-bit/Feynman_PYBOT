# Feynman_PYBOT

一个不依赖大模型、可以「一键换人设」的角色心智框架：右脑管情绪与姿态，左脑管措辞，中间用一条可审计的接口连起来。

- **右脑**：26 个方块（事件评价器）→ 7 维情感状态 → 语气/篇幅/检索策略 → 真 3×3×3 魔方记录对话历史，带疲劳、冷却、睡眠复位。
- **左脑**：活字排版。只从词库里选字、排序、留白，从不自己造句子。
- **人设**：一个 `persona.json` 定义性格、口头禅、触发器和知识卡目录，不写代码就能换角色。

> 状态：左右脑合并进行中，尚未发布。当前仓库为开发骨架。

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
```

不需要 API key、不需要联网。对话状态默认保存在 `~/.feynman_pybot/<id>.mind_state.json`，
不想保存就加 `--no-state`。左脑排版器默认关闭（`--bridge` 或 `OMEGA_LEFT_BRIDGE=1` 才开，
需要自备 vault，见 `docs/bridge.md`）。

## 两套示例人设

| 人设 | 状态 | 语料 |
|---|---|---|
| 费曼 | 开发中 | 随仓库附带自建知识卡；原书语料需自备 |
| 老庄 | 规划中 | 《道德经》《庄子》原文属公有领域，可整本附带 |

## 语料与版权

本仓库**不附带任何受版权保护的语料**。知识卡为项目自建整理，按 CC-BY-4.0 发布；
引擎只提供加载器，语料路径由使用者自行配置。详见 `contracts/persona.md`。

## 许可

代码 MIT；人设文本 CC-BY-4.0。第三方语料不在许可范围内。
