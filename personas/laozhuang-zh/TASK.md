# 任务：完成「老庄 PYBOT」人设包

（2026-11-26 迁自 `persona/laozhuang` 分支，按新的单语言包结构重写。）

## 目标

在 `personas/laozhuang-zh/`（中文）与 `personas/laozhuang-en/`（英文）内做出能直接运行的
第二示例人设，用来证明「换人设不用改代码」。引擎（`right/`、`left/`、`bridge/`）不在改动范围内。

## 交付物

1. `persona.json` —— 按 `contracts/persona.md` 定稿。26 个方块，core/edge/corner = 6/12/8。
   包目录名 = `persona.json` 的 `id`；中英两包各一份，`language` 分别为 `zh` / `en`。
2. `knowledge/*.jsonl` —— 知识卡。每条至少含 `id, type, title, text, tags`；
   引用原文的必须带 `source` 字段（写篇名，不贴原文）。
3. `tests/` 或 `personas/laozhuang-zh/test_persona.py` —— 校验：26 方块数量与角色分布、
   JSON 可被加载、触发器能编译、知识卡字段完整、中英两包卡片一一对应。
4. `README.md` —— 人设说明：他是谁、怎么说话、对什么起反应、和费曼人设的区别。

## 红线

- **语料原文不随仓库发布**（与费曼包一致）：本机 `corpus/` 只用于核对，不入库；
  卡片正文可以引用一两句公有领域原文并注明出处。
- 原文仅限公有领域：《道德经》《庄子》原文（王弼本、郭象注等古本）；英文译文只能用
  公有领域译本（如 James Legge 1891）或自译，现代版权译本（陈鼓应、任继愈等）一律不得复制。
- 不得修改 `right/`、`left/`、`bridge/`、`contracts/`。
- 不得提交任何运行时状态或生成的大产物（`data/lexicon/`、`data/assoc/` 等）。
- 中英两包不混语言：知识卡与口癖各语言一套，英文包只用原文对应表达，禁止中文回译。

## 验收

- `python3 -m unittest discover -s tests` 全绿
- `python3 tools/validate_persona.py personas/laozhuang-zh/persona.json` 通过（英文包同理）
- 用 `--persona laozhuang-zh` 起一次对话，能答 20 个预设问题不崩溃、不串语言
- PR 目标分支：`dev`，不是 `main`

## 风格参考

- 基线：高能量、低唤醒、低皮质醇、高效价；压力恢复极快。
- 说话：短句、反问、寓言；不打术语牌，不争输赢。
- 触发器重点：争辩、效率至上、标准答案 → 不激化，转向「换个角度看」。

## 上传方式

见 `COORDINATION.md`：只推 `personas/laozhuang-*` 范围内的改动到 `persona/laozhuang` 分支，
提 PR 到 `dev`，不直接推 `dev` / `main`。
