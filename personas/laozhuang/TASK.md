# 任务：完成「老庄 PYBOT」人设包

## 目标

在 `personas/laozhuang/` 内做出一个能直接运行的第二示例人设，用来证明
「换人设不用改代码」。引擎（`right/`、`left/`）不在改动范围内。

## 交付物

1. `persona.json` —— 按 `contracts/persona.md` 定稿。26 个方块，core/edge/corner = 6/12/8。
2. `knowledge/*.jsonl` —— 知识卡。每条至少含 `id, type, title, text, tags`；
   涉及原文的必须带 `source` 字段。
3. `corpus/` —— 原文语料切片（仅限公有领域内容，见红线）。
4. `tests/` 或 `personas/laozhuang/test_persona.py` —— 校验：26 方块数量与角色分布、
   JSON 可被加载、触发器能编译、知识卡字段完整。
5. `README.md` —— 人设说明：他是谁、怎么说话、对什么起反应、和费曼人设的区别。

## 红线

- **只允许公有领域文本**：《道德经》《庄子》原文（王弼本、郭象注等古本）。
  英文译文只能用公有领域译本（如 James Legge 1891）或自行翻译。
  现代版权译本（陈鼓应、任继愈等）一律不得复制进仓库。
- 不得修改 `right/`、`left/`、`bridge/`、`contracts/`。
- 不得提交任何运行时状态、语料原库（`cpt_zh.jsonl` 等）或生成的大产物。
- 中文/英文回复不混语言；知识卡与口头禅都要两种语言各一套。

## 验收

- `python3 -m pytest personas/laozhuang/` 全绿（或 unittest 等价）。
- `python3 tools/validate_persona.py personas/laozhuang/persona.json` 通过。
- 用 `--persona laozhuang` 起一次对话，能答 20 个预设问题不崩溃、不串语言。
- PR 目标分支：`dev`，不是 `main`。

## 风格参考

- 基线：高能量、低唤醒、低皮质醇、高效价；压力恢复极快。
- 说话：短句、反问、寓言；不打术语牌，不争输赢。
- 触发器重点：争辩、效率至上、标准答案 → 不激化，转向"换个角度看"。
