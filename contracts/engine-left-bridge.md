# 变更说明：MindEngine 接入左脑排版器（附加式）

状态：已由主负责人实施（2026-11-26）。**纯附加，默认关闭**。

## 改动

1. `MindEngine.__init__` 新增可选参数 `left_responder=None`（鸭子类型，只要求有
   `.reply(text, state, persona, cube_depth, cube_verbosity, nonce, session_id)`）。
   右脑仍然不导入 `bridge/`，`None` 时构造路径与之前完全一致。
2. `turn()` 内部把 `self.cube_feedback.bias(self.cube)` 只算一次，变量名
   `cube_bias`，传给 `left_brain.plan_for` 和 `_base_reply`。数值与之前相同。
3. `_base_reply` 中文分支顺序变为：**KB 卡片 → 左脑排版器（可选）→ 内置
   knowledge → fallback**。英文分支不变。
4. 新私有方法 `_left_reply`：responder 为 None、抛异常、返回 None、返回非中文
   或空串，一律返回 None，走原路。

## 不变量

- `turn(text, base_reply=..., user=...)` 签名不变，`TurnResult` 字段不变。
- `left_responder` 为 None 时，`turn()` 的返回与改动前逐字段相同
  （`tests/test_engine_bridge.py::TestOffPathIdentical` 断言 reply/base_reply/source）。
- 左脑永远不提供事实：它只读 `left/` 的 vault 并排版，事实仍来自卡片。
- `source` 新增一个取值 `"left:typesetter"`，只在该路径命中时出现。

## 给联想分支的提醒

`engine.py` 归主负责人。若 `assoc/association` 也要改 `_base_reply` 或
`turn()` 调用顺序，先在这个文件加一节，再改代码，避免合并时互相覆盖。
