# 变更说明：MindEngine 接入左脑排版器（附加式）

## 知识包回答的证据排版

知识包命中卡片或选中已审核的联想边后，右脑可以调用
`left_responder.plan_evidence(evidence, *, source, language, detail, dialogue_act, candidates)`。
`candidates` 是右脑从原句中识别出的安全停顿位置，使用 Python 字符偏移。
左脑返回其中一组递增的偏移。右脑逐项核验，再在这些位置断行。短追问可在现有分号处断行，
较长的说明可在完整句子间分段。引号、括号内部与公式标记受保护。原句次序、文字和标点保留。

这条路径沿用 `--bridge` 与 `OMEGA_LEFT_BRIDGE=1`，使用随仓库提供的规则排版器。
证据排版无需 vault、外部模型或派生词表。右脑保留原 `base_reply` 和 `source`，
并在 `retrieval.evidence_layout` 中记录实际使用的断行偏移。
左脑缺失、返回越界位置或无可用断行时按原流程回答。拒答、社交回应、原样复述和调用方提供
的正文直接走原流程。知识包缺证据时仍拒答，独立原型保留原有的 vault 排版接口。

## 独立原型的 vault 排版

状态：已由主负责人实施（2026-11-26）。**纯附加，默认关闭**。

## 改动

1. `MindEngine.__init__` 新增可选参数 `left_responder=None`（鸭子类型，只要求有
   `.reply(text, state, persona, cube_depth, cube_verbosity, nonce, session_id)`）。
   右脑仍然不导入 `bridge/`，`None` 时构造路径与之前完全一致。
2. `turn()` 内部把 `self.cube_feedback.bias(self.cube)` 只算一次，变量名
   `cube_bias`，传给 `left_brain.plan_for` 和 `_base_reply`。数值与之前相同。
3. 独立原型的中文分支顺序为：**KB 卡片 → 左脑排版器（可选）→ 内置
   knowledge → fallback**。
4. 新私有方法 `_left_reply`：responder 为 None、抛异常、返回 None、返回非中文
   或空串，一律返回 None，走原路。

## 不变量

- `turn(text, base_reply=..., user=...)` 签名不变，`TurnResult` 字段不变。
- `left_responder` 为 None 时，`turn()` 的返回与改动前逐字段相同
  （`tests/test_engine_bridge.py::TestOffPathIdentical` 断言 reply/base_reply/source）。
- `source` 的 `"left:typesetter"` 只在独立原型的 vault 路径命中时出现。

## 给联想分支的提醒

`engine.py` 归主负责人。若 `assoc/association` 也要改 `_base_reply` 或
`turn()` 调用顺序，先在这个文件加一节，再改代码，避免合并时互相覆盖。
