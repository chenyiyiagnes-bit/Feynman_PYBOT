# 文档索引

| 文件 | 讲什么 |
|---|---|
| `contracts/persona.md` | `persona.json` 的字段格式（schema 1），写人设前先看这个 |
| `contracts/engine-left-bridge.md` | 右脑引擎接入左脑排版器的接口约定、失败关闭语义 |
| `docs/bridge.md` | 左右脑桥的映射表、环境变量、端到端验证结果与已知限制 |
| `docs/cards.md` | 拆书工具：覆盖率/台账、`support` 出处校验、匿名盲评；每个工具都写明「不证明什么」 |
| `docs/segmentation.md` | 切词的唯一真源、每包自带的 golden 题集、`eval_segmentation` 的判据与 CI 闸门、词表图的开关 |
| `docs/reviews/` | 审核报告存放处与模板（Gemini 分支提交到这里） |
| `COORDINATION.md` | 分支流、文件所有权、语料红线 |

## 按目的找

- **想跑起来**：`README.md` 快速上手 → `python3 tools/doctor.py` 自检
- **想做人设**：`contracts/persona.md` → `tools/new_persona.py` → `personas/laozhuang-zh/README.md` 的交付清单
- **想喂知识**：`personas/<id>/knowledge/README.md` 或 `tools/import_cards.py`
- **想改桥**：`docs/bridge.md` 先读「已知限制」，再改 `contracts/`
- **想看测试怎么跑**：`README.md` 的测试一节；CI 在 `.github/workflows/tests.yml`
