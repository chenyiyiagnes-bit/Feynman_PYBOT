# PYBOT 对话与证据改进

## 运行时行为

个人事实从明确的自述、迁移和撤销中提取。以轮次和时间记录生效与终止，当前询问只读当前事实，过去住址可从历史回答。疑问、假设、引语和转述跳过事实写入。旧快照可载入，撤销随新快照保存。会话用户分别保存近期话题，显式回到旧话题时检查卡片版本，多个主题同时匹配时请用户说清。每人保留最近八个话题，最多保留三十二位用户的主题列表。

知识包按固定检索预算与词项证据选事实。情绪和魔方影响详略和表达姿态。已选卡片的版本、候选、证据位置和已审核联想边写入内部检索诊断。明确虚构对象的实测值或既成事件会触发拒答。未达到证据门槛或结论冲突时走拒答与澄清。纯文本旧卡沿用原句；已有结构的教学单元、物理边、方法卡直接读取本包字段，内容负责人也可用下面的字段明确标注。

## 证据与联想标注

- 卡片可加上 \`passages\` 映射。键为 \`conclusion\`、\`mechanism\`、\`example\` 或 \`condition\`，值为原句或原句列表。每段必须逐字出现在卡片详细正文里。运行时记录字符范围，简答保留所有条件段。
- 证据块沿用 \`support\`，含 \`source\`、\`quote\`、\`quote_sha256\`、本机 \`source_file\`。内容负责人核实引文支持该段后标记 \`reviewed_by\`。运行时区分官方卡片摘要和已记录审核的原句。验证工具单独核验本机原文及其哈希。
- 同一范围内互相矛盾的卡片用相同 \`claim_key\` 和不同 \`claim_value\` 登记。卡片竞争时触发条件澄清。不同范围的结论可使用不同键。
- 新的人工联想边使用本包 \`language\`、\`relation\`、变量对 \`mapping\` 和 \`source\`。共同机制配 \`mechanism\`，差别与边界配 \`difference\`，英文沿用 \`mechanism_en\` 与 \`difference_en\`。可登记 \`misconceptions\` 作为反例。路径只走一条人工边。
- 右脑 CLI 的 \`/analogy CARD_ID\` 用已审核的变量对应、共同关系与边界提问。评分核对人工定义的短语和反例，语义判断仍由人工验收。普通教学单元也可配置 \`mechanism_rubric\` 的 \`required\` 和 \`misconceptions\`。

核查原文、生成首批物理卡清单的入口如下。清单排序取两套冻结题中目标卡的出现次数，作为审核优先级。原文、派生状态、审核材料留在本机。

\`\`\`sh
python3 tools/card_support.py
python3 tools/card_support.py --package feynman-zh --pilot 20 --json /tmp/pybot-evidence-pilot.json
\`\`\`

英文费曼包当前 293 行带回译标记。20 张中文物理卡试点的原文对照、独立科学审核和英文包的来源决定仍由内容负责人验收。

## 回放、盲评与回滚

| 变量 | 默认值 | 作用 |
|---|---|---|
| \`OMEGA_RETRIEVAL\` | \`legacy\` | \`shadow\` 同时记录 BM25 与 RRF，同样的候选预算。 \`rrf\` 只在已有词项证据且证据段足够时调整近分候选。 |
| \`OMEGA_POSTURE\` | \`affect_cube\` | \`fixed\` 使用基准表达策略，\`affect\` 只使用情绪，\`affect_cube\` 加入魔方反馈。 |
| \`OMEGA_STRUCTURED_EVIDENCE\` | 已装配的包为 \`1\` | 设 \`0\` 回到整卡表达方式。 |
| \`OMEGA_REVIEW_LEARNING\` | 已装配的包为 \`1\` | 梦后纠正、表扬建议和新类比进入审核队列。设 \`0\` 供隔离复现旧自动学习。 |
| \`OMEGA_LEFT_BRIDGE\` | \`0\` | 设 \`1\` 或传 \`--bridge\`，仅给已经选定的原句安排断行。 |

\`\`\`sh
python3 tools/dialogue_eval.py --mode legacy
python3 tools/dialogue_eval.py --mode shadow --full /tmp/pybot-holdout.json
python3 tools/dialogue_eval.py --mode rrf --posture affect
python3 tools/dialogue_eval.py --confirm-set tests/data/confirm_set_v1.json \\
    --confirmation-packages feynman-zh feynman-en
\`\`\`

\`dialogue_holdout_v1.json\` 含 8 组 31 轮人工编写对话，覆盖四个包。保存结果带代码版本、
包指纹、用例指纹、开关、候选召回、选卡、拒答以及本机单轮 p95。使用 \`--strict\` 时任何用例
不符即退出失败。跨情绪配置的人工评分用 \`tools/ab_eval.py make\` 的 \`--mode-a\`、\`--mode-b\`
与 \`turns\` 输入。评审只读匿名对话，可以分别投给连贯性、自然表达、角色及科学正确性。
科学和角色偏好需要各自的评审记录。

审核人员使用 \`tools/learning_review.py PACKAGE inspect|replay|approve|revoke\` 处理本机候选。
\`replay\` 和 \`approve\` 要求 \`--id\`、\`--proof\`、\`--holdout\`。证明文件包含
\`reviewer\`、\`source\`、\`quote\`、\`quote_sha256\`、\`source_file\`。
确认集是 JSON 数组，每项为 \`question\` 与预期 \`card\`，拒答值为 \`null\`。
试用副本要命中候选目标并保留黄金题和确认集已有正确结果。审核员确认引文支持事实后才批准。
\`revoke\` 按决策 ID 撤销，运行中的包下一轮重新载入决定。只读复制本机状态目录即可保留旧快照基线。

## 本地验证结果

比较基线为 \`66b44e6\`。实际对话通过 \`app.runtime.build()\` 装配。冻结对话集上，预定来源
从 3/11 到 11/11，回复文字检查从 3/10 到 10/10，指定卡片均为 7/8，拒答从 1/2 到 2/2。
仍未命中的题是 \`zh-mechanics\` 第一轮。30 道知识确认题的实际回答命中由 14/30 到 15/30。
同样取前三张，原检索、BM25、RRF 分别为 21/30、23/30、22/30，所以线上默认仍用原排序。

这些数字来自本机冻结对话及确认集。修正语法和续接问题后复跑了同一套对话，因此这里记录
的是可复现的回归结果。独立的科学审核和人工角色盲评结果仍待登记。
