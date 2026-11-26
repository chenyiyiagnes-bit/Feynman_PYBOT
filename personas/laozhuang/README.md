# 老庄（草稿）

第二套示例人设，用于验证「换人设不用改代码」。

- 原文：《道德经》《庄子》——公有领域，可随仓库附带。
- 译文：只用公有领域译本（如理雅各 1891）或自译，不用现代版权译本。
- 待办：知识卡（道、无为、齐物、养生、寓言）、语料切片、与费曼人设的 A/B。

## 交付清单

1. `knowledge/*.jsonl`：一行一张卡，最小字段 `{"id","type":"fact","title","text","tags"}`。
   标题写读者会搜的词（如「无为是什么意思」），正文用原文 + 一句白话。
   手头有 Markdown 笔记时可直接转：
   `python3 tools/import_cards.py 笔记.md -o personas/laozhuang/knowledge/notes.jsonl --tags 老庄`
2. `corpus/` 下的原文切片只放公有领域文本；版权译本一律不放。
3. 每加完一批：
   `python3 tools/doctor.py --persona laozhuang`（看卡片数与告警）
   `python3 scripts/chat.py --persona laozhuang`（试聊十来个问题）
   `python3 -m unittest discover -s tests -t .`（全绿再提交）
4. 与费曼人设做 A/B：同一组问题各问一遍，记录「答对 / 不答 / 答偏」三类的条数。
