# 知识卡：写进去以后，怎么证明它没丢、没跑

一本书拆成卡片，最容易坏的不是「答得不好」，是**卡悄悄没了**：某一行没有
`text` 字段、某一行混进了中文、某一行不是合法 JSON——加载器跳过它，不报错，
从此这个问题永远没人能答。这一页讲三个工具，按「拆完卡 → 上线前 → 上线后」
的顺序用。

## 1. 覆盖率：每一行都有去处

```sh
python3 tools/card_coverage.py                      # 全部包
python3 tools/card_coverage.py --package feynman-zh --json out.json
python3 tools/card_coverage.py --write-ledger data/card_ledger.json
python3 tools/card_coverage.py --check-ledger data/card_ledger.json
```

它用和加载器**同一套规则**重读每一行，把结果分成两类：

- **loaded**：成了 `Doc`，能被检索到；
- **dropped + 原因**：`no-answer-text`（这一 kind 的正文不在 `text` 里，比如
  `people`/`relations` 行）、`not-english`（英文包里混了中文行）、`bad-json`。

出现 `bad-json` 或 `unaccounted`（工具数出来的卡和 KB 里的卡对不上）就以 1 退出。
「有原因地跳过」不算失败——`people.jsonl` 本来就不该被检索；没有原因地消失才算。

`--write-ledger` 给每张卡记下 `block_id`（`包:文件:卡 id`）和整行内容的
`text_sha256`。以后用 `--check-ledger` 就能发现「这张卡被改过」「这张卡不见了」
「新加了卡」——改卡本身不是错，改了却没意识到才是。

**它不证明什么：** 卡能被加载 ≠ 卡答得对。那要看 `tools/eval_segmentation.py`
和 golden 集。

## 2. 出处：`support` 块

卡片可以带可选字段 `support`，记录这条判断是从哪句话来的：

```json
"support": [
  {
    "source": "FLP I-1-1",
    "quote": "先看它是什么样、怎么动。",
    "source_file": "reference/flp-i.txt",
    "quote_sha256": "<quote 的 sha256，utf-8>"
  }
]
```

```sh
python3 tools/card_support.py                 # 校验已有 support 块
python3 tools/card_support.py --strict        # 没写 support 的卡也算失败
python3 tools/card_support.py --list-missing  # 列出还没写的卡
```

`source_file` 是**本机**的原文文件，不入库（语料红线）；相对路径先按包目录、
再按仓库根找。文件在：引文必须逐字出现（空白不计），`quote_sha256` 必须对上；
文件不在：记为 **unverified**，不算 valid。格式坏、哈希对不上、引文找不到：
记 invalid，退出 1。

**它不证明什么：** 引文真实 ≠ 引文支持卡片的说法。一句原话可以被正确地摘下来，
却和卡片的结论无关——这个判断只能人来下。工具管形状，不管含义。

## 3. 盲评：匿名两两对比

改了一版人设（或多教了一批卡），想知道「是不是变好了」：

```sh
python3 tools/ab_eval.py make --a feynman-zh --b feynman-zh-v2 \
    --questions questions.jsonl --pairs pairs.jsonl --key key.json --seed 7
# 把 pairs.jsonl 给评审，只给这个文件，不给 key.json
python3 tools/ab_eval.py score --pairs pairs.jsonl --key key.json --votes votes.jsonl
```

`questions.jsonl` 一行一题（`{"id","question"}`）；`votes.jsonl` 一行一票
（`{"id","pick":"left"|"right"|"tie"}`）。每对的左右顺序按 `--seed` 打乱，
答案文件里不出现任何包名；`key.json` 单独保存，评审看不到。

**它不证明什么：** 读起来更顺 ≠ 答得更对；十几道题的胜率是噪声，报告里同时
给出票数，不要只报赢家。
