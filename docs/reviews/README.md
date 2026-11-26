# 审核报告存放处

分支 `review/gemini` 只放审核报告，不改代码。

- 报告路径：`docs/reviews/YYYY-MM-DD-<reviewer>.md`
- 同一问题多次审核用 `YYYY-MM-DD-<reviewer>-2.md`
- 提交后开 PR 目标 `dev`，标题 `review: <scope>`

报告只写发现，不直接改代码；修复由主负责人在 `dev` 上做。
