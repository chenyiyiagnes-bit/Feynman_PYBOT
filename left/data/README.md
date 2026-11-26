# left/data —— 由使用者本地生成，不随仓库发布

左脑排版器需要一份「字库」和一份「拓扑图」。它们是从语料里统计出来的派生数据，
语料本身受版权保护，所以本仓库不附带任何现成产物，只附带构建脚本。

```bash
python3 scripts/build_vault_p15.py     # 生成 data/feynman_style_vault_p15.json
python3 scripts/build_p15_inventory.py # 生成库存与拓扑
```

语料路径由环境变量指定：

```bash
export QBOT_FEYNMAN_CORPUS=/path/to/your/corpus.jsonl
export QBOT_FEYNMAN_VAULT=data/feynman_style_vault_p15.json
```

没有语料时：`bridge.LeftResponder`（`OMEGA_LEFT_BRIDGE=1` 才启用）会静默降级为不发言；
直接调用 `left/plugins/feynman_typesetter.py` 的 `Vault(...)` 会抛 `FileNotFoundError`，
此时先按上面的环境变量备好语料与字库，或用 `left/scripts/build_vault*.py` 先铸字。
