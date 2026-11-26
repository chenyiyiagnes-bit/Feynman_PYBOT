"""切换产品字库/拓扑图：只改 .env 里两个已知键，其余行逐字节保留。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/switch_vault.py --show
    PYTHONPATH=. .venv/bin/python -B scripts/switch_vault.py \
        --vault data/feynman_style_vault_p15.json \
        --topology data/topology_graph_p15.npz

规则：
- 只写 QBOT_FEYNMAN_VAULT / QBOT_FEYNMAN_TOPOLOGY，不读也不打印其它值；
- 写入前校验字库能被 Vault 加载、拓扑图能被拓扑引擎加载；
- 原文件先备份为 .env.bak-<时间戳>，再原子替换。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

KEYS = ("QBOT_FEYNMAN_VAULT", "QBOT_FEYNMAN_TOPOLOGY")
DEFAULT_VAULT = "data/feynman_style_vault_p15.json"
DEFAULT_TOPOLOGY = "data/topology_graph_p15.npz"


def read_env(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def current(lines: list[str]) -> dict:
    got = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k in KEYS:
            got[k] = v.strip()
    return got


def rewrite(lines: list[str], updates: dict) -> list[str]:
    out, seen = [], set()
    for line in lines:
        s = line.strip()
        k = None
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
        if k in updates:
            out.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    return out


def _validate(vault: str, topology: str) -> None:
    from plugins.feynman_typesetter import Vault

    if vault:
        p = Path(vault)
        if not p.is_file():
            raise SystemExit(f"字库不存在：{p}")
        v = Vault(p)
        print(f"  字库 OK：{p}（bank_id {v.bank_id}，"
              f"{sum(len(x) for x in v.by_slot.values())} 个字粒）")
    if topology:
        import plugins.topology_builder as tb

        p = Path(topology)
        if not p.is_file():
            raise SystemExit(f"拓扑图不存在：{p}")
        old = tb.TOPOLOGY_PATH
        tb.TOPOLOGY_PATH = p
        try:
            eng = tb.TopologyManifoldEngine.load()
        finally:
            tb.TOPOLOGY_PATH = old
        print(f"  拓扑图 OK：{p}（topo_hash {eng.topo_hash}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env")
    ap.add_argument("--vault", default=None)
    ap.add_argument("--topology", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env_path = Path(args.env)
    lines = read_env(env_path)
    cur = current(lines)
    if args.show or (not args.vault and not args.topology):
        print(f"当前（{env_path}）：")
        for k in KEYS:
            print(f"  {k} = {cur.get(k) or '（未设置，用代码默认）'}")
        if args.show:
            return

    updates = {}
    if args.vault:
        updates["QBOT_FEYNMAN_VAULT"] = args.vault
    if args.topology:
        updates["QBOT_FEYNMAN_TOPOLOGY"] = args.topology
    print("待切换：")
    _validate(args.vault, args.topology)

    changed = {k: v for k, v in updates.items() if cur.get(k) != v}
    if not changed:
        print("已经是目标配置，未改动。")
        return
    if args.check_only:
        print("校验通过；--check-only 不写文件。")
        return
    new_lines = rewrite(lines, updates)
    if args.dry_run:
        print("dry-run 结果（只列这两个键）：")
        for k in changed:
            print(f"  {k}={updates[k]}")
        return

    if env_path.exists():
        backup = env_path.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(env_path, backup)
        print(f"已备份 → {backup.name}")
    tmp = env_path.with_suffix(".tmp-switch")
    tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.replace(tmp, env_path)
    print(f"已写入 {env_path}：")
    for k in changed:
        print(f"  {k}={updates[k]}")
    print("重启 uvicorn 后生效。")


if __name__ == "__main__":
    main()
