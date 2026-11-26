"""把人工确认过的「笔记概念 → 主库字粒」映射编译成 overlay 文件。

不做自动相似度猜测：每条都必须显式给出 target 与 slot，编译时逐条核对
target 是否真在当前字库的该槽位里、bank_id 是否匹配。不合法就整份不写。

输入 JSONL（一行一个映射）：
    {"concept": "心跳", "target": "振动", "slot": "action", "note_ref": "notes/生物.md#L12"}

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_user_overlay.py \
        --notes notes_map.jsonl --out data/overlays/my-notes.json --overlay-id my-notes
    ... --vault data/feynman_style_vault_p15.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plugins.feynman_typesetter import Vault
from plugins.user_overlay import OVERLAY_VERSION, validate_overlay


def _notes(path: Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n} JSON 解析失败：{exc}")
        concept = str(row.get("concept") or "").strip()
        if not concept:
            raise SystemExit(f"{path}:{n} 缺少 concept")
        out.append({"alias": concept, "target": row.get("target", ""),
                    "slot": row.get("slot", ""),
                    "note_ref": str(row.get("note_ref") or ""),
                    "provenance": "user_note"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True, help="人工映射 JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay-id", default="")
    ap.add_argument("--vault", default="data/feynman_style_vault_p15.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vault = Vault(Path(args.vault))
    entries = _notes(Path(args.notes))
    overlay_id = args.overlay_id or Path(args.out).stem
    data = {"version": OVERLAY_VERSION, "overlay_id": overlay_id,
            "bank_id": vault.bank_id, "entries": entries}
    errors = validate_overlay(data, vault)
    if errors:
        print("❌ overlay 不合规：", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    if args.dry_run:
        print(f"✅ 可编译：{len(entries)} 条映射，绑定 {vault.bank_id}（dry-run，未写文件）")
        return
    out = Path(args.out)
    if out.exists():
        print(f"❌ {out} 已存在：overlay 也是版本化产物，请换新文件名或先人工移走",
              file=sys.stderr)
        raise SystemExit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"✅ {out}：{len(entries)} 条映射，绑定 {vault.bank_id}")


if __name__ == "__main__":
    main()
