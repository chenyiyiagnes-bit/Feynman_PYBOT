"""Scaffold a new persona: personas/<id>/persona.json + knowledge/.

    python3 tools/new_persona.py mybot
    python3 tools/new_persona.py laozi --name 老子 --from feynman
    python3 tools/new_persona.py mybot --dry-run

The default base is the neutral ``generic`` persona: all 26 cubies with the
stock triggers, flat baselines. ``--from <existing>`` copies another persona
(baselines, voice, switches, triggers) and only changes id/name — the fastest
way to make a variant of a character you already tuned.

The new persona starts with an empty knowledge/ directory: it will chat in
character but has no facts to answer from until cards are added.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge import load as load_persona  # noqa: E402
from tools.export_persona import build as export_config  # noqa: E402

KNOWLEDGE_README = """\
# 知识卡目录

把 `.jsonl` 知识卡放在这里，一行一张。最小字段：

    {"id": "fact-001", "type": "fact", "title": "主题", "text": "答案正文", "tags": ["关键词"]}

引擎会按标题（权重 4）、标签（3）、正文（1）建索引；命不中就不答，不会编。
写卡时优先写"能一句话说清的事实"，再配 2-4 个别人真的会搜的词。
"""

PERSONA_README = """\
# {name}

由 `tools/new_persona.py` 从 `{base}` 生成。

- `persona.json`：情绪基线、语速衰减、口头禅、26 个方块触发器。
- `knowledge/`：知识卡，目前为空。

下一步：

1. 在 `knowledge/` 放 `.jsonl` 卡片（格式见该目录 README）。
2. `python3 tools/validate_persona.py personas/{pid}/persona.json` 校验人设。
3. `python3 scripts/chat.py --persona {pid}` 试聊，再按手感调
   `baselines` / `taus` / `voice`。
"""


def scaffold(persona_id: str, name: str, base: str, root: Path,
             dry_run: bool = False) -> Path:
    target = root / persona_id / "persona.json"
    if target.exists() and not dry_run:
        raise SystemExit(f"refusing to overwrite: {target}")

    data = export_config(base)
    data["id"] = persona_id
    data["name"] = name or persona_id
    data["description"] = data.get("description", "")
    data["knowledge"]["kb_dirs"] = ["knowledge"]

    if dry_run:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:400] + " ...")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    knowledge = target.parent / "knowledge"
    knowledge.mkdir(exist_ok=True)
    (knowledge / "README.md").write_text(KNOWLEDGE_README, encoding="utf-8")
    (target.parent / "README.md").write_text(
        PERSONA_README.format(name=name or persona_id, base=base, pid=persona_id),
        encoding="utf-8")

    # Fail loudly here rather than at chat time.
    load_persona(target)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("id", help="persona id, also the directory name")
    parser.add_argument("--name", default="", help="display name (default: id)")
    parser.add_argument("--from", dest="base", default="generic",
                        help="base persona: generic|feynman|laozhuang (default: generic)")
    parser.add_argument("--root", default=str(ROOT / "personas"),
                        help="personas root (default: personas/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the persona.json instead of writing files")
    args = parser.parse_args(argv)

    target = scaffold(args.id, args.name, args.base, Path(args.root),
                      dry_run=args.dry_run)
    if not args.dry_run:
        print(f"created {target.parent}")
        print(f"next: put cards in {target.parent / 'knowledge'}, then")
        print(f"      python3 scripts/chat.py --persona {args.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
