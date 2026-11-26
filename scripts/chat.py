#!/usr/bin/env python3
"""Talk to a persona PYBOT in the terminal.

    python3 scripts/chat.py                 # 费曼，中文
    python3 scripts/chat.py --persona laozhuang
    python3 scripts/chat.py --lang en --persona feynman
    python3 scripts/chat.py --debug         # 显示 fact source / 左脑策略

No network, no API keys: the mind runs on the cards in personas/<id>/knowledge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime import (DEFAULT_PERSONAS, build, default_state_path,  # noqa: E402
                         resolve_persona)
from bridge import discover, load  # noqa: E402

HELP = """\
/help   显示这份帮助
/state  看看此刻的情绪状态和姿态
/save   立刻存盘（默认每轮自动存）
/quit   退出"""


def show_state(engine) -> str:
    state = engine.state.values
    cubie = getattr(engine.cube, "current", None) or getattr(engine.cube, "name", "")
    lines = [f"  状态 {k}={v:+.2f}" for k, v in state.items()]
    lines.append(f"  姿态 {cubie or engine.last_left_brain.get('posture', '-')}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", default="feynman-zh",
                        help="persona id under personas/ (default: feynman)")
    parser.add_argument("--lang", choices=("zh", "en"), default=None,
                        help="reply language (default: persona file)")
    parser.add_argument("--bridge", action="store_true",
                        help="format selected evidence with the left brain")
    parser.add_argument("--state", default=None,
                        help="state file to save between runs")
    parser.add_argument("--no-state", action="store_true",
                        help="do not persist anything")
    parser.add_argument("--debug", action="store_true",
                        help="show the fact source after each reply")
    parser.add_argument("--list", action="store_true", help="list personas and exit")
    args = parser.parse_args(argv)

    if args.list:
        for pid in discover(DEFAULT_PERSONAS):
            print(pid)
        return 0

    state = None
    if not args.no_state:
        state = args.state or default_state_path(args.persona)
        Path(state).parent.mkdir(parents=True, exist_ok=True)

    engine = build(args.persona, language=args.lang,
                   enable_bridge=args.bridge or None, state_path=state)

    if args.bridge:
        responder = engine.left_responder
        if responder is None or not responder.evidence_available():
            reason = (responder.last_error if responder is not None
                      else "LeftResponder not constructed")
            print(f"证据排版不可用：{reason}（见 docs/bridge.md）")

    persona = load(resolve_persona(args.persona))
    speaker = persona.name or args.persona
    print(f"PYBOT · {args.persona} · {engine.language} · "
          f"知识卡 {len(engine.kb.docs)} 张 · 输入 /help 看命令")
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit", "quit", "exit"):
            return 0
        if text == "/help":
            print(HELP)
            continue
        if text == "/state":
            print(show_state(engine))
            continue
        if text == "/save":
            engine.save()
            print("  已存盘" + (f" → {state}" if state else "（未开启持久化）"))
            continue
        result = engine.turn(text)
        print(f"{speaker}> {result.reply}")
        if args.debug:
            print(f"    [{result.source}] style={engine.last_left_brain.get('posture', '-')}")


if __name__ == "__main__":
    raise SystemExit(main())
