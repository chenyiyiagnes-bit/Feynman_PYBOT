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

from app.runtime import DEFAULT_PERSONAS, build, default_state_path  # noqa: E402
from bridge import discover  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", default="feynman",
                        help="persona id under personas/ (default: feynman)")
    parser.add_argument("--lang", choices=("zh", "en"), default=None,
                        help="reply language (default: persona file)")
    parser.add_argument("--bridge", action="store_true",
                        help="enable the left-brain typesetter (needs OMEGA_LEFT_BRIDGE data)")
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
        if responder is None or not responder.available():
            reason = (responder.last_error if responder is not None
                      else "LeftResponder not constructed")
            print(f"桥未启用：{reason}（需要 OMEGA_LEFT_BRIDGE 数据，见 docs/bridge.md）")

    print(f"PYBOT · {args.persona} · {engine.language} · "
          f"知识卡 {len(engine.kb.docs)} 张 · 输入 /quit 退出")
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
        result = engine.turn(text)
        print(f"费曼> {result.reply}" if args.persona == "feynman" else f"{args.persona}> {result.reply}")
        if args.debug:
            print(f"    [{result.source}] style={engine.last_left_brain.get('posture', '-')}")


if __name__ == "__main__":
    raise SystemExit(main())
