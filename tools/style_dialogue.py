"""A short scripted dialogue per package, with the style audit switched on.

This is the P2 check: the audit is installed exactly as the App installs it
(``app.runtime.build`` + ``PYBOT_STYLE_GATE=1``), a coherent six-turn
conversation is run against the real engine, and every reply is printed with
the verdict it got. Nothing is blocked — the reply shown is what a user
would see.

State goes to a throwaway directory, so a test run never touches
``~/.pybot`` or the live persona state.

Usage:
  python3 tools/style_dialogue.py
  python3 tools/style_dialogue.py --package feynman-zh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIALOGUES = {
    "feynman-zh": [
        "你好，你是谁？",
        "什么是量子电动力学？",
        "能不能一句话讲清楚光到底是什么？",
        "我没听懂，再说慢一点。",
        "那引力为什么不能和量子力学统一？",
        "听了半天，你倒是给个准话啊。",
    ],
    "feynman-en": [
        "Hello, who are you?",
        "What is a photon?",
        "Explain that like I'm twelve.",
        "I still don't get it. Say it again, slower.",
        "Why can't gravity fit together with quantum mechanics?",
        "You keep dodging — just give me a straight answer.",
    ],
    "laozhuang-zh": [
        "你是谁？",
        "什么是无为？",
        "上善若水是什么意思？",
        "我最近很焦虑，怎么办？",
        "那你觉得我该不该辞职？",
        "说了半天，你也没给我答案。",
    ],
    "laozhuang-en": [
        "Who are you?",
        "What is wu wei?",
        "What does 'the highest good is like water' mean?",
        "I've been anxious lately. What should I do?",
        "So should I quit my job or not?",
        "You keep talking and never answer me.",
    ],
}


def run_package(name: str, lines: list[str], workdir: Path) -> list[dict]:
    os.environ["PYBOT_STYLE_GATE"] = "1"
    os.environ["PYBOT_STATE_DIR"] = str(workdir / "state")
    os.environ["PYBOT_STYLE_AUDIT_DIR"] = str(workdir / "audit")
    import importlib
    from app import runtime
    importlib.reload(runtime)
    engine = runtime.build(name)
    log = workdir / "audit" / f"{name}.jsonl"
    records = []
    for i, line in enumerate(lines, 1):
        result = engine.turn(line)
        # The last line the audit wrote is this turn's verdict.
        if log.exists():
            last = log.read_text(encoding="utf-8").strip().splitlines()[-1]
            verdict = json.loads(last)
        else:
            verdict = {"ok": None, "failed": [], "metrics": {}}
        records.append({"turn": i, "user": line, "reply": result.reply,
                        "source": result.source, "verdict": verdict})
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", action="append", default=[])
    ap.add_argument("--json", action="store_true", help="dump records as JSON")
    args = ap.parse_args(argv)
    names = args.package or list(DIALOGUES)

    summary = {}
    with tempfile.TemporaryDirectory(prefix="pybot-style-") as tmp:
        workdir = Path(tmp)
        for name in names:
            lines = DIALOGUES[name]
            records = run_package(name, lines, workdir)
            passed = sum(1 for r in records
                         if (r["verdict"].get("ok") is True))
            summary[name] = {"turns": len(records), "ok": passed}
            if args.json:
                print(json.dumps({"package": name, "records": records},
                                 ensure_ascii=False, indent=2))
                continue
            print(f"\n===== {name}: {passed}/{len(records)} replies inside band")
            for r in records:
                v = r["verdict"]
                m = v.get("metrics") or {}
                flag = "ok  " if v.get("ok") else "FAIL"
                print(f"  [{r['turn']}] {flag} {v.get('failed') or ''} "
                      f"len={m.get('mean_sentence_chars')} "
                      f"max={m.get('max_sentence_chars')} "
                      f"src={r['source']}")
                print(f"      U: {r['user']}")
                print(f"      B: {r['reply'][:200]}")
    print("\nsummary: " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
