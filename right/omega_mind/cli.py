"""Interactive CLI: `python -m omega_mind --persona feynman`."""

from __future__ import annotations

import argparse
import select
import sys

from .config import get_persona
from .cube import render_net
from .engine import LANGUAGES, MindEngine, has_cjk
from .feynman_kb import DEFAULT_DIRS
from .profiler import format_profile, profile_corpus
from .view import render_ascii


def reflect_summary(audit: dict, language: str) -> str:
    """One line about the dream's self-audit, from its stored dict."""
    findings = audit.get("findings") or []
    repaired = audit.get("repaired", 0)
    if language == "en":
        if not findings:
            return "dream audit: nothing went wrong"
        if audit.get("read_only"):
            return (f"dream audit: {len(findings)} trouble(s) noted, "
                    f"read-only (cortisol too high)")
        return (f"dream audit: {len(findings)} trouble(s) noted, "
                f"{repaired} repair(s)")
    if not findings:
        return "梦后复盘：没有发现明显不对劲的地方"
    if audit.get("read_only"):
        return f"梦后复盘：发现 {len(findings)} 处问题，情绪太高只记录不改"
    return f"梦后复盘：发现 {len(findings)} 处问题，自动修了 {repaired} 处"


def print_profile(engine: MindEngine) -> None:
    """Profile the knowledge base's corpus; say so when there is none.

    Printing nothing at all when ``kb.dir`` is unset looked like a broken
    command, so fall back to the default KB directories and report a miss.
    """
    kb_dir = getattr(engine.kb, "dir", None)
    dirs = [kb_dir] if kb_dir else list(DEFAULT_DIRS)
    shown = False
    for directory in dirs:
        if directory is not None and directory.is_dir():
            print(format_profile(profile_corpus(directory)))
            shown = True
    if not shown:
        print("（没有知识库目录可分析）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omega_mind")
    parser.add_argument("--persona", default="generic", choices=["generic", "feynman"])
    parser.add_argument("--save", default="mind_state.json")
    parser.add_argument("--load", action="store_true", help="resume from --save if it exists")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--lang", choices=LANGUAGES, default=None,
                        help="reply language mode (default: saved or zh)")
    parser.add_argument("--serve", type=int, metavar="PORT", default=None,
                        help="run HTTP/WebSocket server instead of the CLI")
    parser.add_argument("--profile", action="store_true",
                        help="print the corpus profile and exit")
    args = parser.parse_args(argv)

    persona = get_persona(args.persona)

    import os
    if args.load and os.path.exists(args.save):
        engine = MindEngine.load(args.save, persona=persona, language=args.lang)
        if engine.load_warning:
            print(f"[warn] {engine.load_warning} — 从空状态开始")
        else:
            print(f"[resumed] {args.save}")
    else:
        engine = MindEngine(persona=persona, store_path=args.save,
                            language=args.lang or "zh")
    if args.lang:
        engine.set_language(args.lang)

    if args.serve is not None:
        from .server import serve
        serve(engine, port=args.serve)
        return 0
    if args.profile:
        print_profile(engine)
        return 0

    print(f"OmegaMind · persona={persona.name} · lang={engine.language} · "
          "/state /save /reset /cube /turn /memory /relations /dream "
          "/profile /challenge /lang /exit")
    print(f"[kb] {engine.kb.mode} · [pybot] {engine.bridge.mode}")
    challenge_on = False
    hinted: set[str] = set()

    def prompt() -> str:
        return "you> " if engine.language == "en" else "你> "

    def say(text: str) -> None:
        print(("mind> " if engine.language == "en" else "他> ") + text)

    def read_line() -> str | None:
        """Blocking read that still lets idle time do its work."""
        sys.stdout.write(prompt())
        sys.stdout.flush()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 5.0)
            if ready:
                line = sys.stdin.readline()
                return line.strip() if line else None
            dream = engine.maintenance()
            if dream is not None:
                print(f"\n[dream {dream.index}] {dream.narrative}")
            message = engine.tick()
            if message:
                print("\n" + ("mind> " if engine.language == "en" else "他> ") + message)
            sys.stdout.write(prompt())
            sys.stdout.flush()

    while True:
        line = read_line()
        if line is None:
            print()
            break
        text = line
        if not text:
            continue
        if text in ("/lang", "/language"):
            print(f"current language: {engine.language} (switch: /lang zh | /lang en)")
            continue
        if text.startswith("/lang"):
            parts = text.split()
            if len(parts) != 2 or parts[1] not in LANGUAGES:
                print("用法：/lang zh 或 /lang en")
                continue
            engine.set_language(parts[1])
            print(f"language mode: {engine.language}")
            continue
        if text in ("/zh", "/en"):
            engine.set_language(text[1:])
            print(f"language mode: {engine.language}")
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/state":
            print(engine.state.snapshot())
            continue
        if text == "/save":
            engine.save()
            print(f"saved -> {args.save}")
            continue
        if text == "/reset":
            engine.reset()
            print("reset")
            continue
        if text == "/cube":
            print(render_net(engine.cube))
            continue
        if text.startswith("/turn"):
            parts = text.split()
            if len(parts) != 2:
                print("用法：/turn U  ·  /turn R'  ·  /turn F2")
                continue
            try:
                moved = engine.turn_cube(parts[1])
            except (ValueError, KeyError) as exc:
                print(f"无效动作：{exc}")
                continue
            print(f"转动 {parts[1]} · 动了 {len(moved)} 块")
            print(render_net(engine.cube))
            continue
        if text == "/memory":
            snap = engine.tiers.snapshot()
            print(f"working={len(snap['working'])} episodes={len(snap['episodes'])} "
                  f"undreamed={len(engine.tiers.undreamed())}")
            for fact in snap["semantic"].values():
                print(f"  · {fact['value']}")
            continue
        if text == "/profile":
            print_profile(engine)
            continue
        if text == "/relations":
            for name, node in engine.relations.nodes.items():
                print(f"  {name}: affinity={node.affinity:.2f} trust={node.trust:.2f} "
                      f"interactions={node.interactions}")
            continue
        if text == "/dream":
            dream = engine.dream()
            if dream is None:
                print("（没有可整理的片段）")
            else:
                print(f"梦 #{dream.index} · {dream.narrative}")
                if dream.promoted:
                    print("  记住：" + "；".join(dream.promoted))
                if dream.audit and not dream.audit.get("error"):
                    print("  " + reflect_summary(dream.audit, engine.language))
            continue
        if text == "/challenge":
            session = engine.challenge(reset=True)
            if not session.available():
                print(("知识库里没有可教学单元，先检查 ~/Projects/feynman-open/knowledge"
                       if engine.language == "zh" else
                       "No teachable units in the knowledge base — check the corpus."))
                challenge_on = False
            else:
                print(session.next_unit())
                print("（直接回答；/next 换一题，/stop 退出挑战）" if engine.language == "zh"
                      else "(answer in your own words; /next for another, /stop to leave)")
                challenge_on = True
            continue
        if challenge_on:
            session = engine.challenge()
            if text in ("/stop", "/exit-challenge"):
                challenge_on = False
                print("(left challenge mode)" if engine.language == "en"
                      else "（离开挑战模式）")
                continue
            if text == "/next":
                print(session.next_unit())
                continue
            grade = session.submit(text)
            print(grade.feedback)
            if args.verbose:
                print(f"  verdict={grade.verdict} score={grade.score} "
                      f"matched={grade.matched} missing={grade.missing}")
            if session.unit is None and grade.verdict == "strong":
                print(session.next_unit())
            continue

        mismatch = (
            (engine.language == "zh" and not has_cjk(text))
            or (engine.language == "en" and has_cjk(text))
        )
        if mismatch and engine.language not in hinted and not challenge_on:
            hinted.add(engine.language)
            if engine.language == "en":
                print("(you are in English mode; switch with /lang zh)")
            else:
                print("（当前是中文模式；切换：/lang en）")
        result = engine.turn(text)
        say(result.reply)
        if args.verbose:
            print(render_ascii(result.directive, result.activations))
            print(f"  base={result.base_reply!r} source={result.source}")
    engine.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
