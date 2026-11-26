"""Keyframe report for a scripted run: verbosity peak and dream before/after.

This is a diagnostic, not part of the mind. It drives a real ``MindEngine``
through a deterministic 40-turn script and captures three keyframes:

* the turn where the renderer's ``verbosity`` peaked;
* the cube net immediately before a dream;
* the cube net immediately after that dream restored part of the displacement.

The two nets are printed side by side so a human can see what the dream moved.

Run it with::

    python3 -m omega_mind.keyframe_report
    python3 -m omega_mind.keyframe_report --persona generic --turns 60
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import get_persona
from .cube import render_net
from .cube_feedback import CubeFeedback, displacement
from .engine import MindEngine
from .feynman_kb import FeynmanKB
from .pybot_bridge import PyBotBridge

# A mix of praise, curiosity, correction and indifference. It is deliberately
# short: the point is to move the cube and the renderer, not to test language.
DEFAULT_LINES: Sequence[str] = (
    "你真厉害，这个想法太漂亮了",
    "再讲深一点，我想知道为什么",
    "这不对吧？",
    "有点意思，继续说",
    "我不懂这里",
    "算了，换个话题",
    "你刚才那个解释很清楚",
    "真的吗？我怎么觉得不对劲",
    "太棒了，再来一个",
    "随便聊聊吧",
)


@dataclass
class Keyframe:
    turn: int
    text: str
    verbosity: float
    detail: str
    displacement: int
    bias: Dict[str, float]
    net: str


@dataclass
class DreamSnapshot:
    turn: int
    before_net: str
    after_net: str
    before_displacement: int
    after_displacement: int
    cube_report: Dict[str, object] = field(default_factory=dict)
    salience: float = 0.0
    restore_ratio: Optional[float] = None
    audit: Dict[str, object] = field(default_factory=dict)


@dataclass
class KeyframeReport:
    persona: str
    frames: List[Keyframe]
    peak: Keyframe
    dream: Optional[DreamSnapshot]

    def to_text(self) -> str:
        out = [
            f"persona={self.persona}  turns={len(self.frames)}  "
            f"peak_turn={self.peak.turn}  peak_verbosity={self.peak.verbosity:.3f}",
            "",
            "turn  verbosity  detail   displaced  bias",
        ]
        for f in self.frames:
            out.append(
                f"{f.turn:>4}  {f.verbosity:>9.3f}  {f.detail:<8} "
                f"{f.displacement:>9}  d={f.bias['depth']:+d} v={f.bias['verbosity']:+.3f}"
            )
        out.append("")
        out.append(f"--- verbosity peak at turn {self.peak.turn}: {self.peak.text}")
        out.append(self.peak.net)
        if self.dream is not None:
            out.append("")
            out.append(
                f"--- dream at turn {self.dream.turn}: "
                f"displacement {self.dream.before_displacement} -> "
                f"{self.dream.after_displacement}  "
                f"salience={self.dream.salience:.2f}  "
                f"restore_ratio={self.dream.restore_ratio}"
            )
            out.append(render_side_by_side(
                "before dream", self.dream.before_net,
                "after dream", self.dream.after_net,
            ))
            audit = self.dream.audit or {}
            if audit and not audit.get("error"):
                out.append("")
                out.append("--- dream & evolution summary")
                out.append(f"findings={len(audit.get('findings') or [])}  "
                           f"repairs={audit.get('repaired', 0)}  "
                           f"read_only={bool(audit.get('read_only'))}  "
                           f"gate_ok={bool(audit.get('gate_ok', True))}")
                for action in (audit.get("actions") or [])[:6]:
                    mark = "ok" if action.get("applied") else "no"
                    line = (f"  [{mark}] {action.get('kind')} "
                            f"{action.get('target') or '-'}")
                    if action.get("reason"):
                        line += f"  ({action.get('reason')})"
                    out.append(line)
        else:
            out.append("")
            out.append("--- dream: not triggered (no undreamed episodes)")
        return "\n".join(out)


def render_side_by_side(left_title: str, left: str,
                        right_title: str, right: str,
                        gap: int = 4) -> str:
    """Two multi-line blocks with matching line counts, side by side."""
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    height = max(len(left_lines), len(right_lines))
    left_width = max([len(line) for line in left_lines] + [len(left_title)])
    right_width = max([len(line) for line in right_lines] + [len(right_title)])

    def pad(text: str, width: int) -> str:
        return text + " " * max(0, width - len(text))

    lines = [
        pad(left_title, left_width) + " " * gap + right_title,
        pad("-" * len(left_title), left_width) + " " * gap + "-" * len(right_title),
    ]
    for i in range(height):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        lines.append(pad(l, left_width) + " " * gap + pad(r, right_width).rstrip())
    return "\n".join(lines)


def pick_peak(frames: Sequence[Keyframe]) -> Keyframe:
    """Highest verbosity; on a tie the earlier turn wins (max keeps first)."""
    return max(frames, key=lambda frame: frame.verbosity)


def run_report(persona: str = "feynman", turns: int = 40,
               dream_at: int = 32,
               lines: Sequence[str] = DEFAULT_LINES,
               step_seconds: float = 20.0,
               with_kb: bool = False) -> KeyframeReport:
    if turns < 1:
        raise ValueError("turns must be at least 1")
    if not lines:
        raise ValueError("lines must not be empty")
    # The scripted lines are not knowledge questions, so the default run does
    # not load the corpus. ``with_kb`` turns it on (into a throwaway learned
    # directory) when the point of the run is the dream audit itself.
    kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-keyframe-")) \
        if with_kb else FeynmanKB.disabled()
    engine = MindEngine(
        persona=get_persona(persona),
        bridge=PyBotBridge.disabled(),
        kb=kb,
        auto_save=False,
        language="zh",
    )
    feedback = CubeFeedback()
    frames: List[Keyframe] = []
    dream: Optional[DreamSnapshot] = None
    now = 1000.0
    for i in range(turns):
        now += step_seconds
        text = lines[i % len(lines)]
        result = engine.turn(text, now=now)
        bias = feedback.bias(engine.cube)
        frames.append(Keyframe(
            turn=i + 1,
            text=text,
            verbosity=round(result.directive.verbosity, 4),
            detail=str(result.left_brain.get("detail", "")),
            displacement=displacement(engine.cube),
            bias=bias.to_dict(),
            net=render_net(engine.cube),
        ))
        if dream is None and (i + 1) == dream_at:
            before = render_net(engine.cube)
            before_displacement = displacement(engine.cube)
            now += step_seconds
            dream_record = engine.dream(now=now, force=True)
            cube_report: Dict[str, object] = {}
            if engine.event_log and "cube_restore" in engine.event_log[-1]:
                cube_report = dict(engine.event_log[-1]["cube_restore"])
            dream = DreamSnapshot(
                turn=i + 1,
                before_net=before,
                after_net=render_net(engine.cube),
                before_displacement=before_displacement,
                after_displacement=displacement(engine.cube),
                cube_report=cube_report,
                salience=getattr(dream_record, "salience", 0.0),
                restore_ratio=cube_report.get("restore_ratio"),
                audit=dict(getattr(dream_record, "audit", {}) or {}),
            )
    return KeyframeReport(
        persona=persona,
        frames=frames,
        peak=pick_peak(frames),
        dream=dream,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="feynman",
                        help="generic or feynman (default: feynman)")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--dream-at", type=int, default=32,
                        help="turn number that triggers a forced dream")
    parser.add_argument("--with-kb", action="store_true",
                        help="load the corpus (into a throwaway learned dir) "
                             "so the dream audit has something to inspect")
    args = parser.parse_args(argv)
    try:
        report = run_report(persona=args.persona, turns=args.turns,
                            dream_at=args.dream_at, with_kb=args.with_kb)
    except ValueError as exc:
        parser.error(str(exc))
    print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
