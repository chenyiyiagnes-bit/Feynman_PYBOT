"""Read-only cube projection.

The cube is a view. Nothing reads state back from here.
"""

from __future__ import annotations

from typing import Dict, List

from .appraisal import Activation
from .cubies import CUBIES_BY_NAME
from .policy import StyleDirective

FACES = ("F", "B", "L", "R", "U", "D")


def face_view(activations: List[Activation]) -> Dict[str, List[str]]:
    by_face: Dict[str, List[str]] = {f: [] for f in FACES}
    for act in activations:
        cubie = CUBIES_BY_NAME[act.name]
        by_face[cubie.face].append(f"{act.name}({act.strength:.1f})")
    return by_face


def render_ascii(directive: StyleDirective, activations: List[Activation]) -> str:
    faces = face_view(activations)
    lines = [f"tone={directive.tone} warmth={directive.warmth:.2f} "
             f"tempo={directive.tempo:.2f} expr={directive.expression}"]
    for face in FACES:
        cells = ", ".join(faces[face]) or "-"
        lines.append(f"  {face}: {cells}")
    return "\n".join(lines)
