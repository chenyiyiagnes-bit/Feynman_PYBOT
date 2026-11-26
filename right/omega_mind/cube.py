"""The physical cube: 26 cubies that actually move.

The appraisal layer decides *which* cubie fires; this module is the cube those
cubies sit on. A strong activation faces the mind one quarter turn, so after a
conversation the cube records where the mind has been — the arrangement is the
history, not a decoration.

Positions are integer coordinates in {-1, 0, 1}³ excluding the centre.
Faces: U y=+1, D y=-1, R x=+1, L x=-1, F z=+1, B z=-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cubies import CUBIES

Pos = Tuple[int, int, int]

FACE_LAYER = {"U": ("y", 1), "D": ("y", -1), "R": ("x", 1), "L": ("x", -1),
              "F": ("z", 1), "B": ("z", -1)}
FACE_ORDER = ("U", "D", "L", "R", "F", "B")

_CENTRES = {  # core cubie -> its home position (matches cubies.CUBIES face)
    "core_superego": (0, 1, 0),
    "core_memory": (0, -1, 0),
    "core_logic": (-1, 0, 0),
    "core_intuition": (1, 0, 0),
    "core_ego": (0, 0, 1),
    "core_id": (0, 0, -1),
}

ALL_POSITIONS: List[Pos] = [
    (x, y, z)
    for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
    if (x, y, z) != (0, 0, 0)
]


def _initial_arrangement() -> Dict[Pos, str]:
    """Edges and corners fill the remaining slots in a fixed order."""
    layout: Dict[Pos, str] = dict((pos, name) for name, pos in _CENTRES.items())
    free = [p for p in ALL_POSITIONS if p not in layout]
    edges = [c.name for c in CUBIES if c.role == "edge"]
    corners = [c.name for c in CUBIES if c.role == "corner"]
    corner_slots = [p for p in free if all(v != 0 for v in p)]
    edge_slots = [p for p in free if sum(1 for v in p if v != 0) == 2]
    for pos, name in zip(corner_slots, corners):
        layout[pos] = name
    for pos, name in zip(edge_slots, edges):
        layout[pos] = name
    return layout


# Where every cubie was born. Displacement (position != home) is what the
# feedback layer treats as "this conversation has moved me".
HOME_OF: Dict[str, Pos] = {
    name: pos for pos, name in _initial_arrangement().items()
}


def _rotate(pos: Pos, axis: str, quarters: int) -> Pos:
    x, y, z = pos
    for _ in range(quarters % 4):
        if axis == "x":
            x, y, z = x, -z, y
        elif axis == "y":
            x, y, z = z, y, -x
        else:
            x, y, z = -y, x, z
    return (x, y, z)


@dataclass
class Move:
    face: str
    quarters: int = 1

    def __post_init__(self):
        if self.face not in FACE_LAYER:
            raise ValueError(f"unknown face: {self.face}")
        self.quarters %= 4

    def __str__(self) -> str:
        # Must round-trip through parse_move: history is replayed backwards
        # by the dream restore, so "R'" has to stay "R'" and not become "R3".
        suffix = "" if self.quarters == 1 else ("'" if self.quarters == 3
                                                else str(self.quarters))
        return f"{self.face}{suffix}"


def parse_move(text: str) -> Move:
    text = text.strip()
    if not text:
        raise ValueError("empty move")
    face = text[0].upper()
    rest = text[1:]
    if rest in ("", "+"):
        return Move(face, 1)
    if rest in ("'", "’", "-"):
        return Move(face, 3)
    if rest in ("2", "''"):
        return Move(face, 2)
    raise ValueError(f"bad move: {text}")


@dataclass
class CubeState:
    occupancy: Dict[Pos, str] = field(default_factory=_initial_arrangement)
    history: List[str] = field(default_factory=list)
    # Which side the mind is looking at: whole-cube quarter-turns about y.
    # A reframe changes nothing physical. It only changes which face counts
    # as "front" for attention, gains and front-corner bias — the mechanical
    # form of reappraisal: same facts, different salience.
    orientation: int = 0

    # -- moves -------------------------------------------------------------

    def reframe(self, quarters: int = 1) -> int:
        """Turn the whole cube in the mind's hands. Occupancy is untouched."""
        self.orientation = (self.orientation + quarters) % 4
        return self.orientation

    def view(self, pos: Pos) -> Pos:
        """Where a body position appears in the current viewing frame."""
        return _rotate(pos, "y", self.orientation % 4)

    def front_positions(self) -> List[Pos]:
        """Body positions currently on the front face of the view."""
        return [pos for pos in self.occupancy if self.view(pos)[2] == 1]

    def turn(self, face: str, quarters: int = 1) -> List[str]:
        """Rotate one face; returns the names of the cubies it moved."""
        axis, layer = FACE_LAYER[face]
        quarters %= 4
        if quarters == 0:
            return []
        index = {"x": 0, "y": 1, "z": 2}[axis]
        moved = [pos for pos in self.occupancy if pos[index] == layer]
        new_occupancy = dict(self.occupancy)
        for pos in moved:
            new_occupancy.pop(pos)
        names = []
        for pos in moved:
            target = _rotate(pos, axis, quarters)
            name = self.occupancy[pos]
            new_occupancy[target] = name
            names.append(name)
        self.occupancy = new_occupancy
        self.history.append(str(Move(face, quarters)))
        self.history = self.history[-50:]
        return names

    def apply(self, move_text: str) -> List[str]:
        """One move ("R'") or a sequence ("R U R' U'")."""
        names: List[str] = []
        for token in move_text.split():
            move = parse_move(token)
            names.extend(self.turn(move.face, move.quarters))
        return names

    # -- queries -----------------------------------------------------------

    def position_of(self, name: str) -> Optional[Pos]:
        for pos, cubie in self.occupancy.items():
            if cubie == name:
                return pos
        return None

    def cubie_at(self, pos: Pos) -> Optional[str]:
        return self.occupancy.get(pos)

    def validate(self) -> bool:
        return len(self.occupancy) == 26 and set(self.occupancy.values()) == {
            c.name for c in CUBIES
        }

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> Dict:
        return {
            "occupancy": [[pos[0], pos[1], pos[2], name]
                          for pos, name in sorted(self.occupancy.items())],
            "history": self.history,
            "orientation": self.orientation,
        }

    def restore(self, data: Dict) -> None:
        if not data:
            return
        self.occupancy = {
            (int(x), int(y), int(z)): str(name)
            for x, y, z, name in data.get("occupancy", [])
        }
        self.history = list(data.get("history", []))
        self.orientation = int(data.get("orientation", 0)) % 4


# -- rendering -------------------------------------------------------------

_SHORT = {
    "core_ego": "EGO", "core_id": " ID", "core_logic": "LOG",
    "core_intuition": "INT", "core_superego": "SUP", "core_memory": "MEM",
}


def short_name(name: str) -> str:
    if name in _SHORT:
        return _SHORT[name]
    role, _, rest = name.partition("_")
    token = rest[:2].upper() if role == "corner" else rest[:2].lower()
    return f"{token}{role[0].upper()}"


def _face_grid(cube: CubeState, face: str) -> List[List[str]]:
    """3x3 grid of short names for one face, viewed from outside."""
    rows: List[List[str]] = []
    for row in range(3):
        line = []
        for col in range(3):
            a = col - 1
            b = 1 - row
            if face == "F":
                pos = (a, b, 1)
            elif face == "B":
                pos = (-a, b, -1)
            elif face == "U":
                pos = (a, 1, -b)
            elif face == "D":
                pos = (a, -1, b)
            elif face == "R":
                pos = (1, -b, -a)
            else:  # L
                pos = (-1, -b, a)
            name = cube.cubie_at(pos) or "??"
            line.append(short_name(name))
        rows.append(line)
    return rows


def render_net(cube: CubeState) -> str:
    faces = {f: _face_grid(cube, f) for f in FACE_ORDER}
    lines: List[str] = []
    lines.append("        " + "  ".join(faces["U"][0]))
    lines.append("        " + "  ".join(faces["U"][1]))
    lines.append("        " + "  ".join(faces["U"][2]))
    for r in range(3):
        parts = [faces["L"][r][c] for c in range(3)] + \
                [faces["F"][r][c] for c in range(3)] + \
                [faces["R"][r][c] for c in range(3)] + \
                [faces["B"][r][c] for c in range(3)]
        lines.append("  ".join(parts))
    lines.append("        " + "  ".join(faces["D"][0]))
    lines.append("        " + "  ".join(faces["D"][1]))
    lines.append("        " + "  ".join(faces["D"][2]))
    if cube.history:
        lines.append("moves: " + " ".join(cube.history[-8:]))
    if cube.orientation % 4:
        lines.append(f"frame: y×{cube.orientation % 4}（重框不改摆位）")
    return "\n".join(lines)
