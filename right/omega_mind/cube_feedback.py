"""The return path of the loop: where the cubies sit biases the mind.

cubies.py calls the cube projection "a view over the activations". This module
adds the small feedback edge, so the arrangement is no longer write-only:

- a cubie that has been *moved* onto the front face appraises a little
  stronger (bounded boost, then fatigue if it overstays the stage);
- a scrambled cube lifts the baseline that stress and defensiveness return to
  and trims dopamine's baseline, capped at ``MAX_BASELINE_SHIFT``;
- the front corners colour how deep the left brain looks and how much of a
  card it speaks;
- a dream walks the cube partway home along the inverse of its own history.

One rule keeps the whole thing honest: only *displacement* counts. The
arrangement a cubie was born in is not a bias — being moved is. A fresh mind
starts with every gain at 1.0 and every baseline at the persona value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import BOUNDS, DIMENSIONS, PersonaConfig
from .cube import HOME_OF, CubeState, Move, Pos, _rotate, parse_move

# A quarter-turn boost of at most +20% for a cubie standing on the front face
# after having been moved there, decaying by 5% per extra turn on stage until
# the floor. Nothing here can push a gain outside [MIN_GAIN, 1 + MAX_GAIN].
MAX_GAIN = 0.20
GAIN_PER_DWELL = 0.08
HABITUATION_TURNS = 3
FATIGUE_STEP = 0.05
MIN_GAIN = 0.90

# Chronic mood: at most this much offset from the persona baseline, reached
# only if every movable cubie is out of place. Defensiveness deliberately
# stays at its persona baseline: both personas define it as a purely reactive
# posture (baseline 0), and the acceptance scenario requires it back under
# 0.10 within three turns of a criticism.
MAX_BASELINE_SHIFT = 0.12
_SHIFTED_UP = ("cortisol",)
_SHIFTED_DOWN = (("dopamine", 0.5),)

# A boosted cubie needs more than the plain threshold to twist the cube again
# (hysteresis), a raw strength below the floor cannot be inflated over the
# line by gain alone, and a cubie that just turned sits out a cooldown.
TURN_THRESHOLD = 0.75
HYSTERESIS = 0.08
RAW_FLOOR = 0.62
COOLDOWN_TURNS = 2

# Dream: close roughly 45% of the displacement gap, using at most this many
# inverse moves so a dream stays a short settling pass, not a solve.
RESTORE_RATIO = 0.45
MAX_RESTORE_MOVES = 12
# A dream may land this many facelets past its target before it is treated as
# a wipe rather than a settling: one inverse move can drop displacement by 6,
# which would erase far more stance than a 10% dream asked for. With one
# facelet of slack, a bereavement-level batch restores nothing rather than 37%.
MAX_OVERSHOOT_FACELETS = 1

# A dream that carries heavy material should not wipe the stance those turns
# produced. The ratio falls linearly with the batch's peak salience; at the
# top a dream keeps this fraction of the base ratio. The scale is calibrated
# to measured values from the real appraisal path, not the 0..10 cap of
# ``salience_of``: ordinary chatter lands near 0.02-0.15, an insult or a hard
# correction near 0.9, and the heaviest lines seen (a death, being told the
# answer was wrong while grieving) near 2.4. A scale of 10 would have left the
# curve flat.
DREAM_SALIENCE_SCALE = 2.5
DREAM_RESTORE_MIN_FRACTION = 0.20


def dream_restore_ratio(salience: float,
                        base: float = RESTORE_RATIO) -> float:
    """How much displacement a dream may close, given how heavy it was.

    Low salience (ordinary chatter) gets the full settling pass. High
    salience (a fight, a loss, a breakthrough) keeps most of its stance, so
    the next morning still carries the weight of what happened. Bounded by
    ``base`` at the bottom and ``base * DREAM_RESTORE_MIN_FRACTION`` at the
    top; a dream never increases displacement at any ratio.

    The curve is calibrated on ``DREAM_SALIENCE_SCALE`` (see above), not on
    the 0..10 cap of ``salience_of``; passing a cap-scale number just
    saturates it at the floor.
    """
    weight = min(1.0, max(0.0, salience) / DREAM_SALIENCE_SCALE)
    ratio = base * (1.0 - (1.0 - DREAM_RESTORE_MIN_FRACTION) * weight)
    return round(max(0.0, min(1.0, ratio)), 4)

FRONT_CORNERS: tuple[Pos, ...] = ((-1, 1, 1), (1, 1, 1), (-1, -1, 1), (1, -1, 1))

# Front-corner -> (retrieval-depth steps, verbosity delta). Adapted from the
# corner's job: perfectionism/pedantry look deeper and say more; the defensive
# corners close the aperture.
CORNER_WEIGHTS: Dict[str, tuple[int, float]] = {
    "corner_perfectionism": (2, 0.35),
    "corner_pedantry": (1, 0.30),
    "corner_paranoia": (1, 0.15),
    "corner_empathy": (0, 0.10),
    "corner_impulsivity": (-1, -0.25),
    "corner_shame": (-1, -0.20),
    "corner_grudge": (-1, -0.30),
    "corner_denial": (-2, -0.35),
}
MAX_DEPTH_BIAS = 2
MAX_VERBOSITY_BIAS = 0.35

# The raw sum is linear in the number of displaced corners, which made one
# moved corner nearly invisible to the left brain's +/-0.20 gate. A tanh with
# this gain keeps a single displacement visible while the output stays bounded
# by MAX_*; bias is recomputed from the current arrangement every turn, so it
# can never accumulate into a latch. 1.35 (not 1.65) is the largest value that
# keeps one quarter turn under 0.20 verbosity: a single move may tilt the
# reply, but only a second displaced corner can close the aperture.
SENSITIVITY_GAIN = 1.35


def front_names(cube: CubeState) -> Set[str]:
    """Cubies on the front face *as currently framed*. A reframe moves this
    set without moving a single cubie."""
    return {name for pos, name in cube.occupancy.items()
            if cube.view(pos)[2] == 1}


def displacement(cube: CubeState) -> int:
    return sum(1 for pos, name in cube.occupancy.items()
               if HOME_OF.get(name) != pos)


@dataclass(frozen=True)
class CubeSignal:
    """The arrangement read as a continuous signal, not just a stance.

    ``displacement`` is the share of cubies away from home: the slow channel
    the baselines already relax toward. ``delta`` is how many displaced
    cubies this mind gained or lost since the last observed turn — the fast
    channel, and the only one the left brain consumes. ``front_load`` is the
    share of the front face that is displaced rather than default.
    """

    displacement: float = 0.0
    delta: int = 0
    front_load: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "displacement": round(self.displacement, 4),
            "delta": self.delta,
            "front_load": round(self.front_load, 4),
        }


@dataclass(frozen=True)
class CubeBias:
    depth: int = 0          # steps added to the retrieval candidate limit
    verbosity: float = 0.0  # added to the reply-detail pressure

    def to_dict(self) -> Dict[str, Any]:
        return {"depth": self.depth, "verbosity": round(self.verbosity, 3)}


def tuned_bias(raw_depth: float, raw_verbosity: float,
               gain: float = SENSITIVITY_GAIN) -> CubeBias:
    """Squash the averaged front-face sum onto the bounded bias range.

    Properties the tests pin down: zero raw input gives exactly zero (a solved
    cube must not drift), each output is monotonic in its raw input, and the
    magnitude never exceeds ``MAX_DEPTH_BIAS`` / ``MAX_VERBOSITY_BIAS`` no
    matter how scrambled the front face gets.
    """
    if raw_depth == 0 and raw_verbosity == 0:
        return CubeBias()
    gain = max(0.0, gain)
    depth = MAX_DEPTH_BIAS * math.tanh(raw_depth * gain / MAX_DEPTH_BIAS)
    verbosity = MAX_VERBOSITY_BIAS * math.tanh(
        raw_verbosity * gain / MAX_VERBOSITY_BIAS)
    depth = int(max(-MAX_DEPTH_BIAS, min(MAX_DEPTH_BIAS, round(depth))))
    verbosity = max(-MAX_VERBOSITY_BIAS,
                    min(MAX_VERBOSITY_BIAS, verbosity))
    return CubeBias(depth=depth, verbosity=round(verbosity, 3))


class CubeFeedback:
    """Per-mind bookkeeping for the return path. Snapshot/restore keeps it."""

    def __init__(self) -> None:
        self.dwell: Dict[str, int] = {}
        self.last_turn: Dict[str, int] = {}
        self.observed_turn: Optional[int] = None
        # Momentum baseline: displacement measured at the previous observed
        # turn. None until the first observation, so a fresh mind does not
        # report a spurious jump from an unknown past.
        self.last_displaced: Optional[int] = None
        self.signal = CubeSignal()

    # -- per-turn observation ---------------------------------------------

    def observe(self, cube: CubeState, turn: int) -> None:
        """Count how long each displaced cubie has held the front face.

        Called once at the start of a turn, before appraisal, so gains apply
        to the very turn after a cubie is moved there.
        """
        if turn == self.observed_turn:
            return
        self.observed_turn = turn
        live = {name for name in front_names(cube)
                if HOME_OF.get(name) != cube.position_of(name)}
        for name in list(self.dwell):
            if name not in live:
                del self.dwell[name]
        for name in live:
            self.dwell[name] = self.dwell.get(name, 0) + 1
        # Measure the arrangement once per turn: displacement is the slow
        # stance, delta is the fast momentum, front_load is how much of the
        # focus face is a moved stance. First observation has no baseline,
        # so its delta is zero rather than "whatever the past was".
        displaced = displacement(cube)
        delta = 0 if self.last_displaced is None else displaced - self.last_displaced
        self.last_displaced = displaced
        front = sum(1 for pos, name in cube.occupancy.items()
                    if cube.view(pos)[2] == 1 and HOME_OF.get(name) != pos)
        self.signal = CubeSignal(displacement=displaced / 26.0,
                                 delta=delta,
                                 front_load=front / 9.0)

    def gains(self, cube: CubeState) -> Dict[str, float]:
        """Appraisal multiplier for cubies currently moved onto the front."""
        out: Dict[str, float] = {}
        for name in front_names(cube):
            if HOME_OF.get(name) == cube.position_of(name):
                continue
            dwell = self.dwell.get(name, 0)
            if dwell <= 0:
                continue
            if dwell <= HABITUATION_TURNS:
                boost = min(MAX_GAIN, GAIN_PER_DWELL * dwell)
            else:
                boost = MAX_GAIN - FATIGUE_STEP * (dwell - HABITUATION_TURNS)
            out[name] = round(max(MIN_GAIN, 1.0 + boost), 4)
        return out

    def baselines(self, persona: PersonaConfig, cube: CubeState) -> Dict[str, float]:
        """Effective decay targets: static persona baseline plus a capped
        chronic offset proportional to how scrambled the cube is."""
        shift = displacement(cube) / 26.0 * MAX_BASELINE_SHIFT
        out = {dim: persona.baseline(dim) for dim in DIMENSIONS}
        for dim in _SHIFTED_UP:
            low, high = BOUNDS[dim]
            out[dim] = round(max(low, min(high, out[dim] + shift)), 6)
        for dim, weight in _SHIFTED_DOWN:
            low, high = BOUNDS[dim]
            out[dim] = round(max(low, min(high, out[dim] - shift * weight)), 6)
        return out

    def bias(self, cube: CubeState) -> CubeBias:
        """How the front corners tilt retrieval depth and reply detail.

        Only *displaced* front corners count, and the summed weight is averaged
        over the four front slots: the corner bias is a property of the whole
        front face, not of whichever two corners a single turn happened to
        move. A lone ``R`` therefore tilts the average mildly — a tilt, not a
        latch — while a face carrying several displaced defensive corners can
        still close the aperture. The averaged sum then goes through
        ``tuned_bias``, which makes the first displacement more visible without
        changing the bound.
        """
        depth = 0
        verbosity = 0.0
        back = -cube.orientation % 4
        for slot in FRONT_CORNERS:
            body = _rotate(slot, "y", back)
            name = cube.cubie_at(body)
            if not name or HOME_OF.get(name) == body:
                continue
            step, delta = CORNER_WEIGHTS.get(name, (0, 0.0))
            depth += step
            verbosity += delta
        return tuned_bias(depth / len(FRONT_CORNERS),
                          verbosity / len(FRONT_CORNERS))

    # -- turn gate ---------------------------------------------------------

    def choose_turn(self, activations, cube: CubeState, turn: int,
                    gains: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Which cubie, if any, is strong enough to twist the cube this turn.

        Hysteresis keeps a cubie that is already benefiting from the front
        face from flipping itself away over a marginal event; the raw floor
        stops gain from turning noise into a breakthrough.
        """
        gains = self.gains(cube) if gains is None else gains
        best: Optional[str] = None
        best_margin = 0.0
        for act in activations:
            gain = gains.get(act.name, 1.0)
            raw = act.strength / gain if gain else act.strength
            if raw < RAW_FLOOR:
                continue
            if turn - self.last_turn.get(act.name, -10 ** 9) <= COOLDOWN_TURNS:
                continue
            threshold = TURN_THRESHOLD + (HYSTERESIS if gain > 1.0 else 0.0)
            margin = act.strength - threshold
            if margin >= 0.0 and margin > best_margin:
                best_margin = margin
                best = act.name
        return best

    def note_turn(self, name: str, turn: int) -> None:
        self.last_turn[name] = turn
        if len(self.last_turn) > 26:
            for old in sorted(self.last_turn, key=self.last_turn.get)[:6]:
                del self.last_turn[old]

    # -- dream -------------------------------------------------------------

    def partial_restore(self, cube: CubeState, ratio: float = RESTORE_RATIO,
                        max_moves: int = MAX_RESTORE_MOVES) -> Dict[str, Any]:
        """Walk the cube partway home by inverting its own recent history.

        Only legal moves are used, so the arrangement stays one this cube
        could have reached. Deterministic: the same history gives the same
        dream. The walk is dry-run first: reversing the most recent moves can
        pass through a *worse* arrangement on the way, so only a prefix that
        actually lowers displacement is applied — and if none does, a dream
        leaves the cube alone rather than scrambling it.

        A single inverse move can jump most of the way home (16 -> 10 in one
        turn), so the ratio is a target, not a guarantee. The dream takes the
        reachable prefix closest to the target; if that prefix lands more than
        MAX_OVERSHOOT_FACELETS below it, the gentlest prefix still above the
        target is used instead. Without that guard a 10% dream after a
        bereavement restored 37% of the stance.
        """
        initial = displacement(cube)
        report: Dict[str, Any] = {
            "initial_displaced": initial,
            "final_displaced": initial,
            "target_remaining": initial,
            "moves": [],
            "actual_reduction": 0.0,
        }
        if initial == 0 or ratio <= 0:
            # ratio 0 means "settle nothing": the target equals the current
            # displacement, and any inverse move would restore more than asked.
            return report
        target = max(0, round(initial * (1.0 - ratio)))
        report["target_remaining"] = target

        sim = CubeState()
        sim.restore(cube.snapshot())
        prefix: List[str] = []
        candidates: List[Tuple[int, List[str]]] = []
        for entry in reversed(list(cube.history)):
            if len(prefix) >= max_moves:
                break
            move = parse_move(entry)
            inverse = Move(move.face, (4 - move.quarters) % 4)
            sim.apply(str(inverse))
            prefix.append(str(inverse))
            value = displacement(sim)
            if value < initial:
                candidates.append((value, list(prefix)))

        # Closest reachable displacement to the target; on a tie prefer the
        # gentler side (above the target, i.e. more stance left standing).
        moves: List[str] = []
        best = min(candidates,
                   key=lambda c: (abs(c[0] - target), c[0] < target, c[0]),
                   default=None)
        if best is not None:
            value, chosen = best
            if target - value > MAX_OVERSHOOT_FACELETS:
                gentler = [(v, p) for v, p in candidates if v >= target]
                value, chosen = min(gentler, key=lambda c: c[0], default=(initial, []))
            moves = chosen

        for text in moves:
            cube.apply(text)
            report["moves"].append(text)
        final = displacement(cube)
        report["final_displaced"] = final
        report["actual_reduction"] = round((initial - final) / initial, 3)
        return report

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "dwell": dict(self.dwell),
            "last_turn": dict(self.last_turn),
            "observed_turn": self.observed_turn,
            "last_displaced": self.last_displaced,
        }

    def restore(self, data: Dict[str, Any]) -> None:
        self.dwell = {str(k): int(v) for k, v in (data.get("dwell") or {}).items()}
        self.last_turn = {str(k): int(v) for k, v in (data.get("last_turn") or {}).items()}
        observed = data.get("observed_turn")
        self.observed_turn = int(observed) if observed is not None else None
        displaced = data.get("last_displaced")
        self.last_displaced = int(displaced) if displaced is not None else None
        # The signal itself is cheap to recompute; keep it neutral until the
        # next observation so a restored mind never speaks a stale delta.
        self.signal = CubeSignal()

    def reset(self) -> None:
        self.dwell.clear()
        self.last_turn.clear()
        self.observed_turn = None
        self.last_displaced = None
        self.signal = CubeSignal()
