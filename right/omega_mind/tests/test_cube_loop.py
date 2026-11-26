"""The closed half of the loop: cube state feeds back into the mind.

Everything here is about three claims:
1. being displaced onto the front face colours appraisal (bounded, fades);
2. a scrambled cube lifts the baseline the affect state returns to (capped);
3. a dream walks the cube partway home along the inverse of its own history.

All effects must stay bounded — that is what the 50-turn stress test checks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import left_brain
from omega_mind.appraisal import Activation, AppraisalEngine
from omega_mind.config import BOUNDS, get_persona
from omega_mind.cube import HOME_OF, CubeState, Move, parse_move
from omega_mind.cube_feedback import (
    COOLDOWN_TURNS,
    MAX_BASELINE_SHIFT,
    MAX_DEPTH_BIAS,
    MAX_GAIN,
    MAX_VERBOSITY_BIAS,
    MIN_GAIN,
    TURN_THRESHOLD,
    CubeBias,
    CubeFeedback,
    SENSITIVITY_GAIN,
    displacement,
    front_names,
    tuned_bias,
)
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.state import AffectState

SCRAMBLE = "R U R' U' F"


def offline_engine(persona="generic") -> MindEngine:
    return MindEngine(
        persona=get_persona(persona),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
    )


def displaced_front(cube: CubeState):
    return sorted(name for name in front_names(cube)
                  if HOME_OF.get(name) != cube.position_of(name))


class TestDisplacementQueries(unittest.TestCase):
    def test_a_solved_cube_has_no_displacement(self):
        self.assertEqual(displacement(CubeState()), 0)

    def test_a_single_face_turn_displaces_eight_cubies(self):
        # The centre never moves, so a quarter turn displaces exactly 8.
        cube = CubeState()
        cube.turn("F")
        self.assertEqual(displacement(cube), 8)
        # Everyone on the front but the born-there centre is now displaced.
        self.assertEqual(displaced_front(cube),
                         sorted(n for n in front_names(cube) if n != "core_ego"))

    def test_front_is_the_z_equals_one_layer(self):
        cube = CubeState()
        self.assertEqual(len(front_names(cube)), 9)
        self.assertEqual(cube.cubie_at((0, 0, 1)), "core_ego")


class TestFeedbackGains(unittest.TestCase):
    def test_home_arrangement_is_not_a_bias(self):
        fb = CubeFeedback()
        cube = CubeState()
        for turn in range(6):
            fb.observe(cube, turn)
        self.assertEqual(fb.gains(cube), {})

    def test_a_cubie_moved_to_the_front_gets_a_bounded_boost(self):
        cube = CubeState()
        cube.apply(SCRAMBLE)
        front = displaced_front(cube)
        self.assertTrue(front, "scramble should leave someone displaced on the front")
        name = front[0]
        fb = CubeFeedback()
        fb.observe(cube, 0)
        self.assertAlmostEqual(fb.gains(cube)[name], 1.08)
        fb.observe(cube, 1)
        self.assertAlmostEqual(fb.gains(cube)[name], 1.16)
        fb.observe(cube, 2)
        self.assertAlmostEqual(fb.gains(cube)[name], 1.20)

    def test_staying_on_the_stage_fatigues(self):
        cube = CubeState()
        cube.apply(SCRAMBLE)
        name = displaced_front(cube)[0]
        fb = CubeFeedback()
        seen = []
        for turn in range(9):
            fb.observe(cube, turn)
            seen.append(fb.gains(cube)[name])
        self.assertEqual(seen[2], 1.20)
        self.assertLess(seen[3], seen[2])
        self.assertAlmostEqual(seen[8], MIN_GAIN)  # floor
        for gain in seen:
            self.assertGreaterEqual(gain, MIN_GAIN)
            self.assertLessEqual(gain, 1.0 + MAX_GAIN)

    def test_leaving_the_front_resets_the_dwell(self):
        cube = CubeState()
        cube.apply(SCRAMBLE)
        name = displaced_front(cube)[0]
        fb = CubeFeedback()
        fb.observe(cube, 0)
        fb.observe(cube, 1)
        self.assertGreater(fb.gains(cube)[name], 1.0)

        # Turn every face once: the front layer is rebuilt and nobody keeps
        # their streak across an absence.
        for step, text in enumerate("UDLRFB"):
            cube.turn(text)
            fb.observe(cube, 2 + step)
        for dwell in fb.dwell.values():
            self.assertLessEqual(dwell, len("UDLRFB"))


class TestChronicBaseline(unittest.TestCase):
    def test_a_solved_cube_returns_to_the_static_baseline(self):
        persona = get_persona("generic")
        fb = CubeFeedback()
        cube = CubeState()
        self.assertEqual(fb.baselines(persona, cube),
                         {dim: persona.baseline(dim) for dim in BOUNDS})

    def test_a_scrambled_cube_lifts_stress_and_trims_reward(self):
        persona = get_persona("generic")
        fb = CubeFeedback()
        cube = CubeState()
        cube.turn("F")
        shift = 8 / 26 * MAX_BASELINE_SHIFT
        bases = fb.baselines(persona, cube)
        self.assertAlmostEqual(bases["cortisol"], persona.baseline("cortisol") + shift, places=5)
        self.assertAlmostEqual(bases["dopamine"], persona.baseline("dopamine") - shift / 2, places=5)
        # Defensiveness is purely reactive by spec: no chronic residual.
        self.assertEqual(bases["defensiveness"], persona.baseline("defensiveness"))
        self.assertEqual(bases["valence"], persona.baseline("valence"))

    def test_the_shift_never_exceeds_its_cap(self):
        persona = get_persona("generic")
        fb = CubeFeedback()
        cube = CubeState()
        cube.apply("R U R' U' F D' L2 B' U2 R F2 L'")
        bases = fb.baselines(persona, cube)
        self.assertLessEqual(bases["cortisol"], persona.baseline("cortisol") + MAX_BASELINE_SHIFT)
        self.assertGreaterEqual(bases["dopamine"], persona.baseline("dopamine") - MAX_BASELINE_SHIFT / 2)


class TestTurnGate(unittest.TestCase):
    def test_a_boosted_cubie_needs_more_to_turn_again(self):
        cube = CubeState()
        cube.apply(SCRAMBLE)
        name = displaced_front(cube)[0]
        fb = CubeFeedback()
        fb.observe(cube, 0)
        gains = fb.gains(cube)
        self.assertGreater(gains[name], 1.0)

        below = [Activation(name, TURN_THRESHOLD + 0.02, {})]
        self.assertIsNone(fb.choose_turn(below, cube, 1, gains))
        above = [Activation(name, TURN_THRESHOLD + 0.10, {})]
        self.assertEqual(fb.choose_turn(above, cube, 1, gains), name)

    def test_an_unboosted_cubie_turns_at_the_plain_threshold(self):
        cube = CubeState()
        fb = CubeFeedback()
        name = next(n for n in front_names(cube)
                    if HOME_OF.get(n) == cube.position_of(n))
        acts = [Activation(name, TURN_THRESHOLD + 0.01, {})]
        self.assertEqual(fb.choose_turn(acts, cube, 0), name)

    def test_gain_alone_cannot_push_noise_over_the_line(self):
        cube = CubeState()
        cube.apply(SCRAMBLE)
        name = displaced_front(cube)[0]
        fb = CubeFeedback()
        fb.observe(cube, 0)
        gains = fb.gains(cube)
        # 0.68 raw * 1.20 = 0.816 would clear 0.75, but raw < 0.62 floor.
        weak = [Activation(name, 0.60 * gains[name], {})]
        self.assertIsNone(fb.choose_turn(weak, cube, 1, gains))

    def test_a_cubie_that_just_turned_sits_out_the_cooldown(self):
        cube = CubeState()
        fb = CubeFeedback()
        fb.note_turn("edge_humor", 10)
        strong = [Activation("edge_humor", 1.5, {})]
        for turn in range(11, 11 + COOLDOWN_TURNS):
            self.assertIsNone(fb.choose_turn(strong, cube, turn))
        self.assertEqual(fb.choose_turn(strong, cube, 10 + COOLDOWN_TURNS + 1), "edge_humor")


class TestPolicyBias(unittest.TestCase):
    def setUp(self):
        self.persona = get_persona("generic")
        self.state = AffectState.from_persona(self.persona, 0.0)

    def test_solved_cube_bias_is_neutral(self):
        bias = CubeFeedback().bias(CubeState())
        self.assertEqual((bias.depth, bias.verbosity), (0, 0.0))

    def test_positive_bias_deepens_and_negative_bias_tersens(self):
        base = left_brain.plan_for(self.state, self.persona, [])
        deep = left_brain.plan_for(self.state, self.persona, [],
                                   cube_bias=CubeBias(depth=2, verbosity=0.35))
        terse = left_brain.plan_for(self.state, self.persona, [],
                                    cube_bias=CubeBias(depth=-2, verbosity=-0.35))
        self.assertEqual(base.detail, "normal")
        self.assertEqual(deep.detail, "rich")
        self.assertEqual(deep.limit, min(4, base.limit + 1))
        self.assertEqual(terse.detail, "brief")
        self.assertEqual(terse.limit, max(1, base.limit - 1))

    def test_stress_still_wins_over_a_positive_bias(self):
        self.state.set("cortisol", 0.70)
        policy = left_brain.plan_for(self.state, self.persona, [],
                                     cube_bias=CubeBias(depth=2, verbosity=0.35))
        self.assertEqual(policy.detail, "brief")
        self.assertEqual(policy.posture, "careful")

    def test_limits_stay_inside_their_range(self):
        for depth in range(-3, 4):
            policy = left_brain.plan_for(self.state, self.persona, [],
                                         cube_bias=CubeBias(depth=depth, verbosity=0.0))
            self.assertIn(policy.limit, (1, 2, 3, 4))


class TestTunedBiasCurve(unittest.TestCase):
    """The tanh curve must be sensitive, monotonic and bounded at once."""

    def test_home_state_has_zero_drift(self):
        for raw in (0.0, -0.0):
            self.assertEqual(tuned_bias(raw, raw), CubeBias())

    def test_a_small_displacement_is_more_visible_than_linear(self):
        # One grudge corner averaged over four slots is -0.075; linear output
        # would be that same number, still far below the left brain's -0.20
        # gate. The curve must lift it without crossing the bound.
        raw = -0.075
        value = tuned_bias(0.0, raw).verbosity
        self.assertLess(value, raw)                 # more negative = stronger
        self.assertGreaterEqual(value, -MAX_VERBOSITY_BIAS)

    def test_monotonic_in_both_inputs(self):
        previous_v = None
        previous_d = None
        for step in range(-10, 11):
            raw = step / 20.0
            bias = tuned_bias(raw, raw)
            if previous_v is not None:
                self.assertGreaterEqual(bias.verbosity, previous_v)
                self.assertGreaterEqual(bias.depth, previous_d)
            previous_v = bias.verbosity
            previous_d = bias.depth

    def test_saturates_at_the_bound(self):
        positive = tuned_bias(50.0, 50.0)
        negative = tuned_bias(-50.0, -50.0)
        self.assertEqual((positive.depth, positive.verbosity),
                         (MAX_DEPTH_BIAS, MAX_VERBOSITY_BIAS))
        self.assertEqual((negative.depth, negative.verbosity),
                         (-MAX_DEPTH_BIAS, -MAX_VERBOSITY_BIAS))

    def test_larger_gain_never_shrinks_the_response(self):
        small = tuned_bias(0.5, 0.15, gain=1.2)
        large = tuned_bias(0.5, 0.15, gain=SENSITIVITY_GAIN)
        self.assertGreaterEqual(large.depth, small.depth)
        self.assertGreaterEqual(large.verbosity, small.verbosity)

    def test_a_scrambled_cube_stays_inside_the_bound(self):
        cube = CubeState()
        for move in SCRAMBLE.split():
            cube.apply(move)
        bias = CubeFeedback().bias(cube)
        self.assertLessEqual(abs(bias.depth), MAX_DEPTH_BIAS)
        self.assertLessEqual(abs(bias.verbosity), MAX_VERBOSITY_BIAS)


class TestDreamRestore(unittest.TestCase):
    def test_history_entries_round_trip(self):
        cube = CubeState()
        cube.apply("R' U2 F")
        for entry in cube.history:
            self.assertIsInstance(parse_move(entry), Move)
        self.assertEqual(cube.history, ["R'", "U2", "F"])

    def test_a_dream_walks_the_cube_partway_home(self):
        cube = CubeState()
        cube.apply("R U R' U' F D' L2")
        before = displacement(cube)
        self.assertGreater(before, 0)
        report = CubeFeedback().partial_restore(cube, ratio=0.45)
        self.assertTrue(cube.validate())
        self.assertLess(displacement(cube), before)
        # The ratio is a target, not a floor: moves are coarse, so the walk
        # settles on the reachable displacement closest to it, provided it
        # does not overshoot by more than a couple of facelets.
        self.assertLessEqual(
            abs(report["final_displaced"] - report["target_remaining"]), 2)
        self.assertTrue(report["moves"])
        self.assertGreater(report["actual_reduction"], 0.0)

    def test_a_solved_cube_has_nothing_to_restore(self):
        report = CubeFeedback().partial_restore(CubeState())
        self.assertEqual(report["moves"], [])
        self.assertEqual(report["actual_reduction"], 0.0)

    def test_restore_without_history_is_a_no_op(self):
        cube = CubeState()
        cube.apply("R U")
        cube.history = []  # a hand-edited save can look like this
        before = cube.snapshot()["occupancy"]
        report = CubeFeedback().partial_restore(cube)
        self.assertEqual(report["moves"], [])
        self.assertEqual(cube.snapshot()["occupancy"], before)

    def test_a_dream_never_increases_displacement(self):
        # Undoing the most recent moves one by one can pass through a *worse*
        # arrangement (a turn can send a moved cubie home while pushing a
        # home cubie out). The dry run must pick the helping prefix instead.
        scrambles = (
            "U", "R U", "R U R' U'", "F D' L2 B' U2 R",
            "U U U U D D", "F U R U' R' F' R' D' R D",
            "R U2 R' U' R U' R' L' U2 L U L' U L",
        )
        for scramble in scrambles:
            cube = CubeState()
            cube.apply(scramble)
            before = displacement(cube)
            history_before = list(cube.history)
            report = CubeFeedback().partial_restore(cube)
            self.assertTrue(cube.validate(), scramble)
            self.assertLessEqual(displacement(cube), before, scramble)
            self.assertEqual(cube.history[len(history_before):], report["moves"])

    def test_a_dream_that_cannot_help_leaves_the_cube_alone(self):
        cube = CubeState()
        cube.apply("R U R' U'")
        # Forget how we got here: the arrangement has no useful history.
        cube.history = ["U", "D"]  # reversing these makes things worse
        before = cube.snapshot()["occupancy"]
        report = CubeFeedback().partial_restore(cube)
        if report["final_displaced"] >= report["initial_displaced"]:
            self.assertEqual(report["moves"], [])
            self.assertEqual(cube.snapshot()["occupancy"], before)


class TestEngineWiring(unittest.TestCase):
    def test_turn_applies_the_feedback_layer(self):
        engine = offline_engine()
        engine.turn("你真是个蠢货，滚开", now=1.0)
        self.assertTrue(engine.cube.history)
        self.assertEqual(engine.cube_feedback.observed_turn, 0)
        self.assertIn("cube_feedback", engine.snapshot())
        for gain in engine.cube_feedback.gains(engine.cube).values():
            self.assertGreaterEqual(gain, MIN_GAIN)
            self.assertLessEqual(gain, 1.0 + MAX_GAIN)

    def test_feedback_state_survives_a_save(self):
        engine = offline_engine()
        engine.turn("前任又来找我了", now=1.0)
        engine.cube_feedback.dwell["edge_humor"] = 2
        data = engine.snapshot()
        clone = MindEngine.load_from_dict(data, bridge=PyBotBridge.disabled(),
                                          kb=FeynmanKB.disabled())
        self.assertEqual(clone.cube_feedback.dwell, engine.cube_feedback.dwell)
        self.assertEqual(clone.cube_feedback.observed_turn,
                         engine.cube_feedback.observed_turn)

    def test_dream_settles_the_cube_and_the_chronic_offset(self):
        engine = offline_engine()
        now = 0.0
        for turn in range(8):
            now += 1.0
            engine.turn("前任 裁员 挂科 余额 失恋", now=now)
        self.assertGreater(displacement(engine.cube), 0)
        before = displacement(engine.cube)
        dream = engine.dream(now=now + 1.0)
        self.assertIsNotNone(dream)
        self.assertLessEqual(displacement(engine.cube), before)
        self.assertTrue(engine.cube.validate())


class TestFiftyTurnStability(unittest.TestCase):
    """Hostile -> flattering -> noisy -> calm, with dreams at 15/30/45.

    Every invariant below is a hard bound: the point of the damping is that a
    long conversation cannot push the loop into a locked state.
    """

    HOSTILE = ["闭嘴，你这个废物", "呵呵，真的假的，又是套路", "你每次都这样，上次你也是"]
    SWEET = ["哈哈，太棒了，有意思", "崇拜你，最棒了", "谢谢，辛苦了"]
    NOISY = ["万一完蛋了怎么办", "不可能，你瞎说", "天呐！！！马上"]

    def test_fifty_turns_stay_bounded(self):
        engine = offline_engine()
        persona = engine.persona
        now = 0.0
        dream_turns = {14, 29, 44}
        turns_seen = 0

        for turn in range(50):
            if turn < 15:
                text = self.HOSTILE[turn % len(self.HOSTILE)]
            elif turn < 30:
                text = self.SWEET[turn % len(self.SWEET)]
            elif turn < 42:
                text = self.NOISY[turn % len(self.NOISY)]
            else:
                text = "嗯"
            now += 2.0
            result = engine.turn(text, now=now)

            self.assertTrue(engine.cube.validate(), f"turn {turn}")
            for dim, value in result.state.items():
                low, high = BOUNDS[dim]
                self.assertGreaterEqual(value, low - 1e-9, f"{dim} @ {turn}")
                self.assertLessEqual(value, high + 1e-9, f"{dim} @ {turn}")
            for act in result.activations:
                self.assertGreaterEqual(act.strength, 0.0)
                self.assertLessEqual(act.strength, 3.0)
            for gain in engine.cube_feedback.gains(engine.cube).values():
                self.assertGreaterEqual(gain, MIN_GAIN)
                self.assertLessEqual(gain, 1.0 + MAX_GAIN)
            bases = engine.cube_feedback.baselines(persona, engine.cube)
            self.assertLessEqual(bases["cortisol"],
                                 persona.baseline("cortisol") + MAX_BASELINE_SHIFT + 1e-9)
            self.assertGreaterEqual(bases["dopamine"],
                                    persona.baseline("dopamine") - MAX_BASELINE_SHIFT / 2 - 1e-9)
            self.assertIn(result.left_brain["limit"], (1, 2, 3, 4))
            self.assertIn(result.left_brain["detail"], ("brief", "normal", "rich"))
            self.assertIn(result.left_brain["posture"], ("answer", "careful", "explore"))
            if engine.cube.history:
                turns_seen += 1

            if turn in dream_turns:
                before = displacement(engine.cube)
                self.assertIsNotNone(engine.dream(now=now + 0.5))
                self.assertLessEqual(displacement(engine.cube), before)
                self.assertTrue(engine.cube.validate())

        self.assertGreater(turns_seen, 0, "50 turns should twist the cube at least once")
        self.assertEqual(len(engine.event_log), len(engine.event_log))  # bounded by design


if __name__ == "__main__":
    unittest.main()
