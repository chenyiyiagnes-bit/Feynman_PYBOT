"""Tests for the affect → typesetter mapping."""

from __future__ import annotations

import unittest

from bridge.mapping import OPERATORS, map_affect

BASELINES = {
    "dopamine": 0.60,
    "cortisol": 0.12,
    "valence": 0.15,
    "arousal": 0.40,
    "affinity": 0.25,
    "defensiveness": 0.0,
    "energy": 0.85,
}


def state(**overrides):
    s = dict(BASELINES)
    s.update(overrides)
    return s


def params(**overrides):
    return map_affect(state(**overrides), BASELINES)


class TestBounds(unittest.TestCase):
    def test_all_operator_weights_within_left_brain_bounds(self):
        for cur in (0.0, 0.5, 1.0):
            for stress in (0.0, 0.5, 1.0):
                for energy in (0.0, 0.5, 1.0):
                    p = params(dopamine=cur, cortisol=stress, energy=energy)
                    for name, value in p.operators.items():
                        self.assertGreaterEqual(value, 0.0, name)
                        self.assertLessEqual(value, 2.0, name)

    def test_style_within_bounds(self):
        for cur in (0.0, 1.0):
            for stress in (0.0, 1.0):
                p = params(dopamine=cur, cortisol=stress)
                self.assertGreaterEqual(p.style_deg, 0.0)
                self.assertLessEqual(p.style_deg, 120.0)

    def test_operators_have_exactly_eight_keys(self):
        p = params()
        self.assertEqual(tuple(sorted(p.operators)), tuple(sorted(OPERATORS)))
        self.assertEqual(len(OPERATORS), 8)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        first = map_affect(state(dopamine=0.8, cortisol=0.3), BASELINES,
                           cube_depth=0.4, cube_verbosity=-0.2)
        second = map_affect(state(dopamine=0.8, cortisol=0.3), BASELINES,
                            cube_depth=0.4, cube_verbosity=-0.2)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_at_baseline_curiosity_and_stress_are_zero(self):
        p = map_affect(state(), BASELINES)
        self.assertAlmostEqual(p.affects["curiosity"], 0.0, places=6)
        self.assertAlmostEqual(p.affects["stress"], 0.0, places=6)

    def test_at_baseline_energy_is_not_tired(self):
        p = map_affect(state(), BASELINES)
        self.assertAlmostEqual(p.affects["fatigue"], 0.0, places=6)


class TestMonotonicity(unittest.TestCase):
    def test_curiosity_raises_cross_and_anti_intuition(self):
        low = params(dopamine=0.60)
        high = params(dopamine=0.95)
        self.assertGreater(high.operators["cross"], low.operators["cross"])
        self.assertGreater(high.operators["anti_intuition"],
                           low.operators["anti_intuition"])

    def test_stress_raises_micro_and_jargon_strip(self):
        low = params(cortisol=0.12)
        high = params(cortisol=0.80)
        self.assertGreater(high.operators["micro"], low.operators["micro"])
        self.assertGreater(high.operators["jargon_strip"],
                           low.operators["jargon_strip"])

    def test_stress_lowers_style(self):
        low = params(cortisol=0.12)
        high = params(cortisol=0.80)
        self.assertLess(high.style_deg, low.style_deg)

    def test_fatigue_raises_least_action(self):
        rested = params(energy=0.85)
        tired = params(energy=0.20)
        self.assertGreater(tired.operators["least_action"],
                           rested.operators["least_action"])

    def test_guard_raises_cargo_cult(self):
        calm = params(defensiveness=0.0)
        guarded = params(defensiveness=0.7)
        self.assertGreater(guarded.operators["cargo_cult"],
                           calm.operators["cargo_cult"])


class TestFlags(unittest.TestCase):
    def test_high_stress_locks_topic(self):
        self.assertTrue(params(cortisol=0.9).flags["topic_lock"])
        self.assertFalse(params(cortisol=0.12).flags["topic_lock"])

    def test_isomorphism_needs_curiosity_and_depth(self):
        curious_moved = map_affect(state(dopamine=0.95), BASELINES, cube_depth=0.5)
        self.assertTrue(curious_moved.flags["use_isomorphism"])
        curious_still = map_affect(state(dopamine=0.95), BASELINES, cube_depth=0.0)
        self.assertFalse(curious_still.flags["use_isomorphism"])
        calm_moved = map_affect(state(dopamine=0.60), BASELINES, cube_depth=0.5)
        self.assertFalse(calm_moved.flags["use_isomorphism"])

    def test_conditioned_chain_needs_engagement_without_fatigue(self):
        self.assertTrue(params(arousal=0.8, energy=0.85).flags["conditioned_chain"])
        self.assertFalse(params(arousal=0.8, energy=0.1).flags["conditioned_chain"])

    def test_rough_exchange_asks_for_resolution(self):
        self.assertTrue(params(cortisol=0.9).flags["dissonance_resolution"])
        self.assertTrue(params(defensiveness=0.8).flags["dissonance_resolution"])
        self.assertFalse(
            params(cortisol=0.12, defensiveness=0.0).flags["dissonance_resolution"])


class TestJargonAversion(unittest.TestCase):
    def test_aversion_adds_to_jargon_strip(self):
        plain = map_affect(state(), BASELINES, jargon_aversion=False)
        averse = map_affect(state(), BASELINES, jargon_aversion=True)
        self.assertGreater(averse.operators["jargon_strip"],
                           plain.operators["jargon_strip"])


class TestLeftBrainAgreement(unittest.TestCase):
    """The local constants must match the left brain's pydantic models."""

    def test_operator_names_match_left_brain(self):
        try:
            import plugins.feynman_typesetter as typesetter
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"left brain not importable: {exc}")
        fields = set(typesetter.OperatorWeights.model_fields)
        self.assertEqual(fields, set(OPERATORS))

    def test_style_bounds_match_left_brain(self):
        try:
            import plugins.paragraph_weaver as weaver
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"left brain not importable: {exc}")
        field = weaver.ParagraphRequest.model_fields["style_deg"]
        metadata = field.metadata
        bounds = {getattr(m, "ge", None) for m in metadata} | {
            getattr(m, "le", None) for m in metadata}
        self.assertIn(0.0, bounds)
        self.assertIn(120.0, bounds)


if __name__ == "__main__":
    unittest.main()
