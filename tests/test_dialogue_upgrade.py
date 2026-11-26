"""Exercise the new conversation paths through normal package assembly."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.runtime import build
from omega_mind import dialogue, evidence, review
from omega_mind.memory import Episode
from tools import ab_eval, learning_review

CONDENSATION = {
    "id": "condense", "type": "physics_card", "title": "Condensation",
    "text": "Vapor can form drops. Cooling lowers the saturation pressure. The pressure must reach saturation.",
    "passages": {
        "conclusion": "Vapor can form drops.",
        "mechanism": "Cooling lowers the saturation pressure.",
        "condition": "The pressure must reach saturation.",
    },
    "claim_key": "condensation", "claim_value": "drops",
}
EVAPORATION = {
    "id": "evaporate", "type": "physics_card", "title": "Evaporation",
    "text": "Liquid particles can escape from the surface.",
}
EDGE = {
    "a": "condense", "b": "evaporate",
    "kind": "same_mechanism", "origin": "human", "language": "en",
    "mechanism_en": "Particle exchange at a surface",
    "difference_en": "The net transfer reverses",
    "mapping": [["vapor pressure", "surface flux"]],
    "relation": "Exchange follows imbalance",
    "source": "Synthetic review example",
    "misconceptions": ["exchange ignores imbalance"],
}


class DialogueUpgrade(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        env = patch.dict(os.environ, {
            "PYBOT_STATE_DIR": str(self.root / "state"),
            "OMEGA_RETRIEVAL": "legacy", "OMEGA_POSTURE": "affect_cube",
            "OMEGA_REVIEW_LEARNING": "1", "OMEGA_STRUCTURED_EVIDENCE": "1",
        })
        env.start()
        self.addCleanup(env.stop)

    def package(self, name="sample-en", rows=None, edges=()):
        from bridge.persona import ROLE_COUNT
        from omega_mind.config import generic_persona

        package = self.root / name
        knowledge = package / "knowledge"
        knowledge.mkdir(parents=True, exist_ok=True)
        config = generic_persona()
        (package / "persona.json").write_text(json.dumps({
            "id": name, "language": "en", "schema": 1, "baselines": config.baselines,
            "taus": config.taus, "knowledge": {"kb_dirs": ["knowledge"]},
            "voice": {"prefixes_en": [], "endings_en": []},
            "cubies": [{"name": f"{role}{index}", "role": role}
                       for role, n in ROLE_COUNT.items() for index in range(n)],
        }))
        (knowledge / "cards.jsonl").write_text("\n".join(
            json.dumps(row) for row in (rows if rows is not None else [CONDENSATION, EVAPORATION])))
        if edges:
            path = package / "data" / "assoc" / "assoc_edges.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(edge) for edge in edges))
        return package

    def make(self, path=None, state=None):
        return build(path or self.package(), enable_bridge=True, state_path=state,
                     clock=lambda: 100.0)

    def test_temporal_memory_survives_dream_and_resume(self):
        path = self.package()
        state = self.root / "dialogue.state.json"
        mind = self.make(path, state)
        mind.turn("I live in Solport.", now=100)
        mind.turn("I moved from Solport to Westmere.", now=110)
        mind.turn("I no longer live in Westmere.", now=120)
        mind.turn("I no longer live in Solport.", now=125)
        mind.dream(now=130)
        mind = self.make(path, state)
        mind.tiers.record(Episode(0, 100, "default", "I live in Solport.", "", ""))
        self.assertEqual(mind.turn("Where do I live?").source, "memory:unknown")
        self.assertIn("Westmere", mind.turn("Where did I live before?").base_reply)
        self.assertEqual(mind.turn("Where do I live?", user="other").source, "memory:unknown")
        history = [entry for entry in mind.tiers.fact_log if entry.get("operation") == "set"]
        self.assertEqual([entry.get("valid_until") for entry in history], [110, 120])

    def test_hypothesis_reporting_and_quotation_do_not_write_memory(self):
        mind = self.make()
        for text in ("If I live in Solport, can I commute?", "Ari said, I live in Solport.",
                     'Ari wrote "I live in Solport."', "Where do I live?"):
            self.assertNotEqual(mind.turn(text).source, "memory:update")
        self.assertEqual(mind.tiers.semantic, {})

    def test_topic_history_is_specific_to_user_and_card_revision(self):
        path = self.package()
        state = self.root / "dialogue.state.json"
        mind = self.make(path, state)
        mind.turn("Condensation", user="a")
        mind.turn("Evaporation", user="a")
        mind = self.make(path, state)
        self.assertEqual(mind.turn("Go back to condensation.", user="b").source,
                         "clarification:en")
        self.assertEqual(mind.turn("Go back to condensation.", user="a").retrieval["dialogue_act"], "resume")
        changed = dict(CONDENSATION, description="new revision")
        self.package(rows=[changed, EVAPORATION])
        mind = self.make(path, state)
        self.assertEqual(mind.turn("Go back to condensation.", user="a").source,
                         "clarification:en")

    def test_foreign_save_is_rejected_without_writing(self):
        path = self.package()
        state = self.root / "dialogue.state.json"
        mind = self.make(path, state)
        mind.turn("My name is Rui.")
        saved = state.read_bytes()
        with self.assertRaisesRegex(ValueError, "another persona"):
            self.make(self.package("other-en"), state)
        self.assertEqual(saved, state.read_bytes())

    def test_structured_evidence_keeps_condition_and_offsets(self):
        row = dict(CONDENSATION)
        quote = row["passages"]["mechanism"]
        row["support"] = [{"source": "Synthetic source", "quote": quote,
                           "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                           "reviewed_by": "Synthetic reviewer"}]
        mind = self.make(self.package(rows=[row, EVAPORATION]))
        found = mind.kb.lookup("Condensation", structured=True, detail="brief", mode="shadow")
        self.assertIn("reach saturation", found.text)
        self.assertIn("condense", found.diagnostics["candidates"]["legacy"])
        card = mind.kb.by_id["condense"]
        rich = card.answer_for("rich")
        for passage in found.diagnostics["evidence_plan"]:
            start, end = passage["span"]
            self.assertEqual(rich[start:end], passage["text"])
            self.assertEqual(passage["revision"], dialogue.card_revision(card))
        why, reason = evidence.compose(card, "mechanism", "brief")
        self.assertEqual(reason, "")
        self.assertEqual(why[0]["provenance"], "reviewed_quote")
        self.assertEqual(evidence.compose(card, "example", "brief")[1], "insufficient_passages")

    def test_structured_conflict_clarifies_without_speaking_either_claim(self):
        conflict = dict(CONDENSATION, id="conflict", claim_value="gas")
        filler = [dict(EVAPORATION, id=f"other-{i}") for i in range(8)]
        mind = self.make(self.package(rows=[CONDENSATION, conflict, *filler]))
        result = mind.turn("Condensation")
        self.assertEqual(result.source, "clarification:en")
        self.assertEqual(result.retrieval["decision"], "conflict")
        self.assertNotIn("Vapor can form drops", result.reply)

    def test_typed_edge_tracks_provenance_and_analogy_misconception(self):
        candidate = dict(EDGE, origin="cooccur", relation="Incorrect relation")
        mind = self.make(self.package(edges=[candidate, EDGE]))
        found = mind.turn("Condensation")
        self.assertEqual(found.retrieval["association_edge"]["origin"], "human")
        self.assertEqual(found.retrieval["association_edge"]["source"], EDGE["source"])
        session = mind.challenge()
        self.assertIn("Evaporation", session.open_analogy("condense"))
        terms = "Vapor pressure and surface flux. Exchange follows imbalance. The net transfer reverses."
        wrong = session.submit(terms + " But exchange ignores imbalance.")
        self.assertEqual(wrong.verdict, "weak")
        self.assertEqual(session.submit(terms).verdict, "strong")

    def test_posture_modes_share_fact_selection(self):
        path = self.package()
        results = []
        for mode in ab_eval.POSTURES:
            mind = self.make(path)
            mind.posture_mode = mode
            mind.state.set("cortisol", .9)
            results.append(mind.turn("Condensation"))
        self.assertEqual({turn.source for turn in results}, {"sample-en:condense"})
        self.assertEqual(results[0].left_brain["detail"], "normal")
        self.assertEqual(results[1].left_brain["detail"], "brief")

    def test_review_replay_approval_and_live_rollback(self):
        package = self.package()
        mind = self.make(package)
        candidate = review.queue_candidate(mind.kb.learned, "correction", {
            "question": "What is a velodrift?", "text": "A velodrift is a synthetic flow example.",
        })
        self.assertTrue(candidate)
        self.assertTrue(mind.turn("What is a velodrift?").source.startswith("fallback:"))
        quote = "A velodrift is a synthetic flow example."
        (self.root / "test-source.txt").write_text(quote)
        proof = self.root / "proof.json"
        proof.write_text(json.dumps({"reviewer": "Example reviewer", "source": "Synthetic source",
                                     "quote": quote, "quote_sha256": review.digest(quote),
                                     "source_file": "test-source.txt"}))
        holdout = self.root / "comparison.json"
        holdout.write_text(json.dumps([{"question": "Evaporation", "card": "evaporate"}]))
        args = ["--id", candidate, "--proof", str(proof), "--holdout", str(holdout)]
        self.assertEqual(learning_review.main([str(package), "replay", *args]), 0)
        self.assertEqual(review.active(review.read(mind.kb.learned)), [])
        self.assertEqual(learning_review.main([str(package), "approve", *args]), 0)
        self.assertIn(quote, mind.turn("What is a velodrift?").base_reply)
        self.assertEqual(learning_review.main([str(package), "revoke", "--id", candidate]), 0)
        self.assertTrue(mind.turn("What is a velodrift?").source.startswith("fallback:"))

    def test_dream_records_an_explicit_correction_for_a_confident_hit(self):
        mind = self.make()
        self.assertEqual(mind.turn("Condensation").source, "sample-en:condense")
        mind.turn("Correction: cooling only explains this example under the stated pressure.")
        mind.dream(now=150)
        pending = review.read(mind.kb.learned)["candidates"]
        corrections = [item for item in pending.values() if item["kind"] == "correction"]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["previous_source"], "sample-en:condense")
        self.assertTrue(corrections[0]["text"].startswith("cooling"))
        self.assertEqual(review.active(review.read(mind.kb.learned)), [])

    def test_blind_transcript_keeps_turns_and_hides_mode(self):
        package = self.package()
        questions = self.root / "questions.jsonl"
        questions.write_text(json.dumps({"id": "one", "turns": ["I live in Solport.", "Where do I live?"]}))
        pairs, key = self.root / "pairs.jsonl", self.root / "key.json"
        self.assertEqual(ab_eval.make(self.root, package.name, package.name, questions,
                                      pairs, key, 7, "fixed", "affect_cube"), 0)
        pair = json.loads(pairs.read_text())
        self.assertEqual(len(pair["left"]), 2)
        self.assertNotIn("affect_cube", pairs.read_text())
        key_data = json.loads(key.read_text())
        self.assertNotEqual(key_data["_packages"]["a"], key_data["_packages"]["b"])
        report = ab_eval.score([pair], key_data, [{"id": "one", "axis": "role", "pick": "left"}])
        self.assertEqual(sum(report["axes"]["role"]["wins"].values()), 1)


if __name__ == "__main__":
    unittest.main()
