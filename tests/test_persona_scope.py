"""Persona scoping: soft data belongs to the persona, not the engine.

Two minds share one engine; they must not share what they were taught or how
their cards are linked. These tests pin the three places that used to be
global:
  * association edges      -> <persona>/data/assoc/assoc_edges.jsonl
  * learned overlay        -> <state base>/<persona>/kb_learned
  * persona resolution     -> personas/<id>/
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import runtime
from right.omega_mind import assoc
from right.omega_mind.feynman_kb import FeynmanKB

ROOT = Path(__file__).resolve().parents[1]


class TestStatePaths(unittest.TestCase):
    def test_state_base_follows_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {runtime.STATE_DIR_ENV: tmp}):
                self.assertEqual(runtime.state_base(), Path(tmp))

    def test_learned_dir_is_per_persona(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {runtime.STATE_DIR_ENV: tmp}):
                feynman = runtime.load(
                    runtime.resolve_persona("feynman-zh"))
                laozhuang = runtime.load(
                    runtime.resolve_persona("laozhuang-zh"))
                self.assertNotEqual(runtime.learned_dir_for(feynman),
                                    runtime.learned_dir_for(laozhuang))
                for persona in (feynman, laozhuang):
                    self.assertTrue(
                        runtime.learned_dir_for(persona).is_relative_to(
                            Path(tmp)))
                    self.assertIn(persona.id,
                                  str(runtime.learned_dir_for(persona)))

    def test_assoc_path_lives_under_the_persona(self):
        for name in ("feynman-zh", "laozhuang-zh"):
            persona = runtime.load(runtime.resolve_persona(name))
            path = runtime.assoc_path_for(persona)
            self.assertTrue(path.is_relative_to(persona.root))
            self.assertEqual(path.name, "assoc_edges.jsonl")


class TestBuildWiring(unittest.TestCase):
    def test_build_scopes_kb_data_to_the_persona(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {runtime.STATE_DIR_ENV: tmp}):
                engine = runtime.build("feynman-zh")
                self.assertEqual(engine.kb.learned_dir,
                                 Path(tmp) / "feynman-zh" / "kb_learned")
                self.assertTrue(engine.kb.assoc_path.is_relative_to(
                    ROOT / "personas" / "feynman-zh"))
                self.assertEqual(engine.kb.dir,
                                 ROOT / "personas" / "feynman-zh" / "knowledge")

    def test_two_engines_never_share_learned_or_edges(self):
        """Same cards, two ids: each mind gets its own overlay and edge file."""
        import json
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personas"
            source = ROOT / "personas" / "feynman-zh"
            for name in ("feynman-zh", "copycat"):
                target = root / name
                target.mkdir(parents=True)
                shutil.copytree(source / "knowledge", target / "knowledge")
                data = json.loads(
                    (source / "persona.json").read_text(encoding="utf-8"))
                data["id"] = name
                data["name"] = name
                (target / "persona.json").write_text(
                    json.dumps(data, ensure_ascii=False), encoding="utf-8")
            state = Path(tmp) / "state"
            with mock.patch.dict(os.environ, {runtime.STATE_DIR_ENV: str(state)}):
                engines = {name: runtime.build(name, personas_root=root)
                           for name in ("feynman-zh", "copycat")}
                for name, engine in engines.items():
                    self.assertEqual(engine.kb.learned_dir.resolve(),
                                     (state / name / "kb_learned").resolve())
                    self.assertEqual(
                        engine.kb.assoc_path.resolve(),
                        (root / name / "data" / "assoc"
                         / "assoc_edges.jsonl").resolve())
                self.assertNotEqual(engines["feynman-zh"].kb.learned_dir,
                                    engines["copycat"].kb.learned_dir)


class TestAssocIsolation(unittest.TestCase):
    def _edge_file(self, tmp: Path, name: str) -> Path:
        path = tmp / name / "data" / "assoc" / "assoc_edges.jsonl"
        path.parent.mkdir(parents=True)
        row = assoc.Edge("a-1", "b-1", kind="same_domain").to_row()
        path.write_text(
            __import__("json").dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return path

    def test_kb_reads_only_its_own_edge_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mine = self._edge_file(tmp, "mine")
            other = self._edge_file(tmp, "other")
            kb_mine = FeynmanKB(autoload=False, assoc_path=str(mine))
            kb_other = FeynmanKB(autoload=False, assoc_path=str(other))
            kb_none = FeynmanKB(autoload=False,
                                assoc_path=str(tmp / "absent.jsonl"))
            self.assertEqual(kb_mine._assoc_graph().path, mine)
            self.assertEqual(kb_other._assoc_graph().path, other)
            self.assertEqual(len(kb_mine._assoc_graph().edges), 1)
            self.assertEqual(kb_none._assoc_graph().edges, [])
            # Graph objects are per-KB, never module state.
            self.assertIsNot(kb_mine._assoc_graph(), kb_other._assoc_graph())


if __name__ == "__main__":
    unittest.main()
