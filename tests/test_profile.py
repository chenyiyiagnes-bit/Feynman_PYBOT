"""profile.py shows every remembered fact with its quote; it changes nothing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import profile


def write_state(base: Path, persona: str, save: dict) -> Path:
    path = base / f"{persona}.mind_state.json"
    path.write_text(json.dumps(save, ensure_ascii=False), encoding="utf-8")
    return path


class TestProfile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ,
                                    {"PYBOT_STATE_DIR": str(self.base)})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _save(self):
        write_state(self.base, "feynman-zh", {
            "tiers": {
                "semantic": {
                    "name:default": {"value": "用户的名字是 小明", "raw": "小明",
                                     "quote": "我叫小明", "source_turn": 4},
                    "likes:amy": {"value": "用户喜欢 木工", "raw": "木工",
                                  "quote": "我特别喜欢木工", "source_turn": 9},
                },
                "withdrawn": {
                    "lives_in:default": {"raw": "上海", "quote": "我住在上海",
                                         "source_turn": 2},
                },
            },
        })

    def test_each_fact_carries_its_quote_and_turn(self):
        self._save()
        rows = profile.collect()
        self.assertEqual(len(rows), 3)
        name = next(r for r in rows if r["kind"] == "name")
        self.assertEqual(name["quote"], "我叫小明")
        self.assertEqual(name["turn"], 4)
        self.assertEqual(name["state"], "current")

    def test_a_retracted_fact_is_shown_as_withdrawn(self):
        self._save()
        row = next(r for r in profile.collect() if r["kind"] == "lives_in")
        self.assertEqual(row["state"], "withdrawn")
        self.assertEqual(row["raw"], "上海")

    def test_personas_and_users_can_be_narrowed(self):
        self._save()
        self.assertEqual(len(profile.collect(only="feynman-zh")), 3)
        self.assertEqual(profile.collect(only="someone-else"), [])
        self.assertEqual(len(profile.collect(user="amy")), 1)

    def test_the_state_file_is_not_touched(self):
        path = write_state(self.base, "feynman-zh", {"tiers": {"semantic": {
            "name:default": {"value": "v", "raw": "小明", "quote": "我叫小明",
                             "source_turn": 1}}}})
        before = path.read_bytes()
        profile.collect()
        self.assertEqual(path.read_bytes(), before)

    def test_an_unreadable_save_is_skipped(self):
        (self.base / "broken.mind_state.json").write_text("{not json",
                                                          encoding="utf-8")
        self.assertEqual(profile.collect(), [])

    def test_json_mode_prints_one_fact_per_line(self):
        self._save()
        with mock.patch("sys.stdout") as out:
            self.assertEqual(profile.main(["--json"]), 0)
        lines = [call.args[0] for call in out.write.call_args_list]
        payload = [json.loads(line) for line in lines if line.strip()]
        self.assertEqual(len(payload), 3)
        self.assertIn("quote", payload[0])


if __name__ == "__main__":
    unittest.main()
