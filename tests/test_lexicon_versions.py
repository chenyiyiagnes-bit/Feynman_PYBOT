"""A build is whole; a rollback is a swap, never a patch.

The lexicon is derived data, and derived data that only ever moves forward is
unrecoverable. Every build writes one complete snapshot named by its content;
activating an older snapshot puts exactly those files back.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools", ROOT / "right"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import build_lexicon as bl  # noqa: E402
import lexicon_versions as lv  # noqa: E402

CARDS = [
    {"id": "a", "type": "card", "title": "热传导",
     "text": "热量从热的一端流到冷的一端，温差推动流动。"},
    {"id": "b", "type": "card", "title": "扩散",
     "text": "分子扩散让热量在材料里均匀开来，温差逐渐消失。"},
    {"id": "c", "type": "card", "title": "能量",
     "text": "能量守恒，热和功之间可以互相转换。"},
]


def make_package(tmp: Path, cards=None) -> Path:
    package = tmp / "pkg"
    (package / "knowledge").mkdir(parents=True, exist_ok=True)
    (package / "persona.json").write_text(
        json.dumps({"schema": 1, "id": "pkg", "name": "t", "language": "zh"}),
        encoding="utf-8")
    with (package / "knowledge" / "cards.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards if cards is not None else CARDS:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    return package


class Versions(unittest.TestCase):
    def test_same_cards_rebuild_to_the_same_version(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="versions-"))
        package = make_package(tmp)
        first = bl.build(package, built="2026-01-01")
        second = bl.build(package, built="2026-02-02")
        self.assertEqual(first["version"], second["version"])
        self.assertEqual(len(lv.versions(package / "data" / "lexicon")), 1)

    def test_changed_cards_keep_the_old_version(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="versions-"))
        package = make_package(tmp)
        old = bl.build(package)
        cards = CARDS + [{"id": "d", "type": "card", "title": "熵",
                          "text": "熵衡量不可用的能量，熵增是自发过程的方向。"}]
        make_package(tmp, cards)
        new = bl.build(package)
        self.assertNotEqual(old["version"], new["version"])
        rows = {row["version"] for row in lv.versions(package / "data" / "lexicon")}
        self.assertEqual(rows, {old["version"], new["version"]})
        self.assertEqual(lv.current(package / "data" / "lexicon"), new["version"])

    def test_rollback_restores_the_whole_file(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="versions-"))
        package = make_package(tmp)
        old = bl.build(package)
        old_terms = (package / "data" / "lexicon" / "terms.json").read_bytes()
        make_package(tmp, CARDS[:1])
        changed = bl.build(package)
        self.assertNotEqual(
            old_terms, (package / "data" / "lexicon" / "terms.json").read_bytes())
        lv.activate(package / "data" / "lexicon", old["version"])
        self.assertEqual(
            old_terms, (package / "data" / "lexicon" / "terms.json").read_bytes())
        self.assertEqual(lv.current(package / "data" / "lexicon"), old["version"])
        # The newer build is still there to go back to.
        self.assertIn(changed["version"],
                      {row["version"] for row in lv.versions(package / "data" / "lexicon")})

    def test_unknown_version_is_refused(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="versions-"))
        package = make_package(tmp)
        bl.build(package)
        with self.assertRaises(SystemExit):
            lv.activate(package / "data" / "lexicon", "deadbeefcafe")

    def test_the_current_pointer_names_the_manifest(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="versions-"))
        package = make_package(tmp)
        manifest = bl.build(package)
        pointer = json.loads((package / "data" / "lexicon" / "current.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(pointer["version"], manifest["version"])
        self.assertEqual(pointer["source_digest"], manifest["source_digest"])


if __name__ == "__main__":
    unittest.main()
