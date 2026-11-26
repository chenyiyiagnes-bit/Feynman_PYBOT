"""The affect parameters of a mind live in its package, nowhere else.

``get_persona`` takes ``"generic"`` or a package id under ``personas/`` and
reads ``persona.json`` from disk. There is no built-in copy of any persona's
parameters, so a missing package is an error rather than a silent fallback.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from omega_mind import config
from omega_mind.config import get_persona, load_package_persona


class TestPackagePersona(unittest.TestCase):
    def test_a_package_id_loads_that_package_s_manifest(self):
        package = config.PACKAGES_ROOT / "feynman-zh"
        data = json.loads((package / "persona.json").read_text(encoding="utf-8"))
        cfg = get_persona("feynman-zh")
        self.assertEqual(cfg.name, "feynman-zh")
        for dim, value in data.get("baselines", {}).items():
            self.assertAlmostEqual(cfg.baselines[dim], float(value))

    def test_a_package_directory_or_manifest_is_accepted(self):
        package = config.PACKAGES_ROOT / "feynman-zh"
        by_dir = load_package_persona(package)
        by_file = load_package_persona(package / "persona.json")
        self.assertEqual(by_dir, by_file)

    def test_an_unknown_id_is_an_error_not_a_default(self):
        with self.assertRaises(ValueError) as caught:
            get_persona("no-such-package")
        self.assertIn("no-such-package", str(caught.exception))

    def test_generic_stays_the_bare_config(self):
        self.assertEqual(get_persona("generic").name, "generic")

    def test_an_old_saved_name_still_reads_the_same_package(self):
        # Saves written before the package split said "feynman".
        self.assertEqual(get_persona("feynman"), get_persona("feynman-zh"))

    def test_no_persona_parameters_are_compiled_in(self):
        self.assertFalse(hasattr(config, "feynman_persona"))

    def test_the_cli_takes_a_package_id(self):
        from omega_mind import cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--persona", "feynman-zh", "--profile"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertTrue(text.strip())
        self.assertNotIn("没有知识库目录可分析", text)


if __name__ == "__main__":
    unittest.main()
