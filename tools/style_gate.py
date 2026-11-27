"""CLI around ``bridge.style_gate``: band, check, golden suite, selftest.

  python3 tools/style_gate.py --package feynman-zh --text "……"
  python3 tools/style_gate.py --suite            # golden net -> KB answer
  python3 tools/style_gate.py --selftest         # adversarial samples

The band and the check live in ``bridge/style_gate.py`` so the App can run
the same measurement at its reply exit. This file only loads packages and
prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "right"))
sys.path.insert(0, str(ROOT))

from bridge import style_gate as gate  # noqa: E402
from style_probe import load_package  # noqa: E402


def band_for(pkg: dict) -> dict:
    from omega_mind.feynman_kb import FeynmanKB
    kb = FeynmanKB(knowledge_dir=str(ROOT / "personas" / pkg["name"] / "knowledge"),
                   language=pkg["lang"])
    stem_by_id = {row.get("id"): row.get("_stem", "?") for row in pkg["rows"]}
    return gate.band_for_docs(pkg["lang"], kb.docs, stem_by_id)


def _kb_answer(package: str, lang: str, query: str) -> tuple[str, str]:
    from omega_mind.feynman_kb import FeynmanKB
    kb = FeynmanKB(knowledge_dir=str(ROOT / "personas" / package / "knowledge"),
                   language=lang)
    text, source = kb.answer(query)
    return text or "", source


def run_suite(names: list[str]) -> int:
    for name in names:
        pkg = load_package(name)
        bands = band_for(pkg)
        stem_by_id = {row.get("id"): row.get("_stem") for row in pkg["rows"]}
        overall_pass = register_pass = total = 0
        failures = []
        for item in pkg["golden"]["golden"]:
            query = item["query"]
            reply, source = _kb_answer(name, pkg["lang"], query)
            stem = stem_by_id.get(source.split(":", 1)[-1]) or "unknown"
            register = bands["by_source"].get(stem, bands["overall"])
            outside = gate.check(reply, bands["overall"])
            inside = gate.check(reply, register)
            total += 1
            overall_pass += outside["ok"]
            register_pass += inside["ok"]
            if not inside["ok"]:
                bad = [k for k, v in inside["checks"].items() if not v]
                failures.append((query, stem, bad, inside["metrics"]))
        o = bands["overall"]
        print(f"== {name} band overall mean[{o['mean_min']},{o['mean_max']}] "
              f"cap={o['sentence_cap']}  golden replies: "
              f"single-band {overall_pass}/{total}, "
              f"own-register {register_pass}/{total}")
        for query, stem, bad, metrics in failures:
            print(f"   FAIL {query} [{stem}] {bad} {metrics}")
    return 0


def selftest() -> int:
    """Adversarial replies must fail; the corpus itself must mostly pass."""
    pkg = load_package("feynman-zh")
    band = band_for(pkg)["overall"]
    good = "费曼说，如果你不能把一件事讲给大一新生听，就说明你自己没搞懂。"
    bad_samples = {
        "ai_cliche": "首先，我们要理解这个问题的本质。其次，值得注意的是它涉及多个维度。综上所述，这就是答案。",
        "runaway": "这是一个非常长的句子" * 40 + "。",
        "wrong_language": "So anyway, the thing about light is that it just keeps going.",
        "empty": "",
    }
    print("good :", json.dumps(gate.check(good, band), ensure_ascii=False))
    ok = gate.check(good, band)["ok"]
    for label, text in bad_samples.items():
        result = gate.check(text, band)
        print(f"{label:15s}", json.dumps(result, ensure_ascii=False))
        ok = ok and not result["ok"]
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", action="append", default=[])
    ap.add_argument("--text")
    ap.add_argument("--file", help="one reply per line")
    ap.add_argument("--suite", action="store_true", help="golden net through the KB")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    names = args.package or ["feynman-zh", "feynman-en",
                             "laozhuang-zh", "laozhuang-en"]
    if args.suite:
        return run_suite(names)

    name = names[0]
    pkg = load_package(name)
    band = band_for(pkg)["overall"]
    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        texts.extend(Path(args.file).read_text(encoding="utf-8").splitlines())
    if not texts:
        ap.error("give --text, --file, --suite or --selftest")
    print(f"band {name}: {json.dumps(band, ensure_ascii=False)}")
    for text in texts:
        print(json.dumps(gate.check(text, band), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
