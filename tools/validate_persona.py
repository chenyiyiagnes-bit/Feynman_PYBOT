"""Validate a persona.json against contracts/persona.md.

Usage:
    python3 tools/validate_persona.py personas/feynman/persona.json

Exit code 0 = valid. Warnings do not fail the run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIMS = ("dopamine", "cortisol", "valence", "arousal", "affinity",
        "defensiveness", "energy")
ROLE_COUNT = {"core": 6, "edge": 12, "corner": 8}
FACES = {"U", "D", "L", "R", "F", "B"}


def validate(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"cannot read JSON: {exc}"], []

    if not isinstance(data, dict):
        return [f"persona must be a JSON object, got {type(data).__name__}"], []

    if data.get("schema") != 1:
        errors.append(f"schema must be 1, got {data.get('schema')!r}")

    for key in ("id", "name", "language"):
        if not isinstance(data.get(key), str) or not data.get(key):
            errors.append(f"{key} must be a non-empty string")
    if isinstance(data.get("language"), str) and data["language"] not in ("zh", "en"):
        errors.append(f"language must be zh or en, got {data['language']!r}")

    def is_number(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    baselines = data.get("baselines") or {}
    taus = data.get("taus") or {}
    for dim in DIMS:
        if dim not in baselines:
            errors.append(f"baselines missing {dim}")
        elif not is_number(baselines[dim]):
            errors.append(f"baselines[{dim}] must be a number")
        if dim not in taus:
            errors.append(f"taus missing {dim}")
        elif not is_number(taus[dim]) or taus[dim] <= 0:
            errors.append(f"taus[{dim}] must be a positive number")
    if is_number(baselines.get("valence")):
        if not -1.0 <= baselines["valence"] <= 1.0:
            errors.append("baselines[valence] must be within [-1, 1]")
    for dim in DIMS:
        if dim == "valence":
            continue
        v = baselines.get(dim)
        if is_number(v) and not 0.0 <= v <= 1.0:
            errors.append(f"baselines[{dim}] must be within [0, 1]")

    voice = data.get("voice") or {}
    for key in ("prefixes_zh", "endings_zh", "prefixes_en", "endings_en"):
        if not isinstance(voice.get(key), list) or not voice.get(key):
            errors.append(f"voice.{key} must be a non-empty list")

    cubies = data.get("cubies") or []
    if len(cubies) != 26:
        errors.append(f"cubies must be exactly 26, got {len(cubies)}")
    seen: set[str] = set()
    counts = dict.fromkeys(ROLE_COUNT, 0)
    for i, c in enumerate(cubies):
        name = c.get("name")
        if not name:
            errors.append(f"cubies[{i}] missing name")
            continue
        if name in seen:
            errors.append(f"duplicate cubie {name}")
        seen.add(name)
        role = c.get("role")
        if role not in ROLE_COUNT:
            errors.append(f"{name}: role must be one of core/edge/corner")
        else:
            counts[role] += 1
        if c.get("face") not in FACES:
            errors.append(f"{name}: face must be one of U/D/L/R/F/B")
        val = c.get("valence")
        if not is_number(val) or not -1.0 <= val <= 1.0:
            errors.append(f"{name}: valence must be within [-1, 1]")
        triggers = c.get("triggers") or []
        if not triggers:
            warnings.append(f"{name}: no triggers")
        for j, item in enumerate(triggers):
            if not isinstance(item, list) or len(item) != 2:
                errors.append(f"{name}: triggers[{j}] must be [pattern, gain]")
                continue
            pattern, gain = item
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"{name}: bad regex {pattern!r}: {exc}")
            if not is_number(gain) or not 0 < gain <= 3:
                errors.append(f"{name}: triggers[{j}] gain must be within (0, 3]")
        for dim, delta in (c.get("effects") or {}).items():
            if dim not in DIMS:
                errors.append(f"{name}: effects key {dim!r} is not a dimension")
            elif not is_number(delta):
                errors.append(f"{name}: effects[{dim}] must be a number")

    for role, want in ROLE_COUNT.items():
        if counts[role] != want:
            errors.append(f"role {role}: expected {want}, got {counts[role]}")

    for name, patterns in (data.get("override_triggers") or {}).items():
        if name not in seen:
            warnings.append(f"override_triggers references unknown cubie {name}")
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"override_triggers[{name}]: bad regex: {exc}")

    switches = data.get("switches") or {}
    if not isinstance(switches, dict):
        errors.append("switches must be an object")
    else:
        for key in ("correction_as_curiosity", "jargon_aversion"):
            if key in switches and not isinstance(switches[key], bool):
                errors.append(f"switches.{key} must be a boolean")
        if "politeness" in switches and not is_number(switches["politeness"]):
            errors.append("switches.politeness must be a number")

    for name in data.get("disable_cubies") or []:
        if name not in seen:
            warnings.append(f"disable_cubies references unknown cubie {name}")

    extra = data.get("extra_effects") or {}
    if not isinstance(extra, dict):
        errors.append("extra_effects must be an object")
    else:
        for name, effects in extra.items():
            if name not in seen:
                warnings.append(f"extra_effects references unknown cubie {name}")
            if not isinstance(effects, dict):
                errors.append(f"extra_effects[{name}] must be an object")
                continue
            for dim, delta in effects.items():
                if dim not in DIMS:
                    errors.append(f"extra_effects[{name}]: {dim!r} is not a dimension")
                elif not is_number(delta):
                    errors.append(f"extra_effects[{name}][{dim}] must be a number")

    knowledge = data.get("knowledge") or {}
    for d in knowledge.get("kb_dirs") or []:
        target = (path.parent / d).resolve()
        if not target.exists():
            warnings.append(f"knowledge dir missing: {target}")
    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    failed = 0
    for raw in argv[1:]:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        errors, warnings = validate(path)
        status = "OK" if not errors else "FAIL"
        print(f"[{status}] {path}  ({len(warnings)} warnings)")
        for w in warnings:
            print(f"  warn: {w}")
        for e in errors:
            print(f"  error: {e}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
