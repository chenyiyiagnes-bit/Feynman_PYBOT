"""The retrieval regression net, loaded from the packages' own data.

The skeleton holds no card ids. A package that wants its retrieval watched
ships ``evals/golden.json`` next to its knowledge; the file names questions
that must keep hitting the right card, and traps that must stay unanswered.
A package without the file simply has no net.

File shape::

    {
      "schema": 1,
      "language": "zh",
      "golden": [{"query": "熵是什么", "acceptable": ["card-id", "..."]}],
      "negatives": ["一句不该被回答的话"]
    }

Two module-level tuples stay for callers that want the merged net across all
packages (the tests and ``tools/eval_segmentation.py``); :func:`for_package`
gives one package's rows, which is what a running mind audits itself against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import PACKAGES_ROOT

GoldenRow = Tuple[str, str, frozenset]
TrapRow = Tuple[str, str]
Net = Tuple[Tuple[GoldenRow, ...], Tuple[TrapRow, ...]]


def _manifest_language(package: Path) -> str:
    try:
        data = json.loads((package / "persona.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "zh"
    return str(data.get("language") or "zh")


def _read_net(package: Path) -> Optional[Net]:
    path = package / "evals" / "golden.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    language = str(data.get("language") or _manifest_language(package))
    golden: List[GoldenRow] = []
    traps: List[TrapRow] = []
    for row in data.get("golden") or []:
        query = str(row.get("query") or "").strip()
        acceptable: Iterable[str] = row.get("acceptable") or []
        if query:
            golden.append((query, language, frozenset(map(str, acceptable))))
    for row in data.get("negatives") or []:
        query = (row.get("query") if isinstance(row, dict) else row) or ""
        if str(query).strip():
            traps.append((str(query).strip(), language))
    return tuple(golden), tuple(traps)


def nets_by_package(root: Path | str = PACKAGES_ROOT) -> Dict[str, Net]:
    """Every package that ships a net, keyed by package name."""
    nets: Dict[str, Net] = {}
    root = Path(root)
    if not root.is_dir():
        return nets
    for package in sorted(root.iterdir()):
        net = _read_net(package)
        if net is not None:
            nets[package.name] = net
    return nets


def load_golden(root: Path | str = PACKAGES_ROOT) -> Net:
    """Every package's net, merged, in package-name order.

    A missing ``evals/golden.json`` means the package has no net, not an
    error: the file is optional by design.
    """
    golden: List[GoldenRow] = []
    traps: List[TrapRow] = []
    for net in nets_by_package(root).values():
        golden.extend(net[0])
        traps.extend(net[1])
    return tuple(golden), tuple(traps)


def for_package(name: str, root: Path | str = PACKAGES_ROOT) -> Net:
    """One package's rows, or two empty tuples when it ships no net."""
    return nets_by_package(root).get(str(name), ((), ()))


GOLDEN, NEGATIVES = load_golden()
