"""List and roll back whole builds of a package's derived lexicon.

A build is never patched: ``build_lexicon.py`` writes the whole thing into
``versions/<id>/`` and the top-level files are a copy of one version. Rolling
back means copying another version's files back into place — the archive is
kept, so the version that was active before a rebuild can be restored exactly.

Because the id is a hash of the content, two builds of the same cards carry
the same id; a changed card file means a new id and the old one stays.

Usage:
    python3 tools/lexicon_versions.py list personas/feynman-zh
    python3 tools/lexicon_versions.py activate personas/feynman-zh 0a1b2c3d4e5f
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


FILES = ("terms.json", "edges.jsonl", "manifest.json")


def lexicon_dir(persona: Path) -> Path:
    return Path(persona) / "data" / "lexicon"


def current(directory: Path) -> str | None:
    path = Path(directory) / "current.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("version")


def versions(directory: Path) -> List[Dict]:
    root = Path(directory) / "versions"
    out: List[Dict] = []
    if not root.is_dir():
        return out
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        out.append({"version": folder.name, "built": manifest.get("built"),
                    "terms": manifest.get("terms"), "edges": manifest.get("edges"),
                    "source_digest": manifest.get("source_digest")})
    return out


def activate(directory: Path, version: str) -> Dict:
    """Copy one archived version over the active files. Full swap, no merge."""
    directory = Path(directory)
    source = directory / "versions" / version
    if not source.is_dir():
        raise SystemExit(f"no version {version!r} under {directory / 'versions'}")
    for name in FILES:
        if not (source / name).is_file():
            raise SystemExit(f"version {version!r} is incomplete: {name} is missing")
    for name in FILES:
        shutil.copyfile(source / name, directory / name)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    (directory / "current.json").write_text(
        json.dumps({"version": version,
                    "source_digest": manifest.get("source_digest"),
                    "built": manifest.get("built")},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="show archived versions")
    listing.add_argument("persona")
    switching = sub.add_parser("activate", help="make an archived version active")
    switching.add_argument("persona")
    switching.add_argument("version")
    args = parser.parse_args(argv)

    directory = lexicon_dir(Path(args.persona))
    if args.command == "list":
        active = current(directory)
        rows = versions(directory)
        if not rows:
            print(f"{directory}: no versions (run tools/build_lexicon.py)")
            return 0
        for row in rows:
            mark = "*" if row["version"] == active else " "
            print(f"{mark} {row['version']}  {row['built']}  "
                  f"{row['terms']} terms, {row['edges']} edges  "
                  f"cards {row['source_digest']}")
        return 0

    manifest = activate(directory, args.version)
    print(f"active version is now {manifest['version']} "
          f"(cards {manifest['source_digest']}); rebuild the coding with "
          f"tools/code_cards.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
