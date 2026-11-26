"""P21 相位带宽：每个字粒在语料里实际出现时的语体相位分布（圆均值 + 集中度）。

字库里的 `phase_deg` 是首次出现那一句的相位，噪声大；这里统计字粒在全部
出现句上的圆周分布，得到更稳的均值 μ 和集中度 R。运行时按
`align = 1 − r·(1 − cos(μ − target))` 打分，`r = R²·n/(n+10)` 是可靠性：
只出现两三次的字粒不会被当成"相位很准"，偏离目标也几乎不扣分。

确定性：语料顺序固定、累加浮点顺序固定，两次构建逐字节一致。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_phase_bandwidth_p21.py
    PYTHONPATH=. .venv/bin/python -B scripts/build_phase_bandwidth_p21.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from plugins.feynman_vault import _style_of, sentences

CORPUS = Path("data/p8_corpus_v2.jsonl")
ISOMORPHISM = Path("data/isomorphism_p21.json")
OUT = Path("data/phase_bandwidth_p21.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(corpus: Path, words: List[str]) -> Tuple[Dict[str, List[float]], int]:
    by_first: Dict[str, List[str]] = {}
    for w in words:
        by_first.setdefault(w[0], []).append(w)
    for cands in by_first.values():
        cands.sort(key=len, reverse=True)
    acc = {w: [0.0, 0.0, 0] for w in words}  # cos 和、sin 和、句数
    n_sent = 0
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            for sent in sentences(text):
                n_sent += 1
                hit = set()
                for ch in sent:
                    for w in by_first.get(ch, ()):
                        if w in sent:
                            hit.add(w)
                if not hit:
                    continue
                _, phase = _style_of(sent)
                c = math.cos(math.radians(phase))
                s = math.sin(math.radians(phase))
                for w in hit:
                    rec = acc[w]
                    rec[0] += c
                    rec[1] += s
                    rec[2] += 1
    return acc, n_sent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--isomorphism", type=Path, default=ISOMORPHISM)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    iso = json.loads(args.isomorphism.read_text(encoding="utf-8"))
    words = sorted((iso.get("vectors") or {}).keys())
    acc, n_sent = scan(args.corpus, words)

    out: Dict[str, dict] = {}
    for w in words:
        c, s, n = acc[w]
        if n == 0:
            continue
        r = math.hypot(c, s) / n
        mu = math.degrees(math.atan2(s, c)) % 360.0
        out[w] = {
            "mu": round(mu, 1),
            "R": round(r, 4),
            "n": n,
            "reliability": round((r * r) * n / (n + 10.0), 4),
        }
    payload = {
        "schema": "phase_bandwidth_p21.v1",
        "meta": {
            "corpus_sha256": _sha256(args.corpus),
            "isomorphism_sha256": _sha256(args.isomorphism),
            "sentences": n_sent,
            "words": len(words),
            "covered": len(out),
        },
        "bands": out,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if args.check:
        if not args.out.exists():
            raise SystemExit(f"❌ {args.out} 不存在")
        if args.out.read_text(encoding="utf-8") != text:
            raise SystemExit(f"❌ 重建结果与 {args.out} 不一致")
        print(f"✅ {args.out} 与重建结果逐字节一致")
        return
    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在，拒绝覆盖")
    args.out.write_text(text, encoding="utf-8")
    print(f"{len(out)}/{len(words)} 个字粒有相位分布，语料 {n_sent} 句 -> {args.out}")


if __name__ == "__main__":
    main()
