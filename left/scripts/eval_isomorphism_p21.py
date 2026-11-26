"""P21 表征质量评估：留出句上的"共现回填"任务，不需要 LLM 判读。

做法：把语料按行切 9:1，用前 90% 建矩阵，在留出 10% 的句子上出题——
句子里真实出现的实体/动作/后果是正确答案，另抽 3 个同槽干扰项，看向量能否
把真答案排到第一。两个方向都测：

- 后果题：已知 (实体, 动作)，从 4 个后果里选真正同句出现的那个；
- 动作题：已知 (实体, 后果)，从 4 个动作里选真正同句出现的那个。

同时对比几种表示：
- `baseline`：与线上产物同款（句级等权共现，k=12）；
- `k24`：同款共现，k=24（谱很平，检验 12 维是否欠拟合）；
- `dist`：±8 字内距离加权共现（w=1/(1+d)），k=12。

只为选型，不改线上产物。结果写 `data/p21_eval.json` / `.md`，含语料与字库哈希。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/eval_isomorphism_p21.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scripts.build_isomorphism_p21 import (
    CORPUS,
    INIT_SEED,
    VAULT,
    build_ppmi,
    load_glyphs,
    sentences,
    truncated_svd,
)

OUT_JSON = Path("data/p21_eval.json")
OUT_MD = Path("data/p21_eval.md")
LAMBDA_WINDOW = 8      # 距离加权臂的最大字符距离
N_TRIALS = 1500        # 每个任务的题数
N_DISTRACTORS = 3
TRIAL_SEED = 0x21B08


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path: Path) -> List[str]:
    lines = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            lines.append(text)
    return lines


def first_index(words: Sequence[str]) -> Dict[str, List[str]]:
    by_first: Dict[str, List[str]] = {}
    for w in words:
        by_first.setdefault(w[0], []).append(w)
    for cands in by_first.values():
        cands.sort(key=len, reverse=True)
    return by_first


def hits(sent: str, by_first: Dict[str, List[str]]) -> List[str]:
    out = set()
    for ch in sent:
        for w in by_first.get(ch, ()):
            if w in sent:
                out.add(w)
    return sorted(out)


def scan_flat(texts: Sequence[str], words: Sequence[str]):
    """句级等权共现（与线上构建同规则）。"""
    by_first = first_index(words)
    pair: Dict[str, int] = {}
    presence = {w: 0 for w in words}
    n_sent = 0
    for text in texts:
        for sent in sentences(text):
            n_sent += 1
            hit = hits(sent, by_first)
            if not hit:
                continue
            for w in hit:
                presence[w] += 1
            for i in range(len(hit)):
                for j in range(i + 1, len(hit)):
                    key = f"{hit[i]}|{hit[j]}"
                    pair[key] = pair.get(key, 0) + 1
    return pair, presence, n_sent


def scan_dist(texts: Sequence[str], words: Sequence[str]):
    """距离加权共现：同一句内、两词首现位置相距 ≤ LAMBDA_WINDOW 字符，权 1/(1+d)。"""
    by_first = first_index(words)
    pair: Dict[str, float] = {}
    presence = {w: 0 for w in words}
    n_sent = 0
    for text in texts:
        for sent in sentences(text):
            n_sent += 1
            pos: Dict[str, int] = {}
            for ch_i, ch in enumerate(sent):
                for w in by_first.get(ch, ()):
                    if w not in pos and w in sent:
                        pos[w] = sent.find(w)
            if not pos:
                continue
            for w in pos:
                presence[w] += 1
            hw = sorted(pos)
            for i in range(len(hw)):
                for j in range(i + 1, len(hw)):
                    d = abs(pos[hw[i]] - pos[hw[j]])
                    if d > LAMBDA_WINDOW:
                        continue
                    key = f"{hw[i]}|{hw[j]}"
                    pair[key] = pair.get(key, 0.0) + 1.0 / (1.0 + d)
    return pair, presence, n_sent


def vectors_from(pair, presence, n_sent, words: List[str], k: int):
    rows, _ = build_ppmi(words, pair, presence, n_sent)
    X, lam, _ = truncated_svd(rows, len(words), k)
    unit: Dict[str, List[float]] = {}
    for c in range(k):
        col = [X[i][c] for i in range(len(words))]
        peak = max(range(len(col)), key=lambda i: (abs(col[i]), -i))
        if col[peak] < 0:
            col = [-v for v in col]
        for i, w in enumerate(words):
            unit.setdefault(w, [])
            unit[w].append(col[i])
    for w in unit:
        n = math.sqrt(sum(x * x for x in unit[w])) or 1.0
        unit[w] = [x / n for x in unit[w]]
    return unit


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _unit(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def collect_pairs(lines: Sequence[str], slot_of: Dict[str, str],
                  limit: int = 3000) -> Dict[str, List[Tuple[str, str]]]:
    """留出句里的共现对：E→C、E→A（题量大，统计功效够）。"""
    words = sorted(slot_of)
    by_first = first_index(words)
    out: Dict[str, List[Tuple[str, str]]] = {"consequence": [], "action": []}
    seen = {"consequence": set(), "action": set()}
    for text in lines:
        for sent in sentences(text):
            hs = hits(sent, by_first)
            if not hs:
                continue
            ents = [w for w in hs if slot_of[w] == "entity"]
            acts = [w for w in hs if slot_of[w] == "action"]
            cons = [w for w in hs if slot_of[w] == "consequence"]
            for e in ents:
                for c in cons:
                    if (e, c) not in seen["consequence"]:
                        seen["consequence"].add((e, c))
                        out["consequence"].append((e, c))
                for a in acts:
                    if (e, a) not in seen["action"]:
                        seen["action"].add((e, a))
                        out["action"].append((e, a))
            if all(len(out[t]) >= limit for t in out):
                return out
    return out


def collect_triples(lines: Sequence[str], slot_of: Dict[str, str],
                    limit: int = 6000) -> List[Tuple[str, str, str]]:
    """留出句里同时含实体/动作/后果的句子 → (实体, 动作, 后果) 三元组。"""
    words = sorted(slot_of)
    by_first = first_index(words)
    seen = set()
    out: List[Tuple[str, str, str]] = []
    for text in lines:
        for sent in sentences(text):
            hs = hits(sent, by_first)
            if not hs:
                continue
            ents = [w for w in hs if slot_of[w] == "entity"]
            acts = [w for w in hs if slot_of[w] == "action"]
            cons = [w for w in hs if slot_of[w] == "consequence"]
            if not (ents and acts and cons):
                continue
            tri = (ents[0], acts[0], cons[0])
            if tri in seen:
                continue
            seen.add(tri)
            out.append(tri)
            if len(out) >= limit:
                return out
    return out


def _trial_metrics(unit, items, slot_of, gvec_fn, tslot) -> dict:
    vocab = sorted(w for w, sl in slot_of.items() if sl == tslot)
    rng = random.Random(TRIAL_SEED)
    top1 = 0
    mrr = 0.0
    n = 0
    for given, target in items:
        if target not in unit:
            continue
        pool = [w for w in vocab if w != target]
        distract = rng.sample(pool, N_DISTRACTORS)
        cands = [target] + distract
        gvec = gvec_fn(given)
        if gvec is None:
            continue
        scores = [(i, _cos(gvec, unit[w])) for i, w in enumerate(cands) if w in unit]
        if len(scores) < len(cands):
            continue
        scores.sort(key=lambda x: (-x[1], x[0]))
        gold_rank = next(i for i, (idx, _) in enumerate(scores) if idx == 0) + 1
        top1 += gold_rank == 1
        mrr += 1.0 / gold_rank
        n += 1
        if n >= N_TRIALS:
            break
    return {"trials": n,
            "top1": round(top1 / max(1, n), 4),
            "mrr": round(mrr / max(1, n), 4)}


def pair_metrics(unit, pairs: Dict[str, List[Tuple[str, str]]], slot_of) -> dict:
    """已知一个词，从 4 个同槽词里选留出句里与它同现的那个。"""
    def one(w):
        return unit.get(w)
    return {task: _trial_metrics(unit, pairs[task], slot_of, one, task)
            for task in ("consequence", "action")}


def triple_metrics(unit, trials, slot_of) -> dict:
    """已知 (实体,动作) 选后果、已知 (实体,后果) 选动作（题量少，只作旁证）。"""
    def mech2(a, b):
        if a not in unit or b not in unit:
            return None
        return _unit([x + y for x, y in zip(unit[a], unit[b])])
    return {
        "consequence": _trial_metrics(unit, [( (e, a), c) for e, a, c in trials],
                                      slot_of, lambda g: mech2(*g), "consequence"),
        "action": _trial_metrics(unit, [((e, c), a) for e, a, c in trials],
                                 slot_of, lambda g: mech2(*g), "action"),
    }


def main() -> None:
    words, slot_of, _ = load_glyphs(VAULT)
    lines = read_lines(CORPUS)
    cut = int(len(lines) * 0.9)
    train, held = lines[:cut], lines[cut:]
    pairs = collect_pairs(held, slot_of)
    trials = collect_triples(held, slot_of)
    print(f"语料 {len(lines)} 行，训练 {cut} 行，留出 {len(lines) - cut} 行；"
          f"E→C 对 {len(pairs['consequence'])}、E→A 对 {len(pairs['action'])}、"
          f"三元组 {len(trials)}")

    variants = {
        "baseline": lambda: (*scan_flat(train, words), 12),
        "k24": lambda: (*scan_flat(train, words), 24),
        "dist": lambda: (*scan_dist(train, words), 12),
    }
    result = {"meta": {"corpus_sha256": _sha256(CORPUS),
                       "vault_sha256": _sha256(VAULT),
                       "train_lines": cut, "heldout_lines": len(lines) - cut,
                       "pairs_available": {k: len(v) for k, v in pairs.items()},
                       "triples_available": len(trials),
                       "trial_seed": TRIAL_SEED, "window": LAMBDA_WINDOW},
              "variants": {}}
    for name, build in variants.items():
        pair, presence, n_sent, k = build()
        unit = vectors_from(pair, presence, n_sent, words, k)
        row = {"k": k, "pairs": len(pair), "sentences": n_sent}
        row["pair"] = pair_metrics(unit, pairs, slot_of)
        row["triple"] = triple_metrics(unit, trials, slot_of)
        result["variants"][name] = row
        print(name, row)

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    lines_md = ["# P21 表征评估（留出共现回填）", "",
                f"训练 {cut} 行 / 留出 {len(lines) - cut} 行；可用三元组 {len(trials)} 条；"
                f"每任务最多 {N_TRIALS} 题、{N_DISTRACTORS} 个同槽干扰项。", "",
                "| 变体 | k | 词→后果 top1 / MRR | 词→动作 top1 / MRR | 三元组 n |",
                "|---|---|---|---|---|"]
    for name, row in result["variants"].items():
        lines_md.append(
            f"| {name} | {row['k']} | {row['pair']['consequence']['top1']} / "
            f"{row['pair']['consequence']['mrr']} | {row['pair']['action']['top1']} / "
            f"{row['pair']['action']['mrr']} | {row['triple']['consequence']['trials']} |")
    lines_md += ["", "判读：top1 明显高于 0.25（随机）才说明向量学到了句内关系；"
                 "变体之间差 1 个百分点以内视为同水平，不为它换线上产物。"
                 "三元组题量少，只作旁证，不参与选型。", ""]
    OUT_MD.write_text("\n".join(lines_md), encoding="utf-8")
    print(f"-> {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
