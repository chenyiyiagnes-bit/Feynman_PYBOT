"""字库体检：四槽位计数、相位/层级/来源分布、逐字溯源与结构引用完整性。

只读，不修改任何数据。溯源按「来源串里是否有语料文件名」判定：
- 可回源：某一节段与语料 JSONL 的 source 文件名一致；
- 外部书：以「书｜」等前缀引用、语料里没有的书（P15 的非费曼补库），单独计数；
- 无源：既不在语料、也不是登记过的外部书格式。
结构检查覆盖重复字粒、相位越界、共现键的端点、纠缠对端点。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/audit_vault.py
    ... --vault data/feynman_style_vault_p15.json --json
    ... --strict     有结构性错误或无源字粒时非零退出
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOTS = ("hook", "entity", "action", "consequence")
SLOT_KEYS = {"hook": "hooks", "entity": "entities",
             "action": "actions", "consequence": "consequences"}
EXTERNAL_PREFIXES = ("书", "R14", "R15", "R16")  # 探矿脚本登记过的书源前缀

PHASE_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120.01)]


def _default_vault() -> Path:
    env = os.getenv("QBOT_FEYNMAN_VAULT")
    if env:
        p = Path(env)
        return p if p.is_absolute() else ROOT / p
    return ROOT / "data" / "feynman_style_vault_p15.json"


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _corpus_files(corpus: Path) -> set[str]:
    names: set[str] = set()
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = str(rec.get("source") or "")
            if src:
                names.add(os.path.basename(src.split("｜")[-1]).strip().lower())
    return names


def _source_state(source: str, corpus_files: set[str]) -> str:
    toks = [t.strip() for t in source.split("｜") if t.strip()]
    if any(os.path.basename(t).lower() in corpus_files for t in toks):
        return "resolved"
    if toks and toks[0] in EXTERNAL_PREFIXES:
        return "external"
    return "unresolved"


def _bucket(deg: float) -> str:
    for lo, hi in PHASE_BUCKETS:
        if lo <= deg < hi:
            return f"{int(lo)}–{int(min(hi, 120))}"
    return "越界"


def audit(vault_path: Path, corpus_override: Path | None = None) -> dict:
    raw = json.loads(vault_path.read_text(encoding="utf-8"))
    bank_id = "feynman-" + hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
    corpus = _resolve(corpus_override or raw.get("corpus")
                      or "data/p8_corpus_v2.jsonl")
    corpus_files = _corpus_files(corpus) if corpus.is_file() else set()

    report: dict = {
        "vault": str(vault_path),
        "bank_id": bank_id,
        "corpus": str(corpus),
        "corpus_found": corpus.is_file(),
        "corpus_files": len(corpus_files),
        "meta": {"sections": raw.get("sections"), "sentences": raw.get("sentences")},
        "slots": {},
        "errors": [],
        "warnings": [],
        "external_sources": [],
    }

    all_words: dict[str, set[str]] = {}
    dup_words: dict[str, list[str]] = {}
    src_counts = Counter({"resolved": 0, "external": 0, "unresolved": 0})
    unresolved_examples: list[str] = []
    external_examples: list[str] = []
    glyphs_without_source: list[str] = []
    glyphs_external_only: list[str] = []

    for slot in SLOTS:
        glyphs = raw.get(SLOT_KEYS[slot]) or []
        all_words[slot] = set()
        words = Counter(g.get("word") or "" for g in glyphs)
        dup_words[slot] = sorted(w for w, n in words.items() if n > 1 and w)
        phases, tiers, curated, domains, styles = Counter(), Counter(), Counter(), Counter(), Counter()
        orders, valencies = Counter(), Counter()
        no_source = 0
        for g in glyphs:
            w = g.get("word") or ""
            if not w:
                report["errors"].append(f"{slot}: 空字粒")
            if g.get("slot") and g["slot"] != slot:
                report["errors"].append(f"{slot}: 字粒 {w!r} slot 标为 {g['slot']!r}")
            try:
                deg = float(g.get("phase_deg"))
            except (TypeError, ValueError):
                report["errors"].append(f"{slot}: 字粒 {w!r} 相位缺失")
                deg = None
            if deg is not None:
                phases[_bucket(deg)] += 1
                if not (0.0 <= deg <= 120.0):
                    report["errors"].append(f"{slot}: 字粒 {w!r} 相位 {deg} 越界")
            try:
                if float(g.get("count", 0)) <= 0:
                    report["errors"].append(f"{slot}: 字粒 {w!r} 计数非正")
            except (TypeError, ValueError):
                report["errors"].append(f"{slot}: 字粒 {w!r} 计数非数值")
            if "tier" in g:
                tiers[str(g.get("tier"))] += 1
            if "curated" in g:
                curated["是" if g.get("curated") else "否"] += 1
            domains[str(g.get("domain") or "?")] += 1
            styles[str(g.get("style") or "?")] += 1
            if slot == "action":
                orders[str(g.get("order") or "?")] += 1
                valencies[str(g.get("valency") or "?")] += 1
            sources = [s for s in (g.get("sources") or []) if isinstance(s, str)]
            states = [_source_state(s, corpus_files) for s in sources]
            src_counts.update(states)
            for s, st in zip(sources, states):
                if st == "unresolved" and len(unresolved_examples) < 10:
                    unresolved_examples.append(f"{slot}/{w}: {s[:80]}")
                if st == "external" and len(external_examples) < 10:
                    external_examples.append(f"{slot}/{w}: {s[:80]}")
            if sources and "resolved" not in states:
                glyphs_without_source.append(f"{slot}/{w}")
                if all(st == "external" for st in states):
                    glyphs_external_only.append(f"{slot}/{w}")
            if not sources:
                no_source += 1
                glyphs_without_source.append(f"{slot}/{w}（无 sources）")
            all_words[slot].add(w)

        report["slots"][slot] = {
            "glyphs": len(glyphs),
            "duplicates": dup_words[slot],
            "no_sources": no_source,
            "phase": dict(sorted(phases.items())),
            "tier": dict(tiers),
            "curated": dict(curated),
            "domain": dict(domains.most_common()),
            "style": dict(styles.most_common()),
            "order": dict(orders),
            "valency": dict(valencies),
        }

    for slot in SLOTS:
        if dup_words[slot]:
            report["errors"].append(f"{slot}: 重复字粒 {dup_words[slot][:10]}")

    # 结构引用：共现键两端、纠缠对两端
    known = set().union(*all_words.values())
    cooc = raw.get("cooccurrence") or {}
    broken_cooc = []
    for key in cooc:
        a, _, b = str(key).partition("|")
        if not a or not b or a not in known or b not in known:
            broken_cooc.append(str(key))
    if broken_cooc:
        report["errors"].append(
            f"共现键端点缺失 {len(broken_cooc)} 条，例：{broken_cooc[:5]}")
    ent = raw.get("entangled_pairs") or []
    ent_words = all_words["entity"]
    cons_words = all_words["consequence"]
    unusable_ent = [p for p in ent
                    if not (p.get("everyday") in ent_words
                            and p.get("physics") in cons_words)]
    if unusable_ent:
        # 打分配对的键是 (实体, 后果)；端点不在槽位或不在对应槽位的配对永远不会命中。
        # p15 及更早的字库是从分句候选名词生成的对，80 对里 0 对可用——
        # 新一代构建已改为从 E→C 共现边生成，旧库不追改，只报警告。
        report["warnings"].append(
            f"纠缠对不可用于打分 {len(unusable_ent)}/{len(ent)} 条"
            f"（须为 实体×后果 槽位内配对），例：{unusable_ent[:3]}")

    report["sources"] = dict(src_counts)
    report["unresolved_examples"] = unresolved_examples
    report["external_examples"] = external_examples
    report["glyphs_without_resolvable_source"] = glyphs_without_source
    report["glyphs_external_only"] = glyphs_external_only
    report["cooccurrence_pairs"] = len(cooc)
    report["entangled_pairs"] = len(ent)
    return report


def render(r: dict) -> str:
    L = [
        f"字库：{r['vault']}",
        f"bank_id：{r['bank_id']}",
        f"语料：{r['corpus']}"
        f"（{'已找到' if r['corpus_found'] else '缺失'}，{r['corpus_files']} 个来源文件）",
        f"节 {r['meta']['sections']}；句 {r['meta']['sentences']}",
        "",
        "| 槽位 | 字粒 | 重复 | 无源 | 相位分布 | tier | curated |",
        "|---|---|---|---|---|---|---|",
    ]
    for slot in SLOTS:
        s = r["slots"][slot]
        L.append(f"| {slot} | {s['glyphs']} | {len(s['duplicates'])} | "
                 f"{s['no_sources']} | {s['phase']} | {s['tier'] or '—'} | "
                 f"{s['curated'] or '—'} |")
    for slot in SLOTS:
        s = r["slots"][slot]
        L.append("")
        L.append(f"{slot}：域 {s['domain']}；文体 {s['style']}"
                 + (f"；order {s['order']}；valency {s['valency']}"
                    if slot == "action" else ""))
    L += [
        "",
        f"来源串：可回源 {r['sources'].get('resolved', 0)}，"
        f"外部书 {r['sources'].get('external', 0)}，"
        f"无源 {r['sources'].get('unresolved', 0)}",
        f"共现对 {r['cooccurrence_pairs']}；纠缠对 {r['entangled_pairs']}",
    ]
    if r["external_examples"]:
        L.append("外部书示例：" + "；".join(r["external_examples"][:5]))
    if r["unresolved_examples"]:
        L.append("无源示例：" + "；".join(r["unresolved_examples"]))
    if r["glyphs_without_resolvable_source"]:
        L.append(f"无可回源字粒 {len(r['glyphs_without_resolvable_source'])} 条"
                 f"（其中仅外部书 {len(r['glyphs_external_only'])} 条）：" +
                 "、".join(r["glyphs_without_resolvable_source"][:20]))
    if r["errors"]:
        L.append("")
        L.append(f"结构错误 {len(r['errors'])} 条：")
        L += ["- " + e for e in r["errors"][:20]]
    else:
        L.append("结构检查：通过")
    if r["warnings"]:
        L.append("")
        L.append(f"警告 {len(r['warnings'])} 条：")
        L += ["- " + w for w in r["warnings"]]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", type=Path, default=_default_vault())
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--strict", action="store_true",
                    help="有结构错误、无源串或无任何可回源字粒时非零退出")
    args = ap.parse_args()
    vault = _resolve(args.vault)
    if not vault.is_file():
        raise SystemExit(f"字库不存在：{vault}")
    report = audit(vault, args.corpus)
    print(json.dumps(report, ensure_ascii=False, indent=2)
          if args.json else render(report))
    if args.strict:
        no_src = sum(s["no_sources"] for s in report["slots"].values())
        bad = report["errors"] or report["sources"].get("unresolved", 0) or no_src
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
