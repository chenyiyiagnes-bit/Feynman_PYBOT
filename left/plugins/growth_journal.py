"""成长年轮：把系统当前学到的痕迹导出成一份可读、可审计的 Markdown。

只读，不写任何库，也不生成新话术：

- 字库数字来自 `Vault.stats()`（原典只读，不参与进化）；
- 学到的搭配/骨架分来自扩展库，按"偏离中性 1.0 的程度"排序；
- 拓扑数字来自 `get_engine()`，引擎没开就如实写"未启用"；
- 会话只泄漏脱敏后的短哈希，不导出原始 session_id，也不导出任何原话。

含 `DEATH_MARKERS` 的字粒、搭配或会话痕迹一律过滤：年轮是给人看的，
但边界仍与字库同一套判据。输入相同则输出逐字节相同（无时间戳）。
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional

from plugins.feynman_vault import DEATH_MARKERS

JOURNAL_VERSION = 1
MAX_EDGES = 20
MAX_SESSIONS = 10
EMPTY_PLACEHOLDER = "还没有学到任何搭配：扩展库为空，全部输出仍来自费曼原典。"


def _dirty(text: object) -> bool:
    s = str(text or "")
    return any(m in s for m in DEATH_MARKERS)


def _clean(text: object) -> bool:
    return bool(str(text or "").strip()) and not _dirty(text)


def _session_ref(session_id: str) -> str:
    """会话只留 8 位短哈希；同一 id 每次得到同一个值，便于对照但不泄露原 id。"""
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:8]


def learned_edges(ext_data: Dict, limit: int = MAX_EDGES) -> List[dict]:
    """经验乘子；按 |score-1| 从大到小排，同分按字母序，保证确定性。"""
    rows = []
    for key, score in (ext_data.get("edge_scores") or {}).items():
        if "|" not in str(key):
            continue
        entity, action = str(key).split("|", 1)
        if not (_clean(entity) and _clean(action)):
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        rows.append({"entity": entity, "action": action,
                     "score": round(value, 3)})
    rows.sort(key=lambda r: (-abs(r["score"] - 1.0), r["entity"], r["action"]))
    return rows[:limit]


def skeleton_preferences(ext_data: Dict) -> List[dict]:
    rows = []
    for name, score in (ext_data.get("skeleton_scores") or {}).items():
        if not _clean(name):
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        rows.append({"pattern_type": str(name), "score": round(value, 3)})
    rows.sort(key=lambda r: (-abs(r["score"] - 1.0), r["pattern_type"]))
    return rows


def learned_prototypes(ext_data: Dict) -> List[dict]:
    rows = []
    for entity, proto in (ext_data.get("entity_prototypes") or {}).items():
        if not _clean(entity):
            continue
        rows.append({"entity": str(entity), "prototype": str(proto)})
    rows.sort(key=lambda r: r["entity"])
    return rows


def session_traces(sessions: Iterable[Dict], limit: int = MAX_SESSIONS) -> List[dict]:
    """会话只留轮数/振幅/相位/上轮实体；含去世词的整条记录丢弃。"""
    rows = []
    for rec in sessions or ():
        sid = str(rec.get("session_id") or "")
        entities = [str(e) for e in (rec.get("last_entities") or []) if _clean(e)]
        if not sid or _dirty(sid) or any(_dirty(e) for e in (rec.get("last_entities") or [])):
            continue
        try:
            turns = int(rec.get("turns") or 0)
        except (TypeError, ValueError):
            turns = 0
        if turns <= 0:
            continue
        try:
            amplitude = round(float(rec.get("amplitude") or 0.0), 3)
        except (TypeError, ValueError):
            amplitude = 0.0
        try:
            phase = round(float(rec.get("phase_deg") or 0.0), 1)
        except (TypeError, ValueError):
            phase = 0.0
        rows.append({"session_ref": _session_ref(sid), "turns": turns,
                     "amplitude": amplitude, "phase_deg": phase,
                     "last_entities": entities})
    rows.sort(key=lambda r: r["session_ref"])
    return rows[:limit]


def build_journal(vault_stats: Dict, ext_data: Dict, ext_hash: str,
                  topo_stats: Optional[Dict] = None,
                  sessions: Iterable[Dict] = ()) -> Dict:
    """纯函数：同样的输入必然得到同样的字典。"""
    edges = learned_edges(ext_data)
    skeletons = skeleton_preferences(ext_data)
    prototypes = learned_prototypes(ext_data)
    traces = session_traces(sessions)
    return {
        "version": JOURNAL_VERSION,
        "ext_hash": ext_hash or "00000000",
        "vault": {
            "bank_id": vault_stats.get("bank_id"),
            "corpus": vault_stats.get("corpus"),
            "sections": vault_stats.get("sections"),
            "sentences": vault_stats.get("sentences"),
            "glyphs": dict(vault_stats.get("glyphs") or {}),
        },
        "learned_edges": edges,
        "skeleton_preferences": skeletons,
        "learned_prototypes": prototypes,
        "topology": dict(topo_stats) if topo_stats else {"enabled": False},
        "sessions": traces,
    }


def render_markdown(journal: Dict) -> str:
    v = journal.get("vault") or {}
    glyphs = v.get("glyphs") or {}
    glyph_line = " / ".join(f"{k} {glyphs[k]}" for k in
                            ("hook", "entity", "action", "consequence") if k in glyphs)
    lines = [
        "# 费曼心智成长年轮",
        "",
        f"- 年轮版本：{journal.get('version', JOURNAL_VERSION)}",
        f"- 扩展库指纹：`{journal.get('ext_hash', '00000000')}`",
        "",
        "## 一、字库（只读原典，不参与进化）",
        "",
        f"- bank_id：`{v.get('bank_id') or '未加载'}`",
        f"- 语料：{v.get('corpus') or '—'}｜段落 {v.get('sections') or 0}｜"
        f"句子 {v.get('sentences') or 0}",
        f"- 四槽字粒：{glyph_line or '—'}",
        "",
        "## 二、学到的搭配（扩展库经验乘子）",
        "",
    ]
    edges = journal.get("learned_edges") or []
    if edges:
        lines += [f"- `{e['entity']}` × `{e['action']}` → **{e['score']:.3f}**"
                  for e in edges]
    else:
        lines.append(f"_{EMPTY_PLACEHOLDER}_")

    lines += ["", "## 三、骨架偏好", ""]
    skeletons = journal.get("skeleton_preferences") or []
    lines += [f"- `{s['pattern_type']}` → **{s['score']:.3f}**" for s in skeletons] or ["_无_"]

    lines += ["", "## 四、自建实体原型", ""]
    protos = journal.get("learned_prototypes") or []
    lines += [f"- `{p['entity']}` → {p['prototype']}" for p in protos] or ["_无_"]

    lines += ["", "## 五、拓扑层", ""]
    topo = journal.get("topology") or {}
    if topo.get("enabled") is False:
        lines.append("_未启用（FEYNMAN_ENABLE_TOPOLOGY=0 或拓扑图缺失）_")
    else:
        lines += [
            f"- topo_hash：`{topo.get('topo_hash') or '—'}`",
            f"- 词 {topo.get('words', 0)}｜已建复形 {topo.get('triangles', 0)}｜"
            f"支撑≥2 的边 {topo.get('edge_support_ge2', 0)}｜类比对 {topo.get('analogy_pairs', 0)}",
        ]

    lines += ["", "## 六、会话痕迹（脱敏，不导出原话）", ""]
    traces = journal.get("sessions") or []
    if traces:
        lines += [
            f"- `{t['session_ref']}`：{t['turns']} 轮｜振幅 {t['amplitude']}｜"
            f"相位 {t['phase_deg']}°｜上轮实体 {'、'.join(t['last_entities']) or '—'}"
            for t in traces
        ]
    else:
        lines.append("_当前没有活跃会话。_")
    return "\n".join(lines).rstrip() + "\n"
