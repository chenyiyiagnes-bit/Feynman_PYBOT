"""成长年轮与真实会话重置。

锁住 P17 的验收点：
1. 含 DEATH_MARKERS 的字粒/搭配/会话痕迹整条过滤；
2. 空扩展库输出占位文案，不抛异常；
3. 同样输入得到逐字节相同的 Markdown，ext_hash 两次调用一致；
4. 拓扑引擎没开时如实写"未启用"；
5. `/feynman/session/reset` 真的清掉 `_sessions` 与隐式信号上一轮快照。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_growth_journal.py
"""

import hashlib
import json
import os
import tempfile

TMP = tempfile.mkdtemp()
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_EXT"] = os.path.join(TMP, "domain_extensions.json")

from fastapi.testclient import TestClient

import bus.main as bm
from plugins.domain_expander import atomic_save_extensions, compute_canonical_ext_hash
from plugins.growth_journal import (EMPTY_PLACEHOLDER, build_journal,
                                    render_markdown, session_traces)

VAULT_STATS = {"bank_id": "feynman-test", "corpus": "demo", "sections": 3,
               "sentences": 9, "glyphs": {"hook": 1, "entity": 2,
                                          "action": 3, "consequence": 4}}


def _journal(ext, hash_=None, topo=None, sessions=()):
    return build_journal(VAULT_STATS, ext, hash_ or compute_canonical_ext_hash(ext),
                         topo, sessions)


def test_empty_extensions_placeholder():
    md = render_markdown(_journal({}))
    assert EMPTY_PLACEHOLDER in md, md
    assert "00000000" in md
    assert "未启用" in md, md
    assert "feynman-test" in md and "entity 2" in md


def test_death_markers_filtered_everywhere():
    ext = {
        "edge_scores": {"去世|振动": 1.30, "弹簧|振动": 1.30, "追悼|沉淀": 0.40},
        "skeleton_scores": {"悼念骨架": 1.20, "转折骨架": 1.20},
        "entity_prototypes": {"墓碑": "fluid", "弹簧": "mechanical"},
    }
    sessions = [
        {"session_id": "追悼会-1", "turns": 2, "amplitude": 1.0, "phase_deg": 60.0,
         "last_entities": ["弹簧"]},
        {"session_id": "s-ok", "turns": 2, "amplitude": 1.0, "phase_deg": 60.0,
         "last_entities": ["墓碑"]},
        {"session_id": "s-ok2", "turns": 2, "amplitude": 1.0, "phase_deg": 60.0,
         "last_entities": ["弹簧"]},
    ]
    j = _journal(ext, sessions=sessions)
    md = render_markdown(j)
    for bad in ("去世", "追悼", "悼念", "墓碑"):
        assert bad not in md, (bad, md)
    assert "弹簧" in md and "转折骨架" in md and "mechanical" in md
    assert [e["session_ref"] for e in j["sessions"]] == [
        hashlib.sha256(b"s-ok2").hexdigest()[:8]]
    assert "s-ok2" not in md, "原始 session_id 不应出现在年轮里"


def test_deterministic_markdown_and_hash():
    ext = {"edge_scores": {"弹簧|振动": 1.25, "光线|穿过": 0.80}}
    a = _journal(ext)
    b = _journal(ext)
    assert a["ext_hash"] == b["ext_hash"] == compute_canonical_ext_hash(ext)
    assert render_markdown(a) == render_markdown(b)
    # 排序只按偏离中性的幅度，同分按字母序，不依赖字典插入顺序
    assert [e["entity"] for e in a["learned_edges"]] == ["弹簧", "光线"]


def test_topology_stats_passthrough():
    topo = {"enabled": True, "topo_hash": "48cdc9f6", "words": 657,
            "triangles": 343, "edge_support_ge2": 1314, "analogy_pairs": 172}
    md = render_markdown(_journal({}, topo=topo))
    assert "48cdc9f6" in md and "已建复形 343" in md and "未启用" not in md


def test_session_traces_ignore_empty_and_dead():
    rows = session_traces([
        {"session_id": "", "turns": 1},
        {"session_id": "zero", "turns": 0},
        {"session_id": "追悼", "turns": 1},
        {"session_id": "live", "turns": 3, "amplitude": 1.8, "phase_deg": 61.5,
         "last_entities": ["弹簧"]},
    ])
    assert [r["session_ref"] for r in rows] == [
        hashlib.sha256(b"live").hexdigest()[:8]], rows


def test_http_journal_and_session_reset():
    ext = {"edge_scores": {"弹簧|振动": 1.3}}
    atomic_save_extensions(dict(ext))
    client = TestClient(bm.app)

    first = client.get("/feynman/journal")
    second = client.get("/feynman/journal")
    assert first.status_code == second.status_code == 200, first.text
    h1 = first.json()["journal"]["ext_hash"]
    h2 = second.json()["journal"]["ext_hash"]
    assert h1 == h2 == compute_canonical_ext_hash(ext) != "00000000"
    assert first.json()["markdown"] == second.json()["markdown"]
    assert "弹簧" in first.json()["markdown"]

    # 先制造一个真实会话，再确认 /reset 把它清掉
    body = {"user_query": "弹簧为什么会振动", "style_deg": 60.0,
            "nonce": 3, "session_id": "journal-reset-demo"}
    assert client.post("/feynman/paragraph", json=body).status_code == 200
    active = client.get("/feynman/journal").json()["journal"]["sessions"]
    assert any(s["turns"] == 1 for s in active), active

    r = client.post("/feynman/session/reset", json={"session_id": "journal-reset-demo"})
    assert r.status_code == 200, r.text
    assert r.json()["cleared"] is True and r.json()["implicit_turn"] is True, r.json()
    assert client.get("/feynman/journal").json()["journal"]["sessions"] == []

    again = client.post("/feynman/session/reset", json={"session_id": "journal-reset-demo"})
    assert again.json()["cleared"] is False
    assert client.post("/feynman/session/reset", json={"session_id": " "}).status_code == 400


def main():
    test_empty_extensions_placeholder()
    test_death_markers_filtered_everywhere()
    test_deterministic_markdown_and_hash()
    test_topology_stats_passthrough()
    test_session_traces_ignore_empty_and_dead()
    test_http_journal_and_session_reset()
    print("✅ test_growth_journal: 6/6")


if __name__ == "__main__":
    main()
