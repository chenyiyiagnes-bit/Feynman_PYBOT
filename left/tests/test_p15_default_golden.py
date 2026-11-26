"""产品默认档黄金回归：P15 字库 + 四检索开关 + 拓扑层，25 问。

只锁"没人打算改却变了"的排版漂移；检索开关、字库、拓扑图的任何一处
回归都会让首句或前三句变化，测试直接指出是哪一问。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_p15_default_golden.py
"""

import json
import os
from pathlib import Path

os.environ["QBOT_FEYNMAN_VAULT"] = "data/feynman_style_vault_p15.json"
os.environ["QBOT_FEYNMAN_TOPOLOGY"] = "data/topology_graph_p15.npz"
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.topology_builder import get_engine

FIXTURE = Path("tests/fixtures_feynman_p15_default.json")


def _run(vault, engine, query):
    req = PrintRequest(style_deg=60.0, nonce=0, user_query=query,
                       topic_lock=True, retrieve_consequence=True,
                       retrieval_gates=True, specific_pref=True)
    return run_feynman_print(vault, req, topology=engine)


def test_default_config_matches_golden():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    vault = Vault(Path(data["vault"]))
    engine = get_engine(force=True)
    assert engine is not None and engine.topo_hash == data["topology_hash"]
    for query, exp in data["cases"].items():
        out = _run(vault, engine, query)
        assert out["printed_collapse"] == exp["line"], query
        got3 = [p.get("line") for p in (out.get("paths") or [])[:3]]
        assert got3 == exp["top3"], (query, got3, exp["top3"])


def test_topology_layer_actually_engaged():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    vault = Vault(Path(data["vault"]))
    engine = get_engine(force=True)
    query = next(iter(data["cases"]))
    out = _run(vault, engine, query)
    assert out.get("topology_hash") == data["topology_hash"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P15 默认档黄金回归 {len(tests)} 项全部通过")
