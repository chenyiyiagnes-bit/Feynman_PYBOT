"""旁路扩展器回归（P6.3）：默认关闭、只写扩展库、死亡标记拦截、原型覆盖生效。"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import plugins.domain_expander as de
import plugins.domain_crawler as dc
from plugins.feynman_typesetter import (
    PHYSICAL_ONLY_ACTIONS, PrintRequest, Vault, resolve_entity_prototype, run_feynman_print,
)

VAULT_FILE = Path("data/feynman_style_vault.json")
HAS_REAL = VAULT_FILE.is_file()
ACTIONS = {"观察", "计算", "拆解", "旋转", "推动"}


def _setup(tmpdir: Path):
    de.EXT_PATH = tmpdir / "domain_extensions.json"
    dc._cache.clear()


def test_harvest_edges_prefers_specific_verbs_and_caps():
    text = "科学家观察这只股票，并计算它的现金流，最后推动市场。"
    edges = dc.harvest_edges(text, ACTIONS)
    assert edges[:2] == ["观察", "计算"], edges
    assert len(edges) <= dc.MAX_EDGES_PER_ENTITY
    assert dc.harvest_edges("没有任何动作字粒的句子", ACTIONS) == []


def test_disabled_by_default_writes_nothing(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ.pop(dc.ENABLED_ENV, None)
        r = dc.crawl_entity("股票", ACTIONS)
        assert r["status"] == "disabled"
        assert not de.EXT_PATH.exists()


def test_crawl_writes_only_extension_library_and_hash_excludes_log():
    vault_digest = None
    if HAS_REAL:
        vault_digest = hashlib.sha256(VAULT_FILE.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: "这种算法需要观察数据，再计算误差，最后记住结论。"
        r = dc.crawl_entity("量子退火", ACTIONS)
        assert r["status"] == "ok", r
        assert "观察" in r["edges"]
        assert r["prototype"] == "abstract_info", "认知动词上下文应标抽象原型"
        data, h = de.load_domain_extensions_with_hash()
        assert data["entity_prototypes"]["量子退火"] == "abstract_info"
        assert ["量子退火", "观察"] in data["cooccurrence_edges"]
        assert data["crawl_log"]["量子退火"]["source"] == "zh.wikipedia"
        assert h != de.EMPTY_EXT_HASH
        # crawl_log 不进哈希：改日志不漂移
        data["crawl_log"]["量子退火"]["at"] = 0
        assert de.compute_canonical_ext_hash(data) == h
    os.environ.pop(dc.ENABLED_ENV, None)
    if vault_digest is not None:
        assert hashlib.sha256(VAULT_FILE.read_bytes()).hexdigest() == vault_digest, \
            "旁路扩展绝不能改动原典字库"


def test_death_marker_summary_is_dropped_whole():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: "费曼逝世后，人们观察到他留下的讲义。"
        r = dc.crawl_entity("费曼", ACTIONS)
        assert r["status"] == "blocked_death_marker"
        assert not de.EXT_PATH.exists()
    os.environ.pop(dc.ENABLED_ENV, None)


def test_missing_summary_and_404_are_reported_not_invented():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: None
        r = dc.crawl_entity("不存在的条目", ACTIONS)
        assert r["status"] == "no_summary" and r["edges"] == []
        assert not de.EXT_PATH.exists()
    os.environ.pop(dc.ENABLED_ENV, None)


def test_endpoint_forbidden_when_disabled_and_queues_when_enabled():
    from fastapi.testclient import TestClient
    from bus.main import app

    os.environ.pop(dc.ENABLED_ENV, None)
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        client = TestClient(app)
        assert client.post("/feynman/extend", json={"entities": ["股票"]}).status_code == 403

        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: "投资者观察股价，并计算收益。"
        ok = client.post("/feynman/extend", json={"entities": ["股票", "股票；rm"]})
        assert ok.status_code == 200, ok.text
        assert ok.json()["queued"] == ["股票"]
        data, _ = de.load_domain_extensions_with_hash()
        assert any(e[0] == "股票" for e in data["cooccurrence_edges"])
    os.environ.pop(dc.ENABLED_ENV, None)


def test_ext_prototype_override_changes_ranked_verb():
    if not HAS_REAL:
        return
    v = Vault()
    fam, _ = resolve_entity_prototype("光合作用", {"光合作用": "abstract_info"})
    assert fam == "abstract_info"
    req_entities = "光合作用是怎么回事"
    bad = 0
    for style in (30, 60, 90):
        for nonce in range(3):
            out = run_feynman_print(
                v, PrintRequest(style_deg=style, nonce=nonce, user_query=req_entities),
                prototypes={"光合作用": "abstract_info"})
            if out.get("collapsed") and out["paths"][0]["glyphs"]["action"] in PHYSICAL_ONLY_ACTIONS:
                bad += 1
    assert bad == 0, f"扩展库原型覆盖后仍有 {bad} 条物理动作登顶"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 旁路扩展器 {len(tests)} 项回归全部通过")
