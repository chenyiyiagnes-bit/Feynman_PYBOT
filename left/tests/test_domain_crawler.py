"""旁路扩展器回归（P6.3 + 九条优化）：默认关闭、只写扩展库、死亡标记、
小句抽取、限速重试、落盘缓存、provenance 回滚、日志端点。"""

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
    dc.CACHE_PATH = tmpdir / "crawl_cache.json"
    dc.MIN_INTERVAL_SECONDS = 0.0
    dc.clear_cache()


def _cleanup():
    os.environ.pop(dc.ENABLED_ENV, None)
    dc.MIN_INTERVAL_SECONDS = 0.25


class _Resp:
    def __init__(self, status=200, data=None, url="https://zh.wikipedia.org/x",
                 headers=None):
        self.status_code = status
        self._data = data or {}
        self.url = url
        self.headers = headers or {}

    def json(self):
        return self._data


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.responses.pop(0) if self.responses else _Resp(404)


# ---------------------------------------------------------------- 抽取


def test_harvest_edges_prefers_specific_verbs_and_caps():
    text = "科学家观察这只股票，并计算它的现金流，最后推动市场。"
    edges = dc.harvest_edges(text, ACTIONS)
    assert edges[:2] == ["观察", "计算"], edges
    assert len(edges) <= dc.MAX_EDGES_PER_ENTITY
    assert dc.harvest_edges("没有任何动作字粒的句子", ACTIONS) == []


def test_negation_and_condition_clauses_are_skipped():
    text = "如果观察不到，就无法计算。实验者观察了粒子，计算了轨迹。"
    edges = dc.harvest_edges(text, ACTIONS)
    assert "观察" in edges and "计算" in edges, edges
    only_negated = dc.harvest_edges("如果没有观察，也不会计算。", ACTIONS)
    assert only_negated == [], only_negated


def test_nominal_compounds_do_not_count_as_actions():
    assert dc.harvest_edges("观察者拿着计算器。", ACTIONS) == []
    assert dc.harvest_edges("这是一种观察法。", ACTIONS) == []


def test_entity_in_sentence_is_preferred_when_present():
    text = "电子在磁场里旋转。实验者观察了整个装置。"
    assert dc.harvest_edges(text, ACTIONS, entity="电子") == ["旋转"]
    # 实体一次都没出现时不搞全丢弃，退回所有小句
    assert dc.harvest_edges(text, ACTIONS, entity="量子退火") == ["旋转", "观察"]


def test_clean_summary_drops_only_dirty_clauses():
    text = "费曼逝世后学界哀悼。这种算法观察数据，计算误差。"
    cleaned, dropped = dc.clean_summary(text)
    assert dropped == 1
    assert "费曼" not in cleaned and "观察" in cleaned


# ---------------------------------------------------------------- 抓取写入


def test_disabled_by_default_writes_nothing(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ.pop(dc.ENABLED_ENV, None)
        r = dc.crawl_entity("股票", ACTIONS)
        assert r["status"] == "disabled"
        assert not de.EXT_PATH.exists()
    _cleanup()


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
        assert r["run_id"].startswith("run-")
        data, h = de.load_domain_extensions_with_hash()
        assert data["entity_prototypes"]["量子退火"] == "abstract_info"
        assert ["量子退火", "观察"] in data["cooccurrence_edges"]
        assert data["crawl_log"]["量子退火"]["source"] == "zh.wikipedia"
        assert h != de.EMPTY_EXT_HASH
        # crawl_log/runs/provenance 不进哈希：改日志与 provenance 不漂移
        data["crawl_log"]["量子退火"]["at"] = 0
        data["crawl_runs"][-1]["at"] = 0
        data["edge_provenance"]["量子退火|观察"].append("run-other")
        assert de.compute_canonical_ext_hash(data) == h
    _cleanup()
    if vault_digest is not None:
        assert hashlib.sha256(VAULT_FILE.read_bytes()).hexdigest() == vault_digest, \
            "旁路扩展绝不能改动原典字库"


def test_death_marker_summary_is_dropped_whole():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: "费曼逝世后学界哀悼。他去世后人们整理讲义。"
        r = dc.crawl_entity("费曼", ACTIONS)
        assert r["status"] == "blocked_death_marker", r
        data, _ = de.load_domain_extensions_with_hash()
        assert data.get("cooccurrence_edges") == []
        assert data.get("entity_prototypes") == {}
    _cleanup()


def test_partially_dirty_summary_keeps_clean_sentences():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: (
            "费曼逝世后学界哀悼。这种算法需要观察数据并计算误差。")
        r = dc.crawl_entity("量子退火", ACTIONS)
        assert r["status"] == "ok", r
        assert r["dropped_sentences"] == 1
        assert set(r["edges"]) >= {"观察", "计算"}
        data, _ = de.load_domain_extensions_with_hash()
        assert data["crawl_log"]["量子退火"]["dropped_sentences"] == 1
    _cleanup()


def test_missing_summary_and_404_are_reported_not_invented():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: None
        r = dc.crawl_entity("不存在的条目", ACTIONS)
        assert r["status"] == "no_summary" and r["edges"] == []
        data, h = de.load_domain_extensions_with_hash()
        assert data.get("cooccurrence_edges") == []
        assert h == de.EMPTY_EXT_HASH, "只记失败日志不应改变扩展库指纹"
    _cleanup()


def test_crawl_many_dedupes_and_caps():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        calls = []

        def fake(entity, client=None):
            calls.append(entity)
            return None

        dc.fetch_summary = fake
        results = dc.crawl_many(["股票", "股票", "债券", "股票"], ACTIONS)
        assert [r["entity"] for r in results] == ["股票", "债券"]
        assert calls == ["股票", "债券"]
    _cleanup()


# ---------------------------------------------------------------- HTTP / 缓存


def test_retry_once_on_503_and_respect_retry_after():
    client = _Client([
        _Resp(503, headers={"Retry-After": "0"}),
        _Resp(200, {"extract": "电子在磁场里旋转。"}),
    ])
    text, status = dc._summary_text("电子", client, dc.WIKI_HOST)
    assert status == "ok" and "旋转" in text
    assert len(client.calls) == 2


def test_redirect_outside_allowlist_is_rejected():
    client = _Client([
        _Resp(200, {"extract": "不该采信。"},
              url="https://evil.example/wiki/电子"),
    ])
    text, status = dc._summary_text("电子", client, dc.WIKI_HOST)
    assert text is None and status == "error"
    assert dc.allowed_host("zh.wikipedia.org")
    assert not dc.allowed_host("evil.example")


def test_disambiguation_falls_through_to_search():
    responses = [
        _Resp(200, {"type": "disambiguation"}),
        _Resp(200, {"pages": [{"title": "量子退火（物理学）"}]}),
        _Resp(200, {"extract": "量子退火利用量子涨落寻找最优解。"}),
    ]
    client = _Client(responses)
    text, status, host, url = dc.fetch_summary_with_status("量子退火", client=client)
    assert status == "ok" and "量子退火" in text
    assert host == dc.WIKI_HOST
    assert len(client.calls) == 3


def test_cache_persists_and_respects_status_ttl():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        assert dc._ttl("ok") == dc.CACHE_TTL_OK
        assert dc._ttl("not_found") == dc.CACHE_TTL_MISS
        assert dc._ttl("error") == dc.CACHE_TTL_TRANSIENT
        client = _Client([_Resp(200, {"extract": "电子会旋转。"})])
        text, status, _host, _url = dc.fetch_summary_with_status("电子", client=client)
        assert status == "ok" and text
        assert len(client.calls) == 1
        # 内存清空、从磁盘缓存恢复：不再联网
        dc._cache.clear()
        dc._cache_loaded = False
        client2 = _Client([_Resp(500)])
        text2, status2, _h, _u = dc.fetch_summary_with_status("电子", client=client2)
        assert status2 == "ok" and text2 == text
        assert client2.calls == []
        assert dc.CACHE_PATH.is_file()
    _cleanup()


# ---------------------------------------------------------------- 回滚 / 日志


def test_rollback_run_removes_only_its_own_edges():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: "算法观察数据并计算误差。"
        first = dc.crawl_entity("量子退火", ACTIONS)
        dc.fetch_summary = lambda entity, client=None: "算法观察数据。"
        second = dc.crawl_entity("量子退火", ACTIONS)
        assert first["status"] == second["status"] == "ok"

        result = dc.rollback_run(first["run_id"])
        assert result["ok"], result
        data, _ = de.load_domain_extensions_with_hash()
        edges = {tuple(e) for e in data["cooccurrence_edges"]}
        assert ("量子退火", "观察") in edges, "第二个 run 仍贡献这条边"
        assert ("量子退火", "计算") not in edges, "只由第一个 run 贡献的边应删掉"
        assert ["量子退火", "计算"] in result["removed_edges"]

        result2 = dc.rollback_run(second["run_id"])
        assert result2["ok"]
        data, _ = de.load_domain_extensions_with_hash()
        assert data["cooccurrence_edges"] == []
        assert data.get("crawl_log") == {}

        assert dc.rollback_run("run-nope")["reason"] == "no_such_run"
    _cleanup()


def test_crawl_stats_counts_and_recent_records():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: (
            "观察数据。" if entity == "股票" else None)
        dc.crawl_entity("股票", ACTIONS)
        dc.crawl_entity("债券", ACTIONS)
        stats = dc.crawl_stats()
        assert stats["total_runs"] == 2
        assert stats["status_counts"] == {"ok": 1, "no_summary": 1}
        assert stats["entities"] == 2
        assert stats["recent"][0]["entity"] == "债券"
        assert stats["edges"] == 1
    _cleanup()


def test_endpoint_forbidden_when_disabled_and_queues_when_enabled():
    from fastapi.testclient import TestClient
    import bus.main as bm
    from bus.main import app

    # 本仓库 checkout 不一定带 data/feynman_style_vault.json；端点要的是
    # 动作字粒集合，测试里直接给一份，别依赖本机字库。
    original_actions = bm._action_words
    bm._action_words = lambda: ACTIONS
    os.environ.pop(dc.ENABLED_ENV, None)
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        client = TestClient(app)
        assert client.post("/feynman/extend", json={"entities": ["股票"]}).status_code == 403
        assert client.get("/feynman/extend/log").status_code == 403

        os.environ[dc.ENABLED_ENV] = "1"
        dc.fetch_summary = lambda entity, client=None: "投资者观察股价，并计算收益。"
        ok = client.post("/feynman/extend", json={"entities": ["股票", "股票；rm"]})
        assert ok.status_code == 200, ok.text
        assert ok.json()["queued"] == ["股票"]
        data, _ = de.load_domain_extensions_with_hash()
        assert any(e[0] == "股票" for e in data["cooccurrence_edges"])

        log = client.get("/feynman/extend/log")
        assert log.status_code == 200, log.text
        payload = log.json()
        assert payload["status_counts"] == {"ok": 1}
        run_id = payload["recent"][0]["run_id"]

        rolled = client.post("/feynman/extend/rollback", json={"run_id": run_id})
        assert rolled.status_code == 200, rolled.text
        assert rolled.json()["ok"]
        assert client.post("/feynman/extend/rollback",
                           json={"run_id": run_id}).status_code == 404
    bm._action_words = original_actions
    _cleanup()


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
