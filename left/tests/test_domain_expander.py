"""扩展库 ext_hash 与原子写入回归（P6.5）。"""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import plugins.domain_expander as de
from plugins.feynman_typesetter import PrintRequest, Vault, _seed, run_feynman_print

REAL_VAULT = Path("data/feynman_style_vault.json")
HAS_REAL = REAL_VAULT.is_file()


def _tmp_ext(tmpdir: Path):
    de.EXT_PATH = tmpdir / "domain_extensions.json"
    return de.EXT_PATH


def test_empty_extension_hash_is_fixed_and_empty():
    assert de.compute_canonical_ext_hash({}) == de.EMPTY_EXT_HASH
    assert de.compute_canonical_ext_hash(
        {"entity_prototypes": {}, "cooccurrence_edges": [], "edge_scores": {}}) == de.EMPTY_EXT_HASH
    with tempfile.TemporaryDirectory() as d:
        _tmp_ext(Path(d))
        data, h = de.load_domain_extensions_with_hash()
        assert data["entity_prototypes"] == {} and h == de.EMPTY_EXT_HASH


def test_hash_is_order_and_duplicate_invariant():
    d1 = {"entity_prototypes": {"超流体": "fluid", "期权": "abstract_finance"},
          "cooccurrence_edges": [["期权", "观察"], ["超流体", "流动"]]}
    d2 = {"entity_prototypes": {"期权": "abstract_finance", "超流体": "fluid"},
          "cooccurrence_edges": [["超流体", "流动"], ["期权", "观察"], ["期权", "观察"]]}
    assert de.compute_canonical_ext_hash(d1) == de.compute_canonical_ext_hash(d2)


def test_hash_moves_on_edges_scores_and_skeleton_weights():
    base = {"entity_prototypes": {"期权": "abstract_finance"},
            "cooccurrence_edges": [["期权", "观察"]],
            "edge_scores": {"期权|观察": 1.0},
            "skeleton_scores": {"retro": 1.0}}
    h = de.compute_canonical_ext_hash(base)
    for field, mutated in (
        ("entity_prototypes", {**base, "entity_prototypes": {"期权": "abstract_info"}}),
        ("cooccurrence_edges", {**base, "cooccurrence_edges": [["期权", "观察"], ["期权", "拆解"]]}),
        ("edge_scores", {**base, "edge_scores": {"期权|观察": 1.15}}),
        ("skeleton_scores", {**base, "skeleton_scores": {"retro": 1.2}}),
    ):
        assert de.compute_canonical_ext_hash(mutated) != h, f"{field} 未参与哈希"


def test_atomic_save_leaves_no_temp_and_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_ext(Path(d))
        h = de.atomic_save_extensions(
            {"entity_prototypes": {"期权": "abstract_finance"},
             "cooccurrence_edges": [["期权", "观察"]]})
        assert path.is_file()
        assert not list(path.parent.glob(".ext_tmp_*"))
        data, h2 = de.load_domain_extensions_with_hash()
        assert h2 == h and data["entity_prototypes"]["期权"] == "abstract_finance"
        stored = json.loads(path.read_text())
        assert stored["ext_hash"] == h, "文件内快照指纹必须与返回值一致"


def test_concurrent_updates_do_not_lose_writes():
    with tempfile.TemporaryDirectory() as d:
        _tmp_ext(Path(d))
        de.atomic_save_extensions({"edge_scores": {"x|y": 1.0}, "cooccurrence_edges": []})

        def bump():
            def mutate(ext):
                ext.setdefault("edge_scores", {})["x|y"] = round(
                    ext["edge_scores"].get("x|y", 1.0) + 0.1, 3)
            de.update_extensions(mutate)

        threads = [threading.Thread(target=bump) for _ in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        data, _ = de.load_domain_extensions_with_hash()
        assert data["edge_scores"]["x|y"] == 3.0


def test_seed_contract_empty_hash_is_byte_identical():
    old = hashlib.sha256(b"bank|60.0|3").hexdigest()[:16]
    assert _seed("bank", 60.0, 3) == old
    assert _seed("bank", 60.0, 3, "", de.EMPTY_EXT_HASH) == old
    assert _seed("bank", 60.0, 3, "sess") == hashlib.sha256(b"bank|60.0|3|sess").hexdigest()[:16]
    assert _seed("bank", 60.0, 3, "", "abcd1234") != old


def test_print_snapshot_changes_output_but_stays_deterministic():
    if not HAS_REAL:
        return
    v = Vault()
    a = run_feynman_print(v, PrintRequest(style_deg=55, nonce=2))
    b = run_feynman_print(v, PrintRequest(style_deg=55, nonce=2), ext_hash=de.EMPTY_EXT_HASH)
    c = run_feynman_print(v, PrintRequest(style_deg=55, nonce=2), ext_hash="abcd1234")
    d = run_feynman_print(v, PrintRequest(style_deg=55, nonce=2), ext_hash="abcd1234")
    assert a["seed"] == b["seed"]
    assert c["seed"] == d["seed"] != a["seed"]
    assert a["ext_hash"] == de.EMPTY_EXT_HASH and c["ext_hash"] == "abcd1234"


def test_endpoint_exposes_ext_hash_without_creating_file():
    from fastapi.testclient import TestClient
    from bus.main import app

    before = de.EXT_PATH.exists()
    resp = TestClient(app).post("/feynman/print", json={"style_deg": 60, "nonce": 0})
    assert resp.status_code == 200, resp.text
    assert resp.json()["ext_hash"] == de.EMPTY_EXT_HASH
    assert de.EXT_PATH.exists() == before, "读请求不得创建扩展库"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 扩展库与 ext_hash {len(tests)} 项回归全部通过")
