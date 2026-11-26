"""扩展库与字库的溯源绑定：学出来的边必须标明是在哪个字库上学的。

只做标记与曝光，不参与打分：换字库后 /stats.ext.matches_vault 给出 None/True/False，
旧库没有标记返回 None（不知道来源），不会因此丢弃任何经验。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_ext_vault_binding.py
"""

import json
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="ext_vault_"))
os.environ["QBOT_FEYNMAN_EXT"] = str(_TMP / "domain_extensions.json")
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"

from fastapi.testclient import TestClient

import bus.main as bm
import plugins.domain_expander as de
from plugins.evolution_loop import apply_evolution_update

client = TestClient(bm.app)


def test_update_stamps_vault_id():
    de.EXT_PATH.unlink(missing_ok=True)
    apply_evolution_update(["线圈"], "旋转", "", 0.3, "test", vault_id="feynman-deadbeef")
    data, _ = de.load_domain_extensions_with_hash()
    assert data["vault_id"] == "feynman-deadbeef"
    assert data["edge_scores"]["线圈|旋转"] == 1.3


def test_without_vault_id_keeps_previous_stamp():
    before, _ = de.load_domain_extensions_with_hash()
    apply_evolution_update(["线圈"], "旋转", "", 0.3, "test")
    after, _ = de.load_domain_extensions_with_hash()
    assert after["vault_id"] == before["vault_id"]


def test_stats_reports_mismatch():
    de.EXT_PATH.unlink(missing_ok=True)
    body = {"vote": "up", "entities": ["线圈"], "action": "旋转"}
    resp = client.post("/feynman/feedback", json=body)
    assert resp.status_code == 200, resp.text
    stats = client.get("/stats").json()
    assert stats["ext"]["matches_vault"] is True
    assert stats["ext"]["vault_id"] == stats["ext"]["current_vault_id"]

    data = json.loads(de.EXT_PATH.read_text(encoding="utf-8"))
    data["vault_id"] = "feynman-othervault"
    de.EXT_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    stats = client.get("/stats").json()
    assert stats["ext"]["matches_vault"] is False


def test_legacy_ext_without_stamp_is_unknown_not_false():
    de.EXT_PATH.unlink(missing_ok=True)
    apply_evolution_update(["线圈"], "旋转", "", 0.3, "test")
    data = json.loads(de.EXT_PATH.read_text(encoding="utf-8"))
    data.pop("vault_id", None)
    de.EXT_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert client.get("/stats").json()["ext"]["matches_vault"] is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 扩展库溯源绑定 {len(tests)} 项回归全部通过")
