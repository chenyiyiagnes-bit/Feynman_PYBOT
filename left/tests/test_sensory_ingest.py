"""P19 只读感知接入：默认只读，写入要过两道闸，匹配不依赖分词模型。

锁住：
1. 同一段文字得到同一 nonce / 同一读后感（确定性），且读后感只来自现有段落编织器；
2. 默认 `write=false` 时扩展库一分不动，`learned is None`；
3. `write=true` 但环境闸没开 → `write_blocked=True`，仍然一分不动；
4. 环境闸打开后：实体过清洗、动作必须在字库动作表、最多 3 条边，
   `reason="sensory_ingest"`，分数钳位在 [0.15, 1.5]；
5. 匹配不到实体/动作时，开闸也不写（无可学之处），如实说明；
6. 空正文 400；过长正文截断到 2000 字。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_sensory_ingest.py
"""

import os
import tempfile
from pathlib import Path

TMP = tempfile.mkdtemp()
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_EXT"] = os.path.join(TMP, "domain_extensions.json")
os.environ.pop("FEYNMAN_ENABLE_SENSORY_WRITE", None)

from fastapi.testclient import TestClient

import bus.main as bm
from plugins.domain_expander import load_domain_extensions_with_hash
from plugins.feynman_typesetter import Vault
from plugins.sensory_ingest import (
    MAX_DIGEST_CHARS, build_digest, digest_nonce, matched_action,
    matched_entities, plan_write, render_digest_reply, write_requested,
)

VAULT = Vault(Path("data/feynman_style_vault_p15.json"))
ENT = VAULT.by_slot["entity"][0]["word"]
ACT = VAULT.by_slot["action"][0]["word"]
SECOND_ENT = VAULT.by_slot["entity"][1]["word"]

TEXT = f"{ENT}在实验里{ACT}，因为观测到了守恒，所以这件事不奇怪。"


def _reset_ext():
    path = os.environ["QBOT_FEYNMAN_EXT"]
    if os.path.exists(path):
        os.remove(path)


def test_matching_is_substring_and_deterministic():
    ents = matched_entities(TEXT, VAULT)
    assert ENT in ents, ents
    assert matched_action(TEXT, VAULT) == ACT
    assert matched_action("这句话里没有任何动作字粒", VAULT) is None
    assert matched_entities("<script>alert(1)</script>", VAULT) == []
    assert digest_nonce(TEXT) == digest_nonce(TEXT)
    assert digest_nonce(TEXT) != digest_nonce(TEXT + "。")


def test_read_only_digest_makes_no_writes():
    _reset_ext()
    a = build_digest(TEXT, VAULT)
    b = build_digest(TEXT, VAULT)
    assert a["read_only"] is True and a["learned"] is None
    assert a["reflection"] == b["reflection"]
    assert a["sentences"] == b["sentences"]
    assert a["chars"] <= MAX_DIGEST_CHARS
    assert not os.path.exists(os.environ["QBOT_FEYNMAN_EXT"]), "只读路径不应落盘"
    assert 0.0 <= a["absorption"] <= 1.0


def test_write_gate_closed_by_default():
    _reset_ext()
    assert write_requested(True) is False
    plan = plan_write({"matched_entities": [ENT], "matched_action": ACT,
                       "absorption": 0.9}, [ACT])
    assert plan == {"entities": [ENT], "action": ACT, "delta": 0.08,
                    "reason": "sensory_ingest"}
    # 动作不在字库、无实体、无净收益时都不给计划
    assert plan_write({"matched_entities": [ENT], "matched_action": "飞升",
                       "absorption": 0.9}, [ACT]) is None
    assert plan_write({"matched_entities": [], "matched_action": ACT,
                       "absorption": 0.9}, [ACT]) is None
    assert plan_write({"matched_entities": [ENT], "matched_action": ACT,
                       "absorption": 0.5}, [ACT]) is None


def test_http_digest_default_and_gated():
    _reset_ext()
    client = TestClient(bm.app)
    r = client.post("/feynman/digest", json={"raw_content": TEXT, "title": "测试"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["read_only"] is True and body["learned"] is None
    assert ENT in body["matched_entities"] and body["matched_action"] == ACT
    assert body["reflection"] and "📖" in body["reply"]
    assert not os.path.exists(os.environ["QBOT_FEYNMAN_EXT"])

    r2 = client.post("/feynman/digest", json={"raw_content": TEXT, "write": True})
    assert r2.status_code == 200, r2.text
    gated = r2.json()
    assert gated["write_blocked"] is True and gated["learned"] is None
    assert "FEYNMAN_ENABLE_SENSORY_WRITE" in gated["write_note"]
    assert not os.path.exists(os.environ["QBOT_FEYNMAN_EXT"])

    empty = client.post("/feynman/digest", json={"raw_content": "   "})
    assert empty.status_code == 400
    long = client.post("/feynman/digest", json={"raw_content": "零" * 5000})
    assert long.status_code == 200 and long.json()["chars"] == MAX_DIGEST_CHARS


def test_http_digest_learns_only_through_controlled_channel():
    _reset_ext()
    os.environ["FEYNMAN_ENABLE_SENSORY_WRITE"] = "1"
    try:
        client = TestClient(bm.app)
        r = client.post("/feynman/digest", json={"raw_content": TEXT, "write": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["read_only"] is False and body["learned"]
        info = body["learned"]
        assert info["reason"] == "sensory_ingest" and info["action"] == ACT
        assert 0 < len(info["entities"]) <= 3 and 0 < info["delta"] <= 0.10
        assert info["ext_hash"] != info["previous_ext_hash"]
        assert info["ext_hash"] != "00000000"
        assert "学到" in render_digest_reply(body)

        learned = load_domain_extensions_with_hash()[0]
        scores = learned.get("edge_scores") or {}
        assert all(0.15 <= v <= 1.5 for v in scores.values())
        assert any(k.endswith(f"|{ACT}") for k in scores), scores
        assert learned.get("last_evolved_reason") == "sensory_ingest"

        # 没匹配到实体时，开闸也不写，且不报错
        before = load_domain_extensions_with_hash()[1]
        bare = client.post("/feynman/digest",
                           json={"raw_content": "今天天气不错。", "write": True})
        assert bare.status_code == 200
        assert bare.json()["learned"] is None
        assert load_domain_extensions_with_hash()[1] == before
    finally:
        os.environ.pop("FEYNMAN_ENABLE_SENSORY_WRITE", None)


def test_two_entities_are_capped():
    text = f"{ENT}和{SECOND_ENT}都{ACT}，实验观测守恒。"
    d = build_digest(text, VAULT, extra_entities=["墓碑", "追悼会"])
    assert len(d["matched_entities"]) <= 3
    plan = plan_write(d, [ACT])
    assert plan is None or len(plan["entities"]) <= 3


def main():
    test_matching_is_substring_and_deterministic()
    test_read_only_digest_makes_no_writes()
    test_write_gate_closed_by_default()
    test_http_digest_default_and_gated()
    test_http_digest_learns_only_through_controlled_channel()
    test_two_entities_are_capped()
    print("✅ test_sensory_ingest: 6/6")


if __name__ == "__main__":
    main()
