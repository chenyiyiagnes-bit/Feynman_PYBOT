"""P20 五条对话优化：默认全关，开启只加字段，正文逐字节不变。

锁住：
1. 五个开关默认 False；关闭时响应里没有 `dialogue` 键；
2. 逐条开启只加对应字段，段落正文与关闭时完全相同；
3. 苏格拉底反问只由固定句架 + 锚点字粒填充，字粒缺一就不出；
4. 比喻画像只排版既有四槽字粒；
5. 自动相位只给建议，不改 style_deg，也不改正文；
6. 人机共创只给候选边，扩展库一分不动；写入仍归 P19 的双闸通道；
7. 同一输入两次输出完全一致。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_dialogue_modes.py
"""

import os
import tempfile

TMP = tempfile.mkdtemp()
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_EXT"] = os.path.join(TMP, "domain_extensions.json")

from fastapi.testclient import TestClient

import bus.main as bm
from plugins.dialogue_modes import (
    BREATH_LINES, DialogueModes, apply_dialogue_modes, breathing_line,
    cocreate_offer, metaphor_portrait, socratic_question,
)

QUERY = "电子为什么会发光"
BASE = {"user_query": QUERY, "style_deg": 60.0, "nonce": 2, "beam_width": 6}

SAMPLE = {
    "anchor": {"glyphs": {"hook": "你看", "entity": "电子", "action": "振动",
                          "consequence": "辐射出光"}},
    "sentences": [
        {"stage": 0, "line": "你看电子振动。", "glyphs": {"hook": "你看", "entity": "电子",
                                                     "action": "振动", "consequence": "辐射出光"},
         "phase_mean": 58.0},
        {"stage": 1, "line": "像水波扩散。", "glyphs": {"entity": "电子", "action": "振动",
                                                    "consequence": "辐射出光"},
         "phase_mean": 92.0},
    ],
}


def test_all_off_returns_nothing():
    assert DialogueModes().any_on() is False
    assert apply_dialogue_modes(None, SAMPLE, None) == {}
    assert apply_dialogue_modes(None, SAMPLE, DialogueModes()) == {}


def test_socratic_uses_template_and_glyphs_only():
    q = socratic_question(SAMPLE, 0)
    assert q and "电子" in q and "振动" in q
    assert q in (
        "如果电子根本没有振动，你会看到什么不一样？",
        "假设你是电子，振动的时候你最先感觉到什么？",
        "振动这件事，少了电子还成立吗？为什么？",
    ), q
    assert socratic_question({"sentences": [{"glyphs": {"entity": "电子"}}]}) is None
    assert socratic_question({"sentences": []}) is None
    assert socratic_question(SAMPLE, 0) == socratic_question(SAMPLE, 0)


def test_portrait_only_lays_out_existing_glyphs():
    p = metaphor_portrait(SAMPLE)
    assert "🖼 比喻画像" in p
    for word in ("你看", "电子", "振动", "辐射出光"):
        assert word in p
    assert "92°" in p or "58°" in p
    assert metaphor_portrait({"sentences": []}) is None


def test_breathing_and_cocreate():
    a = breathing_line(2, 2)
    assert a["line"] in BREATH_LINES and a["seconds"] in (3, 5, 8)
    assert a == breathing_line(2, 2)
    c = cocreate_offer(SAMPLE)
    assert c["candidate_edges"] == ["电子|振动"]
    assert "write=true" in c["requires"]
    assert cocreate_offer({"sentences": []}) is None


def test_http_default_off_is_baseline():
    client = TestClient(bm.app)
    off = client.post("/feynman/paragraph", json=BASE)
    assert off.status_code == 200, off.text
    assert "dialogue" not in off.json()
    again = client.post("/feynman/paragraph", json=BASE)
    assert again.json()["paragraph"] == off.json()["paragraph"]

    flags = {"socratic": True, "portrait": True, "auto_phase": True,
             "breathing": True, "cocreate": True}
    on = client.post("/feynman/paragraph", json=dict(BASE, **flags))
    assert on.status_code == 200, on.text
    body = on.json()
    assert body["paragraph"] == off.json()["paragraph"], "加字段不能改正文"
    d = body["dialogue"]
    assert set(d) <= set(flags), d
    if d.get("socratic"):
        assert "？" in d["socratic"]
    if d.get("auto_phase"):
        assert 0.0 <= d["auto_phase"]["suggested_style_deg"] <= 120.0
    if d.get("cocreate"):
        assert "write=true" in d["cocreate"]["requires"]
    if d.get("breathing"):
        assert d["breathing"]["line"] in BREATH_LINES


def test_single_flag_only_adds_its_own_field():
    client = TestClient(bm.app)
    r = client.post("/feynman/paragraph", json=dict(BASE, auto_phase=True))
    body = r.json()
    assert list(body.get("dialogue", {}).keys()) in ([], ["auto_phase"])
    assert body["sentences"] == client.post("/feynman/paragraph", json=BASE).json()["sentences"]


def test_cocreate_writes_nothing():
    path = os.environ["QBOT_FEYNMAN_EXT"]
    if os.path.exists(path):
        os.remove(path)
    client = TestClient(bm.app)
    client.post("/feynman/paragraph", json=dict(BASE, cocreate=True))
    assert not os.path.exists(path), "共创建议不应落盘"


def main():
    test_all_off_returns_nothing()
    test_socratic_uses_template_and_glyphs_only()
    test_portrait_only_lays_out_existing_glyphs()
    test_breathing_and_cocreate()
    test_http_default_off_is_baseline()
    test_single_flag_only_adds_its_own_field()
    test_cocreate_writes_nothing()
    print("✅ test_dialogue_modes: 7/7")


if __name__ == "__main__":
    main()
