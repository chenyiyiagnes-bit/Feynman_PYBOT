"""量子活字印刷机回归：铸字质量、确定性、心境敏感性、套话否决、API 契约。

验收标准（对应方案评审里列出的指标）：
1. 同一 (语料, 心境, nonce) 两次刷墨完全一致；
2. 心境扫 0/45/90/135/170°，Top-1 至少出现 3 种；
3. 套话字粒任何心境下都进不了 Top-1；全套话语料如实拒绝印刷；
4. 铸字结果不含虚词开头/结尾的碎词；
5. 27 条路径排版 < 0.2s。
"""

import os
import tempfile
import time

os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FONT_BANK_PATH"] = os.path.join(tempfile.mkdtemp(), "font_banks.json")

from fastapi.testclient import TestClient

from bus.main import app
from plugins.quantum_tokenizer import FUNCTION_CHARS

client = TestClient(app)

SAMPLE = (
    "在微观的量子世界里，孤独的粒子从不沿着单一的轨迹行走。它像一阵春风穿过时间的废墟，"
    "在无数条可能的路径上同时震荡。当观测者的目光折射出好奇的光芒，所有的概率云瞬间坍缩，"
    "在寂静的宇宙尽头唤醒了沉睡的公式与星辰。"
)
CLICHE_CORPUS = (
    "今天天气真的很不错啊，我们觉得这个事情非常有趣，大家可以一起进行讨论。"
    "今天天气真的很好，我们非常开心，这个事情真的不错。"
)
MOODS = (0, 45, 90, 135, 170)


def _quantize(corpus, max_per_slot=3):
    resp = client.post(
        "/movable_type/quantize",
        json={"corpus_text": corpus, "max_per_slot": max_per_slot, "force_refresh": True},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _print(bank_id, mood, nonce=0, top_k=10):
    resp = client.post(
        "/movable_type/print",
        json={"bank_id": bank_id, "observer_mood_deg": mood, "nonce": nonce, "top_k": top_k},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_quantize_fills_three_slots_with_clean_glyphs():
    data = _quantize(SAMPLE)
    bank = data["font_bank"]
    for slot in ("subjects", "verbs", "objects"):
        assert bank[slot], slot
        for g in bank[slot]:
            assert 0.0 <= g["phase_deg"] <= 180.0
            assert -1.0 <= g["amp"] <= 1.0
            assert g["word"][0] not in FUNCTION_CHARS, g
            assert g["word"][-1] not in FUNCTION_CHARS, g
    assert data["cache_hit"] is False
    assert "语料" in data["source_summary"]


def test_quantize_is_cached_by_corpus_hash():
    first = _quantize(SAMPLE)
    second = client.post(
        "/movable_type/quantize",
        json={"corpus_text": SAMPLE, "max_per_slot": 3},
    ).json()
    assert second["cache_hit"] is True
    assert second["bank_id"] == first["bank_id"]


def test_print_is_deterministic_for_same_seed():
    bank_id = _quantize(SAMPLE)["bank_id"]
    a = _print(bank_id, 90)
    b = _print(bank_id, 90)
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a["paths"][0]["sentence"] == b["paths"][0]["sentence"]
    assert a["seed"] == b["seed"]


def test_printed_line_is_composed_of_glyphs_from_the_bank():
    data = _quantize(SAMPLE)
    bank = data["font_bank"]
    words = {g["word"] for g in bank["subjects"]} | {g["word"] for g in bank["verbs"]} | {
        g["word"] for g in bank["objects"]
    }
    printed = _print(data["bank_id"], 90)
    top = printed["top_resonant_line"]
    assert any(top.startswith(w) for w in words)
    assert any(top.endswith(w) for w in words)


def test_mood_sweep_produces_at_least_three_distinct_top_lines():
    bank_id = _quantize(SAMPLE)["bank_id"]
    tops = [_print(bank_id, m)["top_resonant_line"] for m in MOODS]
    assert len(set(tops)) >= 3, tops


def test_cliche_glyphs_never_win_any_mood():
    data = _quantize(SAMPLE + CLICHE_CORPUS, max_per_slot=4)
    bank = data["font_bank"]
    cliche_words = {
        g["word"]
        for slot in ("subjects", "verbs", "objects")
        for g in bank[slot]
        if g["tag"] == "cliche"
    }
    assert cliche_words, "套话语料应产出至少一颗 cliche 字粒"
    for mood in MOODS:
        top = _print(data["bank_id"], mood)["top_resonant_line"]
        assert not any(w in top for w in cliche_words), (mood, top, cliche_words)


def test_all_cliche_corpus_refuses_to_print():
    data = _quantize(CLICHE_CORPUS)
    for mood in MOODS:
        out = _print(data["bank_id"], mood)
        assert out["collapsed"] is False
        assert out["printed_collapse"] == ""
        assert out["message"]
        assert all(p["cliche"] for p in out["paths"])


def test_empty_corpus_is_rejected():
    resp = client.post("/movable_type/quantize", json={"corpus_text": "   "})
    assert resp.status_code == 422


def test_unknown_bank_id_returns_404():
    resp = client.post(
        "/movable_type/print", json={"bank_id": "does-not-exist", "observer_mood_deg": 90}
    )
    assert resp.status_code == 404


def test_print_latency_under_budget():
    data = _quantize(SAMPLE, max_per_slot=4)
    assert _print(data["bank_id"], 90)["paths_total"] == 64
    started = time.perf_counter()
    for _ in range(5):
        _print(data["bank_id"], 90)
    elapsed = (time.perf_counter() - started) / 5
    assert elapsed < 0.2, f"单次刷墨 {elapsed:.3f}s，超过 0.2s 预算"


if __name__ == "__main__":
    for fn in [
        test_quantize_fills_three_slots_with_clean_glyphs,
        test_quantize_is_cached_by_corpus_hash,
        test_print_is_deterministic_for_same_seed,
        test_printed_line_is_composed_of_glyphs_from_the_bank,
        test_mood_sweep_produces_at_least_three_distinct_top_lines,
        test_cliche_glyphs_never_win_any_mood,
        test_all_cliche_corpus_refuses_to_print,
        test_empty_corpus_is_rejected,
        test_unknown_bank_id_returns_404,
        test_print_latency_under_budget,
    ]:
        fn()
    print("✅ 量子活字印刷机 10 项回归全部通过")
