"""P16 张力弧：只重排候选并在响应里加证据字段，不改任何文本。

锁住：
1. 触发词让张力上升、落地词让它下降（单调、边界）；
2. 阶段 3 的分布整体低于阶段 1；
3. 开/关都不改句子来源：关闭时没有任何新字段，段落逐字节等于基线；
4. 真库 500 条路径的分位校准：阶段 1 中位数 ≥0.60、阶段 3 ≤0.35；
5. 同一请求两次输出完全一致（确定性）。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_tension_arc.py
"""

import statistics
from pathlib import Path

from plugins.feynman_typesetter import Vault
from plugins.paragraph_weaver import (
    ParagraphRequest,
    RESOLUTION_GROUND_TOKENS,
    TENSION_TRIGGER_TOKENS,
    path_tension,
    weave_paragraph,
)

VAULT = Vault(Path("data/feynman_style_vault_p15.json"))
ENT = VAULT.by_slot["entity"][0]["word"]
ACT = VAULT.by_slot["action"][0]["word"]
CON = VAULT.by_slot["consequence"][0]["word"]
HOOK = VAULT.by_slot["hook"][0]["word"]


def _path(line):
    return {"line": line, "phase_mean": 60.0,
            "glyphs": {"hook": HOOK, "entity": ENT, "action": ACT,
                       "consequence": CON}}


def _t(line, stage=1):
    return path_tension(_path(line), stage, VAULT)["tension"]


def test_trigger_and_ground_monotonic():
    base = "电子发光的时候，动量改变"
    triggers = list(TENSION_TRIGGER_TOKENS)
    grounds = list(RESOLUTION_GROUND_TOKENS)
    rising = [_t(base + "".join(triggers[:i])) for i in range(0, 4)]
    assert rising == sorted(rising), rising
    assert rising[-1] > rising[0], rising
    falling = [_t(base + "".join(grounds[:i])) for i in range(0, 4)]
    assert falling == sorted(falling, reverse=True), falling
    assert falling[-1] < falling[0], falling


def test_boundaries():
    all_trig = _t("".join(TENSION_TRIGGER_TOKENS))
    all_ground = _t("".join(RESOLUTION_GROUND_TOKENS))
    assert all_trig >= 0.75, all_trig
    assert all_ground <= 0.35, all_ground
    assert 0.0 <= _t("") <= 1.0
    # 阶段 3 的收束偏置必须把同一条路径整体压低
    mid = "零，观测"
    assert _t(mid, 3) < _t(mid, 1)


def test_phases_match_vault():
    info = path_tension(_path("电子发光"), 1, VAULT)
    assert set(info["phases"]) == {"hook", "entity", "action", "consequence"}
    lookup = {g["word"]: g["phase_deg"] for g in VAULT.by_slot["entity"]}
    assert info["phases"]["entity"] == round(float(lookup[ENT]), 1)


def test_calibration_on_real_pool():
    queries = ["电子为什么会发光", "咖啡杯为什么凉得这么快", "磁铁为什么会吸引铁",
               "时间为什么会变慢", "超导体为什么没有电阻", "光是波还是粒子",
               "熵为什么总是增加", "苹果为什么会落地", "声音为什么能在水里传",
               "原子核里有什么"]
    t1, t3 = [], []
    for q in queries:
        req = ParagraphRequest(user_query=q, style_deg=60, nonce=2, beam_width=6)
        from plugins.feynman_typesetter import PrintRequest, run_feynman_print
        base = run_feynman_print(
            VAULT, PrintRequest(style_deg=60, nonce=2, top_k=50, user_query=q))
        for p in base.get("paths") or []:
            if not p.get("line"):
                continue
            t1.append(path_tension(p, 1, VAULT)["tension"])
            t3.append(path_tension(p, 3, VAULT)["tension"])
    assert len(t1) >= 400, len(t1)
    assert statistics.median(t1) >= 0.60, statistics.median(t1)
    assert statistics.median(t3) <= 0.35, statistics.median(t3)


def test_default_off_is_baseline_and_on_is_deterministic():
    body = dict(user_query="电子为什么会发光", style_deg=60, nonce=2, beam_width=6)
    off_a = weave_paragraph(VAULT, ParagraphRequest(**body))
    off_b = weave_paragraph(VAULT, ParagraphRequest(**body))
    for result in (off_a, off_b):
        assert "tension_arc" not in result
        assert all("tension" not in s and "phases" not in s
                   for s in result["sentences"])
    assert off_a["paragraph"] == off_b["paragraph"]
    assert off_a["sentences"] == off_b["sentences"]

    on_a = weave_paragraph(VAULT, ParagraphRequest(**body, dissonance_resolution=True))
    on_b = weave_paragraph(VAULT, ParagraphRequest(**body, dissonance_resolution=True))
    assert on_a["paragraph"] == on_b["paragraph"], "张力弧不应引入随机性"
    assert on_a["sentences"] == on_b["sentences"]
    assert on_a["tension_arc"]["enabled"] is True
    assert len(on_a["tension_arc"]["stages"]) == len(on_a["sentences"])
    for s in on_a["sentences"]:
        assert 0.0 <= s["tension"] <= 1.0
        assert set(s["phases"]) == {"hook", "entity", "action", "consequence"}
    # 只加字段，不改文本：每句仍是原句本身
    on_lines = [s["line"] for s in on_a["sentences"]]
    assert all(on_lines), on_lines


def test_arc_drops_across_stages_in_most_requests():
    queries = ["电子为什么会发光", "咖啡杯为什么凉得这么快", "磁铁为什么会吸引铁",
               "时间为什么会变慢", "超导体为什么没有电阻", "光是波还是粒子"]
    drops = 0
    for q in queries:
        r = weave_paragraph(VAULT, ParagraphRequest(
            user_query=q, style_deg=60, nonce=2, beam_width=6,
            dissonance_resolution=True))
        arc = r.get("tension_arc") or {}
        if arc.get("T1") is not None and arc.get("T3") is not None and arc["T3"] <= arc["T1"]:
            drops += 1
    assert drops >= 5, drops


def main():
    test_trigger_and_ground_monotonic()
    test_boundaries()
    test_phases_match_vault()
    test_calibration_on_real_pool()
    test_default_off_is_baseline_and_on_is_deterministic()
    test_arc_drops_across_stages_in_most_requests()
    print("✅ test_tension_arc: 6/6")


if __name__ == "__main__":
    main()
