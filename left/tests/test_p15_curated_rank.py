"""P15 策展从句排序：specific_pref 下 tier2 → curated → 次数。

背景：P15 从非费曼物理书补进来的专属从句出现次数天然少（书里各出现 1–2 次），
按原始共现次数排序永远输给语料里反复出现的旧句。策展标记是编辑信号，
不是凭空放大权重：只影响 tier2 内部先后，不创造任何边。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_p15_curated_rank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.feynman_typesetter import Vault, _retrieve_consequences


def _vault(tmp: Path) -> Vault:
    def cons(word, count, tier, curated=False, phase=40.0):
        g = {"word": word, "slot": "consequence", "count": count, "amp": 1.0,
             "domain": "physics", "style": "lecture", "phase_deg": phase,
             "sources": ["书｜测试"], "example": word, "tier": tier}
        if curated:
            g["curated"] = True
        return g

    data = {
        "corpus": "x", "sections": 0, "sentences": 0, "slot_offsets": {},
        "skeletons": [], "hooks": [], "actions": [],
        "entities": [{"word": "线圈", "slot": "entity", "count": 9, "amp": 1.0,
                      "domain": "physics", "style": "lecture", "phase_deg": 40.0,
                      "sources": [], "example": ""}],
        "consequences": [
            cons("气体由内部加热线圈缓慢地加热", 2, 2, phase=40.0),
            cons("示波器上出现一个信号", 2, 2, phase=40.0),
            cons("线圈中就产生感应电动势", 1, 2, curated=True, phase=40.0),
            cons("线圈本身就要变冷", 1, 2, curated=True, phase=40.0),
            cons("电流变化", 15, 1),
            cons("温度降低", 5, 1),
        ],
        "cooccurrence": {
            "线圈|气体由内部加热线圈缓慢地加热": 2,
            "线圈|示波器上出现一个信号": 2,
            "线圈|线圈中就产生感应电动势": 1,
            "线圈|线圈本身就要变冷": 1,
            "线圈|电流变化": 15,
            "线圈|温度降低": 5,
        },
        "entangled_pairs": [],
    }
    p = tmp / "vault.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return Vault(p)


def _top(vault, specific=True, k=6):
    return [g["word"] for g in _retrieve_consequences(
        vault, "线圈", vault.by_slot["consequence"], 40.0, top_k=k,
        specific_pref=specific)]


def test_curated_tier2_first_when_specific():
    tmp = Path("/tmp/feynman_test_p15_curated")
    tmp.mkdir(parents=True, exist_ok=True)
    v = _vault(tmp)
    top = _top(v, specific=True)
    # 策展句（次数 1）排在自动句（次数 2）和所有 tier1 之前
    assert top[:2] == ["线圈中就产生感应电动势", "线圈本身就要变冷"], top
    assert top[2:4] == ["气体由内部加热线圈缓慢地加热", "示波器上出现一个信号"], top
    assert top[4:] == ["电流变化", "温度降低"], top


def test_specific_off_keeps_count_order():
    tmp = Path("/tmp/feynman_test_p15_curated")
    tmp.mkdir(parents=True, exist_ok=True)
    v = _vault(tmp)
    top = _top(v, specific=False)
    assert top == ["电流变化", "温度降低", "气体由内部加热线圈缓慢地加热",
                   "示波器上出现一个信号", "线圈中就产生感应电动势", "线圈本身就要变冷"], top


def test_no_curated_flag_changes_nothing():
    tmp = Path("/tmp/feynman_test_p15_curated")
    tmp.mkdir(parents=True, exist_ok=True)
    v = _vault(tmp)
    # 去掉 curated 标记后，specific 只按 tier + 次数
    for g in v.by_slot["consequence"]:
        g.pop("curated", None)
    top = _top(v, specific=True)
    assert top[:2] == ["气体由内部加热线圈缓慢地加热", "示波器上出现一个信号"], top
    assert top[-2:] == ["电流变化", "温度降低"], top


def test_no_invented_edges():
    tmp = Path("/tmp/feynman_test_p15_curated")
    tmp.mkdir(parents=True, exist_ok=True)
    v = _vault(tmp)
    for w in _top(v, specific=True):
        assert v.cooc_count("线圈", w) > 0, w


def test_deterministic():
    tmp = Path("/tmp/feynman_test_p15_curated")
    tmp.mkdir(parents=True, exist_ok=True)
    a = _vault(tmp)
    b = _vault(tmp)
    assert _top(a) == _top(b)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"✅ P15 策展排序 {len(fns)} 项回归全部通过")
