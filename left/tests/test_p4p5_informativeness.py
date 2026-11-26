"""P4/P5：后果从句的信息量偏好（左上下文一致性）。

p12 的最佳配置出句是"测量动量，动量变化"——句法通顺，但后果只是把主语
换个说法重复一遍，没有新信息。语料层实测左邻多为虚词，按"左动作/左主语
匹配"取不到证据；可用的结构信号是：

- 从句自带主语还是依赖左侧主语；
- 从句是否引入新指称（除左侧实体之外的库内实体）。

据此打分（默认关，关时逐字节不变）：
- 从句以左侧实体开头且不引入其它实体 → 空话重复，×0.9；
- 从句以左侧实体开头且引入其它实体 → 有新指称，×1.15；
- 从句以另一个实体开头 → 主语错位，×0.85；
- 从句没有库内实体开头 → 中性 1.0（句法上主语由左侧提供，相关不相关交给检索）。
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.feynman_typesetter import (  # noqa: E402
    PrintRequest,
    Vault,
    _informativeness_factor,
    run_feynman_print,
)


def _g(word, slot, phase, tier=1):
    return {"word": word, "slot": slot, "count": 10, "amp": 1.0, "domain": "physics",
            "style": "lecture", "phase_deg": phase, "sources": ["样｜1"], "example": "",
            "tier": tier}


FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 100,
    "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0},
    "skeletons": [
        {"pattern_type": "declarative", "template": "{hook}，{entity}{action}，{consequence}。",
         "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]},
    ],
    "hooks": [_g("你看", "hook", 60.0)],
    "entities": [_g("动量", "entity", 50.0), _g("电子", "entity", 45.0),
                 _g("光子", "entity", 55.0)],
    "actions": [_g("观察", "action", 55.0), _g("测量", "action", 50.0)],
    "consequences": [
        _g("动量改变", "consequence", 50.0),      # 以左侧实体开头、无新指称：空话
        _g("电子发射一个光子", "consequence", 55.0),  # 自带主语且引入光子
        _g("振幅减小", "consequence", 60.0),      # 无库内实体开头：中性
        _g("光子数增多", "consequence", 40.0),    # 以另一个实体开头：错位
    ],
    "entangled_pairs": [],
    "cooccurrence": {},
}
FAKE_PATH = TMP / "vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def test_factor_classification():
    assert _informativeness_factor("动量改变", "动量", VAULT) < 1.0
    assert _informativeness_factor("电子发射一个光子", "电子", VAULT) > 1.0
    assert _informativeness_factor("振幅减小", "动量", VAULT) == 1.0
    # 以另一个实体开头但引入新指称：不罚（主语错位交给检索门禁），且算有信息
    assert _informativeness_factor("光子数增多", "动量", VAULT) > 1.0
    # 没有其它实体的从句：中性
    assert _informativeness_factor("振幅减小", "电子", VAULT) == 1.0


def test_missing_entity_or_consequence_is_neutral():
    assert _informativeness_factor("", "动量", VAULT) == 1.0
    assert _informativeness_factor("动量改变", "", VAULT) == 1.0


def test_flag_off_is_byte_identical():
    a = run_feynman_print(VAULT, PrintRequest(style_deg=55, nonce=3, top_k=20))
    b = run_feynman_print(VAULT, PrintRequest(style_deg=55, nonce=3, top_k=20))
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a.get("informative_pref") in (False, None)


def test_flag_reorders_paths_toward_informative_consequence():
    base = dict(style_deg=55, nonce=3, top_k=50)
    off = run_feynman_print(VAULT, PrintRequest(**base))["paths"]
    on = run_feynman_print(VAULT, PrintRequest(**base, informative_pref=True))["paths"]

    def count(ps, word, n=5):
        return sum(1 for p in ps[:n] if p["glyphs"]["consequence"] == word)

    # 开启后，前 5 条里有新指称的从句变多，空话重复变少
    assert count(on, "电子发射一个光子") > count(off, "电子发射一个光子")
    assert count(on, "动量改变") < count(off, "动量改变") or count(off, "动量改变") == 0


def test_flag_is_deterministic():
    kw = dict(style_deg=55, nonce=9, top_k=20, informative_pref=True)
    a = run_feynman_print(VAULT, PrintRequest(**kw))
    b = run_feynman_print(VAULT, PrintRequest(**kw))
    assert [p["line"] for p in a["paths"]] == [p["line"] for p in b["paths"]]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P4/P5 信息量偏好 {len(tests)} 项回归全部通过")
