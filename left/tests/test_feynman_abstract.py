"""P6.2 抽象实体原型路由回归。

抽象名词（股票/算法/合同）不该配物理动作（推动/旋转/散热）；
语料共现证明不了的事，靠原型规则压掉，而不是让整池路径一起饿死。
"""

import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

from plugins.feynman_typesetter import (
    ABSTRACT_PROTOTYPE_RULES,
    PHYSICAL_ONLY_ACTIONS,
    PROTOTYPE_MAP,
    SAFE_COGNITIVE_ACTIONS,
    OperatorWeights,
    PrintRequest,
    Vault,
    _evaluate_path,
    _user_glyph,
    extract_entities,
    resolve_entity_prototype,
    run_feynman_print,
)

REAL_VAULT = Path("data/feynman_style_vault.json")
HAS_REAL = REAL_VAULT.is_file()


def _vault():
    return Vault()


def test_resolve_entity_prototype_three_way():
    if not HAS_REAL:
        return
    assert resolve_entity_prototype("股票")[0] == "abstract_finance"
    assert resolve_entity_prototype("算法")[0] == "abstract_info"
    assert resolve_entity_prototype("合同")[0] == "abstract_rule"
    assert resolve_entity_prototype("汽车") == ("physical", "车")
    assert resolve_entity_prototype("咖啡杯")[0] == "default_physical"
    # 规则表之间不重叠
    seen = set(PROTOTYPE_MAP)
    for words in ABSTRACT_PROTOTYPE_RULES.values():
        assert not (seen & set(words))
        seen |= set(words)


def test_extract_entities_keeps_abstract_words_despite_pos_tag():
    if not HAS_REAL:
        return
    # jieba 把"合同"标成形容词，词表命中时要照样收
    assert "合同" in extract_entities("合同为什么有约束力")
    assert extract_entities("股票回购是什么")[:1] == ["股票"]
    assert "算法" in extract_entities("算法是怎么工作的")


def _path(entity, action_word, style=55.0):
    v = _vault()
    act = next(g for g in v.by_slot["action"] if g["word"] == action_word)
    return {
        "hook": v.by_slot["hook"][0],
        "entity": _user_glyph(entity, style, f"{entity}为什么"),
        "action": act,
        "consequence": v.by_slot["consequence"][0],
    }


def test_abstract_entity_fuses_physical_verbs_but_keeps_cognitive():
    if not HAS_REAL:
        return
    v = _vault()
    w = OperatorWeights()
    physical = _evaluate_path(_path("股票", "旋转"), 55.0, v.jargon, set(), vault=v, weights=w)
    cognitive = _evaluate_path(_path("股票", "观察"), 55.0, v.jargon, set(), vault=v, weights=w)
    assert physical["weight"] < cognitive["weight"] * 0.05
    assert any("抽象概念不适用物理动作" in n for n in physical["notes"])
    assert any("认知动作" in n for n in cognitive["notes"])


def test_physical_entity_still_allows_physical_verbs():
    if not HAS_REAL:
        return
    v = _vault()
    w = OperatorWeights()
    car = _evaluate_path(_path("汽车", "旋转"), 55.0, v.jargon, set(), vault=v, weights=w)
    assert car["weight"] > 0
    assert not any("抽象概念" in n for n in car["notes"])


def test_abstract_queries_never_collapse_onto_physical_verbs():
    if not HAS_REAL:
        return
    v = _vault()
    for query in ("股票回购是什么", "算法是怎么工作的", "合同为什么有约束力"):
        for style in (15.0, 55.0, 92.0):
            for nonce in range(4):
                out = run_feynman_print(v, PrintRequest(
                    style_deg=style, nonce=nonce, user_query=query))
                assert out["collapsed"], (query, style, nonce, out.get("message"))
                action = out["paths"][0]["glyphs"]["action"]
                assert action not in PHYSICAL_ONLY_ACTIONS, (query, style, nonce, action)


def test_cognitive_bridge_actions_are_real_vault_glyphs():
    if not HAS_REAL:
        return
    v = _vault()
    out = run_feynman_print(v, PrintRequest(style_deg=55, nonce=0,
                                            user_query="期权为什么有价格"))
    words = {g["word"] for g in v.by_slot["action"]}
    for p in out["paths"]:
        assert p["glyphs"]["action"] in words, "候选动作必须来自语料，不得凭空造词"
    assert any(p["glyphs"]["action"] in SAFE_COGNITIVE_ACTIONS for p in out["paths"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("✅ 抽象实体原型路由 6 项回归全部通过")
