"""反生硬回归：句子不能三句同一个起手式，也不能答非所问。

锁住：
1. 问句谓语在单句打印里优先（"电子为什么会发光"不能坍缩成"电子停止"）；
2. 段落第二、三句已经有连接词时，不再叠"我们看到"式起手式；
3. 三句的起手式尽量互不重复（候选池够用时）；
4. 极端小池（只有一个 hook）仍能凑齐三句，不因去重变成留白。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_paragraph_variety.py
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print  # noqa: E402
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph  # noqa: E402

REAL_VAULT = Path("data/feynman_style_vault_p15.json")


def _mini_vault(hooks=("你看", "注意", "想象一下", "假设")) -> Vault:
    skeleton = {"pattern_type": "declarative",
                "template": "{hook}，{verb}，{consequence}。",
                "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]}
    analogy = {"pattern_type": "analogy",
               "template": "{hook}，{verb}，就像{consequence}。",
               "phase_deg": 92.0, "freq": 10, "provenance": ["样例句"]}
    glyph = lambda w, slot, ph: {"word": w, "slot": slot, "count": 10, "amp": 1.0,
                                 "domain": "physics", "style": "lecture",
                                 "phase_deg": ph, "sources": ["样｜1"], "example": ""}
    data = {
        "corpus": "(synthetic)", "sections": 1, "sentences": 100,
        "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0,
                         "consequence": 20.0},
        "skeletons": [skeleton, analogy],
        "hooks": [glyph(h, "hook", 60.0) for h in hooks],
        "entities": [glyph("电子", "entity", 20.0), glyph("光子", "entity", 30.0)],
        "actions": [glyph("发光", "action", 40.0), glyph("聚集", "action", 50.0)],
        "consequences": [glyph("温度升高", "consequence", 45.0),
                         glyph("光斑变亮", "consequence", 55.0)],
        "entangled_pairs": [],
        "cooccurrence": {"电子|发光": 6, "发光|光斑变亮": 5,
                         "电子|聚集": 3, "聚集|温度升高": 4},
    }
    path = TMP / f"mini_{len(hooks)}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return Vault(path)


def test_query_verb_wins_single_line():
    vault = Vault(REAL_VAULT)
    out = run_feynman_print(vault, PrintRequest(
        style_deg=55.0, nonce=3, top_k=50, user_query="电子为什么会发光"))
    assert out["user_verbs"] == ["发光"]
    assert "发光" in out["printed_collapse"], out["printed_collapse"]


def test_no_double_opening_after_connector():
    vault = Vault(REAL_VAULT)
    out = weave_paragraph(vault, ParagraphRequest(
        user_query="电子为什么会发光", style_deg=55.0, nonce=3, beam_width=6))
    paragraph = out["paragraph"]
    assert paragraph
    assert "打个比方， 我们看到" not in paragraph
    assert "打个比方，我们看到" not in paragraph
    assert "说到底， 我们看到" not in paragraph
    assert "说到底，我们看到" not in paragraph


def test_hooks_do_not_repeat_when_pool_allows():
    vault = Vault(REAL_VAULT)
    out = weave_paragraph(vault, ParagraphRequest(
        user_query="电子为什么会发光", style_deg=55.0, nonce=3, beam_width=6))
    hooks = [(s.get("glyphs") or {}).get("hook") for s in out["sentences"]
             if s.get("stage", 0) > 0]
    assert len(out["sentences"]) == 3
    assert len(hooks) == 2
    assert hooks[0] != hooks[1], hooks


def test_single_hook_pool_still_completes():
    vault = _mini_vault(hooks=("你看",))
    out = weave_paragraph(vault, ParagraphRequest(
        user_query="电子为什么会发光", style_deg=55.0, nonce=1, beam_width=6))
    assert out["complete"] is True
    # 只有一个起手式时允许重复，但连接词后面不再叠它
    assert "打个比方， 你看" not in out["paragraph"]


def test_hook_dropped_only_after_connector():
    vault = _mini_vault()
    out = weave_paragraph(vault, ParagraphRequest(
        user_query="电子为什么会发光", style_deg=55.0, nonce=1, beam_width=6))
    assert out["sentences"][0]["connector"] == ""
    for s in out["sentences"][1:]:
        assert s["connector"]
        assert not s["line"].startswith(("你看", "注意", "想象一下", "假设")), s["line"]


def main():
    test_query_verb_wins_single_line()
    test_no_double_opening_after_connector()
    test_hooks_do_not_repeat_when_pool_allows()
    test_single_hook_pool_still_completes()
    test_hook_dropped_only_after_connector()
    print("✅ test_paragraph_variety: 5/5")


if __name__ == "__main__":
    main()
