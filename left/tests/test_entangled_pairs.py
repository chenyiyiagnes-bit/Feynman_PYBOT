"""纠缠对建设回归：只从 E→C 共现边生成、(实体, 后果) 键可命中、覆盖可实验挂载。

历史 bug：p8–p15 的 entangled_pairs 来自分句候选名词，和打分配对的
`(entity, consequence)` 键永远对不上，80 对里 0 对生效。这里锁死修复后的形状。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_entangled_pairs.py
"""

import json
import os
import tempfile
from pathlib import Path

from plugins.feynman_typesetter import Vault
from plugins.feynman_vault import _entangled_from_cooc

TMP = Path(tempfile.mkdtemp())


def _vocab():
    return {
        "entity": [{"word": "地球"}, {"word": "电路"}],
        "consequence": [{"word": "距离变化"}, {"word": "磁场变化"}],
        "action": [{"word": "产生"}],
        "hook": [{"word": "注意"}],
    }


def test_pairs_only_from_entity_consequence_edges():
    cooc = {
        "地球|距离变化": 9,
        "距离变化|地球": 2,          # 反向也认，合并为 (地球, 距离变化)
        "电路|磁场变化": 5,
        "地球|产生": 7,              # 实体×动作：不收
        "注意|距离变化": 6,          # 起手式×后果：不收
        "未知词|距离变化": 8,        # 不在槽位：不收
    }
    pairs = _entangled_from_cooc(cooc, _vocab(), top=10, min_count=3)
    keys = {(p["everyday"], p["physics"]) for p in pairs}
    assert keys == {("地球", "距离变化"), ("电路", "磁场变化")}, keys
    assert pairs[0] == {"everyday": "地球", "physics": "距离变化", "count": 11}
    assert all(p["count"] >= 3 for p in pairs)


def test_min_count_and_top_cap():
    cooc = {"地球|距离变化": 9, "电路|磁场变化": 5, "地球|磁场变化": 1}
    pairs = _entangled_from_cooc(cooc, _vocab(), top=1, min_count=3)
    assert len(pairs) == 1 and pairs[0]["everyday"] == "地球"


def _vault_file():
    raw = {
        "corpus": "(synthetic)", "sections": 1, "sentences": 1,
        "slot_offsets": {}, "skeletons": [],
        "hooks": [{"word": "注意"}],
        "entities": [{"word": "地球"}],
        "actions": [{"word": "产生"}],
        "consequences": [{"word": "距离变化"}],
        "entangled_pairs": [{"everyday": "地球", "physics": "距离变化", "count": 4}],
        "cooccurrence": {"地球|距离变化": 4, "地球|产生": 2},
    }
    p = TMP / "vault.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return p


def test_vault_override_env():
    vault_path = _vault_file()
    override = TMP / "pairs.json"
    override.write_text(json.dumps(
        [{"everyday": "电子", "physics": "发光", "count": 7}], ensure_ascii=False),
        encoding="utf-8")
    try:
        os.environ.pop("QBOT_ENTANGLED_PAIRS", None)
        assert Vault(vault_path).entangled == [
            {"everyday": "地球", "physics": "距离变化", "count": 4}]
        os.environ["QBOT_ENTANGLED_PAIRS"] = str(override)
        assert Vault(vault_path).entangled == [
            {"everyday": "电子", "physics": "发光", "count": 7}]
    finally:
        os.environ.pop("QBOT_ENTANGLED_PAIRS", None)


def test_override_missing_file_raises():
    os.environ["QBOT_ENTANGLED_PAIRS"] = str(TMP / "nope.json")
    try:
        try:
            Vault(_vault_file())
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("覆盖文件缺失应抛 FileNotFoundError")
    finally:
        os.environ.pop("QBOT_ENTANGLED_PAIRS", None)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 纠缠对建设 {len(tests)} 项回归全部通过")
