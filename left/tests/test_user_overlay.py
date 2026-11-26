"""用户 overlay 字库：只许把笔记别名映射到主库既有字粒。

锁住四条硬规则：
1. target 必须在所声明槽位里真实存在，bank_id 必须与当前字库一致；
2. 重复 alias、alias=target、错误 provenance 一律整份拒绝；
3. 问句映射最长别名优先、单遍替换（替换结果不会被二次改写）；
4. 接口只在 QBOT_USER_OVERLAY 设置时介入，命中后返回 overlay 证据；
   未设置时响应里没有 overlay 字段，行为与基线一致。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_user_overlay.py
"""

import json
import os
import tempfile
from pathlib import Path

TMP = tempfile.mkdtemp()
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_EXT"] = os.path.join(TMP, "domain_extensions.json")
os.environ.pop("QBOT_USER_OVERLAY", None)

from fastapi.testclient import TestClient

import bus.main as bm
from plugins.feynman_typesetter import Vault
from plugins.user_overlay import (OverlayError, UserOverlay, load_overlay,
                                  validate_overlay)

VAULT = Vault(Path("data/feynman_style_vault.json"))
ENTITIES = [g["word"] for g in VAULT.by_slot["entity"]]
ACTIONS = [g["word"] for g in VAULT.by_slot["action"]]
W0, W1, W2 = ENTITIES[0], ENTITIES[1], ENTITIES[2]
ACT0 = ACTIONS[0]


def _entry(alias, target, slot="entity", **kw):
    e = {"alias": alias, "target": target, "slot": slot,
         "note_ref": "notes/demo.md#L1", "provenance": "user_note"}
    e.update(kw)
    return e


def _data(entries, bank_id=None, overlay_id="demo"):
    return {"version": 1, "overlay_id": overlay_id,
            "bank_id": bank_id or VAULT.bank_id, "entries": entries}


def _write(data, name="overlay.json"):
    p = Path(TMP) / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_valid_overlay_maps_and_reports():
    ov = load_overlay(_write(_data([_entry("笔记概念", W0)])), VAULT)
    mapped, hits = ov.map_query("笔记概念为什么会发光")
    assert mapped == f"{W0}为什么会发光", mapped
    assert hits == [{"alias": "笔记概念", "target": W0, "slot": "entity",
                     "note_ref": "notes/demo.md#L1",
                     "provenance": "user_note"}], hits
    assert ov.info()["entries"] == 1


def test_longest_alias_first_and_no_rescan():
    ov = load_overlay(_write(_data([
        _entry("笔记", W1),
        _entry("笔记本", W2),
        _entry("映射甲", W0),
        _entry(W0, W1),          # 替换结果不许被再次改写
    ], overlay_id="order")), VAULT)
    mapped, hits = ov.map_query(f"笔记和笔记本，还有映射甲")
    assert mapped == f"{W1}和{W2}，还有{W0}", mapped
    # 单遍替换：W0 已是替换产物，不应再变成 W1
    assert W1 not in mapped.split("还有")[-1], mapped
    assert [h["alias"] for h in hits] == ["笔记", "笔记本", "映射甲"], hits


def test_target_must_exist_in_declared_slot():
    errs = validate_overlay(_data([_entry("笔记甲", "不存在的词")]), VAULT)
    assert any("不在字库 entity 槽位" in e for e in errs), errs
    errs = validate_overlay(_data([_entry("笔记甲", W0, slot="action")]), VAULT)
    assert any("不在字库 action 槽位" in e for e in errs), errs
    assert any("bank_id 不符" in e
               for e in validate_overlay(_data([], bank_id="feynman-000000000000"),
                                         VAULT))


def test_structural_errors_reject_whole_file():
    errs = validate_overlay(_data([
        _entry("笔记甲", W0),
        _entry("笔记甲", W1),
        _entry("笔记乙", "笔记乙"),
        _entry("笔记丙", W2, provenance="corpus"),
    ]), VAULT)
    assert any("alias 重复" in e for e in errs), errs
    assert any("相同" in e for e in errs), errs
    assert any("provenance" in e for e in errs), errs
    bad = _write(_data([_entry("笔记甲", "不存在的词")]), "bad.json")
    try:
        load_overlay(bad, VAULT)
    except OverlayError as exc:
        assert "校验失败" in str(exc)
    else:
        raise AssertionError("不合规 overlay 应拒绝加载")


def test_missing_file_raises():
    try:
        load_overlay(Path(TMP) / "nope.json", VAULT)
    except OverlayError as exc:
        assert "不存在" in str(exc)
    else:
        raise AssertionError("缺失文件应抛 OverlayError")


def test_http_overlay_hits_and_is_isolated():
    # 路由按 .env 的产品字库加载 overlay，这里以同一个字库绑定
    route_vault = bm._get_feynman_vault()
    rword = route_vault.by_slot["entity"][0]["word"]
    data = _data([_entry("笔记概念", rword)], bank_id=route_vault.bank_id)
    client = TestClient(bm.app)
    body = {"user_query": "笔记概念为什么会发光", "style_deg": 60.0,
            "nonce": 3, "beam_width": 5}
    r = client.post("/feynman/paragraph", json=body)
    assert r.status_code == 200, r.text
    assert "overlay" not in r.json(), "未设置 overlay 不应返回 overlay 字段"

    p = _write(data, "http.json")
    os.environ["QBOT_USER_OVERLAY"] = str(p)
    try:
        r = client.post("/feynman/paragraph", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["overlay"]["overlay_id"] == "demo"
        assert out["overlay"]["hits"][0]["target"] == rword, out["overlay"]
        assert rword in out["user_entities"], out["user_entities"]
        assert "笔记概念" not in out["paragraph"], "别名不应进入打印文本"

        # 单句接口共用同一套 overlay 语义
        r1 = client.post("/feynman/print", json=body)
        assert r1.status_code == 200, r1.text
        assert r1.json()["overlay"]["hits"][0]["target"] == rword

        p.write_text(json.dumps(
            _data([_entry("笔记概念", "不存在的词")], bank_id=route_vault.bank_id),
            ensure_ascii=False), encoding="utf-8")
        r = client.post("/feynman/paragraph", json=body)
        assert r.status_code == 400, r.text
    finally:
        os.environ.pop("QBOT_USER_OVERLAY", None)

    r = client.post("/feynman/paragraph", json=body)
    assert "overlay" not in r.json()


def main():
    test_valid_overlay_maps_and_reports()
    test_longest_alias_first_and_no_rescan()
    test_target_must_exist_in_declared_slot()
    test_structural_errors_reject_whole_file()
    test_missing_file_raises()
    test_http_overlay_hits_and_is_isolated()
    print("✅ test_user_overlay: 6/6")


if __name__ == "__main__":
    main()
