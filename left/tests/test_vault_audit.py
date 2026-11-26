"""字库体检脚本回归：合成小字库覆盖可回源/外部书/无源、结构错误与 --strict。

不读产品数据，全部现场构造，避免和真实字库的版本演进互相绑架。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_vault_audit.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_vault.py"
TMP = Path(tempfile.mkdtemp())


def _corpus() -> Path:
    p = TMP / "corpus.jsonl"
    rows = [
        {"source": "费曼集成库｜lecture｜book.md", "section": "s1", "text": "电子会发光。"},
        {"source": "费曼集成库｜lecture｜book2.md", "section": "s2", "text": "引力使空间弯曲。"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    return p


def _glyph(word, slot, phase=30.0, sources=None, count=5):
    return {"word": word, "slot": slot, "count": count, "amp": 1.0,
            "domain": "physics", "style": "lecture", "phase_deg": phase,
            "sources": sources if sources is not None else ["book.md｜§1"]}


def _vault(path: Path, *, extra_source=None, duplicate=False, bad_phase=False):
    cons = [_glyph("空间弯曲", "consequence", 55.0)]
    if extra_source is not None:
        cons.append(_glyph("体积改变", "consequence", 40.0,
                           sources=[extra_source]))
    if duplicate:
        cons.append(_glyph("空间弯曲", "consequence", 30.0))
    hooks = [_glyph("注意", "hook", 5.0)]
    if bad_phase:
        hooks.append(_glyph("其实", "hook", 200.0))
    raw = {
        "corpus": str(_corpus()), "sections": 2, "sentences": 2,
        "slot_offsets": {"hook": -3.9, "entity": -1.9,
                         "action": 0.1, "consequence": 2.1},
        "skeletons": [{"pattern_type": "declarative",
                       "template": "{hook}，{verb}，{consequence}。",
                       "freq": 0, "phase_deg": 45.0}],
        "hooks": hooks,
        "entities": [_glyph("电子", "entity", 28.7)],
        "actions": [_glyph("产生", "action", 26.1)],
        "consequences": cons,
        "entangled_pairs": [{"everyday": "电子", "physics": "空间弯曲", "count": 3}],
        "cooccurrence": {"电子|产生": 10, "产生|空间弯曲": 7},
    }
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _run(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-B", str(SCRIPT), *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def _load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_vault", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_vault_passes_strict():
    v = _vault(TMP / "clean.json")
    r = _run("--vault", v, "--strict")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "结构检查：通过" in r.stdout
    assert "无源 0" in r.stdout


def test_external_book_counted_not_failed():
    v = _vault(TMP / "external.json", extra_source="书｜当代核物理学的进展")
    mod = _load_module()
    rep = mod.audit(v)
    assert rep["sources"]["external"] == 1
    assert rep["sources"]["unresolved"] == 0
    assert rep["glyphs_external_only"] == ["consequence/体积改变"]
    assert rep["glyphs_without_resolvable_source"] == ["consequence/体积改变"]
    assert _run("--vault", v, "--strict").returncode == 0


def test_unresolved_source_fails_strict_and_json():
    v = _vault(TMP / "bad_src.json", extra_source="不存在.md｜§9")
    mod = _load_module()
    rep = mod.audit(v)
    assert rep["sources"]["unresolved"] == 1
    assert "consequence/体积改变" in rep["glyphs_without_resolvable_source"]
    r = _run("--vault", v, "--strict", "--json")
    assert r.returncode == 1
    body = json.loads(r.stdout)
    assert body["sources"]["unresolved"] == 1


def test_structural_errors_detected():
    mod = _load_module()
    dup = mod.audit(_vault(TMP / "dup.json", duplicate=True))
    assert any("重复字粒" in e for e in dup["errors"])
    bad = mod.audit(_vault(TMP / "phase.json", bad_phase=True))
    assert any("越界" in e for e in bad["errors"])
    assert _run("--vault", TMP / "phase.json", "--strict").returncode == 1


def test_cooccurrence_endpoint_missing():
    v = TMP / "cooc.json"
    raw = json.loads(_vault(v).read_text(encoding="utf-8"))
    raw["cooccurrence"]["电子|不存在词"] = 3
    v.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    rep = _load_module().audit(v)
    assert any("共现键端点缺失" in e for e in rep["errors"])


def test_bank_id_matches_vault_loader():
    v = _vault(TMP / "bank.json")
    rep = _load_module().audit(v)
    import hashlib
    raw = json.loads(v.read_text(encoding="utf-8"))
    want = "feynman-" + hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:12]
    assert rep["bank_id"] == want


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 字库体检 {len(tests)} 项回归全部通过")
