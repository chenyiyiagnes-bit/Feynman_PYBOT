"""switch_vault 回归：只动两个已知键，其它行逐字节保留，秘密值不打印。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_switch_vault.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "switch_vault.py"

ORIGINAL = [
    "# 注释行必须原样保留",
    "DEEPSEEK_API_KEY=sk-should-never-be-printed",
    "QBOT_FEYNMAN_VAULT=data/feynman_style_vault_p10.json",
    "QBOT_DEFAULT_TOPIC_LOCK=1",
    "QBOT_FEYNMAN_TOPOLOGY=data/topology_graph_p10.npz",
    "TRAILING=keep me",
]


def _run(*args):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=180)


def test_rewrite_only_known_keys():
    d = Path(tempfile.mkdtemp(prefix="switch_vault_"))
    env = d / ".env"
    env.write_text("\n".join(ORIGINAL) + "\n", encoding="utf-8")
    p = _run("--env", str(env), "--vault", "data/feynman_style_vault_p15.json",
             "--topology", "data/topology_graph_p15.npz")
    assert p.returncode == 0, p.stderr
    assert "sk-should-never-be-printed" not in p.stdout + p.stderr
    lines = env.read_text(encoding="utf-8").splitlines()
    assert "# 注释行必须原样保留" in lines
    assert "DEEPSEEK_API_KEY=sk-should-never-be-printed" in lines
    assert "TRAILING=keep me" in lines
    assert "QBOT_DEFAULT_TOPIC_LOCK=1" in lines
    assert "QBOT_FEYNMAN_VAULT=data/feynman_style_vault_p15.json" in lines
    assert "QBOT_FEYNMAN_TOPOLOGY=data/topology_graph_p15.npz" in lines
    assert any(x.name.startswith(".env.bak-") for x in d.iterdir())


def test_show_hides_secret_values():
    d = Path(tempfile.mkdtemp(prefix="switch_vault_"))
    env = d / ".env"
    env.write_text("\n".join(ORIGINAL) + "\n", encoding="utf-8")
    p = _run("--env", str(env), "--show")
    assert p.returncode == 0, p.stderr
    assert "sk-should-never-be-printed" not in p.stdout
    assert "QBOT_FEYNMAN_VAULT = data/feynman_style_vault_p10.json" in p.stdout


def test_missing_vault_fails_without_touching_env():
    d = Path(tempfile.mkdtemp(prefix="switch_vault_"))
    env = d / ".env"
    env.write_text("\n".join(ORIGINAL) + "\n", encoding="utf-8")
    before = env.read_bytes()
    p = _run("--env", str(env), "--vault", "data/no_such_vault.json")
    assert p.returncode != 0
    assert "不存在" in p.stderr + p.stdout
    assert env.read_bytes() == before


def test_check_only_does_not_write():
    d = Path(tempfile.mkdtemp(prefix="switch_vault_"))
    env = d / ".env"
    env.write_text("\n".join(ORIGINAL) + "\n", encoding="utf-8")
    before = env.read_bytes()
    p = _run("--env", str(env), "--vault", "data/feynman_style_vault_p15.json",
             "--check-only")
    assert p.returncode == 0, p.stderr
    assert env.read_bytes() == before


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ switch_vault {len(tests)} 项回归全部通过")
