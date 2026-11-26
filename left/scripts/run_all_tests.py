"""一次跑完全部回归：逐个测试文件起独立进程，互不污染环境变量。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/run_all_tests.py
    ... --filter p15         只跑文件名含 p15 的
    ... --fail-fast          首个失败即停
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="单个测试文件超时秒数（默认 120）")
    args = ap.parse_args()

    files = sorted(p for p in TESTS.glob("test_*.py") if args.filter in p.name)
    if not files:
        raise SystemExit(f"没有匹配的测试文件（filter={args.filter!r}）")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    failed, started = [], time.perf_counter()
    for i, path in enumerate(files, 1):
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-B", str(path)],
                                  cwd=ROOT, env=env, capture_output=True,
                                  text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            secs = time.perf_counter() - t0
            failed.append(path.name)
            print(f"[{i:>2}/{len(files)}] TIMEOUT {path.name}  ({secs:.1f}s，"
                  f"上限 {args.timeout:g}s)")
            if args.fail_fast:
                break
            continue
        secs = time.perf_counter() - t0
        if proc.returncode == 0:
            print(f"[{i:>2}/{len(files)}] ok   {path.name}  ({secs:.1f}s)")
        else:
            failed.append(path.name)
            print(f"[{i:>2}/{len(files)}] FAIL {path.name}  ({secs:.1f}s)")
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
            for line in tail:
                print("        " + line)
            if args.fail_fast:
                break

    total = time.perf_counter() - started
    print(f"\n{len(files) - len(failed)}/{len(files)} 通过，用时 {total:.1f}s")
    if failed:
        print("失败：" + "、".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
