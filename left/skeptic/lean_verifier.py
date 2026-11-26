import asyncio
import cmath
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from pydantic import BaseModel
from shared.protocol import AmplitudePacket


class LeanProofDraft(BaseModel):
    bot_id: str
    perspective: str
    theorem_name: str
    lean_code: str
    explanation: str


FORBIDDEN_KEYWORDS = [
    r"\bsorry\b",
    r"\badmit\b",
    r"\baxiom\b",
    r"\bunsafe\b",
    r"System\.IO",
]

ELAN_INSTALL_HINT = (
    "未在系统 PATH 中检测到 `lean` 编译器。安装方式（官方 elan 引导脚本）：\n"
    "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable\n"
    "安装后新开终端（或 source ~/.elan/env），`lean --version` 有输出即接入成功。\n"
    "需要 `omega`/`ring` 等 Mathlib 战术时，用 `QBOT_LEAN_PROJECT` 指向一个 `lake` 项目目录，验证器会改走 `lake env lean`。"
)


def _resolve_lean_binary() -> str | None:
    lean_bin = shutil.which("lean")
    if lean_bin:
        return lean_bin
    # elan 装了但当前 shell 没 source 环境时，直接找默认安装位置
    fallback = Path.home() / ".elan" / "bin" / "lean"
    return str(fallback) if fallback.is_file() else None


def probe_lean_environment() -> dict:
    """探测真实 Lean 4 工具链：路径、版本、是否走 lake 项目模式。"""
    lean_bin = _resolve_lean_binary()
    project_dir = os.getenv("QBOT_LEAN_PROJECT", "").strip() or None
    lake_bin = shutil.which("lake")
    info = {
        "available": bool(lean_bin),
        "lean_path": lean_bin,
        "version": None,
        "project_dir": project_dir,
        "lake_available": bool(lake_bin),
        "mode": "lake-project" if (lean_bin and project_dir) else ("direct" if lean_bin else "simulation"),
        "hint": None if lean_bin else ELAN_INSTALL_HINT,
    }
    if lean_bin:
        try:
            out = __import__("subprocess").run(
                [lean_bin, "--version"], capture_output=True, text=True, timeout=10
            )
            info["version"] = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else None
        except Exception as e:
            info["version"] = f"probe failed: {e}"
    return info


def _check_lean_cheating(lean_code: str) -> str | None:
    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, lean_code):
            return f"触发防作弊熔断：检测到禁用关键字 `{pattern}`（严禁使用 sorry 或私设公理）"
    return None


def _compile_command(lean_bin: str, lean_file: Path, workdir: str) -> tuple[list[str], str]:
    project_dir = os.getenv("QBOT_LEAN_PROJECT", "").strip()
    lake_bin = shutil.which("lake")
    if project_dir and lake_bin and Path(project_dir).is_dir():
        return [lake_bin, "env", "lean", str(lean_file)], project_dir
    return [lean_bin, str(lean_file)], workdir


async def verify_lean_proof_sandbox(draft: LeanProofDraft, timeout_sec: float = 10.0) -> AmplitudePacket:
    cheat_reason = _check_lean_cheating(draft.lean_code)
    if cheat_reason:
        return _build_formal_packet(draft, is_valid=False, compiler_msg=cheat_reason)

    lean_bin = _resolve_lean_binary()
    if not lean_bin:
        return _build_formal_packet(
            draft, is_valid=False,
            compiler_msg=f"[环境提示] {ELAN_INSTALL_HINT}"
        )

    with tempfile.TemporaryDirectory(prefix="feynman_lean_") as tmp_dir:
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", draft.theorem_name) or "theorem"
        lean_file = Path(tmp_dir) / f"{safe_name}.lean"
        lean_file.write_text(draft.lean_code, encoding="utf-8")
        cmd, workdir = _compile_command(lean_bin, lean_file, tmp_dir)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            out_text = stdout.decode("utf-8", errors="ignore").strip()
            err_text = stderr.decode("utf-8", errors="ignore").strip()
            combined_log = "\n".join(filter(None, [out_text, err_text]))

            if proc.returncode == 0 and "error:" not in combined_log.lower():
                return _build_formal_packet(
                    draft, is_valid=True,
                    compiler_msg="✅ Lean 4 形式化类型检查 100% 通过（零错误、零 sorry）"
                )
            else:
                short_err = combined_log[:200] if combined_log else f"Exit code {proc.returncode}"
                return _build_formal_packet(draft, is_valid=False, compiler_msg=f"❌ Lean 编译失败: {short_err}")
        except asyncio.TimeoutError:
            proc.kill()
            return _build_formal_packet(draft, is_valid=False, compiler_msg=f"⏱️ Lean 验证超时（超过 {timeout_sec}s）")
        except FileNotFoundError as e:
            return _build_formal_packet(draft, is_valid=False, compiler_msg=f"[环境提示] Lean 可执行文件无法启动：{e}")


def _build_formal_packet(draft: LeanProofDraft, is_valid: bool, compiler_msg: str) -> AmplitudePacket:
    theta = 0.0 if is_valid else math.pi
    r = 1.0
    regime = "形式化绝对真理 (0° 零幻觉锁定)" if is_valid else "形式化反相熔断 (180° 编译证伪)"
    amp = cmath.rect(r, theta)

    return AmplitudePacket(
        bot_id=draft.bot_id,
        wave_direction="FORMAL_VERIFICATION",
        perspective=draft.perspective,
        conclusion_key=f"定理 [{draft.theorem_name}] 证明成立",
        detail=draft.explanation,
        critical_flaw=compiler_msg,
        real=round(amp.real, 4),
        imag=round(amp.imag, 4),
        magnitude=r,
        phase_rad=round(theta, 4),
        phase_deg=round(math.degrees(theta), 1),
        interference_regime=regime
    )
