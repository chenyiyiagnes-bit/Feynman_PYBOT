"""真实 Lean 4 沙盒回归。

本机无 lean 时验证「如实报错 + 安装提示」路径；装了 lean 则做真实编译。
"""
import asyncio

from skeptic.lean_verifier import (
    LeanProofDraft,
    ELAN_INSTALL_HINT,
    probe_lean_environment,
    verify_lean_proof_sandbox,
)

VALID_CORE_PROOF = """def gateH (q : Int × Int) : Int × Int := (q.1 + q.2, q.1 - q.2)
theorem hadamard_involution (a b : Int) : gateH (gateH (a, b)) = (a + b + (a - b), a + b - (a - b)) := rfl
"""

BUGGY_PROOF = """def gateH (q : Int × Int) : Int × Int := (q.1 + q.2, q.1 - q.2)
theorem hadamard_involution (a b : Int) : gateH (gateH (a, b)) = (2 * a, 0) := rfl
"""

CHEAT_PROOF = """theorem fake : True := by sorry
"""


def make_draft(name: str, code: str) -> LeanProofDraft:
    return LeanProofDraft(bot_id="PyBot-Lean", perspective="形式化验证", theorem_name=name, lean_code=code, explanation="回归用例")


async def run():
    env = probe_lean_environment()
    print(f"🔧 Lean 探测：available={env['available']} mode={env['mode']} version={env['version']}")

    # 防作弊与工具链无关，永远成立
    cheat = await verify_lean_proof_sandbox(make_draft("cheat", CHEAT_PROOF))
    assert cheat.phase_deg == 180.0 and "防作弊" in cheat.critical_flaw

    if not env["available"]:
        assert env["hint"] and "elan" in env["hint"]
        missing = await verify_lean_proof_sandbox(make_draft("no_toolchain", VALID_CORE_PROOF))
        assert missing.phase_deg == 180.0
        assert "环境提示" in missing.critical_flaw and "lean" in missing.critical_flaw.lower()
        assert ELAN_INSTALL_HINT in missing.critical_flaw
        print("⚠️ 本机未装 Lean：已验证「不假装通过 + 给出 elan 安装提示」路径")
        return

    ok = await verify_lean_proof_sandbox(make_draft("valid", VALID_CORE_PROOF))
    assert ok.real > 0 and ok.phase_deg == 0.0 and "通过" in ok.critical_flaw, ok.critical_flaw

    bad = await verify_lean_proof_sandbox(make_draft("buggy", BUGGY_PROOF))
    assert bad.real < 0 and bad.phase_deg == 180.0, bad.critical_flaw

    print("🎉 真实 Lean 4 编译：合法证明 0°、错误证明 180°、作弊 180° 全部符合预期")


if __name__ == "__main__":
    asyncio.run(run())
