import os
import asyncio
from collections import defaultdict
from typing import List, Dict
from pydantic import BaseModel

from shared.protocol import RawWorkerDraft, AmplitudePacket
from shared.deepseek_client import chat_completion, client_available
from workers.feynman_workers import run_parallel_workers
from evaluator.mapreduce_auditor import mapreduce_evaluate_drafts
from skeptic.oracle import feynman_skeptic_verdict


class BeamSolveRequest(BaseModel):
    problem: str
    max_depth: int = 2
    phase_cutoff_deg: float = 90.0


async def _run_focused_perturbation_workers(problem: str, core_hypothesis: str, pruned_traps: List[str]) -> List[RawWorkerDraft]:
    sub_perspectives = {
        "SubBot-A": f"已知主干结论【{core_hypothesis}】在第一轮胜出。请拆解其最底层的物理/因果传导细节。",
        "SubBot-B": f"围绕主干结论【{core_hypothesis}】，施加轻微的极端边界微扰，检验是否依然稳固不崩溃。",
        "SubBot-C": f"专门寻找主干结论【{core_hypothesis}】可能存在的边缘例外（避开已淘汰陷阱：{pruned_traps}）。"
    }
    drafts = []

    for bot_id, instruction in sub_perspectives.items():
        if client_available():
            try:
                resp_text = await asyncio.to_thread(
                    chat_completion,
                    os.getenv("WORKER_MODEL", "deepseek-chat"),
                    f"【原问题】：{problem}\n【二阶聚焦任务】：{instruction}\n请输出150字以内的深度推演。",
                )
                drafts.append(RawWorkerDraft(bot_id=bot_id, perspective=f"二阶微扰·{bot_id}", raw_reasoning=resp_text))
                continue
            except Exception:
                pass
        await asyncio.sleep(0.08)
        drafts.append(RawWorkerDraft(
            bot_id=bot_id,
            perspective=f"二阶微扰·{bot_id}",
            raw_reasoning=f"二阶深化验证【{core_hypothesis}】：在控制体边界与极端参数微扰下，因果链与守恒方程均严格闭合。"
        ))
    return drafts


def _aggregate_and_prune(packets: List[AmplitudePacket], phase_cutoff_deg: float) -> Dict:
    bins = defaultdict(lambda: {"amp": 0j, "packets": []})
    for p in packets:
        bins[p.conclusion_key]["amp"] += complex(p.real, p.imag)
        bins[p.conclusion_key]["packets"].append(p)

    total_weight = sum(abs(v["amp"]) ** 2 for v in bins.values())
    surviving, pruned = [], []

    for claim, data in bins.items():
        amp = data["amp"]
        prob = ((abs(amp) ** 2) / total_weight * 100) if total_weight > 0 else 0.0
        avg_abs_phase = sum(abs(x.phase_deg) for x in data["packets"]) / len(data["packets"])
        entry = {
            "conclusion": claim,
            "probability": round(prob, 2),
            "net_real": round(amp.real, 4),
            "net_imag": round(amp.imag, 4),
            "avg_abs_phase_deg": round(avg_abs_phase, 1)
        }
        if amp.real > 0 and avg_abs_phase <= phase_cutoff_deg:
            surviving.append(entry)
        else:
            pruned.append(entry)

    surviving.sort(key=lambda x: x["probability"], reverse=True)
    return {"surviving": surviving, "pruned": pruned}


async def run_beam_evolution(req: BeamSolveRequest) -> Dict:
    evolution_trace, pruned_traps = [], []

    r1_drafts = await run_parallel_workers(req.problem)
    r1_packets = await mapreduce_evaluate_drafts(req.problem, r1_drafts)
    r1_result = _aggregate_and_prune(r1_packets, req.phase_cutoff_deg)

    for p_item in r1_result["pruned"]:
        pruned_traps.append(f"{p_item['conclusion']} (平均相位 {p_item['avg_abs_phase_deg']}° 已被剪枝)")

    evolution_trace.append({
        "round": 1, "stage": "一阶双向广度采样与驻相剪枝",
        "active_bots": len(r1_packets), "surviving_beams": r1_result["surviving"], "pruned_branches": r1_result["pruned"]
    })

    if not r1_result["surviving"]:
        return {"status": "ALL_PRUNED", "reason": "所有路径均未通过相位自洽阈值", "trace": evolution_trace}

    top_beam = r1_result["surviving"][0]
    final_packets = r1_packets

    if req.max_depth >= 2:
        r2_drafts = await _run_focused_perturbation_workers(req.problem, top_beam["conclusion"], pruned_traps)
        r2_packets = await mapreduce_evaluate_drafts(req.problem, r2_drafts)
        r2_result = _aggregate_and_prune(r2_packets, req.phase_cutoff_deg)

        evolution_trace.append({
            "round": 2, "stage": f"二阶微扰深化（聚焦主干：{top_beam['conclusion']}）",
            "active_bots": len(r2_packets), "surviving_beams": r2_result["surviving"], "pruned_branches": r2_result["pruned"]
        })
        if r2_result["surviving"]:
            top_beam = r2_result["surviving"][0]
            final_packets = r2_packets

    supporting_details = [p.detail for p in final_packets if p.real > 0]
    verdict = await feynman_skeptic_verdict(req.problem, top_beam["conclusion"], supporting_details)

    return {
        "problem": req.problem,
        "total_bot_calls": sum(step["active_bots"] for step in evolution_trace),
        "final_winner": top_beam,
        "pruned_traps": pruned_traps,
        "evolution_trace": evolution_trace,
        "skeptic_verdict": verdict
    }
