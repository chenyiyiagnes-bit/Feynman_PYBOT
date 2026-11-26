import asyncio
import math
import os
import threading
import time
from collections import OrderedDict, defaultdict
from typing import List, Dict, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from dotenv import load_dotenv

# 必须在导入 plugins 之前加载：feynman_typesetter/topology_builder 的字库与
# 拓扑路径是模块级常量，导入时就会读环境变量。
load_dotenv()

from shared.protocol import SolveRequest, AmplitudePacket
from shared.deepseek_client import client_available
from shared.llm_ledger import ledger, request_budget, remaining_budget
from shared.solve_cache import solve_cache
from shared.metrics import RollingMetrics, now_ms
from workers.feynman_workers import run_parallel_workers, offline_domain
from evaluator.mapreduce_auditor import mapreduce_evaluate_drafts
from skeptic.oracle import feynman_skeptic_verdict, generate_second_notebook_report
from skeptic.lean_verifier import probe_lean_environment
from plugins.grover_engine import GroverRequest, run_grover_simulation
from plugins.movable_type_engine import (
    PrintRequest,
    QuantizeRequest,
    bank_store,
    run_quantize,
    run_quantum_print,
)
from plugins.feynman_typesetter import PrintRequest as FeynmanPrintRequest
from plugins.paragraph_weaver import ParagraphRequest as FeynmanParagraphRequest
from bus.beam_evolver import BeamSolveRequest, run_beam_evolution

load_dotenv()
app = FastAPI(title="Feynman-QBot v2.0 Quantum Interference Bus")

COLLAPSE_THRESHOLD = 60.0

# 「边界条件缺失」检测：题目涉及封闭体，却没给密封/敞篷等边界词，
# 且胜出结论也没有任何条件限定 —— 此时即便模型很有信心，也按方案回第二笔记本。
_BOUNDARY_OBJECT_HINTS = ("卡车", "车厢", "货车", "箱子", "箱内", "容器", "水箱", "船舱", "电梯")
_BOUNDARY_WORDS = ("密封", "密闭", "敞篷", "开口", "开放", "封闭", "无顶", "敞口")
_CLAIM_BOUNDARY_WORDS = ("密封", "敞篷", "边界", "取决于", "条件下", "开放", "密闭")


def _boundary_underspecified(problem: str) -> bool:
    return any(k in problem for k in _BOUNDARY_OBJECT_HINTS) and not any(
        k in problem for k in _BOUNDARY_WORDS
    )


def _mode_tag() -> str:
    """缓存键的运行模式标签：离线仿真结果绝不复用给在线模型，反之亦然。"""
    if not client_available():
        return "offline"
    return f"online:{os.getenv('WORKER_MODEL', 'deepseek-chat')}/{os.getenv('AUDITOR_MODEL', 'deepseek-chat')}"


def compute_damped_collapse_spectrum(
    packets: List[AmplitudePacket],
    imag_damping_lambda: float = 1.0,
    veto_phase_deg: float = 150.0
) -> List[Dict]:
    """非线性反相熔断 + 虚部指数阻尼收缩引擎"""
    wave_bins = defaultdict(lambda: {"amp": 0j, "bots": [], "veto_multipliers": []})
    for p in packets:
        claim = p.conclusion_key
        wave_bins[claim]["amp"] += complex(p.real, p.imag)
        wave_bins[claim]["bots"].append(p.bot_id)
        if abs(p.phase_deg) >= veto_phase_deg:
            severity = min(0.95, abs(p.phase_deg) / 180.0)
            wave_bins[claim]["veto_multipliers"].append(1.0 - severity)

    raw_weights, processed_bins = {}, {}
    for claim, data in wave_bins.items():
        amp = data["amp"]
        veto_factor = math.prod(data["veto_multipliers"]) if data["veto_multipliers"] else 1.0
        effective_real = max(0.0, amp.real) * veto_factor
        net_imag = abs(amp.imag)
        weight = (effective_real ** 2) * math.exp(-imag_damping_lambda * net_imag)
        raw_weights[claim] = weight
        processed_bins[claim] = {
            "effective_real": round(effective_real, 4),
            "net_imag": round(net_imag, 4),
            "veto_triggered": len(data["veto_multipliers"]) > 0,
            "bots": data["bots"]
        }

    total_weight = sum(raw_weights.values())
    spectrum = []
    for claim, weight in raw_weights.items():
        prob = (weight / total_weight * 100) if total_weight > 0 else 0.0
        info = processed_bins[claim]
        spectrum.append({
            "conclusion": claim,
            "probability": round(prob, 2),
            "net_real": info["effective_real"],
            "net_imag": info["net_imag"],
            "veto_triggered": info["veto_triggered"],
            "bots": info["bots"]
        })

    spectrum.sort(key=lambda x: x["probability"], reverse=True)
    return spectrum


def _offline_uncovered_response(problem: str) -> Dict:
    """离线内核只覆盖少数物理域。未覆盖时老实反问，不用模板编答案。"""
    return {
        "collapse_status": "SECOND_NOTEBOOK_PENDING",
        "problem": problem,
        "packets": [],
        "spectrum": [],
        "skeptic_verdict": None,
        "second_notebook": {
            "status": "OFFLINE_UNCOVERED",
            "conflict_diagnosis": "当前运行在离线仿真内核，只覆盖『密封车厢悬停无人机』与『麦克斯韦妖』两类基准物理域。"
                                  "这道题不在覆盖范围内，系统不会套模板硬答。",
            "missing_parameters": [
                "在 .env 中配置 DEEPSEEK_API_KEY 以获得真实模型的推演与盲审",
                "或补充已知条件，把问题改写成上述两类已覆盖的物理域之一",
            ],
            "clarifying_questions": [
                "1. 这道题是否可以直接接入在线模型求解（配置 API Key 后重试）？",
                "2. 题目是否有可归约到的标准物理模型（封闭控制体、热力学循环等）？",
            ],
        },
    }


def rescue_by_relative_rank(packets: List[AmplitudePacket]) -> List[Dict]:
    """全员反相熔断后的补救排序：按「相对损耗最小」重组光谱。

    θ = sign·π·loss^1.2 ⇒ loss = (|θ|/π)^(1/1.2)；得分 = r · (1 − loss)。
    同一结论取最强路径得分为主、其余路径加分，保证结论方向仍由证据决定。
    """
    bins = defaultdict(list)
    for p in packets:
        bins[p.conclusion_key].append(p)

    scores = {}
    for claim, items in bins.items():
        losses, evidence = [], []
        for p in items:
            theta = abs(p.phase_rad)
            loss = (theta / math.pi) ** (1.0 / 1.2) if theta > 0 else 0.0
            losses.append(loss)
            evidence.append(p.magnitude)
        best = max((r * (1.0 - l) for r, l in zip(evidence, losses)), default=0.0)
        support = sum(r * (1.0 - l) for r, l in zip(evidence, losses)) - best
        scores[claim] = best + 0.1 * support

    total = sum(scores.values())
    spectrum = [{
        "conclusion": claim,
        "probability": round(score / total * 100, 2) if total > 0 else 0.0,
        "net_real": 0.0,
        "net_imag": 0.0,
        "veto_triggered": False,
        "bots": [p.bot_id for p in bins[claim]],
        "rescued": True,
    } for claim, score in scores.items()]
    spectrum.sort(key=lambda x: x["probability"], reverse=True)
    return spectrum


async def _run_solve(req: SolveRequest) -> Dict:
    if not client_available() and offline_domain(req.problem) is None:
        return _offline_uncovered_response(req.problem)

    drafts = await run_parallel_workers(req.problem)
    packets = await mapreduce_evaluate_drafts(req.problem, drafts)
    spectrum = compute_damped_collapse_spectrum(packets)
    winner = spectrum[0]
    packet_dicts = [p.model_dump() for p in packets]

    # 全部有效结论都被反相熔断 / 稀释到近零：说明模型没能产生任何站得住的结论，
    # 与「结论冲突、题目缺边界」是两回事，单独区分出来。
    total_prob = sum(s["probability"] for s in spectrum)
    effectively_destroyed = total_prob < 0.5 or (len(spectrum) > 0 and all(s["probability"] < 0.5 for s in spectrum))

    if effectively_destroyed and client_available():
        # 补救一：用更强提示词做一轮「相对最优为锚」的重审（预算够才做）
        budget_left = remaining_budget()
        if budget_left is None or budget_left >= 3:
            packets = await mapreduce_evaluate_drafts(req.problem, drafts, calibration=True)
            spectrum = compute_damped_collapse_spectrum(packets)
            winner = spectrum[0]
            packet_dicts = [p.model_dump() for p in packets]
            total_prob = sum(s["probability"] for s in spectrum)
            effectively_destroyed = total_prob < 0.5 or (len(spectrum) > 0 and all(s["probability"] < 0.5 for s in spectrum))

    if effectively_destroyed:
        # 补救二：仍全员反相时，按「相对损耗最小」重组光谱，不让整轮作废
        rescued = rescue_by_relative_rank(packets)
        if rescued:
            top = rescued[0]
            supporting = [p.detail for p in packets if p.conclusion_key == top["conclusion"]]
            verdict = await feynman_skeptic_verdict(req.problem, top["conclusion"], supporting)
            return {
                "collapse_status": "COLLAPSED_RESCUED",
                "problem": req.problem,
                "packets": packet_dicts,
                "spectrum": rescued,
                "skeptic_verdict": verdict,
                "second_notebook": None,
                "recovery_note": (
                    "常规坍缩门限未达成（所有路径都被反相熔断或虚部阻尼压到近零），"
                    "系统按『相对损耗最小』把最自洽的路径选了出来。此结论可信度低于正常坍缩，"
                    "建议补充边界条件后复算。"
                ),
            }
        return {
            "collapse_status": "NO_VALID_COLLAPSE",
            "problem": req.problem,
            "packets": packet_dicts,
            "spectrum": spectrum,
            "skeptic_verdict": None,
            "second_notebook": {
                "status": "NO_VALID_COLLAPSE",
                "conflict_diagnosis": "本轮所有推理路径的有效振幅均被反相熔断或互相抵消，未产生任何可置信的坍缩结论（通常意味着题目信息严重不足，或各路径结论彼此矛盾且无守恒依据支撑）。",
                "missing_parameters": ["请提供更明确的前提条件或约束，重新发起推演"],
                "clarifying_questions": [
                    "1. 这道题的关键前提是否已经完整给出（例如初始速度、边界约束、判定标准）？",
                    "2. 若结论本身应是确定的，是否可补充更多已知条件后重试一次？"
                ]
            }
        }

    if winner["probability"] >= COLLAPSE_THRESHOLD:
        if _boundary_underspecified(req.problem) and not any(
            k in winner["conclusion"] for k in _CLAIM_BOUNDARY_WORDS
        ):
            notebook_report = await generate_second_notebook_report(req.problem, spectrum, packet_dicts)
            return {
                "collapse_status": "SECOND_NOTEBOOK_PENDING",
                "problem": req.problem,
                "packets": packet_dicts,
                "spectrum": spectrum,
                "skeptic_verdict": None,
                "second_notebook": notebook_report,
            }
        supporting = [p.detail for p in packets if p.conclusion_key == winner["conclusion"] and p.real > 0]
        verdict = await feynman_skeptic_verdict(req.problem, winner["conclusion"], supporting)
        return {
            "collapse_status": "COLLAPSED_VALID",
            "problem": req.problem,
            "packets": packet_dicts,
            "spectrum": spectrum,
            "skeptic_verdict": verdict,
            "second_notebook": None
        }
    else:
        notebook_report = await generate_second_notebook_report(req.problem, spectrum, packet_dicts)
        return {
            "collapse_status": "SECOND_NOTEBOOK_PENDING",
            "problem": req.problem,
            "packets": packet_dicts,
            "spectrum": spectrum,
            "skeptic_verdict": None,
            "second_notebook": notebook_report
        }


@app.post("/solve")
async def solve_paradox(req: SolveRequest):
    key = solve_cache.cache_key(req.problem, _mode_tag())

    if not req.force_refresh:
        cached = solve_cache.get(key)
        if cached is not None:
            return cached

    with request_budget() as budget:
        result = await _run_solve(req)

    result["cache_hit"] = False
    result["llm_calls"] = budget.used
    result["llm_budget"] = ledger.snapshot()
    solve_cache.put(key, req.problem, result)
    return result


@app.get("/stats")
async def stats():
    from plugins.domain_expander import load_domain_extensions_with_hash

    ext_data, ext_hash = load_domain_extensions_with_hash()
    try:
        vault_id = _get_feynman_vault().bank_id
    except HTTPException:
        vault_id = None
    ext_vault_id = ext_data.get("vault_id")
    return {
        "mode": _mode_tag(),
        "llm_ledger": ledger.snapshot(),
        "cache_entries": len(solve_cache._entries),
        "cache_ttl_sec": solve_cache.ttl_sec,
        "lean": probe_lean_environment(),
        "feynman": _feynman_metrics.snapshot(),
        "sessions": {"active": len(_sessions),
                     "max": int(os.getenv("QBOT_SESSION_MAX", "") or DEFAULT_SESSION_MAX)},
        "ext": {
            "hash": ext_hash,
            "vault_id": ext_vault_id,
            "current_vault_id": vault_id,
            # 未标记的旧扩展库返回 None：不知道它是在哪个字库上学出来的
            "matches_vault": (None if not ext_vault_id or not vault_id
                              else ext_vault_id == vault_id),
            "edge_scores": len(ext_data.get("edge_scores") or {}),
        },
    }


@app.get("/lean/status")
async def lean_status():
    return probe_lean_environment()


@app.post("/cache/clear")
async def clear_cache():
    return {"cleared": solve_cache.clear()}


@app.post("/counters/reset")
async def reset_counters():
    ledger.reset()
    _feynman_metrics.reset()
    return {"llm_ledger": ledger.snapshot(), "feynman": _feynman_metrics.snapshot()}


@app.post("/grover")
async def grover_search_endpoint(req: GroverRequest):
    return await run_grover_simulation(req)


@app.post("/solve_beam")
async def solve_with_beam_pruning(req: BeamSolveRequest):
    return await run_beam_evolution(req)


# ---------------------------------------------------------------------------
# 量子活字印刷机（Q-MovableType）：铸字一次、刷墨多次
# ---------------------------------------------------------------------------
@app.post("/movable_type/quantize")
async def movable_type_quantize(req: QuantizeRequest):
    """语料 → 三槽位量子字粒库（按语料哈希缓存，重复铸字不重算）。"""
    try:
        with request_budget() as budget:
            result = await run_quantize(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result["llm_calls"] = budget.used
    return result


@app.post("/movable_type/print")
async def movable_type_print(req: PrintRequest):
    """固定字库 + 观测者心境 → 波函数坍缩印句（纯本地计算，毫秒级）。"""
    try:
        return run_quantum_print(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/movable_type/banks")
async def movable_type_banks():
    return {"bank_ids": bank_store.ids()}


# ---------------------------------------------------------------------------
# 费曼心智活字：字粒来自本地费曼语料，四槽位 + 语体相位旋钮 + 三道闸门
# ---------------------------------------------------------------------------
_feynman_vault = None
_action_words_cache = None
# 会话记忆有界：LRU 上限 + 空闲 TTL，长跑客户端换 session_id 不会把进程撑大
_sessions = OrderedDict()
_session_last_seen: Dict[str, float] = {}
_session_lock = threading.Lock()
DEFAULT_SESSION_MAX = 512
DEFAULT_SESSION_TTL_SEC = 7200.0
# 排版请求的滚动指标（延迟/空白），/stats 曝光，不落盘
_feynman_metrics = RollingMetrics(window=500)


def _session_memory(session_id: str):
    from plugins.feynman_memory import SessionMemory

    def _num(name: str, default):
        raw = os.getenv(name, "").strip()
        try:
            return type(default)(raw) if raw else default
        except ValueError:
            return default

    max_sessions = max(1, _num("QBOT_SESSION_MAX", DEFAULT_SESSION_MAX))
    ttl = _num("QBOT_SESSION_TTL_SEC", DEFAULT_SESSION_TTL_SEC)
    now = time.time()
    with _session_lock:
        if ttl > 0:
            cutoff = now - ttl
            for sid in [s for s, t in _session_last_seen.items() if t < cutoff]:
                _sessions.pop(sid, None)
                _session_last_seen.pop(sid, None)
        memory = _sessions.get(session_id)
        if memory is None:
            memory = SessionMemory()
            _sessions[session_id] = memory
        else:
            _sessions.move_to_end(session_id)
        _session_last_seen[session_id] = now
        while len(_sessions) > max_sessions:
            old, _ = _sessions.popitem(last=False)
            _session_last_seen.pop(old, None)
        return memory


def _product_defaults() -> dict:
    """产品默认档：P15 字库 + 专属从句检索优先（2026-11-25 盲评胜出）。

    只补客户端没显式传的字段；显式传 False 仍然尊重。环境变量置 0 可逐项退回。
    """
    def _on(name: str) -> bool:
        return os.getenv(name, "1").strip() != "0"

    return {
        "topic_lock": _on("QBOT_DEFAULT_TOPIC_LOCK"),
        "retrieve_consequence": _on("QBOT_DEFAULT_RETRIEVE"),
        "retrieval_gates": _on("QBOT_DEFAULT_GATES"),
        "specific_pref": _on("QBOT_DEFAULT_SPECIFIC"),
        "use_topology": _on("QBOT_DEFAULT_TOPOLOGY"),
    }


def _with_product_defaults(req: FeynmanPrintRequest) -> FeynmanPrintRequest:
    updates = {k: v for k, v in _product_defaults().items()
               if k not in req.model_fields_set}
    return req.model_copy(update=updates) if updates else req


def _get_feynman_vault():
    global _feynman_vault
    if _feynman_vault is None:
        try:
            from plugins.feynman_typesetter import Vault
            _feynman_vault = Vault()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _feynman_vault


@app.get("/feynman/vault")
async def feynman_vault_status():
    """字库概况：来源语料、段落/句子数、四槽字粒数、实测槽位偏移与骨架。"""
    return _get_feynman_vault().stats()


@app.get("/feynman/journal")
async def feynman_journal():
    """成长年轮：只读导出当前学到的搭配、骨架偏好、原型与会话痕迹。

    不写任何库、不生成新话术；含去世词的记录整条过滤。同样的状态得到同样的
    Markdown（无时间戳），便于逐字节对照。
    """
    from plugins.domain_expander import load_domain_extensions_with_hash
    from plugins.growth_journal import build_journal, render_markdown
    from plugins.topology_builder import get_engine

    ext_data, ext_hash = load_domain_extensions_with_hash()
    engine = get_engine()
    topo_stats = None
    if engine is not None:
        topo_stats = dict(engine.stats(), enabled=True)
    with _session_lock:
        sessions = [dict(memory.snapshot(), session_id=sid)
                    for sid, memory in _sessions.items()]
    journal = build_journal(_get_feynman_vault().stats(), ext_data, ext_hash,
                            topo_stats, sessions)
    return {"journal": journal, "markdown": render_markdown(journal)}


class FeynmanSessionResetRequest(BaseModel):
    session_id: str


@app.post("/feynman/session/reset")
async def feynman_session_reset(req: FeynmanSessionResetRequest):
    """真正清掉一个会话：量子记忆 + 隐式信号上一轮快照都清空。

    扩展库里的全局经验乘子不受影响——那是系统学到的，不属于某个会话。
    """
    from plugins.evolution_loop import implicit_tracker

    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    with _session_lock:
        had_memory = _sessions.pop(sid, None) is not None
        _session_last_seen.pop(sid, None)
    had_signal = implicit_tracker.forget(sid)
    return {"session_id": sid, "cleared": had_memory or had_signal,
            "memory": had_memory, "implicit_turn": had_signal}


@app.get("/feynman/operators")
async def feynman_operators():
    """五个心智算子的预设与标签；前端据此画滑杆与雷达图。"""
    from plugins.feynman_typesetter import OPERATOR_LABELS, OPERATOR_PRESETS
    return {"presets": OPERATOR_PRESETS, "labels": OPERATOR_LABELS, "max_weight": 2.0}


@app.post("/feynman/print")
async def feynman_print(req: FeynmanPrintRequest, background: BackgroundTasks):
    """四槽位相位排版。纯本地、确定性；证据不足时返回留白字粒而不是编句子。"""
    req = _with_product_defaults(req)
    started_ms = now_ms()
    from plugins.feynman_typesetter import run_feynman_print, BLANK_GLYPH
    from plugins.domain_expander import load_domain_extensions_with_hash
    from plugins.evolution_loop import apply_evolution_update, implicit_tracker
    from plugins.topology_builder import get_engine
    from plugins.user_overlay import OverlayError, load_overlay_or_none

    overlay = None
    overlay_hits: list = []
    try:
        overlay = load_overlay_or_none(_get_feynman_vault())
    except OverlayError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if overlay is not None and req.user_query:
        mapped, overlay_hits = overlay.map_query(req.user_query)
        if mapped != req.user_query:
            req = req.model_copy(update={"user_query": mapped})

    # 扩展库默认空 → ext_hash 恒为 00000000，种子与旧基线逐字节一致
    _ext_data, ext_hash = load_domain_extensions_with_hash()
    learned = _ext_data.get("edge_scores") or {}
    skeleton_scores = _ext_data.get("skeleton_scores") or {}

    memory = None
    bias = 0.0
    if req.session_id:
        memory = _session_memory(req.session_id)
        biased = memory.biased_style(req.style_deg)
        bias = round(biased - req.style_deg, 2)
        if bias:
            req = req.model_copy(update={"style_deg": biased})

    try:
        result = run_feynman_print(_get_feynman_vault(), req, ext_hash=ext_hash,
                                   learned_edges=learned, skeleton_scores=skeleton_scores,
                                   prototypes=_ext_data.get("entity_prototypes") or None,
                                   topology=get_engine(force=req.use_topology))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知字库 {exc}")

    top = result.get("paths") or []
    top0 = top[0] if (result.get("collapsed") and top) else None
    if top0:
        action_word = top0["glyphs"]["action"]
        result["entity_weights"] = {
            e: round(float(learned.get(f"{e}|{action_word}", 1.0)), 3)
            for e in (result.get("user_entities") or [])
        }
        # 隐式信号：同一 session 里"顺着追问"才给上一轮加分，换话题则轻微衰减
        signal = implicit_tracker.observe(
            req.session_id,
            list(result.get("user_entities") or []),
            action_word,
            (result.get("skeleton") or {}).get("pattern_type", ""),
            req.style_deg,
        )
        result["implicit_signal"] = signal and signal["reason"]
        if signal:
            background.add_task(
                apply_evolution_update, signal["entities"], signal["action"],
                signal["skeleton"], signal["delta"], signal["reason"], _action_words(),
                vault_id=_get_feynman_vault().bank_id)

    if memory is not None and top0:
        memory.update(
            float(top0.get("phase_mean", req.style_deg)),
            tuple(result.get("user_entities") or ()),
        )
    if memory is not None:
        snap = memory.snapshot()
        snap.update({"session_id": req.session_id, "bias_deg": bias})
        result["session"] = snap
    if overlay is not None:
        note = overlay.info()
        note["hits"] = overlay_hits
        result["overlay"] = note
    _feynman_metrics.record(now_ms() - started_ms,
                            blank=result.get("printed_collapse") in (None, "", BLANK_GLYPH))
    return result


@app.post("/feynman/paragraph")
async def feynman_paragraph(req: FeynmanParagraphRequest, background: BackgroundTasks):
    """一段三句：现象锚点 → 92° 类比 → 15° 收束。只排版一次，重排三次。"""
    from plugins.domain_expander import load_domain_extensions_with_hash
    from plugins.evolution_loop import apply_evolution_update, implicit_tracker
    from plugins.paragraph_weaver import weave_paragraph
    from plugins.topology_builder import get_engine
    from plugins.user_overlay import OverlayError, load_overlay_or_none

    # 段落与单句共用一套产品默认档：客户端显式传值优先，未传的按 _product_defaults 补齐
    req = _with_product_defaults(req)
    started_ms = now_ms()

    # 用户 overlay：只把问句里的笔记别名换成主库既有字粒，再走原链路。
    # 未设 QBOT_USER_OVERLAY 时完全不介入，输出与基线逐字节一致。
    overlay = None
    overlay_hits: list = []
    try:
        overlay = load_overlay_or_none(_get_feynman_vault())
    except OverlayError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if overlay is not None and req.user_query:
        mapped, overlay_hits = overlay.map_query(req.user_query)
        if mapped != req.user_query:
            req = req.model_copy(update={"user_query": mapped})

    ext_data, ext_hash = load_domain_extensions_with_hash()
    memory = None
    bias = 0.0
    if req.session_id:
        memory = _session_memory(req.session_id)
        biased = memory.biased_style(req.style_deg)
        bias = round(biased - req.style_deg, 2)
        if bias:
            req = req.model_copy(update={"style_deg": biased})

    try:
        result = weave_paragraph(
            _get_feynman_vault(), req, ext_hash=ext_hash,
            learned_edges=ext_data.get("edge_scores") or {},
            skeleton_scores=ext_data.get("skeleton_scores") or {},
            prototypes=ext_data.get("entity_prototypes") or None,
            topology=get_engine(force=req.use_topology))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知字库 {exc}")

    # 会话与隐式信号只认阶段 1，一轮对话不给进化库记三次账
    anchor = result.get("anchor") or {}
    if anchor.get("skeleton"):
        signal = implicit_tracker.observe(
            req.session_id, list(anchor.get("entities") or []),
            anchor.get("action", ""), anchor.get("skeleton", ""), req.style_deg)
        result["implicit_signal"] = signal and signal["reason"]
        if signal:
            background.add_task(
                apply_evolution_update, signal["entities"], signal["action"],
                signal["skeleton"], signal["delta"], signal["reason"], _action_words(),
                vault_id=_get_feynman_vault().bank_id)
    if memory is not None:
        if result.get("sentences"):
            memory.update(float(req.style_deg), tuple(anchor.get("entities") or ()))
        snap = memory.snapshot()
        snap.update({"session_id": req.session_id, "bias_deg": bias})
        result["session"] = snap
    if overlay is not None:
        note = overlay.info()
        note["hits"] = overlay_hits
        result["overlay"] = note
    if req.socratic or req.portrait or req.auto_phase or req.breathing or req.cocreate:
        from plugins.dialogue_modes import DialogueModes, apply_dialogue_modes
        turn = int((memory.snapshot().get("turns") if memory else 0) or 0) or 1
        extras = apply_dialogue_modes(
            _get_feynman_vault(), result,
            DialogueModes(socratic=req.socratic, portrait=req.portrait,
                          auto_phase=req.auto_phase, breathing=req.breathing,
                          cocreate=req.cocreate),
            nonce=req.nonce, turn=turn)
        if extras:
            result["dialogue"] = extras
    from plugins.feynman_typesetter import BLANK_GLYPH
    lines = [(s.get("line") or "") for s in (result.get("sentences") or [])]
    _feynman_metrics.record(now_ms() - started_ms,
                            blank=not lines or all(x == BLANK_GLYPH for x in lines))
    return result


class FeynmanDigestRequest(BaseModel):
    """外界丢进来的一段文字。默认只读；写入要过 `write` + 环境闸两道。"""
    raw_content: str = Field("", max_length=20000)
    title: str = Field("", max_length=80)
    source_type: str = Field("text", max_length=32)
    session_id: Optional[str] = Field(None, max_length=64)
    write: bool = False


@app.post("/feynman/digest")
async def feynman_digest(req: FeynmanDigestRequest):
    """只读感知：读一段文字，用现有段落编织器回一段读后感；默认不写任何东西。"""
    from plugins.domain_expander import load_domain_extensions_with_hash
    from plugins.sensory_ingest import (
        apply_digest_write, build_digest, clamp_chars, plan_write,
        render_digest_reply, write_requested, WRITE_HINT,
    )

    text = clamp_chars(req.raw_content)
    if not text:
        raise HTTPException(status_code=400, detail="raw_content 不能为空")
    ext_data, ext_hash = load_domain_extensions_with_hash()
    vault = _get_feynman_vault()
    extra = list((ext_data.get("entity_prototypes") or {}).keys())
    extra += [str(k).split("|", 1)[0] for k in (ext_data.get("edge_scores") or {})]
    result = await asyncio.to_thread(build_digest, text, vault, ext_hash, extra)
    result["title"] = req.title or ""
    result["source_type"] = req.source_type or "text"

    if req.write:
        plan = plan_write(result, _action_words())
        if plan is None:
            result["write_blocked"] = False
            result["write_note"] = "没有同时匹配到可用实体与字库动作，本次无可学之处。"
        elif not write_requested(True):
            result["write_blocked"] = True
            result["write_note"] = (f"写入默认关闭；设 {WRITE_HINT} 并重启后才落库。")
        else:
            new_hash = await asyncio.to_thread(
                apply_digest_write, plan, _action_words(), vault.bank_id)
            result["read_only"] = False
            result["learned"] = {"entities": plan["entities"], "action": plan["action"],
                                 "delta": plan["delta"], "reason": plan["reason"],
                                 "ext_hash": new_hash, "previous_ext_hash": ext_hash}
            result["write_note"] = "已按受控通道写入经验乘子。"
    result["reply"] = render_digest_reply(result)
    return result


@app.get("/feynman/topology")
async def feynman_topology():
    """拓扑层状态：开关关闭或文件缺失时如实返回，不假装启用。

    health 里带证据覆盖率：只有同句三角是"实心"，两句/三句窗与出处相交
    是弱证据；覆盖率长期接近 0 说明这一层在这个字库上没有判别力。
    """
    from plugins.topology_builder import enabled, get_engine
    engine = get_engine()
    health = None
    if engine is not None:
        health = engine.stats()
        try:
            vault = _get_feynman_vault()
            total = (len(vault.by_slot["entity"]) * len(vault.by_slot["action"])
                     * len(vault.by_slot["consequence"]))
            if total:
                health["simplex_coverage"] = round(
                    engine.stats()["triangles"] / total, 6)
        except Exception:
            pass
    return {"enabled": enabled(),
            "loaded": engine is not None,
            "stats": engine.stats() if engine is not None else None,
            "health": health}


class FeynmanFeedbackRequest(BaseModel):
    """一次点赞/点踩。只认真实存在的动作字粒，实体须过白名单清洗。"""
    vote: str
    entities: List[str] = []
    action: str
    skeleton: str = ""
    session_id: Optional[str] = None


@app.post("/feynman/feedback")
async def feynman_feedback(req: FeynmanFeedbackRequest, background: BackgroundTasks):
    """显式反馈：👍 +0.30 / 👎 −0.45，钳位 [0.15, 1.5]，跌破即剪枝。"""
    from plugins.evolution_loop import (
        EXPLICIT_UPVOTE_BONUS, EXPLICIT_DOWNVOTE_PENALTY, apply_evolution_update,
        sanitize_entity,
    )
    from plugins.domain_expander import load_domain_extensions_with_hash

    if req.vote not in ("up", "down"):
        raise HTTPException(status_code=400, detail="vote 只能是 up 或 down")
    allowed = _action_words()
    if req.action not in allowed:
        raise HTTPException(status_code=400, detail=f"动作「{req.action}」不在字库里")
    ents = [w for w in (sanitize_entity(e) for e in req.entities[:3]) if w]
    if not ents:
        raise HTTPException(status_code=400, detail="至少需要一个有效实体（2–8 字名词）")
    delta = EXPLICIT_UPVOTE_BONUS if req.vote == "up" else EXPLICIT_DOWNVOTE_PENALTY
    _, old_hash = load_domain_extensions_with_hash()
    new_hash = await asyncio.to_thread(
        apply_evolution_update, ents, req.action, req.skeleton, delta,
        "explicit_feedback", allowed, _get_feynman_vault().bank_id)
    return {"ok": True, "applied_delta": delta, "entities": ents,
            "action": req.action, "ext_hash": new_hash, "previous_ext_hash": old_hash}


def _action_words() -> set:
    global _action_words_cache
    if _action_words_cache is None:
        vault = _get_feynman_vault()
        _action_words_cache = {g["word"] for g in vault.by_slot["action"]}
    return _action_words_cache


class FeynmanExtendRequest(BaseModel):
    """手动触发旁路扩展：一次最多 5 个实体，只写扩展库。"""
    entities: List[str] = []


class FeynmanRollbackRequest(BaseModel):
    """按 run_id 撤销一次抓取。"""
    run_id: str = ""


@app.post("/feynman/extend")
async def feynman_extend(req: FeynmanExtendRequest, background: BackgroundTasks):
    """旁路爬虫入口。默认关闭；没开开关时直接 403，不偷偷联网。"""
    from plugins.domain_crawler import enabled, crawl_many
    from plugins.evolution_loop import sanitize_entity

    if not enabled():
        raise HTTPException(status_code=403,
                            detail="旁路扩展器默认关闭；设 FEYNMAN_ENABLE_EXTENDER=1 并重启后再试。")
    ents = [w for w in (sanitize_entity(e) for e in req.entities[:5]) if w]
    if not ents:
        raise HTTPException(status_code=400, detail="至少提供一个 2–8 字实体")
    background.add_task(crawl_many, ents, _action_words())
    return {"queued": ents, "note": "抓取在后台进行；只写 data/domain_extensions.json。",
            "log": "/feynman/extend/log"}


@app.get("/feynman/extend/log")
async def feynman_extend_log(limit: int = 20):
    """最近抓取记录与状态计数（只读）。默认关闭时同样 403。"""
    from plugins.domain_crawler import crawl_stats, enabled

    if not enabled():
        raise HTTPException(status_code=403,
                            detail="旁路扩展器默认关闭；设 FEYNMAN_ENABLE_EXTENDER=1 并重启后再试。")
    return await asyncio.to_thread(crawl_stats, limit)


@app.post("/feynman/extend/rollback")
async def feynman_extend_rollback(req: FeynmanRollbackRequest):
    """按 run_id 撤销一次抓取；其他 run 或基线贡献的边保留。"""
    from plugins.domain_crawler import enabled, rollback_run

    if not enabled():
        raise HTTPException(status_code=403,
                            detail="旁路扩展器默认关闭；设 FEYNMAN_ENABLE_EXTENDER=1 并重启后再试。")
    run_id = (req.run_id or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="缺少 run_id")
    result = await asyncio.to_thread(rollback_run, run_id)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=f"没有这次抓取：{run_id}")
    return result
