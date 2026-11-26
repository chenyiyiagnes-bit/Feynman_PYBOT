"""段落织构：现象锚点 → 92° 类比 → 15° 收束，一段三句。

和 PYBOT4 草图的两处关键不同：
- **只排版一次**：草图每阶段各调一次打印函数，一个段落要重算 9–18 次；
  这里一次 `run_feynman_print` 拿到完整候选池，三个阶段只做筛选、重排、
  换骨架重渲染，延迟与单句同一量级。
- **连接词在渲染层拼**：草图把"打个比方，"拼进 user_query 再打印，会被
  `extract_entities` 当成实体抽出来污染候选；这里连接词只加在最终字符串上。

会话与隐式信号只认阶段 1（现象锚点），一轮对话不会给进化库记三次账。
"""

import math
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from plugins.feynman_typesetter import (
    EMPTY_EXT_HASH,
    OperatorWeights,
    PrintRequest,
    Vault,
    _choose_skeleton,
    _render_line,
    _style_band,
    run_feynman_print,
)

CONNECTORS = ("", "打个比方，", "说到底，")
ANCHOR_KINDS = ("declarative", "causal", "conditional", "temporal")
CLOSURE_KINDS = ("declarative", "causal", "contrast")

# P16 张力弧：只做候选重排与字段标注，不改任何字粒、不造新句。
# 触发词制造悬念，落地词把话收回物理稳态；按词表出现率归一，不做逐词累加。
TENSION_TRIGGER_TOKENS = ("零", "无穷", "绝对", "消失", "反常", "奇怪",
                          "居然", "却", "反而", "并非")
RESOLUTION_GROUND_TOKENS = ("守恒", "不变", "抵消", "平衡", "观测",
                            "实验", "证明", "因为", "遵循", "决定")
# 实测 p95 的相位离散度约 0.043；除以它把 0–0.05 拉到 0–1，避免区分度被自己吃掉
_PHASE_SPREAD_SCALE = 0.05
_STAGE_CENTER = {1: 60.0, 2: 92.0, 3: 15.0}
# 阶段偏置只整体平移该阶段的张力分布，不参与同阶段内排序
_STAGE_BIAS = {1: 0.10, 2: 0.0, 3: -0.32}


class ParagraphRequest(BaseModel):
    user_query: str = Field("", max_length=200)
    style_deg: float = Field(70.0, ge=0.0, le=120.0)
    nonce: int = Field(0, ge=0, le=1_000_000)
    session_id: Optional[str] = Field(None, max_length=64)
    beam_width: int = Field(6, ge=1, le=20)
    operators: OperatorWeights = Field(default_factory=OperatorWeights)
    fusion: str = Field("linear", pattern="^(linear|boltzmann)$")
    fusion_tau: float = Field(1.0, gt=0.0, le=10.0)
    use_topology: bool = False
    # P8：条件级联 + 段落接力（阶段 2 沿用阶段 1 的配价，阶段 3 回指锚实体）
    conditioned_chain: bool = False
    # P16：不协和音解决张力弧；默认关闭，关闭时段落与响应结构都与基线一致
    dissonance_resolution: bool = False
    # 检索开关：与单句同一个 PrintRequest 口径；默认档由总线 _product_defaults 补齐
    topic_lock: bool = False
    retrieve_consequence: bool = False
    retrieval_gates: bool = False
    specific_pref: bool = False
    # P21 同构空间：类比阶段按"远实体/近机制"排序，收束阶段用向量相似度
    # 并锁住锚实体；默认关，关时与旧版逐字节一致
    use_isomorphism: bool = False
    # P20：五条对话优化，默认关闭；开启只加响应字段，不改正文
    socratic: bool = False
    portrait: bool = False
    auto_phase: bool = False
    breathing: bool = False
    cocreate: bool = False
    # 切词规则 profile（persona.json 的 language_profile）：与右脑检索同一套
    # 边界；默认 default，classical 包由桥从人设传入
    language_profile: Optional[str] = None


def _pick_skeleton(vault: Vault, kinds, style_deg: float,
                   operators: OperatorWeights, learned: Optional[Dict[str, float]] = None,
                   metaphor_mode: bool = False, abstract_mode: bool = False) -> Optional[dict]:
    """在给定句式类别里选与语体最合的一条骨架；没有就返回 None（直拼）。"""
    if not vault.skeletons or not kinds:
        return None
    chosen = _choose_skeleton(vault, style_deg, operators,
                              metaphor_mode=metaphor_mode,
                              abstract_mode=abstract_mode,
                              skeleton_scores=learned)
    chosen_kind = (chosen or {}).get("pattern_type")
    best, best_gap = None, 1e9
    for sk in vault.skeletons:
        kind = sk.get("pattern_type")
        if kind not in kinds:
            continue
        gap = abs(float(sk.get("phase_deg", 45.0)) - style_deg)
        if kind == chosen_kind:
            gap -= 0.5  # 整体选择器偏好的句式在同类别里优先
        if gap < best_gap:
            best, best_gap = sk, gap
    return best


def _path_key(p: dict) -> tuple:
    g = p.get("glyphs") or {}
    return (g.get("hook"), g.get("entity"), g.get("action"), g.get("consequence"))


def _slot_key(p: dict) -> tuple:
    """内容键，不含起手式：换个 hook 重印同一句，读起来还是同一句话。"""
    g = p.get("glyphs") or {}
    return (g.get("entity"), g.get("action"), g.get("consequence"))


def _stage_index(stage) -> int:
    if isinstance(stage, str):
        return {"anchor": 1, "analogy": 2, "closure": 3}.get(stage, 1)
    try:
        return min(3, max(1, int(stage)))
    except (TypeError, ValueError):
        return 1


def _phase_lookup(vault: Vault) -> Dict[str, Dict[str, float]]:
    return {slot: {g["word"]: float(g.get("phase_deg", 20.0)) for g in glyphs}
            for slot, glyphs in vault.by_slot.items()}


def _token_rate(text: str, tokens) -> float:
    """词表出现率：出现了几个词 / 词表大小，避免逐词 +0.15 累加饱和。"""
    if not text:
        return 0.0
    return sum(1 for t in tokens if t in text) / float(len(tokens))


def _circular_dispersion(phases: List[float]) -> float:
    """四槽相位的圆周离散度 1-R：全同向为 0，完全相反接近 1。"""
    if not phases:
        return 0.0
    xs = sum(math.cos(math.radians(p)) for p in phases) / len(phases)
    ys = sum(math.sin(math.radians(p)) for p in phases) / len(phases)
    return max(0.0, min(1.0, 1.0 - math.hypot(xs, ys)))


def path_tension(path: dict, stage, vault: Vault,
                 skeleton: Optional[dict] = None,
                 lookup: Optional[Dict[str, Dict[str, float]]] = None) -> dict:
    """一条路径在某阶段的认知张力 [0,1]；纯函数，只用已有字形与渲染文本。

    三部分：逐槽相位离散度、文本里的触发词/落地词出现率、阶段骨架偏置。
    骨架偏置只平移该阶段整体分布，不参与同阶段内排序。
    """
    idx = _stage_index(stage)
    lookup = lookup or _phase_lookup(vault)
    glyphs = path.get("glyphs") or {}
    phases = {slot: float(lookup.get(slot, {}).get(glyphs.get(slot), 20.0))
              for slot in ("hook", "entity", "action", "consequence")}
    dispersion = _circular_dispersion(list(phases.values()))
    spread = min(1.0, dispersion / _PHASE_SPREAD_SCALE)
    text = str(path.get("line") or path.get("rendered") or "")
    trigger = _token_rate(text, TENSION_TRIGGER_TOKENS)
    ground = _token_rate(text, RESOLUTION_GROUND_TOKENS)
    sk_phase = float((skeleton or {}).get("phase_deg", _STAGE_CENTER[idx]))
    skeleton_bias = max(-0.05, min(0.05,
                                   (sk_phase - _STAGE_CENTER[idx]) / 120.0 * 0.10))
    bias = _STAGE_BIAS[idx] + skeleton_bias
    tension = 0.55 + 0.20 * spread + 0.20 * trigger - 0.50 * ground + bias
    tension = max(0.0, min(1.0, tension))
    return {
        "tension": round(tension, 4),
        "phases": {k: round(v, 1) for k, v in phases.items()},
        "parts": {
            "dispersion": round(dispersion, 4),
            "spread": round(spread, 4),
            "trigger_rate": round(trigger, 4),
            "ground_rate": round(ground, 4),
            "stage_bias": round(bias, 4),
        },
    }


def _decorate(stage: dict, info: dict) -> dict:
    """把张力学证据挂到句子上；不改 line / glyphs / 任何文本。"""
    stage["phases"] = dict(info["phases"])
    stage["tension"] = info["tension"]
    stage["tension_parts"] = dict(info["parts"])
    return stage


def _closure_sim(path: dict, anchor: Optional[dict]) -> float:
    """收束相似度：与锚点的相位方向 + 是否同一实体，两半各占一半。"""
    if not anchor:
        return 0.5
    try:
        delta = float(path.get("phase_mean") or 0.0) - float(anchor.get("phase_mean") or 0.0)
        phase_sim = max(0.0, math.cos(math.radians(delta)))
    except (TypeError, ValueError):
        phase_sim = 0.5
    same_entity = ((path.get("glyphs") or {}).get("entity")
                   == (anchor.get("glyphs") or {}).get("entity"))
    return 0.5 * phase_sim + (0.5 if same_entity else 0.0)


def _closure_sim_p21(path: dict, anchor: Optional[dict], engine) -> float:
    """阶段 E：在相位相似度上叠同构向量；没有向量证据时退回原函数。"""
    base = _closure_sim(path, anchor)
    if engine is None:
        return base
    try:
        v = engine.closure_vector_sim(path, anchor)
    except Exception:
        return base
    if v is None:
        return base
    return 0.5 * base + 0.5 * ((v + 1.0) / 2.0)


def _compose(vault: Vault, path: dict, skeleton: Optional[dict],
             user_entity: str = "", drop_hook: bool = False) -> str:
    return _render_line(skeleton, path.get("glyphs") or {}, vault,
                        path.get("orders"), user_entity=user_entity,
                        drop_hook=drop_hook)


def weave_paragraph(vault: Vault, req: ParagraphRequest, ext_hash: str = "",
                    learned_edges: Optional[Dict[str, float]] = None,
                    skeleton_scores: Optional[Dict[str, float]] = None,
                    prototypes: Optional[Dict[str, str]] = None,
                    topology=None) -> dict:
    """一次打印拿候选池，三阶段只做重排与重渲染。"""
    started = time.perf_counter()
    # 纯招呼语：直接给模板回复，不拿语料拼一段与对话无关的物理句
    from plugins.feynman_typesetter import detect_smalltalk
    talk = detect_smalltalk(req.user_query)
    if talk:
        band, desc = _style_band(req.style_deg)
        return {
            "paragraph": talk,
            "sentences": [{
                "stage": 0, "kind": "greeting", "connector": "", "line": talk,
                "glyphs": {}, "badges": [], "probability": None, "valency": "post",
            }],
            "complete": False,
            "greeting": True,
            "pool_size": 0,
            "style_band": band,
            "style_desc": desc,
            "seed": None,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "user_entities": [],
            "user_verbs": [],
            "anchor": {},
        }
    # 一次打印本来就评估全部 81 条路径，多取几条输出不增加计算；
    # beam_width 只控制后两阶段从前多少条候选里找（前段用户实体会占满前排）。
    pool_size = 50
    scan_limit = min(pool_size, max(20, req.beam_width * 10))
    print_req = PrintRequest(
        style_deg=req.style_deg, nonce=req.nonce, top_k=pool_size,
        user_query=req.user_query, session_id=req.session_id,
        operators=req.operators, fusion=req.fusion,
        conditioned_chain=req.conditioned_chain,
        use_isomorphism=req.use_isomorphism,
        topic_lock=req.topic_lock,
        retrieve_consequence=req.retrieve_consequence,
        retrieval_gates=req.retrieval_gates,
        specific_pref=req.specific_pref,
        language_profile=req.language_profile,
    )
    base = run_feynman_print(vault, print_req, ext_hash=ext_hash,
                             learned_edges=learned_edges,
                             skeleton_scores=skeleton_scores,
                             prototypes=prototypes, topology=topology)
    base["weaver_pool_size"] = pool_size

    p21_engine = None
    if req.use_isomorphism:
        from plugins.isomorphism_p21 import load_engine
        p21_engine = load_engine()

    pool = [p for p in (base.get("paths") or []) if p.get("line")]
    # 问句带谓语时，现象锚点优先落在"用户问的那件事"上（"电子发光"），
    # 段落接力再沿同一配价往下走；否则锚点常被语料里权重更高的空动作占了
    user_verbs = set(base.get("user_verbs") or [])
    if user_verbs:
        pool.sort(key=lambda p: 0 if (p.get("glyphs") or {}).get("action") in user_verbs else 1)
    user_entities = list(base.get("user_entities") or [])
    abstract_mode = False
    if user_entities:
        from plugins.feynman_typesetter import resolve_entity_prototype
        abstract_mode = any(resolve_entity_prototype(e, prototypes)[0].startswith("abstract")
                            for e in user_entities)
    metaphor_mode = bool(user_entities)

    anchor_sk = _pick_skeleton(vault, ANCHOR_KINDS, max(0.0, min(60.0, req.style_deg)),
                               req.operators, skeleton_scores,
                               metaphor_mode=metaphor_mode, abstract_mode=abstract_mode)
    analogy_sk = _pick_skeleton(vault, ("analogy",), 92.0, req.operators,
                                skeleton_scores, metaphor_mode=metaphor_mode,
                                abstract_mode=abstract_mode)
    closure_sk = _pick_skeleton(vault, CLOSURE_KINDS, 15.0, req.operators,
                                skeleton_scores, metaphor_mode=metaphor_mode,
                                abstract_mode=abstract_mode)

    stages: List[dict] = []
    used_lines: set = set()
    used_keys: set = set()
    used_actions: set = set()
    used_hooks: set = set()
    tension_on = bool(req.dissonance_resolution)
    phase_lookup = _phase_lookup(vault) if tension_on else {}

    def _add(stage_idx: int, path: dict, skeleton: Optional[dict],
             user_entity: str = "", tension: Optional[dict] = None) -> bool:
        connector = CONNECTORS[min(stage_idx, len(CONNECTORS) - 1)]
        # 连接词已经开了头，起手式不再重复念一遍（"打个比方，我们看到，……"）
        line = _compose(vault, path, skeleton, user_entity=user_entity,
                        drop_hook=bool(connector))
        if not line or line in used_lines:
            return False
        if _slot_key(path) in used_keys and stage_idx > 0:
            return False
        stage = {
            "stage": stage_idx,
            "kind": (skeleton or {}).get("pattern_type", "plain"),
            "connector": connector,
            "line": line,
            "glyphs": path.get("glyphs"),
            "sources": path.get("sources") or {},
            "badges": path.get("badges") or [],
            "probability": path.get("probability"),
            "valency": (path.get("orders") or {}).get("action", "post"),
        }
        if path.get("topology"):
            stage["topology"] = path["topology"]
        if tension_on:
            _decorate(stage, tension or path_tension(
                path, stage_idx + 1, vault, skeleton, phase_lookup))
        stages.append(stage)
        used_lines.add(line)
        used_keys.add(_slot_key(path))
        used_actions.add((path.get("glyphs") or {}).get("action"))
        used_hooks.add((path.get("glyphs") or {}).get("hook"))
        return True

    # 阶段 1：现象锚点。默认取第一条能渲染出不同句子的（基线）。
    # 开张力弧时只在 Top-10 里换排序：优先接近目标张力 0.72 的句子，概率作次序。
    anchor = None

    def _anchor_info(p: dict) -> dict:
        return path_tension(p, 1, vault, anchor_sk, phase_lookup)

    if tension_on:
        candidates = []
        for order, p in enumerate(pool[:10]):
            info = _anchor_info(p)
            candidates.append((-abs(info["tension"] - 0.72),
                               -(p.get("probability") or 0.0), order, p, info))
        candidates.sort(key=lambda row: row[:3])
        for _, _, _, p, info in candidates:
            if _add(0, p, anchor_sk, tension=info):
                anchor = p
                break
        if anchor is None:
            for p in pool:
                if _add(0, p, anchor_sk, tension=_anchor_info(p)):
                    anchor = p
                    break
    else:
        for p in pool:
            if _add(0, p, anchor_sk):
                anchor = p
                break

    # 阶段 2：92° 类比。右槽放阶段 1 的实体（用户问的东西优先），左槽是语料里的现象。
    anchor_entity = (anchor or {}).get("glyphs", {}).get("entity") or (
        user_entities[0] if user_entities else "")

    def _differs(p: dict) -> bool:
        """和现象锚点至少两个槽不同：避免整段只是换了起手式。"""
        if not anchor:
            return True
        g, a = p.get("glyphs") or {}, anchor.get("glyphs") or {}
        same = sum(1 for k in ("entity", "action", "consequence") if g.get(k) == a.get(k))
        return same <= 1

    anchor_valency = ((anchor or {}).get("orders") or {}).get("action", "post")
    anchor_mech = None
    if p21_engine is not None and anchor:
        _ag = anchor.get("glyphs") or {}
        anchor_mech = p21_engine.mechanism(_ag.get("action", ""), _ag.get("consequence", ""))
    t1 = (stages[0].get("tension") if tension_on and stages else 0.6) or 0.0

    # 条件束限制了用户实体时，候选里可能只剩这一个实体：换实体的类比找不到，
    # 就退回"同实体、换动作/后果"的第二条路径。接力时配价（及物/不及物）是硬约束：
    # 先在同配价里松"整句不同"，确实没有才放弃配价。
    for same_entity, strict in ((False, True), (True, True), (False, False), (True, False)):
        placed = False
        eligible = []
        for order, p in enumerate(pool[:scan_limit]):
            if p is anchor:
                continue
            same = (p.get("glyphs") or {}).get("entity") == anchor_entity
            if same_entity != same:
                continue
            if strict and not _differs(p):
                continue
            # 同一段里动词不重复：三句都印"产生"时读起来只剩一个动作
            if strict and (p.get("glyphs") or {}).get("action") in used_actions:
                continue
            # 起手式也不重复：三句都"我们看到"会像同一句复读
            if strict and (p.get("glyphs") or {}).get("hook") in used_hooks:
                continue
            # P8 接力：类比的动词要和锚点同配价（不及物接不及物）；
            # 同一配价里实在没有可用的新内容时，最后两轮放开配价也要凑齐一段。
            if (req.conditioned_chain and strict is True
                    and (p.get("orders") or {}).get("action", "post") != anchor_valency):
                continue
            info = None
            score = 0.0
            if tension_on:
                info = path_tension(p, 2, vault, analogy_sk, phase_lookup)
                # 惩罚张力剧烈震荡：让类比停在锚点附近，而不是再起一个峰
                score = math.exp(-abs(info["tension"] - t1 * 0.9) / 0.25)
            if p21_engine is not None:
                analog = p21_engine.analogy_score(anchor_entity, anchor_mech, p)
                if analog is not None:
                    # 远实体 / 近机制：换物种但机制相同才是好类比
                    score = (score if score else 1.0) * analog
            eligible.append((-score, order, p, info))
        if tension_on or p21_engine is not None:
            eligible.sort(key=lambda row: row[:2])
        for _, _, p, info in eligible:
            if _add(1, p, analogy_sk, user_entity=anchor_entity, tension=info):
                placed = True
                break
        if placed:
            break

    # 阶段 3：15° 收束。优先与阶段 1 同实体的路径，让首尾落在同一件事上。
    ordered = sorted(
        (p for p in pool[:scan_limit] if p is not anchor),
        key=lambda p: (
            0 if (p.get("glyphs") or {}).get("entity") == anchor_entity else 1,
            -(p.get("probability") or 0.0),
        ),
    )
    # false 再放开"至少两槽不同"：条件束只剩几条路径时，宁可收束句与锚点近一点，
    # 也不能缺第三句（段落织构的 contract 是三句）。
    for strict in (True, False, None):
        placed = False
        eligible = []
        for order, p in enumerate(ordered):
            if strict is True and not _differs(p):
                continue
            if strict is True and (p.get("glyphs") or {}).get("action") in used_actions:
                continue
            if strict is True and (p.get("glyphs") or {}).get("hook") in used_hooks:
                continue
            # P8 接力：收束句必须回指锚实体（第三人称代词会指代不明，不用）
            if (req.conditioned_chain and strict is not False
                    and (p.get("glyphs") or {}).get("entity") != anchor_entity):
                continue
            # P21 实体锁：开同构时收束句先锁锚实体，修 P16 的 11/40 实体漂移；
            # strict 退到 False/None 时仍然放开，保证三句段落凑得齐
            if (p21_engine is not None and anchor_entity and strict is True
                    and p21_engine.has(anchor_entity)
                    and (p.get("glyphs") or {}).get("entity") != anchor_entity):
                continue
            info = None
            score = 0.0
            if tension_on:
                info = path_tension(p, 3, vault, closure_sk, phase_lookup)
                # 张力落差 × 收束相似度：光"越平越好"会选到没信息量的句子
                score = (math.exp(0.85 * (t1 - info["tension"]))
                         * _closure_sim_p21(p, anchor, p21_engine))
            elif p21_engine is not None:
                score = _closure_sim_p21(p, anchor, p21_engine)
            eligible.append((-score, order, p, info))
        if tension_on or p21_engine is not None:
            eligible.sort(key=lambda row: row[:2])
        for _, _, p, info in eligible:
            if _add(2, p, closure_sk, tension=info):
                placed = True
                break
        if placed:
            break

    paragraph = "".join(s["connector"] + s["line"] for s in stages)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    band, desc = _style_band(req.style_deg)
    result = {
        "paragraph": paragraph,
        "sentences": stages,
        "complete": len(stages) == 3,
        "pool_size": len(pool),
        "style_band": band,
        "style_desc": desc,
        "seed": base.get("seed"),
        "user_entities": user_entities,
        "user_verbs": sorted(user_verbs),
        "ext_hash": base.get("ext_hash") or ext_hash or EMPTY_EXT_HASH,
        "latency_ms": latency_ms,
        "anchor": {
            "entities": user_entities,
            "action": (anchor or {}).get("glyphs", {}).get("action", ""),
            "skeleton": (stages[0]["kind"] if stages else ""),
            "style_deg": req.style_deg,
        },
    }
    if topology is not None:
        result["topology_hash"] = topology.topo_hash
    if tension_on:
        result["tension_arc"] = {
            "enabled": True,
            "stages": [s.get("tension") for s in stages],
            "T1": stages[0].get("tension") if stages else None,
            "T3": stages[-1].get("tension") if len(stages) == 3 else None,
        }
    return result


__all__ = ["ParagraphRequest", "weave_paragraph", "CONNECTORS"]
