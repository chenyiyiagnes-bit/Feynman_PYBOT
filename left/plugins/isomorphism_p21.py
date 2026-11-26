"""P21 同构空间运行时：只读离线基底，算相容乘子、类比/收束相似度与诊断。

离线产物由 `scripts/build_isomorphism_p21.py` 生成（句级共现 → PPMI → 截断 SVD）。
本模块不做训练、不联网、不调用 LLM，只在已有的 657 个字粒向量上做余弦。

三条硬约束：
1. **替换而不是叠加**：用户实体的"没有共现 ×0.35"离散惩罚在开启时被这里的连续
   乘子取代；原型支持 ×1.15、抽象概念 ×0.02 否决等既有语义约束原样保留。
2. **影响有界**：离散惩罚替换只动用户实体路径；池内全局机制项（rerank_pool）
   对所有候选路径生效，但乘子夹在 exp(±λ/2)≈[0.914, 1.094]，只改相对排序，
   不改变"不生成新文字"的任何前提。
3. **默认关**：`use_isomorphism=False` 时本模块不被导入，输出与基线逐字节一致。
"""

from __future__ import annotations

import bisect
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PATH = Path(os.getenv("QBOT_P21_ISOMORPHISM", "data/isomorphism_p21.json"))
BANDWIDTH_PATH = Path(os.getenv("QBOT_P21_BANDWIDTH", "data/phase_bandwidth_p21.json"))

# 相容乘子的中心与斜率：余弦 0.25 附近乘子恰好 1.0；
# 中心值由 `scripts/calibrate_p21.py` 在开发集真问句上实测后写死。
# 池内分位乘子（阶段 C2）启用后，这两个常量只作为单路径 API 的兜底。
COMPAT_CENTER = 0.25
COMPAT_SLOPE = 1.60
COMPAT_LO, COMPAT_HI = 0.60, 1.60
# 池内分位乘子的范围与全局项强度：全局项 exp(±0.18/2) ≈ [0.914, 1.094]
COMPAT_PERC_LO, COMPAT_PERC_HI = 0.60, 1.60
P21_POOL_LAMBDA = 0.18


def _avg_percentile(ordered: Sequence[float], x: float) -> float:
    """分位（0..1）；并列取平均秩，结果与输入顺序无关。"""
    n = len(ordered)
    if n <= 1:
        return 0.5
    lo = bisect.bisect_left(ordered, x)
    hi = bisect.bisect_right(ordered, x)
    return ((lo + hi - 1) / 2.0) / (n - 1)


_cache: Dict[str, Tuple[float, "P21Engine"]] = {}


def _norm(v: Optional[Sequence[float]]) -> Optional[List[float]]:
    if not v:
        return None
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-12:
        return None
    return [float(x) / n for x in v]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cos(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    """余弦；任一侧缺失或为零向量时返回 None（调用方按"缺项"处理）。"""
    if not a or not b:
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return max(-1.0, min(1.0, _dot(a, b) / (na * nb)))


def _mean_vec(pairs: Iterable[Tuple[Sequence[float], float]]) -> Optional[List[float]]:
    dim = None
    acc: List[float] = []
    for vec, w in pairs:
        if not vec:
            continue
        if dim is None:
            dim = len(vec)
            acc = [0.0] * dim
        elif len(vec) != dim:
            continue
        for i, x in enumerate(vec):
            acc[i] += float(w) * float(x)
    return _norm(acc)


def _top_k_sparsify(vec: Optional[List[float]], k: int = 3) -> Optional[List[float]]:
    """只留绝对值最大的 k 维再归一：OOV 投影是拼出来的，稀一点更不容易乱指。"""
    if not vec:
        return None
    idx = sorted(range(len(vec)), key=lambda i: (-abs(vec[i]), i))[:k]
    keep = set(idx)
    out = [v if i in keep else 0.0 for i, v in enumerate(vec)]
    return _norm(out)


class P21Engine:
    """一个只读的同构空间。向量在内存里存成 list[float]，657×12 很小。"""

    def __init__(self, payload: dict):
        self.meta: dict = payload.get("meta") or {}
        self.axes_audit: List[dict] = payload.get("axes_audit") or []
        self.slots: Dict[str, str] = payload.get("slots") or {}
        raw = payload.get("vectors") or {}
        self._raw: Dict[str, List[float]] = {w: [float(x) for x in v] for w, v in raw.items()}
        self._unit: Dict[str, List[float]] = {}
        for w, v in self._raw.items():
            u = _norm(v)
            if u is not None:
                self._unit[w] = u
        self.dim = len(next(iter(self._raw.values()), []) or [])
        # 相位带宽（阶段 D）：字粒在语料里的圆周均值 μ 与可靠性 r。
        # 缺失时 align 完全退回字库自带 phase_deg，行为与基线一致。
        self.bands: Dict[str, dict] = {}
        try:
            bp = BANDWIDTH_PATH
            if bp.exists():
                self.bands = (json.loads(bp.read_text(encoding="utf-8"))
                              .get("bands") or {})
        except (OSError, json.JSONDecodeError):
            self.bands = {}
        # 通用先验：全部动作字粒的单位向量均值。"语料里没见过的词"至少落在
        # 动作语义的重心方向上，比空向量（等于放弃判断）稳。
        acts = [self._unit[w] for w, slot in self.slots.items()
                if slot == "action" and w in self._unit]
        self._generic = _mean_vec([(v, 1.0) for v in acts])

    # ---------- 基础查询 ----------
    def vec(self, word: str) -> Optional[List[float]]:
        return self._unit.get(word) or (self._raw.get(word) if word in self._raw else None)

    def has(self, word: str) -> bool:
        return word in self._raw

    def aligned(self, word: str, phase: float, target: float) -> float:
        """带宽对齐：按出现频率把字库的首次相位替换成语料里的平均相位。

        r 是可靠性（集中度² × n/(n+10)）。r=1 时完全用 μ；r=0 时退回传入的
        phase（字库值）。返回 [-1,1]，与原来的余弦同量纲，可直接替换。
        """
        band = self.bands.get(word)
        if not band:
            return math.cos(math.radians(phase - target))
        r = float(band.get("reliability") or 0.0)
        if r <= 0.0:
            return math.cos(math.radians(phase - target))
        mu = float(band.get("mu") or phase)
        base = math.cos(math.radians(phase - target))
        smooth = math.cos(math.radians(mu - target))
        return (1.0 - r) * base + r * smooth

    def cos(self, a: str, b: str) -> Optional[float]:
        return _cos(self.vec(a), self.vec(b))

    def mechanism(self, action: str, consequence: str) -> Optional[List[float]]:
        """动作 + 后果合成"机制向量"：为什么这件事会这样发生。"""
        av, cv = self.vec(action), self.vec(consequence)
        if av is None and cv is None:
            return None
        if av is None:
            return cv
        if cv is None:
            return av
        return _norm([0.45 * a + 0.55 * c for a, c in zip(av, cv)])

    # ---------- OOV 投影 ----------
    def oov_vector(self, entity: str, actions: Sequence[str] = (),
                   learned: Optional[Dict[str, float]] = None,
                   prototype: str = "") -> Tuple[Optional[List[float]], dict]:
        """问句里语料外的新词 → 四源投影（问句谓语 / 学习边 / 原型 / 主轴锚）。

        权重是固定的，产物进 audit，便于回查这一条路径为什么拿到了这个分数。
        最稀规则：投影后只留 3 维，避免"股票"被拼成四不像的混合体。
        """
        if self.has(entity):
            return self.vec(entity), {"source": "corpus", "sparse": False}
        parts: List[Tuple[Sequence[float], float]] = []
        used: Dict[str, int] = {}
        action_vecs = [(self.vec(a), 1.0) for a in actions if a and self.vec(a)]
        if action_vecs:
            m = _mean_vec(action_vecs)
            if m is not None:
                parts.append((m, 0.45))
                used["actions"] = len(action_vecs)
        if learned:
            edge_vecs = []
            for key, weight in learned.items():
                if not key.startswith(entity + "|"):
                    continue
                aw = key.split("|", 1)[1]
                v = self.vec(aw)
                if v is not None:
                    edge_vecs.append((v, max(0.0, float(weight))))
            m = _mean_vec(edge_vecs)
            if m is not None:
                parts.append((m, 0.30))
                used["edges"] = len(edge_vecs)
        if prototype and self.vec(prototype) is not None:
            parts.append((self.vec(prototype), 0.20))
            used["prototype"] = prototype
        if self._generic is not None:
            parts.append((self._generic, 0.10))
            used["generic"] = 1
        fused = _mean_vec(parts)
        sparse = _top_k_sparsify(fused, 3)
        return sparse, {"source": "oov", "sources": used, "sparse": True,
                        "nonzero": sum(1 for x in (sparse or []) if abs(x) > 1e-9)}

    # ---------- OOV 动词投影 ----------
    def verb_vector(self, verb: str) -> Tuple[Optional[List[float]], dict]:
        """问句谓语不在语料里时，用字面重叠把它放进动作语义里。

        中文动词多是二字词，"下跌/变凉/发霉"这类和已有动作（下沉/变化/生长…）
        常共享一个字。取共享字占比最高的 3 个动作向量的加权均值，再加一点
        通用动作重心，最后 Top-3 稀疏化。没有任何重叠时只留通用重心，
        比空向量（等于放弃）多一点点信息，但权重很低。
        """
        if not verb:
            return None, {"source": "missing"}
        if self.has(verb):
            return self.vec(verb), {"source": "corpus"}
        vset = set(verb)
        cands: List[Tuple[float, str, List[float]]] = []
        for w, slot in self.slots.items():
            if slot != "action" or w not in self._unit:
                continue
            inter = len(vset & set(w))
            if inter:
                cands.append((inter / max(len(verb), len(w)), w, self._unit[w]))
        cands.sort(key=lambda x: (-x[0], x[1]))
        parts: List[Tuple[Sequence[float], float]] = [(vec, wgt) for wgt, _, vec in cands[:3]]
        used: Dict[str, int] = {"overlap": len(parts)}
        if self._generic is not None:
            parts.append((self._generic, 0.15))
            used["generic"] = 1
        fused = _mean_vec(parts)
        sparse = _top_k_sparsify(fused, 3)
        return sparse, {"source": "oov_verb", "sources": used,
                        "overlap": [w for _, w, _ in cands[:3]],
                        "sparse": True,
                        "nonzero": sum(1 for x in (sparse or []) if abs(x) > 1e-9)}

    # ---------- 相容乘子 ----------
    def compat(self, entity_vec: Optional[Sequence[float]], action: str,
               consequence: str) -> dict:

        av, cv = self.vec(action), self.vec(consequence)
        mech = self.mechanism(action, consequence)
        ea = _cos(entity_vec, av)
        ac = _cos(av, cv)
        em = _cos(entity_vec, mech)
        terms = [(ea, 0.40), (ac, 0.30), (em, 0.30)]
        num = sum(v * w for v, w in terms if v is not None)
        den = sum(w for v, w in terms if v is not None)
        score = (num / den) if den else None
        return {
            "entity_action": None if ea is None else round(ea, 4),
            "action_consequence": None if ac is None else round(ac, 4),
            "entity_mechanism": None if em is None else round(em, 4),
            "score": None if score is None else round(score, 4),
        }

    def multiplier(self, entity_vec: Optional[Sequence[float]], action: str,
                   consequence: str) -> Tuple[float, dict]:
        info = self.compat(entity_vec, action, consequence)
        score = info["score"]
        if score is None:
            info["multiplier"] = 1.0
            info["clamped"] = False
            return 1.0, info
        raw = 1.0 + COMPAT_SLOPE * (score - COMPAT_CENTER)
        mult = max(COMPAT_LO, min(COMPAT_HI, raw))
        info["multiplier"] = round(mult, 4)
        info["raw_multiplier"] = round(raw, 4)
        info["clamped"] = abs(mult - raw) > 1e-9
        return mult, info

    # ---------- 段落阶段：类比与收束 ----------
    def analogy_score(self, anchor_entity: str, anchor_mech: Optional[Sequence[float]],
                      cand: dict) -> Optional[float]:
        """远实体 / 近机制：换一个物种、但背后的物理机制相同，才是好类比。"""
        g = cand.get("glyphs") or {}
        ce, ca, cc = g.get("entity"), g.get("action"), g.get("consequence")
        if not (ce and ca and cc):
            return None
        d_entity = self.cos(anchor_entity, ce) if anchor_entity and self.has(ce) else None
        if d_entity is None:
            return None
        cand_mech = self.mechanism(ca, cc)
        s_mech = _cos(anchor_mech, cand_mech)
        if s_mech is None:
            return None
        return (0.35 + 0.65 * (1.0 - (d_entity + 1.0) / 2.0)) * (0.25 + 1.50 * (s_mech + 1.0) / 2.0)

    def closure_vector_sim(self, cand: dict, anchor: Optional[dict]) -> Optional[float]:
        if not anchor:
            return None
        ag, cg = anchor.get("glyphs") or {}, cand.get("glyphs") or {}
        if not (ag.get("entity") and cg.get("entity")):
            return None
        ent = self.cos(ag["entity"], cg["entity"])
        a_mech = self.mechanism(ag.get("action", ""), ag.get("consequence", ""))
        c_mech = self.mechanism(cg.get("action", ""), cg.get("consequence", ""))
        mech = _cos(a_mech, c_mech)
        if ent is None and mech is None:
            return None
        vals = [v for v in (ent, mech) if v is not None]
        return sum(vals) / len(vals)

    # ---------- 诚实诊断（只记录，不改行为） ----------
    def distrust(self, entity: str, action: str, consequence: str,
                 top_m: int = 4) -> dict:
        """正交残差 + Top-M 归一化熵：这条句子的"机制"是不是离题。

        残差 = 后果向量在 [实体, 动作] 张成空间之外的比例；熵 = 载荷分布是否
        只在少数几维上（低熵=方向明确，高熵=哪一维都沾一点，话射得很散）。
        只作日志，阈值要等真实问句分布回来再定。
        """
        ev, av, cv = self.vec(entity), self.vec(action), self.vec(consequence)
        audit = {"residual": None, "entropy": None, "top_m": top_m}
        if cv is None:
            return audit
        basis = [v for v in (ev, av) if v is not None]
        resid = list(cv)
        for b in basis:
            nb = math.sqrt(sum(x * x for x in b))
            if nb <= 1e-12:
                continue
            ub = [x / nb for x in b]
            proj = _dot(resid, ub)
            resid = [r - proj * u for r, u in zip(resid, ub)]
        nc = math.sqrt(sum(x * x for x in cv))
        nr = math.sqrt(sum(x * x for x in resid))
        if nc > 1e-12:
            audit["residual"] = round(nr / nc, 4)
        sq = [x * x for x in cv]
        total = sum(sq) or 1.0
        top = sorted(sq, reverse=True)[:top_m]
        probs = [s / total for s in top if s > 0]
        if probs:
            audit["entropy"] = round(-sum(p * math.log(p) for p in probs)
                                      / math.log(max(2, top_m)), 4)
        return audit


class P21Context:
    """一次打印内的同构上下文：缓存 OOV 投影，保证同实体同乘子。"""

    def __init__(self, engine: P21Engine, user_entities: Sequence[str] = (),
                 user_verbs: Sequence[str] = (), learned: Optional[Dict[str, float]] = None,
                 prototypes: Optional[Dict[str, str]] = None):
        self.engine = engine
        self.user_entities = list(user_entities)
        self.user_verbs = list(user_verbs)
        self.learned = learned or {}
        self.prototypes = prototypes or {}
        self._vec_cache: Dict[str, Tuple[Optional[List[float]], dict]] = {}
        self._verb_cache: Dict[str, Tuple[Optional[List[float]], dict]] = {}

    def entity_vector(self, word: str) -> Tuple[Optional[List[float]], dict]:
        if word in self._vec_cache:
            return self._vec_cache[word]
        proto = ""
        if not self.engine.has(word):
            try:
                from plugins.feynman_typesetter import resolve_entity_prototype
                _, proto = resolve_entity_prototype(word, self.prototypes)
            except Exception:  # 原型解析失败不影响主链路，按无原型处理
                proto = ""
        vec, info = self.engine.oov_vector(word, self.user_verbs, self.learned, proto)
        if proto:
            info["prototype"] = proto
        self._vec_cache[word] = (vec, info)
        return vec, info

    def multiplier(self, word: str, action: str, consequence: str) -> Tuple[float, dict]:
        vec, info = self.entity_vector(word)
        mult, audit = self.engine.multiplier(vec, action, consequence)
        audit["entity_source"] = info
        return mult, audit

    def verb_vector(self, word: str) -> Tuple[Optional[List[float]], dict]:
        if word in self._verb_cache:
            return self._verb_cache[word]
        vec, info = self.engine.verb_vector(word)
        self._verb_cache[word] = (vec, info)
        return vec, info

    def query_mechanism(self, entity_word: str) -> Tuple[Optional[List[float]], dict]:
        """问句语义中心：实体向量（OOV 走投影）⊕ 谓语向量（OOV 走字面重叠）。"""
        if entity_word:
            ev, einfo = self.entity_vector(entity_word)
        else:
            ev, einfo = None, {"source": "missing"}
        vparts: List[Tuple[Sequence[float], float]] = []
        verb_sources: Dict[str, str] = {}
        for v in self.user_verbs[:3]:
            vec, info = self.verb_vector(v)
            if vec is not None:
                vparts.append((vec, 1.0))
                verb_sources[v] = info.get("source", "")
        qv = _mean_vec(vparts)
        if ev is None and qv is None:
            return None, {"source": "empty", "entity": einfo.get("source"),
                          "verbs": verb_sources}
        if qv is None:
            fused = ev
        elif ev is None:
            fused = qv
        else:
            fused = _norm([0.55 * a + 0.45 * b for a, b in zip(ev, qv)])
        return fused, {"entity": einfo.get("source"), "verbs": verb_sources}

    def rerank_pool(self, paths: Sequence[dict], anchor: str = "") -> dict:
        """池内重排（阶段 C2）：全局机制项 + 相容分位乘子。

        以前的相容乘子中心 0.25 是手挑的，PPMI 余弦的绝对尺度又不可比，
        结果全池 96.7% 的路径都挨罚。这里改成池内分位：

        - 全局项：每条路径乘 exp(λ·(pct−0.5))，pct 是它与问句语义中心
          （实体⊕谓语）的机制相似度在本次候选池里的分位，λ=0.18，
          影响夹在 [0.91, 1.09]，对所有路径生效（不只 OOV 实体）。
        - 相容项：`_evaluate_path` 里只记录 score 不乘；这里按同一批
          user-entity 路径的分位给乘子：1.0 + 1.2·(pct−0.5)，夹到
          [0.6, 1.6]。池里只有一条时 pct=0.5，乘子恰好 1.0，
          不再出现"单条路径被固定中心压低"的无依据惩罚。

        全部确定性：分位用平均秩处理并列，与路径顺序无关。
        """
        pool_stats = {"paths": len(paths), "sim_scored": 0, "compat_scored": 0}
        qvec, qinfo = self.query_mechanism(anchor)
        pool_stats["query"] = qinfo
        vvec_parts = [(self.verb_vector(v)[0], 1.0) for v in self.user_verbs[:3]
                      if self.verb_vector(v)[0] is not None]
        vvec = _mean_vec(vvec_parts)

        if qvec is not None:
            sims: List[Tuple[int, float]] = []
            for i, p in enumerate(paths):
                g = p.get("glyphs") or {}
                mech = self.engine.mechanism(g.get("action", ""), g.get("consequence", ""))
                s1 = _cos(qvec, mech)
                s2 = _cos(vvec, self.engine.vec(g.get("action", ""))) if vvec else None
                terms = [(s1, 0.60), (s2, 0.40)]
                num = sum(v * w for v, w in terms if v is not None)
                den = sum(w for v, w in terms if v is not None)
                if not den:
                    continue
                p["p21_sim"] = num / den
                sims.append((i, p["p21_sim"]))
            if sims:
                ordered = sorted(s for _, s in sims)
                n = len(ordered)
                for i, s in sims:
                    pct = _avg_percentile(ordered, s)
                    mult = math.exp(P21_POOL_LAMBDA * (pct - 0.5))
                    p["weight"] *= mult
                    p["p21_pool"] = {"sim": round(s, 4),
                                     "percentile": round(pct, 4),
                                     "multiplier": round(mult, 4)}
                pool_stats["sim_scored"] = n

        compat: List[Tuple[int, float]] = []
        for i, p in enumerate(paths):
            info = p.get("p21")
            if info and info.get("score") is not None:
                compat.append((i, float(info["score"])))
        if compat:
            ordered = sorted(s for _, s in compat)
            n = len(ordered)
            lo = COMPAT_PERC_LO
            hi = COMPAT_PERC_HI
            for i, s in compat:
                pct = _avg_percentile(ordered, s)
                mult = max(lo, min(hi, 1.0 + 1.2 * (pct - 0.5)))
                paths[i]["weight"] *= mult
                info = paths[i]["p21"]
                info["percentile"] = round(pct, 4)
                info["percentile_pool"] = n
                info["multiplier"] = round(mult, 4)
            pool_stats["compat_scored"] = n
        return pool_stats


def load_engine(path: Optional[Path] = None) -> Optional[P21Engine]:
    """读产物并缓存；文件缺失/损坏时返回 None（调用方退回基线离散惩罚）。"""
    p = Path(path or DEFAULT_PATH)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    key = str(p.resolve())
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    engine = P21Engine(payload)
    _cache[key] = (mtime, engine)
    return engine


__all__ = ["P21Engine", "P21Context", "load_engine", "DEFAULT_PATH",
           "COMPAT_CENTER", "COMPAT_SLOPE", "COMPAT_LO", "COMPAT_HI"]
