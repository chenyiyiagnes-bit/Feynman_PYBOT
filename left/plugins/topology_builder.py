"""语义流形与拓扑层：PPMI + 确定性 SVD + 语料同句三角形 + 类比空间正交传输。

和 PYBOT4 草图的三处不同：
- 不依赖 scipy：字库词表约 200 词，稠密 `numpy.linalg.svd` 是毫秒级，`svds` 没必要；
- 2-simplex 不按"任意三词共现"硬凑，只认语料里同一句话同时出现的
  (实体, 动作, 后果) 三元组——没有同句证据的三角形才是"洞"；
- `topo_hash` 只承诺同机重跑一致：对量化到 4 位小数的向量矩阵与排序后的
  三角形做 sha256，不宣称跨平台字节一致（SVD 浮点不同平台会有末位差异）。

本模块只读字库与语料，产出 `data/topology_graph.npz`；运行时由排版引擎按需加载。
"""

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from plugins.feynman_vault import CORPUS_PATH, VAULT_PATH, iter_sections, sentences

TOPOLOGY_PATH = Path(os.getenv("QBOT_FEYNMAN_TOPOLOGY", "data/topology_graph.npz"))
DIM = 64
SIMPLEX_BOOST = 1.45          # 同句三元组：最硬的证据
WINDOW2_BOOST = 1.20          # 相邻两句窗：同一段话里出现
WINDOW3_BOOST = 1.10          # 三句窗：更弱的段落证据
SECTION_BOOST = 1.08          # 同一节文本共现：不限句距，弱于三句窗
PASSAGE_BOOST = 1.15          # 字粒出处（段落键）相交：同一节/同一讲
HOLE_FACTOR = 0.55            # 三对边都强、却从未同处：真正的"空转三角"
HOLE_MIN_SUPPORT = 2          # 低于此支撑的洞不罚：没有证据不等于反证据
# 三对边的分级共现证据：三元组在语料里太稀疏（p8b 普通候选池 0 命中），
# 但"实体-动作""动作-后果"等成对共现是真实存在的。共现越强越可信，
# 用对数压缩后做 ±10% 的倾斜（PAIR_TILT），没有一对共现时保持中性 1.0。
PAIR_TILT = 0.20              # 乘子范围 [1−0.1, 1+0.1]
PAIR_REF = 20                 # 达到这个无向支撑数视为满分证据
GEO_GAIN = 0.60
GEO_CLAMP = (0.25, 1.80)
ANALOGY_MARKERS = ("就像", "好比", "好像", "如同", "仿佛")
MIN_ANALOGY_PAIRS = 30
SLOT_KEYS = ("hooks", "entities", "actions", "consequences")


def _vault_words(vault: dict) -> List[str]:
    words = {g["word"] for slot in SLOT_KEYS for g in (vault.get(slot) or [])}
    return sorted(words)


def _frequencies(vault: dict) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for slot in SLOT_KEYS:
        for g in vault.get(slot) or []:
            freq[g["word"]] = max(freq.get(g["word"], 1), int(g.get("count", 1)))
    return freq


def _ppmi_matrix(vault: dict, words: List[str]) -> np.ndarray:
    """共现对的 PPMI：log( P(a,b) / (P(a)·P(b)) )，负值归零。

    P(a,b) 用共现对计数 / 共现总质量；P(a) 用字粒在语料里的计数 / 字粒总频次。
    """
    idx = {w: i for i, w in enumerate(words)}
    n = len(words)
    freq = _frequencies(vault)
    pairs = vault.get("cooccurrence") or {}
    total_pairs = float(sum(pairs.values()) or 1)
    total_freq = float(sum(freq.values()) or 1)
    mat = np.zeros((n, n), dtype=np.float64)
    for key, count in pairs.items():
        a, _, b = key.partition("|")
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia == ib or count <= 0:
            continue
        fa, fb = freq.get(a, 1), freq.get(b, 1)
        ratio = (count / total_pairs) / ((fa / total_freq) * (fb / total_freq))
        mat[ia, ib] = mat[ib, ia] = max(0.0, math.log(ratio))
    return mat


def _embed(mat: np.ndarray, dim: int = DIM) -> np.ndarray:
    """确定性 SVD 嵌入；逐列定符号，避免同一矩阵两次分解符号翻转。"""
    if not np.any(mat):
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    u, s, _vt = np.linalg.svd(mat, full_matrices=False)
    k = int(min(dim, len(s)))
    vecs = u[:, :k] * np.sqrt(s[:k])
    for j in range(k):
        col = vecs[:, j]
        total = float(col.sum())
        if abs(total) < 1e-12:
            # 和为零时退回绝对值最大分量定符号，仍保持确定性
            total = float(col[int(np.argmax(np.abs(col)))])
        if total < 0:
            vecs[:, j] = -col
    return np.round(vecs, 4)


def _first_marker(sent: str) -> Optional[Tuple[int, str]]:
    best: Optional[Tuple[int, str]] = None
    for m in ANALOGY_MARKERS:
        i = sent.find(m)
        if i < 0:
            continue
        if best is None or i < best[0] or (i == best[0] and len(m) > len(best[1])):
            best = (i, m)
    return best


def collect_tiered_triples(vault: dict, corpus_path: Path = CORPUS_PATH
                           ) -> Dict[int, Counter]:
    """三级同处证据：1=同句，2=相邻两句窗，3=三句窗。

    实测 p8b：同句三元组只有 31 个（66×60×60 的 0.013%），候选路径几乎从不命中；
    放宽到两句/三句窗后分别到约 100/162 个，虽然仍稀疏，但至少给"段落级证据"
    留了位置。级别越高权重越低，见 *_BOOST 常数。
    """
    ents = [g["word"] for g in vault.get("entities") or []]
    acts = [g["word"] for g in vault.get("actions") or []]
    cons = [g["word"] for g in vault.get("consequences") or []]
    tiers: Dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}

    def _add(window: str, tier: int) -> None:
        e_hit = [w for w in ents if w in window][:3]
        a_hit = [w for w in acts if w in window][:2]
        c_hit = [w for w in cons if w in window][:2]
        for e in e_hit:
            for a in a_hit:
                for c in c_hit:
                    tiers[tier][(e, a, c)] += 1

    for _src, _section, text in iter_sections(corpus_path):
        sents = sentences(text)
        for i in range(len(sents)):
            for tier, width in ((1, 1), (2, 2), (3, 3)):
                if i + width <= len(sents):
                    _add("，".join(sents[i:i + width]), tier)
    return tiers


def collect_section_triples(vault: dict, corpus_path: Path = CORPUS_PATH
                            ) -> Counter:
    """段落级共现：三个槽的字粒出现在同一节文本里，不限句距。

    p8b 实测：同句 31、两句窗 103、三句窗 168、出处相交 72，而同一节文本
    共现约 2400 个；480 条候选路径里前三层命中为 0，段落层命中 51 条
    （42/48 个查询至少有一条）。后果字粒多是整句短语，跨句共现是常态，
    只认同句等于永远不给正证据。

    每节每槽最多取 5 个词，只为限制组合爆炸；共现是文本事实，不做语义推断。
    """
    ents = [g["word"] for g in vault.get("entities") or []]
    acts = [g["word"] for g in vault.get("actions") or []]
    cons = [g["word"] for g in vault.get("consequences") or []]
    triples: Counter = Counter()
    for _src, _section, text in iter_sections(corpus_path):
        e_hit = [w for w in ents if w in text][:5]
        if not e_hit:
            continue
        a_hit = [w for w in acts if w in text][:5]
        if not a_hit:
            continue
        c_hit = [w for w in cons if w in text][:5]
        if not c_hit:
            continue
        for e in e_hit:
            for a in a_hit:
                for c in c_hit:
                    triples[(e, a, c)] += 1
    return triples


def _passage_key(source: str) -> str:
    """出处键的段落粒度：4 段（集成库｜类型｜文档｜文档）取前三段为文档；
    其余（文件名｜章节 / 书｜句子号）整串作为段落键。"""
    parts = source.split("｜")
    return "｜".join(parts[:3]) if len(parts) >= 4 else source


def collect_passage_triples(vault: dict) -> Counter:
    """字粒出处相交的三元组：三个槽的字粒至少同处过一个段落（同一节/同一讲）。"""
    ents = [g["word"] for g in vault.get("entities") or []]
    acts = [g["word"] for g in vault.get("actions") or []]
    cons = [g["word"] for g in vault.get("consequences") or []]
    src: Dict[str, set] = {}
    for slot in SLOT_KEYS:
        for g in vault.get(slot) or []:
            src.setdefault(g["word"], set()).update(
                _passage_key(s) for s in g.get("sources") or [])
    triples: Counter = Counter()
    for e in ents:
        se = src.get(e) or set()
        if not se:
            continue
        for a in acts:
            sa = se & (src.get(a) or set())
            if not sa:
                continue
            for c in cons:
                common = sa & (src.get(c) or set())
                if common:
                    triples[(e, a, c)] = len(common)
    return triples


def collect_triangles(vault: dict, corpus_path: Path = CORPUS_PATH
                      ) -> Tuple[Counter, Counter]:
    """同句三角形与类比名词对。只统计字库里已有的字粒，不猜语义。

    三角形：一句话里同时出现 (实体, 动作, 后果) 才算实心——这是最硬的证据。
    类比对：含"就像/好比/如同"的句子里，标记左右两侧各取实体字粒配对，
    给 Procrustes 训练"左边的东西 → 右边的东西"的向量传输。
    """
    ents = [g["word"] for g in vault.get("entities") or []]
    acts = [g["word"] for g in vault.get("actions") or []]
    cons = [g["word"] for g in vault.get("consequences") or []]
    triples: Counter = Counter()
    pairs: Counter = Counter()
    for _src, _section, text in iter_sections(corpus_path):
        for sent in sentences(text):
            e_hit = [w for w in ents if w in sent][:3]
            a_hit = [w for w in acts if w in sent][:2]
            c_hit = [w for w in cons if w in sent][:2]
            for e in e_hit:
                for a in a_hit:
                    for c in c_hit:
                        triples[(e, a, c)] += 1
            found = _first_marker(sent)
            if not found:
                continue
            idx, marker = found
            left, right = sent[:idx], sent[idx + len(marker):]
            l_words = [w for w in ents if w in left][:3]
            r_words = [w for w in ents if w in right][:3]
            for lw in l_words:
                for rw in r_words:
                    if lw != rw:
                        pairs[(lw, rw)] += 1
    return triples, pairs


def train_procrustes(vectors: np.ndarray, idx: Dict[str, int],
                     pairs: Counter) -> Tuple[Optional[np.ndarray], float, int]:
    """类比空间正交传输矩阵 R：加权正交 Procrustes，det=+1。样本不足返回 None。"""
    xs, ys, ws = [], [], []
    for (left, right), count in pairs.items():
        il, ir = idx.get(left), idx.get(right)
        if il is None or ir is None or count <= 0:
            continue
        if not np.any(vectors[il]) or not np.any(vectors[ir]):
            continue
        xs.append(vectors[il])
        ys.append(vectors[ir])
        ws.append(float(count))
    n_pairs = len(xs)
    if n_pairs < MIN_ANALOGY_PAIRS:
        return None, 0.0, n_pairs
    x = np.asarray(xs)
    y = np.asarray(ys)
    w = np.asarray(ws)
    w = w / w.sum()
    cross = (y * w[:, None]).T @ x
    u, _s, vt = np.linalg.svd(cross)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    alpha = float(min(0.85, n_pairs / (n_pairs + 15.0)))
    return np.round(r, 4), alpha, n_pairs


def _content_hash(vectors: np.ndarray, tiers: Dict[int, Counter],
                  passages: Counter, r: Optional[np.ndarray],
                  sections: Optional[Counter] = None) -> str:
    h = hashlib.sha256()
    h.update(np.round(vectors, 4).astype(np.float32).tobytes())
    for tier in sorted(tiers):
        for key in sorted(tiers[tier]):
            h.update(f"{tier}|{key[0]}|{key[1]}|{key[2]}:{tiers[tier][key]}".encode("utf-8"))
    for key in sorted(passages):
        h.update(f"p|{key[0]}|{key[1]}|{key[2]}".encode("utf-8"))
    for key in sorted(sections or ()):
        h.update(f"s|{key[0]}|{key[1]}|{key[2]}".encode("utf-8"))
    if r is not None:
        h.update(np.round(r, 4).astype(np.float32).tobytes())
    return h.hexdigest()[:8]


def _undirected_edge_counts(vault: dict) -> Dict[Tuple[str, str], int]:
    """把有向共现键合成无向边计数：空洞判定只看"这对词有多熟"。"""
    counts: Dict[Tuple[str, str], int] = {}
    for key, count in (vault.get("cooccurrence") or {}).items():
        if "|" not in key:
            continue
        a, b = key.split("|", 1)
        if a == b:
            continue
        edge = (a, b) if a <= b else (b, a)
        counts[edge] = counts.get(edge, 0) + int(count)
    return counts


def _triple_rows(counter: Counter) -> np.ndarray:
    keys = sorted(counter)
    return (np.asarray([list(k) for k in keys], dtype=str) if keys
            else np.zeros((0, 3), dtype=str))


def build_topology(vault_path: Path = VAULT_PATH, corpus_path: Path = CORPUS_PATH,
                   out_path: Path = TOPOLOGY_PATH) -> dict:
    vault = json.loads(Path(vault_path).read_text(encoding="utf-8"))
    words = _vault_words(vault)
    vecs = _embed(_ppmi_matrix(vault, words))
    tiers = collect_tiered_triples(vault, corpus_path)
    passages = collect_passage_triples(vault)
    sections = collect_section_triples(vault, corpus_path)
    _same_sentence, pairs = collect_triangles(vault, corpus_path)
    idx = {w: i for i, w in enumerate(words)}
    r, alpha, n_pairs = train_procrustes(vecs, idx, pairs)
    topo_hash = _content_hash(vecs, tiers, passages, r, sections)
    edge_counts = _undirected_edge_counts(vault)

    tri_keys = sorted(tiers[1])
    meta = {
        "topo_hash": topo_hash,
        "dim": int(vecs.shape[1]),
        "words": len(words),
        "triangles": len(tri_keys),
        "window2_triangles": len(tiers[2]),
        "window3_triangles": len(tiers[3]),
        "section_triangles": len(sections),
        "passage_triangles": len(passages),
        "edges": len(edge_counts),
        "edge_support_ge2": sum(1 for c in edge_counts.values() if c >= HOLE_MIN_SUPPORT),
        "hole_min_support": HOLE_MIN_SUPPORT,
        "analogy_pairs": n_pairs,
        "analogy_alpha": round(alpha, 4),
        "corpus": str(corpus_path),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tier2_keys = sorted(tiers[2])
    tier3_keys = sorted(tiers[3])
    section_keys = sorted(sections)
    passage_keys = sorted(passages)
    edge_keys = sorted(edge_counts)
    np.savez(
        out_path,
        words=np.asarray(words, dtype=str),
        vectors=np.asarray(vecs, dtype=np.float32),
        simplex_words=_triple_rows(tiers[1]),
        simplex_counts=np.asarray([tiers[1][k] for k in tri_keys], dtype=np.int32),
        tier2_words=_triple_rows(tiers[2]),
        tier3_words=_triple_rows(tiers[3]),
        section_words=_triple_rows(sections),
        passage_words=_triple_rows(passages),
        edge_words=np.asarray([list(e) for e in edge_keys], dtype=str)
        if edge_keys else np.zeros((0, 2), dtype=str),
        edge_counts=np.asarray([edge_counts[e] for e in edge_keys], dtype=np.int32),
        analogy_r=np.asarray(r if r is not None else np.zeros((0, 0)), dtype=np.float32),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    return meta


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class TopologyManifoldEngine:
    """只读拓扑图：查询字粒向量、三角形证据与类比空间对齐度。"""

    def __init__(self, words: List[str], vectors: np.ndarray,
                 simples: Dict[Tuple[str, str, str], int],
                 edges: set,
                 analogy_r: Optional[np.ndarray], analogy_alpha: float,
                 analogy_pairs: int, topo_hash: str,
                 simplex_tiers: Optional[Dict[Tuple[str, str, str], int]] = None,
                 passage_simplexes: Optional[set] = None,
                 edge_counts: Optional[Dict[Tuple[str, str], int]] = None,
                 section_simplexes: Optional[set] = None):
        self.words = list(words)
        self._idx = {w: i for i, w in enumerate(words)}
        self._vectors = vectors
        self._unit = _unit_rows(vectors) if len(vectors) else vectors
        self.simples = simples
        self.simplex_tiers = dict(simplex_tiers or {k: 1 for k in simples})
        self.passage_simplexes = set(passage_simplexes or ())
        self.section_simplexes = set(section_simplexes or ())
        self._edges = set(edges)
        self._edge_counts = {(a, b) if a <= b else (b, a): int(c)
                             for (a, b), c in (edge_counts or {}).items()}
        self._r = analogy_r
        self.analogy_alpha = analogy_alpha
        self.analogy_pairs = analogy_pairs
        self.topo_hash = topo_hash

    @classmethod
    def load(cls, path: Path = TOPOLOGY_PATH) -> Optional["TopologyManifoldEngine"]:
        p = Path(path)
        if not p.is_file():
            return None
        data = np.load(p, allow_pickle=False)
        words = [str(w) for w in data["words"]]
        vectors = data["vectors"]
        simplex_words = data["simplex_words"]
        counts = data["simplex_counts"]
        simples = {}
        if len(simplex_words):
            for row, c in zip(simplex_words, counts):
                simples[(str(row[0]), str(row[1]), str(row[2]))] = int(c)
        tiers: Dict[Tuple[str, str, str], int] = {k: 1 for k in simples}
        for tier, key in ((2, "tier2_words"), (3, "tier3_words")):
            if key in data:
                for row in data[key]:
                    tiers.setdefault((str(row[0]), str(row[1]), str(row[2])), tier)
        sections = set()
        if "section_words" in data:
            for row in data["section_words"]:
                sections.add((str(row[0]), str(row[1]), str(row[2])))
        passages = set()
        if "passage_words" in data:
            for row in data["passage_words"]:
                passages.add((str(row[0]), str(row[1]), str(row[2])))
        edge_words = data["edge_words"] if "edge_words" in data else np.zeros((0, 2), dtype=str)
        edges = {(str(row[0]), str(row[1])) for row in edge_words} if len(edge_words) else set()
        edge_counts: Dict[Tuple[str, str], int] = {}
        if "edge_counts" in data and len(edge_words):
            for row, c in zip(edge_words, data["edge_counts"]):
                edge_counts[(str(row[0]), str(row[1]))] = int(c)
        raw_r = data["analogy_r"]
        analogy_r = raw_r if raw_r.size else None
        meta = json.loads(str(data["meta"].item()))
        return cls(words, vectors, simples, edges, analogy_r,
                   float(meta.get("analogy_alpha", 0.0)),
                   int(meta.get("analogy_pairs", 0)),
                   str(meta.get("topo_hash", "00000000")),
                   simplex_tiers=tiers, passage_simplexes=passages,
                   edge_counts=edge_counts, section_simplexes=sections)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        i = self._idx.get(word)
        if i is None:
            return None
        v = self._unit[i]
        if not np.any(v):
            return None
        return v

    def covers(self, *words: str) -> bool:
        """四个槽的词都有非零向量：池内归一只能比较这些路径。"""
        return all(self.get_vector(w) is not None for w in words)

    def has_simplex(self, entity: str, action: str, consequence: str) -> bool:
        key = (entity, action, consequence)
        return (key in self.simplex_tiers or key in self.section_simplexes
                or key in self.passage_simplexes)

    def _simplex_boost(self, key: Tuple[str, str, str]) -> Optional[float]:
        tier = self.simplex_tiers.get(key)
        if tier == 1:
            return SIMPLEX_BOOST
        if tier == 2:
            return WINDOW2_BOOST
        if tier == 3:
            return WINDOW3_BOOST
        if key in self.section_simplexes:
            return SECTION_BOOST
        if key in self.passage_simplexes:
            return PASSAGE_BOOST
        return None

    def pair_evidence(self, entity: str, action: str, consequence: str) -> dict:
        """三对边的分级共现证据（对数压缩）：只奖励、不惩罚。

        三元组在语料里太稀疏，但成对共现是真实数据。用三条边支撑数的均值
        经 log1p 压缩到 [0,1]；score≤0.5 视为弱证据，倾斜恰好为 1.0，
        不给"没有证据"的路径扣分；score>0.5 最多给 +10% 权重。
        """
        supports = [self._edge_support(entity, action),
                    self._edge_support(entity, consequence),
                    self._edge_support(action, consequence)]
        mean_support = sum(supports) / 3.0
        score = 0.0
        if mean_support > 0:
            score = min(1.0, math.log1p(mean_support) / math.log1p(PAIR_REF))
        tilt = 1.0 + PAIR_TILT * max(0.0, score - 0.5)
        return {"supports": supports, "score": round(score, 4), "tilt": round(tilt, 4)}

    def compute_geodesic_multiplier(self, hook: str, entity: str, action: str,
                                    consequence: str,
                                    beta_curvature: float = 0.35) -> Tuple[float, bool]:
        """S = Σ‖Δ‖² + βΣ(1−cos)；乘数 = exp(−gain·平均项)，再乘证据分级因子。"""
        points = [v for v in (self.get_vector(hook), self.get_vector(entity),
                              self.get_vector(action), self.get_vector(consequence))
                  if v is not None]
        mult = 1.0
        if len(points) >= 2:
            steps = [points[i + 1] - points[i] for i in range(len(points) - 1)]
            energy = sum(float(np.dot(d, d)) for d in steps)
            if len(steps) >= 2:
                energy += beta_curvature * sum(
                    1.0 - float(np.dot(steps[i], steps[i + 1])
                                / (np.linalg.norm(steps[i]) * np.linalg.norm(steps[i + 1]) + 1e-9))
                    for i in range(len(steps) - 1))
            terms = len(steps) + max(0, len(steps) - 1)
            mult = math.exp(-GEO_GAIN * energy / max(1, terms))
        covered = all(w in self._idx for w in (entity, action, consequence))
        boost = self._simplex_boost((entity, action, consequence)) if covered else None
        is_simplex = bool(covered and boost is not None)
        if boost is not None:
            mult *= boost
        elif covered and self._is_hole(entity, action, consequence):
            # 三对边都够强、却从未同处：典型的空转三角形
            mult *= HOLE_FACTOR
        # 成对共现是真实语料证据：强于中位时给小幅加成，弱/无证据保持中性
        mult *= self.pair_evidence(entity, action, consequence)["tilt"]
        return max(GEO_CLAMP[0], min(GEO_CLAMP[1], mult)), is_simplex

    def _edge_support(self, a: str, b: str) -> int:
        if self._edge_counts:
            return int(self._edge_counts.get((a, b) if a <= b else (b, a), 0))
        # 旧图没有计数：三边齐全按 1 处理，达不到 HOLE_MIN_SUPPORT，保持中性
        return 1 if ((a, b) in self._edges or (b, a) in self._edges) else 0

    def _is_hole(self, entity: str, action: str, consequence: str) -> bool:
        pairs = ((entity, action), (entity, consequence), (action, consequence))
        return all(self._edge_support(a, b) >= HOLE_MIN_SUPPORT for a, b in pairs)

    def compute_analogy_alignment(self, entity: str, consequence: str) -> Optional[float]:
        """把实体向量平行传输到类比空间，再与后果向量算余弦；没有训练矩阵返回 None。"""
        if self._r is None or self._r.size == 0:
            return None
        ve, vc = self.get_vector(entity), self.get_vector(consequence)
        if ve is None or vc is None:
            return None
        moved = self.analogy_alpha * (self._r @ ve) + (1.0 - self.analogy_alpha) * ve
        norm = float(np.linalg.norm(moved))
        if norm == 0:
            return None
        return float(np.dot(moved / norm, vc))

    def stats(self) -> dict:
        tiers = Counter(self.simplex_tiers.values())
        return {
            "topo_hash": self.topo_hash,
            "words": len(self.words),
            "triangles": len(self.simples),
            "window2_triangles": tiers.get(2, 0),
            "window3_triangles": tiers.get(3, 0),
            "section_triangles": len(self.section_simplexes),
            "passage_triangles": len(self.passage_simplexes),
            "edges": len(self._edges),
            "edge_support_ge2": sum(1 for c in self._edge_counts.values()
                                    if c >= HOLE_MIN_SUPPORT),
            "analogy_pairs": self.analogy_pairs,
            "analogy_alpha": round(self.analogy_alpha, 4),
        }


def enabled() -> bool:
    return os.getenv("FEYNMAN_ENABLE_TOPOLOGY", "0").strip() == "1"


_ENGINE: Optional[TopologyManifoldEngine] = None


def get_engine(force: bool = False) -> Optional[TopologyManifoldEngine]:
    """按开关懒加载；关闭或文件缺失返回 None，排版行为与基线一致。

    force=True 用于请求级开关（PrintRequest.use_topology）：即便 env 没开，
    用户显式打开拓扑层也允许加载；默认仍尊重 FEYNMAN_ENABLE_TOPOLOGY。
    """
    global _ENGINE
    if not (force or enabled()):
        return None
    if _ENGINE is None:
        _ENGINE = TopologyManifoldEngine.load()
    return _ENGINE


if __name__ == "__main__":
    meta = build_topology()
    print("✅ 拓扑图已生成：", TOPOLOGY_PATH)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
