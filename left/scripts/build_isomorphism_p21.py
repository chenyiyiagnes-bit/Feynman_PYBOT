"""P21 同构基底（离线）：句级共现 → PPMI → 截断 SVD → 固定符号的实数同构向量。

只读语料与字库，产出 `data/isomorphism_p21.json`。运行时只读这个产物，
不重新训练、不联网、不调用 LLM。构建是确定性的：同一语料 + 同一字库
两次构建逐字节相同（`--check` 用重建结果比对现有文件）。

设计取舍（见 docs/p21-adjudication.md）：
- 单一联合矩阵：658 个字粒同处一个空间，不比 Gemini 提案的两矩阵共享轴更好实现，
  但避免了"实体空间/动作空间"两条坐标轴不可比的问题；
- 共现取句级（同一句里出现即算），不是 ±5 字窗口——窗口臂在 P8 语料里
  会把"电子旋转得很快"的"很快"也算进去，句级更接近"同一物理情景"；
- PPMI 用 0.75 次幂平滑边缘分布（上下文分布平滑），罕用字不致于被饿死；
- 截断 SVD 用确定性的块幂迭代（纯标准库，无 numpy），固定种子初始化，
  收敛后按"最大载荷为正"逐维翻符号，保证两次构建同号；
- 维数 k 由能量占比自动选（默认累计 55%，下限 6、上限 12），奇异值谱写进 meta。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_isomorphism_p21.py
    PYTHONPATH=. .venv/bin/python -B scripts/build_isomorphism_p21.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

CORPUS = Path("data/p8_corpus_v2.jsonl")
VAULT = Path("data/feynman_style_vault_p15.json")
OUT = Path("data/isomorphism_p21.json")

ALPHA = 0.75          # 边缘分布平滑指数
INIT_SEED = 0x21B07   # 块幂迭代初值种子（固定，保证确定性）
SOLT_ORDER = ("entity", "action", "consequence", "hook")
SLOT_KEYS = {"entity": "entities", "action": "actions",
             "consequence": "consequences", "hook": "hooks"}
# 原型概念词：不在字库里，但语料里出现，用来给问句中的新词做投影锚。
# 取自 plugins/feynman_typesetter.PROTOTYPE_MAP 的 12 个值；已入库的词不重复加。
AUX_WORDS = ("人", "仪器", "动物", "容器", "机器", "气体", "液体",
             "灯", "船", "车", "钟表", "镜")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_glyphs(vault_path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    """字库四槽的并集；同一个词出现在两槽时按 SOLT_ORDER 记主槽。"""
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    slot_of: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for slot in SOLT_ORDER:
        for g in vault.get(SLOT_KEYS[slot], []):
            w = g.get("word")
            if not w:
                continue
            slot_of.setdefault(w, slot)
            counts[slot] = counts.get(slot, 0) + 1
    for w in AUX_WORDS:
        if w not in slot_of:
            slot_of[w] = "aux"
            counts["aux"] = counts.get("aux", 0) + 1
    return sorted(slot_of), slot_of, counts


def sentences(text: str) -> List[str]:
    """与 plugins/feynman_vault.sentences 同规则：句末标点或换行切，15–120 字。"""
    import re
    parts = re.split(r"(?<=[。！？；])|\n+", text)
    return [p.strip(" \t\r\n“”\"'·-—") for p in parts if 15 <= len(p.strip()) <= 120]


def scan(corpus_path: Path, words: List[str]) -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    """返回 (句内共现计数, 字粒出现句数, 句子总数, 语料块数)。

    用首字索引做子串匹配：中文没有空格，658 个字粒逐句 `in` 也可以，
    但首字索引把候选缩到同首字的几个词，语料 12 万块时省一个数量级。
    """
    by_first: Dict[str, List[str]] = {}
    for w in words:
        by_first.setdefault(w[0], []).append(w)
    for cands in by_first.values():
        cands.sort(key=len, reverse=True)  # 长词先匹配，短词照样会单独命中

    pair: Dict[str, int] = {}
    presence: Dict[str, int] = {w: 0 for w in words}
    n_sent = 0
    n_chunk = 0
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_chunk += 1
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            for sent in sentences(text):
                n_sent += 1
                hit = set()
                for ch in sent:
                    for w in by_first.get(ch, ()):
                        if w in sent:
                            hit.add(w)
                if not hit:
                    continue
                for w in hit:
                    presence[w] += 1
                hw = sorted(hit)
                for i in range(len(hw)):
                    a = hw[i]
                    for j in range(i + 1, len(hw)):
                        key = f"{a}|{hw[j]}"
                        pair[key] = pair.get(key, 0) + 1
    return pair, presence, n_sent, n_chunk


def build_ppmi(words: List[str], pair: Dict[str, int], presence: Dict[str, int],
               n_sent: int, alpha: float = ALPHA) -> Tuple[List[List[Tuple[int, float]]], float]:
    """对称 PPMI：pmi(a,b) = log( p(a,b) / (p'(a)·p'(b)) )，负值截 0。

    p' 为平滑后的边缘：presence^alpha 归一。句数作分母；pair 是句内无序对计数。
    返回稀疏行（对称双向存，方便矩阵乘）与 Frobenius 范数平方。
    """
    index = {w: i for i, w in enumerate(words)}
    smoothed = {w: max(0, presence[w]) ** alpha for w in words}
    z = sum(smoothed.values()) or 1.0
    smooth = {w: smoothed[w] / z for w in words}
    total_pairs = sum(pair.values()) or 1
    rows: List[List[Tuple[int, float]]] = [[] for _ in words]
    frob2 = 0.0
    for key, c in pair.items():
        a, b = key.split("|", 1)
        denom = smooth[a] * smooth[b] * total_pairs
        if denom <= 0:
            continue
        v = math.log(c / denom)
        if v <= 0:
            continue
        i, j = index[a], index[b]
        rows[i].append((j, v))
        rows[j].append((i, v))
        frob2 += 2.0 * v * v
    return rows, frob2


def _matmul(rows: List[List[Tuple[int, float]]], X: List[List[float]], k: int) -> List[List[float]]:
    n = len(rows)
    Y = [[0.0] * k for _ in range(n)]
    for i, row in enumerate(rows):
        yi = Y[i]
        for j, val in row:
            xj = X[j]
            for c in range(k):
                yi[c] += val * xj[c]
    return Y


def _orthonormalize(X: List[List[float]], k: int, rng: random.Random) -> List[List[float]]:
    """修正 Gram–Schmidt；数值退化时用确定性的随机向量补齐。"""
    n = len(X)
    cols = [[X[i][c] for i in range(n)] for c in range(k)]
    for c in range(k):
        v = cols[c]
        for _ in range(2):  # 二次正交化，压住浮点误差
            for prev in range(c):
                dot = sum(v[i] * cols[prev][i] for i in range(n))
                if dot:
                    for i in range(n):
                        v[i] -= dot * cols[prev][i]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-9:
            v = [rng.gauss(0.0, 1.0) for _ in range(n)]
            for _ in range(2):
                for prev in range(c):
                    dot = sum(v[i] * cols[prev][i] for i in range(n))
                    for i in range(n):
                        v[i] -= dot * cols[prev][i]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
        cols[c] = [x / norm for x in v]
    return [[cols[c][i] for c in range(k)] for i in range(n)]


def truncated_svd(rows: List[List[Tuple[int, float]]], n: int, k: int,
                  max_iter: int = 400, tol: float = 1e-9) -> Tuple[List[List[float]], List[float], float]:
    """对称矩阵的块幂迭代：返回 (n×k 向量列, k 个奇异值, 最大相对残差)。

    PPMI 对称但不保证半正定；块幂对 |λ| 收敛，借助逐维符号锁把方向固定下来。
    残差用 A v − σ v 的相对范数估计，供 meta 复核迭代是否真的收敛。
    """
    rng = random.Random(INIT_SEED)
    X = [[rng.gauss(0.0, 1.0) for _ in range(k)] for _ in range(n)]
    X = _orthonormalize(X, k, rng)
    prev = [0.0] * k
    for _ in range(max_iter):
        # 迭代 AᵀA：PPMI 对称但不保证半正定，直接在 A 上做块幂会在 |λ| 相近的
        # 正负特征向量之间来回摆（实测残差 2.0）；AᵀA 半正定，收敛稳。
        AX = _matmul(rows, X, k)
        Y = _matmul(rows, AX, k)
        X = _orthonormalize(Y, k, rng)
        sigma = [math.sqrt(sum(AX[i][c] ** 2 for i in range(n))) for c in range(k)]
        if all(abs(sigma[c] - prev[c]) <= tol * max(1.0, sigma[c]) for c in range(k)):
            break
        prev = sigma
    AX = _matmul(rows, X, k)
    residual = 0.0
    lam: List[float] = []
    for c in range(k):
        # PPMI 不半正定：用带符号的瑞利商，残差要跟 λx 比，不是跟 |λ|x 比
        l = sum(X[i][c] * AX[i][c] for i in range(n))
        lam.append(l)
        num = math.sqrt(sum((AX[i][c] - l * X[i][c]) ** 2 for i in range(n)))
        den = math.sqrt(sum((l * X[i][c]) ** 2 for i in range(n))) or 1.0
        residual = max(residual, num / den)
    return X, lam, residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--vault", type=Path, default=VAULT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--kmax", type=int, default=16, help="迭代维数（只用于看谱）")
    ap.add_argument("--k", type=int, default=12, help="落库维数，默认 12")
    ap.add_argument("--check", action="store_true", help="重建并与现有产物逐字节比对")
    args = ap.parse_args()

    words, slot_of, slot_counts = load_glyphs(args.vault)
    pair, presence, n_sent, n_chunk = scan(args.corpus, words)
    rows, frob2 = build_ppmi(words, pair, presence, n_sent)

    X, lam, residual = truncated_svd(rows, len(words), args.kmax)
    # 谱是平的（PPMI 大量小边主导 Frobenius），能量阈值选不出拐点，
    # 按提案的 8–12 区间固定 12 维落库；完整 16 维谱写进 meta 供复核。
    sigma = [abs(x) for x in lam]
    energy = [s * s for s in sigma]
    total = frob2 or sum(energy) or 1.0
    chosen = max(1, min(args.k, args.kmax, len(words)))

    # 符号锁：每维最大绝对载荷为正；并列取下标最小者
    vec_cols: List[List[float]] = []
    for c in range(chosen):
        col = [X[i][c] for i in range(len(words))]
        best_i = max(range(len(col)), key=lambda i: (abs(col[i]), -i))
        if col[best_i] < 0:
            col = [-v for v in col]
        vec_cols.append(col)

    axis_sigma = sigma[:chosen]
    vectors = {
        w: [round(vec_cols[c][i], 6) for c in range(chosen)]
        for i, w in enumerate(words)
    }
    axes_audit = []
    for c in range(chosen):
        ranked = sorted(range(len(words)), key=lambda i: (abs(vec_cols[c][i]), words[i]), reverse=True)
        top = ranked[:12]
        pos = [[words[i], slot_of[words[i]], round(vec_cols[c][i], 4)] for i in top if vec_cols[c][i] > 0][:6]
        neg = [[words[i], slot_of[words[i]], round(vec_cols[c][i], 4)] for i in top if vec_cols[c][i] < 0][:6]
        axes_audit.append({
            "axis": c,
            "sigma": round(axis_sigma[c], 4),
            "energy_ratio": round(energy[c] / total, 4),
            "pos": pos,
            "neg": neg,
        })

    pair_tokens = sum(pair.values())
    payload = {
        "schema": "isomorphism_p21.v1",
        "meta": {
            "vault_bank_id": json.loads(args.vault.read_text(encoding="utf-8")).get("bank_id"),
            "vault_sha256": _sha256(args.vault),
            "corpus_sha256": _sha256(args.corpus),
            "corpus_chunks": n_chunk,
            "sentences": n_sent,
            "words": len(words),
            "pairs": sum(1 for c in pair.values() if c > 0),
            "pair_tokens": pair_tokens,
            "ppmi_alpha": ALPHA,
            "kmax": args.kmax,
            "k": chosen,
            "singular_values": [round(s, 4) for s in sigma],
            "eigenvalues": [round(x, 4) for x in lam],
            "negative_axes": sum(1 for x in lam[:chosen] if x < 0),
            "cumulative_energy": round(sum(energy[:chosen]) / total, 4),
            "frobenius_sq": round(frob2, 1),
            "residual_max": round(residual, 8),
            "init_seed": INIT_SEED,
        },
        "vectors": vectors,
        "slots": slot_of,
        "axes_audit": axes_audit,
        "coverage": {
            "slot_counts": slot_counts,
            "words_without_pairs": sorted(w for w in words if presence[w] == 0),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"

    if args.check:
        if not args.out.exists():
            raise SystemExit(f"❌ {args.out} 不存在，无法比对")
        current = args.out.read_text(encoding="utf-8")
        if current != text:
            raise SystemExit(f"❌ 重建结果与 {args.out} 不一致（内容漂移）")
        print(f"✅ {args.out} 与重建结果逐字节一致")
        return

    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在，拒绝覆盖（旧产物保持不动）")
    args.out.write_text(text, encoding="utf-8")
    print(f"vault 字粒 {len(words)}，语料 {n_chunk} 块 / {n_sent} 句，"
          f"共现对 {payload['meta']['pairs']}，k={chosen}，"
          f"累计能量 {payload['meta']['cumulative_energy']}，残差 {residual:.2e}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
