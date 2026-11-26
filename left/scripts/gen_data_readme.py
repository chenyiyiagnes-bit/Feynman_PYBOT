"""生成 data/README.md：把数据目录里的产物按用途分类，附说明与大小。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/gen_data_readme.py [--check]

--check 只校验 README 是否最新，不写文件（过期则非零退出）。
新增数据文件后重跑；分类规则没覆盖到的文件会落到「未分类」，同时脚本会提示。
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path("data")
OUT = DATA / "README.md"

PRODUCT_VAULT = "feynman_style_vault_p15.json"
PRODUCT_TOPO = "topology_graph_p15.npz"

# 精确文件的补充说明（分类由 _category 统一判定）
DESC = {
    "feynman_style_vault.json": "基础字库（旧基线，测试夹具依赖）",
    "feynman_style_vault_p8.json": "P8 字库：四槽位相位排版首版",
    "feynman_style_vault_p8b.json": "P8b 字库：重建拓扑配套",
    "feynman_style_vault_p9.json": "P9 字库：后果槽扩到 377 条",
    "feynman_style_vault_p10.json": "P10 字库：解码期检索约束配套（回退档）",
    "feynman_style_vault_p15.json": "**当前产品字库**：非费曼书补库，472 条后果",
    "topology_graph.npz": "基础拓扑图（旧基线）",
    "topology_graph_p8.npz": "P8 拓扑图",
    "topology_graph_p8b.npz": "P8b 拓扑图（hash 4a31b8b2）",
    "topology_graph_p8c.npz": "P8c 拓扑图：加同节三元组（hash 281b68b5）",
    "topology_graph_p9.npz": "P9 拓扑图（hash 107af5df）",
    "topology_graph_p10.npz": "P10 拓扑图（hash 1937db81）",
    "topology_graph_p15.npz": "**当前产品拓扑图**：P15 配套（hash 48cdc9f6）",
    "p8_corpus_v2.jsonl": "费曼语料 v2（123,828 节，154 来源）",
    "p8_corpus_combined.jsonl": "P8 合并语料",
    "p8_corpus_integrated_zh.jsonl": "P8 中文整合语料",
    "p9_consequence_allowlist.json": "P9 后果从句人工策展白名单",
    "p9_consequence_inventory.json": "P9 后果库存（377 条，逐词回源）",
    "p10_consequence_inventory.json": "P10 后果库存（391 条 = 377 + 14 净新增）",
    "p15_gap_allowlist.json": "P15 空档实体策展清单（92 条人工过目）",
    "p15_consequence_inventory.json": "P15 后果库存（472 条，含 tier2 专属从句）",
    "p15_consequence_edges.json": "P15 仅来自非语料书的 E→C 补边（53 条）",
    "entangled_pairs_p15.json": "P15 重算的实体×后果纠缠对（80 对，可打分配对；供 QBOT_ENTANGLED_PAIRS 消融）",
    "eval_p15_gap_entities.json": "P15 空档实体检索评估（13 实体，top-3 换入率）",
    "eval_topology_ablation_p15.json": "拓扑层开/关消融：55 问首句变化率与 Jaccard",
    "rating_analysis_topology.json": "拓扑消融盲评分析（配对差 +0.51，CI [0.11,0.98]）",
    "rating_worksheet_topology_blind.jsonl": "拓扑消融盲评表（90 行）",
    "rating_worksheet_p15_blind.jsonl": "P15 vs P10 盲评表（110 行）",
    "rating_analysis_p15.json": "P15 转正分析（配对差 +1.07，CI [0.73,1.42]）",
    "left_context_p9.json": "377 条后果的左窗口上下文统计（P4/P5，结论：左邻多为虚词）",
    "entity_consequence_gaps.json": "66 实体后果缺口普查",
    "entity_consequence_gaps_p10.json": "P10 版缺口普查",
    "solve_cache.json": "解题结果缓存（运行时产物，可删）",
    "font_banks.json": "量子活字印刷机的字库索引",
    "v2_lemma_wiki.json": "词形/lemma 对照表",
    "p9_consequence_inventory_report.json": "P9 库存构建校验报告",
    "p10_consequence_inventory_report.json": "P10 库存构建校验报告",
    "p15_consequence_inventory_report.json": "P15 库存构建校验报告",
    "rating_worksheet_p14_config_map.json": "P14 盲评配置映射（非标准命名）",
    "rating_worksheet_topology_blind.configs.json": "拓扑消融盲评的配置清单",
    "isomorphism_p21.json": "P21 同构基底：668 词 × 12 维（句级共现 → PPMI → 截断 SVD，含每维载荷审计）",
    "phase_bandwidth_p21.json": "P21 相位带宽：每个字粒在语料里的圆周均值 μ 与可靠性",
    "p21_calibration.json": "P21 相容乘子标定明细（开发集问句 × 3 nonce）",
    "p21_calibration.md": "P21 相容乘子标定摘要（score 分位、乘子方向）",
    "p21_eval.json": "P21 表征评估明细（留出共现回填：baseline / k24 / 距离加权）",
    "p21_eval.md": "P21 表征评估摘要（选型结论：维持 k=12 句级等权）",
    "ab_p21_pairs.md": "P21 段落盲评表（40 组，A/B 已随机摆位）",
    "ab_p21_key.json": "P21 盲评揭盲钥匙（评完再看）",
    "ab_p21_scores_model.jsonl": "P21 盲评评分（Claude 模型评，非人评）",
    "ab_p21_summary.json": "P21 盲评汇总（均分、配对差、胜负）",
    "ab_p21_report.md": "P21 盲评报告（模型评）",
    "ab_p21_proxy.json": "P21 盲评前机械指标明细",
    "ab_p21_proxy.md": "P21 机械指标摘要（留白/三句/实体锁定/重复）",
}

CATEGORIES = [
    ("产品字库", "在用的字库与回退档"),
    ("历史字库", "旧版本一律保留，不覆盖"),
    ("产品拓扑图", "在用的拓扑图"),
    ("历史拓扑图", "旧版本一律保留，不覆盖"),
    ("语料", "字库与拓扑的原料，只读"),
    ("后果库存与策展", "库存、白名单、缺口普查、候选"),
    ("评估与消融", "自动评估报告；结论已并入对应记忆/文档"),
    ("盲评工作表", "配置不可见的评分表"),
    ("盲评映射与元信息", "id → 配置；评分阶段不可看"),
    ("盲评结果", "已评分表与统计分析"),
    ("其它", "运行时产物、索引与对照表"),
]


def _category(name: str) -> str:
    if ".v1." in name or ".v2." in name:          # 旧快照优先于“当前版本”前缀
        return "历史拓扑图" if name.startswith("topology_graph") else "历史字库"
    if name in (PRODUCT_VAULT, PRODUCT_TOPO, "feynman_style_vault_p10.json"):
        return "产品字库"
    if name.startswith("feynman_style_vault"):
        return "历史字库"
    if name.startswith("topology_graph_p15"):
        return "产品拓扑图"
    if name.startswith("topology_graph"):
        return "历史拓扑图"
    if name.endswith(".jsonl") and "corpus" in name:
        return "语料"
    if name.startswith(("eval_", "topology_calibration")):
        return "评估与消融"
    if name.startswith("rating_worksheet"):
        if name.endswith("_rated.jsonl"):
            return "盲评结果"
        if name.endswith(".map.json") or name.endswith(".meta.json") \
                or name.endswith(".configs.json"):
            return "盲评映射与元信息"
        return "盲评工作表"
    if name.startswith("rating_analysis"):
        return "盲评结果"
    if "inventory" in name or "allowlist" in name or "gaps" in name \
            or "candidates" in name or name.startswith("left_context") \
            or "consequence_edges" in name:
        return "后果库存与策展"
    return "其它"


def _note(name: str) -> str:
    if name in DESC:
        return DESC[name]
    if ".v1." in name or ".v2." in name:
        return "旧版本快照"
    if "inventory_report" in name:
        return "库存构建校验报告"
    if name.startswith("feynman_style_vault"):
        return "历史版本快照"
    if name.startswith("topology_graph"):
        return "历史版本快照"
    if name.startswith("eval_fusion"):
        return "融合打分评估（top-3 换入率）"
    if name.startswith("topology_calibration"):
        return "拓扑参数网格标定报告"
    if name.startswith("eval_p14"):
        return "specific_pref 开启前后评估"
    if name.startswith("rating_analysis"):
        return "评分分析：均值、CI、配对差"
    if name.startswith("rating_worksheet"):
        if name.endswith("_rated.jsonl"):
            return "已评分盲评表"
        if name.endswith(".map.json"):
            return "盲评映射：id → 配置（评分阶段不可看）"
        if name.endswith(".meta.json"):
            return "盲评参数与种子"
        return "盲评工作表（待评分）"
    if name.startswith("gap_entity_book_candidates"):
        return "非费曼书空档候选（P15 前身）"
    if "candidates" in name:
        return "挖掘候选（未过门禁）"
    return ""


def _size(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def build() -> str:
    files = sorted(p for p in DATA.iterdir() if p.is_file() and p.name != "README.md")
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(_category(p.name), []).append(p)

    lines = [
        "# data/ 产物清单（自动生成）",
        "",
        "由 `scripts/gen_data_readme.py` 生成，勿手改；新增数据文件后重跑。",
        "规则：历史字库/拓扑版本一律保留不覆盖；产品当前指针在 `.env` 的 "
        "`QBOT_FEYNMAN_VAULT` / `QBOT_FEYNMAN_TOPOLOGY`（用 "
        "`scripts/switch_vault.py --show` 查看）。",
        "",
        f"共 {len(files)} 个文件。",
        "",
    ]
    for cat, blurb in CATEGORIES:
        if cat not in groups:
            continue
        lines += [f"## {cat} — {blurb}", "", "| 文件 | 大小 | 说明 |", "|---|---|---|"]
        for p in sorted(groups[cat]):
            lines.append(f"| `{p.name}` | {_size(p.stat().st_size)} | {_note(p.name)} |")
        lines.append("")
    missing = [p.name for p in files if not _note(p.name)]
    if missing:
        lines += ["## 待补说明", ""] + [f"- `{n}`" for n in missing] + [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != text:
            raise SystemExit("data/README.md 已过期，请重跑 scripts/gen_data_readme.py")
        print("data/README.md 是最新的。")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"✅ 已生成 {OUT}")


if __name__ == "__main__":
    main()
