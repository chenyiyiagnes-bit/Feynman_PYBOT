"""生成 ARCHITECTURE.md：路由、字库/拓扑现状、扩展库绑定、模块/脚本/测试清单。

全部信息现场从运行中的对象读取（app.routes、Vault.stats()、engine.stats()、
load_domain_extensions_with_hash()），不是一个会过期的副本。

用法：
    DEEPSEEK_API_KEY= GEMINI_API_KEY= PYTHONPATH=. .venv/bin/python -B scripts/gen_architecture.py
    ... --check     只校验是否最新
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "ARCHITECTURE.md"

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("QBOT_CACHE_TTL_SEC", "0")


def _first_doc(path: Path) -> str:
    try:
        mod = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(mod) or ""
        return doc.strip().splitlines()[0] if doc.strip() else ""
    except Exception:
        return ""


def _count_tests(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return sum(1 for n in walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name.startswith("test_"))


def walk(tree):
    yield from ast.walk(tree)


def _route_rows(routes) -> list[str]:
    skip = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
    rows = []
    for r in routes:
        path = getattr(r, "path", "")
        if path in skip:
            continue
        methods = sorted(getattr(r, "methods", set()) - {"HEAD", "OPTIONS"})
        name = getattr(getattr(r, "endpoint", None), "__name__", "")
        doc = (getattr(getattr(r, "endpoint", None), "__doc__", None) or "").strip()
        doc = doc.splitlines()[0] if doc else ""
        rows.append((path, "/".join(methods), name, doc))
    rows.sort(key=lambda x: (not x[0].startswith("/feynman"), x[0], x[1]))
    return [f"| `{m} {p}` | `{n}` | {d} |" for p, m, n, d in rows]


def build() -> str:
    import bus.main as bm
    from plugins.domain_expander import load_domain_extensions_with_hash
    from plugins.topology_builder import get_engine

    vault = bm._get_feynman_vault()
    vs = vault.stats()
    eng = get_engine(force=True)
    es = eng.stats() if eng else None
    ext, ext_hash = load_domain_extensions_with_hash()

    L: list[str] = [
        "# 架构（自动生成）",
        "",
        "由 `scripts/gen_architecture.py` 从运行中的对象现场读取生成，勿手改。",
        "数据产物清单见 `data/README.md`；产品默认配置见 `bus/main.py` 的 "
        "`_product_defaults()`。",
        "",
        "## 进程与职责",
        "",
        "| 模块 | 职责 |",
        "|---|---|",
        "| `bus/main.py` | FastAPI 总线：路由、产品默认档、会话内存、进化写回、指标 |",
        "| `plugins/feynman_typesetter.py` | 活字排版：槽位/相位/骨架组装，不生成新文字 |",
        "| `plugins/paragraph_weaver.py` | 段落编织：一次 print 取池 + 三级重排（锚点→类比→收束） |",
        "| `plugins/feynman_vault.py` | 字库加载与坏库回退 |",
        "| `plugins/topology_builder.py` | 拓扑流形：团复形、洞检测、类比对、请求级重排 |",
        "| `plugins/isomorphism_p21.py` | 同构基底（默认关）：句级 PPMI-SVD 字粒向量、OOV 投影、相位带宽对齐、相容乘子、诚实诊断 |",
        "| `plugins/evolution_loop.py` | 反馈 → 扩展库边权更新（带跨任务锁、钳位） |",
        "| `plugins/domain_expander.py` | 扩展库读写、EXT_LOCK、vault_id 溯源标记 |",
        "| `shared/metrics.py` | 请求空白率 / 延迟滚动窗口 |",
        "| `shared/solve_cache.py` | 解题缓存（TTL） |",
        "| `shared/llm_ledger.py` | LLM 调用记账（默认关闭，见 .env 开关） |",
        "",
        "## 路由（引自 app.routes）",
        "",
        "| 路由 | 处理函数 | 说明 |",
        "|---|---|---|",
        *_route_rows(bm.app.routes),
        "",
        "## 当前产品字库（Vault.stats()）",
        "",
        f"- bank_id：`{vs['bank_id']}`",
        f"- 语料：{vs.get('corpus')}；节 {vs.get('sections')}；句 {vs.get('sentences')}",
        "- 槽位字粒数：" + "，".join(f"{k} {v}" for k, v in vs["glyphs"].items()),
        "- 槽位相位偏置（度）：" + "，".join(f"{k} {v}" for k, v in vs["slot_offsets"].items()),
        f"- 骨架 {len(vs['skeletons'])} 条；共现对 {vs['cooccurrence_pairs']}；"
        f"纠缠对 {vs['entangled_pairs']}；术语 {vs['jargon_terms']}",
        "",
        "| 骨架 | 相位（度） | 频次 |",
        "|---|---|---|",
        *[f"| {sk['pattern_type']} | {sk['phase_deg']} | {sk['freq']} |"
          for sk in vs["skeletons"]],
        "",
    ]

    if es:
        L += [
            "## 当前拓扑图（engine.stats()）",
            "",
            f"- topo_hash：`{es['topo_hash']}`",
            f"- 词 {es['words']}；已建复形 {es['triangles']}；"
            f"三角按证据分级 tier2 {es['window2_triangles']} / tier3 {es['window3_triangles']}"
            f"（分级表可含未进复形列表的三角）；同节 {es['section_triangles']}；"
            f"同段 {es['passage_triangles']}",
            f"- 边 {es['edges']}，支撑≥2 的边 {es['edge_support_ge2']}；"
            f"类比对 {es['analogy_pairs']}，alpha {es['analogy_alpha']}",
            f"- 引擎开关：请求级 `use_topology` 默认开（产品档），"
            f"插件默认 `FEYNMAN_ENABLE_TOPOLOGY="
            f"{os.getenv('FEYNMAN_ENABLE_TOPOLOGY', '0')}`（新进程缺省 0）",
            "",
        ]

    L += [
        "## 扩展库（学习产物）",
        "",
        f"- 文件：`{os.getenv('QBOT_FEYNMAN_EXT', 'data/domain_extensions.json')}`",
        f"- 内容哈希：`{ext_hash}`；实体原型 {len(ext.get('entity_prototypes', {}))} 条；"
        f"共现边 {len(ext.get('cooccurrence_edges', {}))} 条",
        f"- 溯源标记 vault_id：`{ext.get('vault_id') or '（无，旧库）'}`；"
        f"当前字库 `{vs['bank_id']}` → "
        f"{'一致' if ext.get('vault_id') == vs['bank_id'] else ('未知' if not ext.get('vault_id') else '不一致（换库后旧经验仍在，仅标记）')}",
        "",
        "## 脚本（scripts/）",
        "",
        "| 脚本 | 说明 |",
        "|---|---|",
        *[f"| `{p.name}` | {_first_doc(p)} |" for p in sorted((ROOT / 'scripts').glob('*.py'))],
        "",
        "## 回归（tests/）",
        "",
        f"共 {len(list((ROOT / 'tests').glob('test_*.py')))} 个文件。统一入口：",
        "",
        "```sh",
        "PYTHONPATH=. .venv/bin/python -B scripts/run_all_tests.py",
        "```",
        "",
        "| 文件 | 用例数 | 说明 |",
        "|---|---|---|",
        *[f"| `{p.name}` | {_count_tests(p)} | {_first_doc(p)} |"
          for p in sorted((ROOT / "tests").glob("test_*.py"))],
        "",
        "## 约定",
        "",
        "- 不生成新文字：所有输出字粒必须能在字库中逐字溯源；编辑过的标 `normalized_from`。",
        "- 历史字库/拓扑版本只增不改；换产品指针用 `scripts/switch_vault.py`。",
        "- 训练语料中英一一对应，英文只取费曼原话。",
        "- 默认档由服务端补齐，客户端显式传值优先（pydantic `model_fields_set`）。",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != text:
            print("ARCHITECTURE.md 已过期（重跑不带 --check 即可刷新）。")
            raise SystemExit(1)
        print("ARCHITECTURE.md 是最新的。")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"✅ 已生成 {OUT}")


if __name__ == "__main__":
    main()
