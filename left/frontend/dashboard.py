import os
import uuid
import hashlib
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import httpx

st.set_page_config(page_title="Feynman-QBot v2.0 量子干涉监控大屏", page_icon="⚛️", layout="wide")

with st.sidebar:
    st.title("⚛️ Feynman-QBot v2.0")
    st.caption("双向驻波对撞 · 反相熔断 · 第二笔记本")
    default_url = os.getenv("DEFAULT_BUS_URL", "http://127.0.0.1:8000")
    bus_url = st.text_input("FastAPI 总线地址", value=default_url)

    app_mode = st.radio(
        "选择运行引擎模式",
        [
            "费曼双向波函数对撞推理",
            "Grover 量子搜索插件 (25% → 100%)",
            "量子活字印刷机 (语料波函数坍缩)",
            "费曼心智活字 (本地语料铸字)",
        ],
        index=0
    )
    st.divider()

    # 成长年轮：只读导出，不影响任何运行状态；内容由总线现场汇总
    if st.button("🌱 导出成长年轮", use_container_width=True):
        try:
            jr = httpx.get(f"{bus_url}/feynman/journal", timeout=10.0).json()
            st.session_state.feynman_journal_md = jr.get("markdown", "")
        except Exception as exc:
            st.session_state.feynman_journal_md = ""
            st.warning(f"总线未响应，年轮未导出：{exc}")
    journal_md = st.session_state.get("feynman_journal_md") or ""
    if journal_md:
        with st.expander("🌱 成长年轮预览", expanded=False):
            st.markdown(journal_md)
        st.download_button(
            "⬇️ 下载成长年轮.md", journal_md,
            file_name="feynman_growth_journal.md", mime="text/markdown",
            use_container_width=True,
        )
    st.divider()

    if app_mode.startswith("Grover"):
        target_state = st.selectbox("设定目标量子态 (Oracle)", ["|00>", "|01>", "|10>", "|11>"], index=2)
        grover_step = st.select_slider(
            "量子电路演化阶段",
            options=[0, 1, 2],
            value=2,
            format_func=lambda x: {
                0: "Step 0: 初始均匀叠加 (25%)",
                1: "Step 1: Oracle 180°相位翻转",
                2: "Step 2: 扩散算子干涉放大 (100%)"
            }[x]
        )
        run_btn = st.button("⚡ 执行当前量子门演化", type="primary", use_container_width=True)
    elif app_mode.startswith("量子活字"):
        st.subheader("🖨️ 铸字与观测台")
        default_corpus = (
            "在微观的量子世界里，孤独的粒子从不沿着单一的轨迹行走。它像一阵春风穿过时间的废墟，"
            "在无数条可能的路径上同时震荡。当观测者的目光折射出好奇的光芒，所有的概率云瞬间坍缩，"
            "在寂静的宇宙尽头唤醒了沉睡的公式与星辰。"
        )
        corpus_text = st.text_area("1. 填入语料（字/词/短语/长文）", value=default_corpus, height=160)
        max_per_slot = st.slider("2. 每槽字粒数（决定并行路径数）", 2, 5, 3)
        observer_mood = st.slider(
            "3. 观测者心境偏振角", 0, 180, 90, 5,
            help="0°=理性写实 | 45°=自然写意 | 90°=量子奇幻 | 165°=深沉时空"
        )
        mood_label = (
            "📐 理性公式基矢" if observer_mood <= 30 else
            "🌿 自然写意基矢" if observer_mood <= 65 else
            "⚛️ 量子奇幻基矢" if observer_mood <= 120 else
            "⏳ 深沉时空基矢"
        )
        st.caption(f"当前偏振片：**{mood_label}（{observer_mood}°）**")
        reprint_btn = st.button("🖨️ 重新刷墨（换一次测量种子）", use_container_width=True)
        requantize_btn = st.button("🔁 重新铸字（语料改动后）", type="primary", use_container_width=True)
    elif app_mode.startswith("费曼心智"):
        st.subheader("🧠 费曼心智旋钮")
        try:
            op_meta = httpx.get(f"{bus_url}/feynman/operators", timeout=3.0).json()
        except Exception:
            op_meta = {}
        op_labels = op_meta.get("labels", {
            "micro": "微观拆解", "cross": "跨界类比", "boundary": "极端边界",
            "reversal": "时间反演", "anti_intuition": "反直觉",
        })
        presets = op_meta.get("presets", {})
        preset_name = st.selectbox("心智算子预设", ["自定义", *presets.keys()], index=0)
        preset = presets.get(preset_name, {})
        style_deg = st.slider(
            "语体相位（讲义 ← → 演讲）", 0, 120, 60, 5,
            help="0–40°：黑板前一步步讲；40–80°：桌边用日常东西打比方；80–120°：礼堂里说到哲学那一层"
        )
        feynman_ops = {
            key: st.slider(label, 0.0, 2.0, float(preset.get(key, 0.0)), 0.1)
            for key, label in op_labels.items()
        }
        feynman_band = (
            "📐 讲义" if style_deg <= 40 else ("☕ 对话" if style_deg <= 80 else "🎤 演讲")
        )
        st.caption(f"当前语体：**{feynman_band}（{style_deg}°）** ｜ 算子全 0 = 基线")
        feynman_topic = st.text_input(
            "话题（可留空）", value="",
            help="填话题时，语料里找不到影子的字粒会被挡掉；全挡光就只印留白。"
        )
        feynman_query = st.text_input(
            "想问什么（可留空）", value="",
            help="写一个具体问题，把里面的人/物/概念塞进实体槽；抽象概念会自动换认知动词。"
        )
        feynman_remember = st.checkbox("启用会话记忆", value=True,
                                       help="记住上一轮把话说到哪个语体方向，最多轻推 ±15°。")
        if feynman_remember:
            if "feynman_session_id" not in st.session_state:
                st.session_state.feynman_session_id = uuid.uuid4().hex[:12]
            st.caption(f"会话 `{st.session_state.feynman_session_id}`")
            if st.button("🆕 开新会话", use_container_width=True):
                st.session_state.feynman_session_id = uuid.uuid4().hex[:12]
        feynman_reprint = st.button("🖨️ 重新排版（换一次种子）", use_container_width=True)
        st.divider()
        feynman_paragraph_mode = st.checkbox(
            "📄 段落模式（现象 → 类比 → 收束）", value=False,
            help="一次排版拿候选池，再重排成三段：先落现象，再用 92° 类比，最后 15° 收束。")
        feynman_beam = st.slider("段落候选宽度", 2, 12, 6) if feynman_paragraph_mode else 6
        feynman_fusion = "boltzmann" if st.checkbox(
            "自由能融合（Boltzmann）", value=False,
            help="算子乘子从线性连乘改为 exp(-Σ w·(1−s)/τ)；全零权重时仍精确等于 1.0（基线不变）。"
        ) else "linear"
        feynman_topology = st.checkbox(
            "拓扑层（P7 实验）", value=False,
            help="加载语料拓扑图：同句三角 ×1.45、两句/三句窗与出处相交是弱证据、"
                 "三边都强却从未同处算空洞。乘子按候选池中位数归一，只保留相对差异。默认关。"
        )
        if feynman_topology:
            try:
                th = httpx.get(f"{bus_url}/feynman/topology", timeout=3.0).json()
                h = th.get("health") or {}
                if h:
                    st.caption(
                        f"拓扑 {th.get('stats', {}).get('topo_hash', '—')}｜"
                        f"同句三角 {h.get('triangles', 0)}、两句窗 {h.get('window2_triangles', 0)}、"
                        f"三句窗 {h.get('window3_triangles', 0)}、出处相交 {h.get('passage_triangles', 0)}"
                        f"｜覆盖率 {h.get('simplex_coverage', 0):.4%}")
                elif not th.get("loaded"):
                    st.caption("拓扑图未加载：检查 FEYNMAN_ENABLE_TOPOLOGY 或 data/topology_graph_p8c.npz")
            except Exception:
                pass
        feynman_conditioned = st.checkbox(
            "条件级联（P8 实验）", value=False,
            help="实体→动作→后果逐级看共现边，顺序束搜索；需要 P8 新字库（QBOT_FEYNMAN_VAULT）才有完整效果。关时与旧版逐字节一致。"
        )
        feynman_tension = st.checkbox(
            "不协和音解决（P16 实验）", value=False,
            help="只在段落模式的 Top-10 候选里重排：先拉起适度悬念，类比承接，最后用落地词收束。"
                 "只增加 phases/tension 证据字段，不造新句；默认关时与基线一致。"
        )
        with st.expander("对话优化（P20 实验，全默认关）", expanded=False):
            feynman_socratic = st.checkbox("苏格拉底反问", value=False,
                                           help="用固定句架 + 本段实体/动作填一个反问；只加字段，不改正文。")
            feynman_portrait = st.checkbox("比喻画像", value=False,
                                           help="把三句的四槽字粒与相位排成一张字卡，不生成新内容。")
            feynman_auto_phase = st.checkbox("自动相位（只建议）", value=False,
                                             help="按本段实际相位给下一轮语体档建议，不自动生效。")
            feynman_breathing = st.checkbox("轮间呼吸", value=False,
                                            help="按轮次给一句固定停顿提示与停顿秒数。")
            feynman_cocreate = st.checkbox("人机共创（只提议）", value=False,
                                           help="只列候选搭配；真正写入须走 /feynman/digest 的 write 双闸。")
        if "feynman_nonce" not in st.session_state:
            st.session_state.feynman_nonce = 0
        if feynman_reprint:
            st.session_state.feynman_nonce += 1
    else:
        preset = st.selectbox("选择内置基准测试悖论", [
            "密封车厢内的悬停无人机地磅读数悖论",
            "麦克斯韦妖（Maxwell's Demon）永动机悖论",
            "【模糊陷阱】卡车里飞无人机（触发第二笔记本反问）",
            "自定义输入课题..."
        ])
        if preset.startswith("密封车厢"):
            default_q = "一辆完全密封的厢式货车停在电子地磅上总重5吨，车内一架2公斤无人机起飞悬停在正中央，地磅读数如何变化？换成敞篷车呢？"
        elif preset.startswith("麦克斯韦妖"):
            default_q = "绝热容器中间有无摩擦小门，小妖只放快分子去右边、慢分子去左边从而制造温差发电，是否打破热力学第二定律？"
        elif preset.startswith("【模糊陷阱】"):
            default_q = "一架无人机在卡车里飞起来，卡车会变轻吗？"
        else:
            default_q = ""

        problem = st.text_area("探索问题", value=default_q, height=110)
        force_refresh = st.checkbox("跳过结果缓存（强制重新推演）", value=False)
        run_btn = st.button("🚀 启动双向波函数对撞与盲审", type="primary", use_container_width=True)

st.title("🌌 Feynman-QBot v2.0 分布式量子干涉实时监控台")

# 模式 A：Grover 搜索
if app_mode.startswith("Grover"):
    try:
        resp = httpx.post(f"{bus_url}/grover", json={"target_state": target_state, "step": grover_step}, timeout=10.0)
        g_data = resp.json()
    except Exception as e:
        st.error(f"请先启动后端服务 (`uvicorn bus.main:app --reload`)！错误：{e}")
        st.stop()

    st.info(f"**当前电路状态：** {g_data['stage_desc']}")
    df_p = pd.DataFrame(g_data["packets"])
    df_s = pd.DataFrame(g_data["spectrum"])

    col_l, col_r = st.columns([1, 1.15])
    with col_l:
        fig = go.Figure()
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='gray', dash='dot'), showlegend=False))
        for _, row in df_p.iterrows():
            is_tgt = (row["conclusion_key"] == f"量子态 {target_state}")
            color = "#FF4B4B" if row["real"] < -0.1 else ("#00FFAA" if is_tgt else "#00B4D8")
            y_offset = 0.0 if is_tgt else (int(row["bot_id"][-1]) - 1.5) * 0.03
            fig.add_trace(go.Scatter(
                x=[0, row["real"]], y=[0, row["imag"] + y_offset],
                mode='lines+markers+text', line=dict(color=color, width=4 if is_tgt else 2),
                marker=dict(size=10 if is_tgt else 7),
                text=["", f"{row['bot_id']} ({row['real']:+.2f})"], textposition="top center", showlegend=False
            ))
        fig.update_layout(
            title="🧭 Grover 复平面振幅矢量罗盘",
            xaxis=dict(title="实部 Real", range=[-1.2, 1.2], zeroline=True),
            yaxis=dict(title="虚部 Imag", range=[-1.2, 1.2], zeroline=True, scaleanchor="x", scaleratio=1),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig_bar = go.Figure(go.Bar(
            x=df_s["probability"], y=df_s["conclusion"], orientation='h',
            text=[f"{p:.1f}% (振幅: {r:+.2f})" for p, r in zip(df_s["probability"], df_s["net_real"])],
            textposition='auto',
            marker=dict(color=["#00FFAA" if "[目标]" in c else "#3B82F6" for c in df_s["conclusion"]])
        ))
        fig_bar.update_layout(title="📊 量子测量概率分布 (|Ψ|²)", xaxis=dict(title="坍缩概率 (%)", range=[0, 110]), height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

# 模式 C：量子活字印刷机
elif app_mode.startswith("量子活字"):
    st.title("🖨️ 量子活字印刷机 · 实时显影台")

    key = hashlib.sha256(f"{corpus_text}|{max_per_slot}".encode("utf-8")).hexdigest()[:16]
    needs_quantize = (
        st.session_state.get("mt_key") != key
        or requantize_btn
        or "mt_bank" not in st.session_state
    )
    if needs_quantize:
        with st.spinner("⛏️ 正在把语料铸成量子字粒..."):
            try:
                resp = httpx.post(
                    f"{bus_url}/movable_type/quantize",
                    json={"corpus_text": corpus_text, "max_per_slot": int(max_per_slot)},
                    timeout=120.0,
                )
            except Exception as e:
                st.error(f"无法连接后端总线 ({bus_url})！错误：{e}")
                st.stop()
        if resp.status_code != 200:
            st.error(f"铸字失败：{resp.status_code} {resp.text[:200]}")
            st.stop()
        st.session_state.mt_bank = resp.json()
        st.session_state.mt_key = key
        st.session_state.mt_nonce = 0

    if reprint_btn:
        st.session_state.mt_nonce = st.session_state.get("mt_nonce", 0) + 1
    nonce = st.session_state.get("mt_nonce", 0)
    bank_data = st.session_state.mt_bank

    try:
        pr = httpx.post(
            f"{bus_url}/movable_type/print",
            json={
                "bank_id": bank_data["bank_id"],
                "observer_mood_deg": float(observer_mood),
                "nonce": nonce,
                "top_k": 12,
            },
            timeout=15.0,
        )
    except Exception as e:
        st.error(f"刷墨失败：{e}")
        st.stop()
    if pr.status_code == 404:
        st.session_state.pop("mt_bank", None)
        st.warning("字库在后端已过期（服务可能重启过），请再点一次「重新铸字」。")
        st.stop()
    if pr.status_code != 200:
        st.error(f"刷墨失败：{pr.status_code} {pr.text[:200]}")
        st.stop()
    printed = pr.json()

    st.caption(
        f"⚙️ {bank_data['source_summary']} ｜ 并行路径 {printed['paths_total']} 条 ｜ "
        f"采样种子 {printed['seed']}（同种子同心境必复现同一句）"
    )

    slots = bank_data["font_bank"]
    col_s, col_v, col_o = st.columns(3)
    for col, title, items in (
        (col_s, "字槽 1 · 主语意象", slots["subjects"]),
        (col_v, "字槽 2 · 量子动作", slots["verbs"]),
        (col_o, "字槽 3 · 时空归宿", slots["objects"]),
    ):
        with col:
            st.markdown(f"**【{title}】**")
            for g in items:
                flag = "🚫" if g["tag"] == "cliche" else "•"
                st.markdown(
                    f"{flag} `{g['word']}` 振幅 {g['amp']:+.2f} · 相位 {g['phase_deg']:.0f}° · #{g['tag']}"
                )

    st.divider()
    res_l, res_r = st.columns(2)
    with res_l:
        if printed["collapsed"]:
            st.success(f"### 📜 刷墨落纸（本次测量）\n## 『 {printed['printed_collapse']} 』")
        else:
            st.warning("### 🫥 波函数完全相消\n" + printed["message"])
    with res_r:
        st.info(f"### 🏆 最强相长共鸣（确定性 Top-1）\n## 『 {printed['top_resonant_line']} 』")

    df_paths = pd.DataFrame(printed["paths"])
    if df_paths.empty:
        st.stop()

    chart_l, chart_r = st.columns([1, 1.15])
    with chart_l:
        fig = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode='lines',
            line=dict(color='gray', dash='dot'), showlegend=False
        ))
        for _, row in df_paths.iterrows():
            if row["magnitude"] <= 0.02:
                continue
            color = "#FF4B4B" if row["cliche"] else (
                "#00FFAA" if row["sentence"] == printed["top_resonant_line"] else "#00B4D8"
            )
            fig.add_trace(go.Scatter(
                x=[0, row["real"]], y=[0, row["imag"]],
                mode='lines+markers+text',
                line=dict(color=color, width=3),
                marker=dict(size=7),
                text=["", row["sentence"][:8] + "…"],
                textposition="top center", showlegend=False
            ))
        fig.update_layout(
            title=f"🧭 心境 ({observer_mood:.0f}°) 投影复平面（右=同向可落墨，左=相消）",
            xaxis=dict(title="同向投影 Real", range=[-1.0, 1.0], zeroline=True),
            yaxis=dict(title="正交张力 Imag", range=[-1.0, 1.0], zeroline=True,
                       scaleanchor="x", scaleratio=1),
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_r:
        fig_bar = go.Figure(go.Bar(
            x=df_paths["probability"], y=df_paths["sentence"], orientation='h',
            text=[f"{p:.1f}% · {n}" for p, n in zip(df_paths["probability"], df_paths["note"])],
            textposition='auto',
            marker=dict(color=df_paths["probability"], colorscale='Tealgrn')
        ))
        fig_bar.update_layout(
            title="📊 坍缩概率光谱（Top 12）",
            xaxis=dict(title="坍缩概率 (%)", range=[0, 105]),
            yaxis=dict(autorange="reversed"),
            height=430
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("🔍 查看并行字粒路径明细", expanded=False):
        show = df_paths[["sentence", "components", "tags", "phase_deg", "real", "imag", "probability", "note"]].copy()
        show["components"] = show["components"].apply(lambda x: " + ".join(x))
        show["tags"] = show["tags"].apply(lambda x: " / ".join(x))
        show.columns = ["印句", "字粒组合", "纠缠标签", "合成相位(°)", "有效实部", "虚部", "坍缩概率(%)", "相位注记"]
        st.dataframe(show, use_container_width=True, hide_index=True)

# 模式 B：费曼双向波函数对撞
elif not app_mode.startswith("费曼心智"):
    if run_btn or "result" not in st.session_state:
        with st.spinner("⚡ 正在执行双向波函数对撞 -> Map-Reduce 并发盲审 -> 非线性反相熔断..."):
            try:
                resp = httpx.post(f"{bus_url}/solve", json={"problem": problem, "force_refresh": force_refresh}, timeout=120.0)
                st.session_state.result = resp.json()
            except Exception as e:
                st.error(f"无法连接后端总线 ({bus_url})！错误：{e}")
                st.stop()

    data = st.session_state.result

    if not data.get("packets") or not data.get("spectrum"):
        nb = data.get("second_notebook") or {}
        st.warning(f"**未进行波函数推演：** {nb.get('conflict_diagnosis', '本轮没有产生任何推理路径')}")
        for item in nb.get("missing_parameters", []):
            st.markdown(f"- {item}")
        for q in nb.get("clarifying_questions", []):
            st.markdown(f"**{q}**")
        st.stop()

    df_packets = pd.DataFrame(data["packets"])
    df_spec = pd.DataFrame(data["spectrum"])
    winner = df_spec.iloc[0]
    loser = df_spec.iloc[-1] if len(df_spec) > 1 else winner

    if data.get("cache_hit"):
        st.caption(
            f"⚡ 命中 /solve 结果缓存（写入于 {data.get('cache_age_sec', 0)} 秒前），本轮未产生新的 LLM 调用。"
            " 需要强制重算是非题，勾选侧栏「跳过缓存」后重跑。"
        )
    else:
        st.caption(
            f"🛰️ 本轮 LLM 调用 {data.get('llm_calls', 0)} 次"
            f"（进程累计 {data.get('llm_budget', {}).get('granted', 0)} 次，被预算拦截 "
            f"{data.get('llm_budget', {}).get('denied', 0)} 次）。"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("双向对撞节点", f"{len(df_packets)} 个", "3 正向 + 2 逆向")
    c2.metric("胜出坍缩概率", f"{winner['probability']:.2f}%", f"有效实部 {winner['net_real']:+.2f}")
    c3.metric("伪命题残余概率", f"{loser['probability']:.2f}%", "反相熔断+虚部阻尼", delta_color="inverse")
    c4.metric("波函数坍缩状态", data["collapse_status"], "门限 >= 60%")

    st.divider()
    col_left, col_right = st.columns([1, 1.15])

    with col_left:
        fig = go.Figure()
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='gray', dash='dot'), showlegend=False))
        for _, row in df_packets.iterrows():
            color = "#FF4B4B" if abs(row["phase_deg"]) > 90 else ("#00FFAA" if "ADVANCED" in row["wave_direction"] else "#00B4D8")
            fig.add_trace(go.Scatter(
                x=[0, row["real"]], y=[0, row["imag"]],
                mode='lines+markers+text', line=dict(color=color, width=3), marker=dict(size=8),
                text=["", f"{row['bot_id']} ({row['phase_deg']:+.0f}°)"], textposition="top center", showlegend=False
            ))
        fig.update_layout(
            title="🧭 复平面相位罗盘 (青=正向波 | 绿=逆向驻波 | 红=反相熔断)",
            xaxis=dict(title="实部 Real (逻辑自洽度)", range=[-1.2, 1.2], zeroline=True),
            yaxis=dict(title="虚部 Imag (对称退相干张力)", range=[-1.2, 1.2], zeroline=True, scaleanchor="x", scaleratio=1),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig_bar = go.Figure(go.Bar(
            x=df_spec["probability"], y=df_spec["conclusion"], orientation='h',
            text=[f"{p:.2f}% (有效实部:{r:+.2f}, 虚部:{i:.2f})" for p, r, i in zip(df_spec["probability"], df_spec["net_real"], df_spec["net_imag"])],
            textposition='auto',
            marker=dict(color=df_spec["probability"], colorscale='Tealgrn')
        ))
        fig_bar.update_layout(title="📊 非线性反相熔断 + 虚部阻尼坍缩光谱", xaxis=dict(title="坍缩概率 (%)", range=[0, 110]), height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 渲染费曼第一性原理裁决 或 第二笔记本交互追问
    if data.get("collapse_status") == "COLLAPSED_VALID":
        st.subheader("🎓 Skeptic Bot 费曼第一性原理裁决")
        st.success(data["skeptic_verdict"])
    elif data.get("collapse_status") == "COLLAPSED_RESCUED":
        st.subheader("🛟 反相熔断后的相对排序补救")
        st.warning(data.get("recovery_note", ""))
        st.info(data["skeptic_verdict"])
    else:
        nb = data["second_notebook"]
        st.subheader("📓 触发《费曼第二笔记本》：波函数未能坍缩（最高概率 < 60%）")
        st.warning(f"**退相干分歧诊断：** {nb['conflict_diagnosis']}")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🔍 探测到的缺失关键参数")
            for item in nb["missing_parameters"]:
                st.markdown(f"- {item}")
        with col_m2:
            st.markdown("#### ❓ 请补充以下边界条件以促成坍缩")
            for q in nb["clarifying_questions"]:
                st.markdown(f"**{q}**")
        user_patch = st.text_input("在此输入补充条件（例如：卡车是完全密封的厢式货车）：")
        if st.button("🔄 注入补充条件，重新执行波函数坍缩", type="primary"):
            if user_patch.strip():
                new_problem = f"{data['problem']} 【补充边界条件：{user_patch.strip()}】"
                resp = httpx.post(f"{bus_url}/solve", json={"problem": new_problem}, timeout=120.0)
                st.session_state.result = resp.json()
                st.rerun()

    with st.expander("🔍 查看 Map-Reduce 并发审计批注与双向推演明细", expanded=True):
        show_df = df_packets[["bot_id", "perspective", "conclusion_key", "phase_deg", "interference_regime", "critical_flaw", "detail"]]
        show_df.columns = ["节点ID", "波函数方向与偏置", "归并基矢", "相位角(°)", "干涉相区", "盲审审计批注", "原始推演"]
        st.dataframe(show_df, use_container_width=True, hide_index=True)

# 模式 D：费曼心智活字（字粒来自本地费曼语料）
elif app_mode.startswith("费曼心智"):
    st.title("🧠 费曼心智活字 · 语体显影台")
    st.caption("字粒全部铸自费曼本人作品（讲义、演讲、书信、访谈），每颗可追出处；四槽位=起手式×实体×动作×可观测后果。")

    try:
        vr = httpx.get(f"{bus_url}/feynman/vault", timeout=20.0)
    except Exception as e:
        st.error(f"无法连接后端总线 ({bus_url})！错误：{e}")
        st.stop()
    if vr.status_code == 503:
        st.error("后端没有费曼字库。先在项目根目录运行：`python plugins/feynman_vault.py`")
        st.stop()
    if vr.status_code != 200:
        st.error(f"读取字库失败：{vr.status_code} {vr.text[:200]}")
        st.stop()
    vault_info = vr.json()

    try:
        pr = httpx.post(
            f"{bus_url}/feynman/print",
            json={
                "bank_id": vault_info["bank_id"],
                "style_deg": float(style_deg),
                "nonce": st.session_state.get("feynman_nonce", 0),
                "top_k": 10,
                "topic": feynman_topic,
                "user_query": feynman_query,
                "session_id": (st.session_state.get("feynman_session_id")
                               if feynman_remember else ""),
                "operators": feynman_ops,
                "fusion": feynman_fusion,
                "use_topology": feynman_topology,
                "conditioned_chain": bool(feynman_conditioned),
            },
            timeout=20.0,
        )
    except Exception as e:
        st.error(f"排版失败：{e}")
        st.stop()
    if pr.status_code != 200:
        st.error(f"排版失败：{pr.status_code} {pr.text[:200]}")
        st.stop()
    fout = pr.json()

    st.caption(
        f"⚙️ 语料 {vault_info['sections']} 段 / {vault_info['sentences']} 句 ｜ "
        f"字粒 " + " · ".join(f"{k} {v}" for k, v in vault_info["glyphs"].items()) +
        f" ｜ 术语闸门 {vault_info['jargon_terms']} 条 ｜ 种子 {fout['seed']} ｜ "
        f"扩展指纹 `{fout.get('ext_hash', '00000000')}`"
    )

    if fout.get("session"):
        sess = fout["session"]
        s_l, s_m, s_r = st.columns([1.4, 1, 1])
        with s_l:
            st.caption("会话记忆振幅（上限 2.0，退相干会清零）")
            st.progress(min(1.0, float(sess.get("amplitude", 0.0)) / 2.0),
                        text=f"|A| = {sess.get('amplitude', 0.0):.2f}")
        with s_m:
            st.metric("会话相位", f"{sess.get('phase_deg', 0.0):.0f}°")
        with s_r:
            st.metric("本轮旋钮偏移", f"{sess.get('bias_deg', 0.0):+.1f}°",
                      help="记忆最多把旋钮轻推 ±15°，不会替你决定语体。")

    purity = float(fout.get("purity", 1.0 if fout.get("collapsed") else 0.0))

    if not fout.get("collapsed"):
        st.warning("### 🫥 印不出可核对的句子\n" + fout["message"])
        st.markdown(f"## 『 {fout['printed_collapse']} 』")
        st.stop()

    kpi_l, kpi_m, kpi_r = st.columns([1.1, 1.2, 1.7])
    with kpi_l:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=purity * 100,
            number={"suffix": "%"},
            title={"text": "纯净度（术语路径质量占比反向）"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2A9D8F" if purity >= 0.8 else "#E76F51"},
                "steps": [
                    {"range": [0, 50], "color": "#F4A261"},
                    {"range": [50, 80], "color": "#E9C46A"},
                    {"range": [80, 100], "color": "#A8DADC"},
                ],
            },
        ))
        gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)
    with kpi_m:
        st.metric("路径总数", fout.get("paths_total", 0))
        st.metric("存活路径", fout.get("paths_alive", 0))
        st.metric("术语枪毙", len(fout.get("killed", [])))
        if fout.get("skeleton"):
            st.caption(f"骨架：{fout['skeleton'].get('pattern_type')} · "
                       f"{fout['skeleton'].get('phase_deg')}° · 语料命中 {fout['skeleton'].get('freq')} 次")
    with kpi_r:
        radar = go.Figure(go.Scatterpolar(
            r=[float(feynman_ops.get(k, 0.0)) for k in op_labels],
            theta=list(op_labels.values()),
            fill="toself",
            line=dict(color="#457B9D"),
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
            showlegend=False, height=240, margin=dict(t=30, b=10, l=40, r=40),
            title="五算子权重",
        )
        st.plotly_chart(radar, use_container_width=True)

    res_l, res_r = st.columns(2)
    with res_l:
        badges = " ".join(fout.get("top_badges") or [])
        st.success(f"### 🖨️ 本次排版\n## 『 {fout['printed_collapse']} 』")
        if badges:
            st.caption(badges)
    with res_r:
        st.info(f"### 🏆 最强共鸣（确定性 Top-1）\n## 『 {fout['top_resonant_line']} 』")

    if feynman_paragraph_mode:
        st.divider()
        st.markdown("### 📄 段落模式 · 现象锚点 → 92° 类比 → 15° 收束")
        wout, werr = None, ""
        try:
            wpr = httpx.post(
                f"{bus_url}/feynman/paragraph",
                json={
                    "user_query": feynman_query,
                    "style_deg": float(style_deg),
                    "nonce": st.session_state.get("feynman_nonce", 0),
                    "beam_width": int(feynman_beam),
                    "session_id": (st.session_state.get("feynman_session_id")
                                   if feynman_remember else ""),
                    "operators": feynman_ops,
                    "fusion": feynman_fusion,
                    "use_topology": feynman_topology,
                    "conditioned_chain": bool(feynman_conditioned),
                    "dissonance_resolution": bool(feynman_tension),
                    "socratic": bool(feynman_socratic),
                    "portrait": bool(feynman_portrait),
                    "auto_phase": bool(feynman_auto_phase),
                    "breathing": bool(feynman_breathing),
                    "cocreate": bool(feynman_cocreate),
                },
                timeout=30.0,
            )
            if wpr.status_code == 200:
                wout = wpr.json()
            else:
                werr = f"{wpr.status_code} {wpr.text[:200]}"
        except Exception as e:
            werr = str(e)
        if werr:
            st.error(f"段落排版失败：{werr}")
        if wout:
            st.success(f"### 📜 {wout['paragraph']}")
            stage_names = ("现象锚点", "类比", "收束")
            cols = st.columns(3)
            for col, s in zip(cols, wout["sentences"]):
                with col:
                    st.markdown(f"**{stage_names[s['stage']]}** · `{s['kind']}`")
                    st.markdown(f"> {s['connector']}{s['line']}")
                    if s.get("probability") is not None:
                        st.caption(f"坍缩概率 {s['probability'] * 100:.1f}%")
                    if s.get("tension") is not None:
                        st.caption(f"张力 {s['tension']:.2f}")
                    if s.get("topology"):
                            st.caption(f"拓扑乘子 ×{s['topology']['multiplier']:.2f}"
                                       + ("（实心三角形）" if s["topology"].get("is_simplex") else ""))
            st.caption(
                f"🌡️ 语体 {wout['style_band']} ｜ 候选池 {wout['pool_size']} 条 ｜ "
                f"种子 {wout['seed']} ｜ 耗时 {wout['latency_ms']:.1f} ms"
                + (f" ｜ 拓扑指纹 `{wout['topology_hash']}`" if wout.get("topology_hash") else "")
            )
            if not wout.get("complete") and not wout.get("greeting"):
                st.warning("候选池里凑不齐三句互不重复的话，这段先短着，不拿重复句充数。")
            dlg = wout.get("dialogue") or {}
            if dlg:
                with st.expander("🧪 对话优化结果（P20）", expanded=True):
                    if dlg.get("socratic"):
                        st.markdown(f"**反问**：{dlg['socratic']}")
                    if dlg.get("portrait"):
                        st.code(dlg["portrait"], language="text")
                    if dlg.get("auto_phase"):
                        ap = dlg["auto_phase"]
                        st.caption(f"建议语体 {ap['suggested_style_deg']}°"
                                   f"（本段实测 {ap['observed_phase_mean']}°；只建议，不自动生效）")
                    if dlg.get("breathing"):
                        st.caption(f"呼吸 {dlg['breathing']['seconds']} 秒：{dlg['breathing']['line']}")
                    if dlg.get("cocreate"):
                        st.caption("共创候选：" + "、".join(dlg["cocreate"]["candidate_edges"])
                                   + f"（{dlg['cocreate']['requires']}；此处只提议）")

    top_glyphs = (fout.get("paths") or [{}])[0].get("glyphs", {})
    entities = fout.get("user_entities") or []
    if entities:
        ew = fout.get("entity_weights") or {}
        st.caption("🎯 用户实体 → 动作经验乘子：" + " ｜ ".join(
            f"`{e}` ×{ew.get(e, 1.0):.2f}" for e in entities))
        if top_glyphs:
            fb_l, fb_r, fb_note = st.columns([1, 1, 3])
            payload = {
                "entities": entities,
                "action": top_glyphs.get("action", ""),
                "skeleton": (fout.get("skeleton") or {}).get("pattern_type", ""),
                "session_id": (st.session_state.get("feynman_session_id")
                               if feynman_remember else ""),
            }
            with fb_l:
                up = st.button("👍 这句对味", use_container_width=True)
            with fb_r:
                down = st.button("👎 这句不对", use_container_width=True)
            if up or down:
                try:
                    fr = httpx.post(f"{bus_url}/feynman/feedback",
                                    json={**payload, "vote": "up" if up else "down"},
                                    timeout=10.0)
                    if fr.status_code == 200:
                        fb = fr.json()
                        st.session_state.feynman_last_feedback = (
                            f"{'👍' if up else '👎'} 已记下："
                            f"{'、'.join(fb['entities'])} × {fb['action']} "
                            f"（{fb['applied_delta']:+.2f}）→ 扩展指纹 "
                            f"`{fb['previous_ext_hash']}` → `{fb['ext_hash']}`")
                    else:
                        st.session_state.feynman_last_feedback = (
                            f"反馈被拒：{fr.status_code} {fr.text[:120]}")
                except Exception as e:
                    st.session_state.feynman_last_feedback = f"反馈提交失败：{e}"
            with fb_note:
                tip = st.session_state.get("feynman_last_feedback")
                if tip:
                    st.success(tip)
                else:
                    st.caption("点赞 +0.30 / 点踩 −0.45，钳位 [0.15, 1.5]；点踩到底会自动剪掉这条搭配。")

    if fout.get("killed"):
        with st.container():
            st.error("### 🚫 本轮被枪毙的最高权重路径（术语堆叠）")
            for krow in fout["killed"]:
                st.markdown(f"- 『 {krow.get('rendered', krow['line'])} 』 ← 术语：{'、'.join(krow['jargon_hits'])}")

    df_paths = pd.DataFrame(fout["paths"])
    if df_paths.empty:
        st.stop()
    df_paths["glyph_str"] = df_paths["glyphs"].apply(lambda d: " × ".join(d.values()))
    df_paths["note_str"] = df_paths["notes"].apply(lambda xs: " / ".join(xs) if xs else "—")
    df_paths["badge_str"] = df_paths.get("badges", pd.Series([[]] * len(df_paths))).apply(
        lambda xs: " ".join(xs) if xs else "—")
    df_paths["shown"] = df_paths["rendered"] if "rendered" in df_paths.columns else df_paths["line"]

    chart_l, chart_r = st.columns(2)
    with chart_l:
        st.markdown("**📊 候选路径概率（Top 10）**")
        fig_bar = go.Figure(go.Bar(
            x=df_paths["probability"] * 100, y=df_paths["shown"], orientation="h",
            text=[f"{p*100:.1f}%" for p in df_paths["probability"]],
            textposition="auto",
            marker=dict(color=df_paths["coherence"], colorscale="Tealgrn"),
        ))
        fig_bar.update_layout(xaxis=dict(title="坍缩概率 (%)", range=[0, 110]),
                              yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_r:
        st.markdown("**🧭 语体相位分布**")
        fig_ph = go.Figure()
        # 用路径里记录的相位总合画线，不猜单颗相位
        for row in fout["paths"][:10]:
            fig_ph.add_trace(go.Scatter(
                x=[0, 3], y=[row["phase_total_deg"], row["phase_total_deg"]],
                mode="lines", opacity=0.35, showlegend=False,
                line=dict(color="#00B4D8")
            ))
        fig_ph.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_ph.update_layout(
            title=f"🎛️ 旋钮 {fout['style_band']} · 每行是一条路径的相位总和",
            xaxis=dict(showticklabels=False, range=[-0.2, 3.2]),
            yaxis=dict(title="相位总合 (°)", range=[0, 360]),
            height=420,
        )
        st.plotly_chart(fig_ph, use_container_width=True)

    with st.expander("🔍 路径明细与字粒出处", expanded=True):
        show = df_paths[["shown", "glyph_str", "phase_total_deg", "coherence", "badge_str", "note_str", "probability"]].copy()
        show.columns = ["印句", "四槽字粒", "相位总合(°)", "语体相干度", "算子徽章", "闸门注记", "概率"]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.markdown("**每颗字粒的语料出处（前两条）**")
        for slot in ("hook", "entity", "action", "consequence"):
            word = fout["paths"][0]["glyphs"].get(slot, "?")
            srcs = fout["paths"][0]["sources"].get(slot, [])
            st.markdown(f"- **{slot}** `{word}` ← " + (" ； ".join(srcs) if srcs else "（无）"))
