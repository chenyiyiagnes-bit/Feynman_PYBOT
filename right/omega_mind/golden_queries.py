"""The retrieval regression net, importable at runtime.

The tests own the expectations, but a dream-time self-audit has to check the
same questions before it may change the learned store. Keeping the list here
means the audit and the test suite can never drift apart.
"""

from __future__ import annotations

# (query, language, acceptable card ids)
GOLDEN = (
    ("热水为什么会自己变凉", "zh", {"U0_RELAX"}),
    ("为什么端一杯满水走路反而稳", "zh", {"U0_DAMP"}),
    ("钱不会凭空多出来，那它会去哪", "zh", {"U0_FLUX"}),
    ("为什么有些事一开始不起眼，后来收不住", "zh", {"U0_FEEDBACK"}),
    ("什么是科学", "zh", {"persona-science-001"}),
    ("费曼是谁", "zh", {"self-001"}),
    ("费曼怎么看待权威", "zh", {"card-003"}),
    ("什么是阻尼", "zh", {"U0_DAMP", "C_DAMP"}),
    # sm-005 (S = k ln W) lists this very question as an alias and is the
    # direct definition, so it is as right an answer as the chapter card.
    ("熵是什么", "zh", {"thermo-entropy-disorder", "flp-i-44", "sm-005"}),
    ("什么是能量守恒", "zh", {"flp-i-04", "C_FLUX"}),
    ("什么是扩散", "zh", {"C_DIFFUSE"}),
    ("重整化是什么", "zh", {"S_RG", "qed-005"}),
    ("What is entropy?", "en", {"flp-i-44", "thermo-entropy-disorder", "sm-005"}),
    ("Why does hot water cool down?", "en", {"U0_RELAX"}),
    ("What is science?", "en", {"persona-science-001"}),
    ("What is damping?", "en", {"C_DAMP", "U0_DAMP"}),
    ("Who are you?", "en", {"self-001"}),
    ("What is an atom?", "en", {"flp-i-01"}),
    # Physics-domain cards added 2026-11-25 (QED, particles, condensed
    # matter, statistical mechanics, optics, gravitation, computing).
    ("量子电动力学是讲什么的", "zh", {"qed-001"}),
    ("费曼图怎么读", "zh", {"qed-003"}),
    ("路径积分是什么意思", "zh", {"qed-002"}),
    ("电子磁矩为什么不是 2", "zh", {"qed-006"}),
    ("宇称不守恒是谁发现的", "zh", {"part-005"}),
    ("夸克为什么不能单独存在", "zh", {"part-003"}),
    ("部分子模型是什么", "zh", {"part-002"}),
    ("中微子是什么", "zh", {"part-006"}),
    ("库珀对是什么", "zh", {"cm-002"}),
    ("约瑟夫森效应是什么", "zh", {"cm-003"}),
    ("受激辐射是什么", "zh", {"opt-001"}),
    # Optics, added 2026-11-26 after the standard changed from "what the corpus
    # happens to contain" to "what Feynman actually taught": FLP I ch. 26-32
    # (least time, index/dispersion, scattering) and Newton's Opticks for the
    # seven-colour convention. Both rainbow cards are acceptable: one explains
    # dispersion, the other corrects the "seven separate colours" premise.
    ("为什么彩虹总是七种颜色呢", "zh", {"opt-009", "opt-010"}),
    ("天空为什么是蓝色的", "zh", {"opt-012"}),
    ("光为什么在水里拐弯", "zh", {"opt-011"}),
    ("why is a rainbow always seven colours", "en", {"opt-009", "opt-010"}),
    # Remaining FLP I optics block, added 2026-11-26 and since checked
    # against the local New Millennium text: 27 geometrical optics, 34
    # relativistic radiation, 35 colour vision, 36 mechanisms of seeing.
    ("透镜为什么能成像", "zh", {"opt-013"}),
    # "什么是X" must not be blocked by the glue bigram 么是 counting as a
    # second subject token; this locks the _STOP_BIGRAMS entry.
    ("什么是像差", "zh", {"opt-013"}),
    ("同步辐射是什么", "zh", {"opt-014"}),
    ("为什么暗处看不出颜色", "zh", {"opt-015"}),
    ("蜜蜂为什么能看见紫外", "zh", {"opt-016"}),
    ("what is synchrotron radiation", "en", {"opt-014"}),
    ("why can bees see ultraviolet", "en", {"opt-016"}),
    # FLP I ch. 29 interference, 30 diffraction, 33 polarization: dedicated
    # cards written from the local full text, replacing the weak coverage by
    # opt-003 / opt-004. "什么是干涉" tokenises to 是干 + 干涉; it locks the
    # diffraction card's title against carrying that glue bigram again.
    ("干涉是什么", "zh", {"opt-017"}),
    ("什么是干涉", "zh", {"opt-017"}),
    ("衍射是什么", "zh", {"opt-018"}),
    ("光栅是怎么分光的", "zh", {"opt-018"}),
    # Feynman's own answer is "no one has ever been able to define the
    # difference"; the guardrail card says the same thing.
    ("干涉和衍射有什么区别", "zh", {"opt-018", "wave-interference-diffraction"}),
    ("偏振是什么", "zh", {"opt-019"}),
    ("布儒斯特角是什么", "zh", {"opt-019"}),
    # The single-photon card is FLP III 11-4, not chapter 33.
    ("单个光子怎么过偏振片", "zh", {"opt-004"}),
    # One-word English questions tie at 1.0 on every card with the word in
    # its title; these lock the verbatim-alias tie-break. The optics card
    # must win "polarization", and must not steal "vacuum polarization".
    ("what is interference", "en", {"opt-017"}),
    ("what is diffraction", "en", {"opt-018"}),
    ("what is polarization", "en", {"opt-019"}),
    ("what is vacuum polarization", "en", {"qed-007"}),
    # Dielectric polarization is FLP II ch. 10 (P, chi, kappa); before this
    # card "极化是什么" was answered by the vacuum-polarization mention in
    # qed-005. The guardrail card em-dielectric-conductor still owns the
    # "dielectric versus conductor" question and ties on the bare English one.
    ("极化是什么", "zh", {"opt-020"}),
    ("电介质极化是什么", "zh", {"opt-020"}),
    ("介电常数是什么", "zh", {"opt-020"}),
    ("真空极化是什么", "zh", {"qed-007", "qed-005"}),
    ("什么是电介质", "zh", {"em-dielectric-conductor", "opt-020"}),
    ("what is dielectric polarization", "en", {"opt-020"}),
    ("what is the dielectric constant", "en", {"opt-020"}),
    ("what is a dielectric", "en", {"opt-020", "em-dielectric-conductor"}),
    ("what is electric susceptibility", "en", {"opt-020"}),
    ("how does a diffraction grating work", "en", {"opt-018"}),
    ("相变是什么", "zh", {"sm-003"}),
    ("引力波有能量吗", "zh", {"grav-002"}),
    ("量子计算是谁提出的", "zh", {"comp-002"}),
)

# These must not be answered from the cards: either a different layer owns
# them (memory, chat) or the corpus has no honest answer.
NEGATIVES = (
    ("我昨天吃了什么", "zh"),
    ("我们上次聊了什么", "zh"),
    ("讲个笑话", "zh"),
    ("为什么天气会变", "zh"),
    ("为什么冰块浮在水上", "zh"),
    ("tell me a joke", "en"),
    ("what is the capital of France", "en"),
    ("why can't I remember yesterday", "en"),
    ("How does memory work?", "en"),
    ("Why does ice float?", "en"),
)
