"""实体槽残余抽象词门禁：词级黑名单 + 窄语素后缀。

背景：p8b 真字库实体槽 60 颗里有 19 颗是"材料/分量/区域"这类 jieba 判 n、
语法合法但没有可操作画面的边界词。宽后缀（度/量/率/性）不能一刀切——
质量/速度/温度/动量/频率/强度/距离是费曼原话里的常规主语（"温度升高"）。
"""

from plugins.feynman_vault import GENERIC_ENTITY_WORDS, is_pure_noun_phrase, entity_tier


def test_generic_abstract_words_blocked():
    blocked = [
        # 上位泛指 / 集合总称
        "材料", "原料", "物体", "事物", "产物", "装置", "设备", "仪器",
        "器件", "工具", "介质", "载体", "样本", "样品", "零件", "元件",
        # 数学与度量分划（注意：分量 拦，质量/速度 不拦）
        "分量", "矢量", "标量", "张量", "坐标", "投影", "参数", "变量",
        "常量", "系数", "数值", "读数", "比例", "差值", "总和", "均值",
        # 时空与框架关系
        "方向", "位置", "地点", "表面", "内部", "外部", "中心", "终点",
        "起点", "轨迹", "路径", "区域", "范围", "领域", "原理", "模型",
        # 泛化 -体/-物（白名单之外的）
        "整体", "个体", "主体", "客体",
    ]
    for w in blocked:
        assert is_pure_noun_phrase(w) is False, f"残余抽象词未被拦截: {w}"


def test_concrete_quantities_and_objects_survive():
    kept = [
        # 物理量：能量能升高、速度能变快，是费曼真在用的主语
        "质量", "速度", "温度", "动量", "频率", "强度", "距离", "概率",
        # 具体物件
        "电子", "光子", "磁铁", "弹簧", "弹珠", "棱镜", "线圈", "咖啡杯",
        "水面", "磁体", "晶体", "流体", "障碍物", "重物", "狭缝", "屏幕",
    ]
    for w in kept:
        assert is_pure_noun_phrase(w) is True, f"合法实体被误杀: {w}"


def test_morpheme_suffix_blocks_compounds():
    """黑名单之外的复合词：靠语素后缀拦（材料/物件类收尾）。"""
    for w in ("金属材料", "实验装置", "测量仪器", "某种介质", "运动轨迹"):
        assert is_pure_noun_phrase(w) is False, f"复合抽象词未被拦截: {w}"


def test_entity_tiers():
    """边界抽象词不判死刑：第一梯队具体物，第二梯队留待搭配证据，其余不收。"""
    assert entity_tier("电子") == 1
    assert entity_tier("狭缝") == 1
    assert entity_tier("表面") == 2
    assert entity_tier("物体") == 2
    assert entity_tier("分量") == 2
    assert entity_tier("基本") == 0     # 形容词素不进兜底队列
    assert entity_tier("金属材料") == 0  # 复合衍生物连第二梯队都不进


def test_tier2_sorts_after_tier1():
    """第二梯队不挤第一梯队：相位再合也不排到具体物前面。"""
    from plugins.feynman_typesetter import _select_per_slot

    class FakeVault:
        slot_offsets = {"hook": 0.0, "entity": 0.0, "action": 0.0, "consequence": 0.0}
        by_slot = {"hook": [], "action": [], "consequence": [], "entity": [
            {"word": "表面", "count": 90, "phase_deg": 60.0, "tier": 2, "amp": 0.9},
            {"word": "弹珠", "count": 10, "phase_deg": 10.0, "tier": 1, "amp": 1.0},
        ]}

    picked = _select_per_slot(FakeVault(), 60.0, per_slot=2, topic="")
    assert [g["word"] for g in picked["entity"]] == ["弹珠", "表面"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 实体槽抽象词门禁 {len(tests)} 项回归全部通过")
