"""署名作品登记表：书库白名单之外的费曼本人作品也要进字库。

书库（cpt_zh.jsonl）里除了几本带"费曼"书名的书，还有他本人的论文、讲座、
短文的单篇稿子，文件名里不一定有"费曼"。这些要放行；讲他的书、剧评、
合著者的文章、书的序言版面要拦下——字粒要能当"他本人会说的话"用。
"""

from plugins.feynman_vault import _clean_source


def test_authored_papers_and_talks_pass():
    admitted = [
        "Cargo Cult Science（货物崇拜科学）中文版.md",
        "Plenty of Room at the Bottom（底部有广阔空间）中文版.md",
        "The Value of Science（科学的价值）中文版.md",
        "Feynman's Lost Lecture- The Motion of Planets Around the Sun（费曼遗失的讲座）中文版.md",
        "Simulating Physics with Computers（用计算机模拟物理）中文版.md",
        "Space-Time Approach to Non-Relativistic Quantum Mechanics 1948（非相对论量子力学的时空方法）中文版.md",
        "The Theory of Positrons（正电子理论）中文版.md",
        "THE DEVELOPMENT OF THE SPACE-TIME VIEW OF QUANTUM ELECTRODYNAMICS 1965（量子电动力学时空观的发展）中文版.md",
        "Appendix F - Personal observations on the reliability of the Shuttle（关于航天飞机可靠性的个人观察）中文版.md",
        "Los Alamos from Below（从下面看洛斯阿拉莫斯）中文版.md",
        "Mr. Feynman Goes to Washington（费曼先生去华盛顿）中文版.md",
        "The Dignified Professor（有尊严的教授）中文版.md",
        "The Relation of Science and Religion（科学与宗教的关系）中文版.md",
        "The Problem of Teaching Physics in Latin America（在拉丁美洲教物理的问题）中文版.md",
        "Letters（来信：科学与宗教的新假说）中文版.md",
        "An Historic Moment in Physics（物理学的一个历史性时刻）中文版.md",
        "量子电动力学讲义_理查德·费曼,张邦固,朱重远.md",
    ]
    for src in admitted:
        assert _clean_source(src), src


def test_about_feynman_sources_stay_out():
    rejected = [
        "QED 舞台剧 费曼剧评.md",
        "Resource Letter QEDV-1- The QED vacuum.md",
        "Matthew Sands（费曼讲义合作者）- 同步加速器.md",
        "读者致敬 Robert Leighton（费曼讲义合著者）.md",
        "Robert Leighton 荣休（费曼讲义合作者）.md",
        "QED 前言（Mautner 序、Leighton 序、费曼致谢）中文版.md",
        "费曼传（詹姆斯·格雷克）书评.md",
        "费曼讣告.md",
    ]
    for src in rejected:
        assert not _clean_source(src), src


def test_book_whitelist_still_works():
    assert _clean_source("费恩曼物理学讲义.md")
    assert _clean_source("别闹了，费曼先生！.md")
    assert not _clean_source("费曼学习法导读.md")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"✅ {name}")
    print("全部通过")
