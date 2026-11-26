"""字库语料的会话格式（chat JSONL）适配：assistant 正文才是费曼的话。

审计后的库是 {"messages":[system,user,assistant],"meta":{...}}，旧铸字器只认
{"source","section","text"}。这里验证自动识别：中文 assistant 正文入样、
英文行与去世词行照旧被过滤，旧格式逐字节不受影响。
"""

import json
import tempfile
from pathlib import Path

from plugins.feynman_vault import iter_sections

ZH = ("你问为什么磁铁能吸铁，其实并没有一种叫磁的东西流出来。"
      "磁场的想法不过是说：把电荷放在那里，它会受到一个力。")
EN = ("You ask why a magnet attracts iron. There is no stuff called magnetism "
      "flowing out; the field is only a way of saying a charge feels a force there.")


def _row(lang, text, work_name="Surely You're Joking", span="syj__001:1-3",
         messages=None):
    return {
        "uid": f"{work_name}-{span}-{lang}",
        "messages": messages or [
            {"role": "system", "content": "你是理查德·费曼。只依据有据可查的原始材料作答。"},
            {"role": "user", "content": "磁铁为什么能吸铁？"},
            {"role": "assistant", "content": text},
        ],
        "meta": {"lang": lang, "work_name": work_name, "span_id": span,
                 "training_ready": True},
    }


def _write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def test_chat_rows_use_assistant_text():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "chat.jsonl"
        _write(corpus, [
            _row("zh", ZH, work_name="QED 讲义", span="qed__004:12-15"),
            _row("en", EN),
        ])
        got = list(iter_sections(corpus, trusted=True))
        assert len(got) == 1, got
        src, section, text = got[0]
        assert src == "QED 讲义", src
        assert section == "qed__004:12-15", section
        assert text == ZH, text


def test_chat_english_rows_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "chat.jsonl"
        _write(corpus, [_row("en", EN)])
        assert list(iter_sections(corpus, trusted=True)) == []


def test_chat_death_marker_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "chat.jsonl"
        _write(corpus, [_row("zh", ZH + "他去世以后，大家还记得这场讲座。")])
        assert list(iter_sections(corpus, trusted=True)) == []


def test_old_format_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "legacy.jsonl"
        corpus.write_text(json.dumps({
            "source": "费恩曼物理学讲义", "section": "第一卷 1-2",
            "text": ZH,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        got = list(iter_sections(corpus, trusted=True))
        assert len(got) == 1, got
        assert got[0] == ("费恩曼物理学讲义", "第一卷 1-2", ZH), got


def test_chat_multiple_assistant_messages_joined():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "chat.jsonl"
        row = _row("zh", "", messages=[
            {"role": "system", "content": "你是理查德·费曼。"},
            {"role": "user", "content": "磁铁呢？"},
            {"role": "assistant", "content": ZH[:20]},
            {"role": "user", "content": "那电场呢？"},
            {"role": "assistant", "content": ZH[20:]},
        ])
        _write(corpus, [row])
        got = list(iter_sections(corpus, trusted=True))
        assert len(got) == 1, got
        assert got[0][2].replace("\n", "") == ZH, got[0][2]


def test_chat_row_without_meta_lang_defaults_to_zh():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "chat.jsonl"
        row = _row("zh", ZH)
        row["meta"].pop("lang")
        _write(corpus, [row])
        assert len(list(iter_sections(corpus, trusted=True))) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"✅ {name}")
    print("全部通过")
