"""P18 Discord 网关：离线用假总线测，不需要 bot token，也不装 discord.py。

锁住：
1. session_id 自动按 频道+用户 生成；
2. 反馈参数取自真实字段（user_entities / paths[0].glyphs.action / skeleton.pattern_type）；
3. 留白或空实体/动作时不出按钮、不发必然 400 的请求；
4. `/reset` 真打到 `/feynman/session/reset`，并按 cleared 区分文案；
5. 段落回复只从真实 `sentences[].line` 取值；
6. 无 discord.py 时模块仍可导入，`bot is None`、`main()` 明确报错而不是崩栈。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_discord_gateway.py
"""

import asyncio

import bots.discord_gateway as gw


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class FakeClient:
    """记录每一次调用，按 path 返回预设响应。"""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    async def post(self, url, json=None):
        path = url.replace(gw.BUS_URL, "")
        self.calls.append((path, json))
        if path not in self.routes:
            return FakeResponse(404, {"detail": "not found"})
        status, body = self.routes[path]
        return FakeResponse(status, body)


def _print_ok():
    return {
        "collapsed": True,
        "printed_collapse": "电子这个家伙，一动就有脾气。",
        "user_entities": ["电子"],
        "skeleton": {"pattern_type": "转折骨架"},
        "ext_hash": "abc12345",
        "paths": [{
            "glyphs": {"hook": "你看", "entity": "电子", "action": "振动",
                       "consequence": "辐射出光"},
            "sources": {"hook": ["讲义I-2"], "entity": ["讲义I-2"],
                        "action": ["讲义I-3"], "consequence": ["讲义I-2"]},
        }],
    }


def test_session_id_auto():
    assert gw.session_id_for(123, 456) == "discord-123-456"
    assert gw.session_id_for("dm", "u") == "discord-dm-u"


def test_print_then_feedback_payload():
    data = _print_ok()
    fb = gw.feedback_payload(data, "up", "discord-1-2")
    assert fb == {"vote": "up", "entities": ["电子"], "action": "振动",
                  "skeleton": "转折骨架", "session_id": "discord-1-2"}
    assert gw.feedback_payload(data, "down")["session_id"] is None
    assert gw.feedback_payload(data, "sideways") is None


def test_blank_and_missing_fields_mean_no_buttons():
    blank = dict(_print_ok(), printed_collapse="【此处没有可测的东西】")
    assert gw.is_blank(blank) is True
    assert gw.feedback_payload(blank, "up") is None

    uncollapsed = dict(_print_ok(), collapsed=False)
    assert gw.is_blank(uncollapsed) is True
    assert gw.feedback_payload(uncollapsed, "up") is None

    no_entity = dict(_print_ok(), user_entities=[])
    assert gw.feedback_payload(no_entity, "up") is None

    no_action = dict(_print_ok(), paths=[{"glyphs": {"entity": "电子"}}])
    assert gw.feedback_payload(no_action, "up") is None

    good = _print_ok()
    view_needed = not gw.is_blank(good) and gw.feedback_payload(good, "up") is not None
    assert view_needed is True


def test_source_lines_use_real_fields():
    text = gw.source_lines(_print_ok())
    assert "**电子**" in text and "*讲义I-2*" in text and "ABC" not in text
    assert "`ext_hash`: `abc12345`" in text
    assert gw.source_lines({"paths": []}).count("ext_hash") == 1


def test_paragraph_reply_from_real_sentences():
    data = {"paragraph": "三段", "sentences": [
        {"stage": 0, "line": "先看现象。"},
        {"stage": 1, "line": "再打个比方。"},
        {"stage": 2, "line": "最后落回守恒。"},
    ]}
    text = gw.format_paragraph_reply(data)
    assert "现象锚点" in text and "92° 类比" in text and "15° 收束" in text
    for line in ("先看现象。", "再打个比方。", "最后落回守恒。"):
        assert line in text
    assert "留白" in gw.format_paragraph_reply({"sentences": []})


def test_async_print_feedback_reset():
    routes = {
        "/feynman/print": (200, _print_ok()),
        "/feynman/feedback": (200, {"ok": True, "edge_score": 1.12}),
        "/feynman/session/reset": (200, {"session_id": "discord-1-2",
                                         "cleared": True, "memory": None,
                                         "implicit_turn": True}),
    }
    client = FakeClient(routes)

    async def run():
        status, data = await gw.ask_print(client, "电子为什么会发光", "discord-1-2")
        assert status == 200 and gw.is_blank(data) is False
        body = await gw.send_feedback(client, data, "down", "discord-1-2")
        assert body == {"ok": True, "edge_score": 1.12}
        rs, rb = await gw.reset_session(client, "discord-1-2")
        assert rs == 200 and rb["cleared"] is True and rb["implicit_turn"] is True

    asyncio.run(run())
    paths = [c[0] for c in client.calls]
    assert paths == ["/feynman/print", "/feynman/feedback", "/feynman/session/reset"]
    assert client.calls[0][1]["session_id"] == "discord-1-2"
    assert client.calls[0][1]["user_query"] == "电子为什么会发光"
    assert client.calls[1][1]["action"] == "振动"
    assert client.calls[2][1] == {"session_id": "discord-1-2"}


def test_blank_reply_skips_feedback_request():
    routes = {"/feynman/print": (200, dict(_print_ok(), printed_collapse=""))}
    client = FakeClient(routes)

    async def run():
        status, data = await gw.ask_print(client, "随便", "s")
        assert status == 200 and gw.is_blank(data) is True
        assert await gw.send_feedback(client, data, "up", "s") is None

    asyncio.run(run())
    assert [c[0] for c in client.calls] == ["/feynman/print"]


def test_no_discord_import_is_graceful():
    """当前环境装没装 discord.py 都要成立：没有时模块可导入、main() 明确退出。"""
    saved_discord, saved_bot = gw.discord, gw.bot
    gw.discord, gw.bot = None, None
    try:
        try:
            gw.main()
        except SystemExit as exc:
            assert "discord.py" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError("没有 discord.py 时 main() 必须明确退出")
    finally:
        gw.discord, gw.bot = saved_discord, saved_bot
    assert gw.bot is not None, "本机已装 discord.py，机器人实例应存在"


def test_deterministic_nonce_and_mention_strip():
    assert gw.stable_nonce("电子发光") == gw.stable_nonce("电子发光")
    assert gw.stable_nonce("电子发光") != gw.stable_nonce("电子发光?")
    assert gw.strip_mention("<@123> 电子为什么会发光") == "电子为什么会发光"
    assert gw.strip_mention("<@&456> 电子为什么会发光") == "电子为什么会发光"
    assert gw.strip_mention("<#789> 电子为什么会发光") == "电子为什么会发光"
    assert gw.strip_mention("<@!987> ") == ""


def test_paragraph_plain_and_view_data():
    data = {"paragraph": "先看现象。打个比方，光斑变亮。说到底，温度升高。",
            "user_entities": ["电子"], "ext_hash": "deadbeef",
            "sentences": [{"stage": 0, "kind": "declarative",
                           "line": "先看现象。",
                           "glyphs": {"hook": "你看", "entity": "电子",
                                      "action": "发光", "consequence": "光斑变亮"},
                           "sources": {"entity": ["讲义I-2"]}}]}
    plain = gw.format_paragraph_plain(data)
    assert plain.startswith("⚛️ ") and "现象锚点" not in plain
    view = gw.paragraph_view_data(data)
    assert gw.feedback_payload(view, "up") == {
        "vote": "up", "entities": ["电子"], "action": "发光",
        "skeleton": "declarative", "session_id": None}
    assert "讲义I-2" in gw.source_lines(view)
    assert gw.paragraph_view_data({"sentences": []}) == {}
    assert "留白" in gw.format_paragraph_plain({})


def test_turn_nonce_changes_within_session():
    sid = "unit-turn-nonce"
    gw._TURNS.pop(sid, None)
    first = gw.turn_nonce("电子为什么会发光", sid)
    second = gw.turn_nonce("电子为什么会发光", sid)
    assert first == gw.stable_nonce("电子为什么会发光")
    assert first != second
    assert gw.turn_nonce("电子为什么会发光", "unit-turn-other") == first


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_role_mention_counts_as_mentioning_bot():
    """用户 @ 的常常是 bot 的身份组，不是用户提及；必须认得。"""
    bot = _Obj(user=_Obj(id=111))
    role = _Obj(id=222, tags=_Obj(bot_id=111))
    msg = _Obj(mentions=[], role_mentions=[role], guild=_Obj(me=_Obj(roles=[])))
    assert gw.mentions_bot(msg, bot) is True

    own_role = _Obj(id=333, tags=None)
    msg2 = _Obj(mentions=[], role_mentions=[own_role],
                guild=_Obj(me=_Obj(roles=[own_role])))
    assert gw.mentions_bot(msg2, bot) is True

    other = _Obj(id=444, tags=None)
    msg3 = _Obj(mentions=[], role_mentions=[other],
                guild=_Obj(me=_Obj(roles=[])))
    assert gw.mentions_bot(msg3, bot) is False
    assert gw.mentions_bot(_Obj(mentions=[bot.user], role_mentions=[], guild=None), bot) is True
    assert gw.mentions_bot(msg3, None) is False


def main():
    test_session_id_auto()
    test_print_then_feedback_payload()
    test_blank_and_missing_fields_mean_no_buttons()
    test_source_lines_use_real_fields()
    test_paragraph_reply_from_real_sentences()
    test_async_print_feedback_reset()
    test_blank_reply_skips_feedback_request()
    test_no_discord_import_is_graceful()
    test_deterministic_nonce_and_mention_strip()
    test_paragraph_plain_and_view_data()
    test_turn_nonce_changes_within_session()
    test_role_mention_counts_as_mentioning_bot()
    print("✅ test_discord_gateway: 12/12")


if __name__ == "__main__":
    main()
