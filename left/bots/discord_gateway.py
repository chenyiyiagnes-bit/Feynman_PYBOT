"""Discord 轻量网关：把总线当唯一大脑，网关只做消息转发与按钮。

设计约束（与 doc-6 的草稿不同，这里按真实响应字段写）：
- 回复取 `printed_collapse`，不取不存在的 `sentence/slots`；
- 按钮参数取 `paths[0].glyphs.entity/action` 与 `skeleton.pattern_type`；
- 留白句不出按钮，避免发一条必然 400 的反馈；
- `/reset` 真调 `POST /feynman/session/reset`，不是只回一句"已重置"；
- 不引入 discord.py 作为测试依赖：没装时模块仍可导入，纯函数与 HTTP
  助手可离线测；`bot.run()` 只在手动执行且 token 存在时触发。

启动（手动）：
    pip install discord.py httpx
    PYTHONPATH=. python bots/discord_gateway.py
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import httpx

try:  # 本地 .env 里的 token 不必每开一个终端就 export 一次
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BUS_URL = os.getenv("DEFAULT_BUS_URL", "http://127.0.0.1:8000").rstrip("/")
BLANK_GLYPH = "【此处没有可测的东西】"
SLOTS = ("hook", "entity", "action", "consequence")

try:  # 测试环境通常不装 discord.py
    import discord
    from discord import app_commands
    from discord.ext import commands
except Exception:  # pragma: no cover - 只在没有 discord.py 时走这里
    discord = None
    app_commands = None
    commands = None


def system_proxy() -> Optional[str]:
    """Discord 被墙时走本机代理；优先 DISCORD_PROXY，其次 macOS 系统代理。
    只影响连 Discord 的这条连接，总线仍是 127.0.0.1 直连。"""
    env = os.getenv("DISCORD_PROXY", "").strip()
    if env:
        return env
    if os.uname().sysname != "Darwin":
        return None
    try:
        out = subprocess.run(["scutil", "--proxy"], capture_output=True,
                             text=True, timeout=2).stdout
    except Exception:
        return None
    host = re.search(r"HTTPSProxy\s*:\s*(\S+)", out)
    port = re.search(r"HTTPSPort\s*:\s*(\d+)", out)
    if host and port:
        return f"http://{host.group(1)}:{port.group(1)}"
    return None


def session_id_for(channel_id: object, user_id: object) -> str:
    """频道 + 用户自动绑定，手机端不用手输 session_id。"""
    return f"discord-{channel_id}-{user_id}"


def stable_nonce(text: str) -> int:
    """跨进程稳定：不用 Python 的随机化 hash()，确定性契约不能破。"""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10000


def print_payload(query: str, session_id: str, nonce: Optional[int] = None) -> dict:
    return {"user_query": query, "style_deg": 55.0, "operators": {},
            "nonce": stable_nonce(query) if nonce is None else int(nonce),
            "session_id": session_id}


def paragraph_payload(query: str, session_id: str, nonce: Optional[int] = None) -> dict:
    return {"user_query": query, "style_deg": 55.0,
            "operators": {"cross_analogy": 1.2, "jargon_strip": 1.3},
            "nonce": stable_nonce(query) if nonce is None else int(nonce),
            "session_id": session_id}


def is_blank(data: dict) -> bool:
    """留白：没坍缩或文本就是留白字粒。"""
    text = (data.get("printed_collapse") or "").strip()
    return not data.get("collapsed") or not text or text == BLANK_GLYPH


def top_path(data: dict) -> dict:
    paths = data.get("paths") or []
    return paths[0] if paths and isinstance(paths[0], dict) else {}


def feedback_payload(data: dict, vote: str, session_id: str = "") -> Optional[dict]:
    """实体/动作任一为空就返回 None：客户端自己挡下，不发必然 400 的请求。"""
    if vote not in ("up", "down") or is_blank(data):
        return None
    entities = [str(e) for e in (data.get("user_entities") or []) if str(e).strip()]
    action = ((top_path(data).get("glyphs") or {}).get("action") or "").strip()
    if not entities or not action:
        return None
    skeleton = ((data.get("skeleton") or {}).get("pattern_type") or "")
    return {"vote": vote, "entities": entities[:3], "action": action,
            "skeleton": skeleton, "session_id": session_id or None}


def source_lines(data: dict) -> str:
    """按钮"查出处"的私密内容：四槽字粒 + 语料来源。"""
    path = top_path(data)
    glyphs = path.get("glyphs") or {}
    sources = path.get("sources") or {}
    lines = ["**📚 零幻觉字粒原典溯源：**"]
    for slot in SLOTS:
        word = glyphs.get(slot)
        if not word:
            continue
        srcs = sources.get(slot) or []
        src = srcs[0] if srcs else "unknown"
        lines.append(f"- `{slot.upper()}`：**{word}** ← *{src}*")
    lines.append(f"\n`ext_hash`: `{data.get('ext_hash', '00000000')}`")
    return "\n".join(lines)


def format_print_reply(data: dict) -> str:
    text = (data.get("printed_collapse") or "").strip()
    return f"⚛️ {text}" if text else "⚛️ （总线没有返回可打印的字粒）"


def format_paragraph_reply(data: dict) -> str:
    labels = {0: "现象锚点", 1: "92° 类比", 2: "15° 收束"}
    rows = []
    for s in data.get("sentences") or []:
        line = (s.get("line") or "").strip()
        if not line:
            continue
        rows.append(f"**{labels.get(s.get('stage'), '句子')}**：{line}")
    if not rows:
        return "⚛️ （候选池凑不齐三句，这次先只给留白）"
    return "⚛️ **费曼三段式讲解**\n\n" + "\n\n".join(rows)


def paragraph_view_data(data: dict) -> dict:
    """把段落结果摊成按钮 / 查出处认的结构；三句同出一次候选池。"""
    sentences = [s for s in (data.get("sentences") or [])
                 if (s.get("line") or "").strip()]
    if not sentences:
        return {}
    first = sentences[0]
    return {
        "collapsed": True,
        "printed_collapse": data.get("paragraph") or first["line"],
        "paragraph": data.get("paragraph") or "",
        "user_entities": data.get("user_entities") or [],
        "paths": [{"glyphs": first.get("glyphs") or {},
                   "sources": first.get("sources") or {}}],
        "skeleton": {"pattern_type": first.get("kind") or ""},
        "ext_hash": data.get("ext_hash", "00000000"),
    }


def format_paragraph_plain(data: dict) -> str:
    """Discord 里直接读的版本：不加"现象锚点/类比/收束"这类调试标签。"""
    text = (data.get("paragraph") or "").strip()
    return f"⚛️ {text}" if text else "⚛️ （候选池凑不齐三句，这次先只给留白）"


_TURNS: dict = {}


def turn_nonce(text: str, session_id: str) -> int:
    """同一句重复问也要换一次排版；按会话计次，进程内确定、重启后从头来。"""
    turn = int(_TURNS.get(session_id, 0))
    _TURNS[session_id] = turn + 1
    return (stable_nonce(text) + 7 * turn) % 10000


async def post_json(client: httpx.AsyncClient, path: str, payload: dict) -> Tuple[int, dict]:
    r = await client.post(f"{BUS_URL}{path}", json=payload)
    try:
        body = r.json()
    except Exception:
        body = {}
    return r.status_code, body


async def ask_print(client: httpx.AsyncClient, query: str, session_id: str,
                    nonce: Optional[int] = None) -> Tuple[int, dict]:
    return await post_json(client, "/feynman/print",
                           print_payload(query, session_id, nonce))


async def ask_paragraph(client: httpx.AsyncClient, query: str, session_id: str,
                        nonce: Optional[int] = None) -> Tuple[int, dict]:
    return await post_json(client, "/feynman/paragraph",
                           paragraph_payload(query, session_id, nonce))


async def send_feedback(client: httpx.AsyncClient, data: dict, vote: str,
                        session_id: str = "") -> Optional[dict]:
    payload = feedback_payload(data, vote, session_id)
    if payload is None:
        return None
    status, body = await post_json(client, "/feynman/feedback", payload)
    return body if status == 200 else None


async def reset_session(client: httpx.AsyncClient, session_id: str) -> Tuple[int, dict]:
    return await post_json(client, "/feynman/session/reset",
                           {"session_id": session_id})


async def digest_text(client: httpx.AsyncClient, text: str, session_id: str) -> Tuple[int, dict]:
    return await post_json(client, "/feynman/digest",
                           {"source_type": "discord_share", "title": text[:15],
                            "raw_content": text, "session_id": session_id})


def strip_mention(text: str) -> str:
    """去掉用户提及、身份组提及和频道提及，剩下的才是问题正文。"""
    text = re.sub(r"<@[!&]?\d+>", "", text or "")
    return re.sub(r"<#\d+>", "", text).strip()


def mentions_bot(message, bot) -> bool:
    """Discord 里 @机器人 常常命中它的身份组（<@&role>），不是用户提及；
    两种都要认，否则用户明明 @ 了，网关却当没叫它。"""
    if bot is None or bot.user is None:
        return False
    if bot.user in message.mentions:
        return True
    bot_id = str(bot.user.id)
    for role in message.role_mentions:
        tags = getattr(role, "tags", None)
        if getattr(tags, "bot_id", None) is not None and str(tags.bot_id) == bot_id:
            return True
        me = getattr(message.guild, "me", None) if message.guild else None
        if me is not None and role in me.roles:
            return True
    return False


if discord is not None:
    class FeedbackView(discord.ui.View):
        def __init__(self, data: dict, query: str, session_id: str,
                     nonce: Optional[int] = None, mode: str = "paragraph"):
            super().__init__(timeout=600)
            self.data = data
            self.query = query
            self.session_id = session_id
            self.mode = mode
            self.nonce = stable_nonce(query) if nonce is None else int(nonce)

        @discord.ui.button(label="👍 妙哉", style=discord.ButtonStyle.success)
        async def upvote(self, interaction, button):  # noqa: ANN001
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                await send_feedback(client, self.data, "up", self.session_id)
            button.disabled = True
            button.label = "👍 已记住共鸣"
            await interaction.response.edit_message(view=self)

        @discord.ui.button(label="👎 生硬·换一句", style=discord.ButtonStyle.danger)
        async def downvote(self, interaction, button):  # noqa: ANN001
            async with httpx.AsyncClient(timeout=12.0, trust_env=False) as client:
                await send_feedback(client, self.data, "down", self.session_id)
                self.nonce = (self.nonce + 1) % 10000
                if self.mode == "paragraph":
                    status, fresh = await ask_paragraph(client, self.query,
                                                        self.session_id, self.nonce)
                    text = format_paragraph_plain(fresh)
                    view_data = paragraph_view_data(fresh)
                else:
                    status, fresh = await ask_print(client, self.query,
                                                    self.session_id, self.nonce)
                    text = format_print_reply(fresh)
                    view_data = fresh
            if status == 200:
                self.data = view_data or fresh
                await interaction.response.edit_message(content=text, view=self)
            else:
                await interaction.response.send_message(
                    "⚠️ 重排失败，请稍后再试。", ephemeral=True)

        @discord.ui.button(label="🔍 查出处", style=discord.ButtonStyle.secondary)
        async def source(self, interaction, button):  # noqa: ANN001
            await interaction.response.send_message(source_lines(self.data),
                                                    ephemeral=True)


    class FeynmanBot(commands.Bot):
        def __init__(self, proxy=None):
            intents = discord.Intents.default()
            intents.message_content = True
            super().__init__(command_prefix="!", intents=intents, proxy=proxy)

        async def setup_hook(self):
            await self.tree.sync()


    bot = FeynmanBot(system_proxy())

    @bot.tree.command(name="explain", description="三段式讲透一个问题")
    @app_commands.describe(question="你想探索的问题")
    async def explain_cmd(interaction, question: str):  # noqa: ANN001
        await interaction.response.defer(thinking=True)
        sid = session_id_for(interaction.channel_id, interaction.user.id)
        nonce = turn_nonce(question, sid)
        async with httpx.AsyncClient(timeout=12.0, trust_env=False) as client:
            status, data = await ask_paragraph(client, question, sid, nonce)
        text = (format_paragraph_plain(data) if status == 200
                else "⚠️ 量子活字总线暂时开小差了，请稍后再试。")
        await interaction.followup.send(text)

    @bot.tree.command(name="reset", description="清空当前对话的会话记忆")
    async def reset_cmd(interaction):  # noqa: ANN001
        sid = session_id_for(interaction.channel_id, interaction.user.id)
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            status, data = await reset_session(client, sid)
        _TURNS.pop(sid, None)
        if status == 200 and data.get("cleared"):
            msg = "🔄 会话波函数已重置，随时开始新话题。"
        elif status == 200:
            msg = "🔄 这个会话本来就还没有记忆，可以重新开始。"
        else:
            msg = "⚠️ 重置失败，请稍后再试。"
        await interaction.response.send_message(msg, ephemeral=True)

    @bot.event
    async def on_message(message):  # noqa: ANN001
        if message.author.bot:
            return
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentioned = mentions_bot(message, bot)
        if not (is_dm or mentioned):
            return
        text = strip_mention(message.content)
        if not text:
            return
        sid = session_id_for(message.channel.id, message.author.id)
        async with message.channel.typing():
            async with httpx.AsyncClient(timeout=12.0, trust_env=False) as client:
                if len(text) > 80:
                    status, data = await digest_text(client, text, sid)
                    if status == 200:
                        await message.reply(
                            f"📖 **费曼读后感**：\n{data.get('feynman_reflection', '')}")
                    else:
                        await message.reply("⚠️ 这段分享暂时消化不了，请稍后再试。")
                    return
                nonce = turn_nonce(text, sid)
                status, data = await ask_paragraph(client, text, sid, nonce)
        if status != 200:
            await message.reply("⚠️ 量子活字总线暂时开小差了，请稍后再试。")
            return
        view_data = paragraph_view_data(data)
        view = (FeedbackView(view_data, text, sid, nonce=nonce, mode="paragraph")
                if view_data else None)
        await message.reply(format_paragraph_plain(data), view=view)
else:  # pragma: no cover - 没有 discord.py 时只暴露纯函数
    bot = None


def main() -> None:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if discord is None:
        raise SystemExit("未安装 discord.py：pip install discord.py 后再启动网关。")
    if not token:
        raise SystemExit("请在 .env 中设置 DISCORD_BOT_TOKEN 后再启动 Discord 网关。")
    bot.run(token)


if __name__ == "__main__":
    main()
