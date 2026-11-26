"""HTTP + WebSocket server and a tiny dashboard, standard library only.

The mind lives in one process behind one lock: every request sees the same
engine, and a chat turn cannot interleave with a snapshot. WebSocket is a
minimal RFC 6455 implementation (text frames, ping/pong, close) — enough for
the dashboard and for scripts that want a live stream of replies.

    python -m omega_mind --serve 8765
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from . import profiler
from .engine import MindEngine
from .feynman_kb import DEFAULT_DIRS
from .cube import render_net

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_FRAME = 1 << 20  # 1 MiB


def _state_payload(engine: MindEngine) -> Dict[str, Any]:
    dream = engine.dreamer.last()
    return {
        "persona": engine.persona.name,
        "language": engine.language,
        "turn": engine.turn_index,
        "state": engine.state.snapshot(),
        "relations": engine.relations.snapshot(),
        "memory": {
            "working": len(engine.tiers.working),
            "episodes": len(engine.tiers.episodes),
            "undreamed": len(engine.tiers.undreamed()),
            "semantic": dict(engine.tiers.semantic),
        },
        "dreams": len(engine.dreamer.dreams),
        "last_dream": dream.to_dict() if dream else None,
        "cube": {"net": render_net(engine.cube), "history": engine.cube.history[-8:]},
        "resonance": engine.resonance_snapshot(),
        "left_brain": engine.last_left_brain,
        "retrieval": engine.last_retrieval,
        "challenge": (engine.challenge_session.status()
                      if engine.challenge_session is not None else {}),
    }


DASHBOARD = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>OmegaMind</title>
<style>
body{font-family:ui-monospace,Menlo,monospace;background:#111;color:#ddd;margin:2rem}
h1{font-size:1rem;letter-spacing:.2em;color:#8bd}
pre{background:#1a1a1a;padding:1rem;border-radius:6px;white-space:pre-wrap}
#log div{margin:.2rem 0}#log .u{color:#8bd}#log .m{color:#ce8}
input{width:70%;background:#1a1a1a;border:1px solid #333;color:#ddd;padding:.5rem}
button{background:#2a4a5a;color:#ddd;border:0;padding:.5rem 1rem}
.row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
</style></head><body>
<h1>OMEGAMIND <small id="lang"></small></h1>
<div><button onclick="setLang('zh')">中文</button>
<button onclick="setLang('en')">English</button>
<button onclick="startChallenge()">挑战</button></div>
<div id="log"></div>
<form id="f"><input id="t" autocomplete="off" placeholder="说点什么"> <button>发送</button></form>
<div class="row"><div><h3>cube</h3><pre id="net"></pre></div>
<div><h3>state</h3><pre id="st"></pre><h3>relations</h3><pre id="rel"></pre></div></div>
<script>
async function setLang(lang){
  await fetch('/api/lang',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({lang})});refresh();
}
async function refresh(){
  const s = await (await fetch('/api/state')).json();
  document.getElementById('lang').textContent = '['+s.language+']';
  document.getElementById('st').textContent = JSON.stringify(
    {state:s.state,left_brain:s.left_brain,retrieval:s.retrieval,
     resonance:s.resonance},null,1);
  document.getElementById('net').textContent = s.cube.net;
  document.getElementById('rel').textContent = Object.values(s.relations.nodes)
    .map(n=>n.name+' a='+n.affinity.toFixed(2)).join('\\n');
}
function add(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;
  document.getElementById('log').appendChild(d);}
let challengeOn=false;
async function startChallenge(){
  const r=await (await fetch('/api/challenge',{method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify({action:'start'})})).json();
  if(r.error){add('m',r.error);return;}
  challengeOn=true;add('m','他> '+r.question);refresh();
}
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const i=document.getElementById('t');const text=i.value.trim();
  if(!text)return;i.value='';add('u','你> '+text);
  if(challengeOn){
    const r=await (await fetch('/api/challenge',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({action:'answer',answer:text})})).json();
    add('m','他> '+(r.feedback||r.error||''));
    if(r.next_question){add('m','他> '+r.next_question);}
    if(!r.status||!r.status.active){challengeOn=false;
      add('m','(challenge done)');}
    refresh();return;
  }
  const r=await (await fetch('/api/chat',{method:'POST',
    headers:{'content-type':'application/json'},body:JSON.stringify({text})})).json();
  add('m','他> '+r.reply);refresh();
};
let timer=null;
function schedule(){
  if(timer){clearInterval(timer);timer=null;}
  if(!document.hidden){timer=setInterval(refresh,3000);}
}
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();schedule();});
refresh();schedule();
</script></body></html>
"""


def _ws_accept(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()


class _WSClosed(Exception):
    pass


def _read_exact(rfile, n: int) -> bytes:
    data = rfile.read(n)
    if data is None or len(data) != n:
        raise _WSClosed()
    return data


def _read_frame(rfile) -> Tuple[int, bytes]:
    """Read one client frame; returns (opcode, payload)."""
    b1, b2 = _read_exact(rfile, 2)
    opcode = b1 & 0x0F
    masked = b2 & 0x80
    length = b2 & 0x7F
    if length == 126:
        length = int.from_bytes(_read_exact(rfile, 2), "big")
    elif length == 127:
        length = int.from_bytes(_read_exact(rfile, 8), "big")
    if length > MAX_FRAME:
        raise _WSClosed()
    mask = _read_exact(rfile, 4) if masked else b"\0\0\0\0"
    payload = bytearray(_read_exact(rfile, length))
    if masked:
        for i in range(len(payload)):
            payload[i] ^= mask[i % 4]
    return opcode, bytes(payload)


def _frame(opcode: int, payload: bytes) -> bytes:
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([n])
    elif n < (1 << 16):
        header += bytes([126]) + n.to_bytes(2, "big")
    else:
        header += bytes([127]) + n.to_bytes(8, "big")
    return header + payload


class MindHandler(BaseHTTPRequestHandler):
    server_version = "OmegaMind/1.0"
    engine: MindEngine
    lock: threading.Lock

    # -- helpers -----------------------------------------------------------

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, body: str, status: int = 200, ctype: str = "text/html; charset=utf-8") -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # quieter by default
        pass

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._text(DASHBOARD)
        if path == "/ws":
            return self._websocket()
        with self.lock:
            if path == "/api/state":
                return self._json(_state_payload(self.engine))
            if path == "/api/cube":
                return self._json({"net": render_net(self.engine.cube),
                                   "history": self.engine.cube.history})
            if path == "/api/memory":
                return self._json(self.engine.tiers.snapshot())
            if path == "/api/relations":
                return self._json(self.engine.relations.snapshot())
            if path == "/api/dreams":
                return self._json(self.engine.dreamer.snapshot())
            if path == "/api/profile":
                kb_dir = getattr(self.engine.kb, "dir", None)
                dirs = [kb_dir] if kb_dir else list(DEFAULT_DIRS)
                out = [profiler.profile_corpus(d) for d in dirs if d.is_dir()]
                return self._json({"profiles": out})
            if path == "/api/challenge":
                return self._json(self.engine.challenge().status())
        return self._json({"error": "not found", "path": path}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, status=400)
        if path == "/api/lang":
            lang = str(body.get("lang") or "")
            try:
                with self.lock:
                    self.engine.set_language(lang)
            except ValueError:
                return self._json({"error": f"language must be zh or en"}, status=400)
            return self._json({"language": self.engine.language})
        if path == "/api/chat":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._json({"error": "empty text"}, status=400)
            user = str(body.get("user") or "default")
            with self.lock:
                result = self.engine.turn(text, user=user)
                payload = {
                    "reply": result.reply,
                    "source": result.source,
                    "directive": asdict(result.directive),
                    "state": result.state,
                    "resonance": result.resonance,
                    "left_brain": result.left_brain,
                    "turn": self.engine.turn_index,
                }
            return self._json(payload)
        if path == "/api/turn-cube":
            move = str(body.get("move") or "")
            with self.lock:
                try:
                    moved = self.engine.turn_cube(move)
                except (ValueError, KeyError):
                    return self._json({"error": f"bad move: {move}"}, status=400)
                return self._json({"moved": moved, "net": render_net(self.engine.cube)})
        if path == "/api/dream":
            with self.lock:
                dream = self.engine.dream(force=True)
            return self._json({"dream": dream.to_dict() if dream else None})
        if path == "/api/challenge":
            action = str(body.get("action") or
                         ("answer" if body.get("answer") else "start"))
            with self.lock:
                session = self.engine.challenge()
                if not session.available():
                    return self._json(
                        {"error": "no teachable units in the knowledge base"},
                        status=503)
                if action == "start":
                    question = session.next_unit()
                    if self.engine.auto_save:
                        self.engine.save()
                    return self._json({"question": question,
                                       "status": session.status()})
                if action != "answer":
                    return self._json({"error": f"unknown action: {action}"},
                                      status=400)
                answer = str(body.get("answer") or "").strip()
                if not answer:
                    return self._json({"error": "empty answer"}, status=400)
                grade = session.submit(answer)
                payload = {
                    "verdict": grade.verdict,
                    "score": grade.score,
                    "feedback": grade.feedback,
                    "matched": grade.matched,
                    "missing": grade.missing,
                    "revealed": grade.revealed,
                    "status": session.status(),
                }
                if session.unit is None:
                    payload["next_question"] = session.next_unit()
                    payload["status"] = session.status()
                if self.engine.auto_save:
                    self.engine.save()
                return self._json(payload)
        return self._json({"error": "not found", "path": path}, status=404)

    # -- websocket ---------------------------------------------------------

    def _websocket(self) -> None:
        key = self.headers.get("sec-websocket-key")
        if not key:
            return self._json({"error": "websocket upgrade required"}, status=400)
        self.send_response(101)
        self.send_header("upgrade", "websocket")
        self.send_header("connection", "Upgrade")
        self.send_header("sec-websocket-accept", _ws_accept(key))
        self.end_headers()
        self.close_connection = True
        rfile = self.rfile
        try:
            while True:
                opcode, payload = _read_frame(rfile)
                if opcode == 0x8:  # close
                    self.wfile.write(_frame(0x8, b""))
                    break
                if opcode == 0x9:  # ping
                    self.wfile.write(_frame(0xA, payload))
                    continue
                if opcode != 0x1:  # only text is meaningful here
                    continue
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self.wfile.write(_frame(0x1, json.dumps(
                        {"error": "bad json"}, ensure_ascii=False).encode()))
                    continue
                text = str(msg.get("text") or "").strip()
                user = str(msg.get("user") or "default")
                lang = msg.get("lang")
                with self.lock:
                    if lang in ("zh", "en"):
                        self.engine.set_language(lang)
                if not text:
                    reply = {"error": "empty text"}
                else:
                    with self.lock:
                        result = self.engine.turn(text, user=user)
                        reply = {"reply": result.reply, "source": result.source,
                                 "directive": asdict(result.directive),
                                 "state": result.state,
                                 "resonance": result.resonance,
                                 "left_brain": result.left_brain,
                                 "language": self.engine.language,
                                 "turn": self.engine.turn_index}
                self.wfile.write(_frame(0x1, json.dumps(reply, ensure_ascii=False).encode()))
        except (_WSClosed, ConnectionError, OSError):
            pass

    def handle_one_request(self) -> None:
        # ThreadingHTTPServer handles concurrency; keep BaseHTTPRequestHandler's
        # HTTP/1.0 close semantics but tolerate clients that keep sockets open.
        super().handle_one_request()


def build_server(engine: MindEngine, host: str = "127.0.0.1",
                 port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (MindHandler,), {
        "engine": engine,
        "lock": threading.Lock(),
    })
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(engine: MindEngine, host: str = "127.0.0.1", port: int = 8765,
          idle_interval: float = 30.0) -> None:
    httpd = build_server(engine, host, port)
    handler = httpd.RequestHandlerClass
    stop = threading.Event()

    def idle_loop() -> None:
        """Dreams and proactive speech need a clock, not a user request."""
        while not stop.wait(idle_interval):
            with handler.lock:
                dream = engine.maintenance()
                message = engine.tick()
                if dream is not None or message:
                    engine.save()

    threading.Thread(target=idle_loop, daemon=True, name="mind-idle").start()
    print(f"OmegaMind serving on http://{host}:{port}  (dashboard /, ws /ws)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        engine.save()
