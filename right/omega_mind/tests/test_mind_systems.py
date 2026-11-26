"""Memory tiers, dreams, relations, the physical cube, the profiler and server."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.cube import CubeState, parse_move, render_net
from omega_mind.dream import Dreamer
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.memory import Episode, MemoryTiers, salience_of
from omega_mind.profiler import profile_corpus
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.relations import RelationshipGraph


def offline_engine(persona="feynman-zh") -> MindEngine:
    return MindEngine(
        persona=get_persona(persona),
        store_path=tempfile.mktemp(suffix=".json"),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
    )


def ep(turn: int, text: str = "x", salience: float = 1.0, dreamed: bool = False) -> Episode:
    return Episode(turn=turn, now=float(turn), user="u", text=text,
                   reply="r", source="s", salience=salience, dreamed=dreamed)


class TestMemoryTiers(unittest.TestCase):
    def test_working_is_bounded(self):
        tiers = MemoryTiers()
        for i in range(50):
            tiers.record(ep(i))
        self.assertEqual(len(tiers.working), 16)
        self.assertEqual(len(tiers.episodes), 50)
        self.assertEqual(tiers.recent(1)[0].turn, 49)

    def test_facts_promoted_to_semantic(self):
        tiers = MemoryTiers()
        tiers.record(ep(1, "我叫王小明"))
        tiers.record(ep(2, "我特别喜欢拉面"))
        values = [v["value"] for v in tiers.about("u")]
        self.assertTrue(any("王小明" in v for v in values))
        self.assertTrue(any("拉面" in v for v in values))
        self.assertEqual(tiers.about("other"), [])

    def test_snapshot_round_trip(self):
        tiers = MemoryTiers()
        tiers.record(ep(1, "我叫李雷"))
        clone = MemoryTiers()
        clone.restore(json.loads(json.dumps(tiers.snapshot())))
        self.assertEqual(clone.snapshot(), tiers.snapshot())

    def test_cap_keeps_undreamed_and_top_salient(self):
        tiers = MemoryTiers()
        for i in range(510):
            tiers.record(ep(i, salience=float(i % 10), dreamed=(i % 2 == 0)))
        self.assertLessEqual(len(tiers.episodes), 500)
        self.assertEqual(len(tiers.undreamed()), 255)
        self.assertEqual(tiers.episodes, sorted(tiers.episodes, key=lambda e: e.turn))

    def test_salience_is_bounded(self):
        class A:
            strength = 5.0
        self.assertLessEqual(salience_of([A(), A()], {"cortisol": 3.0}), 10.0)


class TestDreams(unittest.TestCase):
    def test_threshold_and_consolidation(self):
        tiers = MemoryTiers()
        for i in range(7):
            tiers.record(ep(i, text=f"第{i}件事"))
        dreamer = Dreamer(threshold=8)
        engine = offline_engine()
        self.assertFalse(dreamer.due(tiers))
        self.assertIsNone(dreamer.consolidate(tiers, engine.state, engine.persona, 100.0))
        tiers.record(ep(7, text="最后一件事"))
        self.assertTrue(dreamer.due(tiers))
        dream = dreamer.consolidate(tiers, engine.state, engine.persona, 100.0)
        self.assertIsNotNone(dream)
        self.assertEqual(dream.episodes, 8)
        self.assertIn("最后一件事", dream.narrative)
        self.assertEqual(tiers.undreamed(), [])

    def test_dream_narrative_follows_language(self):
        engine = mixed_engine("en", FakeRuntime())
        for i in range(8):
            engine.tiers.record(ep(i, text=f"event {i}"))
        dream = engine.dream(force=True)
        self.assertIsNotNone(dream)
        self.assertNotIn("梦里", dream.narrative)
        lowered = dream.narrative.lower()
        self.assertTrue("dream" in lowered or "nothing heavy" in lowered)

    def test_dream_settles_stress(self):
        engine = offline_engine()
        engine.state.set("cortisol", 0.9)
        for i in range(8):
            engine.tiers.record(ep(i, text="烦"))
        dream = engine.dream(force=True)
        self.assertIsNotNone(dream)
        self.assertLess(engine.state.get("cortisol"), 0.9)
        self.assertIn("cortisol", dream.settled)

    def test_tick_dreams_when_due(self):
        engine = offline_engine()
        for i in range(8):
            engine.tiers.record(ep(i, text=f"片段{i}"))
        engine.last_interaction = 1000.0
        engine.state.set("affinity", 0.7)
        engine.state.set("dopamine", 0.1)
        message = engine.tick(now=1000.0 + 700.0)
        self.assertTrue(message)
        self.assertEqual(len(engine.dreamer.dreams), 1)
        self.assertTrue(all(e.dreamed for e in engine.tiers.episodes))

    def test_engine_snapshot_round_trip(self):
        engine = offline_engine()
        engine.turn("你好", user="alice")
        engine.turn("我很讨厌等待", user="alice")
        engine.turn_cube("R")
        data = json.loads(json.dumps(engine.snapshot()))
        clone = MindEngine.load_from_dict(data, persona=engine.persona)
        self.assertEqual(clone.cube.snapshot(), engine.cube.snapshot())
        self.assertEqual(clone.relations.snapshot(), engine.relations.snapshot())
        self.assertEqual(len(clone.tiers.episodes), len(engine.tiers.episodes))
        self.assertEqual(clone.tiers.snapshot()["semantic"], engine.tiers.snapshot()["semantic"])


class TestRelations(unittest.TestCase):
    def test_affinity_per_user(self):
        engine = offline_engine()
        engine.relations.node("alice").affinity = 0.9
        engine.turn("你好", user="bob")
        self.assertAlmostEqual(engine.relations.nodes["alice"].affinity, 0.9, places=6)
        self.assertNotEqual(engine.relations.nodes["bob"].affinity, 0.9)

    def test_affinity_follows_state_within_a_turn(self):
        engine = offline_engine()
        engine.turn("谢谢你，你真好", user="alice")
        self.assertAlmostEqual(engine.relations.nodes["alice"].affinity,
                               engine.state.get("affinity"), places=6)

    def test_mentions_become_nodes(self):
        graph = RelationshipGraph()
        found = graph.mentions("我妈妈叫王芳", 10.0, "alice")
        self.assertEqual(found, ["王芳"])
        self.assertIn("王芳", graph.nodes)
        self.assertIn(("alice", "王芳", "妈妈"), graph.edges)

    def test_unnamed_roles_do_not_create_nodes(self):
        graph = RelationshipGraph()
        self.assertEqual(graph.mentions("我妈今天又催我", 1.0, "alice"), [])
        self.assertEqual(set(graph.nodes), {"self"})


class TestCube(unittest.TestCase):
    def test_quarter_turns_cancel(self):
        cube = CubeState()
        before = cube.snapshot()["occupancy"]
        for _ in range(4):
            cube.turn("U")
        self.assertEqual(cube.snapshot()["occupancy"], before)
        self.assertEqual(cube.history, ["U", "U", "U", "U"])

    def test_move_parsing(self):
        self.assertEqual((parse_move("U").face, parse_move("U").quarters), ("U", 1))
        self.assertEqual(parse_move("R'").quarters, 3)
        self.assertEqual(parse_move("F2").quarters, 2)
        with self.assertRaises(ValueError):
            parse_move("Q")

    def test_every_face_preserves_the_set(self):
        cube = CubeState()
        for face in ("U", "D", "L", "R", "F", "B"):
            cube.turn(face)
            self.assertTrue(cube.validate(), face)
            self.assertTrue(cube.history[-1].startswith(face))
        self.assertEqual(len(cube.occupancy), 26)

    def test_render_has_all_cubies(self):
        net = render_net(CubeState())
        for token in ("EGO", " ID", "LOG", "INT", "SUP", "MEM"):
            self.assertIn(token, net)

    def test_restore_round_trip(self):
        cube = CubeState()
        cube.apply("R U R' U'")
        clone = CubeState()
        clone.restore(cube.snapshot())
        self.assertEqual(clone.snapshot(), cube.snapshot())
        self.assertTrue(clone.validate())

    def test_strong_activation_turns_the_cube(self):
        engine = offline_engine()
        engine.turn("你真是个蠢货，滚开", user="alice")
        self.assertGreater(len(engine.cube.history), 0)

    def test_inverse_move_restores_the_layout(self):
        for face in "UDLRFB":
            with self.subTest(face=face):
                cube = CubeState()
                before = cube.snapshot()["occupancy"]
                cube.apply(f"{face} {face}'")
                self.assertEqual(cube.snapshot()["occupancy"], before)

    def test_turn_direction_follows_the_right_hand_rule(self):
        # The corner at (+1,+1,+1) is the anchor: each axis sends it to the
        # position a positive quarter turn about that axis must produce.
        expected = {"U": (1, 1, -1), "R": (1, -1, 1), "F": (-1, 1, 1)}
        for face, target in expected.items():
            with self.subTest(face=face):
                cube = CubeState()
                corner = cube.cubie_at((1, 1, 1))
                cube.turn(face)
                self.assertEqual(cube.position_of(corner), target)

    def test_a_turn_moves_exactly_one_layer(self):
        for face in "UDLRFB":
            with self.subTest(face=face):
                cube = CubeState()
                self.assertEqual(len(cube.turn(face)), 9)
        cube = CubeState()
        self.assertEqual(cube.turn("U", 0), [])
        self.assertEqual(cube.history, [])

    def test_centre_cubies_never_leave_home(self):
        from omega_mind.cube import _CENTRES
        cube = CubeState()
        cube.apply("R U R' U' F D' L2 B' U2 R")
        for name, home in _CENTRES.items():
            self.assertEqual(cube.position_of(name), home, name)

    def test_commutator_repeats_to_identity(self):
        cube = CubeState()
        before = cube.snapshot()["occupancy"]
        for _ in range(5):
            cube.apply("R U R' U'")
        self.assertNotEqual(cube.snapshot()["occupancy"], before)
        cube.apply("R U R' U'")
        self.assertEqual(cube.snapshot()["occupancy"], before)

    def test_history_is_bounded(self):
        cube = CubeState()
        for _ in range(60):
            cube.turn("R")
        self.assertEqual(len(cube.history), 50)


class TestProfiler(unittest.TestCase):
    def test_profiles_synthetic_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "units.jsonl"
            rows = [
                {"id": "a", "title": "t", "type": "teachable_unit",
                 "entry_question": "q", "invariant": "i", "phenomenon": "p",
                 "tags": ["热", "熵"]},
                {"id": "a", "title": "dup"},
                {"title": "missing id"},
                {"id": "b", "title": "t2", "tags": ["热"]},
            ]
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                            + "\nnot json\n", encoding="utf-8")
            profile = profile_corpus(tmp)
            self.assertEqual(profile["totals"]["rows"], 4)
            self.assertEqual(profile["totals"]["invalid_json"], 1)
            self.assertEqual(profile["totals"]["duplicate_ids"], 1)
            self.assertIn("id", profile["files"][0]["missing_required"])
            self.assertEqual(profile["top_tags"][0], ("热", 2))

    def test_real_corpus_when_present(self):
        directory = Path.home() / "Projects" / "feynman-open" / "knowledge"
        if not directory.is_dir():
            self.skipTest("corpus not on this machine")
        profile = profile_corpus(directory)
        self.assertGreaterEqual(profile["totals"]["rows"], 300)


class FakeRuntime:
    def __init__(self, reply: str = "An English answer."):
        self.reply_text = reply
        self.calls = 0
        self.memory = {}

    def turn(self, text, memory):
        self.calls += 1
        memory["last"] = text
        memory["count"] = int(memory.get("count", 0)) + 1
        return {"reply": self.reply_text, "module": "fake"}


def mixed_engine(language: str, runtime: FakeRuntime) -> MindEngine:
    return MindEngine(
        persona=get_persona("feynman-zh"),
        store_path=tempfile.mktemp(suffix=".json"),
        bridge=PyBotBridge(runtime=runtime),
        kb=FeynmanKB.disabled(),
        auto_save=False,
        language=language,
    )


class TestLanguageModes(unittest.TestCase):
    def test_zh_mode_never_calls_english_pybot(self):
        runtime = FakeRuntime()
        engine = mixed_engine("zh", runtime)
        result = engine.turn("你好")
        self.assertEqual(runtime.calls, 0)
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in result.reply))

    def test_en_mode_uses_pybot_and_stays_english(self):
        runtime = FakeRuntime()
        engine = mixed_engine("en", runtime)
        result = engine.turn("hello there")
        self.assertEqual(runtime.calls, 1)
        self.assertFalse(any("\u4e00" <= ch <= "\u9fff" for ch in result.reply))
        self.assertIn("An English answer", result.base_reply)

    def test_en_mode_rejects_chinese_base_reply(self):
        runtime = FakeRuntime(reply="这是一句中文回答。")
        engine = mixed_engine("en", runtime)
        result = engine.turn("tell me something")
        self.assertFalse(any("\u4e00" <= ch <= "\u9fff" for ch in result.reply))
        self.assertTrue(result.source.startswith("fallback"))

    def test_zh_mode_rejects_english_base_reply(self):
        runtime = FakeRuntime(reply="English only.")
        engine = mixed_engine("zh", runtime)
        result = engine.turn("随便聊聊")
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in result.reply))

    def test_language_survives_snapshot(self):
        engine = mixed_engine("en", FakeRuntime())
        data = json.loads(json.dumps(engine.snapshot()))
        clone = MindEngine.load_from_dict(
            data, persona=engine.persona, bridge=PyBotBridge(runtime=FakeRuntime()),
        )
        self.assertEqual(clone.language, "en")

    def test_provided_reply_must_match_the_mode(self):
        engine = mixed_engine("zh", FakeRuntime())
        result = engine.turn("聊聊", base_reply="An English sentence.")
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in result.reply))
        self.assertTrue(result.source.startswith("fallback"))

    def test_mode_switch_is_validated(self):
        engine = mixed_engine("zh", FakeRuntime())
        engine.set_language("en")
        self.assertEqual(engine.language, "en")
        with self.assertRaises(ValueError):
            engine.set_language("fr")


class TestBridgePersistence(unittest.TestCase):
    def test_pybot_memory_survives_snapshot(self):
        runtime = FakeRuntime()
        engine = mixed_engine("en", runtime)
        engine.turn("first message")
        runtime.turn("second message", engine.bridge.memory)  # PyBot remembers more
        expected = dict(engine.bridge.memory)
        data = json.loads(json.dumps(engine.snapshot()))
        fresh = FakeRuntime()
        clone = MindEngine.load_from_dict(
            data, persona=engine.persona, bridge=PyBotBridge(runtime=fresh),
        )
        self.assertEqual(clone.bridge.memory, expected)
        self.assertEqual(clone.bridge.memory["last"], "second message")

    def test_unserializable_memory_is_dropped_not_fatal(self):
        runtime = FakeRuntime()
        engine = mixed_engine("en", runtime)
        engine.bridge.memory["weird"] = object()
        data = json.loads(json.dumps(engine.snapshot()))
        self.assertEqual(data["pybot"]["memory"], {})
        self.assertTrue(data["pybot"].get("memory_dropped"))


class TestIdleMaintenance(unittest.TestCase):
    def test_maintenance_dreams_behind_the_proactive_gates(self):
        engine = offline_engine()
        for i in range(8):
            engine.tiers.record(ep(i, text=f"片段{i}"))
        engine.last_interaction = 1000.0
        engine.state.set("affinity", 0.1)   # tick() would refuse
        self.assertIsNone(engine.tick(now=2000.0))
        dream = engine.maintenance(now=2000.0)
        self.assertIsNotNone(dream)
        self.assertEqual(len(engine.dreamer.dreams), 1)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from omega_mind.server import build_server
        cls.engine = offline_engine()
        cls.httpd = build_server(cls.engine, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_state_and_dashboard(self):
        html = urllib.request.urlopen(self.url("/")).read().decode()
        self.assertIn("OMEGAMIND", html)
        state = json.load(urllib.request.urlopen(self.url("/api/state")))
        self.assertIn("state", state)
        self.assertIn("cube", state)

    def test_resonance_is_exposed(self):
        state = json.load(urllib.request.urlopen(self.url("/api/state")))
        self.assertIn("resonance", state)
        self.assertIn("energy", state["resonance"])
        body = json.dumps({"text": "你好", "user": "res"}).encode()
        req = urllib.request.Request(self.url("/api/chat"), data=body,
                                     headers={"content-type": "application/json"})
        payload = json.load(urllib.request.urlopen(req))
        self.assertIn("resonance", payload)

    def test_chat_post(self):
        body = json.dumps({"text": "你好", "user": "web"}).encode()
        req = urllib.request.Request(self.url("/api/chat"), data=body,
                                     headers={"content-type": "application/json"})
        payload = json.load(urllib.request.urlopen(req))
        self.assertTrue(payload["reply"])
        self.assertIn("state", payload)

    def test_bad_requests(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.url("/api/nope"))
        self.assertEqual(ctx.exception.code, 404)
        req = urllib.request.Request(self.url("/api/chat"), data=b"not json",
                                     headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_language_switch_endpoint(self):
        body = json.dumps({"lang": "en"}).encode()
        req = urllib.request.Request(self.url("/api/lang"), data=body,
                                     headers={"content-type": "application/json"})
        payload = json.load(urllib.request.urlopen(req))
        self.assertEqual(payload["language"], "en")
        state = json.load(urllib.request.urlopen(self.url("/api/state")))
        self.assertEqual(state["language"], "en")
        # back to Chinese for the remaining tests
        req = urllib.request.Request(self.url("/api/lang"),
                                     data=json.dumps({"lang": "zh"}).encode(),
                                     headers={"content-type": "application/json"})
        urllib.request.urlopen(req)
        bad = urllib.request.Request(self.url("/api/lang"),
                                     data=json.dumps({"lang": "fr"}).encode(),
                                     headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(bad)
        self.assertEqual(ctx.exception.code, 400)

    def test_cube_turn_and_dream_endpoints(self):
        body = json.dumps({"move": "U"}).encode()
        req = urllib.request.Request(self.url("/api/turn-cube"), data=body,
                                     headers={"content-type": "application/json"})
        payload = json.load(urllib.request.urlopen(req))
        self.assertTrue(payload["moved"])
        req = urllib.request.Request(self.url("/api/dream"), data=b"{}",
                                     headers={"content-type": "application/json"})
        json.load(urllib.request.urlopen(req))  # 200 either way

    def test_websocket_handshake_and_turn(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        sock.sendall(
            f"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        response = sock.recv(4096).decode("latin-1")
        self.assertIn("101 Switching Protocols", response)
        self.assertIn("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", response)  # RFC 6455 example accept
        payload = json.dumps({"text": "你好", "user": "ws"}).encode()
        mask = b"\x01\x02\x03\x04"
        header = bytes([0x81, 0x80 | len(payload)])
        sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
        b1, b2 = sock.recv(2)
        length = b2 & 0x7F
        if length == 126:
            length = int.from_bytes(sock.recv(2), "big")
        elif length == 127:
            length = int.from_bytes(sock.recv(8), "big")
        data = b""
        while len(data) < length:
            data += sock.recv(length - len(data))
        reply = json.loads(data)
        self.assertTrue(reply["reply"])
        # close
        sock.sendall(bytes([0x88, 0x80]) + mask)
        sock.close()


if __name__ == "__main__":
    unittest.main()


def _has_cjk_in_values(value) -> bool:
    if isinstance(value, str):
        return any("\u4e00" <= ch <= "\u9fff" for ch in value)
    if isinstance(value, dict):
        return any(_has_cjk_in_values(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_cjk_in_values(v) for v in value)
    return False


class TestEnglishKnowledge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zh = real_kb("zh")
        cls.en = real_kb("en")

    def test_english_package_covers_every_prose_row(self):
        # One package per language: the English package is its own corpus,
        # not an overlay on the Chinese one. Every Chinese prose row has one
        # English row, by id.
        learned = {doc.id for doc in self.zh.docs
                   if doc.id.startswith("learned-")}
        zh_ids = {doc.id for doc in self.zh.docs} - learned
        en_ids = {doc.id for doc in self.en.docs}
        # Known gap: the English package ships no verified_facts.jsonl, so
        # the web-quote rows have no translation yet. Answering in English is
        # unaffected (those rows are quotes from third-party pages, not
        # teachings), but the gap is pinned here: translating them shrinks
        # the set, adding an untranslated row fails this test.
        facts_file = Path(self.zh.dir) / "verified_facts.jsonl"
        untranslated = set()
        if facts_file.is_file():
            untranslated = {
                row["id"] for row in
                (json.loads(line) for line in
                 facts_file.read_text(encoding="utf-8").splitlines() if line.strip())
                if row.get("id")
            }
        self.assertEqual(en_ids, zh_ids - untranslated)
        self.assertEqual(zh_ids - en_ids, untranslated)
        self.assertIn(":en:", self.en.mode)
        self.assertEqual(self.en.teachable_units(language="en"),
                         [doc for doc in self.en.docs if doc.unit is not None])
        self.assertEqual(self.en.teachable_units(language="zh"), [])

    def test_english_answers_never_leak_chinese(self):
        for doc in self.en.docs:
            self.assertFalse(any("一" <= ch <= "鿿" for ch in doc.answer),
                             msg=doc.id)
        reply, source = self.en.answer("What is an atom?")
        self.assertIsNotNone(reply)
        self.assertTrue(source.startswith(self.en.source_root + ":"))
        self.assertFalse(any("一" <= ch <= "鿿" for ch in reply))

    def test_chinese_package_does_not_answer_english(self):
        # A Chinese-package miss is a miss; it never falls through to the
        # English index.
        text, source = self.zh.answer("What is an atom?", language="en")
        self.assertIsNone(text)
        self.assertTrue(source.endswith(":no_match"), source)

    def test_personal_memory_question_does_not_hit_a_biography(self):
        # A question about the speaker's own memory is not a question about
        # Feynman's teaching award; the card layer must stay out of it.
        for query in ("Why can I not remember yesterday clearly?",
                      "Why can't I remember yesterday clearly?"):
            reply, source = self.en.answer(query)
            self.assertIsNone(reply, msg=query)
            self.assertEqual(source, f"{self.en.source_root}:no_match")

    def test_general_memory_question_does_not_hit_unrelated_card(self):
        reply, source = self.en.answer("How does memory work?")
        self.assertIsNone(reply)
        self.assertEqual(source, f"{self.en.source_root}:no_match")

    def test_overlay_rows_are_provenanced_and_english(self):
        import json
        rows = []
        for path in sorted((Path(__file__).resolve().parents[3] / "personas"
                                / "feynman-en" / "knowledge").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append((path.name, json.loads(line)))
        self.assertGreaterEqual(len(rows), 200)
        for name, row in rows:
            self.assertTrue(row.get("id"), name)
            self.assertEqual(row.get("translation_of"), row["id"], name)
            self.assertEqual(row.get("source_language"), "zh", name)
            self.assertFalse(_has_cjk_in_values(row), name)

    def test_persona_science_answer_omits_internal_usage_note(self):
        # persona-science-001 carries a note telling the assistant when to
        # recall it. That is metadata, not something the persona should say.
        for docs in (self.zh.docs, self.en.docs):
            doc = next((d for d in docs if d.id == "persona-science-001"), None)
            self.assertIsNotNone(doc)
            self.assertNotIn("这个记忆应", doc.answer)
            self.assertNotIn("This memory should be triggered", doc.answer)
            self.assertNotIn("during meals and casual conversation", doc.answer.lower())

    def test_chinese_answers_stay_chinese(self):
        reply, source = self.zh.answer("原子是什么？")
        self.assertIsNotNone(reply)
        self.assertTrue(source.startswith("feynman_kb:"))
        self.assertTrue(any("一" <= ch <= "鿿" for ch in reply))

    def test_chinese_science_question_hits_the_science_card(self):
        reply, source = self.zh.answer("什么是科学？")
        self.assertIsNotNone(reply)
        self.assertEqual(source, "feynman_kb:persona-science-001")
        self.assertNotIn("这个记忆应", reply)

    def test_engine_english_mode_prefers_the_cards(self):
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=self.en,
            auto_save=False,
            language="en",
        )
        result = engine.turn("What is an atom?")
        self.assertTrue(result.source.startswith(self.en.source_prefix + ":"))
        self.assertFalse(any("一" <= ch <= "鿿" for ch in result.reply))

    def test_engine_chinese_mode_is_unchanged(self):
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=self.zh,
            auto_save=False,
            language="zh",
        )
        result = engine.turn("原子是什么？")
        self.assertTrue(result.source.startswith(self.zh.source_prefix + ":"))
        self.assertTrue(any("一" <= ch <= "鿿" for ch in result.reply))


class TestEnglishChallenge(unittest.TestCase):
    def setUp(self):
        self.kb = real_kb("en")
        self.engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=self.kb,
            auto_save=False,
            language="en",
        )
        from omega_mind.challenge import ChallengeSession
        self.session = ChallengeSession(self.kb, self.engine, language="en")

    def test_units_and_prompts_are_english(self):
        self.assertTrue(self.session.available())
        question = self.session.next_unit()
        self.assertTrue(question)
        self.assertFalse(any("一" <= ch <= "鿿" for ch in question))

    def test_english_answer_is_graded_and_feedback_has_no_cjk(self):
        unit = next(u for u in self.session.units if u.id == "U0_RELAX")
        self.session.unit = unit
        self.session.tries = 0
        grade = self.session.submit(
            "The deviation from equilibrium shrinks by the same time constant; "
            "the half-life is independent of how large the starting deviation is."
        )
        self.assertEqual(grade.verdict, "strong")
        self.assertFalse(any("一" <= ch <= "鿿" for ch in grade.feedback))

    def test_a_vague_answer_stays_weak(self):
        unit = next(u for u in self.session.units if u.id == "U0_RELAX")
        self.session.unit = unit
        self.session.tries = 0
        grade = self.session.submit("It gets colder over time.")
        self.assertEqual(grade.verdict, "weak")


class TestResonance(unittest.TestCase):
    def test_constants_match_the_design(self):
        from omega_mind import resonance as r
        self.assertEqual(r.OMEGA_RESONANCE, 2.25)
        self.assertAlmostEqual(r.FIRST_ORDER_INERTIA, 1.5)
        self.assertAlmostEqual(r.DAMPING_FACTOR, 2.0 / 3.0)
        self.assertEqual(r.HALF_LIFE_SECONDS, 45.0)
        self.assertAlmostEqual(r.HARMONY_THRESHOLD, 2.0 / 9.0)
        self.assertEqual(r.TORQUE_DEG_PER_UNIT, 40.0)
        self.assertEqual(r.SPRING_BACK_DEG, 20.0)

    def test_breakthrough_at_two_and_a_quarter(self):
        from omega_mind import resonance as r
        res = r.assess([], bonus=2.25)
        self.assertTrue(res.breakthrough)
        self.assertTrue(res.in_harmony)
        self.assertEqual(res.depth, 2)
        self.assertAlmostEqual(res.torque_deg, 90.0)

    def test_half_life_and_depth(self):
        from omega_mind import resonance as r
        from omega_mind.appraisal import Activation
        self.assertAlmostEqual(r.residual_energy(2.25, 45.0), 1.125)
        res = r.assess([Activation("curiosity", 4.5, {})])
        self.assertTrue(res.breakthrough)
        self.assertEqual(res.depth, 4)
        self.assertEqual(res.torque_deg, 180.0)

    def test_rumination_sequence_is_valid_and_bounded(self):
        from omega_mind import resonance as r
        moves = r.rumination_moves("F", 4)
        self.assertEqual(moves, ["F", "D", "F'", "D'"])
        for move in moves:
            parsed = parse_move(move)
            self.assertIn(parsed.face, "UDLRFB")
            self.assertIn(parsed.quarters, (1, 2, 3))
        self.assertEqual(r.rumination_moves("U", 9), ["U", "R", "U'", "R'"])


    def test_residual_energy_is_carried_across_turns(self):
        from omega_mind import resonance as r
        clock = [1000.0]
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            clock=lambda: clock[0],
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB.disabled(),
            auto_save=False,
            language="zh",
        )
        first = engine.turn("你真是个蠢货，滚开")
        self.assertGreater(first.resonance["retained"], 0.0)
        clock[0] += r.HALF_LIFE_SECONDS
        second = engine.turn("嗯")
        expected = r.residual_energy(first.resonance["retained"], r.HALF_LIFE_SECONDS)
        self.assertAlmostEqual(second.resonance["carried"], expected, places=3)

    def test_damping_and_spring_back_are_applied(self):
        from omega_mind import resonance as r
        res = r.assess([], bonus=2.25)
        expected = 2.25 * r.DAMPING_FACTOR - r.SPRING_BACK_DEG / r.TORQUE_DEG_PER_UNIT
        self.assertAlmostEqual(res.retained, expected, places=3)
        self.assertAlmostEqual(res.spring_back_deg, r.SPRING_BACK_DEG)

    def test_resonance_is_snapshotted(self):
        engine = offline_engine()
        engine.turn("你真是个蠢货，滚开")
        data = json.loads(json.dumps(engine.snapshot()))
        self.assertIn("resonance", data)
        clone = MindEngine.load_from_dict(data, persona=engine.persona)
        self.assertAlmostEqual(clone._resonance_energy, engine._resonance_energy, places=6)
        self.assertEqual(clone._resonance_ts, engine._resonance_ts)


class TestFactSkepticism(unittest.TestCase):
    RUMOUR_ZH = "费曼先生，听说 OpenAI 刚刚证明了在任意域上的矩阵乘法指数就是精确的 2.25！"
    RUMOUR_EN = ("I heard OpenAI just proved the matrix multiplication exponent "
                 "is exactly 2.25 over arbitrary fields.")

    def test_the_rumour_is_corrected_not_repeated(self):
        from omega_mind import skepticism
        hit = skepticism.check(self.RUMOUR_ZH, "zh")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.rule, "omega_2_25")
        self.assertIn("并没有证明", hit.reply)
        self.assertIn("开放问题", hit.reply)
        en = skepticism.check(self.RUMOUR_EN, "en")
        self.assertIsNotNone(en)
        self.assertIn("has not proved", en.reply)
        self.assertIn("still", en.reply)

    def test_an_ordinary_question_is_not_a_claim(self):
        from omega_mind import skepticism
        self.assertIsNone(skepticism.check("What is the matrix multiplication exponent?", "en"))
        self.assertIsNone(skepticism.check("矩阵乘法的指数现在最好上界是多少？", "zh"))

    def test_engine_reacts_with_a_jolt_and_a_breakthrough(self):
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB.disabled(),
            auto_save=False,
            language="zh",
        )
        before = engine.state.get("dopamine")
        result = engine.turn(self.RUMOUR_ZH)
        self.assertEqual(result.source, "skepticism:omega_2_25")
        self.assertGreater(engine.state.get("dopamine"), before)
        self.assertTrue(result.resonance["breakthrough"])
        self.assertTrue(any("一" <= ch <= "鿿" for ch in result.reply))
        self.assertTrue(engine.cube.history)

    def test_english_mode_answers_the_english_rumour_in_english(self):
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB.disabled(),
            auto_save=False,
            language="en",
        )
        result = engine.turn(self.RUMOUR_EN)
        self.assertEqual(result.source, "skepticism:omega_2_25")
        self.assertFalse(any("一" <= ch <= "鿿" for ch in result.reply))
        self.assertTrue(result.resonance["breakthrough"])


def real_kb(language: str = "zh") -> FeynmanKB:
    kb = FeynmanKB(language=language)
    if not kb.available() or kb.language != language:
        raise unittest.SkipTest(f"no {language!r} knowledge package installed")
    return kb


class TestEventLog(unittest.TestCase):
    def test_event_log_is_bounded(self):
        engine = offline_engine()
        for i in range(engine.EVENT_LOG_LIMIT + 15):
            engine.turn(f"消息 {i}", base_reply="好。")
        self.assertLessEqual(len(engine.event_log), engine.EVENT_LOG_LIMIT)
        # The newest entries survive; the oldest are dropped.
        self.assertIn("消息", str(engine.event_log[-1]))
        self.assertNotIn("消息 0", str(engine.event_log[0]))


class TestChallengeSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def engine(self, language="zh"):
        # No KB supplied: the mind owns its package and may switch to the
        # other package's mind on set_language.
        return MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            auto_save=False,
            language=language,
        )

    def test_engine_owns_one_session(self):
        engine = self.engine()
        session = engine.challenge()
        self.assertIs(engine.challenge(), session)
        self.assertIsNot(engine.challenge(reset=True), session)
        self.assertTrue(session.available())

    def test_snapshot_and_status(self):
        engine = self.engine()
        session = engine.challenge()
        question = session.next_unit()
        self.assertTrue(question)
        status = session.status()
        self.assertTrue(status["available"])
        self.assertTrue(status["question"])
        self.assertEqual(status["tries"], 0)
        self.assertEqual(status["total"], len(session.units))

    def test_progress_survives_snapshot_round_trip(self):
        engine = self.engine()
        session = engine.challenge()
        session.next_unit()
        first_id = session.unit.id
        session.submit("我不知道，但大概和这个单元有关。")
        data = engine.snapshot()
        self.assertEqual(data["challenge"]["active"], first_id)

        restored = MindEngine.load_from_dict(
            data, persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(), kb=self.kb, auto_save=False)
        resumed = restored.challenge()
        self.assertEqual(resumed.snapshot()["active"], first_id)
        self.assertEqual(resumed.snapshot()["tries"], 1)
        self.assertEqual(resumed.snapshot()["index"], session.index)

    def test_strong_answer_masters_the_unit(self):
        engine = self.engine()
        session = engine.challenge()
        session.next_unit()
        unit = session.unit
        answer = unit.unit.get("invariant") or unit.title
        grade = session.submit(answer)
        self.assertEqual(grade.verdict, "strong")
        self.assertEqual(session.mastery[unit.id], {"asked": 1, "strong": 1})
        self.assertEqual(session.status()["mastered"], 1)

    def test_mastered_units_are_served_after_unmastered_ones(self):
        engine = self.engine()
        session = engine.challenge()
        if len(session.units) < 2:
            self.skipTest("needs at least two teachable units")
        session.next_unit()
        first = session.unit.id
        session.mastery[first] = {"asked": 1, "strong": 1}
        session.unit = None
        session.next_unit()
        self.assertNotEqual(session.unit.id, first)
        # Once every unit is mastered, practice cycles through all again.
        for doc in session.units:
            session.mastery[doc.id] = {"asked": 0, "strong": 1}
        seen = set()
        for _ in range(len(session.units)):
            session.next_unit()
            seen.add(session.unit.id)
        self.assertEqual(seen, {doc.id for doc in session.units})

    def test_mastery_survives_snapshot_round_trip(self):
        engine = self.engine()
        session = engine.challenge()
        session.next_unit()
        uid = session.unit.id
        session.mastery[uid] = {"asked": 2, "strong": 1}
        data = engine.snapshot()
        restored = MindEngine.load_from_dict(
            data, persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(), kb=self.kb, auto_save=False)
        self.assertEqual(restored.challenge().mastery.get(uid),
                         {"asked": 2, "strong": 1})
        self.assertEqual(restored.challenge().status()["mastered"], 1)

    def test_language_switch_keeps_progress_pointer(self):
        engine = self.engine()
        session = engine.challenge()
        session.next_unit()
        first_id = session.unit.id
        engine.set_language("en")
        migrated = engine.challenge()
        self.assertEqual(migrated.language, "en")
        self.assertEqual(migrated.status()["active"], first_id)

    def test_reset_clears_challenge(self):
        engine = self.engine()
        engine.challenge().next_unit()
        engine.reset()
        self.assertIsNone(engine.challenge_session)
        self.assertEqual(engine.snapshot()["challenge"], {})


class TestChallengeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from omega_mind.server import build_server
        cls.kb = real_kb()
        cls.engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=cls.kb,
            auto_save=False,
        )
        cls.httpd = build_server(cls.engine, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url(path), data=json.dumps(body).encode(),
            headers={"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    def test_state_exposes_retrieval_and_challenge(self):
        self.post("/api/chat", {"text": "你好"})
        state = json.load(urllib.request.urlopen(self.url("/api/state")))
        self.assertIn("retrieval", state)
        self.assertIn("challenge", state)

    def test_start_answer_and_get_status(self):
        started = self.post("/api/challenge", {"action": "start"})
        self.assertTrue(started["question"])
        self.assertTrue(started["status"]["active"])
        status = json.load(urllib.request.urlopen(self.url("/api/challenge")))
        self.assertEqual(status["active"], started["status"]["active"])
        answered = self.post("/api/challenge",
                             {"action": "answer", "answer": "我说不上来。"})
        self.assertIn(answered["verdict"], {"strong", "partial", "weak"})
        self.assertEqual(answered["status"]["tries"], 1)

    def test_bad_challenge_requests(self):
        bad = urllib.request.Request(
            self.url("/api/challenge"),
            data=json.dumps({"action": "dance"}).encode(),
            headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(bad)
        self.assertEqual(ctx.exception.code, 400)
        empty = urllib.request.Request(
            self.url("/api/challenge"),
            data=json.dumps({"action": "answer", "answer": ""}).encode(),
            headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(empty)
        self.assertEqual(ctx.exception.code, 400)


class TestEnglishFactsAndSaveSafety(unittest.TestCase):
    def test_english_facts_become_semantic_memory(self):
        tiers = MemoryTiers()
        tiers.record(ep(1, "My name is Alice."))
        tiers.record(ep(2, "I live in Berlin"))
        tiers.record(ep(3, "I really love jazz"))
        tiers.record(ep(4, "I hate waiting."))
        tiers.record(ep(5, "I work at CERN"))
        values = [v["value"] for v in tiers.about("u")]
        self.assertIn("User's name is Alice", values)
        self.assertIn("User lives in Berlin", values)
        self.assertIn("User likes jazz", values)
        self.assertIn("User dislikes waiting", values)
        self.assertIn("User works at CERN", values)
        # Quotes keep the matched phrase, trimmed of punctuation.
        quotes = [v["quote"] for v in tiers.about("u")]
        self.assertIn("My name is Alice", quotes)

    def test_english_fact_extraction_rejects_junk_values(self):
        tiers = MemoryTiers()
        tiers.record(ep(1, "I love it"))
        tiers.record(ep(2, "I hate this"))
        self.assertEqual(tiers.about("u"), [])

    def test_corrupt_save_falls_back_and_keeps_a_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mind.json"
            path.write_text('{"state": {"cortisol": 0.5}', encoding="utf-8")
            engine = MindEngine.load(
                str(path), persona=get_persona("generic"),
                clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
                kb=FeynmanKB.disabled())
            self.assertIsNotNone(engine.load_warning)
            self.assertEqual(engine.turn_index, 0)
            backup = Path(str(path) + ".corrupt")
            self.assertTrue(backup.exists())
            self.assertIn("cortisol", backup.read_text(encoding="utf-8"))
            # The engine still saves over the broken file on the next turn.
            engine.turn("你好", now=1000.0)
            json.loads(path.read_text(encoding="utf-8"))

    def test_event_log_cap_applies_when_loading(self):
        engine = offline_engine()
        data = engine.snapshot()
        data["event_log"] = [{"turn": i, "now": 0.0} for i in range(500)]
        clone = MindEngine.load_from_dict(data, persona=engine.persona)
        self.assertLessEqual(len(clone.event_log), MindEngine.EVENT_LOG_LIMIT)
        self.assertEqual(clone.event_log[-1]["turn"], 499)
