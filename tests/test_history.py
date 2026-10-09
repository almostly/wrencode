"""Conversation history: the store, embedded PGlite paths, the mirror, the loop's use of it."""

from __future__ import annotations

import datetime
import io
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

from wrencode import (
    app,
    backends,
    history,
    ui,
)

try:
    import psycopg
except ImportError:  # the Postgres history tests below skip without it
    psycopg = None


@unittest.skipIf(
    psycopg is None or shutil.which("node") is None, "psycopg or Node.js not installed"
)
class TestHistoryStore(unittest.TestCase):
    """The Postgres history store, against a real embedded PGlite."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls._tmp.name) / "pglite"
        cls.server = history.EmbeddedPGlite(cls.root)
        try:
            cls.store = history.Store(cls.server.start(), cls.server)
            cls.store.init_schema()
        except RuntimeError as err:
            cls.server.stop()
            cls._tmp.cleanup()
            raise unittest.SkipTest(f"embedded PGlite unavailable: {err}") from err
        except BaseException:
            cls.server.stop()
            cls._tmp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls._tmp.cleanup()

    def test_sessions_and_messages_roundtrip(self):
        ws = "/work/roundtrip"
        self.assertIsNone(self.store.latest_session(ws))
        sid = self.store.new_session(ws, "openai", "gpt-x")
        msgs = [
            {"role": "user", "content": "fix the parser"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "1| x"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
        self.store.save(sid, msgs)
        self.assertEqual(self.store.load(sid), msgs)
        self.assertEqual(self.store.latest_session(ws), sid)
        row = self.store.sessions(ws)[0]
        self.assertEqual(
            (row["id"], row["title"], row["chats"]), (sid, "fix the parser", 1)
        )
        self.store.save(sid, msgs[:1])  # a save replaces the list (compaction)
        self.assertEqual(self.store.load(sid), msgs[:1])

    def test_search_is_per_workspace(self):
        a, b = "/work/search-a", "/work/search-b"
        sa = self.store.new_session(a, "x", "y")
        sb = self.store.new_session(b, "x", "y")
        self.store.save(sa, [{"role": "user", "content": "the flux capacitor leaks"}])
        self.store.save(sb, [{"role": "user", "content": "flux elsewhere"}])
        hits = self.store.search(a, "flux")
        self.assertEqual([h["session_id"] for h in hits], [sa])
        self.assertIn("capacitor", hits[0]["text"])
        self.assertEqual(self.store.search(a, "nomatchxyz"), [])

    def test_mirror_copies_sessions_to_a_second_postgres(self):
        with tempfile.TemporaryDirectory() as d:
            remote_server = history.EmbeddedPGlite(pathlib.Path(d) / "pglite")
            remote = history.Store(remote_server.start(), remote_server)
            mirror = history.Mirror(remote, label="second")
            self.store.mirror = mirror
            try:
                ws = "/work/mirrored"
                sid = self.store.new_session(ws, "openai", "gpt-x")
                msgs = [{"role": "user", "content": "copy me"}]
                self.store.save(sid, msgs)
                self.assertTrue(mirror.flush(30))
                rows = remote.sessions(ws)
                self.assertEqual(
                    [(r["title"], r["chats"]) for r in rows], [("copy me", 1)]
                )
                self.assertEqual(remote.load(rows[0]["id"]), msgs)
                # a later save replaces the copy; /sync re-queues everything
                self.store.save(sid, [*msgs, {"role": "assistant", "content": "done"}])
                self.assertTrue(mirror.flush(30))
                self.assertEqual(len(remote.load(rows[0]["id"])), 2)
                queued, left = self.store.sync(ws)
                self.assertEqual((queued, left), (1, 0))
                self.assertEqual(
                    len(remote.sessions(ws)), 1
                )  # matched by uid, not duplicated
            finally:
                self.store.mirror = None
                mirror.close()

    def test_data_survives_a_server_restart(self):
        ws = "/work/persist"
        sid = self.store.new_session(ws, "x", "y")
        self.store.save(sid, [{"role": "user", "content": "keep me"}])
        self.store.close()  # the connection and the server
        server = history.EmbeddedPGlite(self.root)
        store = history.Store(server.start(), server)
        type(self).server, type(self).store = server, store  # for the other tests
        self.assertEqual(store.latest_session(ws), sid)
        self.assertEqual(store.load(sid)[0]["content"], "keep me")


class TestEmbeddedPaths(unittest.TestCase):
    """The embedded server's socket must fit a Unix socket path; errors read clean."""

    def test_socket_stays_under_a_short_root_and_moves_for_a_long_one(self):
        short = pathlib.Path("/home/me/.wrencode/pglite")
        self.assertEqual(history.redact("postgres://u:p@host:abc/db"), "host/db")
        shared = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, shared, ignore_errors=True)
        history._private_dir(shared / "fresh")
        self.assertEqual(stat.S_IMODE((shared / "fresh").stat().st_mode), 0o700)
        history._private_dir(shared / "fresh")  # ours and private: fine
        (shared / "open").mkdir(mode=0o755)
        with self.assertRaisesRegex(RuntimeError, "not a private directory"):
            history._private_dir(shared / "open")
        (shared / "link").symlink_to(shared / "fresh")
        with self.assertRaisesRegex(RuntimeError, "not a private directory"):
            history._private_dir(shared / "link")
        self.assertEqual(history._socket_dir(short), short / "run")
        long = pathlib.Path("/tmp/" + "x" * 100 + "/pglite")
        moved = history._socket_dir(long)
        self.assertLessEqual(len(str(moved / history.SOCKET_NAME).encode()), 108)
        self.assertTrue(str(moved).startswith(tempfile.gettempdir()))
        self.assertEqual(moved, history._socket_dir(long))  # stable per root
        self.assertNotEqual(
            moved, history._socket_dir(pathlib.Path("/tmp/" + "y" * 100))
        )
        pg = history.EmbeddedPGlite(long)
        self.assertEqual(pg.socket.parent, moved)
        self.assertIn(f"host={moved}", pg.dsn())

    def test_node_errors_lose_their_color_codes(self):
        self.assertEqual(
            history._ANSI.sub("", "\x1b[90mat main\x1b[39m {"), "at main {"
        )

    def test_history_json_lives_in_the_config_dir(self):
        with (
            mock.patch.dict(os.environ, {"WRENCODE_HISTORY_FILE": ""}),
            mock.patch.object(backends, "CONFIG_DIR", pathlib.Path("/cfg")),
        ):
            os.environ.pop("WRENCODE_HISTORY_FILE", None)
            self.assertEqual(app.history_file_path(), pathlib.Path("/cfg/history.json"))


class TestMirrorQueue(unittest.TestCase):
    """The mirror's worker: coalescing, retry after failure, and never blocking the caller."""

    class FlakyRemote:
        def __init__(self, failures):
            self.failures = failures
            self.applied = []
            self.schema_inits = 0

        def init_schema(self):
            self.schema_inits += 1

        def upsert(self, row, messages):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("connection refused")
            self.applied.append((row["uid"], list(messages)))

        def close(self):
            pass

    def test_latest_snapshot_wins_and_failures_are_retried(self):
        remote = self.FlakyRemote(failures=1)
        mirror = history.Mirror(remote, label="fake")
        with mock.patch("sys.stdout", io.StringIO()) as out:
            mirror.enqueue({"uid": "u1"}, [{"role": "user", "content": "v1"}])
            mirror.enqueue({"uid": "u1"}, [{"role": "user", "content": "v2"}])
            self.assertTrue(mirror.flush(10))
            mirror.close()
        self.assertEqual(remote.applied, [("u1", [{"role": "user", "content": "v2"}])])
        self.assertEqual(remote.schema_inits, 1)
        self.assertIn("unreachable", out.getvalue())
        self.assertIn("is back", out.getvalue())
        self.assertFalse(mirror.failing)

    def test_enqueue_never_blocks_on_a_dead_remote(self):
        remote = self.FlakyRemote(failures=10**6)
        mirror = history.Mirror(remote, label="dead")
        with mock.patch("sys.stdout", io.StringIO()) as out:
            t0 = time.monotonic()
            for i in range(50):
                mirror.enqueue({"uid": f"u{i}"}, [])
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertFalse(mirror.flush(0.3))
            self.assertEqual(mirror.pending(), 50)
            mirror._stop.set()
            mirror._wake.set()
        self.assertEqual(out.getvalue().count("unreachable"), 1)

    def test_redact_hides_credentials(self):
        self.assertEqual(
            history.redact("postgres://alice:s3cret@db.example.com:5433/wren"),
            "db.example.com:5433/wren",
        )
        self.assertEqual(history.redact("host=/tmp/x dbname=postgres"), "mirror")


class TestHistoryWiring(unittest.TestCase):
    """The loop's use of the store, with the store mocked."""

    def setUp(self):
        self.store = mock.Mock()
        self.store.load.return_value = [{"role": "user", "content": "hi"}]
        self.store.new_session.return_value = 7
        self.store.sessions.return_value = [
            {
                "id": 7,
                "title": "hi",
                "model": "m",
                "updated_at": datetime.datetime(
                    2026, 10, 8, 12, 0, tzinfo=datetime.timezone.utc
                ),
                "chats": 1,
            }
        ]
        self.store.search.return_value = [
            {"session_id": 7, "role": "user", "text": "hi there"}
        ]
        for p in (
            mock.patch.object(app, "_STORE", self.store),
            mock.patch.object(app, "_SESSION_ID", 7),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_load_and_save_go_to_the_store(self):
        self.assertEqual(app.load_history(), [{"role": "user", "content": "hi"}])
        app.save_history([{"role": "user", "content": "x"}])
        self.store.save.assert_called_once_with(7, [{"role": "user", "content": "x"}])

    def test_clear_starts_a_new_session_and_keeps_the_old(self):
        msgs = [{"role": "user", "content": "old"}]
        action, _ = app.handle_slash_command("/clear", msgs, None)
        self.assertEqual(action, "handled")
        self.assertEqual(msgs, [])
        self.store.new_session.assert_called_once()
        self.store.save.assert_not_called()
        self.assertIn("new session #7", sys.stdout.getvalue())

    def test_sessions_resume_and_search(self):
        app.handle_slash_command("/sessions", [], None)
        self.assertIn("#7", sys.stdout.getvalue())
        msgs: list = []
        app.handle_slash_command("/resume 7", msgs, None)
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])
        app.handle_slash_command("/resume 99", [], None)
        self.assertIn("No session #99", sys.stdout.getvalue())
        app.handle_slash_command("/search hi", [], None)
        out = sys.stdout.getvalue()
        self.assertIn("hi there", out)  # the matching line, under the session's row
        self.assertIn("2026-10-08 12:00", out)
        self.assertNotIn("model", out.split("/search hi")[-1])
        self.store.search.assert_called_with(mock.ANY, "hi")

    def test_bare_resume_offers_a_picker_on_a_terminal(self):
        msgs: list = []
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch.object(ui, "pick_from_list", return_value=0) as pick,
        ):
            app.handle_slash_command("/resume", msgs, None)
        self.assertEqual(pick.call_args[0][1], ["7"])
        self.assertIn("#7", pick.call_args[1]["labels"][0])
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])
        with mock.patch("sys.stdin.isatty", return_value=False):
            app.handle_slash_command("/resume", [], None)
        self.assertIn("Usage: /resume <id>", sys.stdout.getvalue())

    def test_sync_reports_the_mirror(self):
        self.store.mirror = None
        app.handle_slash_command("/sync", [], None)
        self.assertIn("No mirror configured", sys.stdout.getvalue())
        self.store.mirror = mock.Mock(label="db.example.com/wren")
        self.store.sync.return_value = (3, 0)
        app.handle_slash_command("/sync", [{"role": "user", "content": "x"}], None)
        self.assertIn(
            "Mirrored 3 sessions to db.example.com/wren", sys.stdout.getvalue()
        )
        self.store.save.assert_called()  # the current conversation is saved first
        self.store.sync.return_value = (3, 2)
        app.handle_slash_command("/sync", [], None)
        self.assertIn("2 still pending", sys.stdout.getvalue())

    def test_without_the_store_the_commands_explain(self):
        with mock.patch.object(app, "_STORE", None):
            app.handle_slash_command("/sessions", [], None)
        self.assertIn("history.json", sys.stdout.getvalue())

    def test_a_failed_save_is_reported_not_raised(self):
        self.store.save.side_effect = RuntimeError("db gone")
        app.save_history([])
        self.assertIn("Could not save history", sys.stdout.getvalue())
