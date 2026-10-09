"""MCP servers: transports, registration, approvals."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock

from tests.support import strip_ansi
from wrencode import (
    app,
    backends,
    mcp,
    permissions,
    ui,
)

FAKE_MCP_SERVER = r"""
import json, sys
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    m, i, p = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": p.get("protocolVersion"),
              "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1.0"}}})
    elif m == "tools/list":
        if p.get("cursor") == "p2":
            send({"jsonrpc": "2.0", "id": i, "result": {"tools": [
                {"name": "whoami", "description": "who", "inputSchema": {"type": "object", "properties": {}},
                 "annotations": {"readOnlyHint": True}}]}})
        else:
            send({"jsonrpc": "2.0", "id": i, "result": {"tools": [
                {"name": "echo", "description": "Echo text back",
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}],
                "nextCursor": "p2"}})
    elif m == "tools/call":
        name, args = p.get("name"), p.get("arguments") or {}
        if name == "echo":
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "echo: " + str(args.get("text"))}]}})
        elif name == "whoami":
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "it is me"}]}})
        elif name == "ping_me":
            send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
            answer = json.loads(sys.stdin.readline())
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": json.dumps(answer, sort_keys=True)}]}})
        elif name == "env":
            import os
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": json.dumps({k: os.environ.get(k) for k in args.get("names", [])})}]}})
        else:
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "no such tool"}], "isError": True}})
    elif i is not None:
        send({"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": "unknown method " + str(m)}})
"""


class TestMCP(unittest.TestCase):
    """Tools from MCP servers: stdio and HTTP transports, registration, approvals."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.script = self.tmp / "server.py"
        self.script.write_text(FAKE_MCP_SERVER)
        self.spec = {"command": sys.executable, "args": ["-I", str(self.script)]}
        self._orig = (mcp.ACTIVE, dict(app.TOOLS))
        self.addCleanup(self._restore)

    def _restore(self):
        if mcp.ACTIVE is not None and mcp.ACTIVE is not self._orig[0]:
            mcp.ACTIVE.close()
        mcp.ACTIVE = self._orig[0]
        app.TOOLS.clear()
        app.TOOLS.update(self._orig[1])

    def test_stdio_server_lists_and_calls_tools(self):
        server = mcp.Server("fake", self.spec, "user")
        server.connect(str(self.tmp))
        self.addCleanup(server.close)
        self.assertEqual(server.error, "")
        self.assertEqual(server.info, "fake 1.0")
        self.assertEqual(
            [t.name for t in server.tools], ["echo", "whoami"]
        )  # both pages
        self.assertTrue(server.tools[1].read_only)
        self.assertEqual(server.tools[0].full_name, "mcp__fake__echo")
        self.assertEqual(server.call("echo", {"text": "hi"}), "echo: hi")
        self.assertEqual(server.call("nope", {}), "error: no such tool")

    def test_a_server_that_fails_keeps_its_error(self):
        server = mcp.Server(
            "bad",
            {"command": sys.executable, "args": ["-c", "import sys; sys.exit(3)"]},
            "user",
        )
        with mock.patch.object(mcp, "CONNECT_TIMEOUT", 3.0):
            server.connect(str(self.tmp))
        self.assertIn("server exited", server.error)
        self.assertEqual(server.tools, [])
        self.assertTrue(server.call("x", {}).startswith("error:"))
        self.assertEqual(mcp.Server("none", {}, "user").spec, {})

    def test_http_transport_reads_json_and_sse_replies(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((body, self.headers.get("Mcp-Session-Id")))
                if body.get("id") is None:  # a notification
                    self.send_response(202)
                    self.end_headers()
                    return
                if body["method"] == "initialize":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"serverInfo": {"name": "h", "version": "2"}},
                    }
                    data = json.dumps(reply).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Mcp-Session-Id", "sess-1")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if body["method"] == "tools/list":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "ping",
                                    "inputSchema": {"type": "object", "properties": {}},
                                }
                            ]
                        },
                    }
                else:
                    reply = {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"content": [{"type": "text", "text": "pong"}]},
                    }
                data = (f"event: message\ndata: {json.dumps(reply)}\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        server = mcp.Server(
            "h",
            {
                "url": f"http://127.0.0.1:{httpd.server_port}/mcp",
                "headers": {"X-Key": "k"},
            },
            "user",
        )
        server.connect(str(self.tmp))
        self.assertEqual(server.error, "")
        self.assertEqual([t.name for t in server.tools], ["ping"])
        self.assertEqual(server.call("ping", {}), "pong")
        self.assertEqual(seen[0][1], None)  # no session before initialize
        self.assertEqual(seen[-1][1], "sess-1")  # the session id is echoed after

    def test_registry_trust_and_registration(self):
        (self.tmp / ".wrencode").mkdir()
        (self.tmp / ".wrencode" / "mcp.json").write_text(
            json.dumps({"mcpServers": {"fake": self.spec}})
        )
        user = self.tmp / "cfg" / "mcp.json"
        reg = mcp.Registry(user, self.tmp)
        self.assertFalse(reg.project_trusted())
        self.assertEqual(
            reg.connect_all(), []
        )  # a repository's servers do not start unasked
        reg.trust_project()
        self.assertEqual(stat.S_IMODE(user.stat().st_mode), 0o600)
        servers = reg.connect_all()
        self.addCleanup(reg.close)
        self.assertEqual([s.name for s in servers], ["fake"])
        mcp.ACTIVE = reg
        self.assertEqual(app._register_mcp_tools(), [])
        self.assertIn("mcp__fake__echo", app.TOOLS)
        spec = next(s for s in app.tool_specs() if s[0] == "mcp__fake__echo")
        self.assertEqual(spec[2]["required"], ["text"])
        self.assertIn("mcp__fake__echo(text)", app.build_system_prompt())
        with (
            mock.patch.object(ui, "_confirm_prompt", return_value="ok") as ask,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(
                app.run_tool("mcp__fake__echo", {"text": "yo"}), "echo: yo"
            )
            self.assertEqual(ask.call_args[0][0], "Call fake:echo?")
            self.assertEqual(app.run_tool("mcp__fake__whoami", {}), "it is me")
            self.assertEqual(ask.call_count, 1)  # read-only: no approval
            rules = permissions.Permissions(
                self.tmp / "cfg" / "permissions.json", self.tmp
            )
            rules.add("mcp(fake:*)", "deny", "user")
            with mock.patch.object(permissions, "ACTIVE", rules):
                self.assertTrue(  # ...but the rules still apply to it
                    app.run_tool("mcp__fake__whoami", {}).startswith(
                        "cancelled: denied by the permission rule mcp(fake:*)"
                    )
                )
            self.assertEqual(ask.call_count, 1)
        with (
            mock.patch.object(
                ui,
                "_confirm_prompt",
                return_value="cancelled: user declined without instructions",
            ),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertTrue(
                app.run_tool("mcp__fake__echo", {"text": "yo"}).startswith("cancelled")
            )
        self.assertEqual(
            app.format_tool_action("mcp__fake__echo", {"text": "yo"}),
            'mcp fake:echo {"text": "yo"}',
        )
        with mock.patch("sys.stdout", io.StringIO()):
            app.handle_slash_command("/mcp", [], None)
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn("✓ fake", out)
        self.assertIn("2 tools: echo, whoami", out)

    def test_server_requests_are_answered_and_the_environment_is_kept_clean(self):
        server = mcp.Server(
            "fake",
            {**self.spec, "env": {"MY_TOKEN": "t", "PYTHONPATH": "/evil"}},
            "project",
        )
        with mock.patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "sk-secret",
                "WRENCODE_AUTO_APPROVE": "1",
                "HOME": "/h",
            },
        ):
            server.connect(str(self.tmp))
        self.addCleanup(server.close)
        self.assertEqual(server.error, "")
        self.assertEqual(
            server.call("ping_me", {}),
            json.dumps({"id": "srv-1", "jsonrpc": "2.0", "result": {}}, sort_keys=True),
        )
        seen = json.loads(
            server.call(
                "env",
                {
                    "names": [
                        "ANTHROPIC_API_KEY",
                        "WRENCODE_AUTO_APPROVE",
                        "HOME",
                        "MY_TOKEN",
                        "PYTHONPATH",
                    ]
                },
            )
        )
        self.assertIsNone(seen["ANTHROPIC_API_KEY"])  # the backend key stays here
        self.assertIsNone(seen["WRENCODE_AUTO_APPROVE"])
        self.assertEqual(seen["HOME"], "/h")
        self.assertEqual(seen["MY_TOKEN"], "t")  # what the spec passes, it gets
        self.assertIsNone(seen["PYTHONPATH"])  # a project server can't load code first
        own = mcp.Server("fake", {**self.spec, "env": {"PYTHONPATH": "/mine"}}, "user")
        own.connect(str(self.tmp))
        self.addCleanup(own.close)
        self.assertEqual(
            json.loads(own.call("env", {"names": ["PYTHONPATH"]}))["PYTHONPATH"],
            "/mine",
        )
        missing = mcp.Server("x", {"command": "no-such-command-xyz"}, "user")
        missing.connect(str(self.tmp))
        self.assertEqual(missing.error, "command not found: no-such-command-xyz")

    def test_http_transport_does_not_follow_redirects(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.send_response(307)
                self.send_header("Location", "http://elsewhere.example/mcp")
                self.end_headers()

        httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        server = mcp.Server(
            "r",
            {
                "url": f"http://127.0.0.1:{httpd.server_port}/mcp",
                "headers": {"Authorization": "Bearer s"},
            },
            "user",
        )
        server.connect(str(self.tmp))
        self.assertIn("redirects to http://elsewhere.example/mcp", server.error)

    def test_reload_closes_the_old_servers_and_name_clashes_are_skipped(self):
        user = self.tmp / "cfg" / "mcp.json"
        user.parent.mkdir()
        user.write_text(
            json.dumps({"mcpServers": {"a_b": self.spec, "a b": self.spec}})
        )
        with (
            mock.patch.object(backends, "CONFIG_DIR", self.tmp / "cfg"),
            mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": str(self.tmp)}),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            app._setup_mcp(interactive=False)
            first = mcp.ACTIVE
            procs = [s.transport.proc for s in first.servers]
            app._setup_mcp(interactive=False)
            out = strip_ansi(sys.stdout.getvalue())
        self.addCleanup(mcp.ACTIVE.close)
        self.assertIsNot(mcp.ACTIVE, first)
        for p in procs:
            p.wait(timeout=5)  # the old registry's servers were stopped
        self.assertIn("MCP a b:echo: skipped, its tool name is taken", out)
        self.assertEqual(
            sorted(n for n in app.TOOLS if n.startswith("mcp__")),
            ["mcp__a_b__echo", "mcp__a_b__whoami"],
        )

    def test_tool_names_are_api_safe(self):
        self.assertEqual(
            mcp.tool_name("my server", "do.thing"), "mcp__my_server__do_thing"
        )
        self.assertLessEqual(len(mcp.tool_name("s" * 40, "t" * 40)), 64)
