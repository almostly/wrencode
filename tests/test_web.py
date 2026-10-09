"""The fetch tool and web search."""

from __future__ import annotations

import io
import sys
import threading
import unittest
from unittest import mock

from tests.support import strip_ansi
from wrencode import (
    app,
    backends,
    permissions,
    ui,
    web,
)


class TestWeb(unittest.TestCase):
    """The fetch tool and Anthropic's server-side web search."""

    def _serve(self, routes):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.enterContext(mock.patch.object(web, "ALLOW_LOCAL", True))

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                ctype, body = routes.get(self.path, ("text/plain", b"nope"))
                if ctype == "redirect":
                    self.send_response(302)
                    self.send_header("Location", body.decode())
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_port}"

    def test_html_becomes_readable_text(self):
        title, text = web.html_to_text(
            "<html><head><title> Doc </title><style>x{}</style></head><body>"
            "<nav>menu</nav><h2>Install</h2><p>Run <code>pip</code> then <a href='/next'>next</a>.</p>"
            "<ul><li>one</li><li>two</li></ul><pre>a\n  b</pre><script>bad()</script></body></html>",
            "https://ex.com/docs/",
        )
        self.assertEqual(title, "Doc")
        self.assertIn("## Install", text)
        self.assertIn("Run pip then next (https://ex.com/next).", text)
        self.assertIn("- one\n\n- two", text)
        self.assertIn("```\na\n  b\n```", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn("x{}", text)

    def test_fetch_handles_html_json_binary_and_offsets(self):
        base = self._serve(
            {
                "/page": (
                    "text/html; charset=utf-8",
                    b"<title>T</title><h1>Hello</h1><p>world</p>",
                ),
                "/data": ("application/json", b'{"a": [1, 2]}'),
                "/bin": ("application/octet-stream", b"\x00\x01\x02"),
                "/long": ("text/plain", b"x" * 100),
            }
        )
        page = web.fetch(base + "/page")
        self.assertTrue(page.startswith(f"T\n{base}/page\n\n# Hello\n\nworld"), page)
        self.assertIn('"a": [\n    1,', web.fetch(base + "/data"))
        self.assertIn(
            "[application/octet-stream, 3 bytes; not text]", web.fetch(base + "/bin")
        )
        with mock.patch.object(web, "MAX_CHARS", 40):
            first = web.fetch(base + "/long")
            self.assertIn("[60 more characters; fetch again with offset=40]", first)
            second = web.fetch(base + "/long", 40)
            self.assertIn("x" * 40, second)
            self.assertIn("offset=80", second)
            self.assertTrue(web.fetch(base + "/long", 500).startswith("error: offset"))
        self.assertTrue(
            web.fetch(base + "/missing").startswith(f"{base}/missing\n\nnope")
        )
        self.assertTrue(web.fetch("ftp://x/y").startswith("error: not an http(s) URL"))
        self.assertTrue(
            web.fetch("http://127.0.0.1:9/").startswith("error: could not fetch")
        )

    def test_fetch_stays_on_public_hosts(self):
        with mock.patch.object(web, "ALLOW_LOCAL", False):
            for host in (
                "127.0.0.1",
                "localhost",
                "169.254.169.254",
                "10.1.2.3",
                "[::1]",
            ):
                self.assertIn(
                    "is not a public address",
                    web.fetch(f"http://{host}/latest/meta-data"),
                )
            self.assertTrue(web._public("example.com"))
        base = self._serve(
            {
                "/away": ("redirect", b"https://other.example/x"),
                "/home": ("redirect", b"/landed"),
                "/landed": ("text/plain", b"here"),
                "/private": ("redirect", b"http://169.254.169.254/"),
            }
        )
        self.assertTrue(web.fetch(base + "/home").endswith("\n\nhere"))
        self.assertIn(
            "redirects to https://other.example/x; fetch that URL",
            web.fetch(base + "/away"),
        )
        with (
            mock.patch.object(web, "ALLOW_LOCAL", False),
            mock.patch.object(web, "_public", lambda h: h.startswith("127.")),
        ):
            self.assertIn(
                "redirects to http://169.254.169.254/", web.fetch(base + "/private")
            )
        self.assertEqual(
            web.subject("https://U:p@EVIL.com:443/x?a=1#f"), "evil.com/x?a=1"
        )
        self.assertEqual(web.subject("http://h:8080/"), "h:8080/")
        self.assertEqual(web.subject("http://h:80"), "h/")

    def test_fetch_caps_a_compressed_body_and_reads_a_page_without_head_end(self):
        import gzip as gz

        with mock.patch.object(web, "MAX_BYTES", 1000):
            base = self._serve({"/z": ("text/plain", gz.compress(b"a" * 100000))})

            class R:
                def __init__(self, fp):
                    self.fp = fp

            real = web._OPENER.open

            def open_gz(req, timeout):
                resp = real(req, timeout=timeout)
                resp.headers["Content-Encoding"] = "gzip"
                return resp

            with mock.patch.object(web._OPENER, "open", open_gz):
                out = web.fetch(base + "/z")
        self.assertIn("a" * 1000, out)
        self.assertNotIn("a" * 1002, out)
        self.assertIn("longer than the size cap", out)
        self.assertEqual(
            web.html_to_text(
                "<html><head><title>T</title><body><h1>Hi</h1><p>body text</p>"
            ),
            ("T", "# Hi\n\nbody text"),
        )

    def test_fetch_tool_asks_and_follows_rules(self):
        base = self._serve({"/p": ("text/plain", b"hi")})
        self.assertEqual(
            web.subject(base + "/p?x=1"), f"127.0.0.1:{base.rsplit(':', 1)[1]}/p?x=1"
        )
        self.assertEqual(
            permissions.suggest("fetch", "docs.python.org/3/library"),
            "fetch(docs.python.org/*)",
        )
        with (
            mock.patch.object(ui, "_confirm_prompt", return_value="ok") as ask,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            out = app.run_tool("fetch", {"url": base + "/p"})
            self.assertEqual(ask.call_args[0][0], f"Fetch {web.subject(base + '/p')}?")
        self.assertTrue(out.endswith("\n\nhi"), out)
        with (
            mock.patch.object(
                ui,
                "_confirm_prompt",
                return_value="cancelled: user declined without instructions",
            ),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertTrue(
                app.run_tool("fetch", {"url": base + "/p"}).startswith("cancelled")
            )
        self.assertIn("fetch(url, offset)", app.build_system_prompt())
        self.assertEqual(
            app.format_tool_action("fetch", {"url": "https://a/b", "offset": 40}),
            "fetch https://a/b  offset=40",
        )

    def test_web_search_tool_version_follows_the_model(self):
        self.assertEqual(
            backends.web_search_tool("claude-sonnet-5-5")["type"], "web_search_20260209"
        )
        self.assertEqual(
            backends.web_search_tool("claude-opus-4-6")["type"], "web_search_20260209"
        )
        self.assertEqual(
            backends.web_search_tool("claude-haiku-5-5")["type"], "web_search_20250305"
        )
        self.assertEqual(
            backends.web_search_tool("claude-haiku-4-5-20251001")["type"],
            "web_search_20250305",
        )

    def test_web_search_tool_is_offered_on_anthropic_only(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "API_BASE", "https://x/v1/messages"),
            mock.patch.object(backends, "API_KEY", "k"),
            mock.patch.object(
                backends, "_http_post", return_value={"content": [], "usage": {}}
            ) as post,
        ):
            specs = [("read", "Read", {"type": "object", "properties": {}})]
            backends.get_response([{"role": "user", "content": "x"}], "s", None, specs)
            tools = post.call_args[0][1]["tools"]
            self.assertEqual(tools[0], backends.web_search_tool(backends.MODEL))
            self.assertIn(
                "cache_control", tools[-1]
            )  # the cache marker stays on the last of ours
            with mock.patch.object(backends, "WEB_SEARCH", False):
                backends.get_response(
                    [{"role": "user", "content": "x"}], "s", None, specs
                )
            self.assertEqual(post.call_args[0][1]["tools"][0]["name"], "read")

    def test_search_blocks_stream_in_order_and_are_billed(self):
        notes = []
        seen = []
        msg = backends._stream_anthropic(
            iter(
                [
                    {
                        "type": "message_start",
                        "message": {
                            "role": "assistant",
                            "content": [],
                            "usage": {"input_tokens": 10},
                        },
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"query": "wren',
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": 'code"}'},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://a",
                                    "title": "A page",
                                    "encrypted_content": "x",
                                },
                                {
                                    "type": "web_search_result",
                                    "url": "https://b",
                                    "title": "B",
                                },
                            ],
                        },
                    },
                    {"type": "content_block_stop", "index": 1},
                    {
                        "type": "content_block_start",
                        "index": 2,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 2,
                        "delta": {"type": "text_delta", "text": "Found it."},
                    },
                    {"type": "content_block_stop", "index": 2},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {
                            "output_tokens": 30,
                            "server_tool_use": {"web_search_requests": 1},
                        },
                    },
                ]
            ),
            seen.append,
            notes.append,
        )
        self.assertEqual(seen, ["Found it."])
        self.assertEqual(
            [b["type"] for b in notes], ["server_tool_use", "web_search_tool_result"]
        )
        self.assertEqual(msg["content"][0]["input"], {"query": "wrencode"})
        self.assertEqual(
            backends.describe_server_block(msg["content"][0]), 'web_search "wrencode"'
        )
        self.assertEqual(
            backends.describe_server_block(msg["content"][1]), "2 results: A page, B"
        )
        self.assertEqual(
            backends.describe_server_block(
                {
                    "type": "web_search_tool_result",
                    "content": {
                        "type": "web_search_tool_result_error",
                        "error_code": "max_uses_exceeded",
                    },
                }
            ),
            "search failed: max_uses_exceeded",
        )
        orig = backends.USAGE
        backends.USAGE = backends.Usage()
        try:
            with (
                mock.patch.object(backends, "BACKEND", "anthropic"),
                mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
            ):
                text, calls = backends._parse_native_response(msg)
            self.assertEqual((text, calls), ("Found it.", []))
            self.assertEqual(backends.USAGE.turn_searches, 1)
            self.assertAlmostEqual(
                backends.USAGE.turn_cost, (10 * 2 + 30 * 10) / 1e6 + 0.01
            )
            self.assertEqual(backends.USAGE.as_dict()["web_searches"], 1)
            self.assertTrue(
                any(
                    line.startswith("web searches: 1 this turn")
                    for line in backends.usage_report()
                )
            )
        finally:
            backends.USAGE = orig

    def test_filter_code_is_hidden_and_a_whole_input_survives_the_stream(self):
        code = {
            "type": "server_tool_use",
            "name": "code_execution",
            "input": {"code": "x"},
        }
        self.assertEqual(backends.describe_server_block(code), "")
        whole = backends._stream_anthropic(
            iter(
                [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "server_tool_use",
                            "name": "web_search",
                            "input": {"query": "given whole"},
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                ]
            ),
            lambda t: None,
        )
        self.assertEqual(whole["content"][0]["input"], {"query": "given whole"})

    def test_server_blocks_print_without_streaming(self):
        block = {
            "type": "server_tool_use",
            "name": "web_search",
            "input": {"query": "q"},
        }
        self.assertEqual(strip_ansi(app._server_block_line(block)), '● web_search "q"')
        result = {"type": "web_search_tool_result", "content": []}
        self.assertEqual(strip_ansi(app._server_block_line(result)), "  ⎿ no results")
        with mock.patch("sys.stdout", io.StringIO()):
            p = ui.StreamPrinter()
            p.feed("Looking")
            p.note('● web_search "q"')
            p.feed("done\n")
            p.close()
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn('● Looking\n● web_search "q"\n  done\n', out)
