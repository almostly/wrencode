"""Streamed replies: assembling them, and printing them as they arrive."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from unittest import mock

from tests.support import strip_ansi
from wrencode import (
    backends,
    ui,
)


class TestStreaming(unittest.TestCase):
    """Streamed replies: parsed from server-sent events, assembled whole, shown as they come."""

    def _events(self, *objs):
        return iter(objs)

    def test_anthropic_stream_assembles_text_tools_and_thinking(self):
        seen = []
        msg = backends._stream_anthropic(
            self._events(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "role": "assistant",
                        "model": "m",
                        "content": [],
                        "usage": {"input_tokens": 10, "cache_read_input_tokens": 4},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hm"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Hel"},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "lo"},
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": 'th": "a.py"}',
                    },
                },
                {"type": "content_block_stop", "index": 2},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 9},
                },
                {"type": "message_stop"},
            ),
            seen.append,
        )
        self.assertEqual(seen, ["Hel", "lo"])
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(
            msg["usage"],
            {"input_tokens": 10, "cache_read_input_tokens": 4, "output_tokens": 9},
        )
        self.assertEqual(
            msg["content"][0],
            {"type": "thinking", "thinking": "hm", "signature": "sig"},
        )
        self.assertEqual(msg["content"][1], {"type": "text", "text": "Hello"})
        self.assertEqual(
            msg["content"][2],
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read",
                "input": {"path": "a.py"},
            },
        )
        with mock.patch.object(backends, "BACKEND", "anthropic"):
            text, calls = backends._parse_native_response(msg)
        self.assertEqual(
            (text, calls[0].name, calls[0].input), ("Hello", "read", {"path": "a.py"})
        )

    def test_anthropic_stream_error_event_raises(self):
        with self.assertRaises(RuntimeError):
            backends._stream_anthropic(
                self._events(
                    {
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "busy"},
                    }
                ),
                lambda t: None,
            )

    def test_empty_text_blocks_are_dropped_from_the_assembled_message(self):
        msg = backends._stream_anthropic(
            iter(
                [
                    {"type": "message_start", "message": {"role": "assistant"}},
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "t",
                            "name": "read",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"path": "a"}',
                        },
                    },
                    {"type": "content_block_stop", "index": 1},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 3},
                    },
                ]
            ),
            lambda t: None,
        )
        self.assertEqual([b["type"] for b in msg["content"]], ["tool_use"])

    def test_openai_stream_takes_whole_tool_calls_without_an_index(self):
        data = backends._stream_openai(
            iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": "a",
                                            "type": "function",
                                            "function": {
                                                "name": "read",
                                                "arguments": '{"path":"a"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": "b",
                                            "type": "function",
                                            "function": {
                                                "name": "grep",
                                                "arguments": '{"pat":"x"}',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                ]
            ),
            lambda t: None,
        )
        calls = data["choices"][0]["message"]["tool_calls"]
        self.assertEqual([c["id"] for c in calls], ["a", "b"])
        self.assertEqual(
            calls[1]["function"], {"name": "grep", "arguments": '{"pat":"x"}'}
        )

    def test_openai_stream_assembles_a_choice(self):
        seen = []
        data = backends._stream_openai(
            self._events(
                {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]},
                {"choices": [{"delta": {"content": " there"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "grep",
                                            "arguments": '{"pa',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": 't": "x"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
            ),
            seen.append,
        )
        self.assertEqual(seen, ["Hi", " there"])
        msg = data["choices"][0]["message"]
        self.assertEqual(msg["content"], "Hi there")
        self.assertEqual(
            msg["tool_calls"][0]["function"],
            {"name": "grep", "arguments": '{"pat": "x"}'},
        )
        self.assertEqual(data["usage"]["completion_tokens"], 7)
        with mock.patch.object(backends, "BACKEND", "openai"):
            text, calls = backends._parse_native_response(data)
        self.assertEqual((text, calls[0].input), ("Hi there", {"pat": "x"}))

    def test_http_stream_reads_data_lines_and_skips_the_rest(self):
        body = b'event: ping\n: keep-alive\n\ndata: {"a": 1}\n\ndata: not json\n\ndata: [DONE]\n'
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(body)
        cm.__iter__ = lambda self: iter(io.BytesIO(body))
        cm.__exit__.return_value = False
        with mock.patch("urllib.request.urlopen", return_value=cm) as urlopen:
            events = list(backends._http_stream("https://x/y", {"q": 1}, {"h": "v"}))
        self.assertEqual(events, [{"a": 1}])
        sent = json.loads(urlopen.call_args[0][0].data)
        self.assertEqual(sent, {"q": 1})

    def test_get_response_streams_when_asked_and_not_otherwise(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 1},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
        ]
        seen = []
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "API_BASE", "https://x/v1/messages"),
            mock.patch.object(backends, "API_KEY", "k"),
            mock.patch.object(
                backends, "_http_stream", return_value=iter(events)
            ) as stream,
            mock.patch.object(
                backends, "_http_post", return_value={"content": [], "usage": {}}
            ) as post,
        ):
            raw = backends.get_response(
                [{"role": "user", "content": "x"}], "s", None, [], seen.append
            )
            self.assertTrue(stream.call_args[0][1]["stream"])
            self.assertEqual(json.loads(raw)["content"][0]["text"], "ok")
            backends.get_response([{"role": "user", "content": "x"}], "s", None, [])
            self.assertNotIn("stream", post.call_args[0][1])
            with mock.patch.object(backends, "STREAM", False):
                backends.get_response(
                    [{"role": "user", "content": "x"}], "s", None, [], seen.append
                )
            self.assertEqual(post.call_count, 2)
        self.assertEqual(seen, ["ok"])

    def test_stream_printer_renders_as_lines_complete(self):
        started = []
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch(
                "shutil.get_terminal_size", return_value=os.terminal_size((100, 40))
            ),
        ):
            p = ui.StreamPrinter(on_first=lambda: started.append(1))
            p.feed("Use `x`")
            p.feed(" now\n")
            p.feed("```py\nx = 1\n```\n")
            p.feed("done")
            p.close()
            out = sys.stdout.getvalue()
        self.assertEqual(started, [1])
        plain = strip_ansi(out)
        # raw while it streams, then the finished line redrawn in place, dot kept
        self.assertTrue(
            plain.startswith("● Use `x`\r\x1b[J● Use x now\n"), repr(plain[:40])
        )
        self.assertIn("  ┌─ py\n", plain)
        self.assertIn("│ x = 1\n", plain)
        self.assertIn("  └─\n", plain)
        self.assertTrue(plain.endswith("done\n\n"), repr(plain[-20:]))
        self.assertIn(f"{ui.CODE_TEXT}", out)

    def test_stream_printer_redraws_a_wrapped_line_from_its_first_row(self):
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch(
                "shutil.get_terminal_size", return_value=os.terminal_size((20, 40))
            ),
        ):
            p = ui.StreamPrinter()
            p.feed("- **bold** " + "x" * 30)  # 2 + 41 cells: three rows
            p.feed("\n")
            out = sys.stdout.getvalue()
        plain = strip_ansi(out)
        self.assertIn("\x1b[2A\r\x1b[J", out)  # back to the first row, then clear
        self.assertIn("● - bold " + "x" * 30 + "\n", plain)

    def test_stream_printer_counts_the_gutter_of_a_wrapped_code_line(self):
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch(
                "shutil.get_terminal_size", return_value=os.terminal_size((20, 40))
            ),
        ):
            p = ui.StreamPrinter()
            p.feed("```\n" + "y" * 17)  # "  │ " + 17 cells: two rows
            p.feed("\n")
            out = sys.stdout.getvalue()
        self.assertIn("\x1b[1A\r\x1b[J", out)

    def test_stream_printer_drops_text_after_close(self):
        with mock.patch("sys.stdout", io.StringIO()):
            p = ui.StreamPrinter()
            p.feed("partial")
            p.close()
            p.feed(" still streaming after Escape\n")
            p.close()
            out = strip_ansi(sys.stdout.getvalue())
        self.assertEqual(out, "● partial\n\n")

    def test_stream_printer_stays_quiet_without_text(self):
        with mock.patch("sys.stdout", io.StringIO()):
            p = ui.StreamPrinter()
            p.close()
            self.assertEqual(sys.stdout.getvalue(), "")
        self.assertFalse(p.started)
