"""The tools: arguments, file tools, the python tool, parsing tool calls."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from typing import Any
from unittest import mock

from tests.support import strip_ansi
from wrencode import (
    app,
    backends,
    sandbox,
)


class TestToolArgs(unittest.TestCase):
    def test_bash_accepts_command_alias(self):
        args = app.normalize_tool_args("bash", {"command": "echo hi"})
        self.assertEqual(args["cmd"], "echo hi")

    def test_format_bash_shows_full_command(self):
        text = app.format_tool_action("bash", {"command": "cat << 'EOF'\nhello\nEOF"})
        self.assertIn("$ cat << 'EOF'", text)
        self.assertIn("hello", text)

    def test_run_tool_bash_with_command_alias(self):
        self._orig = os.environ.get("WRENCODE_AUTO_APPROVE")
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        try:
            result = app.run_tool("bash", {"command": "echo wrencode-test"})
        finally:
            if self._orig is not None:
                os.environ["WRENCODE_AUTO_APPROVE"] = self._orig
            elif "WRENCODE_AUTO_APPROVE" in os.environ:
                del os.environ["WRENCODE_AUTO_APPROVE"]
        self.assertIn("wrencode-test", result)


# ---------------------------------------------------------------------------
# Parse tool calls
# ---------------------------------------------------------------------------
class TestParseToolCalls(unittest.TestCase):
    def test_single_call_with_end_tag(self):
        text = '<tool_call>{"tool": "read", "args": {"path": "foo.py"}}</tool_call>'
        calls = app.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "tool_use")
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(calls[0]["input"], {"path": "foo.py"})

    def test_call_id_format(self):
        text = '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        calls = app.parse_tool_calls(text)
        self.assertEqual(calls[0]["id"], "call_0")

    def test_single_call_without_end_tag(self):
        text = '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}'
        calls = app.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "glob")

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(app.parse_tool_calls(""), [])

    def test_no_tool_calls_returns_empty_list(self):
        self.assertEqual(app.parse_tool_calls("Just some plain text"), [])

    def test_unknown_tool_is_ignored(self):
        text = '<tool_call>{"tool": "not_a_real_tool", "args": {}}</tool_call>'
        self.assertEqual(app.parse_tool_calls(text), [])

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"tool": "read", "args": {"path": "a.py"}}</tool_call>'
            '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        )
        calls = app.parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(calls[1]["name"], "glob")

    def test_ids_are_sequential(self):
        text = (
            '<tool_call>{"tool": "read", "args": {"path": "a.py"}}</tool_call>'
            '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        )
        calls = app.parse_tool_calls(text)
        self.assertEqual(calls[0]["id"], "call_0")
        self.assertEqual(calls[1]["id"], "call_1")

    def test_call_surrounded_by_text(self):
        text = 'thinking... <tool_call>{"tool": "grep", "args": {"pat": "TODO"}}</tool_call> done.'
        calls = app.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "grep")

    def test_nested_braces_in_args(self):
        # args value with nested JSON object
        payload = json.dumps({"path": "f.py", "content": '{"key": "val"}'})
        text = f'<tool_call>{{"tool": "write", "args": {payload}}}</tool_call>'
        calls = app.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write")

    def test_malformed_block_without_json_is_skipped(self):
        text = (
            "<tool_call>not-json</tool_call>"
            '<tool_call>{"tool": "read", "args": {"path": "ok.py"}}</tool_call>'
        )
        calls = app.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read")


# ---------------------------------------------------------------------------
# File tools (read, write, edit, glob, grep) with workspace sandbox
# ---------------------------------------------------------------------------
class TestFileTools(unittest.TestCase):
    def setUp(self):
        # Resolve so macOS /var → /private/var symlinks don't cause startswith() mismatches
        self._tmp = str(pathlib.Path(tempfile.mkdtemp()).resolve())
        self._orig_workspace = os.environ.get("WRENCODE_WORKSPACE")
        self._orig_auto_approve = os.environ.get("WRENCODE_AUTO_APPROVE")
        self._orig_unrestricted = os.environ.get("WRENCODE_UNRESTRICTED_PATHS")
        os.environ["WRENCODE_WORKSPACE"] = self._tmp
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        os.environ.pop("WRENCODE_UNRESTRICTED_PATHS", None)

    def tearDown(self):

        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._orig_workspace is not None:
            os.environ["WRENCODE_WORKSPACE"] = self._orig_workspace
        elif "WRENCODE_WORKSPACE" in os.environ:
            del os.environ["WRENCODE_WORKSPACE"]
        if self._orig_auto_approve is not None:
            os.environ["WRENCODE_AUTO_APPROVE"] = self._orig_auto_approve
        elif "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]
        if self._orig_unrestricted is not None:
            os.environ["WRENCODE_UNRESTRICTED_PATHS"] = self._orig_unrestricted
        elif "WRENCODE_UNRESTRICTED_PATHS" in os.environ:
            del os.environ["WRENCODE_UNRESTRICTED_PATHS"]

    def test_write_creates_file(self):
        result = app.write({"path": "hello.txt", "content": "world"})
        self.assertEqual(result, "ok")
        self.assertTrue(pathlib.Path(self._tmp, "hello.txt").exists())

    def test_write_errors_when_content_missing(self):
        # Guards against the MAX_TOKENS-truncation failure mode where the
        # model emits a tool_use block with `path` but no `content`. Returning
        # an error lets _track_error halt the loop after a few retries.
        result = app.write({"path": "hello.txt"})
        self.assertTrue(result.startswith("error:"))
        self.assertFalse(pathlib.Path(self._tmp, "hello.txt").exists())

    def test_read_returns_content_with_line_numbers(self):
        app.write({"path": "hello.txt", "content": "line one\nline two"})
        result = app.read({"path": "hello.txt"})
        self.assertIn("line one", result)
        self.assertIn("line two", result)
        # Line numbers should be present
        self.assertIn("1|", result)
        self.assertIn("2|", result)

    def test_write_then_read_roundtrip(self):
        content = "alpha\nbeta\ngamma"
        app.write({"path": "data.txt", "content": content})
        result = app.read({"path": "data.txt"})
        self.assertIn("alpha", result)
        self.assertIn("gamma", result)

    def test_read_nonexistent_file(self):
        result = app.read({"path": "ghost.txt"})
        self.assertIn("error", result.lower())

    def test_edit_replaces_unique_string(self):
        app.write({"path": "f.txt", "content": "hello world"})
        result = app.edit({"path": "f.txt", "old": "hello", "new": "goodbye"})
        self.assertEqual(result, "ok")
        read_back = app.read({"path": "f.txt"})
        self.assertIn("goodbye", read_back)
        self.assertNotIn("hello", read_back)

    def test_edit_preserves_indentation(self):
        # `old` must match exactly: a stripped match anchor combined with an
        # indented `new` would duplicate the leading whitespace (8 -> 16 spaces).
        app.write(
            {"path": "f.py", "content": "def f(x):\n    if x:\n        return x\n"}
        )
        result = app.edit(
            {"path": "f.py", "old": "        return x", "new": "        return -x"}
        )
        self.assertEqual(result, "ok")
        read_back = strip_ansi(app.read({"path": "f.py"}))
        self.assertIn("        return -x", read_back)
        self.assertNotIn("                return", read_back)

    def test_edit_errors_on_missing_old_string(self):
        app.write({"path": "f.txt", "content": "hello world"})
        result = app.edit({"path": "f.txt", "old": "notfound", "new": "x"})
        self.assertIn("error", result.lower())
        self.assertIn("not found", result)

    def test_edit_errors_on_ambiguous_match(self):
        app.write({"path": "f.txt", "content": "foo foo foo"})
        result = app.edit({"path": "f.txt", "old": "foo", "new": "bar"})
        self.assertIn("error", result.lower())
        self.assertIn("3", result)  # count of occurrences

    def test_edit_all_flag_replaces_all(self):
        app.write({"path": "f.txt", "content": "foo foo foo"})
        result = app.edit({"path": "f.txt", "old": "foo", "new": "bar", "all": True})
        self.assertEqual(result, "ok")
        read_back = strip_ansi(app.read({"path": "f.txt"}))
        self.assertNotIn("foo", read_back)
        self.assertEqual(read_back.count("bar"), 3)

    def test_edit_no_op_is_rejected(self):
        app.write({"path": "f.txt", "content": "unchanged"})
        result = app.edit({"path": "f.txt", "old": "unchanged", "new": "unchanged"})
        self.assertIn("error", result.lower())
        self.assertIn("no change", result.lower())

    def test_edit_invalid_python_is_rejected(self):
        app.write({"path": "bad.py", "content": "def ok():\n    return 1\n"})
        result = app.edit({"path": "bad.py", "old": "return 1", "new": "return ("})
        self.assertIn("error", result.lower())
        self.assertIn("invalid python", result.lower())

    def test_edit_invalid_json_is_rejected(self):
        app.write({"path": "bad.json", "content": '{"a": 1}'})
        result = app.edit({"path": "bad.json", "old": "1", "new": "}"})
        self.assertIn("error", result.lower())
        self.assertIn("invalid json", result.lower())

    def test_edit_nonexistent_file(self):
        result = app.edit({"path": "nope.txt", "old": "x", "new": "y"})
        self.assertIn("error", result.lower())

    def test_glob_finds_matching_files(self):
        app.write({"path": "a.py", "content": "x"})
        app.write({"path": "b.py", "content": "y"})
        app.write({"path": "c.txt", "content": "z"})
        result = app.glob({"pat": "*.py"})
        self.assertIn("a.py", result)
        self.assertIn("b.py", result)
        self.assertNotIn("c.txt", result)

    def test_glob_no_match_returns_none(self):
        result = app.glob({"pat": "*.xyz"})
        self.assertEqual(result.strip(), "none")

    def test_glob_accepts_pattern_key_alias(self):
        app.write({"path": "sample.py", "content": "pass"})
        result = app.glob({"pattern": "*.py"})
        self.assertIn("sample.py", result)

    def test_grep_finds_pattern(self):
        app.write({"path": "code.py", "content": "# TODO: fix this\nx = 1"})
        result = app.grep({"pat": "TODO", "path": "."})
        # Should find the match (output may vary with rg vs grep)
        self.assertIn("TODO", result)

    def test_grep_no_match_returns_none(self):
        app.write({"path": "code.py", "content": "nothing here"})
        result = app.grep({"pat": "XYZNOTFOUND", "path": "."})
        self.assertEqual(result.strip(), "none")

    def test_grep_accepts_file_path(self):
        app.write({"path": "single.py", "content": "needle = 1"})
        result = app.grep({"pat": "needle", "path": "single.py"})
        self.assertIn("needle", result)

    def test_grep_missing_path_returns_error(self):
        result = app.grep({"pat": "x", "path": "missing.py"})
        self.assertIn("error", result.lower())
        self.assertIn("not found", result.lower())

    def test_outside_workspace_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            app.resolve_tool_path("/etc/passwd")
        self.assertIn("outside workspace", str(cm.exception))

    def test_outside_workspace_allowed_with_unrestricted_flag(self):
        os.environ["WRENCODE_UNRESTRICTED_PATHS"] = "1"
        # Should not raise
        p = app.resolve_tool_path("/etc/passwd")
        self.assertIsInstance(p, pathlib.Path)
        del os.environ["WRENCODE_UNRESTRICTED_PATHS"]

    def test_relative_path_resolves_under_workspace(self):
        p = app.resolve_tool_path("subdir/file.txt")
        self.assertTrue(str(p).startswith(self._tmp))

    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            app.resolve_tool_path("")


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self._tmp = str(pathlib.Path(tempfile.mkdtemp()).resolve())
        self._orig_workspace = os.environ.get("WRENCODE_WORKSPACE")
        self._orig_auto_approve = os.environ.get("WRENCODE_AUTO_APPROVE")
        os.environ["WRENCODE_WORKSPACE"] = self._tmp
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"

    def tearDown(self):

        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._orig_workspace is not None:
            os.environ["WRENCODE_WORKSPACE"] = self._orig_workspace
        elif "WRENCODE_WORKSPACE" in os.environ:
            del os.environ["WRENCODE_WORKSPACE"]
        if self._orig_auto_approve is not None:
            os.environ["WRENCODE_AUTO_APPROVE"] = self._orig_auto_approve
        elif "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]

    def test_unknown_tool_returns_error(self):
        result = app.run_tool("not_a_tool", {})
        self.assertIn("error", result.lower())

    def test_run_tool_truncates_large_output(self):
        # The read tool caps at MAX_READ_LINES (default 800) before run_tool's
        # MAX_OUT character cap fires.  To breach MAX_OUT we need lines long
        # enough that 800 of them exceed 48 000 chars (800 * 60 = 48 000, so
        # use 65-char lines to stay safely above the threshold).
        line = "X" * 65
        big_content = (line + "\n") * 900  # 900 lines × 66 bytes = 59 400 bytes
        app.write({"path": "big.txt", "content": big_content})
        result = app.run_tool("read", {"path": "big.txt"})
        self.assertIn("truncated", result)


try:
    import pydantic_monty
except ImportError:  # the python tool's tests skip without the sandbox
    pydantic_monty = None


@unittest.skipIf(pydantic_monty is None, "pydantic-monty not installed")
class TestPythonTool(unittest.TestCase):
    """The python tool as the agent sees it."""

    @classmethod
    def tearDownClass(cls):
        sandbox.close()

    def test_registered_and_described(self):
        self.assertIn("python", app.TOOLS)
        self.assertIn("python", [name for name, _, _ in app.tool_specs()])
        self.assertIn("python(code)", app.build_system_prompt())

    def test_runs_with_the_workspace_tools(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "f.txt").write_text("needle here\nsecond\n")
            pathlib.Path(d, "sub").mkdir()
            pathlib.Path(d, "sub", "g.txt").write_text("nothing\n")
            with mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": d}):
                out = app.python(
                    {
                        "code": "paths = sorted(glob('**/*.txt'))\n"
                        "print(paths)\n"
                        "print(len(read('f.txt').splitlines()))\n"
                        "print(grep('needle'))\n"
                        "open('f.txt').read().startswith('needle')"
                    }
                )
                missing = app.python({"code": "read('nope.txt')"})
        self.assertIn("['f.txt', 'sub/g.txt']", out)  # relative paths, a real list
        self.assertIn("\n2\n", out)  # raw text, not numbered lines
        self.assertIn("f.txt:1:needle here", out)
        self.assertTrue(out.endswith("True"))
        self.assertTrue(missing.startswith("error:"))
        self.assertIn("FileNotFoundError", missing)


class TestForgivingEdit(unittest.TestCase):
    CART = (
        "class Cart:\n"
        "    def remove(self, sku: str):\n"
        "        del self.items[sku]\n"
        "\n"
        "    def subtotal(self) -> int:\n"
        "        return 0\n"
    )

    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        self._env = mock.patch.dict(
            os.environ,
            {"WRENCODE_WORKSPACE": str(self._tmp), "WRENCODE_AUTO_APPROVE": "1"},
        )
        self._env.start()
        self._out = mock.patch("sys.stdout", io.StringIO())
        self._out.start()
        (self._tmp / "cart.py").write_text(self.CART)

    def tearDown(self):

        self._out.stop()
        self._env.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def edit(self, old, new):
        return app.edit({"path": "cart.py", "old": old, "new": new})

    def test_dedented_quote_is_reindented(self):
        # The exact failure Qwen3-8B hit 22 times: a method quoted at column 0.
        result = self.edit(
            "def remove(self, sku: str):\n    del self.items[sku]",
            "def remove(self, sku: str):\n    self.items.pop(sku, None)",
        )
        self.assertTrue(result.startswith("ok (matched lines 2-3"), result)
        self.assertIn("added 4 chars", result)
        text = (self._tmp / "cart.py").read_text()
        self.assertIn(
            "    def remove(self, sku: str):\n        self.items.pop(sku, None)\n", text
        )
        self.assertIn("    def subtotal", text)

    def test_over_indented_quote_is_dedented(self):
        result = self.edit("            return 0", "            return 42")
        self.assertIn("removed 4 chars", result)
        self.assertIn("        return 42\n", (self._tmp / "cart.py").read_text())

    def test_inconsistent_shift_is_rejected(self):
        # def is 2 spaces short of the file, body 4 short: no single shift fits.
        result = self.edit(
            "  def remove(self, sku: str):\n    del self.items[sku]", "x"
        )
        self.assertTrue(result.startswith("error:"), result)
        self.assertEqual((self._tmp / "cart.py").read_text(), self.CART)

    def test_ambiguous_reindent_is_rejected(self):
        (self._tmp / "cart.py").write_text(
            "def a():\n    x = 1\n\ndef b():\n    x = 1\n"
        )
        result = self.edit("x = 1", "x = 2")
        self.assertIn("appears 2 times", result)

    def test_not_found_shows_closest_lines(self):
        result = self.edit("def remove(self, sku):\n    del items[sku]", "x")
        self.assertIn("Closest match", result)
        self.assertIn("    2|     def remove(self, sku: str):", result)
        self.assertIn("including indentation", result)

    def test_unrelated_text_has_no_closest_match(self):
        result = self.edit("import numpy as np", "x")
        self.assertNotIn("Closest match", result)
        self.assertIn("Re-read the file", result)

    def test_wrong_file_points_to_the_right_one(self):
        # Qwen3-8B's next failure: editing to_cents in cart.py; it lives in money.py.
        (self._tmp / "shop").mkdir()
        (self._tmp / "shop" / "money.py").write_text(
            "import os\n\ndef to_cents(s: str) -> int:\n    return 0\n"
        )
        result = self.edit(
            "def to_cents(s: str) -> int:\n    return int(float(s) * 100)", "x"
        )
        self.assertIn("isn't in cart.py but appears in shop/money.py:3", result)

    def test_no_hint_when_nowhere_else(self):
        self.assertNotIn(
            "did you mean", self.edit("def missing_function():\n    pass", "x")
        )

    def test_reindented_result_still_syntax_checked(self):
        result = self.edit("return 0", "return (")
        self.assertIn("invalid Python", result)


class TestRepeatedFailingCalls(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        (self._tmp / "f.py").write_text("x = 1\n")
        self._patches = [
            mock.patch.dict(
                os.environ,
                {"WRENCODE_WORKSPACE": str(self._tmp), "WRENCODE_AUTO_APPROVE": "1"},
            ),
            mock.patch.object(backends, "BACKEND", "ollama"),
            mock.patch("sys.stdout", io.StringIO()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_interleaved_identical_failures_hint_then_stop(self):
        bad = '<tool_call>{"tool": "edit", "args": {"path": "f.py", "old": "y = 2", "new": "y = 3"}}</tool_call>'
        ok = '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        replies = iter([bad, ok] * 10)
        msgs = [{"role": "user", "content": "go"}]
        with mock.patch.object(backends, "get_response", lambda *a: next(replies)):
            reason = app.run_agent_turn(msgs, "sys", None)
        self.assertEqual(reason, "tool_errors")
        results = [
            b["content"]
            for m in msgs
            if isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_result"
        ]
        edits = [r for r in results if r.startswith("error:")]
        self.assertEqual(len(edits), app.REPEATED_CALL_STOP)
        self.assertNotIn("failed 2 times", edits[1])
        self.assertIn("has now failed 3 times", edits[2])

    def test_skipped_calls_still_get_results(self):
        bad = '{"tool": "edit", "args": {"path": "f.py", "old": "nope", "new": "z"}}'
        batch = (
            f"<tool_call>{bad}</tool_call><tool_call>{bad}</tool_call><tool_call>"
            + '{"tool": "glob", "args": {"pat": "*"}}</tool_call>'
        )
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        with (
            mock.patch.object(app, "TOOL_ERROR_REPEAT_LIMIT", 2),
            mock.patch.object(backends, "get_response", lambda *a: batch),
        ):
            self.assertEqual(app.run_agent_turn(msgs, "sys", None), "tool_errors")
        ids_called = [
            b["id"] for b in msgs[1]["content"] if b.get("type") == "tool_use"
        ]
        ids_answered = [b["tool_use_id"] for b in msgs[2]["content"]]
        self.assertEqual(ids_called, ids_answered)
        self.assertTrue(msgs[2]["content"][-1]["content"].startswith("skipped"))


# Trimmed from a real Qwen3-8B reply on vLLM: a "+" join inside the JSON arguments.
QWEN3_GARBLED_CALL = '<tool_call>\n{"name": "edit", "arguments": {"path": "shop/money.py", "old": "def to_cents(amount: str) -> int:\\n        \\"\\"\\"Parse \'12.34\' or \'12\' into cents.\\"\\"\\"\\n        if \'.\' in amount:\\n            whole, frac = amount.split(\'.\')" + "\\n            return int(whole) * 100 + int(frac.ljust(</tool_call>'


class TestGarbledToolCalls(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        (self._tmp / "a.txt").write_text("hello")
        self._patches = [
            mock.patch.dict(
                os.environ,
                {"WRENCODE_WORKSPACE": str(self._tmp), "WRENCODE_AUTO_APPROVE": "1"},
            ),
            mock.patch("sys.stdout", io.StringIO()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def loop(self, backend, replies):
        replies = iter(replies)
        msgs = [{"role": "user", "content": "go"}]
        with (
            mock.patch.object(backends, "BACKEND", backend),
            mock.patch.object(backends, "get_response", lambda *a: next(replies)),
        ):
            reason = app.run_agent_turn(msgs, "sys", None)
        return reason, msgs

    @staticmethod
    def openai(content, calls=None):
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        return json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]})

    def test_real_qwen3_call_is_explained(self):
        why = app._garbled_tool_call(QWEN3_GARBLED_CALL, native=True)
        self.assertIn("invalid JSON at character", why)
        self.assertIn("Expecting ',' delimiter", why)
        self.assertIn("split('.')\" + \"", why)

    def test_native_garbled_call_is_resent_not_final(self):
        good = self.openai(
            None,
            [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
                }
            ],
        )
        reason, msgs = self.loop(
            "openai", [self.openai(QWEN3_GARBLED_CALL), good, self.openai("Read it.")]
        )
        self.assertEqual(reason, "done")
        nudge = msgs[2]["content"]
        self.assertTrue(
            nudge.startswith(
                "Your last message contained a tool call that couldn't be run"
            )
        )
        self.assertEqual(msgs[-1]["content"], "Read it.")
        out = backends._to_openai_messages(
            msgs
        )  # history stays valid; the tag is defanged
        self.assertEqual([m["role"] for m in out], [m["role"] for m in msgs])
        self.assertNotIn("<tool_call>", json.dumps(out))

    def test_xml_garbled_call_is_resent(self):
        bad = '<tool_call>{"tool": "read", "args": {"path": "a" + ".txt"}}</tool_call>'
        good = '<tool_call>{"tool": "read", "args": {"path": "a.txt"}}</tool_call>'
        reason, msgs = self.loop("ollama", [bad, good, "It says hello."])
        self.assertEqual(reason, "done")
        results = [
            b["content"]
            for m in msgs
            if isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_result"
        ]
        self.assertTrue(any("hello" in r for r in results))

    def test_gives_up_after_repeated_garbling(self):
        reason, _ = self.loop("openai", [self.openai(QWEN3_GARBLED_CALL)] * 5)
        self.assertEqual(reason, "malformed_tool_call")

    def test_native_call_written_as_text(self):
        text = '<tool_call>{"name": "read", "arguments": {"path": "a.txt"}}</tool_call>'
        self.assertIn(
            "function-calling interface", app._garbled_tool_call(text, native=True)
        )

    def test_xml_unknown_tool(self):
        why = app._garbled_tool_call(
            '<tool_call>{"tool": "fly", "args": {}}</tool_call>', native=False
        )
        self.assertIn("unknown tool", why)

    def test_plain_final_answer_unaffected(self):
        self.assertEqual(
            app._garbled_tool_call("All done, tests pass.", native=True), ""
        )

    def test_hermes_shape_parsed_on_xml_path(self):
        calls = app.parse_tool_calls(
            '<tool_call>{"name": "read", "arguments": {"path": "a.txt"}}</tool_call>'
            '<tool_call>{"name": "glob", "arguments": "{\\"pat\\": \\"*.txt\\"}"}</tool_call>'
        )
        self.assertEqual(
            [(c["name"], c["input"]) for c in calls],
            [("read", {"path": "a.txt"}), ("glob", {"pat": "*.txt"})],
        )
