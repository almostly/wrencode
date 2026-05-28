"""Tests for wrencode.py — stdlib unittest, no extra dependencies.

Run (from the repo root — stdlib only, no extra deps):
    python -m unittest test_wrencode -v
    # or:  python test_wrencode.py
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import wrencode

# ANSI escape codes that wrencode emits
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdown(unittest.TestCase):

    def test_bold(self):
        result = wrencode.render_markdown("**hello**")
        self.assertIn(BOLD, result)
        self.assertIn("hello", result)
        self.assertIn(RESET, result)

    def test_bold_not_in_output_as_asterisks(self):
        result = wrencode.render_markdown("**hello**")
        # The raw ** markers should be consumed, not left in output
        self.assertNotIn("**hello**", result)

    def test_inline_code(self):
        result = wrencode.render_markdown("`foo`")
        self.assertIn(CYAN, result)
        self.assertIn("foo", result)

    def test_fenced_code_block_has_border(self):
        result = wrencode.render_markdown("```python\ndef f(): pass\n```")
        # Border characters produced by render_markdown
        self.assertIn("┌─", result)
        self.assertIn("└─", result)
        # Language label is shown in the header
        self.assertIn("python", result)

    def test_fenced_code_block_content_highlighted(self):
        result = wrencode.render_markdown("```python\ndef f(): pass\n```")
        # 'def' should be colorized (BLUE keyword)
        self.assertIn(BLUE, result)
        self.assertIn("def", result)

    def test_bold_inside_fenced_block_is_not_expanded(self):
        """** inside a code fence must NOT be rendered as bold."""
        result = wrencode.render_markdown("```\n**not bold**\n```")
        # The BOLD escape should not appear in the fenced section,
        # or if it does, the literal '**' markers should still be in the output.
        plain = strip_ansi(result)
        # After stripping ANSI codes the raw ** should still be present
        self.assertIn("**not bold**", plain)

    def test_mixed_text(self):
        result = wrencode.render_markdown("Use **bold** and `code` together")
        self.assertIn(BOLD, result)
        self.assertIn(CYAN, result)

    def test_no_markdown(self):
        result = wrencode.render_markdown("plain text")
        self.assertEqual(result, "plain text")

    def test_fenced_block_without_language(self):
        result = wrencode.render_markdown("```\nsome code\n```")
        self.assertIn("┌─", result)
        self.assertIn("some code", result)


# ---------------------------------------------------------------------------
# Message blocks & confirm
# ---------------------------------------------------------------------------

class TestMessageBlocks(unittest.TestCase):

    def test_print_agent_message_uses_muted_grey_no_label(self):
        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            wrencode.print_agent_message("**done**")
        out = buf.getvalue()
        self.assertIn(wrencode.AGENT_TEXT, out)
        self.assertNotIn("Wren", out)
        self.assertNotIn("You", out)
        self.assertNotIn("\x1b[48;", out)  # no background color
        self.assertIn("done", strip_ansi(out))

    def test_print_tool_action_has_no_background(self):
        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            wrencode.print_tool_action("read", {"path": "foo.py"})
        out = buf.getvalue()
        self.assertNotIn("\x1b[48;", out)
        self.assertIn("read", strip_ansi(out).lower())
        self.assertIn("foo.py", strip_ansi(out))
        self.assertNotIn("read read", strip_ansi(out).lower())

    def test_print_tool_action_no_duplicate_write(self):
        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            wrencode.print_tool_action(
                "write", {"path": "/tmp/x.txt", "content": ""}
            )
        plain = strip_ansi(buf.getvalue()).lower()
        self.assertIn("write", plain)
        self.assertNotIn("write write", plain)

    def test_print_system_uses_banner_cyan(self):
        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf), mock.patch.object(
            wrencode, "colors_enabled", return_value=True
        ):
            wrencode.print_system("Cleared")
        out = buf.getvalue()
        self.assertIn("\033[96m", out)
        self.assertIn("\033[1m", out)
        self.assertIn("Cleared", strip_ansi(out))

    def test_context_loader_frame(self):
        with mock.patch.object(wrencode, "BACKEND", "anthropic"), mock.patch.object(
            wrencode, "MODEL", "claude-sonnet-4-20250514"
        ):
            plain = wrencode.loader_display(0)
            self.assertIn("anthropic", strip_ansi(plain))
            self.assertIn("claude-sonnet", strip_ansi(plain))
            self.assertIn("waiting", strip_ansi(plain))

    def test_loader_display_gradient(self):
        with mock.patch.object(wrencode, "colors_enabled", return_value=True):
            out = wrencode.loader_display(0)
        self.assertIn("\033[96m", out)
        self.assertIn("\033[2m", out)
        self.assertIn("waiting", strip_ansi(out))

    def test_format_input_line_slash_is_bold_cyan(self):
        with mock.patch.object(wrencode, "colors_enabled", return_value=True):
            slash = wrencode.format_input_line("/backend")
            slash_args = wrencode.format_input_line("/backend fgggg")
            chat = wrencode.format_input_line("hello")
        self.assertIn("\033[1m", slash)
        self.assertIn("/backend", strip_ansi(slash))
        self.assertIn("\033[1m", slash_args)
        plain_args = strip_ansi(slash_args)
        self.assertIn("/backend fgggg", plain_args)
        bold_parts = slash_args.split("\033[1m")
        self.assertEqual(len(bold_parts), 2)  # bold wraps /backend only
        self.assertIn("fgggg", bold_parts[-1])
        self.assertNotIn("\033[1m", chat)
        self.assertIn("hello", strip_ansi(chat))

    def test_read_user_input_fallback(self):
        import io

        with mock.patch("sys.stdin", io.StringIO("hello\n")), mock.patch.object(
            wrencode, "colors_enabled", return_value=False
        ):
            self.assertEqual(wrencode.read_user_input(), "hello")


class TestConfirm(unittest.TestCase):

    def setUp(self):
        self._orig_auto = os.environ.get("WRENCODE_AUTO_APPROVE")
        self._orig_session = wrencode._SESSION_AUTO_APPROVE
        wrencode._SESSION_AUTO_APPROVE = False
        if "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]

    def tearDown(self):
        wrencode._SESSION_AUTO_APPROVE = self._orig_session
        if self._orig_auto is not None:
            os.environ["WRENCODE_AUTO_APPROVE"] = self._orig_auto
        elif "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]

    def test_confirm_no_duplicate_write_prompt(self):
        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf), mock.patch("builtins.input", return_value=""):
            wrencode.confirm("write")
        plain = strip_ansi(buf.getvalue())
        self.assertNotIn("PosixPath", plain)
        self.assertNotIn("⚠ write", plain)
        self.assertIn("approve once", plain)

    def test_enter_approves(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(wrencode.confirm("write"), "ok")

    def test_allow_all_sets_session_flag(self):
        with mock.patch("builtins.input", return_value="a"):
            self.assertEqual(wrencode.confirm("write"), "ok")
        self.assertTrue(wrencode._SESSION_AUTO_APPROVE)

    def test_decline_collects_feedback(self):
        with mock.patch("builtins.input", return_value="n"), mock.patch.object(
            wrencode, "read_feedback_line", return_value="use edit instead"
        ):
            result = wrencode.confirm("write")
        self.assertEqual(result, "cancelled: user declined — use edit instead")

    def test_decline_without_feedback(self):
        with mock.patch("builtins.input", return_value="n"), mock.patch.object(
            wrencode, "read_feedback_line", return_value=""
        ):
            result = wrencode.confirm("write")
        self.assertEqual(result, "cancelled: user declined without instructions")

    def test_env_auto_approve(self):
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        self.assertEqual(wrencode.confirm("run"), "ok")

    def test_ctrl_c_returns_interrupted(self):
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertEqual(
                wrencode.confirm("write"),
                "cancelled: user interrupted",
            )

class TestChooseBackendInteractive(unittest.TestCase):

    def setUp(self):
        self._orig_config_file = wrencode.CONFIG_FILE
        self._orig_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        self._tmp = tempfile.mkdtemp()
        wrencode.CONFIG_FILE = pathlib.Path(self._tmp) / "config.json"

    def tearDown(self):
        wrencode.CONFIG_FILE = self._orig_config_file
        if self._orig_anthropic_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._orig_anthropic_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reconfigure_preserves_saved_api_key(self):
        wrencode.save_config(
            {
                "backend": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "api_key": "sk-secret",
            }
        )
        with mock.patch.object(wrencode, "pick_from_list", return_value=0), mock.patch.object(
            wrencode, "pick_model_interactive",
            return_value="claude-haiku-4-5-20251001",
        ), mock.patch("getpass.getpass", return_value=""):
            wrencode.choose_backend_interactive()
        saved = json.loads(wrencode.CONFIG_FILE.read_text())
        self.assertEqual(saved["api_key"], "sk-secret")
        self.assertEqual(wrencode.API_KEY, "sk-secret")


class TestToolArgs(unittest.TestCase):

    def test_bash_accepts_command_alias(self):
        args = wrencode.normalize_tool_args("bash", {"command": "echo hi"})
        self.assertEqual(args["cmd"], "echo hi")

    def test_format_bash_shows_full_command(self):
        text = wrencode.format_tool_action(
            "bash", {"command": "cat << 'EOF'\nhello\nEOF"}
        )
        self.assertIn("$ cat << 'EOF'", text)
        self.assertIn("hello", text)

    def test_run_tool_bash_with_command_alias(self):
        self._orig = os.environ.get("WRENCODE_AUTO_APPROVE")
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        try:
            result = wrencode.run_tool("bash", {"command": "echo wrencode-test"})
        finally:
            if self._orig is not None:
                os.environ["WRENCODE_AUTO_APPROVE"] = self._orig
            elif "WRENCODE_AUTO_APPROVE" in os.environ:
                del os.environ["WRENCODE_AUTO_APPROVE"]
        self.assertIn("wrencode-test", result)


class TestModelPicker(unittest.TestCase):

    def test_list_models_includes_current_and_custom(self):
        wrencode.apply_backend("anthropic", model="claude-haiku-4-5-20251001")
        models = wrencode.list_models_for_backend("anthropic")
        self.assertIn("claude-haiku-4-5-20251001", models)
        self.assertIn(wrencode.CUSTOM_MODEL_OPTION, models)

    def test_pick_from_list_numbered(self):
        with mock.patch("builtins.input", return_value="2"):
            idx = wrencode.pick_from_list(
                "Pick", ["a", "b", "c"], labels=["A", "B", "C"]
            )
        self.assertEqual(idx, 1)

    def test_switch_model_direct_id(self):
        wrencode.apply_backend("anthropic", model="claude-haiku-4-5-20251001")
        self._orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        self._orig_config = wrencode.CONFIG_FILE
        self._tmp = tempfile.mkdtemp()
        wrencode.CONFIG_FILE = pathlib.Path(self._tmp) / "config.json"
        try:
            result = wrencode.switch_model_runtime("claude-sonnet-4-20250514")
            self.assertIsNone(result)
            self.assertEqual(wrencode.MODEL, "claude-sonnet-4-20250514")
        finally:
            wrencode.CONFIG_FILE = self._orig_config
            if self._orig is not None:
                os.environ["ANTHROPIC_API_KEY"] = self._orig
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# _highlight_code
# ---------------------------------------------------------------------------

class TestHighlightCode(unittest.TestCase):

    def test_keyword_is_blue(self):
        result = wrencode._highlight_code("def foo():")
        self.assertIn(BLUE, result)
        self.assertIn("def", result)

    def test_string_is_green(self):
        result = wrencode._highlight_code('x = "hello"')
        self.assertIn(GREEN, result)

    def test_comment_is_dim(self):
        result = wrencode._highlight_code("x = 1  # comment")
        self.assertIn(DIM, result)
        self.assertIn("comment", result)

    def test_number_is_yellow(self):
        result = wrencode._highlight_code("return 42")
        self.assertIn(YELLOW, result)

    def test_multiple_tokens(self):
        result = wrencode._highlight_code('if x == "ok": return True')
        self.assertIn(BLUE, result)   # 'if', 'return', 'True' → blue
        self.assertIn(GREEN, result)  # "ok" → green

    def test_no_tokens_unchanged(self):
        code = "x y z"
        result = wrencode._highlight_code(code)
        self.assertIn("x y z", strip_ansi(result))


# ---------------------------------------------------------------------------
# strip_gptoss_tokens
# ---------------------------------------------------------------------------

class TestStripGptossTokens(unittest.TestCase):

    def test_strips_channel_prefix(self):
        text = "<|channel|>final<|message|>actual content"
        self.assertEqual(wrencode.strip_gptoss_tokens(text), "actual content")

    def test_strips_generic_tokens(self):
        result = wrencode.strip_gptoss_tokens("<|start|>hello<|end|>")
        self.assertEqual(result, "hello")

    def test_no_tokens_unchanged(self):
        self.assertEqual(wrencode.strip_gptoss_tokens("hello world"), "hello world")

    def test_strips_and_strips_whitespace(self):
        result = wrencode.strip_gptoss_tokens("  <|tok|>  hello  ")
        self.assertEqual(result, "hello")

    def test_multiple_channel_segments_takes_last(self):
        text = "<|channel|>final<|message|>first<|channel|>final<|message|>second"
        # split on the token takes the last segment
        self.assertEqual(wrencode.strip_gptoss_tokens(text), "second")


# ---------------------------------------------------------------------------
# truncate_at_turn_leak
# ---------------------------------------------------------------------------

class TestTruncateAtTurnLeak(unittest.TestCase):

    def test_truncates_at_user_marker(self):
        result = wrencode.truncate_at_turn_leak("Hello\nUser: leaked text")
        self.assertEqual(result, "Hello")

    def test_truncates_at_system_marker(self):
        result = wrencode.truncate_at_turn_leak("Hello\nSystem: leaked")
        self.assertEqual(result, "Hello")

    def test_truncates_at_human_marker(self):
        result = wrencode.truncate_at_turn_leak("Answer\nHuman: next turn")
        self.assertEqual(result, "Answer")

    def test_double_newline_user_marker(self):
        result = wrencode.truncate_at_turn_leak("Hello\n\nUser: next")
        self.assertEqual(result, "Hello")

    def test_double_newline_system_marker(self):
        result = wrencode.truncate_at_turn_leak("Hello\n\nSystem: next")
        self.assertEqual(result, "Hello")

    def test_no_leak_returns_unchanged(self):
        self.assertEqual(wrencode.truncate_at_turn_leak("No leak here"), "No leak here")

    def test_empty_string(self):
        self.assertEqual(wrencode.truncate_at_turn_leak(""), "")


# ---------------------------------------------------------------------------
# _tool_call_complete
# ---------------------------------------------------------------------------

class TestToolCallComplete(unittest.TestCase):

    def test_complete_with_end_tag(self):
        text = '<tool_call>{"tool": "read", "args": {"path": "foo.py"}}</tool_call>'
        end = wrencode._tool_call_complete(text)
        # Should return the position right after </tool_call>
        self.assertEqual(end, len(text))

    def test_complete_without_end_tag_uses_brace_matching(self):
        text = '<tool_call>{"tool": "read", "args": {"path": "foo.py"}}'
        end = wrencode._tool_call_complete(text)
        # Should return the position after the closing brace
        self.assertGreater(end, 0)
        # The character just before end should be '}'
        self.assertEqual(text[end - 1], "}")

    def test_no_tool_call_returns_minus_one(self):
        self.assertEqual(wrencode._tool_call_complete("no tool call"), -1)

    def test_open_tag_no_brace_returns_minus_one(self):
        self.assertEqual(wrencode._tool_call_complete("<tool_call>"), -1)

    def test_incomplete_json_returns_minus_one(self):
        # JSON opened but not closed
        self.assertEqual(wrencode._tool_call_complete("<tool_call>{incomplete"), -1)

    def test_nested_braces(self):
        # args value contains a JSON object itself
        text = '<tool_call>{"tool": "write", "args": {"path": "f", "content": "{a: 1}"}}'
        end = wrencode._tool_call_complete(text)
        self.assertGreater(end, 0)
        self.assertEqual(text[end - 1], "}")


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------

class TestParseToolCalls(unittest.TestCase):

    def test_single_call_with_end_tag(self):
        text = '<tool_call>{"tool": "read", "args": {"path": "foo.py"}}</tool_call>'
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "tool_use")
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(calls[0]["input"], {"path": "foo.py"})

    def test_call_id_format(self):
        text = '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(calls[0]["id"], "call_0")

    def test_single_call_without_end_tag(self):
        text = '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}'
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "glob")

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(wrencode.parse_tool_calls(""), [])

    def test_no_tool_calls_returns_empty_list(self):
        self.assertEqual(wrencode.parse_tool_calls("Just some plain text"), [])

    def test_unknown_tool_is_ignored(self):
        text = '<tool_call>{"tool": "not_a_real_tool", "args": {}}</tool_call>'
        self.assertEqual(wrencode.parse_tool_calls(text), [])

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"tool": "read", "args": {"path": "a.py"}}</tool_call>'
            '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        )
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(calls[1]["name"], "glob")

    def test_ids_are_sequential(self):
        text = (
            '<tool_call>{"tool": "read", "args": {"path": "a.py"}}</tool_call>'
            '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        )
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(calls[0]["id"], "call_0")
        self.assertEqual(calls[1]["id"], "call_1")

    def test_call_surrounded_by_text(self):
        text = 'thinking... <tool_call>{"tool": "grep", "args": {"pat": "TODO"}}</tool_call> done.'
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "grep")

    def test_nested_braces_in_args(self):
        # args value with nested JSON object
        payload = json.dumps({"path": "f.py", "content": '{"key": "val"}'})
        text = f'<tool_call>{{"tool": "write", "args": {payload}}}</tool_call>'
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write")

    def test_malformed_block_without_json_is_skipped(self):
        text = (
            "<tool_call>not-json</tool_call>"
            '<tool_call>{"tool": "read", "args": {"path": "ok.py"}}</tool_call>'
        )
        calls = wrencode.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read")


# ---------------------------------------------------------------------------
# apply_backend
# ---------------------------------------------------------------------------

class TestApplyBackend(unittest.TestCase):

    def setUp(self):
        # Save and clear any env vars that would affect tests
        self._saved_model = os.environ.pop("MODEL", None)
        self._saved_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._saved_model is not None:
            os.environ["MODEL"] = self._saved_model
        elif "MODEL" in os.environ:
            del os.environ["MODEL"]
        if self._saved_anthropic_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._saved_anthropic_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_anthropic_defaults(self):
        wrencode.apply_backend("anthropic")
        self.assertEqual(wrencode.BACKEND, "anthropic")
        self.assertEqual(wrencode.MODEL, wrencode.BACKEND_SPECS["anthropic"]["model"])
        self.assertEqual(wrencode.API_BASE, wrencode.BACKEND_SPECS["anthropic"]["api_base"])

    def test_anthropic_model_override(self):
        wrencode.apply_backend("anthropic", model="claude-opus-4-5")
        self.assertEqual(wrencode.MODEL, "claude-opus-4-5")

    def test_anthropic_env_model_beats_explicit(self):
        os.environ["MODEL"] = "claude-env-model"
        wrencode.apply_backend("anthropic", model="claude-explicit")
        self.assertEqual(wrencode.MODEL, "claude-env-model")

    def test_anthropic_env_api_key_beats_explicit(self):
        os.environ["ANTHROPIC_API_KEY"] = "env-key-abc"
        wrencode.apply_backend("anthropic", api_key="explicit-key")
        self.assertEqual(wrencode.API_KEY, "env-key-abc")

    def test_ollama_defaults(self):
        wrencode.apply_backend("ollama")
        self.assertEqual(wrencode.BACKEND, "ollama")
        self.assertEqual(wrencode.MODEL, wrencode.BACKEND_SPECS["ollama"]["model"])
        # Ollama uses a dummy key so the auth field is non-empty
        self.assertEqual(wrencode.API_KEY, "ollama")

    def test_ollama_api_base_uses_localhost(self):
        os.environ.pop("OLLAMA_HOST", None)
        wrencode.apply_backend("ollama")
        self.assertIn("localhost:11434", wrencode.API_BASE)

    def test_ollama_respects_ollama_host_env(self):
        os.environ["OLLAMA_HOST"] = "http://192.168.1.5:11434"
        wrencode.apply_backend("ollama")
        self.assertIn("192.168.1.5:11434", wrencode.API_BASE)
        del os.environ["OLLAMA_HOST"]

    def test_sets_backend_global(self):
        wrencode.apply_backend("anthropic")
        self.assertEqual(wrencode.BACKEND, "anthropic")
        wrencode.apply_backend("ollama")
        self.assertEqual(wrencode.BACKEND, "ollama")


# ---------------------------------------------------------------------------
# resolve_configuration
# ---------------------------------------------------------------------------

class TestResolveConfiguration(unittest.TestCase):

    def setUp(self):
        self._saved_backend = os.environ.pop("BACKEND", None)
        self._orig_config_file = wrencode.CONFIG_FILE
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        if self._saved_backend is not None:
            os.environ["BACKEND"] = self._saved_backend
        elif "BACKEND" in os.environ:
            del os.environ["BACKEND"]
        wrencode.CONFIG_FILE = self._orig_config_file
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_backend_env_var_wins(self):
        os.environ["BACKEND"] = "anthropic"
        wrencode.resolve_configuration()
        self.assertEqual(wrencode.BACKEND, "anthropic")

    def test_backend_env_var_unknown_raises(self):
        os.environ["BACKEND"] = "totally_unknown_backend"
        with self.assertRaises(SystemExit) as cm:
            wrencode.resolve_configuration()
        self.assertEqual(cm.exception.code, 1)

    def test_saved_config_is_used(self):
        # Point CONFIG_FILE at a tmpdir with a known config
        config_path = pathlib.Path(self._tmp) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"backend": "ollama"}, f)
        wrencode.CONFIG_FILE = config_path
        wrencode.resolve_configuration()
        self.assertEqual(wrencode.BACKEND, "ollama")

    def test_saved_config_with_model(self):
        config_path = pathlib.Path(self._tmp) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"backend": "ollama", "model": "mistral"}, f)
        wrencode.CONFIG_FILE = config_path
        wrencode.resolve_configuration()
        # MODEL should be mistral unless overridden by env
        saved_model_env = os.environ.pop("MODEL", None)
        try:
            wrencode.resolve_configuration()
            self.assertEqual(wrencode.MODEL, "mistral")
        finally:
            if saved_model_env is not None:
                os.environ["MODEL"] = saved_model_env

    def test_non_interactive_no_config_raises_system_exit(self):
        # Point to an empty tmpdir (no config.json) and make stdin non-tty
        wrencode.CONFIG_FILE = pathlib.Path(self._tmp) / "config.json"
        orig_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: False
        try:
            with self.assertRaises(SystemExit) as cm:
                wrencode.resolve_configuration()
            self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stdin.isatty = orig_isatty


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
        import shutil
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
        result = wrencode.write({"path": "hello.txt", "content": "world"})
        self.assertEqual(result, "ok")
        self.assertTrue(pathlib.Path(self._tmp, "hello.txt").exists())

    def test_read_returns_content_with_line_numbers(self):
        wrencode.write({"path": "hello.txt", "content": "line one\nline two"})
        result = wrencode.read({"path": "hello.txt"})
        self.assertIn("line one", result)
        self.assertIn("line two", result)
        # Line numbers should be present
        self.assertIn("1|", result)
        self.assertIn("2|", result)

    def test_write_then_read_roundtrip(self):
        content = "alpha\nbeta\ngamma"
        wrencode.write({"path": "data.txt", "content": content})
        result = wrencode.read({"path": "data.txt"})
        self.assertIn("alpha", result)
        self.assertIn("gamma", result)

    def test_read_nonexistent_file(self):
        result = wrencode.read({"path": "ghost.txt"})
        self.assertIn("error", result.lower())


    def test_edit_replaces_unique_string(self):
        wrencode.write({"path": "f.txt", "content": "hello world"})
        result = wrencode.edit({"path": "f.txt", "old": "hello", "new": "goodbye"})
        self.assertEqual(result, "ok")
        read_back = wrencode.read({"path": "f.txt"})
        self.assertIn("goodbye", read_back)
        self.assertNotIn("hello", read_back)

    def test_edit_errors_on_missing_old_string(self):
        wrencode.write({"path": "f.txt", "content": "hello world"})
        result = wrencode.edit({"path": "f.txt", "old": "notfound", "new": "x"})
        self.assertIn("error", result.lower())
        self.assertIn("not found", result)

    def test_edit_errors_on_ambiguous_match(self):
        wrencode.write({"path": "f.txt", "content": "foo foo foo"})
        result = wrencode.edit({"path": "f.txt", "old": "foo", "new": "bar"})
        self.assertIn("error", result.lower())
        self.assertIn("3", result)  # count of occurrences

    def test_edit_all_flag_replaces_all(self):
        wrencode.write({"path": "f.txt", "content": "foo foo foo"})
        result = wrencode.edit({"path": "f.txt", "old": "foo", "new": "bar", "all": True})
        self.assertEqual(result, "ok")
        read_back = strip_ansi(wrencode.read({"path": "f.txt"}))
        self.assertNotIn("foo", read_back)
        self.assertEqual(read_back.count("bar"), 3)

    def test_edit_no_op_is_rejected(self):
        wrencode.write({"path": "f.txt", "content": "unchanged"})
        result = wrencode.edit({"path": "f.txt", "old": "unchanged", "new": "unchanged"})
        self.assertIn("error", result.lower())
        self.assertIn("no change", result.lower())

    def test_edit_invalid_python_is_rejected(self):
        wrencode.write({"path": "bad.py", "content": "def ok():\n    return 1\n"})
        result = wrencode.edit({"path": "bad.py", "old": "return 1", "new": "return ("})
        self.assertIn("error", result.lower())
        self.assertIn("invalid python", result.lower())

    def test_edit_invalid_json_is_rejected(self):
        wrencode.write({"path": "bad.json", "content": '{"a": 1}'})
        result = wrencode.edit({"path": "bad.json", "old": "1", "new": "}"})
        self.assertIn("error", result.lower())
        self.assertIn("invalid json", result.lower())

    def test_edit_nonexistent_file(self):
        result = wrencode.edit({"path": "nope.txt", "old": "x", "new": "y"})
        self.assertIn("error", result.lower())


    def test_glob_finds_matching_files(self):
        wrencode.write({"path": "a.py", "content": "x"})
        wrencode.write({"path": "b.py", "content": "y"})
        wrencode.write({"path": "c.txt", "content": "z"})
        result = wrencode.glob({"pat": "*.py"})
        self.assertIn("a.py", result)
        self.assertIn("b.py", result)
        self.assertNotIn("c.txt", result)

    def test_glob_no_match_returns_none(self):
        result = wrencode.glob({"pat": "*.xyz"})
        self.assertEqual(result.strip(), "none")

    def test_glob_accepts_pattern_key_alias(self):
        wrencode.write({"path": "sample.py", "content": "pass"})
        result = wrencode.glob({"pattern": "*.py"})
        self.assertIn("sample.py", result)


    def test_grep_finds_pattern(self):
        wrencode.write({"path": "code.py", "content": "# TODO: fix this\nx = 1"})
        result = wrencode.grep({"pat": "TODO", "path": "."})
        # Should find the match (output may vary with rg vs grep)
        self.assertIn("TODO", result)

    def test_grep_no_match_returns_none(self):
        wrencode.write({"path": "code.py", "content": "nothing here"})
        result = wrencode.grep({"pat": "XYZNOTFOUND", "path": "."})
        self.assertEqual(result.strip(), "none")

    def test_grep_accepts_file_path(self):
        wrencode.write({"path": "single.py", "content": "needle = 1"})
        result = wrencode.grep({"pat": "needle", "path": "single.py"})
        self.assertIn("needle", result)

    def test_grep_missing_path_returns_error(self):
        result = wrencode.grep({"pat": "x", "path": "missing.py"})
        self.assertIn("error", result.lower())
        self.assertIn("not found", result.lower())


    def test_outside_workspace_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            wrencode.resolve_tool_path("/etc/passwd")
        self.assertIn("outside workspace", str(cm.exception))

    def test_outside_workspace_allowed_with_unrestricted_flag(self):
        os.environ["WRENCODE_UNRESTRICTED_PATHS"] = "1"
        # Should not raise
        p = wrencode.resolve_tool_path("/etc/passwd")
        self.assertIsInstance(p, pathlib.Path)
        del os.environ["WRENCODE_UNRESTRICTED_PATHS"]

    def test_relative_path_resolves_under_workspace(self):
        p = wrencode.resolve_tool_path("subdir/file.txt")
        self.assertTrue(str(p).startswith(self._tmp))

    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            wrencode.resolve_tool_path("")


# ---------------------------------------------------------------------------
# Agent loop smoke test (XML tool-call path via mocked get_response)
# ---------------------------------------------------------------------------

class TestAgentLoopSmoke(unittest.TestCase):
    """Smoke-test run_agent_turn using a mock backend (no real LLM calls)."""

    def setUp(self):
        self._tmp = str(pathlib.Path(tempfile.mkdtemp()).resolve())
        self._orig_workspace = os.environ.get("WRENCODE_WORKSPACE")
        self._orig_auto_approve = os.environ.get("WRENCODE_AUTO_APPROVE")
        os.environ["WRENCODE_WORKSPACE"] = self._tmp
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        # Use ollama so the agent loop takes the XML tool-call path
        self._orig_backend = wrencode.BACKEND
        wrencode.apply_backend("ollama")
        self._orig_get_response = wrencode.get_response

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._orig_workspace is not None:
            os.environ["WRENCODE_WORKSPACE"] = self._orig_workspace
        elif "WRENCODE_WORKSPACE" in os.environ:
            del os.environ["WRENCODE_WORKSPACE"]
        if self._orig_auto_approve is not None:
            os.environ["WRENCODE_AUTO_APPROVE"] = self._orig_auto_approve
        elif "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]
        wrencode.get_response = self._orig_get_response
        wrencode.BACKEND = self._orig_backend

    def test_one_tool_call_then_final_answer(self):
        """Agent reads a file (one tool call) then gives a final answer."""
        # Write a file for the agent to read
        target = pathlib.Path(self._tmp) / "greeting.txt"
        target.write_text("Hello from wrencode!", encoding="utf-8")

        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state):
            call_count[0] += 1
            if call_count[0] == 1:
                return '<tool_call>{"tool": "read", "args": {"path": "greeting.txt"}}</tool_call>'
            return "The file says: Hello from wrencode!"

        wrencode.get_response = mock_get_response

        messages = [{"role": "user", "content": "Read greeting.txt for me"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=5)

        # Two get_response calls: one tool call + one final answer
        self.assertEqual(call_count[0], 2)
        # There should be a tool_result in the message history
        has_tool_result = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in messages
        )
        self.assertTrue(has_tool_result, "Expected a tool_result message in history")
        # Final assistant message should contain the final answer text
        assistant_texts = [
            block.get("text", "")
            for m in messages
            if m["role"] == "assistant"
            for block in (
                m["content"] if isinstance(m["content"], list) else []
            )
            if block.get("type") == "text"
        ]
        self.assertTrue(
            any("Hello from wrencode" in t for t in assistant_texts),
            f"Expected final answer in assistant messages, got: {assistant_texts}",
        )

    def test_no_tool_calls_terminates_immediately(self):
        """Agent with no tool calls in response terminates after one round."""
        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state):
            call_count[0] += 1
            return "Just a plain answer, no tools needed."

        wrencode.get_response = mock_get_response

        messages = [{"role": "user", "content": "Say hello"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=5)

        self.assertEqual(call_count[0], 1)

    def test_tool_call_dispatches_real_tool(self):
        """Verify that run_tool dispatches to the real read implementation."""
        target = pathlib.Path(self._tmp) / "real.txt"
        target.write_text("real content", encoding="utf-8")

        result = wrencode.run_tool("read", {"path": "real.txt"})
        self.assertIn("real content", result)

    def test_max_iters_cap(self):
        """Agent stops after max_iters if it keeps returning tool calls."""
        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state):
            call_count[0] += 1
            return '<tool_call>{"tool": "glob", "args": {"pat": "*.txt"}}</tool_call>'

        wrencode.get_response = mock_get_response

        messages = [{"role": "user", "content": "loop forever"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=3)

        self.assertEqual(call_count[0], 3)

    def test_repeated_identical_tool_error_stops_loop(self):
        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state):
            call_count[0] += 1
            return '<tool_call>{"tool": "read", "args": {"path": "missing.txt"}}</tool_call>'

        wrencode.get_response = mock_get_response

        messages = [{"role": "user", "content": "keep retrying"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=20)

        self.assertEqual(call_count[0], wrencode.TOOL_ERROR_REPEAT_LIMIT)


# ---------------------------------------------------------------------------
# Additional helpers / edge cases
# ---------------------------------------------------------------------------

class TestFlattenContent(unittest.TestCase):

    def test_plain_string_unchanged(self):
        self.assertEqual(wrencode.flatten_content("hello"), "hello")

    def test_none_returns_empty_string(self):
        self.assertEqual(wrencode.flatten_content(None), "")

    def test_text_block_extracted(self):
        content = [{"type": "text", "text": "hello world"}]
        self.assertEqual(wrencode.flatten_content(content), "hello world")

    def test_tool_use_block_serialized(self):
        content = [{"type": "tool_use", "name": "read", "input": {"path": "f.py"}}]
        result = wrencode.flatten_content(content)
        self.assertIn("<tool_call>", result)
        self.assertIn("read", result)

    def test_tool_result_block_included(self):
        content = [{"type": "tool_result", "content": "file contents"}]
        result = wrencode.flatten_content(content)
        self.assertIn("file contents", result)

    def test_multiple_blocks_joined(self):
        content = [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]
        result = wrencode.flatten_content(content)
        self.assertIn("part1", result)
        self.assertIn("part2", result)


class TestRunTool(unittest.TestCase):

    def setUp(self):
        self._tmp = str(pathlib.Path(tempfile.mkdtemp()).resolve())
        self._orig_workspace = os.environ.get("WRENCODE_WORKSPACE")
        self._orig_auto_approve = os.environ.get("WRENCODE_AUTO_APPROVE")
        os.environ["WRENCODE_WORKSPACE"] = self._tmp
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"

    def tearDown(self):
        import shutil
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
        result = wrencode.run_tool("not_a_tool", {})
        self.assertIn("error", result.lower())

    def test_run_tool_truncates_large_output(self):
        # The read tool caps at MAX_READ_LINES (default 800) before run_tool's
        # MAX_OUT character cap fires.  To breach MAX_OUT we need lines long
        # enough that 800 of them exceed 48 000 chars (800 * 60 = 48 000, so
        # use 65-char lines to stay safely above the threshold).
        line = "X" * 65
        big_content = (line + "\n") * 900  # 900 lines × 66 bytes = 59 400 bytes
        wrencode.write({"path": "big.txt", "content": big_content})
        result = wrencode.run_tool("read", {"path": "big.txt"})
        self.assertIn("truncated", result)


if __name__ == "__main__":
    unittest.main()
