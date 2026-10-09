"""Tests for wrencode.py — stdlib unittest, no extra dependencies.

Run (from the repo root — stdlib only, no extra deps):
    python -m unittest test_wrencode -v
    # or:  python test_wrencode.py
"""

from __future__ import annotations

import datetime
import io
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from email.message import Message
from typing import Any, ClassVar
from unittest import mock

import wrencode
import wrencode_backends as backends
import wrencode_configure as configure
import wrencode_history as history
import wrencode_mcp as mcp
import wrencode_permissions as permissions
import wrencode_sandbox as sandbox
import wrencode_sdk as agent_sdk
import wrencode_synthesize as synthesize
import wrencode_ui as ui
import wrencode_web as web

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
# Render markdown
# ---------------------------------------------------------------------------
class TestRenderMarkdown(unittest.TestCase):
    def test_bold(self):
        result = ui.render_markdown("**hello**")
        self.assertIn(BOLD, result)
        self.assertIn("hello", result)
        self.assertIn(RESET, result)

    def test_bold_not_in_output_as_asterisks(self):
        result = ui.render_markdown("**hello**")
        # The raw ** markers should be consumed, not left in output
        self.assertNotIn("**hello**", result)

    def test_inline_code(self):
        result = ui.render_markdown("`foo`")
        self.assertIn(CYAN, result)
        self.assertIn("foo", result)

    def test_fenced_code_block_has_border(self):
        result = ui.render_markdown("```python\ndef f(): pass\n```")
        # Border characters produced by render_markdown
        self.assertIn("┌─", result)
        self.assertIn("└─", result)
        # Language label is shown in the header
        self.assertIn("python", result)

    def test_fenced_code_block_content_highlighted(self):
        result = ui.render_markdown("```python\ndef f(): pass\n```")
        # 'def' should be colorized (BLUE keyword)
        self.assertIn(BLUE, result)
        self.assertIn("def", result)

    def test_bold_inside_fenced_block_is_not_expanded(self):
        """** inside a code fence must NOT be rendered as bold."""
        result = ui.render_markdown("```\n**not bold**\n```")
        # The BOLD escape should not appear in the fenced section,
        # or if it does, the literal '**' markers should still be in the output.
        plain = strip_ansi(result)
        # After stripping ANSI codes the raw ** should still be present
        self.assertIn("**not bold**", plain)

    def test_mixed_text(self):
        result = ui.render_markdown("Use **bold** and `code` together")
        self.assertIn(BOLD, result)
        self.assertIn(CYAN, result)

    def test_no_markdown(self):
        result = ui.render_markdown("plain text")
        self.assertEqual(result, "plain text")

    def test_fenced_block_without_language(self):
        result = ui.render_markdown("```\nsome code\n```")
        self.assertIn("┌─", result)
        self.assertIn("some code", result)


# ---------------------------------------------------------------------------
# Message blocks & confirm
# ---------------------------------------------------------------------------
class TestMessageBlocks(unittest.TestCase):
    def test_print_agent_message_opens_with_a_dot_and_hangs_the_text(self):
        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            ui.print_agent_message("**done**\nsecond line")
        out = buf.getvalue()
        self.assertIn(ui.AGENT_TEXT, out)
        self.assertTrue(out.startswith(ui.AGENT_MARK + " "), repr(out[:30]))
        self.assertIn("\n  " + ui.AGENT_TEXT + "second line", out)
        self.assertNotIn("Wren", out)
        self.assertNotIn("You", out)
        self.assertNotIn("\x1b[48;", out)  # no background color
        self.assertIn("done", strip_ansi(out))

    def test_code_blocks_keep_their_tint_and_headings_are_bold(self):
        out = ui.render_markdown("## Plan\n```python\nx = 1  # one\n```")
        self.assertIn(f"{ui.BOLD}Plan{ui.RESET}", out)
        self.assertIn(f"{ui.DIM}│{ui.RESET} {ui.CODE_TEXT}", out)
        # after a highlighted token the code color comes back, not the prose color
        self.assertIn(f"{ui.YELLOW}1{ui.RESET}{ui.CODE_TEXT}", out)
        self.assertIn(f"{ui.DIM}# one{ui.RESET}{ui.CODE_TEXT}", out)
        self.assertNotIn("\x1b[48;", out)  # no background color
        spaced = ui.render_markdown("before:\n\n```\nx\n```\n\nafter")
        self.assertNotIn("\n\n\n", spaced)

    def test_search_snippets_fit_one_line(self):
        self.assertEqual(wrencode._snippet("  first line\nsecond"), "first line…")
        self.assertEqual(wrencode._snippet("\n\nonly\n"), "only")
        self.assertEqual(wrencode._snippet("a" * 20, width=8), "aaaaaaaa…")

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
            wrencode.print_tool_action("write", {"path": "/tmp/x.txt", "content": ""})
        plain = strip_ansi(buf.getvalue()).lower()
        self.assertIn("write", plain)
        self.assertNotIn("write write", plain)

    def test_print_system_is_plain_text(self):
        import io

        buf = io.StringIO()
        with (
            mock.patch("sys.stdout", buf),
            mock.patch.object(ui, "colors_enabled", return_value=True),
        ):
            ui.print_system("Cleared")
        out = buf.getvalue()
        self.assertNotIn("\033[", out)  # notices read as replies, not banners
        self.assertEqual(out.strip(), "Cleared")

    def test_context_loader_frame(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-4-20250514"),
        ):
            plain = ui.loader_display(
                0, ui.loader_context(backends.BACKEND, backends.MODEL)
            )
            self.assertIn("thinking", strip_ansi(plain))
            self.assertIn("claude-sonnet", strip_ansi(plain))
            self.assertIn("claude-sonnet", strip_ansi(plain))

    def test_loader_display_gradient(self):
        with mock.patch.object(ui, "colors_enabled", return_value=True):
            out = ui.loader_display(0, "anthropic · claude · waiting…")
        self.assertIn("\033[96m", out)
        self.assertIn("\033[2m", out)
        self.assertIn("waiting", strip_ansi(out))

    def test_format_input_line_slash_is_bold_cyan(self):
        with mock.patch.object(ui, "colors_enabled", return_value=True):
            slash = ui.format_input_line("/backend")
            slash_args = ui.format_input_line("/backend fgggg")
            chat = ui.format_input_line("hello")
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

    def test_format_input_line_unknown_slash_is_plain(self):
        with mock.patch.object(ui, "colors_enabled", return_value=True):
            path = ui.format_input_line("/usr/bin is missing")
            prefix = ui.format_input_line("/mo")
            alias = ui.format_input_line("/q")
        self.assertNotIn("\033[1m", path)
        self.assertIn("\033[1m", prefix)
        self.assertIn("\033[1m", alias)

    def test_slash_matches_prefix(self):
        self.assertEqual(ui.slash_matches("/mo"), ["/model"])
        self.assertEqual(ui.slash_matches("/c"), ["/configure", "/compact", "/clear"])
        self.assertEqual(ui.slash_matches("/model gpt"), [])
        self.assertEqual(ui.slash_matches("hello"), [])
        self.assertIn("/help", ui.slash_matches("/"))

    def test_slash_aliases_dispatch(self):
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
        with mock.patch.object(wrencode, "save_history"):
            action, _ = wrencode.handle_slash_command("/c", msgs, None)
            self.assertEqual((action, msgs), ("handled", []))
            for cmd in ("/q", "/quit", "/exit"):
                action, _ = wrencode.handle_slash_command(cmd, msgs, None)
                self.assertEqual(action, "quit")

    def test_read_user_input_fallback(self):
        import io

        with (
            mock.patch("sys.stdin", io.StringIO("hello\n")),
            mock.patch.object(ui, "colors_enabled", return_value=False),
        ):
            self.assertEqual(ui.read_user_input(), "hello")


class TestConfirm(unittest.TestCase):
    def setUp(self):
        self._orig_auto = os.environ.get("WRENCODE_AUTO_APPROVE")
        self._orig_session = ui.SESSION_AUTO_APPROVE
        ui.SESSION_AUTO_APPROVE = False
        if "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]

    def tearDown(self):
        ui.SESSION_AUTO_APPROVE = self._orig_session
        if self._orig_auto is not None:
            os.environ["WRENCODE_AUTO_APPROVE"] = self._orig_auto
        elif "WRENCODE_AUTO_APPROVE" in os.environ:
            del os.environ["WRENCODE_AUTO_APPROVE"]

    def test_confirm_no_duplicate_write_prompt(self):
        import io

        buf = io.StringIO()
        with (
            mock.patch("sys.stdout", buf),
            mock.patch("builtins.input", return_value=""),
        ):
            ui.confirm("write")
        plain = strip_ansi(buf.getvalue())
        self.assertNotIn("PosixPath", plain)
        self.assertNotIn("⚠ write", plain)
        self.assertIn("Enter yes", plain)

    def test_enter_approves(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(ui.confirm("write"), "ok")

    def test_allow_all_sets_session_flag(self):
        with mock.patch("builtins.input", return_value="a"):
            self.assertEqual(ui.confirm("write"), "ok")
        self.assertTrue(ui.SESSION_AUTO_APPROVE)

    def test_decline_collects_feedback(self):
        with (
            mock.patch("builtins.input", return_value="n"),
            mock.patch.object(
                ui, "read_feedback_line", return_value="use edit instead"
            ),
        ):
            result = ui.confirm("write")
        self.assertEqual(result, "cancelled: user declined — use edit instead")

    def test_decline_without_feedback(self):
        with (
            mock.patch("builtins.input", return_value="n"),
            mock.patch.object(ui, "read_feedback_line", return_value=""),
        ):
            result = ui.confirm("write")
        self.assertEqual(result, "cancelled: user declined without instructions")

    def test_env_auto_approve(self):
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        self.assertEqual(ui.confirm("run"), "ok")

    def test_ctrl_c_returns_interrupted(self):
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertEqual(
                ui.confirm("write"),
                "cancelled: user interrupted",
            )


class TestChooseBackendInteractive(unittest.TestCase):
    def setUp(self):
        self._orig_config_file = backends.CONFIG_FILE
        self._orig_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        self._tmp = tempfile.mkdtemp()
        backends.CONFIG_FILE = pathlib.Path(self._tmp) / "config.json"

    def tearDown(self):
        backends.CONFIG_FILE = self._orig_config_file
        if self._orig_anthropic_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._orig_anthropic_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reconfigure_preserves_saved_api_key(self):
        backends.save_config(
            {
                "backend": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "api_key": "sk-secret",
            }
        )
        with (
            mock.patch.object(ui, "pick_from_list", return_value=0),
            mock.patch.object(
                configure,
                "pick_model_interactive",
                return_value="claude-haiku-4-5-20251001",
            ),
            mock.patch("getpass.getpass", return_value=""),
            mock.patch("builtins.input", return_value=""),
            mock.patch.object(configure, "verify_api_key", return_value=("ok", "")),
        ):
            configure.choose_backend_interactive()
        saved = json.loads(backends.CONFIG_FILE.read_text())
        self.assertEqual(saved["api_key"], "sk-secret")
        self.assertEqual(backends.API_KEY, "sk-secret")

    def test_reconfigure_saves_anthropic_workspace_id(self):
        backends.save_config(
            {
                "backend": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "api_key": "sk-secret",
            }
        )
        with (
            mock.patch.object(ui, "pick_from_list", return_value=0),
            mock.patch.object(
                configure,
                "pick_model_interactive",
                return_value="claude-haiku-4-5-20251001",
            ),
            mock.patch("getpass.getpass", return_value=""),
            mock.patch("builtins.input", return_value="wrkspc_test123"),
            mock.patch.object(configure, "verify_api_key", return_value=("ok", "")),
        ):
            configure.choose_backend_interactive()
        saved = json.loads(backends.CONFIG_FILE.read_text())
        self.assertEqual(saved["anthropic_workspace_id"], "wrkspc_test123")
        self.assertEqual(backends.ANTHROPIC_WORKSPACE_ID, "wrkspc_test123")

    def _configure_anthropic(self, typed_key):
        with (
            mock.patch.object(ui, "pick_from_list", return_value=0),
            mock.patch.object(
                configure, "pick_model_interactive", return_value="claude-x"
            ),
            mock.patch("getpass.getpass", return_value=typed_key),
            mock.patch("builtins.input", return_value=""),
            mock.patch.object(configure, "verify_api_key", return_value=("ok", "")),
        ):
            configure.choose_backend_interactive()
        return json.loads(backends.CONFIG_FILE.read_text())

    def test_reconfigure_replaces_env_key(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-stale-from-dotenv"
        saved = self._configure_anthropic("sk-fresh")
        self.assertEqual(saved["api_key"], "sk-fresh")
        self.assertEqual(saved["api_key_overrides_env"], "1")
        self.assertEqual(backends.API_KEY, "sk-fresh")
        # Next launch: .env sets the stale key again, the saved key still wins.
        os.environ["ANTHROPIC_API_KEY"] = "sk-stale-from-dotenv"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BACKEND", None)
            configure.resolve_configuration()
        self.assertEqual(backends.API_KEY, "sk-fresh")

    def test_reconfigure_blank_keeps_env_key(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-from-env"
        saved = self._configure_anthropic("")
        self.assertFalse(saved.get("api_key_overrides_env"))
        self.assertEqual(backends.API_KEY, "sk-from-env")


class TestAnthropicPromptCaching(unittest.TestCase):
    def test_request_caches_tools_system_and_history(self):
        captured: dict[str, Any] = {}

        def fake_post(url, body, headers):
            captured["body"] = body
            return {"content": [{"type": "text", "text": "ok"}], "usage": {}}

        orig = (backends.BACKEND, backends.MODEL, backends.API_KEY)
        try:
            backends.apply_backend("anthropic", model="claude-x", api_key="sk-t")
            with mock.patch.object(backends, "_http_post", fake_post):
                backends.get_response(
                    [{"role": "user", "content": "hi"}],
                    "sys-prompt",
                    None,
                    wrencode.tool_specs(),
                )
        finally:
            backends.BACKEND, backends.MODEL, backends.API_KEY = orig
        body = captured["body"]
        ephemeral = {"type": "ephemeral"}
        self.assertEqual(body["cache_control"], ephemeral)  # growing history
        self.assertEqual(body["system"][-1]["cache_control"], ephemeral)
        self.assertEqual(body["tools"][-1]["cache_control"], ephemeral)
        self.assertEqual(body["max_tokens"], backends.CLAUDE_MAX_TOKENS)

    def test_effort_is_sent_only_when_set(self):
        self.assertEqual(backends._claude_output_config(), {})
        with mock.patch.object(backends, "CLAUDE_EFFORT", "high"):
            self.assertEqual(
                backends._claude_output_config(), {"output_config": {"effort": "high"}}
            )


try:
    import claude_agent_sdk
except ImportError:  # optional extra; message tests need its real types
    claude_agent_sdk = None


class TestAgentSDKBackend(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp())
        self._patches = [
            mock.patch.object(backends, "CONFIG_DIR", self._tmp),
            mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": str(self._tmp)}),
        ]
        for p in self._patches:
            p.start()
        self._saved = (
            backends.BACKEND,
            backends.MODEL,
            backends.API_KEY,
            backends.ANTHROPIC_WORKSPACE_ID,
        )
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "MODEL"):
            os.environ.pop(var, None)

    def tearDown(self):
        import shutil

        (
            backends.BACKEND,
            backends.MODEL,
            backends.API_KEY,
            backends.ANTHROPIC_WORKSPACE_ID,
        ) = self._saved
        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_apply_backend_uses_anthropic_key_and_workspace(self):
        backends.apply_backend(
            "claude-agent-sdk", api_key="sk-t", anthropic_workspace_id="wrkspc_1"
        )
        self.assertEqual(backends.API_KEY, "sk-t")
        self.assertEqual(backends.ANTHROPIC_WORKSPACE_ID, "wrkspc_1")
        self.assertEqual(backends.MODEL, "claude-opus-5-5")

    def test_env_forces_api_key_billing(self):
        backends.apply_backend(
            "claude-agent-sdk", api_key="sk-t", anthropic_workspace_id="wrkspc_1"
        )
        with mock.patch.dict(os.environ, {"ANTHROPIC_CUSTOM_HEADERS": "x-a: 1"}):
            env = agent_sdk._agent_sdk_env()
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-t")
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "")
        self.assertEqual(
            env["ANTHROPIC_CUSTOM_HEADERS"], "x-a: 1\nanthropic-workspace-id: wrkspc_1"
        )

    def test_env_omits_workspace_header_when_unset(self):
        backends.apply_backend("claude-agent-sdk", api_key="sk-t")
        self.assertNotIn("ANTHROPIC_CUSTOM_HEADERS", agent_sdk._agent_sdk_env())

    def test_tool_action_summaries(self):
        fmt = agent_sdk.format_sdk_tool_action
        self.assertEqual(fmt("Bash", {"command": "ls -la"}), "$ ls -la")
        self.assertEqual(fmt("Read", {"file_path": "a.py"}), "Read a.py")
        self.assertEqual(fmt("Grep", {"pattern": "TODO"}), "Grep TODO")
        edit = fmt("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"})
        self.assertEqual(edit, "Edit a.py\n- x\n+ y")
        self.assertTrue(fmt("Odd", {"k": 1}).startswith("Odd("))

    def test_hidden_from_frozen_binary(self):
        with mock.patch.object(configure, "is_frozen", return_value=True):
            self.assertNotIn("claude-agent-sdk", configure.available_backends())
        with mock.patch.object(configure, "is_frozen", return_value=False):
            self.assertIn("claude-agent-sdk", configure.available_backends())

    def test_session_id_saved_per_workspace(self):
        agent_sdk._save_agent_sdk_session_id(wrencode.workspace_root(), "sess-1")
        self.assertEqual(
            agent_sdk._load_agent_sdk_session_id(wrencode.workspace_root()), "sess-1"
        )
        agent_sdk._save_agent_sdk_session_id(wrencode.workspace_root(), "")
        self.assertEqual(
            agent_sdk._load_agent_sdk_session_id(wrencode.workspace_root()), ""
        )

    def test_clear_resets_sdk_session(self):
        session = mock.MagicMock()
        with (
            mock.patch.object(backends, "BACKEND", "claude-agent-sdk"),
            mock.patch.object(wrencode, "agent_sdk_session", return_value=session),
            mock.patch.object(wrencode, "save_history"),
        ):
            action, _ = wrencode.handle_slash_command("/clear", [], None)
            wrencode.handle_slash_command("/compact", [], None)
        self.assertEqual(action, "handled")
        session.reset.assert_called_once()
        session.run.assert_called_once_with("/compact")

    def test_headless_routes_to_sdk_and_reports_cost(self):
        fake = mock.MagicMock()
        fake.run.return_value = agent_sdk.AgentSDKTurn(text="Done.", cost_usd=0.0123)
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(configure, "resolve_configuration", lambda: None),
            mock.patch.object(backends, "load_model", lambda: None),
            mock.patch.object(backends, "BACKEND", "claude-agent-sdk"),
            mock.patch.object(agent_sdk, "AgentSDKSession", return_value=fake),
            mock.patch.object(ui, "HEADLESS", False),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", err),
        ):
            code = wrencode.run_headless("do it", "json")
        data = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(data["result"], "Done.")
        self.assertEqual(data["cost_usd"], 0.0123)
        fake.run.assert_called_once_with("do it")
        fake.close.assert_called_once()

    @staticmethod
    def _sdk():
        assert claude_agent_sdk is not None  # the tests below skip without it
        return claude_agent_sdk

    def _session(self):
        with mock.patch("asyncio.new_event_loop"):
            return agent_sdk.AgentSDKSession(cwd=wrencode.workspace_root())

    @unittest.skipIf(claude_agent_sdk is None, "claude-agent-sdk not installed")
    def test_result_message_reports_turn_cost(self):
        sdk = self._sdk()
        session = self._session()
        turn = agent_sdk.AgentSDKTurn()

        def result(total, sid="s1"):
            return sdk.ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id=sid,
                total_cost_usd=total,
                result="ok",
            )

        with mock.patch("sys.stdout", io.StringIO()) as out:
            self.assertTrue(session._handle_message(result(0.05), turn))
            self.assertTrue(session._handle_message(result(0.08), turn))
        self.assertAlmostEqual(turn.cost_usd, 0.03)
        self.assertIn("$0.0300 this turn", out.getvalue())
        self.assertEqual(
            agent_sdk._load_agent_sdk_session_id(wrencode.workspace_root()), "s1"
        )

    @unittest.skipIf(claude_agent_sdk is None, "claude-agent-sdk not installed")
    def test_workspace_error_gets_a_hint(self):
        sdk = self._sdk()
        session = self._session()
        turn = agent_sdk.AgentSDKTurn()
        msg = sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="s1",
            result="API Error: 400 must include the anthropic-workspace-id header",
        )
        with mock.patch("sys.stdout", io.StringIO()) as out:
            session._handle_message(msg, turn)
        self.assertTrue(turn.is_error)
        self.assertIn("/configure", out.getvalue())

    @unittest.skipIf(claude_agent_sdk is None, "claude-agent-sdk not installed")
    def test_warns_when_not_billed_to_api_key(self):
        sdk = self._sdk()
        session = self._session()
        msg = sdk.SystemMessage(
            subtype="init", data={"apiKeySource": "claude.ai", "session_id": "s2"}
        )
        with mock.patch("sys.stdout", io.StringIO()) as out:
            session._handle_message(msg, agent_sdk.AgentSDKTurn())
        self.assertIn("may not bill", out.getvalue())

    @unittest.skipIf(claude_agent_sdk is None, "claude-agent-sdk not installed")
    def test_assistant_text_and_tool_calls_print(self):
        sdk = self._sdk()
        session = self._session()
        turn = agent_sdk.AgentSDKTurn()
        msg = sdk.AssistantMessage(
            content=[
                sdk.TextBlock(text="Looking."),
                sdk.ToolUseBlock(id="t1", name="Bash", input={"command": "ls"}),
            ],
            model="claude-opus-5-5",
        )
        with mock.patch("sys.stdout", io.StringIO()) as out:
            self.assertFalse(session._handle_message(msg, turn))
        self.assertEqual(turn.text, "Looking.")
        self.assertIn("$ ls", out.getvalue())


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
        backends.apply_backend("anthropic", model="claude-haiku-4-5-20251001")
        with mock.patch.object(
            configure,
            "fetch_anthropic_models",
            return_value=["claude-haiku-4-5-20251001", "claude-sonnet-4-5"],
        ):
            models = configure.list_models_for_backend("anthropic")
        self.assertIn("claude-haiku-4-5-20251001", models)
        self.assertIn("claude-sonnet-4-5", models)
        self.assertIn(configure.CUSTOM_MODEL_OPTION, models)

    def test_list_models_fetches_from_anthropic(self):
        backends.apply_backend("anthropic", model="claude-haiku-4-5-20251001")
        with mock.patch.object(
            configure,
            "fetch_anthropic_models",
            return_value=["claude-opus-4-6", "claude-haiku-4-5-20251001"],
        ) as fetch:
            models = configure.list_models_for_backend("anthropic")
        fetch.assert_called_once()
        self.assertEqual(models[0], "claude-opus-4-6")
        self.assertIn(configure.CUSTOM_MODEL_OPTION, models)

    def test_anthropic_headers_include_workspace_id(self):
        orig_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        orig_ws = os.environ.pop("ANTHROPIC_WORKSPACE_ID", None)
        try:
            backends.apply_backend(
                "anthropic",
                model="claude-haiku-4-5-20251001",
                api_key="sk-test",
                anthropic_workspace_id="wrkspc_abc",
            )
            headers = backends._anthropic_headers()
            self.assertEqual(headers["anthropic-workspace-id"], "wrkspc_abc")
            self.assertEqual(headers["x-api-key"], "sk-test")
        finally:
            if orig_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig_key
            if orig_ws is not None:
                os.environ["ANTHROPIC_WORKSPACE_ID"] = orig_ws

    def test_anthropic_headers_omit_workspace_when_unset(self):
        orig_ws = os.environ.pop("ANTHROPIC_WORKSPACE_ID", None)
        try:
            backends.apply_backend(
                "anthropic", model="claude-haiku-4-5-20251001", api_key="sk-test"
            )
            headers = backends._anthropic_headers()
            self.assertNotIn("anthropic-workspace-id", headers)
        finally:
            if orig_ws is not None:
                os.environ["ANTHROPIC_WORKSPACE_ID"] = orig_ws

    def test_fetch_anthropic_models_parses_provider_response(self):
        payload = {
            "data": [
                {"id": "claude-opus-4-6", "type": "model"},
                {"id": "claude-haiku-4-5-20251001", "type": "model"},
            ],
            "has_more": False,
            "last_id": "claude-haiku-4-5-20251001",
        }
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
        cm.__exit__.return_value = False
        self._tmp = tempfile.mkdtemp()
        orig_cache = backends.ANTHROPIC_MODELS_CACHE
        orig_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        orig_ws = os.environ.pop("ANTHROPIC_WORKSPACE_ID", None)
        backends.ANTHROPIC_MODELS_CACHE = (
            pathlib.Path(self._tmp) / "anthropic_models.json"
        )
        backends.apply_backend(
            "anthropic", model="x", api_key="sk-test", anthropic_workspace_id="wrkspc_1"
        )
        try:
            with mock.patch("urllib.request.urlopen", return_value=cm) as urlopen:
                ids = configure.fetch_anthropic_models()
            self.assertEqual(ids, ["claude-opus-4-6", "claude-haiku-4-5-20251001"])
            req = urlopen.call_args[0][0]
            hdrs = {k.lower(): v for k, v in req.headers.items()}
            self.assertEqual(hdrs.get("x-api-key"), "sk-test")
            self.assertEqual(hdrs.get("anthropic-workspace-id"), "wrkspc_1")
        finally:
            backends.ANTHROPIC_MODELS_CACHE = orig_cache
            if orig_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig_key
            if orig_ws is not None:
                os.environ["ANTHROPIC_WORKSPACE_ID"] = orig_ws
            import shutil

            shutil.rmtree(self._tmp, ignore_errors=True)

    def test_pick_from_list_numbered(self):
        with mock.patch("builtins.input", return_value="2"):
            idx = ui.pick_from_list("Pick", ["a", "b", "c"], labels=["A", "B", "C"])
        self.assertEqual(idx, 1)

    def test_switch_model_direct_id(self):
        backends.apply_backend("anthropic", model="claude-haiku-4-5-20251001")
        self._orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        self._orig_config = backends.CONFIG_FILE
        self._tmp = tempfile.mkdtemp()
        backends.CONFIG_FILE = pathlib.Path(self._tmp) / "config.json"
        try:
            result = configure.switch_model_runtime("claude-sonnet-4-20250514")
            self.assertIsNone(result)
            self.assertEqual(backends.MODEL, "claude-sonnet-4-20250514")
        finally:
            backends.CONFIG_FILE = self._orig_config
            if self._orig is not None:
                os.environ["ANTHROPIC_API_KEY"] = self._orig
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            import shutil

            shutil.rmtree(self._tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Highlight Code
# ---------------------------------------------------------------------------
class TestHighlightCode(unittest.TestCase):
    def test_keyword_is_blue(self):
        result = ui._highlight_code("def foo():")
        self.assertIn(BLUE, result)
        self.assertIn("def", result)

    def test_string_is_green(self):
        result = ui._highlight_code('x = "hello"')
        self.assertIn(GREEN, result)

    def test_comment_is_dim(self):
        result = ui._highlight_code("x = 1  # comment")
        self.assertIn(DIM, result)
        self.assertIn("comment", result)

    def test_number_is_yellow(self):
        result = ui._highlight_code("return 42")
        self.assertIn(YELLOW, result)

    def test_multiple_tokens(self):
        result = ui._highlight_code('if x == "ok": return True')
        self.assertIn(BLUE, result)  # 'if', 'return', 'True' → blue
        self.assertIn(GREEN, result)  # "ok" → green

    def test_no_tokens_unchanged(self):
        code = "x y z"
        result = ui._highlight_code(code)
        self.assertIn("x y z", strip_ansi(result))


# ---------------------------------------------------------------------------
# Strip GPT OSS tokens
# ---------------------------------------------------------------------------
class TestStripGptossTokens(unittest.TestCase):
    def test_strips_channel_prefix(self):
        text = "<|channel|>final<|message|>actual content"
        self.assertEqual(backends.strip_gptoss_tokens(text), "actual content")

    def test_strips_generic_tokens(self):
        result = backends.strip_gptoss_tokens("<|start|>hello<|end|>")
        self.assertEqual(result, "hello")

    def test_no_tokens_unchanged(self):
        self.assertEqual(backends.strip_gptoss_tokens("hello world"), "hello world")

    def test_strips_and_strips_whitespace(self):
        result = backends.strip_gptoss_tokens("  <|tok|>  hello  ")
        self.assertEqual(result, "hello")

    def test_multiple_channel_segments_takes_last(self):
        text = "<|channel|>final<|message|>first<|channel|>final<|message|>second"
        # split on the token takes the last segment
        self.assertEqual(backends.strip_gptoss_tokens(text), "second")


# ---------------------------------------------------------------------------
# Truncate at turn leak
# ---------------------------------------------------------------------------
class TestTruncateAtTurnLeak(unittest.TestCase):
    def test_truncates_at_user_marker(self):
        result = backends.truncate_at_turn_leak("Hello\nUser: leaked text")
        self.assertEqual(result, "Hello")

    def test_truncates_at_system_marker(self):
        result = backends.truncate_at_turn_leak("Hello\nSystem: leaked")
        self.assertEqual(result, "Hello")

    def test_truncates_at_human_marker(self):
        result = backends.truncate_at_turn_leak("Answer\nHuman: next turn")
        self.assertEqual(result, "Answer")

    def test_double_newline_user_marker(self):
        result = backends.truncate_at_turn_leak("Hello\n\nUser: next")
        self.assertEqual(result, "Hello")

    def test_double_newline_system_marker(self):
        result = backends.truncate_at_turn_leak("Hello\n\nSystem: next")
        self.assertEqual(result, "Hello")

    def test_no_leak_returns_unchanged(self):
        self.assertEqual(backends.truncate_at_turn_leak("No leak here"), "No leak here")

    def test_empty_string(self):
        self.assertEqual(backends.truncate_at_turn_leak(""), "")


# ---------------------------------------------------------------------------
# Complete Tool Call
# ---------------------------------------------------------------------------
class TestToolCallComplete(unittest.TestCase):
    def test_complete_with_end_tag(self):
        text = '<tool_call>{"tool": "read", "args": {"path": "foo.py"}}</tool_call>'
        end = backends._tool_call_complete(text)
        # Should return the position right after </tool_call>
        self.assertEqual(end, len(text))

    def test_complete_without_end_tag_uses_brace_matching(self):
        text = '<tool_call>{"tool": "read", "args": {"path": "foo.py"}}'
        end = backends._tool_call_complete(text)
        # Should return the position after the closing brace
        self.assertGreater(end, 0)
        # The character just before end should be '}'
        self.assertEqual(text[end - 1], "}")

    def test_no_tool_call_returns_minus_one(self):
        self.assertEqual(backends._tool_call_complete("no tool call"), -1)

    def test_open_tag_no_brace_returns_minus_one(self):
        self.assertEqual(backends._tool_call_complete("<tool_call>"), -1)

    def test_incomplete_json_returns_minus_one(self):
        # JSON opened but not closed
        self.assertEqual(backends._tool_call_complete("<tool_call>{incomplete"), -1)

    def test_nested_braces(self):
        # args value contains a JSON object itself
        text = (
            '<tool_call>{"tool": "write", "args": {"path": "f", "content": "{a: 1}"}}'
        )
        end = backends._tool_call_complete(text)
        self.assertGreater(end, 0)
        self.assertEqual(text[end - 1], "}")


# ---------------------------------------------------------------------------
# Parse tool calls
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
# Apply backend
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
        backends.apply_backend("anthropic")
        self.assertEqual(backends.BACKEND, "anthropic")
        self.assertEqual(backends.MODEL, backends.BACKEND_SPECS["anthropic"]["model"])
        self.assertEqual(
            backends.API_BASE, backends.BACKEND_SPECS["anthropic"]["api_base"]
        )

    def test_anthropic_model_override(self):
        backends.apply_backend("anthropic", model="claude-opus-4-5")
        self.assertEqual(backends.MODEL, "claude-opus-4-5")

    def test_anthropic_env_model_beats_explicit(self):
        os.environ["MODEL"] = "claude-env-model"
        backends.apply_backend("anthropic", model="claude-explicit")
        self.assertEqual(backends.MODEL, "claude-env-model")

    def test_anthropic_env_api_key_beats_explicit(self):
        os.environ["ANTHROPIC_API_KEY"] = "env-key-abc"
        backends.apply_backend("anthropic", api_key="explicit-key")
        self.assertEqual(backends.API_KEY, "env-key-abc")

    def test_ollama_defaults(self):
        backends.apply_backend("ollama")
        self.assertEqual(backends.BACKEND, "ollama")
        self.assertEqual(backends.MODEL, backends.BACKEND_SPECS["ollama"]["model"])
        # Ollama uses a dummy key so the auth field is non-empty
        self.assertEqual(backends.API_KEY, "ollama")

    def test_ollama_api_base_uses_localhost(self):
        os.environ.pop("OLLAMA_HOST", None)
        backends.apply_backend("ollama")
        self.assertIn("localhost:11434", backends.API_BASE)

    def test_ollama_respects_ollama_host_env(self):
        os.environ["OLLAMA_HOST"] = "http://192.168.1.5:11434"
        backends.apply_backend("ollama")
        self.assertIn("192.168.1.5:11434", backends.API_BASE)
        del os.environ["OLLAMA_HOST"]

    def test_sets_backend_global(self):
        backends.apply_backend("anthropic")
        self.assertEqual(backends.BACKEND, "anthropic")
        backends.apply_backend("ollama")
        self.assertEqual(backends.BACKEND, "ollama")


# ---------------------------------------------------------------------------
# Resolve configuration
# ---------------------------------------------------------------------------
class TestResolveConfiguration(unittest.TestCase):
    def setUp(self):
        self._saved_backend = os.environ.pop("BACKEND", None)
        self._orig_config_file = backends.CONFIG_FILE
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        if self._saved_backend is not None:
            os.environ["BACKEND"] = self._saved_backend
        elif "BACKEND" in os.environ:
            del os.environ["BACKEND"]
        backends.CONFIG_FILE = self._orig_config_file
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_backend_env_var_wins(self):
        os.environ["BACKEND"] = "anthropic"
        configure.resolve_configuration()
        self.assertEqual(backends.BACKEND, "anthropic")

    def test_backend_env_var_unknown_raises(self):
        os.environ["BACKEND"] = "totally_unknown_backend"
        with self.assertRaises(SystemExit) as cm:
            configure.resolve_configuration()
        self.assertEqual(cm.exception.code, 1)

    def test_saved_config_is_used(self):
        # Point CONFIG_FILE at a tmpdir with a known config
        config_path = pathlib.Path(self._tmp) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"backend": "ollama"}, f)
        backends.CONFIG_FILE = config_path
        configure.resolve_configuration()
        self.assertEqual(backends.BACKEND, "ollama")

    def test_saved_config_with_model(self):
        config_path = pathlib.Path(self._tmp) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"backend": "ollama", "model": "mistral"}, f)
        backends.CONFIG_FILE = config_path
        configure.resolve_configuration()
        # MODEL should be mistral unless overridden by env
        saved_model_env = os.environ.pop("MODEL", None)
        try:
            configure.resolve_configuration()
            self.assertEqual(backends.MODEL, "mistral")
        finally:
            if saved_model_env is not None:
                os.environ["MODEL"] = saved_model_env

    def test_saved_api_backend_without_key_reprompts_on_tty(self):
        # Saved config names a hosted backend but has no api_key, and the
        # backend's key env var is unset — in a terminal we should re-run the
        # chooser instead of dead-ending in load_model() (the "exits
        # immediately" bug when run outside a dir whose .env supplied the key).
        config_path = pathlib.Path(self._tmp) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"backend": "anthropic", "model": "claude-x"}, f)
        backends.CONFIG_FILE = config_path
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        called = []
        try:
            with (
                mock.patch.object(sys.stdin, "isatty", return_value=True),
                mock.patch.object(
                    configure,
                    "choose_backend_interactive",
                    side_effect=lambda: called.append(True),
                ),
            ):
                configure.resolve_configuration()
            self.assertEqual(called, [True])
        finally:
            if saved_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved_key

    def test_saved_api_backend_without_key_non_tty_applies(self):
        # Same missing-key config but without a terminal: don't re-prompt,
        # apply the backend so load_model() can surface the actionable error.
        config_path = pathlib.Path(self._tmp) / "config.json"
        with open(config_path, "w") as f:
            json.dump({"backend": "anthropic", "model": "claude-x"}, f)
        backends.CONFIG_FILE = config_path
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        called = []
        try:
            with (
                mock.patch.object(sys.stdin, "isatty", return_value=False),
                mock.patch.object(
                    configure,
                    "choose_backend_interactive",
                    side_effect=lambda: called.append(True),
                ),
            ):
                configure.resolve_configuration()
            self.assertEqual(called, [])
            self.assertEqual(backends.BACKEND, "anthropic")
        finally:
            if saved_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved_key

    def test_non_interactive_no_config_raises_system_exit(self):
        # Point to an empty tmpdir (no config.json) and make stdin non-tty
        backends.CONFIG_FILE = pathlib.Path(self._tmp) / "config.json"
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=False),
            self.assertRaises(SystemExit) as cm,
        ):
            configure.resolve_configuration()
        self.assertEqual(cm.exception.code, 1)


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

    def test_write_errors_when_content_missing(self):
        # Guards against the MAX_TOKENS-truncation failure mode where the
        # model emits a tool_use block with `path` but no `content`. Returning
        # an error lets _track_error halt the loop after a few retries.
        result = wrencode.write({"path": "hello.txt"})
        self.assertTrue(result.startswith("error:"))
        self.assertFalse(pathlib.Path(self._tmp, "hello.txt").exists())

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

    def test_edit_preserves_indentation(self):
        # `old` must match exactly: a stripped match anchor combined with an
        # indented `new` would duplicate the leading whitespace (8 -> 16 spaces).
        wrencode.write(
            {"path": "f.py", "content": "def f(x):\n    if x:\n        return x\n"}
        )
        result = wrencode.edit(
            {"path": "f.py", "old": "        return x", "new": "        return -x"}
        )
        self.assertEqual(result, "ok")
        read_back = strip_ansi(wrencode.read({"path": "f.py"}))
        self.assertIn("        return -x", read_back)
        self.assertNotIn("                return", read_back)

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
        result = wrencode.edit(
            {"path": "f.txt", "old": "foo", "new": "bar", "all": True}
        )
        self.assertEqual(result, "ok")
        read_back = strip_ansi(wrencode.read({"path": "f.txt"}))
        self.assertNotIn("foo", read_back)
        self.assertEqual(read_back.count("bar"), 3)

    def test_edit_no_op_is_rejected(self):
        wrencode.write({"path": "f.txt", "content": "unchanged"})
        result = wrencode.edit(
            {"path": "f.txt", "old": "unchanged", "new": "unchanged"}
        )
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
        self._orig_backend = backends.BACKEND
        backends.apply_backend("ollama")

    def _mock_get_response(self, fn):
        patcher = mock.patch.object(backends, "get_response", fn)
        patcher.start()
        self.addCleanup(patcher.stop)

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
        backends.BACKEND = self._orig_backend

    def test_one_tool_call_then_final_answer(self):
        """Agent reads a file (one tool call) then gives a final answer."""
        # Write a file for the agent to read
        target = pathlib.Path(self._tmp) / "greeting.txt"
        target.write_text("Hello from wrencode!", encoding="utf-8")

        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return '<tool_call>{"tool": "read", "args": {"path": "greeting.txt"}}</tool_call>'
            return "The file says: Hello from wrencode!"

        self._mock_get_response(mock_get_response)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Read greeting.txt for me"}
        ]
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
            for block in (m["content"] if isinstance(m["content"], list) else [])
            if block.get("type") == "text"
        ]
        self.assertTrue(
            any("Hello from wrencode" in t for t in assistant_texts),
            f"Expected final answer in assistant messages, got: {assistant_texts}",
        )

    def test_no_tool_calls_terminates_immediately(self):
        """Agent with no tool calls in response terminates after one round."""
        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            call_count[0] += 1
            return "Just a plain answer, no tools needed."

        self._mock_get_response(mock_get_response)

        messages = [{"role": "user", "content": "Say hello"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=5)

        self.assertEqual(call_count[0], 1)

    def _task_reply(self, *prompts):
        return "".join(
            "<tool_call>"
            + json.dumps({"tool": "task", "args": {"prompt": p}})
            + "</tool_call>"
            for p in prompts
        )

    def test_task_tool_runs_subagents(self):
        """Two task calls in one reply each run a subagent; results reach the parent."""
        answers = {"count a": "a is 1", "count b": "b is 2"}

        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            first = backends.flatten_content(messages[0]["content"])
            if first in answers:
                return answers[first]
            if len(messages) == 1:
                return self._task_reply("count a", "count b")
            return "Both counted."

        self._mock_get_response(mock_get_response)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "count both"}]
        with mock.patch("sys.stdout", io.StringIO()):
            reason = wrencode.run_agent_turn(messages, "sys", None, max_iters=5)
        self.assertEqual(reason, "done")
        results = messages[2]["content"]
        # Results keep the order of the calls, whichever subagent finished first.
        self.assertIn("a is 1", results[0]["content"])
        self.assertIn("b is 2", results[1]["content"])
        self.assertEqual(
            backends.flatten_content(messages[-1]["content"]), "Both counted."
        )

    def test_subagents_run_at_the_same_time(self):
        """Three subagents must all be waiting at once to pass the barrier."""
        import threading

        barrier = threading.Barrier(3, timeout=5)

        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            first = backends.flatten_content(messages[0]["content"])
            if first.startswith("part"):
                barrier.wait()  # raises BrokenBarrierError if they ran in turn
                return f"{first} done"
            if len(messages) == 1:
                return self._task_reply("part 1", "part 2", "part 3")
            return "All parts done."

        self._mock_get_response(mock_get_response)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            reason = wrencode.run_agent_turn(messages, "sys", None, max_iters=5)
        self.assertEqual(reason, "done")
        results = [b["content"] for b in messages[2]["content"]]
        self.assertEqual(results, ["part 1 done", "part 2 done", "part 3 done"])
        text = strip_ansi(out.getvalue())
        self.assertIn("running 3 subagents in parallel", text)
        for tag in ("[1]", "[2]", "[3]"):
            self.assertIn(tag, text)

    def test_local_weights_run_subagents_in_order(self):
        calls = [backends.ToolCall(str(i), "task", {"prompt": "x"}) for i in range(2)]
        with mock.patch.object(backends, "BACKEND", "mlx"):
            self.assertEqual(wrencode._run_parallel_tasks(calls), {})
        with mock.patch.object(wrencode, "MAX_PARALLEL_SUBAGENTS", 1):
            self.assertEqual(wrencode._run_parallel_tasks(calls), {})

    def test_cancel_stops_the_batch(self):
        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            first = backends.flatten_content(messages[0]["content"])
            if first.startswith("slow"):
                ui._CANCEL_REQUESTED.set()  # as if Escape was pressed
                return "partial"
            return self._task_reply("slow 1", "slow 2")

        self._mock_get_response(mock_get_response)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        with mock.patch("sys.stdout", io.StringIO()):
            reason = wrencode.run_agent_turn(messages, "sys", None, max_iters=5)
        self.assertEqual(reason, "cancelled")
        self.assertFalse(ui._CANCEL_REQUESTED.is_set())

    def test_subagent_depth_is_per_thread(self):
        import threading

        seen: list[int] = []

        def probe():
            seen.append(ui._subagent_depth())

        ui._AGENT_LOCAL.depth = 1
        try:
            t = threading.Thread(target=probe)
            t.start()
            t.join()
        finally:
            ui._AGENT_LOCAL.depth = 0
        self.assertEqual(seen, [0])

    def test_approval_holds_other_agents_output(self):
        import threading

        real = io.StringIO()
        out = ui._AgentStdout(real)
        ui._AGENT_LOCAL.tag = "1"
        try:
            out.hold()  # agent 1 is at an approval prompt
            other = threading.Thread(
                target=lambda: (
                    setattr(ui._AGENT_LOCAL, "tag", "2"),
                    out.write("from two\n"),
                )
            )
            other.start()
            other.join()
            out.write("Approve? ")
            self.assertEqual(real.getvalue(), "Approve? ")  # agent 2 is held
            out.release()
        finally:
            del ui._AGENT_LOCAL.tag
        self.assertIn("from two", strip_ansi(real.getvalue()))
        self.assertIn("[2] from two", strip_ansi(real.getvalue()))

    def test_tool_call_dispatches_real_tool(self):
        """Verify that run_tool dispatches to the real read implementation."""
        target = pathlib.Path(self._tmp) / "real.txt"
        target.write_text("real content", encoding="utf-8")

        result = wrencode.run_tool("read", {"path": "real.txt"})
        self.assertIn("real content", result)

    def test_max_iters_cap(self):
        """Agent stops after max_iters if it keeps returning tool calls."""
        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            call_count[0] += 1
            return '<tool_call>{"tool": "glob", "args": {"pat": "*.txt"}}</tool_call>'

        self._mock_get_response(mock_get_response)

        messages = [{"role": "user", "content": "loop forever"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=3)

        self.assertEqual(call_count[0], 3)

    def test_repeated_identical_tool_error_stops_loop(self):
        call_count = [0]

        def mock_get_response(messages, system_prompt, mlx_state, tools=None):
            call_count[0] += 1
            return '<tool_call>{"tool": "read", "args": {"path": "missing.txt"}}</tool_call>'

        self._mock_get_response(mock_get_response)

        messages = [{"role": "user", "content": "keep retrying"}]
        wrencode.run_agent_turn(messages, "You are helpful.", None, max_iters=20)

        self.assertEqual(call_count[0], wrencode.TOOL_ERROR_REPEAT_LIMIT)


# ---------------------------------------------------------------------------
# Additional helpers / edge cases
# ---------------------------------------------------------------------------
class TestFlattenContent(unittest.TestCase):
    def test_plain_string_unchanged(self):
        self.assertEqual(backends.flatten_content("hello"), "hello")

    def test_none_returns_empty_string(self):
        self.assertEqual(backends.flatten_content(None), "")

    def test_text_block_extracted(self):
        content = [{"type": "text", "text": "hello world"}]
        self.assertEqual(backends.flatten_content(content), "hello world")

    def test_tool_use_block_serialized(self):
        content = [{"type": "tool_use", "name": "read", "input": {"path": "f.py"}}]
        result = backends.flatten_content(content)
        self.assertIn("<tool_call>", result)
        self.assertIn("read", result)

    def test_tool_result_block_included(self):
        content = [{"type": "tool_result", "content": "file contents"}]
        result = backends.flatten_content(content)
        self.assertIn("file contents", result)

    def test_multiple_blocks_joined(self):
        content = [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]
        result = backends.flatten_content(content)
        self.assertIn("part1", result)
        self.assertIn("part2", result)


class TestTruncationWarning(unittest.TestCase):
    """Cover _warn_if_truncated — the safety net for max_tokens-truncated tool calls."""

    def _capture_stderr(self, fn, *args, **kwargs):
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    def test_anthropic_warns_on_max_tokens_stop_reason(self):
        data = {
            "stop_reason": "max_tokens",
            "usage": {"output_tokens": 4096},
            "content": [],
        }
        with mock.patch.object(backends, "BACKEND", "anthropic"):
            out = self._capture_stderr(backends._warn_if_truncated, data)
        self.assertIn("truncated", out.lower())
        self.assertIn("4096", out)

    def test_anthropic_silent_on_tool_use_stop_reason(self):
        data = {
            "stop_reason": "tool_use",
            "usage": {"output_tokens": 500},
            "content": [],
        }
        with mock.patch.object(backends, "BACKEND", "anthropic"):
            out = self._capture_stderr(backends._warn_if_truncated, data)
        self.assertEqual(out, "")

    def test_openai_warns_on_length_finish_reason(self):
        data = {
            "choices": [{"finish_reason": "length", "message": {}}],
            "usage": {"completion_tokens": 4096},
        }
        with mock.patch.object(backends, "BACKEND", "openai"):
            out = self._capture_stderr(backends._warn_if_truncated, data)
        self.assertIn("truncated", out.lower())


class TestAnthropicPromptCache(unittest.TestCase):
    """Confirm the anthropic request wires cache_control onto system + last tool."""

    def test_system_and_tools_are_marked_ephemeral(self):
        captured = {}

        def fake_post(url, payload, headers):
            captured["payload"] = payload
            # minimal valid response shape
            return {
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-4-6"),
            mock.patch.object(backends, "API_KEY", "test-key"),
            mock.patch.object(
                backends, "API_BASE", "https://api.anthropic.com/v1/messages"
            ),
            mock.patch.object(backends, "_http_post", fake_post),
        ):
            backends.get_response(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="you are a test",
                mlx_state=None,
                tools=wrencode.tool_specs(),
            )

        payload = captured["payload"]
        # system is now a list with a cache breakpoint
        self.assertIsInstance(payload["system"], list)
        self.assertEqual(payload["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(payload["system"][0]["text"], "you are a test")
        # the last tool entry carries a cache breakpoint
        self.assertIn("cache_control", payload["tools"][-1])
        self.assertEqual(payload["tools"][-1]["cache_control"], {"type": "ephemeral"})
        # earlier tools do not
        if len(payload["tools"]) > 1:
            self.assertNotIn("cache_control", payload["tools"][0])


class TestCertifiTrustStore(unittest.TestCase):
    """The startup block points SSL at certifi so the bundled binary can do TLS."""

    def test_ssl_cert_file_points_at_existing_bundle(self):
        # certifi is a build dep of the binary; if it's importable, the startup
        # block must set SSL_CERT_FILE to a real CA bundle file. Import in a
        # fresh process with the variable unset: the running test process may
        # have inherited one (corporate or proxy CA bundles), which the startup
        # block rightly leaves alone.
        try:
            import certifi
        except ImportError:
            self.skipTest("certifi not installed in this environment")
        env = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, wrencode; print(os.environ.get('SSL_CERT_FILE', ''))",
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        cert_file = out.stdout.strip()
        self.assertEqual(cert_file, certifi.where())
        self.assertTrue(os.path.exists(cert_file))

    def test_existing_override_is_preserved(self):
        # setdefault must not clobber a user-provided SSL_CERT_FILE.
        with mock.patch.dict(os.environ, {"SSL_CERT_FILE": "/custom/path.pem"}):
            os.environ.setdefault("SSL_CERT_FILE", "/should/not/win.pem")
            self.assertEqual(os.environ["SSL_CERT_FILE"], "/custom/path.pem")


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


# ---------------------------------------------------------------------------
# AWS Bedrock backend (SigV4 signing, request shape, credential resolution)
# ---------------------------------------------------------------------------
class TestBedrockSigV4(unittest.TestCase):
    # Fixed inputs; the expected signature was cross-validated against
    # botocore's SigV4Auth (the reference AWS SDK implementation) at runtime.
    AK = "AKIDEXAMPLE"
    SK = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
    AMZ = "20150830T123600Z"

    def test_sigv4_signature_matches_known_vector(self):
        import hashlib
        import urllib.parse

        body = b'{"max_tokens":16}'
        host = "bedrock-runtime.us-east-1.amazonaws.com"
        wire = (
            "/model/"
            + urllib.parse.quote("us.anthropic.claude-3-5-haiku-20241022-v1:0", safe="")
            + "/invoke"
        )
        canonical_uri = urllib.parse.quote(wire, safe="/~")
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": self.AMZ,
        }
        auth, signed = backends._sigv4_authorization(
            "POST",
            canonical_uri,
            "",
            headers,
            payload_hash,
            "bedrock-runtime",
            "us-east-1",
            self.AMZ,
            self.AK,
            self.SK,
        )
        self.assertEqual(signed, "host;x-amz-content-sha256;x-amz-date")
        self.assertTrue(
            auth.endswith(
                "Signature=3ba9edc434e4059fc725fb046d32d62d8c482fdc79ad7e91543530225bc1764d"
            )
        )
        self.assertIn(
            f"Credential={self.AK}/20150830/us-east-1/bedrock-runtime/aws4_request",
            auth,
        )

    def test_canonical_uri_double_encodes_colon(self):
        # Non-S3 SigV4 double-encodes the path: a model id's ':' -> %253A.
        import urllib.parse

        wire = (
            "/model/"
            + urllib.parse.quote("us.anthropic.claude-3-5-haiku-20241022-v1:0", safe="")
            + "/invoke"
        )
        canonical_uri = urllib.parse.quote(wire, safe="/~")
        self.assertIn("%253A", canonical_uri)
        self.assertNotIn("%3A0", canonical_uri)  # the single-encoded form is gone

    def test_signed_headers_include_session_token_when_present(self):
        with mock.patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": self.AK,
                "AWS_SECRET_ACCESS_KEY": self.SK,
                "AWS_SESSION_TOKEN": "tok123",
                "AWS_REGION": "us-east-1",
            },
        ):
            headers = backends._sigv4_signed_headers(
                "POST",
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
                b"{}",
                "bedrock-runtime",
                "us-east-1",
            )
        self.assertEqual(headers["X-Amz-Security-Token"], "tok123")
        self.assertIn("x-amz-security-token", headers["Authorization"])

    def test_missing_credentials_raises(self):
        # Mock the resolver empty — clearing env alone won't do it, since there's
        # a ~/.aws/credentials fallback that may exist on the dev machine.
        with (
            mock.patch.object(backends, "_aws_credentials", return_value=("", "", "")),
            self.assertRaises(RuntimeError),
        ):
            backends._sigv4_signed_headers(
                "POST", "https://x/y", b"{}", "bedrock-runtime", "us-east-1"
            )


class TestBedrockBackend(unittest.TestCase):
    def test_spec_is_aws_kind_with_no_key_env(self):
        spec = backends.BACKEND_SPECS["bedrock"]
        self.assertEqual(spec["kind"], "aws")
        self.assertNotIn("key_env", spec)

    def test_bedrock_is_native_and_hosted_but_not_anthropic_format(self):
        # Bedrock uses the Converse format, so it must NOT be parsed as Anthropic.
        self.assertNotIn("bedrock", backends.ANTHROPIC_FORMAT_BACKENDS)
        self.assertIn("bedrock", backends.NATIVE_TOOL_BACKENDS)
        self.assertIn("bedrock", backends.HOSTED_BACKENDS)
        self.assertNotIn("bedrock", backends.API_BACKENDS)

    def test_apply_backend_aws_sets_region_and_no_key(self):
        with mock.patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}, clear=False):
            os.environ.pop("MODEL", None)
            backends.apply_backend("bedrock")
        self.assertEqual(backends.BACKEND, "bedrock")
        self.assertEqual(backends.AWS_REGION, "eu-west-1")
        self.assertEqual(backends.API_KEY, "")

    def test_get_response_builds_converse_request(self):
        # get_response should hit /converse with a Converse-shaped body:
        # system as [{text}], inferenceConfig.maxTokens, toolConfig.tools[].toolSpec,
        # string message content normalized to [{text}], and no anthropic_version.
        captured = {}

        def fake_post_raw(url, data, headers):
            captured["url"] = url
            captured["body"] = json.loads(data)
            captured["headers"] = headers
            return {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "ok"}]}
                },
                "stopReason": "end_turn",
            }

        with mock.patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "AKIDEXAMPLE",
                "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                "AWS_REGION": "us-east-1",
            },
            clear=False,
        ):
            os.environ.pop("MODEL", None)
            backends.apply_backend("bedrock", model="openai.gpt-oss-120b-1:0")
            with mock.patch.object(backends, "_http_post_raw", fake_post_raw):
                raw = backends.get_response(
                    [{"role": "user", "content": "hi"}],
                    "sys-prompt",
                    None,
                    wrencode.tool_specs(),
                )
        body = captured["body"]
        self.assertTrue(captured["url"].endswith("/converse"))
        self.assertIn("bedrock-runtime.us-east-1.amazonaws.com", captured["url"])
        self.assertNotIn("anthropic_version", body)
        self.assertEqual(body["system"], [{"text": "sys-prompt"}])
        self.assertEqual(body["inferenceConfig"]["maxTokens"], backends.MAX_TOKENS)
        self.assertEqual(
            body["messages"][0]["content"], [{"text": "hi"}]
        )  # str -> [{text}]
        self.assertIn("toolSpec", body["toolConfig"]["tools"][0])
        self.assertIn("Authorization", captured["headers"])
        # The host is bedrock-runtime.*, but the SigV4 credential scope must name
        # the signing service "bedrock" — AWS 403s on "bedrock-runtime" here.
        self.assertIn(
            "/us-east-1/bedrock/aws4_request", captured["headers"]["Authorization"]
        )
        self.assertNotIn(
            "/bedrock-runtime/aws4_request", captured["headers"]["Authorization"]
        )
        # The returned raw JSON parses back to text via the native path.
        self.assertEqual(
            json.loads(raw)["output"]["message"]["content"][0]["text"], "ok"
        )

    def test_bedrock_response_parse_and_append_roundtrip(self):
        # Converse response -> parsed text + ToolCall, and the history append uses
        # Converse blocks (assistant content + toolResult) so the next turn is valid.
        with mock.patch.object(backends, "BACKEND", "bedrock"):
            data = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "let me read"},
                            {
                                "toolUse": {
                                    "toolUseId": "tu1",
                                    "name": "read",
                                    "input": {"path": "x"},
                                }
                            },
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 5, "outputTokens": 7},
            }
            text, calls = backends._parse_native_response(data)
            self.assertEqual(text, "let me read")
            self.assertEqual(calls[0].id, "tu1")
            self.assertEqual(calls[0].name, "read")

            msgs = []
            backends._append_assistant(msgs, text, calls, data)
            self.assertEqual(msgs[0]["role"], "assistant")
            self.assertIn("toolUse", msgs[0]["content"][1])

            backends._append_tool_results(msgs, [(calls[0], "file contents")])
            tr = msgs[1]["content"][0]["toolResult"]
            self.assertEqual(tr["toolUseId"], "tu1")
            self.assertEqual(tr["content"], [{"text": "file contents"}])
            self.assertEqual(tr["status"], "success")


class TestSynthesize(unittest.TestCase):
    """The /synthesize transcript-fusion pipeline."""

    def _jsonl(self, *objs):
        return "\n".join(json.dumps(o) for o in objs)

    def test_normalize_claude_code_keeps_only_user_and_assistant_text(self):
        raw = self._jsonl(
            {"type": "mode", "mode": "default"},  # non-message line ignored
            {"type": "user", "message": {"role": "user", "content": "fix the bug"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "hmm"},  # dropped
                        {"type": "text", "text": "Found it in foo.py"},
                        {"type": "tool_use", "name": "read", "input": {}},  # dropped
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "bytes"}],
                },
            },  # dropped (list)
            {
                "type": "user",
                "message": {"role": "user", "content": "<system-reminder>x"},
            },  # skipped
        )
        turns = synthesize._parse_claude_code_jsonl(raw)
        self.assertEqual(
            turns,
            [
                {"role": "user", "text": "fix the bug"},
                {"role": "assistant", "text": "Found it in foo.py"},
            ],
        )

    def test_normalize_rejects_non_jsonl(self):
        self.assertIsNone(synthesize._parse_claude_code_jsonl("# just markdown\nhello"))

    def test_generic_jsonl_adapter_sniffs_role_and_content(self):
        # Codex/OpenAI-style: each line a flat {role, content} record, varied keys.
        raw = self._jsonl(
            {"role": "user", "content": "build X"},
            {"sender": "ai", "text": "done, edited y.py"},
            {"role": "system", "content": "ignored"},  # non-user/assistant dropped
        )
        self.assertEqual(
            synthesize._adapt_generic_jsonl(raw),
            [
                {"role": "user", "text": "build X"},
                {"role": "assistant", "text": "done, edited y.py"},
            ],
        )

    def test_messages_json_adapter_handles_array_and_wrapper(self):
        arr = json.dumps(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"text": "yo"}]},
            ]
        )
        self.assertEqual(
            synthesize._adapt_messages_json(arr),
            [
                {"role": "user", "text": "hi"},
                {"role": "assistant", "text": "yo"},
            ],
        )
        wrapped = json.dumps({"messages": [{"role": "user", "content": "hey"}]})
        self.assertEqual(
            synthesize._adapt_messages_json(wrapped), [{"role": "user", "text": "hey"}]
        )
        self.assertIsNone(synthesize._adapt_messages_json('{"no": "messages"}'))

    def test_coerce_text_flattens_blocks(self):
        self.assertEqual(
            synthesize._coerce_text([{"text": "a"}, {"content": "b"}]), "a\nb"
        )
        self.assertEqual(synthesize._coerce_text("plain"), "plain")
        self.assertEqual(synthesize._coerce_text({"text": "nested"}), "nested")

    def test_normalize_reports_adapter_source(self):
        cc = self._jsonl({"type": "user", "message": {"role": "user", "content": "hi"}})
        generic = self._jsonl({"role": "user", "content": "hi"})
        with tempfile.TemporaryDirectory() as d:
            for name, raw, want in [
                ("a.jsonl", cc, "claude-code"),
                ("b.jsonl", generic, "jsonl"),
            ]:
                p = pathlib.Path(d) / name
                p.write_text(raw)
                self.assertEqual(synthesize._synth_normalize(str(p))["source"], want)

    def test_reconcile_dispatches_mode_to_system_prompt(self):
        seen = {}

        def fake(system, user, prefill="", mlx_state=None):
            seen["sys"] = system
            return "doc"

        with mock.patch.object(synthesize, "_synth_complete", side_effect=fake):
            synthesize._synth_reconcile([{"chat": "a"}], mode="diff")
            self.assertIs(seen["sys"], synthesize.SYNTH_DIFF_SYS)
            synthesize._synth_reconcile([{"chat": "a"}], mode="log")
            self.assertIs(seen["sys"], synthesize.SYNTH_LOG_SYS)
            synthesize._synth_reconcile([{"chat": "a"}])
            self.assertIs(seen["sys"], synthesize.SYNTH_RECONCILE_SYS)

    def test_normalize_falls_back_to_text_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "notes.md"
            p.write_text("# design\nplain prose")
            chat = synthesize._synth_normalize(str(p))
            self.assertEqual(chat["source"], "text")
            self.assertEqual(chat["turns"][0]["role"], "user")

    def test_json_slice_tolerates_surrounding_prose(self):
        self.assertEqual(synthesize._json_slice('here: {"a": 1} done'), '{"a": 1}')
        self.assertEqual(synthesize._json_slice("no json"), "no json")

    def test_parse_selection_handles_numbers_ranges_and_all(self):
        self.assertEqual(synthesize._parse_selection("all", 3), [0, 1, 2])
        self.assertEqual(synthesize._parse_selection("1 3", 3), [0, 2])
        self.assertEqual(synthesize._parse_selection("1-3", 5), [0, 1, 2])
        self.assertEqual(synthesize._parse_selection("2,2,9", 3), [1])  # dedup + clamp
        self.assertEqual(synthesize._parse_selection("nope", 3), [])

    def test_extract_forces_json_and_tags_provenance(self):
        chat = {"id": "abcd1234", "turns": [{"role": "user", "text": "hi"}]}
        payload = (
            '"decisions": ["use Converse"], "problems_solved": [], '
            '"files_touched": ["wrencode.py"], "open_questions": []}'
        )
        with mock.patch.object(
            synthesize, "_synth_complete", return_value="{" + payload
        ) as m:
            facts = synthesize._synth_extract(chat)
        # extraction is forced via assistant prefill "{"
        self.assertEqual(m.call_args.kwargs.get("prefill"), "{")
        self.assertEqual(facts["chat"], "abcd1234")
        self.assertEqual(facts["decisions"], ["use Converse"])

    def test_extract_survives_unparseable_output(self):
        chat = {"id": "ffff", "turns": [{"role": "user", "text": "hi"}]}
        with mock.patch.object(
            synthesize,
            "_synth_complete",
            return_value="Let me continue the chat instead...",
        ):
            facts = synthesize._synth_extract(chat)
        self.assertEqual(facts["chat"], "ffff")
        self.assertEqual(facts["decisions"], [])
        self.assertIn("_parse_error", facts)

    def test_run_synthesize_writes_out_file(self):
        with tempfile.TemporaryDirectory() as d:
            a = pathlib.Path(d) / "a.jsonl"
            b = pathlib.Path(d) / "b.jsonl"
            a.write_text(
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": "task A"}}
                )
            )
            b.write_text(
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": "task B"}}
                )
            )
            out = pathlib.Path(d) / "synthesis.md"
            extract_json = (
                '{"decisions": ["d"], "problems_solved": [], '
                '"files_touched": [], "open_questions": []}'
            )
            # one extract call per file, then one reconcile call
            with mock.patch.object(
                synthesize,
                "_synth_complete",
                side_effect=[extract_json, extract_json, "# SYNTHESIS\nmerged"],
            ):
                synthesize.run_synthesize([str(a), str(b)], out=str(out))
            self.assertEqual(out.read_text(), "# SYNTHESIS\nmerged")

    def test_run_synthesize_errors_when_no_transcripts(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            synthesize.run_synthesize([str(pathlib.Path(d) / "missing.jsonl")])

    def test_run_synthesize_rejects_unknown_mode(self):
        with self.assertRaises(SystemExit):
            synthesize.run_synthesize([], mode="rebase")

    def test_chat_ids_are_unique_per_run(self):
        # uuid-like names keep their 8-char prefix; clashing prefixes fall back to
        # the whole name; identical names (different dirs) get a suffix.
        ids = synthesize._chat_ids(
            [
                "/x/0123abcd-rest.jsonl",
                "/x/session-2026-10-01.jsonl",
                "/x/session-2026-10-02.jsonl",
                "/a/notes.md",
                "/b/notes.md",
            ]
        )
        self.assertEqual(
            ids,
            [
                "0123abcd",
                "session-2026-10-01",
                "session-2026-10-02",
                "notes",
                "notes-2",
            ],
        )
        self.assertEqual(len(set(ids)), len(ids))

    def test_run_synthesize_cites_unique_ids(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name in ("session-01.jsonl", "session-02.jsonl"):
                p = pathlib.Path(d) / name
                p.write_text(json.dumps({"role": "user", "content": "task"}))
                paths.append(str(p))
            seen = []

            def fake(system, user, prefill="", mlx_state=None):
                seen.append(user)
                return '{"decisions": []}' if prefill else "doc"

            with mock.patch.object(synthesize, "_synth_complete", side_effect=fake):
                synthesize.run_synthesize(paths, out=str(pathlib.Path(d) / "o.md"))
        self.assertIn("chat_id=session-01>", seen[0])
        self.assertIn("chat_id=session-02>", seen[1])

    def test_render_keeps_start_and_mostly_end_when_over_budget(self):
        chat = {
            "turns": [
                {"role": "user", "text": "START " + "a" * 100},
                {"role": "assistant", "text": "b" * 400},
                {"role": "user", "text": "c" * 100 + " LATEST"},
            ]
        }
        out = synthesize._synth_render(chat, max_chars=200)
        self.assertTrue(out.startswith("[USER] START"))
        self.assertTrue(out.endswith("LATEST"))
        self.assertIn("characters omitted", out)
        self.assertLess(len(out), 260)
        # under budget: untouched
        full = synthesize._synth_render(chat, max_chars=10_000)
        self.assertNotIn("omitted", full)

    def test_render_budget_follows_context_window(self):
        with mock.patch.object(backends, "CONTEXT_TOKENS", 1000):
            self.assertEqual(synthesize._transcript_budget(), 2400)

    def test_extract_normalizes_fact_shapes(self):
        chat = {"id": "abcd1234", "turns": [{"role": "user", "text": "hi"}]}
        payload = '{"decisions": "not a list", "files_touched": ["a.py"], "extra": 1}'
        with mock.patch.object(synthesize, "_synth_complete", return_value=payload):
            facts = synthesize._synth_extract(chat)
        self.assertEqual(facts["decisions"], [])
        self.assertEqual(facts["files_touched"], ["a.py"])
        self.assertEqual(facts["open_questions"], [])
        self.assertNotIn("extra", facts)
        self.assertNotIn("_parse_error", facts)

    def test_synth_complete_is_deterministic_and_passes_prefill(self):
        with mock.patch.object(backends, "complete", return_value="{}") as m:
            synthesize._synth_complete("sys", "user", prefill="{", mlx_state=("m", "t"))
        self.assertEqual(m.call_args.args, ("sys", "user"))
        self.assertEqual(m.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(m.call_args.kwargs["prefill"], "{")
        self.assertEqual(m.call_args.kwargs["mlx_state"], ("m", "t"))


try:
    import pydantic_monty
except ImportError:  # the sandbox tests below skip without it
    pydantic_monty = None


class TestSandboxUnavailable(unittest.TestCase):
    def test_tool_reports_the_missing_dependency(self):
        with mock.patch.object(sandbox, "pydantic_monty", None):
            self.assertFalse(sandbox.available())
            out = sandbox.run("1", workspace=pathlib.Path("."), functions={})
        self.assertTrue(out.startswith("error: the python tool needs pydantic-monty"))

    def test_result_formatting(self):
        self.assertEqual(sandbox._result("hi\n", value=2), "hi\n2")
        self.assertEqual(sandbox._result("", value="s"), "'s'")
        self.assertEqual(sandbox._result("", value=None), "(no output)")
        err = sandbox._result(
            "partial\n", error="ZeroDivisionError: division by zero\n"
        )
        self.assertTrue(err.startswith("error: ZeroDivisionError"))
        self.assertIn("partial", err)

    def test_python_tool_formatting_and_aliases(self):
        self.assertEqual(
            wrencode.normalize_tool_args("python", {"source": "x = 1"})["code"], "x = 1"
        )
        shown = wrencode.format_tool_action("python", {"code": "x = 1\nprint(x)"})
        self.assertEqual(shown, "python x = 1  (+1 lines)")
        self.assertEqual(
            wrencode.format_tool_action("python", {"code": "y"}), "python y"
        )


@unittest.skipIf(pydantic_monty is None, "pydantic-monty not installed")
class TestSandbox(unittest.TestCase):
    """The python tool's sandbox, run for real against pydantic-monty."""

    @classmethod
    def tearDownClass(cls):
        sandbox.close()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = pathlib.Path(self._tmp.name)
        (self.ws / "a.txt").write_text("alpha\nbeta\n")
        (self.ws / "pkg").mkdir()
        (self.ws / "pkg" / "b.py").write_text("B = 1\n")

    def tearDown(self):
        self._tmp.cleanup()

    def run_code(self, code, functions=None):
        return sandbox.run(code, workspace=self.ws, functions=functions or {})

    def test_prints_and_trailing_value(self):
        self.assertEqual(self.run_code("print('hi')\n1 + 1"), "hi\n2")
        self.assertEqual(self.run_code("x = 1"), "(no output)")
        self.assertEqual(self.run_code("[i * i for i in range(3)]"), "[0, 1, 4]")

    def test_workspace_is_mounted_read_only_and_is_the_cwd(self):
        self.assertEqual(
            self.run_code("open('/workspace/a.txt').read()"), "'alpha\\nbeta\\n'"
        )
        self.assertEqual(self.run_code("open('pkg/b.py').read().strip()"), "'B = 1'")
        out = self.run_code("open('/workspace/new.txt', 'w').write('x')")
        self.assertTrue(out.startswith("error:"))
        self.assertIn("PermissionError", out)
        self.assertFalse((self.ws / "new.txt").exists())

    def test_host_functions_are_callable_by_name(self):
        calls = []

        def read(path, offset=None, limit=None):
            calls.append((path, offset, limit))
            return f"contents of {path}"

        out = self.run_code("read('a.txt', limit=5)", functions={"read": read})
        self.assertEqual(out, "'contents of a.txt'")
        self.assertEqual(calls, [("a.txt", None, 5)])

    def test_errors_come_back_as_error_text_with_prior_output(self):
        out = self.run_code("print('before')\n1 / 0")
        self.assertTrue(out.startswith("error:"))
        self.assertIn("ZeroDivisionError", out)
        self.assertIn("before", out)
        self.assertTrue(self.run_code("def (:").startswith("error:"))

    def test_no_shell_network_or_environment(self):
        for code in (
            "import subprocess",
            "import urllib.request",
            "import os\nos.environ['HOME']",
        ):
            self.assertTrue(self.run_code(code).startswith("error:"), code)

    def test_time_limit(self):
        with mock.patch.object(sandbox, "TIMEOUT", 0.3):
            out = self.run_code("while True:\n    pass")
        self.assertTrue(out.startswith("error:"))
        self.assertIn("Timeout", out)


@unittest.skipIf(pydantic_monty is None, "pydantic-monty not installed")
class TestPythonTool(unittest.TestCase):
    """The python tool as the agent sees it."""

    @classmethod
    def tearDownClass(cls):
        sandbox.close()

    def test_registered_and_described(self):
        self.assertIn("python", wrencode.TOOLS)
        self.assertIn("python", [name for name, _, _ in wrencode.tool_specs()])
        self.assertIn("python(code)", wrencode.build_system_prompt())

    def test_runs_with_the_workspace_tools(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "f.txt").write_text("needle here\nsecond\n")
            pathlib.Path(d, "sub").mkdir()
            pathlib.Path(d, "sub", "g.txt").write_text("nothing\n")
            with mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": d}):
                out = wrencode.python(
                    {
                        "code": "paths = sorted(glob('**/*.txt'))\n"
                        "print(paths)\n"
                        "print(len(read('f.txt').splitlines()))\n"
                        "print(grep('needle'))\n"
                        "open('f.txt').read().startswith('needle')"
                    }
                )
                missing = wrencode.python({"code": "read('nope.txt')"})
        self.assertIn("['f.txt', 'sub/g.txt']", out)  # relative paths, a real list
        self.assertIn("\n2\n", out)  # raw text, not numbered lines
        self.assertIn("f.txt:1:needle here", out)
        self.assertTrue(out.endswith("True"))
        self.assertTrue(missing.startswith("error:"))
        self.assertIn("FileNotFoundError", missing)


class TestHardening(unittest.TestCase):
    """What an untrusted repository, file or model reply must not be able to do."""

    DOTENV = (
        'OPENAI_API_KEY="sk-from-repo"\n'
        "export ANTHROPIC_WORKSPACE_ID=wrkspc_1\n"
        "BACKEND=openai-compatible\n"
        "WRENCODE_AUTO_APPROVE=1\n"
        "OPENAI_COMPATIBLE_BASE_URL=https://evil.example/v1\n"
        "# comment\n"
    )
    NAMES = (
        "OPENAI_API_KEY",
        "ANTHROPIC_WORKSPACE_ID",
        "BACKEND",
        "WRENCODE_AUTO_APPROVE",
        "OPENAI_COMPATIBLE_BASE_URL",
    )

    def _clean_env(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in self.NAMES:
            os.environ.pop(name, None)

    def test_project_dotenv_sets_only_credentials(self):
        self._clean_env()
        with tempfile.TemporaryDirectory() as d:
            env = pathlib.Path(d, ".env")
            env.write_text(self.DOTENV)
            skipped = wrencode.load_dotenv(str(env), trusted=False)
        self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-from-repo")
        self.assertEqual(os.environ["ANTHROPIC_WORKSPACE_ID"], "wrkspc_1")
        for name in ("BACKEND", "WRENCODE_AUTO_APPROVE", "OPENAI_COMPATIBLE_BASE_URL"):
            self.assertNotIn(name, os.environ)
        self.assertEqual(
            sorted(skipped),
            ["BACKEND", "OPENAI_COMPATIBLE_BASE_URL", "WRENCODE_AUTO_APPROVE"],
        )

    def test_trusted_dotenv_sets_anything_but_the_real_environment_wins(self):
        self._clean_env()
        os.environ["BACKEND"] = "anthropic"
        with tempfile.TemporaryDirectory() as d:
            env = pathlib.Path(d, ".env")
            env.write_text(self.DOTENV)
            skipped = wrencode.load_dotenv(str(env), trusted=True)
        self.assertEqual(skipped, [])
        self.assertEqual(os.environ["BACKEND"], "anthropic")  # setdefault semantics
        self.assertEqual(os.environ["WRENCODE_AUTO_APPROVE"], "1")

    def test_missing_dotenv_is_fine(self):
        self.assertEqual(wrencode.load_dotenv("/nonexistent/.env", trusted=False), [])

    def test_visible_shows_control_characters(self):
        self.assertEqual(ui.visible("ls\x1b[2K\rrm -rf /"), "ls^[[2K^Mrm -rf /")
        self.assertEqual(ui.visible("a\tb\nc\x7f\x07"), "a\tb\nc^?^G")
        self.assertEqual(ui.visible("plain"), "plain")

    def test_approval_prompt_shows_exactly_what_would_run(self):
        with mock.patch("sys.stdout", io.StringIO()) as out:
            wrencode.print_tool_action("bash", {"cmd": "echo safe\r\x1b[2Krm -rf ~"})
        self.assertIn("echo safe^M^[[2Krm -rf ~", out.getvalue())
        self.assertNotIn("\r", out.getvalue())
        self.assertEqual(ui._AGENT_LOCAL.last_action, "$ echo safe^M^[[2Krm -rf ~")

    def test_tool_output_and_replies_are_sanitized(self):
        with mock.patch("sys.stdout", io.StringIO()) as out:
            wrencode.print_tool_result("x\x1b]0;evil\x07y")
            ui.print_agent_message("done\x1b[2J")
        self.assertIn("x^[]0;evil^Gy", out.getvalue())
        self.assertIn("done^[[2J", out.getvalue())

    def test_hidden_paths_are_flagged_in_approvals(self):
        flagged = wrencode.format_tool_action(
            "write", {"path": ".github/workflows/ci.yml", "content": "x"}
        )
        self.assertIn("⚠ hidden/config path", flagged)
        flagged = wrencode.format_tool_action(
            "edit", {"path": ".git/hooks/pre-commit", "old": "a", "new": "b"}
        )
        self.assertIn("⚠ hidden/config path", flagged)
        for path in ("src/app.py", "./src/app.py", "../sibling/x.py"):
            self.assertNotIn(
                "⚠", wrencode.format_tool_action("write", {"path": path, "content": ""})
            )

    @unittest.skipIf(shutil.which("rg") is None, "ripgrep not installed")
    def test_grep_never_parses_a_file_name_as_a_flag(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "--pre=sh").write_text("touch pwned\n")
            with mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": d}):
                out = wrencode.grep({"pat": "touch", "path": "--pre=sh"})
            self.assertFalse(out.startswith("error:"), out)
            self.assertIn("touch pwned", out)
            self.assertFalse(pathlib.Path(d, "pwned").exists())

    def test_history_file_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d, "h.json")
            p.write_text("[]")
            os.chmod(p, 0o644)
            with mock.patch.dict(os.environ, {"WRENCODE_HISTORY_FILE": str(p)}):
                wrencode.save_history([{"role": "user", "content": "x"}])
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
            self.assertEqual(json.loads(p.read_text())[0]["content"], "x")

    def test_git_status_runs_with_fsmonitor_disabled(self):
        with mock.patch.object(wrencode.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=1, stdout="")
            wrencode.git_context()
        self.assertIn("core.fsmonitor=false", run.call_args.args[0])


class TestRepoRules(unittest.TestCase):
    def test_no_noqa_markers(self):
        here = pathlib.Path(__file__).parent
        files = [*sorted(here.glob("wrencode*.py")), here / "test_wrencode.py"]
        offenders = [
            f"{f.name}:{n}"
            for f in files
            for n, line in enumerate(f.read_text().splitlines(), 1)
            if "noqa" in line and "test_no_noqa_markers" not in line
        ]
        self.assertEqual(offenders, [], "lint exceptions belong in pyproject.toml")


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
            self.assertEqual(
                wrencode.history_file_path(), pathlib.Path("/cfg/history.json")
            )


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
            mock.patch.object(wrencode, "_STORE", self.store),
            mock.patch.object(wrencode, "_SESSION_ID", 7),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_load_and_save_go_to_the_store(self):
        self.assertEqual(wrencode.load_history(), [{"role": "user", "content": "hi"}])
        wrencode.save_history([{"role": "user", "content": "x"}])
        self.store.save.assert_called_once_with(7, [{"role": "user", "content": "x"}])

    def test_clear_starts_a_new_session_and_keeps_the_old(self):
        msgs = [{"role": "user", "content": "old"}]
        action, _ = wrencode.handle_slash_command("/clear", msgs, None)
        self.assertEqual(action, "handled")
        self.assertEqual(msgs, [])
        self.store.new_session.assert_called_once()
        self.store.save.assert_not_called()
        self.assertIn("new session #7", sys.stdout.getvalue())

    def test_sessions_resume_and_search(self):
        wrencode.handle_slash_command("/sessions", [], None)
        self.assertIn("#7", sys.stdout.getvalue())
        msgs: list = []
        wrencode.handle_slash_command("/resume 7", msgs, None)
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])
        wrencode.handle_slash_command("/resume 99", [], None)
        self.assertIn("No session #99", sys.stdout.getvalue())
        wrencode.handle_slash_command("/search hi", [], None)
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
            wrencode.handle_slash_command("/resume", msgs, None)
        self.assertEqual(pick.call_args[0][1], ["7"])
        self.assertIn("#7", pick.call_args[1]["labels"][0])
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])
        with mock.patch("sys.stdin.isatty", return_value=False):
            wrencode.handle_slash_command("/resume", [], None)
        self.assertIn("Usage: /resume <id>", sys.stdout.getvalue())

    def test_sync_reports_the_mirror(self):
        self.store.mirror = None
        wrencode.handle_slash_command("/sync", [], None)
        self.assertIn("No mirror configured", sys.stdout.getvalue())
        self.store.mirror = mock.Mock(label="db.example.com/wren")
        self.store.sync.return_value = (3, 0)
        wrencode.handle_slash_command("/sync", [{"role": "user", "content": "x"}], None)
        self.assertIn(
            "Mirrored 3 sessions to db.example.com/wren", sys.stdout.getvalue()
        )
        self.store.save.assert_called()  # the current conversation is saved first
        self.store.sync.return_value = (3, 2)
        wrencode.handle_slash_command("/sync", [], None)
        self.assertIn("2 still pending", sys.stdout.getvalue())

    def test_without_the_store_the_commands_explain(self):
        with mock.patch.object(wrencode, "_STORE", None):
            wrencode.handle_slash_command("/sessions", [], None)
        self.assertIn("history.json", sys.stdout.getvalue())

    def test_a_failed_save_is_reported_not_raised(self):
        self.store.save.side_effect = RuntimeError("db gone")
        wrencode.save_history([])
        self.assertIn("Could not save history", sys.stdout.getvalue())


class TestUsage(unittest.TestCase):
    """Token usage from every backend format, and the line the person sees."""

    def setUp(self):
        self._orig = backends.USAGE
        backends.USAGE = backends.Usage()
        self.addCleanup(setattr, backends, "USAGE", self._orig)

    def test_anthropic_counts_cached_and_written_separately(self):
        with mock.patch.object(backends, "BACKEND", "anthropic"):
            backends._record_usage(
                {
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 169,
                        "cache_creation_input_tokens": 1472,
                        "cache_read_input_tokens": 0,
                    }
                }
            )
            backends._record_usage(
                {
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 217,
                        "cache_creation_input_tokens": 264,
                        "cache_read_input_tokens": 1472,
                    }
                }
            )
        u = backends.USAGE
        self.assertEqual(
            (u.turn_uncached, u.turn_cache_read, u.turn_cache_write), (6, 1472, 1736)
        )
        self.assertEqual(u.turn_out, 386)
        self.assertEqual(u.prompt, 2 + 264 + 1472)  # the latest request's prompt
        d = u.as_dict()
        self.assertEqual(d["input_tokens"], 6 + 1472 + 1736)
        self.assertEqual(d["model_calls"], 2)

    def test_openai_cached_tokens_are_inside_prompt_tokens(self):
        with mock.patch.object(backends, "BACKEND", "openai"):
            backends._record_usage(
                {
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {"cached_tokens": 900},
                    }
                }
            )
        u = backends.USAGE
        self.assertEqual(
            (u.turn_uncached, u.turn_cache_read, u.turn_out), (100, 900, 50)
        )
        self.assertEqual(u.prompt, 1000)

    def test_bedrock_and_missing_usage(self):
        with mock.patch.object(backends, "BACKEND", "bedrock"):
            backends._record_usage(
                {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "cacheReadInputTokens": 7,
                    }
                }
            )
            backends._record_usage({"output": {}})  # no usage block: ignored
        self.assertEqual(backends.USAGE.turn_calls, 1)
        self.assertEqual(backends.USAGE.prompt, 17)

    def test_usage_line_reads_well(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "CONTEXT_TOKENS", 128000),
        ):
            backends._record_usage(
                {
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 386,
                        "cache_creation_input_tokens": 264,
                        "cache_read_input_tokens": 1472,
                    }
                }
            )
            line = backends.usage_line()
            backends._record_usage({"usage": {"input_tokens": 100, "output_tokens": 1}})
            report = backends.usage_report()
            title = backends.usage_title()
        # no MODEL here, so no price: the line and title carry no dollar amount
        self.assertEqual(line, "↑ 1.9k  ↓ 386  ▱▱▱▱▱▱▱▱▱▱ 2%")
        self.assertEqual(
            report[0].split(), ["input", "cached", "written", "output", "calls", "cost"]
        )
        self.assertEqual(
            report[1].split(), ["this", "turn", "2.0k", "1.5k", "264", "387", "2", "$?"]
        )
        self.assertTrue(report[3].startswith("context: 100 of 128k (0%)"))
        self.assertTrue(report[4].startswith("price: unknown"))
        self.assertIn("WRENCODE_PRICE", report[4])
        self.assertEqual(title, "wrencode · ctx 0% · ↑2.0k ↓387")

    def test_usage_meter_fills_and_warns_at_the_compaction_threshold(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "CONTEXT_TOKENS", 1000),
            mock.patch.object(ui, "colors_enabled", return_value=True),
        ):
            backends._record_usage({"usage": {"input_tokens": 800, "output_tokens": 1}})
            warned = backends.usage_line(0.75)
            calm = backends.usage_line(0.9)
        self.assertIn("▰▰▰▰▰▰▰▰▱▱ 80%", strip_ansi(warned))
        self.assertIn(ui.YELLOW, warned)
        self.assertNotIn(ui.YELLOW, calm)

    def test_usage_command_prints_the_report(self):
        with mock.patch("sys.stdout", io.StringIO()):
            action, _ = wrencode.handle_slash_command("/usage", [], None)
            out = sys.stdout.getvalue()
        self.assertEqual(action, "handled")
        self.assertIn("this turn", out)
        self.assertIn("context:", out)

    def test_turn_resets_but_session_accumulates(self):
        with mock.patch.object(backends, "BACKEND", "anthropic"):
            backends._record_usage({"usage": {"input_tokens": 10, "output_tokens": 1}})
            backends.USAGE.begin_turn()
            backends._record_usage({"usage": {"input_tokens": 20, "output_tokens": 2}})
        u = backends.USAGE
        self.assertEqual((u.turn_uncached, u.session_uncached), (20, 30))
        self.assertEqual((u.turn_calls, u.session_calls), (1, 2))

    def test_get_response_records_the_openrouter_reply(self):
        with (
            mock.patch.object(backends, "BACKEND", "openrouter"),
            mock.patch.object(backends, "API_BASE", "https://x/y"),
            mock.patch.object(
                backends,
                "_http_post",
                return_value={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 3},
                },
            ),
        ):
            backends.get_response([{"role": "user", "content": "x"}], "sys", None)
        self.assertEqual(
            (backends.USAGE.turn_uncached, backends.USAGE.turn_out), (30, 3)
        )


class TestPricing(unittest.TestCase):
    """Prices per token: the table, the overrides, the cost math, and what's shown."""

    def setUp(self):
        self._orig = backends.USAGE
        backends.USAGE = backends.Usage()
        self.addCleanup(setattr, backends, "USAGE", self._orig)
        self._tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        patch = mock.patch.object(backends, "PRICES_FILE", self._tmp / "prices.json")
        patch.start()
        self.addCleanup(patch.stop)
        env = mock.patch.dict(os.environ, {"WRENCODE_PRICE": ""})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("WRENCODE_PRICE", None)

    def test_built_in_table_matches_every_provider_spelling(self):
        cases = {
            ("anthropic", "claude-sonnet-4-5-20250929"): "$3/$15 per MTok",
            ("anthropic", "claude-opus-5-5"): "$4/$20 per MTok",
            ("anthropic", "claude-opus-5"): "$5/$25 per MTok",
            ("anthropic", "claude-haiku-5-5"): "$0.10/$0.50 per MTok",
            ("openrouter", "anthropic/claude-3.5-sonnet"): "$3/$15 per MTok",
            (
                "bedrock",
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ): "$1/$5 per MTok",
            ("bedrock", "us.amazon.nova-pro-v1:0"): "$0.80/$3.20 per MTok",
            ("openai", "gpt-4o-mini"): "$0.15/$0.60 per MTok",
        }
        for (backend, model), label in cases.items():
            price = backends.price_for(model, backend)
            self.assertIsNotNone(price, model)
            self.assertEqual(price.label(), label, model)
        self.assertEqual(backends.price_for("claude-opus-5-5").source, "built-in")
        self.assertIsNone(backends.price_for("gpt-oss-20b", "local"))
        self.assertIsNone(backends.price_for("claude-opus-5-5", "mlx"))
        self.assertIsNone(backends.price_for("llama-3", "ollama"))

    def test_claude_cache_rates_follow_the_price_list(self):
        opus = backends.price_for("claude-opus-5-5", "anthropic")
        sonnet = backends.price_for("claude-sonnet-5-5", "anthropic")
        fable = backends.price_for("claude-fable-5-1", "anthropic")
        self.assertEqual((opus.cache_read, opus.cache_write), (0.2, 5.0))
        self.assertEqual((sonnet.cache_read, sonnet.cache_write), (0.2, 2.5))
        self.assertEqual((fable.cache_read, fable.cache_write), (0.25, 12.5))

    def test_env_override_then_saved_catalog_then_table(self):
        backends.PRICES_FILE.write_text(
            json.dumps({"openrouter/anthropic/claude-3.5-sonnet": [1, 2, 0.1, 1.25]})
        )
        catalog = backends.price_for("anthropic/claude-3.5-sonnet", "openrouter")
        self.assertEqual((catalog.input, catalog.output), (1.0, 2.0))
        self.assertEqual(catalog.source, "openrouter catalog")
        # the file is re-read when it changes
        backends.PRICES_FILE.write_text(
            json.dumps({"openrouter/anthropic/claude-3.5-sonnet": [7, 8]})
        )
        os.utime(backends.PRICES_FILE, (1, 1))
        again = backends.price_for("anthropic/claude-3.5-sonnet", "openrouter")
        self.assertEqual((again.input, again.output, again.cache_read), (7.0, 8.0, 7.0))
        with mock.patch.dict(os.environ, {"WRENCODE_PRICE": "0.5,1.5"}):
            env = backends.price_for("gpt-oss-20b", "local")
            self.assertEqual(env.label(), "$0.50/$1.50 per MTok")
            self.assertEqual(env.source, "WRENCODE_PRICE")
        with mock.patch.dict(os.environ, {"WRENCODE_PRICE": "junk"}):
            self.assertIsNone(backends.price_for("claude-opus-5-5", "anthropic"))
        with mock.patch.dict(os.environ, {"WRENCODE_PRICE": "1,2,3,4,5"}):
            self.assertIsNone(backends.price_for("claude-opus-5-5", "anthropic"))

    def test_call_cost_uses_each_rate_and_the_long_prompt_card(self):
        sonnet = backends.price_for("claude-sonnet-5-5", "anthropic")
        cost = backends.call_cost(sonnet, 200, 1472, 264, 386)
        expected = (200 * 2 + 1472 * 0.2 + 264 * 2.5 + 386 * 10) / 1e6
        self.assertAlmostEqual(cost, expected)
        haiku = backends.price_for("claude-haiku-5-5", "anthropic")
        short = backends.call_cost(haiku, 1000, 0, 0, 100)
        long = backends.call_cost(haiku, 150_000, 0, 0, 100)
        self.assertAlmostEqual(short, (1000 * 0.10 + 100 * 0.50) / 1e6)
        self.assertAlmostEqual(long, (150_000 * 0.50 + 100 * 2.50) / 1e6)
        self.assertIsNone(backends.call_cost(None, 1, 1, 1, 1))

    def test_money_keeps_small_amounts_visible(self):
        self.assertEqual(backends._money(0), "$0")
        self.assertEqual(backends._money(0.00004), "<$0.0001")
        self.assertEqual(backends._money(0.0052144), "$0.0052")
        self.assertEqual(backends._money(0.2134), "$0.213")
        self.assertEqual(backends._money(12.345), "$12.35")
        self.assertEqual(backends._money(1234.5), "$1,234")

    def test_spend_is_on_the_line_the_report_and_the_title(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
            mock.patch.object(backends, "CONTEXT_TOKENS", 128000),
        ):
            backends._record_usage(
                {
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 386,
                        "cache_creation_input_tokens": 264,
                        "cache_read_input_tokens": 1472,
                    }
                }
            )
            first = backends.usage_line()
            backends.USAGE.begin_turn()
            backends._record_usage(
                {"usage": {"input_tokens": 1000, "output_tokens": 100}}
            )
            second = backends.usage_line()
            report = backends.usage_report()
            title = backends.usage_title()
            as_dict = backends.USAGE.as_dict()
        self.assertTrue(first.endswith("▱▱▱▱▱▱▱▱▱▱ 2%  $0.0052"), first)
        self.assertTrue(second.endswith("1%  $0.0030 · total $0.0082"), second)
        self.assertEqual(report[1].split()[-1], "$0.0030")
        self.assertEqual(report[2].split()[-1], "$0.0082")
        self.assertEqual(
            report[4],
            "price: $2/$10 per MTok (cache read $0.20, write $2.50; built-in)",
        )
        self.assertEqual(title, "wrencode · ctx 1% · ↑2.9k ↓486 · $0.0082")
        self.assertAlmostEqual(as_dict["cost_usd"], 0.008214, places=6)

    def test_a_call_without_a_price_withholds_the_total(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
        ):
            backends._record_usage({"usage": {"input_tokens": 10, "output_tokens": 1}})
        with (
            mock.patch.object(backends, "BACKEND", "openai-compatible"),
            mock.patch.object(backends, "MODEL", "some-local-thing"),
        ):
            backends._record_usage(
                {"usage": {"prompt_tokens": 10, "completion_tokens": 1}}
            )
            line = backends.usage_line()
            report = backends.usage_report()
            title = backends.usage_title()
        self.assertNotIn("$", line)
        self.assertNotIn("$", title)
        self.assertEqual(report[2].split()[-1], "$?")
        self.assertNotIn("cost_usd", backends.USAGE.as_dict())

    def test_send_estimate_prices_the_next_request(self):
        with (
            mock.patch.object(backends, "MODEL", "x"),
            mock.patch.object(backends, "BACKEND", "local"),
        ):
            self.assertEqual(backends.send_estimate(100), "")
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
        ):
            # before the first call: the estimated context at the input rate
            first = backends.send_estimate(400, context_tokens=2000)
            self.assertEqual(
                first, f"≈ {backends._money((2000 + 101) * 2 / 1e6)} input"
            )
            backends._record_usage(
                {
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 300,
                        "cache_creation_input_tokens": 500,
                        "cache_read_input_tokens": 1500,
                    }
                }
            )
            # then: last prompt read from the cache, the reply and the text written to it
            hint = backends.send_estimate(40)
        cached = 2000 * 0.2
        fresh = (300 + 11) * 2.5
        self.assertEqual(hint, f"≈ {backends._money((cached + fresh) / 1e6)} input")

    def test_catalog_prices_are_saved_from_a_models_fetch(self):
        entries = [
            {
                "id": "anthropic/claude-sonnet-4.5",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "input_cache_read": "0.0000003",
                    "input_cache_write": "0.00000375",
                },
            },
            {"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "odd/model", "pricing": {"prompt": "n/a", "completion": "1"}},
            {"id": "bare/model"},
        ]
        self.assertEqual(configure._save_catalog_prices("openrouter", entries), 2)
        saved = json.loads(backends.PRICES_FILE.read_text())
        self.assertEqual(
            saved["openrouter/anthropic/claude-sonnet-4.5"], [3, 15, 0.3, 3.75]
        )
        self.assertEqual(saved["openrouter/free/model"], [0, 0, 0, 0])
        self.assertEqual(stat.S_IMODE(backends.PRICES_FILE.stat().st_mode), 0o600)
        # a later fetch merges, keeping what it did not list
        configure._save_catalog_prices(
            "nanogpt",
            [{"id": "m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}],
        )
        saved = json.loads(backends.PRICES_FILE.read_text())
        self.assertIn("openrouter/free/model", saved)
        self.assertEqual(saved["nanogpt/m"], [1, 2, 1, 1])
        price = backends.price_for("anthropic/claude-sonnet-4.5", "openrouter")
        self.assertEqual(price.source, "openrouter catalog")
        self.assertEqual(
            backends.price_for("free/model", "openrouter").label(), "$0/$0 per MTok"
        )

    def test_hosted_models_fetch_saves_the_catalog_prices(self):
        payload = {
            "data": [
                {
                    "id": "b/two",
                    "pricing": {"prompt": "0.000002", "completion": "0.000004"},
                },
                {"id": "a/one"},
            ]
        }
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
        cm.__exit__.return_value = False
        cache = self._tmp / "models.json"
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}),
            mock.patch("urllib.request.urlopen", return_value=cm),
        ):
            ids = configure._fetch_hosted_models(
                "openrouter", "https://x/models", cache, "X"
            )
        self.assertEqual(ids, ["a/one", "b/two"])
        saved = json.loads(backends.PRICES_FILE.read_text())
        self.assertEqual(list(saved), ["openrouter/b/two"])

    def test_model_chooser_shows_prices_beside_known_models(self):
        with mock.patch.object(ui, "colors_enabled", return_value=False):
            labels = configure._model_labels(
                "anthropic",
                ["claude-opus-5-5", "mystery-model", configure.CUSTOM_MODEL_OPTION],
            )
        self.assertEqual(
            strip_ansi(labels[0]).split(), ["claude-opus-5-5", "$4/$20", "per", "MTok"]
        )
        self.assertEqual(labels[1], "mystery-model")
        self.assertEqual(labels[2], configure.CUSTOM_MODEL_OPTION)

    def test_hint_is_drawn_under_the_prompt_and_cleared_by_the_menu(self):
        with mock.patch.object(ui, "colors_enabled", return_value=False):
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line("fix the bug", hint="≈ $0.0031 input")
                hinted = sys.stdout.getvalue()
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line("/mo", ["/model"], 0, hint="≈ $0.0031 input")
                menu = sys.stdout.getvalue()
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line("fix the bug")
                bare = sys.stdout.getvalue()
        self.assertIn("\n  ≈ $0.0031 input", hinted)
        self.assertTrue(
            hinted.endswith("\033[1A\r\033[13C"), repr(hinted)
        )  # back up to the text
        self.assertNotIn("≈", menu)
        self.assertIn("/model", menu)
        self.assertNotIn("\n", bare)

    def test_read_user_input_ignores_the_hint_without_a_terminal(self):
        calls = []
        with (
            mock.patch("sys.stdin", io.StringIO("hello\n")),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            text = ui.read_user_input(lambda t: calls.append(t) or "x")
        self.assertEqual(text, "hello")
        self.assertEqual(calls, [])

    def test_typing_hint_is_wired_into_the_prompt(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
            mock.patch.object(ui, "read_user_input", side_effect=EOFError),
            mock.patch.object(wrencode, "SHOW_USAGE", True),
            mock.patch.object(wrencode, "load_history", return_value=[]),
            mock.patch.object(wrencode, "build_system_prompt", return_value="s" * 400),
            mock.patch.object(wrencode, "find_agents_files", return_value=[]),
            mock.patch.object(history, "open_store", return_value=None),
            mock.patch.object(configure, "resolve_configuration"),
            mock.patch.object(backends, "load_model", return_value=None),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch.object(sys, "argv", ["wrencode"]),
        ):
            wrencode.main()
            banner = sys.stdout.getvalue()
            hint = ui.read_user_input.call_args[0][0]
            shown = hint("fix the bug")
        self.assertIn("$2/$10 per MTok", strip_ansi(banner))
        self.assertTrue(shown.startswith("≈ $"), shown)
        self.assertTrue(shown.endswith(" input"))


class TestApprovalAndResults(unittest.TestCase):
    """What the person sees at an approval, under a tool call, and while waiting."""

    def test_print_diff_is_a_colored_unified_diff_with_line_numbers(self):
        before = "a\nb\nc\nd\ne\nf\n"
        after = "a\nb\nc\nD\ne\nf\n"
        with mock.patch("sys.stdout", io.StringIO()):
            ui.print_diff("app.py", before, after)
            out = sys.stdout.getvalue()
        plain = strip_ansi(out)
        self.assertIn("@@ app.py:2", plain)
        self.assertIn(f"{ui.RED}-d{ui.RESET}", out)
        self.assertIn(f"{ui.GREEN}+D{ui.RESET}", out)
        self.assertNotIn("---", plain)
        self.assertTrue(all(line.startswith("    ") for line in plain.splitlines()))

    def test_print_diff_folds_a_long_change(self):
        before = "\n".join(str(i) for i in range(100))
        with mock.patch("sys.stdout", io.StringIO()):
            ui.print_diff("x", before, "", limit=10)
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn("… +", out)
        self.assertLessEqual(len(out.splitlines()), 11)

    def test_edit_shows_the_diff_and_asks_one_question(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "app.py").write_text("x = 1\ny = 2\n")
        with (
            mock.patch.object(wrencode, "workspace_root", return_value=tmp),
            mock.patch.object(ui, "_confirm_prompt", return_value="ok") as ask,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            result = wrencode.edit(
                {"path": str(tmp / "app.py"), "old": "y = 2", "new": "y = 3"}
            )
            out = strip_ansi(sys.stdout.getvalue())
        self.assertEqual(result, "ok")
        self.assertEqual(ask.call_args[0][0], "Apply to app.py?")
        self.assertIn("-y = 2", out)
        self.assertIn("+y = 3", out)
        self.assertIn("@@ app.py:1", out)

    def test_confirm_prompt_is_one_line(self):
        with (
            mock.patch("builtins.input", return_value=""),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(ui._confirm_prompt("Run it?"), "ok")
            out = strip_ansi(sys.stdout.getvalue())
        self.assertEqual(
            out.strip().splitlines(), ["Run it?  Enter yes · a always · n no"]
        )

    def _result(self, text, name=""):
        with mock.patch("sys.stdout", io.StringIO()):
            wrencode.print_tool_result(text, name)
            return strip_ansi(sys.stdout.getvalue())

    def test_tool_results_fold(self):
        self.assertEqual(self._result("ok"), "  ⎿ ok\n")
        self.assertEqual(self._result("  "), "  ⎿ (empty)\n")
        self.assertEqual(self._result("1: a\n2: b\n3: c", "read"), "  ⎿ 3 lines read\n")
        self.assertEqual(self._result("line\nline", "bash"), "  ⎿ 2 lines\n")
        self.assertEqual(self._result("ok", "bash"), "  ⎿ 1 line\n")  # streamed already
        self.assertEqual(
            self._result("error: nope\ndetail"), "  ⎿ error: nope\n    detail\n"
        )
        long = self._result("\n".join(f"m{i}" for i in range(20)), "grep")
        self.assertTrue(long.startswith("  ⎿ m0\n    m1\n"))
        self.assertIn("… +12 more lines", long)
        self.assertEqual(len(long.splitlines()), 9)

    def test_tool_action_opens_with_the_dot_and_edit_is_one_line(self):
        with mock.patch("sys.stdout", io.StringIO()):
            wrencode.print_tool_action("edit", {"path": "a.py", "old": "x", "new": "y"})
            wrencode.print_tool_action(
                "write", {"path": "b.py", "content": "1\n2\n3\n4\n5\n6\n7\n8"}
            )
            out = sys.stdout.getvalue()
        plain = strip_ansi(out)
        self.assertTrue(out.startswith(ui.TOOL_MARK + " edit a.py\n"))
        self.assertNotIn("- x", plain)
        self.assertIn("write b.py  (8 lines)", plain)
        self.assertIn("  … +2 lines", plain)

    def test_loader_shows_activity_elapsed_and_the_way_out(self):
        ctx = ui.loader_context("anthropic", "claude-sonnet-5-5", "running python")
        self.assertEqual(ctx, "running python · claude-sonnet-5-5")
        with mock.patch.object(ui, "colors_enabled", return_value=False):
            self.assertEqual(
                ui.loader_display(0, ctx, 4),
                "⠋ running python · claude-sonnet-5-5 · 4s · esc to cancel",
            )
            self.assertEqual(
                ui.loader_display(1, ctx), "⠙ running python · claude-sonnet-5-5"
            )


class TestIntuitiveUI(unittest.TestCase):
    """Startup line, key decoding, error hints, and the theme switch."""

    def _key(self, raw: bytes) -> str:
        r, w = os.pipe()
        try:
            os.write(w, raw)
            os.close(w)
            return ui._read_tty_key(r)
        finally:
            os.close(r)

    def test_editing_keys_are_decoded(self):
        self.assertEqual(self._key(b"\x1b[D"), "left")
        self.assertEqual(self._key(b"\x1b[H"), "home")
        self.assertEqual(self._key(b"\x1bOF"), "end")
        self.assertEqual(self._key(b"\x1b[1~"), "home")
        self.assertEqual(self._key(b"\x1b[3~"), "delete")
        self.assertEqual(self._key(b"\x01"), "home")
        self.assertEqual(self._key(b"\x05"), "end")
        self.assertEqual(self._key(b"\x17"), "ctrl_w")
        self.assertEqual(self._key(b"\x15"), "ctrl_u")
        self.assertEqual(self._key(b"\x0b"), "ctrl_k")
        self.assertEqual(self._key(b"\x1b"), "esc")
        self.assertEqual(self._key(b"x"), "x")

    def test_redraw_puts_the_cursor_back_mid_line(self):
        with mock.patch.object(ui, "colors_enabled", return_value=False):
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line("hello", cursor=2)
                mid = sys.stdout.getvalue()
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line("hello", hint="≈ $0.001 input", cursor=2)
                hinted = sys.stdout.getvalue()
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line("hello")
                end = sys.stdout.getvalue()
        self.assertTrue(mid.endswith("hello\r\033[4C"), repr(mid))  # "❯ " + 2
        self.assertTrue(hinted.endswith("\033[1A\r\033[4C"), repr(hinted))
        self.assertTrue(end.endswith("hello\r\033[7C"), repr(end))  # cursor at the end

    def test_errors_get_a_sentence_and_a_next_step(self):
        self.assertIn("/configure", ui.explain_error('HTTP 401: {"type":"error"}'))
        self.assertIn("/model", ui.explain_error("HTTP 404: not_found_error"))
        self.assertIn("rate limiting", ui.explain_error("HTTP 429: slow down"))
        self.assertIn("/compact", ui.explain_error("prompt is too long: 210000 tokens"))
        self.assertIn(
            "network",
            ui.explain_error("<urlopen error [Errno 111] Connection refused>"),
        )
        self.assertEqual(ui.explain_error("something odd"), "")
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch.dict(os.environ, {"WRENCODE_DEBUG": ""}),
        ):
            os.environ.pop("WRENCODE_DEBUG", None)
            ui.print_error(
                'HTTP 401: {"type":"error","error":{"message":"invalid x-api-key"}}'
            )
            ui.print_error("something odd")
            out = strip_ansi(sys.stdout.getvalue())
        lines = out.splitlines()
        self.assertEqual(
            lines[0],
            "Error: The API key was rejected. Run /configure to enter a new one.",
        )
        self.assertTrue(lines[1].startswith("HTTP 401"))
        self.assertEqual(lines[2], "Error: something odd")

    def test_light_background_detection(self):
        with mock.patch.dict(
            os.environ, {"WRENCODE_THEME": "light", "COLORFGBG": "15;0"}
        ):
            self.assertTrue(ui._light_background())
        with mock.patch.dict(os.environ, {"WRENCODE_THEME": "", "COLORFGBG": "15;0"}):
            self.assertFalse(ui._light_background())
        with mock.patch.dict(os.environ, {"WRENCODE_THEME": "", "COLORFGBG": "0;15"}):
            self.assertTrue(ui._light_background())
        with mock.patch.dict(os.environ, {"WRENCODE_THEME": "", "COLORFGBG": ""}):
            self.assertFalse(ui._light_background())

    def test_status_line_says_model_price_session_and_history(self):
        store = mock.Mock(mirror=None)
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
            mock.patch.object(wrencode, "_STORE", store),
            mock.patch.object(wrencode, "_SESSION_ID", 3),
        ):
            line = strip_ansi(
                wrencode._status_line([{"role": "user", "content": "a"}] * 2)
            )
        self.assertEqual(
            line,
            "claude-sonnet-5-5 · $2/$10 per MTok · session #3, 2 chats · history in Postgres",
        )
        with (
            mock.patch.object(backends, "BACKEND", "ollama"),
            mock.patch.object(backends, "MODEL", "llama3"),
            mock.patch.object(wrencode, "_STORE", None),
        ):
            self.assertEqual(strip_ansi(wrencode._status_line([])), "llama3")

    def test_picker_falls_back_to_numbers_without_a_terminal(self):
        with (
            mock.patch("sys.stdin.isatty", return_value=False),
            mock.patch("builtins.input", return_value="2"),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(ui.pick_from_list("Pick", ["a", "b"]), 1)


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

    def test_stream_printer_stays_quiet_without_text(self):
        with mock.patch("sys.stdout", io.StringIO()):
            p = ui.StreamPrinter()
            p.close()
            self.assertEqual(sys.stdout.getvalue(), "")
        self.assertFalse(p.started)


class TestLineEditor(unittest.TestCase):
    """The input buffer: single and multi-line editing without a terminal."""

    def _run(self, keys, **kw):
        ed = ui.LineEditor(**kw)
        out = None
        for k in keys:
            out = ed.apply(k)
        return ed, out

    def test_typing_and_submit(self):
        _ed, out = self._run([*list("hi"), "enter"])
        self.assertEqual(out, "hi")

    def test_backslash_enter_continues_and_enter_submits_everything(self):
        _ed, out = self._run([*list("one\\"), "enter", *list("two"), "enter"])
        self.assertEqual(out, "one\ntwo")

    def test_alt_enter_and_paste_insert_newlines(self):
        _ed, out = self._run(
            [*list("a"), "alt_enter", *list("b"), "paste:x\ny", "enter"]
        )
        self.assertEqual(out, "a\nbx\ny")

    def test_up_and_down_move_between_lines_before_touching_history(self):
        ui._INPUT_HISTORY[:] = ["older"]
        try:
            ed, _ = self._run(["paste:first line\nsecond"], history=True)
            self.assertEqual(ed.cur, len("first line\nsecond"))
            ed.apply("up")
            self.assertEqual(ed.cur, 6)  # same column on the first line
            ed.apply("down")
            self.assertEqual(ed.cur, len("first line\n") + 6)
            ed.apply("home")
            self.assertEqual(ed.cur, len("first line\n"))
            ed.apply("end")
            self.assertEqual(ed.cur, len("first line\nsecond"))
            ed.apply("up")
            ed.apply("up")  # from the first line: history
            self.assertEqual(ed.text, "older")
            ed.apply("down")
            self.assertEqual(ed.text, "")
        finally:
            ui._INPUT_HISTORY.clear()

    def test_word_and_line_deletes_stay_on_their_line(self):
        ed, _ = self._run(["paste:keep this\ndrop that"])
        ed.apply("ctrl_w")
        self.assertEqual(ed.text, "keep this\ndrop ")
        ed.apply("ctrl_u")
        self.assertEqual(ed.text, "keep this\n")
        ed.apply("up")
        ed.apply("ctrl_k")
        self.assertEqual(ed.text, "\n")

    def test_slash_menu_only_on_a_single_line(self):
        ed, _ = self._run(list("/mo"), complete=True)
        self.assertEqual(ed.matches(), ["/model"])
        self.assertEqual(ed.apply("enter"), "/model")
        ed2, _ = self._run([*list("/mo"), "alt_enter"], complete=True)
        self.assertEqual(ed2.matches(), [])

    def test_control_keys_raise(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(["ctrl_c"])
        with self.assertRaises(EOFError):
            self._run(["ctrl_d"])
        ed, out = self._run([*list("x"), "ctrl_d"])
        self.assertEqual((ed.text, out), ("x", None))

    def _key(self, raw: bytes) -> str:
        r, w = os.pipe()
        try:
            os.write(w, raw)
            os.close(w)
            return ui._read_tty_key(r)
        finally:
            os.close(r)

    def test_alt_enter_and_bracketed_paste_are_decoded(self):
        self.assertEqual(self._key(b"\x1b\r"), "alt_enter")
        self.assertEqual(
            self._key(b"\x1b[200~line one\r\nline two\x1b[201~"),
            "paste:line one\nline two",
        )

    def test_multi_line_draw_counts_wrapped_rows(self):
        with (
            mock.patch.object(ui, "colors_enabled", return_value=False),
            mock.patch.object(ui, "_cols", return_value=20),
        ):
            ui._LAST_CURSOR_ROW = 0
            with mock.patch("sys.stdout", io.StringIO()):
                # first line wraps to two rows (2 + 25 cells); cursor at the start of line two
                ui._redraw_input_line("a" * 25 + "\nbb", cursor=26)
                out = sys.stdout.getvalue()
            self.assertEqual(ui._LAST_CURSOR_ROW, 2)
            self.assertTrue(
                out.startswith("\r\x1b[J❯ " + "a" * 25 + "\n  bb"), repr(out[:20])
            )
            self.assertTrue(
                out.endswith("\r\x1b[2C"), repr(out[-12:])
            )  # cursor after the margin
            with mock.patch("sys.stdout", io.StringIO()):
                ui._redraw_input_line(
                    "a" * 25 + "\nbb", cursor=3
                )  # back up two rows first
                out = sys.stdout.getvalue()
            self.assertTrue(out.startswith("\x1b[2A\r\x1b[J"), repr(out[:12]))
            self.assertTrue(
                out.endswith("\x1b[2A\r\x1b[5C"), repr(out[-12:])
            )  # row 0, col 2+3
            self.assertEqual(ui._LAST_CURSOR_ROW, 0)


class TestPermissions(unittest.TestCase):
    """Rules that approve or refuse without asking, and where they come from."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.user = self.tmp / "cfg" / "permissions.json"
        self.project = self.tmp / "proj"
        self.project.mkdir()
        self.rules = permissions.Permissions(self.user, self.project)
        self._orig = permissions.ACTIVE
        permissions.ACTIVE = self.rules
        self.addCleanup(setattr, permissions, "ACTIVE", self._orig)

    def test_rule_syntax_and_matching(self):
        self.assertEqual(permissions.parse("bash(git *)"), ("bash", "git *"))
        with self.assertRaises(ValueError):
            permissions.parse("read(x)")
        with self.assertRaises(ValueError):
            permissions.parse("bash()")
        r = permissions.Rule("bash", "git *", "allow", "user")
        self.assertTrue(r.matches("bash", "git status"))
        self.assertFalse(r.matches("bash", "gitk"))
        self.assertFalse(r.matches("edit", "git status"))
        self.assertTrue(
            permissions.Rule("bash", "pytest:*", "allow", "user").matches(
                "bash", "pytest -q tests"
            )
        )
        self.assertTrue(
            permissions.Rule("edit", "src/*", "allow", "user").matches(
                "edit", "src/a/b.py"
            )
        )
        self.assertFalse(
            permissions.Rule("edit", "src/*", "allow", "user").matches(
                "edit", "lib/b.py"
            )
        )
        self.assertTrue(
            permissions.Rule("write", ".env", "deny", "user").matches("write", ".env")
        )

    def test_suggestions(self):
        self.assertEqual(
            permissions.suggest("bash", "npm test -- --watch"), "bash(npm test:*)"
        )
        self.assertEqual(permissions.suggest("bash", "make"), "bash(make)")
        self.assertEqual(permissions.suggest("edit", "src/app/x.py"), "edit(src/app/*)")
        self.assertEqual(permissions.suggest("write", "README.md"), "write(README.md)")

    def test_user_rules_apply_and_deny_wins(self):
        self.rules.add("bash(git *)", "allow", "user")
        self.rules.add("bash(git push *)", "deny", "user")
        self.assertEqual(stat.S_IMODE(self.user.stat().st_mode), 0o600)
        self.assertEqual(str(self.rules.check("bash", "git status")), "bash(git *)")
        self.assertEqual(
            self.rules.check("bash", "git push origin main").effect, "deny"
        )
        self.assertIsNone(self.rules.check("bash", "rm -rf x"))
        self.assertEqual(self.rules.remove("bash(git *)"), 1)
        self.assertIsNone(self.rules.check("bash", "git status"))

    def test_project_allow_rules_need_acceptance_but_deny_rules_do_not(self):
        pf = self.project / permissions.PROJECT_FILE
        pf.parent.mkdir()
        pf.write_text(json.dumps({"allow": ["bash(*)"], "deny": ["write(.env)"]}))
        rules = permissions.Permissions(self.user, self.project)
        self.assertFalse(rules.project_trusted())
        self.assertIsNone(
            rules.check("bash", "anything")
        )  # shipped by the repo: ignored
        self.assertEqual(rules.check("write", ".env").effect, "deny")
        rules.trust_project()
        self.assertTrue(rules.project_trusted())
        self.assertEqual(rules.check("bash", "anything").source, "project")
        pf.write_text(
            json.dumps({"allow": ["bash(*)", "edit(*)"]})
        )  # changed since: ask again
        rules = permissions.Permissions(self.user, self.project)
        self.assertFalse(rules.project_trusted())
        rules.add(
            "edit(docs/*)", "allow", "project"
        )  # written by the person: trusted as it stands
        self.assertTrue(rules.project_trusted())

    def test_confirm_follows_the_rules(self):
        self.rules.add("edit(src/*)", "allow", "user")
        self.rules.add("bash(rm *)", "deny", "user")
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch.object(ui, "SESSION_AUTO_APPROVE", True),
        ):
            self.assertEqual(ui.confirm("edit", "Apply to src/a.py?", "src/a.py"), "ok")
            denied = ui.confirm("bash", "Run it?", "rm -rf /")
            out = strip_ansi(sys.stdout.getvalue())
        self.assertTrue(
            denied.startswith("cancelled: denied by the permission rule bash(rm *)")
        )
        self.assertIn("✓ edit src/a.py [allowed by rule edit(src/*)]", out)
        self.assertIn("⊘ bash rm -rf / [denied by rule bash(rm *)]", out)

    def test_s_at_the_prompt_saves_a_project_rule(self):
        with (
            mock.patch("builtins.input", return_value="s"),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(
                ui._confirm_prompt("Run it?", "bash", "npm test -- -q"), "ok"
            )
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn("s allow bash(npm test:*)", out)
        self.assertIn("Saved bash(npm test:*)", out)
        saved = json.loads((self.project / permissions.PROJECT_FILE).read_text())
        self.assertEqual(saved["allow"], ["bash(npm test:*)"])
        self.assertEqual(
            str(self.rules.check("bash", "npm test --watch")), "bash(npm test:*)"
        )

    def test_permissions_command(self):
        with mock.patch("sys.stdout", io.StringIO()):
            wrencode.handle_slash_command("/permissions", [], None)
            wrencode.handle_slash_command(
                "/permissions allow bash(git *) --user", [], None
            )
            wrencode.handle_slash_command("/permissions deny write(.env)", [], None)
            wrencode.handle_slash_command("/permissions", [], None)
            wrencode.handle_slash_command("/permissions forget bash(git *)", [], None)
            wrencode.handle_slash_command("/permissions allow nope", [], None)
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn("No rules", out)
        self.assertIn("allow  bash(git *)", out)
        self.assertIn("user", out)
        self.assertIn("deny   write(.env)", out)
        self.assertIn("Removed 1 rule", out)
        self.assertIn("not a rule", out)
        self.assertEqual([str(r) for r in self.rules.rules()], ["write(.env)"])


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
        self._orig = (mcp.ACTIVE, dict(wrencode.TOOLS))
        self.addCleanup(self._restore)

    def _restore(self):
        if mcp.ACTIVE is not None and mcp.ACTIVE is not self._orig[0]:
            mcp.ACTIVE.close()
        mcp.ACTIVE = self._orig[0]
        wrencode.TOOLS.clear()
        wrencode.TOOLS.update(self._orig[1])

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
        self.assertEqual(wrencode._register_mcp_tools(), 2)
        self.assertIn("mcp__fake__echo", wrencode.TOOLS)
        spec = next(s for s in wrencode.tool_specs() if s[0] == "mcp__fake__echo")
        self.assertEqual(spec[2]["required"], ["text"])
        self.assertIn("mcp__fake__echo(text)", wrencode.build_system_prompt())
        with (
            mock.patch.object(ui, "_confirm_prompt", return_value="ok") as ask,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(
                wrencode.run_tool("mcp__fake__echo", {"text": "yo"}), "echo: yo"
            )
            self.assertEqual(ask.call_args[0][0], "Call fake:echo?")
            self.assertEqual(wrencode.run_tool("mcp__fake__whoami", {}), "it is me")
            self.assertEqual(ask.call_count, 1)  # read-only: no approval
        with (
            mock.patch.object(
                ui,
                "_confirm_prompt",
                return_value="cancelled: user declined without instructions",
            ),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertTrue(
                wrencode.run_tool("mcp__fake__echo", {"text": "yo"}).startswith(
                    "cancelled"
                )
            )
        self.assertEqual(
            wrencode.format_tool_action("mcp__fake__echo", {"text": "yo"}),
            'mcp fake:echo {"text": "yo"}',
        )
        with mock.patch("sys.stdout", io.StringIO()):
            wrencode.handle_slash_command("/mcp", [], None)
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn("✓ fake", out)
        self.assertIn("2 tools: echo, whoami", out)

    def test_tool_names_are_api_safe(self):
        self.assertEqual(
            mcp.tool_name("my server", "do.thing"), "mcp__my_server__do_thing"
        )
        self.assertLessEqual(len(mcp.tool_name("s" * 40, "t" * 40)), 64)


class TestWeb(unittest.TestCase):
    """The fetch tool and Anthropic's server-side web search."""

    def _serve(self, routes):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                ctype, body = routes.get(self.path, ("text/plain", b"nope"))
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

    def test_fetch_tool_asks_and_follows_rules(self):
        base = self._serve({"/p": ("text/plain", b"hi")})
        self.assertEqual(
            web.subject(base + "/p?x=1"), f"127.0.0.1:{base.rsplit(':', 1)[1]}/p"
        )
        self.assertEqual(
            permissions.suggest("fetch", "docs.python.org/3/library"),
            "fetch(docs.python.org/*)",
        )
        with (
            mock.patch.object(ui, "_confirm_prompt", return_value="ok") as ask,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            out = wrencode.run_tool("fetch", {"url": base + "/p"})
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
                wrencode.run_tool("fetch", {"url": base + "/p"}).startswith("cancelled")
            )
        self.assertIn("fetch(url, offset)", wrencode.build_system_prompt())
        self.assertEqual(
            wrencode.format_tool_action("fetch", {"url": "https://a/b", "offset": 40}),
            "fetch https://a/b  offset=40",
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
            self.assertEqual(tools[0], backends.WEB_SEARCH_TOOL)
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

    def test_server_blocks_print_without_streaming(self):
        block = {
            "type": "server_tool_use",
            "name": "web_search",
            "input": {"query": "q"},
        }
        self.assertEqual(
            strip_ansi(wrencode._server_block_line(block)), '● web_search "q"'
        )
        result = {"type": "web_search_tool_result", "content": []}
        self.assertEqual(
            strip_ansi(wrencode._server_block_line(result)), "  ⎿ no results"
        )
        with mock.patch("sys.stdout", io.StringIO()):
            p = ui.StreamPrinter()
            p.feed("Looking")
            p.note('● web_search "q"')
            p.feed("done\n")
            p.close()
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn('● Looking\n● web_search "q"\n  done\n', out)


class TestSpeed(unittest.TestCase):
    """Output tokens per second per turn, and the arrow against the previous turn."""

    def setUp(self):
        self._orig = backends.USAGE
        backends.USAGE = backends.Usage()
        self.addCleanup(setattr, backends, "USAGE", self._orig)

    def _call(self, out: int, seconds: float) -> None:
        with mock.patch("time.monotonic", side_effect=[100.0, 100.0 + seconds]):
            backends.USAGE.begin_call()
            backends._record_usage(
                {"usage": {"input_tokens": 500, "output_tokens": out}}
            )

    def test_rate_trend_and_where_it_shows(self):
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
        ):
            backends.USAGE.begin_turn()
            self._call(300, 6.0)
            first = backends.usage_line()
            backends.USAGE.begin_turn()
            self._call(400, 5.0)
            self._call(200, 5.0)
            faster = backends.usage_line()
            report = backends.usage_report()
            title = backends.usage_title()
            backends.USAGE.begin_turn()
            self._call(50, 2.0)
            slower = backends.usage_line()
            backends.USAGE.begin_turn()
            self._call(270, 10.0)  # 27 tok/s against 25: within a tenth
            steady = backends.usage_line()
            as_dict = backends.USAGE.as_dict()
        self.assertIn("  50 tok/s  ", first)  # no arrow on the first turn
        self.assertNotIn("↗", first)
        self.assertIn("  60 tok/s ↗  ", faster)
        self.assertIn("  25 tok/s ↘  ", slower)
        self.assertIn("  27 tok/s →  ", steady)
        self.assertEqual(
            report[-1],
            "speed: 60 tok/s this turn (↗ from 50 tok/s last turn); 56 tok/s this "
            "session; output tokens over the request's wall time",
        )
        self.assertTrue(title.endswith(" · 60 tok/s ↗"), title)
        self.assertEqual(as_dict["output_tokens_per_second"], 43.6)

    def test_untimed_calls_show_no_rate(self):
        with mock.patch.object(backends, "BACKEND", "anthropic"):
            backends._record_usage({"usage": {"input_tokens": 5, "output_tokens": 50}})
        self.assertNotIn("tok/s", backends.usage_line())
        self.assertNotIn("tok/s", backends.usage_title())
        self.assertNotIn("output_tokens_per_second", backends.USAGE.as_dict())
        self.assertFalse(any("speed:" in line for line in backends.usage_report()))
        self.assertEqual(backends._speed(7.25), "7.2 tok/s")

    def test_get_response_starts_the_clock(self):
        with (
            mock.patch.object(backends, "BACKEND", "openrouter"),
            mock.patch.object(backends, "API_BASE", "https://x/y"),
            mock.patch.object(
                backends,
                "_http_post",
                return_value={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 40},
                },
            ),
            mock.patch("time.monotonic", side_effect=[10.0, 12.0]),
        ):
            backends.get_response([{"role": "user", "content": "x"}], "sys", None)
        self.assertEqual(backends.USAGE.turn_seconds, 2.0)
        self.assertEqual(backends.USAGE.turn_rate(), 20.0)
        self.assertEqual(backends.USAGE.call_started, 0.0)


class TestComplete(unittest.TestCase):
    """backends.complete(): one-shot completions shared by compaction and synthesize."""

    def setUp(self):
        self._orig = (
            backends.BACKEND,
            backends.MODEL,
            backends.API_KEY,
            backends.API_BASE,
        )
        self._env_model = os.environ.pop("MODEL", None)

    def tearDown(self):
        backends.BACKEND, backends.MODEL, backends.API_KEY, backends.API_BASE = (
            self._orig
        )
        if self._env_model is not None:
            os.environ["MODEL"] = self._env_model

    def test_anthropic_prefill_is_sent_and_returned(self):
        backends.apply_backend("anthropic", model="claude-x", api_key="sk-t")
        captured = {}

        def fake_post(url, payload, headers):
            captured["payload"] = payload
            return {"content": [{"type": "text", "text": '"a": 1}'}]}

        with mock.patch.object(backends, "_http_post", fake_post):
            out = backends.complete("sys", "user", prefill="{", max_tokens=64)
        self.assertEqual(out, '{"a": 1}')
        self.assertEqual(captured["payload"]["system"], "sys")
        self.assertEqual(captured["payload"]["max_tokens"], 64)
        self.assertEqual(
            captured["payload"]["messages"],
            [
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "{"},
            ],
        )
        self.assertNotIn("tools", captured["payload"])

    def test_openai_format_sends_temperature_and_ignores_prefill(self):
        backends.apply_backend("openai", model="gpt-x", api_key="sk-t")
        captured = {}

        def fake_post(url, payload, headers):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "  hello  "}}]}

        with mock.patch.object(backends, "_http_post", fake_post):
            out = backends.complete("sys", "user", temperature=0.0, prefill="{")
        self.assertEqual(out, "hello")
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["messages"][0]["role"], "system")
        self.assertNotIn("tools", captured["payload"])

    def test_ollama_uses_chat_completions(self):
        backends.apply_backend("ollama", model="llama3.2")
        with mock.patch.object(
            backends,
            "_http_post",
            return_value={"choices": [{"message": {"content": "ok"}}]},
        ) as m:
            self.assertEqual(backends.complete("sys", "user", max_tokens=5), "ok")
        self.assertEqual(m.call_args.args[1]["max_tokens"], 5)

    def test_bedrock_prefill_through_converse(self):
        with mock.patch.object(backends, "_aws_region", return_value="us-east-1"):
            backends.apply_backend("bedrock", model="us.anthropic.claude-x")
        captured = {}

        def fake_converse(body):
            captured["body"] = body
            return {"output": {"message": {"content": [{"text": '"b": 2}'}]}}}

        with mock.patch.object(backends, "_bedrock_converse_call", fake_converse):
            out = backends.complete("sys", "user", prefill="{", temperature=0.0)
        self.assertEqual(out, '{"b": 2}')
        self.assertEqual(captured["body"]["inferenceConfig"]["temperature"], 0.0)
        self.assertEqual(captured["body"]["messages"][1]["role"], "assistant")

    def test_local_model_goes_through_generate_local(self):
        # The old compaction path called the mlx generator for transformers too.
        backends.apply_backend("transformers")
        with mock.patch.object(
            backends, "_generate_local", return_value="<tool_call>{}</tool_call> text"
        ) as gen:
            out = backends.complete("sys", "user", mlx_state=("model", "tok"))
        self.assertEqual(out, "text")
        chat, state, max_tokens, temperature = gen.call_args.args
        self.assertEqual([m["role"] for m in chat], ["system", "user"])
        self.assertEqual(state, ("model", "tok"))
        self.assertEqual(max_tokens, backends.MAX_TOKENS)
        self.assertEqual(temperature, 0.3)

    def test_local_model_without_weights_is_an_error(self):
        backends.apply_backend("mlx")
        with self.assertRaises(RuntimeError):
            backends.complete("sys", "user")


class TestNanoGPTBackend(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("NANOGPT_API_KEY")
        os.environ["NANOGPT_API_KEY"] = "nano-test-key"
        backends.apply_backend("nanogpt")

    def tearDown(self):
        if self._env is None:
            os.environ.pop("NANOGPT_API_KEY", None)
        else:
            os.environ["NANOGPT_API_KEY"] = self._env

    def test_defaults(self):
        self.assertEqual(backends.API_KEY, "nano-test-key")
        self.assertEqual(
            backends.API_BASE, "https://nano-gpt.com/api/v1/chat/completions"
        )
        self.assertIn("nanogpt", backends.NATIVE_TOOL_BACKENDS)

    def test_system_prompt_has_no_xml_tool_tags(self):
        # GLM models on NanoGPT 503 when <tool_call> appears in the prompt.
        self.assertNotIn("<tool_call>", wrencode.build_system_prompt())

    def test_anthropic_history_converted_to_openai_tool_calls(self):
        history = [
            {"role": "user", "content": "count lines"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"cmd": "wc -l f"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "bash",
                        "input": {"cmd": "ls"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "3 f"}
                ],
            },
        ]
        out = backends._to_openai_messages(history)
        self.assertEqual(out[0], {"role": "user", "content": "count lines"})
        self.assertEqual(out[1]["content"], "Checking.")
        # unanswered t2 is dropped so every tool_call id has a result
        self.assertEqual([c["id"] for c in out[1]["tool_calls"]], ["t1"])
        self.assertEqual(
            json.loads(out[1]["tool_calls"][0]["function"]["arguments"]),
            {"cmd": "wc -l f"},
        )
        self.assertEqual(
            out[2], {"role": "tool", "tool_call_id": "t1", "content": "3 f"}
        )
        self.assertEqual(len(out), 3)

    def test_xml_tool_tags_in_history_are_defanged(self):
        history = [
            {"role": "assistant", "content": '<tool_call>{"tool": "ls"}</tool_call>'}
        ]
        out = backends._to_openai_messages(history)
        self.assertNotIn("<tool_call>", out[0]["content"])
        self.assertNotIn("</tool_call>", out[0]["content"])

    def test_orphan_tool_result_becomes_user_text(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "ok"}
                ],
            }
        ]
        self.assertEqual(
            backends._to_openai_messages(history),
            [{"role": "user", "content": "Tool result: ok"}],
        )

    def test_native_openai_messages_pass_through(self):
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
        ]
        self.assertEqual(backends._to_openai_messages(history), history)


class TestAgentsMd(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        self.repo = self._tmp / "repo"
        self.ws = self.repo / "pkg"
        self.ws.mkdir(parents=True)
        (self.repo / ".git").mkdir()
        self._patches = [
            mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": str(self.ws)}),
            mock.patch.object(backends, "CONFIG_DIR", self._tmp / "config"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        import shutil

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_git_root_down_to_workspace(self):
        (self._tmp / "AGENTS.md").write_text("outside the repo")
        (self.repo / "AGENTS.md").write_text("root rules")
        (self.ws / "CLAUDE.md").write_text("pkg rules")
        self.assertEqual(
            wrencode.find_agents_files(),
            [self.repo / "AGENTS.md", self.ws / "CLAUDE.md"],
        )
        ctx = wrencode.agents_md_context()
        self.assertLess(ctx.index("root rules"), ctx.index("pkg rules"))
        self.assertNotIn("outside the repo", ctx)

    def test_agents_md_preferred_over_claude_md(self):
        (self.ws / "AGENTS.md").write_text("agents")
        (self.ws / "CLAUDE.md").write_text("claude")
        self.assertEqual(wrencode.find_agents_files(), [self.ws / "AGENTS.md"])

    def test_user_wide_file_comes_first(self):
        (self._tmp / "config").mkdir()
        (self._tmp / "config" / "AGENTS.md").write_text("mine")
        (self.ws / "AGENTS.md").write_text("project")
        self.assertEqual(
            wrencode.find_agents_files()[0], self._tmp / "config" / "AGENTS.md"
        )

    def test_outside_git_only_workspace(self):
        import shutil

        shutil.rmtree(self.repo / ".git")
        (self.repo / "AGENTS.md").write_text("parent")
        self.assertEqual(wrencode.find_agents_files(), [])

    def test_in_system_prompt_and_capped(self):
        (self.ws / "AGENTS.md").write_text("x" * 50_000)
        prompt = wrencode.build_system_prompt()
        self.assertIn("Project instructions from AGENTS.md", prompt)
        self.assertIn("x" * 100, prompt)
        self.assertNotIn("x" * (wrencode.MAX_AGENTS_MD_CHARS + 1), prompt)

    def test_no_files_no_section(self):
        self.assertNotIn("AGENTS.md", wrencode.build_system_prompt())


class TestHeadless(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        env = {"WRENCODE_WORKSPACE": str(self._tmp)}
        self._patches = [
            mock.patch.dict(os.environ, env),
            mock.patch.object(configure, "resolve_configuration", lambda: None),
            mock.patch.object(backends, "load_model", lambda: None),
            mock.patch.object(ui, "HEADLESS", False),
            mock.patch.object(backends, "BACKEND", "ollama"),
            mock.patch.object(backends, "MODEL", "llama3.2"),
        ]
        for p in self._patches:
            p.start()
        os.environ.pop("WRENCODE_AUTO_APPROVE", None)

    def tearDown(self):
        import shutil

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def run_headless(self, replies, *args):
        replies = iter(replies)
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(backends, "get_response", lambda *a: next(replies)),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", err),
        ):
            code = wrencode.run_headless("do it", *args)
        return code, out.getvalue(), err.getvalue()

    def test_stdout_is_only_the_answer(self):
        (self._tmp / "a.txt").write_text("hi")
        code, out, err = self.run_headless(
            [
                '<tool_call>{"tool": "read", "args": {"path": "a.txt"}}</tool_call>',
                "Done.",
            ]
        )
        self.assertEqual((code, out), (0, "Done.\n"))
        self.assertIn("read a.txt", err)

    def test_json_output(self):
        code, out, _ = self.run_headless(["All good."], "json")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["result"], "All good.")
        self.assertFalse(data["is_error"])
        self.assertEqual(data["stop_reason"], "done")
        self.assertEqual((data["backend"], data["num_turns"]), ("ollama", 1))

    def test_writes_declined_without_yes(self):
        code, _, _ = self.run_headless(
            [
                '<tool_call>{"tool": "write", "args": {"path": "b.txt", "content": "x"}}</tool_call>',
                "Couldn't write.",
            ]
        )
        self.assertEqual(code, 0)
        self.assertFalse((self._tmp / "b.txt").exists())

    def test_writes_allowed_with_yes(self):
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
        self.run_headless(
            [
                '<tool_call>{"tool": "write", "args": {"path": "b.txt", "content": "x"}}</tool_call>',
                "Wrote it.",
            ]
        )
        self.assertEqual((self._tmp / "b.txt").read_text(), "x")

    def test_max_turns_is_an_error(self):
        call = '<tool_call>{"tool": "glob", "args": {"pat": "*"}}</tool_call>'
        code, out, _ = self.run_headless([call, call, call], "json", 2)
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(data["stop_reason"], "max_turns")
        self.assertTrue(data["is_error"])

    def test_backend_error_reported(self):
        def boom(*a):
            raise RuntimeError("HTTP 401")

        out = io.StringIO()
        with (
            mock.patch.object(backends, "get_response", boom),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            code = wrencode.run_headless("x", "json")
        data = json.loads(out.getvalue())
        self.assertEqual(
            (code, data["error"], data["stop_reason"]), (1, "HTTP 401", "error")
        )

    def test_setup_failure_still_reports_json(self):
        def no_backend():
            raise SystemExit(1)

        out = io.StringIO()
        with (
            mock.patch.object(configure, "resolve_configuration", no_backend),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            code = wrencode.run_headless("x", "json")
        data = json.loads(out.getvalue())
        self.assertEqual((code, data["stop_reason"]), (1, "error"))
        self.assertIn("configuration error", data["error"])

    def test_arg_value(self):
        self.assertEqual(wrencode._arg_value(["-p", "hi"], "-p", "--print"), "hi")
        self.assertEqual(wrencode._arg_value(["--print=hi"], "-p", "--print"), "hi")
        self.assertEqual(wrencode._arg_value(["-p", "-"], "-p"), "-")
        self.assertIsNone(wrencode._arg_value(["-p", "--yes"], "-p"))
        self.assertIsNone(wrencode._arg_value(["-p"], "-p"))

    def test_main_dispatch(self):
        argv = [
            "wrencode",
            "--yes",
            "-p",
            "fix it",
            "--output-format",
            "json",
            "--max-turns",
            "3",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(wrencode, "run_headless", return_value=0) as run,
            mock.patch.dict(os.environ, {}),
            self.assertRaises(SystemExit) as cm,
        ):
            wrencode.main()
        self.assertEqual(cm.exception.code, 0)
        run.assert_called_once_with("fix it", "json", 3, None, "")

    def test_main_reads_piped_stdin(self):
        stdin = io.StringIO("from a pipe")
        with (
            mock.patch.object(sys, "argv", ["wrencode", "-p"]),
            mock.patch("sys.stdin", stdin),
            mock.patch.object(wrencode, "run_headless", return_value=0) as run,
            self.assertRaises(SystemExit),
        ):
            wrencode.main()
        run.assert_called_once_with("from a pipe", "text", 0, None, "")


class TestOpenAICompatibleBackend(unittest.TestCase):
    """Drive the backend against a stub OpenAI-compatible server (like vLLM/llama.cpp)."""

    def setUp(self):
        import http.server
        import threading

        self.requests = []
        self.models = ["qwen2.5-coder"]
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _send(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"data": [{"id": m} for m in test.models]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.requests.append((self.path, dict(self.headers), req))
                if len(test.requests) == 1:
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "glob",
                                    "arguments": '{"pat": "*.txt"}',
                                },
                            }
                        ],
                    }
                else:
                    msg = {"role": "assistant", "content": "Found notes.txt."}
                self._send({"choices": [{"message": msg, "finish_reason": "stop"}]})

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        (self._tmp / "notes.txt").write_text("x")
        base = f"http://127.0.0.1:{self.server.server_port}/v1"
        self._patches = [
            mock.patch.dict(
                os.environ,
                {
                    "OPENAI_COMPATIBLE_BASE_URL": base,
                    "WRENCODE_WORKSPACE": str(self._tmp),
                },
            ),
            mock.patch.object(ui, "HEADLESS", False),
            mock.patch.object(backends, "CONFIG_DIR", self._tmp / "config"),
        ]
        for p in self._patches:
            p.start()
        for var in ("MODEL", "OPENAI_COMPATIBLE_API_KEY"):
            os.environ.pop(var, None)
        self._saved = (
            backends.BACKEND,
            backends.MODEL,
            backends.API_KEY,
            backends.API_BASE,
        )

    def tearDown(self):
        import shutil

        self.server.shutdown()
        self.server.server_close()
        for p in self._patches:
            p.stop()
        (backends.BACKEND, backends.MODEL, backends.API_KEY, backends.API_BASE) = (
            self._saved
        )
        shutil.rmtree(self._tmp, ignore_errors=True)

    def headless(self):
        out = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"BACKEND": "openai-compatible"}),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            code = wrencode.run_headless("find text files", "json")
        return code, json.loads(out.getvalue())

    def test_base_url_normalized(self):
        with mock.patch.dict(
            os.environ,
            {"OPENAI_COMPATIBLE_BASE_URL": "http://h:8080/v1/chat/completions/"},
        ):
            backends.apply_backend("openai-compatible")
        self.assertEqual(backends.API_BASE, "http://h:8080/v1/chat/completions")
        self.assertEqual(backends.API_KEY, "EMPTY")
        self.assertIn("openai-compatible", backends.OPENAI_FORMAT_BACKENDS)

    def test_end_to_end_native_tool_calls(self):
        code, data = self.headless()
        self.assertEqual(code, 0)
        self.assertEqual(data["result"], "Found notes.txt.")
        self.assertEqual(data["model"], "qwen2.5-coder")  # the server's only model
        (path, headers, first), (_, _, second) = self.requests
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(first["model"], "qwen2.5-coder")
        self.assertIn("glob", [t["function"]["name"] for t in first["tools"]])
        self.assertNotIn("<tool_call>", first["messages"][0]["content"])
        tool_msg = second["messages"][-1]
        self.assertEqual((tool_msg["role"], tool_msg["tool_call_id"]), ("tool", "c1"))
        self.assertIn("notes.txt", tool_msg["content"])
        self.assertEqual(headers["Authorization"], "Bearer EMPTY")

    def test_api_key_sent(self):
        with mock.patch.dict(os.environ, {"OPENAI_COMPATIBLE_API_KEY": "hf_abc"}):
            self.headless()
        self.assertEqual(self.requests[0][1]["Authorization"], "Bearer hf_abc")

    def test_several_models_need_explicit_model(self):
        self.models = ["a", "b"]
        code, data = self.headless()
        self.assertEqual((code, data["stop_reason"]), (1, "error"))
        self.assertIn("configuration error", data["error"])
        self.assertEqual(self.requests, [])
        with mock.patch.dict(os.environ, {"MODEL": "b"}):
            code, data = self.headless()
        self.assertEqual((code, data["model"]), (0, "b"))

    def test_rejected_key_explained(self):
        def deny(*a, **k):
            import urllib.error

            raise urllib.error.HTTPError("u", 401, "x", Message(), io.BytesIO(b""))

        err = io.StringIO()
        with (
            mock.patch("urllib.request.urlopen", deny),
            mock.patch.object(backends, "BACKEND", "openai-compatible"),
            mock.patch.object(backends, "MODEL", ""),
            mock.patch("sys.stdout", err),
            self.assertRaises(SystemExit),
        ):
            backends.load_model()
        self.assertIn("rejected the key (HTTP 401)", err.getvalue())

    def test_lists_served_models(self):
        self.models = ["m2", "m1"]
        backends.apply_backend("openai-compatible")
        self.assertEqual(configure.fetch_openai_compatible_models(), ["m1", "m2"])
        self.assertIn("m1", configure.list_models_for_backend("openai-compatible"))


class TestAutoCompact(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(backends, "CONTEXT_TOKENS", 4000),
            mock.patch.object(backends, "complete", return_value="SUMMARY"),
            mock.patch("sys.stdout", io.StringIO()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def openai_history(self, rounds):
        msgs = [{"role": "user", "content": "build the thing"}]
        for i in range(rounds):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append(
                {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 1500}
            )
        return msgs

    def test_keeps_tail_paired_and_task_verbatim(self):
        msgs = self.openai_history(20)
        wrencode.auto_compact(msgs, None)
        note, first = msgs[0], msgs[1]
        self.assertEqual(note["role"], "user")
        self.assertIn("SUMMARY", note["content"])
        self.assertIn("build the thing", note["content"])
        self.assertEqual(first["role"], "assistant")
        self.assertEqual(msgs[2]["tool_call_id"], first["tool_calls"][0]["id"])
        self.assertEqual(msgs[-1]["tool_call_id"], "c19")
        self.assertLess(len(msgs), 41)
        self.assertLessEqual(wrencode.estimate_tokens(msgs[1:], ""), 1000)
        self.assertEqual(backends._to_openai_messages(msgs), msgs)  # nothing orphaned

    def test_anthropic_blocks_stay_paired(self):
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(12):
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"t{i}", "name": "read", "input": {}}
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{i}",
                            "content": "y" * 2000,
                        }
                    ],
                }
            )
        wrencode.auto_compact(msgs, None)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[:2], ["user", "assistant"])
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])))  # alternates
        self.assertEqual(
            msgs[1]["content"][0]["id"], msgs[2]["content"][0]["tool_use_id"]
        )

    def test_request_survives_repeated_compaction(self):
        msgs = self.openai_history(20)
        wrencode.auto_compact(msgs, None)
        msgs.extend(self.openai_history(20)[1:])
        wrencode.auto_compact(msgs, None)
        self.assertEqual(msgs[0]["content"].count("build the thing"), 1)
        self.assertEqual(wrencode._latest_request(msgs[:1]), "build the thing")

    def test_nothing_to_gain_is_a_noop(self):
        msgs = self.openai_history(20)
        wrencode.auto_compact(msgs, None)
        msgs = msgs[:1] + msgs[-2:]
        before = list(msgs)
        wrencode.auto_compact(msgs, None)
        self.assertEqual(msgs, before)

    def test_summary_failure_drops_history_with_note(self):
        msgs = self.openai_history(20)
        with mock.patch.object(backends, "complete", side_effect=RuntimeError("nope")):
            wrencode.auto_compact(msgs, None)
        self.assertIn("dropped to fit the context window", msgs[0]["content"])
        self.assertIn("build the thing", msgs[0]["content"])

    def test_transcript_defangs_and_cuts_middle(self):
        msgs = [
            {"role": "assistant", "content": '<tool_call>{"tool": "x"}</tool_call>'}
        ]
        msgs += [{"role": "user", "content": f"m{i} " + "z" * 1000} for i in range(50)]
        out = wrencode._transcript(msgs, 5000)
        self.assertNotIn("<tool_call>", out)
        self.assertIn("middle of the conversation omitted", out)
        self.assertIn("m49", out)
        self.assertLess(len(out), 5200)


class TestAutoCompactInLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        self._patches = [
            mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": str(self._tmp)}),
            mock.patch.object(backends, "BACKEND", "ollama"),
            mock.patch.object(backends, "CONTEXT_TOKENS", 4000),
            mock.patch("sys.stdout", io.StringIO()),
        ]
        for p in self._patches:
            p.start()
        (self._tmp / "big.txt").write_text("w" * 3000)

    def tearDown(self):
        import shutil

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_compacts_when_over_threshold(self):
        replies = iter(
            ['<tool_call>{"tool": "read", "args": {"path": "big.txt"}}</tool_call>'] * 6
            + ["done"]
        )
        msgs = [{"role": "user", "content": "read it a lot"}]
        with (
            mock.patch.object(backends, "get_response", lambda *a: next(replies)),
            mock.patch.object(
                wrencode, "auto_compact", wraps=wrencode.auto_compact
            ) as ac,
            mock.patch.object(backends, "complete", return_value="S"),
        ):
            reason = wrencode.run_agent_turn(msgs, "sys", None)
        self.assertEqual(reason, "done")
        self.assertGreaterEqual(ac.call_count, 1)
        self.assertTrue(msgs[0]["content"].startswith(wrencode._COMPACTION_NOTE))

    def test_context_error_compacts_and_retries_once(self):
        calls = []

        def get_response(*a):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError(
                    "HTTP 400: prompt is too long: 210000 tokens > 200000 maximum"
                )
            return "ok"

        msgs = [{"role": "user", "content": "hi"}]
        with (
            mock.patch.object(backends, "get_response", get_response),
            mock.patch.object(wrencode, "auto_compact") as ac,
        ):
            self.assertEqual(wrencode.run_agent_turn(msgs, "sys", None), "done")
        self.assertEqual((len(calls), ac.call_count), (2, 1))

    def test_repeated_context_error_raises(self):
        def get_response(*a):
            raise RuntimeError(
                "HTTP 400: This model's maximum context length is 8192 tokens"
            )

        with (
            mock.patch.object(backends, "get_response", get_response),
            mock.patch.object(wrencode, "auto_compact"),
            self.assertRaisesRegex(Exception, "maximum context length"),
        ):
            wrencode.run_agent_turn([{"role": "user", "content": "hi"}], "sys", None)

    def test_other_errors_not_retried(self):
        def get_response(*a):
            raise RuntimeError("HTTP 401: bad key")

        with (
            mock.patch.object(backends, "get_response", get_response),
            mock.patch.object(wrencode, "auto_compact") as ac,
            self.assertRaisesRegex(Exception, "HTTP 401"),
        ):
            wrencode.run_agent_turn([{"role": "user", "content": "hi"}], "sys", None)
        ac.assert_not_called()

    def test_disabled_with_zero(self):
        replies = iter(
            ['<tool_call>{"tool": "read", "args": {"path": "big.txt"}}</tool_call>'] * 6
            + ["done"]
        )
        with (
            mock.patch.object(wrencode, "COMPACT_AT", 0.0),
            mock.patch.object(backends, "get_response", lambda *a: next(replies)),
            mock.patch.object(wrencode, "auto_compact") as ac,
        ):
            wrencode.run_agent_turn([{"role": "user", "content": "x"}], "sys", None)
        ac.assert_not_called()


class TestHttpRetry(unittest.TestCase):
    def http_error(self, code):
        import urllib.error

        return urllib.error.HTTPError("u", code, "x", Message(), io.BytesIO(b"busy"))

    def test_retries_server_errors_then_succeeds(self):
        ok = mock.MagicMock()
        ok.__enter__.return_value = io.BytesIO(b'{"ok": 1}')
        with (
            mock.patch(
                "urllib.request.urlopen", side_effect=[self.http_error(503), ok]
            ) as op,
            mock.patch("time.sleep") as sleep,
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(backends._http_post_raw("http://x", b"{}", {}), {"ok": 1})
        self.assertEqual(op.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertEqual(op.call_args.kwargs["timeout"], backends.HTTP_TIMEOUT)

    def test_gives_up_after_retries(self):
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=lambda *a, **k: (_ for _ in ()).throw(self.http_error(429)),
            ) as op,
            mock.patch("time.sleep"),
            mock.patch("sys.stderr", io.StringIO()),
            self.assertRaisesRegex(Exception, "HTTP 429: busy"),
        ):
            backends._http_post_raw("http://x", b"{}", {})
        self.assertEqual(op.call_count, backends.HTTP_RETRIES + 1)

    def ok_response(self):
        ok = mock.MagicMock()
        ok.__enter__.return_value = io.BytesIO(b'{"ok": 1}')
        return ok

    def test_network_errors_are_retried(self):
        import http.client
        import urllib.error

        for err in (
            http.client.RemoteDisconnected(
                "Remote end closed connection without response"
            ),
            TimeoutError("The read operation timed out"),
            urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
            ConnectionResetError(54, "Connection reset by peer"),
        ):
            with (
                mock.patch(
                    "urllib.request.urlopen", side_effect=[err, self.ok_response()]
                ) as op,
                mock.patch("time.sleep") as sleep,
                mock.patch("sys.stderr", io.StringIO()) as stderr,
            ):
                self.assertEqual(
                    backends._http_post_raw("http://x", b"{}", {}), {"ok": 1}
                )
            self.assertEqual(op.call_count, 2, err)
            sleep.assert_called_once_with(2)
            self.assertIn("Network error", stderr.getvalue())

    def test_network_error_raised_after_retries(self):
        with (
            mock.patch(
                "urllib.request.urlopen", side_effect=TimeoutError("timed out")
            ) as op,
            mock.patch("time.sleep") as sleep,
            mock.patch("sys.stderr", io.StringIO()),
            self.assertRaisesRegex(TimeoutError, "timed out"),
        ):
            backends._http_post_raw("http://x", b"{}", {})
        self.assertEqual(op.call_count, backends.HTTP_RETRIES + 1)
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list], [2, 4][: backends.HTTP_RETRIES]
        )

    def test_client_errors_not_retried(self):
        with (
            mock.patch(
                "urllib.request.urlopen", side_effect=self.http_error(400)
            ) as op,
            self.assertRaisesRegex(Exception, "HTTP 400"),
        ):
            backends._http_post_raw("http://x", b"{}", {})
        self.assertEqual(op.call_count, 1)


class TestTruncationRecovery(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(backends, "BACKEND", "nanogpt"),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    @staticmethod
    def reply(content, finish):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish,
                    }
                ]
            }
        )

    def test_nudges_after_empty_truncation(self):
        replies = iter([self.reply("", "length"), self.reply("All done.", "stop")])
        msgs = [{"role": "user", "content": "build it"}]
        with mock.patch.object(backends, "get_response", lambda *a: next(replies)):
            self.assertEqual(wrencode.run_agent_turn(msgs, "sys", None), "done")
        self.assertEqual(
            [m["role"] for m in msgs], ["user", "assistant", "user", "assistant"]
        )
        self.assertEqual(msgs[2]["content"], wrencode.TRUNCATION_NUDGE)
        self.assertEqual(msgs[-1]["content"], "All done.")
        self.assertEqual(backends._to_openai_messages(msgs), msgs)

    def test_gives_up_after_repeated_truncation(self):
        with mock.patch.object(
            backends, "get_response", lambda *a: self.reply("", "length")
        ) as _:
            msgs = [{"role": "user", "content": "x"}]
            reason = wrencode.run_agent_turn(msgs, "sys", None)
        self.assertEqual(reason, "max_tokens")
        self.assertEqual(
            msgs.count({"role": "user", "content": wrencode.TRUNCATION_NUDGE}),
            wrencode.MAX_TRUNCATION_RETRIES,
        )

    def test_normal_stop_unaffected(self):
        with mock.patch.object(
            backends, "get_response", lambda *a: self.reply("hi", "stop")
        ):
            msgs = [{"role": "user", "content": "x"}]
            self.assertEqual(wrencode.run_agent_turn(msgs, "sys", None), "done")
        self.assertEqual(len(msgs), 2)


class TestValidateJson(unittest.TestCase):
    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "verdict": {"enum": ["pass", "fail"]},
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "files": {
                "type": "array",
                "items": {"type": "string", "pattern": r"\.py$"},
            },
            "note": {"type": ["string", "null"], "maxLength": 5},
        },
        "required": ["verdict", "score"],
        "additionalProperties": False,
    }

    def test_valid(self):
        ok = {"verdict": "pass", "score": 7, "files": ["a.py"], "note": None}
        self.assertEqual(wrencode.validate_json(ok, self.SCHEMA), [])

    def test_errors_have_paths(self):
        bad = {
            "verdict": "maybe",
            "score": 11.5,
            "files": ["a.txt"],
            "note": "toolong",
            "x": 1,
        }
        errs = wrencode.validate_json(bad, self.SCHEMA)
        joined = "\n".join(errs)
        for frag in (
            "$.verdict: must be one of",
            "$.score: expected integer",
            "$.files[0]: doesn't match pattern",
            "$.note: longer than 5",
            "unexpected property 'x'",
        ):
            self.assertIn(frag, joined)

    def test_missing_required_and_bool_is_not_int(self):
        errs = wrencode.validate_json({"verdict": "pass", "score": True}, self.SCHEMA)
        self.assertTrue(any("$.score: expected integer" in e for e in errs))
        self.assertIn(
            "$: missing required property 'score'",
            wrencode.validate_json({"verdict": "fail"}, self.SCHEMA),
        )

    def test_combinators(self):
        s = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        self.assertEqual(wrencode.validate_json(3, s), [])
        self.assertTrue(wrencode.validate_json(3.5, s))
        one = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
        self.assertTrue(wrencode.validate_json(3, one))  # matches both


class TestStructuredOutput(unittest.TestCase):
    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "bugs": {"type": "integer"},
            "files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["bugs", "files"],
    }

    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        self._patches = [
            mock.patch.dict(os.environ, {"WRENCODE_WORKSPACE": str(self._tmp)}),
            mock.patch.object(configure, "resolve_configuration", lambda: None),
            mock.patch.object(backends, "load_model", lambda: None),
            mock.patch.object(ui, "HEADLESS", False),
            mock.patch.object(wrencode, "_OUTPUT_SCHEMA", None),
            mock.patch.object(backends, "BACKEND", "ollama"),
        ]
        for p in self._patches:
            p.start()
        os.environ.pop("WRENCODE_AUTO_APPROVE", None)

    def tearDown(self):
        import shutil

        for p in self._patches:
            p.stop()
        wrencode._STRUCTURED_RESULT.clear()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def run_headless(self, replies, schema, fmt="json"):
        replies = iter(replies)
        seen = []

        def get_response(messages, system_prompt, mlx_state, tools=None):
            seen.append(system_prompt)
            return next(replies)

        out = io.StringIO()
        with (
            mock.patch.object(backends, "get_response", get_response),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            code = wrencode.run_headless("count bugs", fmt, 0, schema)
        return code, out.getvalue(), seen

    @staticmethod
    def call(args):
        return (
            f'<tool_call>{{"tool": "respond", "args": {json.dumps(args)}}}</tool_call>'
        )

    def test_valid_answer_ends_run(self):
        code, out, prompts = self.run_headless(
            [self.call({"bugs": 2, "files": ["a.py"]}), "should not be asked"],
            self.SCHEMA,
        )
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["structured_output"], {"bugs": 2, "files": ["a.py"]})
        self.assertEqual(len(prompts), 1)
        self.assertIn('"bugs"', prompts[0])  # schema shown in the prompt

    def test_invalid_answer_gets_errors_then_retries(self):
        code, out, _ = self.run_headless(
            [self.call({"bugs": "two"}), self.call({"bugs": 2, "files": []})],
            self.SCHEMA,
        )
        self.assertEqual(
            (code, json.loads(out)["structured_output"]), (0, {"bugs": 2, "files": []})
        )

    def test_text_mode_prints_the_json(self):
        code, out, _ = self.run_headless(
            [self.call({"bugs": 0, "files": []})], self.SCHEMA, "text"
        )
        self.assertEqual((code, json.loads(out)), (0, {"bugs": 0, "files": []}))

    def test_plain_reply_is_nudged_then_fails(self):
        code, out, _ = self.run_headless(
            ["two bugs", "still text", "and again"], self.SCHEMA
        )
        data = json.loads(out)
        self.assertEqual((code, data["stop_reason"]), (1, "no_structured_output"))
        self.assertIsNone(data["structured_output"])

    def test_non_object_schema_is_wrapped(self):
        _code, out, _ = self.run_headless(
            [self.call({"value": ["x", "y"]})],
            {"type": "array", "items": {"type": "string"}},
        )
        self.assertEqual(json.loads(out)["structured_output"], ["x", "y"])

    def test_native_tool_schema_and_subagents(self):
        with mock.patch.object(wrencode, "_OUTPUT_SCHEMA", self.SCHEMA):
            names = [
                t["function"]["name"]
                for t in backends._build_tool_schemas("openai", wrencode.tool_specs())
            ]
            self.assertIn("respond", names)
            with mock.patch.object(ui, "_subagent_depth", return_value=1):
                names = [
                    t["function"]["name"]
                    for t in backends._build_tool_schemas(
                        "openai", wrencode.tool_specs()
                    )
                ]
                self.assertNotIn("respond", names)
                self.assertIn(
                    "only available", wrencode.respond({"bugs": 1, "files": []})
                )
        self.assertNotIn(
            "respond",
            [
                t["name"]
                for t in backends._build_tool_schemas(
                    "anthropic", wrencode.tool_specs()
                )
            ],
        )

    def test_cli_schema_from_file_and_inline(self):
        path = self._tmp / "s.json"
        path.write_text(json.dumps(self.SCHEMA))
        for value in (str(path), json.dumps(self.SCHEMA)):
            with (
                mock.patch.object(
                    sys, "argv", ["wrencode", "-p", "x", "--json-schema", value]
                ),
                mock.patch.object(wrencode, "run_headless", return_value=0) as run,
                self.assertRaises(SystemExit),
            ):
                wrencode.main()
            run.assert_called_once_with("x", "text", 0, self.SCHEMA, "")

    def test_cli_bad_schema(self):
        with (
            mock.patch.object(
                sys, "argv", ["wrencode", "-p", "x", "--json-schema", "{nope"]
            ),
            mock.patch("sys.stdout", io.StringIO()),
            self.assertRaises(SystemExit) as cm,
        ):
            wrencode.main()
        self.assertEqual(cm.exception.code, 2)


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
        import shutil

        self._out.stop()
        self._env.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def edit(self, old, new):
        return wrencode.edit({"path": "cart.py", "old": old, "new": new})

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
        import shutil

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_interleaved_identical_failures_hint_then_stop(self):
        bad = '<tool_call>{"tool": "edit", "args": {"path": "f.py", "old": "y = 2", "new": "y = 3"}}</tool_call>'
        ok = '<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>'
        replies = iter([bad, ok] * 10)
        msgs = [{"role": "user", "content": "go"}]
        with mock.patch.object(backends, "get_response", lambda *a: next(replies)):
            reason = wrencode.run_agent_turn(msgs, "sys", None)
        self.assertEqual(reason, "tool_errors")
        results = [
            b["content"]
            for m in msgs
            if isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_result"
        ]
        edits = [r for r in results if r.startswith("error:")]
        self.assertEqual(len(edits), wrencode.REPEATED_CALL_STOP)
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
            mock.patch.object(wrencode, "TOOL_ERROR_REPEAT_LIMIT", 2),
            mock.patch.object(backends, "get_response", lambda *a: batch),
        ):
            self.assertEqual(wrencode.run_agent_turn(msgs, "sys", None), "tool_errors")
        ids_called = [
            b["id"] for b in msgs[1]["content"] if b.get("type") == "tool_use"
        ]
        ids_answered = [b["tool_use_id"] for b in msgs[2]["content"]]
        self.assertEqual(ids_called, ids_answered)
        self.assertTrue(msgs[2]["content"][-1]["content"].startswith("skipped"))


class TestVerify(unittest.TestCase):
    def setUp(self):
        self._tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        self._patches = [
            mock.patch.dict(
                os.environ,
                {"WRENCODE_WORKSPACE": str(self._tmp), "WRENCODE_AUTO_APPROVE": "1"},
            ),
            mock.patch.object(configure, "resolve_configuration", lambda: None),
            mock.patch.object(backends, "load_model", lambda: None),
            mock.patch.object(ui, "HEADLESS", False),
            mock.patch.object(wrencode, "_OUTPUT_SCHEMA", None),
            mock.patch.object(backends, "BACKEND", "ollama"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        import shutil

        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def headless(self, replies, verify):
        replies = iter(replies)
        prompts = []

        def get_response(messages, *a):
            prompts.append(messages[-1]["content"])
            return next(replies)

        out = io.StringIO()
        with (
            mock.patch.object(backends, "get_response", get_response),
            mock.patch("sys.stdout", out),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            code = wrencode.run_headless("make ok.txt", "json", 0, None, verify)
        return code, json.loads(out.getvalue()), prompts

    def test_false_done_is_sent_back_then_fixed(self):
        write = '<tool_call>{"tool": "write", "args": {"path": "ok.txt", "content": "y"}}</tool_call>'
        code, data, prompts = self.headless(
            ["All done!", write, "Now really done."], "test -f ok.txt"
        )
        self.assertEqual(
            (code, data["verified"], data["stop_reason"]), (0, True, "done")
        )
        self.assertIn("the check `test -f ok.txt` failed (exit code 1)", prompts[1])

    def test_gives_up_after_attempts(self):
        code, data, _ = self.headless(["done"] * 3, "echo nope; exit 3")
        self.assertEqual(
            (code, data["verified"], data["stop_reason"]), (1, False, "verify_failed")
        )
        self.assertIn("exit code 3", data["verify_output"])
        self.assertIn("nope", data["verify_output"])

    def test_passes_first_time(self):
        code, data, prompts = self.headless(["done"], "true")
        self.assertEqual((code, data["verified"], len(prompts)), (0, True, 1))

    def test_no_verify_field_without_flag(self):
        _, data, _ = self.headless(["done"], "")
        self.assertNotIn("verified", data)

    def test_cli_flag(self):
        with (
            mock.patch.object(
                sys, "argv", ["wrencode", "-p", "x", "--verify", "make test"]
            ),
            mock.patch.object(wrencode, "run_headless", return_value=0) as run,
            self.assertRaises(SystemExit),
        ):
            wrencode.main()
        run.assert_called_once_with("x", "text", 0, None, "make test")


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
        import shutil

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
            reason = wrencode.run_agent_turn(msgs, "sys", None)
        return reason, msgs

    @staticmethod
    def openai(content, calls=None):
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        return json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]})

    def test_real_qwen3_call_is_explained(self):
        why = wrencode._garbled_tool_call(QWEN3_GARBLED_CALL, native=True)
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
            "function-calling interface", wrencode._garbled_tool_call(text, native=True)
        )

    def test_xml_unknown_tool(self):
        why = wrencode._garbled_tool_call(
            '<tool_call>{"tool": "fly", "args": {}}</tool_call>', native=False
        )
        self.assertIn("unknown tool", why)

    def test_plain_final_answer_unaffected(self):
        self.assertEqual(
            wrencode._garbled_tool_call("All done, tests pass.", native=True), ""
        )

    def test_hermes_shape_parsed_on_xml_path(self):
        calls = wrencode.parse_tool_calls(
            '<tool_call>{"name": "read", "arguments": {"path": "a.txt"}}</tool_call>'
            '<tool_call>{"name": "glob", "arguments": "{\\"pat\\": \\"*.txt\\"}"}</tool_call>'
        )
        self.assertEqual(
            [(c["name"], c["input"]) for c in calls],
            [("read", {"path": "a.txt"}), ("glob", {"pat": "*.txt"})],
        )


if __name__ == "__main__":
    unittest.main()
