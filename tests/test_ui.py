"""The terminal: markdown rendering, message blocks, prompts, the line editor."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from tests.support import BOLD, RESET, strip_ansi
from wrencode import (
    app,
    backends,
    ui,
)


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
        self.assertIn(ui.CYAN, result)
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
        self.assertIn(ui.PALETTE.keyword, result)  # 'def' is a keyword
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
        self.assertIn(ui.CYAN, result)

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
        self.assertIn(f"{ui.PALETTE.heading}{ui.BOLD}Plan{ui.RESET}", out)
        self.assertIn(f"{ui.DIM}│{ui.RESET} {ui.CODE_TEXT}", out)
        # after a highlighted token the code color comes back, not the prose color
        self.assertIn(f"{ui.PALETTE.number}1{ui.RESET}{ui.CODE_TEXT}", out)
        self.assertIn(f"{ui.PALETTE.comment}# one{ui.RESET}{ui.CODE_TEXT}", out)
        self.assertNotIn("\x1b[48;", out)  # no background color
        spaced = ui.render_markdown("before:\n\n```\nx\n```\n\nafter")
        self.assertNotIn("\n\n\n", spaced)

    def test_search_snippets_fit_one_line(self):
        self.assertEqual(app._snippet("  first line\nsecond"), "first line…")
        self.assertEqual(app._snippet("\n\nonly\n"), "only")
        self.assertEqual(app._snippet("a" * 20, width=8), "aaaaaaaa…")

    def test_print_tool_action_has_no_background(self):
        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            app.print_tool_action("read", {"path": "foo.py"})
        out = buf.getvalue()
        self.assertNotIn("\x1b[48;", out)
        self.assertIn("read", strip_ansi(out).lower())
        self.assertIn("foo.py", strip_ansi(out))
        self.assertNotIn("read read", strip_ansi(out).lower())

    def test_print_tool_action_no_duplicate_write(self):
        import io

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            app.print_tool_action("write", {"path": "/tmp/x.txt", "content": ""})
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
        self.assertIn(ui.PALETTE.accent, out)
        self.assertIn(ui.DIM, out)
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
        with mock.patch.object(app, "save_history"):
            action, _ = app.handle_slash_command("/c", msgs, None)
            self.assertEqual((action, msgs), ("handled", []))
            for cmd in ("/q", "/quit", "/exit"):
                action, _ = app.handle_slash_command(cmd, msgs, None)
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


# ---------------------------------------------------------------------------
# Highlight Code
# ---------------------------------------------------------------------------
class TestHighlightCode(unittest.TestCase):
    """Code is colored the way Baseline colors it."""

    def test_keyword(self):
        result = ui._highlight_code("def foo():")
        self.assertIn(f"{ui.PALETTE.keyword}def", result)
        self.assertIn(f"{ui.PALETTE.name}foo", result)  # a defined name: the text color

    def test_string(self):
        result = ui._highlight_code('x = "hello"')
        self.assertIn(f'{ui.PALETTE.string}"hello"', result)

    def test_comment(self):
        result = ui._highlight_code("x = 1  # comment")
        self.assertIn(f"{ui.PALETTE.comment}# comment", result)

    def test_numbers_constants_and_decorators(self):
        self.assertIn(f"{ui.PALETTE.number}42", ui._highlight_code("return 42"))
        self.assertIn(f"{ui.PALETTE.number}None", ui._highlight_code("x = None"))
        self.assertIn(
            f"{ui.PALETTE.number}@cache", ui._highlight_code("@cache\ndef f(): ...")
        )

    def test_calls_attributes_and_punctuation(self):
        result = ui._highlight_code("os.path.join(a, b)", "<code>")
        self.assertIn(f"{ui.PALETTE.call}path", result)  # an attribute
        self.assertIn(f"{ui.PALETTE.call}join", result)  # a call
        self.assertIn(f"{ui.PALETTE.name}(", result)  # punctuation: the text color
        self.assertIn(f"{ui.RESET}<code>a", result)  # an identifier: the code color

    def test_multiple_tokens(self):
        result = ui._highlight_code('if x == "ok": return True')
        self.assertIn(f"{ui.PALETTE.keyword}if", result)
        self.assertIn(f'{ui.PALETTE.string}"ok"', result)
        self.assertIn(f"{ui.PALETTE.number}True", result)

    def test_no_tokens_unchanged(self):
        code = "x y z"
        result = ui._highlight_code(code)
        self.assertIn("x y z", strip_ansi(result))


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
            mock.patch.object(app, "workspace_root", return_value=tmp),
            mock.patch.object(ui, "_confirm_prompt", return_value="ok") as ask,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            result = app.edit(
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
            app.print_tool_result(text, name)
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
            app.print_tool_action("edit", {"path": "a.py", "old": "x", "new": "y"})
            app.print_tool_action(
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

    def test_baseline_is_the_default_and_zed_files_load(self):
        dark = ui.baseline_palette("dark")
        self.assertEqual(dark.text, "\x1b[38;2;171;178;191m")  # #abb2bf
        self.assertEqual(dark.code, "\x1b[38;2;98;175;239m")  # #62afef: identifiers
        self.assertEqual(dark.keyword, "\x1b[38;2;225;109;118m")  # #e16d76
        self.assertEqual(dark.string, "\x1b[38;2;209;154;102m")  # #d19a66
        self.assertEqual(dark.call, dark.string)  # calls and attributes: orange
        self.assertEqual(dark.number, "\x1b[38;2;198;120;222m")  # #c678de
        self.assertEqual(dark.comment, "\x1b[38;2;92;99;112m")  # #5c6370
        self.assertEqual(dark.muted, dark.comment)
        self.assertEqual(dark.accent, "\x1b[38;2;82;139;255m")  # #528bff
        self.assertEqual(
            (dark.banner_face, dark.name, dark.heading),
            (dark.accent, dark.text, dark.blue),
        )
        light = ui.baseline_palette("light")
        self.assertEqual(light.text, "\x1b[38;2;56;58;66m")  # #383a42
        self.assertEqual(light.keyword, dark.keyword)  # the same syntax colors
        self.assertEqual(ui.baseline_palette(""), dark)  # unknown background: dark
        self.assertEqual(ui.resolve_palette("", ""), (dark, ""))
        self.assertEqual(ui.resolve_palette("", "light"), (light, ""))
        self.assertEqual(ui.resolve_palette("light", "dark"), (light, ""))  # forced
        ansi, err = ui.resolve_palette("ansi", "")
        self.assertEqual((ansi.text, ansi.red, err), ("\x1b[39m", ui._ANSI_RED, ""))
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        theme = tmp / "mine.json"
        theme.write_text(
            json.dumps(
                {
                    "themes": [
                        {
                            "name": "Mine Dark",
                            "appearance": "dark",
                            "style": {
                                "text": "#aabbcc",
                                "terminal.ansi.red": "#ff0000",
                            },
                        },
                        {
                            "name": "Mine Light",
                            "appearance": "light",
                            "style": {
                                "text": "#112233",
                                "syntax": {"keyword": {"color": "#00ff00"}},
                            },
                        },
                    ]
                }
            )
        )
        by_name = ui.load_zed_theme(str(theme), "Mine Light")
        self.assertEqual(by_name.text, "\x1b[38;2;17;34;51m")
        self.assertEqual(by_name.keyword, "\x1b[38;2;0;255;0m")
        self.assertEqual(by_name.red, ui._ANSI_RED)  # not given: the terminal's
        self.assertEqual(ui.load_zed_theme(str(theme)).text, "\x1b[38;2;170;187;204m")
        self.assertEqual(
            ui.load_zed_theme(str(theme), background="light").text, by_name.text
        )
        with self.assertRaisesRegex(ValueError, "no theme called 'Nope'"):
            ui.load_zed_theme(str(theme), "Nope")
        palette, err = ui.resolve_palette(f"{theme}#Mine Light", "")
        self.assertEqual((palette.text, err), (by_name.text, ""))
        palette, err = ui.resolve_palette(str(tmp / "missing.json"), "")
        self.assertIn("could not read the theme", err)
        self.assertEqual(palette.red, ui._ANSI_RED)
        palette, err = ui.resolve_palette("solarized", "")
        self.assertIn("is not a theme", err)
        with mock.patch.object(ui, "PALETTE", dark):
            banner = ui.render_banner(True)
        self.assertIn(dark.accent + "██", banner)
        self.assertIn(dark.muted + "╗", banner)
        with self.assertRaises(ValueError):
            ui.truecolor("#12345")

    def test_theme_detection_and_text_colors(self):
        with mock.patch.dict(
            os.environ, {"WRENCODE_THEME": "light", "COLORFGBG": "15;0"}
        ):
            self.assertEqual(ui.terminal_theme(), "light")
        with mock.patch.dict(os.environ, {"WRENCODE_THEME": "", "COLORFGBG": "15;0"}):
            self.assertEqual(ui.terminal_theme(), "dark")
        with mock.patch.dict(os.environ, {"WRENCODE_THEME": "", "COLORFGBG": "0;15"}):
            self.assertEqual(ui.terminal_theme(), "light")
        with mock.patch.dict(os.environ, {"WRENCODE_THEME": "", "COLORFGBG": ""}):
            self.assertEqual(ui.terminal_theme(), "")  # unknown: no guess
        # Known backgrounds get tints; an unknown one keeps the terminal's own
        # text color, which reads on both (a light grey on white did not).
        self.assertEqual(ui.text_colors("light"), ("\033[38;5;236m", "\033[38;5;94m"))
        self.assertEqual(ui.text_colors("dark"), ("\033[38;5;252m", "\033[38;5;223m"))
        self.assertEqual(ui.text_colors(""), ("\033[39m", "\033[39m"))

    def test_status_line_says_model_price_session_and_history(self):
        store = mock.Mock(mirror=None)
        with (
            mock.patch.object(backends, "BACKEND", "anthropic"),
            mock.patch.object(backends, "MODEL", "claude-sonnet-5-5"),
            mock.patch.object(app, "_STORE", store),
            mock.patch.object(app, "_SESSION_ID", 3),
        ):
            line = strip_ansi(app._status_line([{"role": "user", "content": "a"}] * 2))
        self.assertEqual(
            line,
            "claude-sonnet-5-5 · $2/$10 per MTok · session #3, 2 chats · history in Postgres",
        )
        with (
            mock.patch.object(backends, "BACKEND", "ollama"),
            mock.patch.object(backends, "MODEL", "llama3"),
            mock.patch.object(app, "_STORE", None),
        ):
            self.assertEqual(strip_ansi(app._status_line([])), "llama3")

    def test_picker_falls_back_to_numbers_without_a_terminal(self):
        with (
            mock.patch("sys.stdin.isatty", return_value=False),
            mock.patch("builtins.input", return_value="2"),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(ui.pick_from_list("Pick", ["a", "b"]), 1)


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

    def test_modifier_arrows_and_tilde_keys_are_decoded(self):
        self.assertEqual(self._key(b"\x1b[1;5C"), "right")  # Ctrl+Right
        self.assertEqual(self._key(b"\x1b[1;2D"), "left")  # Shift+Left
        self.assertEqual(self._key(b"\x1b[3~"), "delete")
        self.assertEqual(self._key(b"\x1b[1;3H"), "home")
        self.assertEqual(self._key(b"\x1b[27~"), "esc")

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
