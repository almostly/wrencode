"""The claude-agent-sdk backend."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from wrencode import (
    app,
    backends,
    configure,
    ui,
)
from wrencode import sdk as agent_sdk

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
        agent_sdk._save_agent_sdk_session_id(app.workspace_root(), "sess-1")
        self.assertEqual(
            agent_sdk._load_agent_sdk_session_id(app.workspace_root()), "sess-1"
        )
        agent_sdk._save_agent_sdk_session_id(app.workspace_root(), "")
        self.assertEqual(agent_sdk._load_agent_sdk_session_id(app.workspace_root()), "")

    def test_clear_resets_sdk_session(self):
        session = mock.MagicMock()
        with (
            mock.patch.object(backends, "BACKEND", "claude-agent-sdk"),
            mock.patch.object(app, "agent_sdk_session", return_value=session),
            mock.patch.object(app, "save_history"),
        ):
            action, _ = app.handle_slash_command("/clear", [], None)
            app.handle_slash_command("/compact", [], None)
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
            code = app.run_headless("do it", "json")
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
            return agent_sdk.AgentSDKSession(cwd=app.workspace_root())

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
            agent_sdk._load_agent_sdk_session_id(app.workspace_root()), "s1"
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
