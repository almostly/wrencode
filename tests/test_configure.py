"""Choosing a backend and model, and applying the configuration."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from wrencode import (
    backends,
    configure,
    ui,
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
