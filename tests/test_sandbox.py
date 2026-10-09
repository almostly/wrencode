"""The pydantic-monty sandbox."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

from wrencode import (
    app,
    sandbox,
)

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
            app.normalize_tool_args("python", {"source": "x = 1"})["code"], "x = 1"
        )
        shown = app.format_tool_action("python", {"code": "x = 1\nprint(x)"})
        self.assertEqual(shown, "python x = 1  (+1 lines)")
        self.assertEqual(app.format_tool_action("python", {"code": "y"}), "python y")


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
