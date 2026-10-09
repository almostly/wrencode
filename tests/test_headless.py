"""Headless mode: -p, JSON output, schemas, --verify."""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from typing import Any, ClassVar
from unittest import mock

from wrencode import (
    app,
    backends,
    configure,
    ui,
)


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
            code = app.run_headless("do it", *args)
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
            code = app.run_headless("x", "json")
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
            code = app.run_headless("x", "json")
        data = json.loads(out.getvalue())
        self.assertEqual((code, data["stop_reason"]), (1, "error"))
        self.assertIn("configuration error", data["error"])

    def test_arg_value(self):
        self.assertEqual(app._arg_value(["-p", "hi"], "-p", "--print"), "hi")
        self.assertEqual(app._arg_value(["--print=hi"], "-p", "--print"), "hi")
        self.assertEqual(app._arg_value(["-p", "-"], "-p"), "-")
        self.assertIsNone(app._arg_value(["-p", "--yes"], "-p"))
        self.assertIsNone(app._arg_value(["-p"], "-p"))

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
            mock.patch.object(app, "run_headless", return_value=0) as run,
            mock.patch.dict(os.environ, {}),
            self.assertRaises(SystemExit) as cm,
        ):
            app.main()
        self.assertEqual(cm.exception.code, 0)
        run.assert_called_once_with("fix it", "json", 3, None, "")

    def test_main_reads_piped_stdin(self):
        stdin = io.StringIO("from a pipe")
        with (
            mock.patch.object(sys, "argv", ["wrencode", "-p"]),
            mock.patch("sys.stdin", stdin),
            mock.patch.object(app, "run_headless", return_value=0) as run,
            self.assertRaises(SystemExit),
        ):
            app.main()
        run.assert_called_once_with("from a pipe", "text", 0, None, "")


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
        self.assertEqual(app.validate_json(ok, self.SCHEMA), [])

    def test_errors_have_paths(self):
        bad = {
            "verdict": "maybe",
            "score": 11.5,
            "files": ["a.txt"],
            "note": "toolong",
            "x": 1,
        }
        errs = app.validate_json(bad, self.SCHEMA)
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
        errs = app.validate_json({"verdict": "pass", "score": True}, self.SCHEMA)
        self.assertTrue(any("$.score: expected integer" in e for e in errs))
        self.assertIn(
            "$: missing required property 'score'",
            app.validate_json({"verdict": "fail"}, self.SCHEMA),
        )

    def test_combinators(self):
        s = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        self.assertEqual(app.validate_json(3, s), [])
        self.assertTrue(app.validate_json(3.5, s))
        one = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
        self.assertTrue(app.validate_json(3, one))  # matches both


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
            mock.patch.object(app, "_OUTPUT_SCHEMA", None),
            mock.patch.object(backends, "BACKEND", "ollama"),
        ]
        for p in self._patches:
            p.start()
        os.environ.pop("WRENCODE_AUTO_APPROVE", None)

    def tearDown(self):

        for p in self._patches:
            p.stop()
        app._STRUCTURED_RESULT.clear()
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
            code = app.run_headless("count bugs", fmt, 0, schema)
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
        with mock.patch.object(app, "_OUTPUT_SCHEMA", self.SCHEMA):
            names = [
                t["function"]["name"]
                for t in backends._build_tool_schemas("openai", app.tool_specs())
            ]
            self.assertIn("respond", names)
            with mock.patch.object(ui, "_subagent_depth", return_value=1):
                names = [
                    t["function"]["name"]
                    for t in backends._build_tool_schemas("openai", app.tool_specs())
                ]
                self.assertNotIn("respond", names)
                self.assertIn("only available", app.respond({"bugs": 1, "files": []}))
        self.assertNotIn(
            "respond",
            [
                t["name"]
                for t in backends._build_tool_schemas("anthropic", app.tool_specs())
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
                mock.patch.object(app, "run_headless", return_value=0) as run,
                self.assertRaises(SystemExit),
            ):
                app.main()
            run.assert_called_once_with("x", "text", 0, self.SCHEMA, "")

    def test_cli_bad_schema(self):
        with (
            mock.patch.object(
                sys, "argv", ["wrencode", "-p", "x", "--json-schema", "{nope"]
            ),
            mock.patch("sys.stdout", io.StringIO()),
            self.assertRaises(SystemExit) as cm,
        ):
            app.main()
        self.assertEqual(cm.exception.code, 2)


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
            mock.patch.object(app, "_OUTPUT_SCHEMA", None),
            mock.patch.object(backends, "BACKEND", "ollama"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):

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
            code = app.run_headless("make ok.txt", "json", 0, None, verify)
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
            mock.patch.object(app, "run_headless", return_value=0) as run,
            self.assertRaises(SystemExit),
        ):
            app.main()
        run.assert_called_once_with("x", "text", 0, None, "make test")
