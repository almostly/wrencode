"""Talking to models: request shapes, usage and pricing, retries, each backend."""

from __future__ import annotations

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
import unittest
from email.message import Message
from typing import Any
from unittest import mock

from tests.support import strip_ansi
from wrencode import (
    app,
    backends,
    configure,
    history,
    ui,
)


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
                    app.tool_specs(),
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
                tools=app.tool_specs(),
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
        root = pathlib.Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = str(root / "src")
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, wrencode.app; print(os.environ.get('SSL_CERT_FILE', ''))",
            ],
            cwd=str(root),
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
                    app.tool_specs(),
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
            action, _ = app.handle_slash_command("/usage", [], None)
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
            mock.patch.object(app, "SHOW_USAGE", True),
            mock.patch.object(app, "load_history", return_value=[]),
            mock.patch.object(app, "build_system_prompt", return_value="s" * 400),
            mock.patch.object(app, "find_agents_files", return_value=[]),
            mock.patch.object(history, "open_store", return_value=None),
            mock.patch.object(configure, "resolve_configuration"),
            mock.patch.object(backends, "load_model", return_value=None),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch.object(sys, "argv", ["wrencode"]),
        ):
            app.main()
            banner = sys.stdout.getvalue()
            hint = ui.read_user_input.call_args[0][0]
            shown = hint("fix the bug")
        self.assertIn("$2/$10 per MTok", strip_ansi(banner))
        self.assertTrue(shown.startswith("≈ $"), shown)
        self.assertTrue(shown.endswith(" input"))


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
        self.assertNotIn("<tool_call>", app.build_system_prompt())

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


class TestOpenAICompatibleBackend(unittest.TestCase):
    """Drive the backend against a stub OpenAI-compatible server (like vLLM/llama.cpp)."""

    def setUp(self):
        import http.server

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
            code = app.run_headless("find text files", "json")
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
