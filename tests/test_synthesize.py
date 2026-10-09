"""The synthesize subcommand."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from wrencode import (
    backends,
    synthesize,
)


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
            '"files_touched": ["app.py"], "open_questions": []}'
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
