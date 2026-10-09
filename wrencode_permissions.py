"""Permission rules: what wrencode may do without asking, and what it must never do.

A rule is `tool(pattern)`: the tool is `bash`, `edit` or `write`, the pattern a
glob against the command (bash) or the workspace-relative path (edit, write),
where `*` matches anything and a trailing `:*` means "this prefix". Examples:

    bash(npm test)        exactly that command
    bash(git *)           any git command
    bash(pytest:*)        anything starting with pytest
    edit(src/*)           any file under src/
    write(.env)           that file

Deny rules win over allow rules, over "allow all for this session" and over
--yes. Allow rules come from two files, user-wide `permissions.json` in the
config directory and the project's `.wrencode/permissions.json`. A project's
allow rules are part of the repository, so they only take effect once the
person has seen and accepted them (their hash is then recorded under the
user file's `trusted`); its deny rules apply regardless, since they can only
hold the agent back.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass

TOOLS = ("bash", "edit", "write")
PROJECT_FILE = pathlib.Path(".wrencode") / "permissions.json"
_RULE = re.compile(r"^(?P<tool>[a-z]+)\((?P<pattern>.*)\)$", re.DOTALL)


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str
    effect: str  # "allow" or "deny"
    source: str  # "user", "project" or "session"

    def __str__(self) -> str:
        return f"{self.tool}({self.pattern})"

    def matches(self, tool: str, subject: str) -> bool:
        return tool == self.tool and _glob(self.pattern, subject)


def parse(text: str) -> tuple[str, str]:
    """Split `tool(pattern)` into its parts; raises ValueError for anything else."""
    m = _RULE.match(text.strip())
    if not m or m.group("tool") not in TOOLS or not m.group("pattern").strip():
        raise ValueError(
            f"not a rule: {text!r} (expected tool(pattern) with tool bash, edit or write)"
        )
    return m.group("tool"), m.group("pattern").strip()


def _glob(pattern: str, subject: str) -> bool:
    subject = subject.strip()
    if pattern.endswith(":*"):
        return subject.startswith(pattern[:-2].strip())
    regex = "".join(
        ".*" if ch == "*" else "." if ch == "?" else re.escape(ch) for ch in pattern
    )
    return re.fullmatch(regex, subject, re.DOTALL) is not None


def suggest(tool: str, subject: str) -> str:
    """The rule offered at the prompt for "always allow this": the command's
    first two words as a prefix for bash, the file's directory for edits (the
    file itself when it sits at the project root)."""
    if tool == "bash":
        words = subject.strip().split()
        if len(words) <= 2:
            return f"bash({' '.join(words)})"
        return f"bash({' '.join(words[:2])}:*)"
    path = pathlib.PurePosixPath(subject.strip())
    if str(path.parent) == ".":  # a file at the root: just that file
        return f"{tool}({path})"
    return f"{tool}({path.parent}/*)"


class Permissions:
    """The rules in effect for one project, loaded from both files."""

    def __init__(self, user_file: pathlib.Path, project_root: pathlib.Path) -> None:
        self.user_file = user_file
        self.project_root = project_root
        self.project_file = project_root / PROJECT_FILE
        self.session: list[Rule] = []
        self.reload()

    # ---- files -------------------------------------------------------------
    @staticmethod
    def _read(path: pathlib.Path) -> dict:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, path: pathlib.Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        if path == self.user_file:
            os.chmod(path, 0o600)

    def reload(self) -> None:
        self._user = self._read(self.user_file)
        self._project = self._read(self.project_file)

    @staticmethod
    def _rules(data: dict, source: str) -> list[Rule]:
        out: list[Rule] = []
        for effect in ("allow", "deny"):
            for text in data.get(effect) or []:
                try:
                    tool, pattern = parse(str(text))
                except ValueError:
                    continue
                out.append(Rule(tool, pattern, effect, source))
        return out

    def rules(self) -> list[Rule]:
        """Every rule, user then project then session."""
        return (
            self._rules(self._user, "user")
            + self._rules(self._project, "project")
            + list(self.session)
        )

    # ---- project trust -----------------------------------------------------
    def _project_hash(self) -> str:
        try:
            return hashlib.sha256(self.project_file.read_bytes()).hexdigest()
        except OSError:
            return ""

    def project_allow_rules(self) -> list[Rule]:
        return [r for r in self._rules(self._project, "project") if r.effect == "allow"]

    def project_trusted(self) -> bool:
        """Whether the project file's allow rules were accepted as they stand now."""
        if not self.project_allow_rules():
            return True
        trusted = self._user.get("trusted") or {}
        return trusted.get(str(self.project_root)) == self._project_hash()

    def trust_project(self) -> None:
        trusted = dict(self._user.get("trusted") or {})
        trusted[str(self.project_root)] = self._project_hash()
        self._user["trusted"] = trusted
        self._write(self.user_file, self._user)

    # ---- decisions ---------------------------------------------------------
    def check(self, tool: str, subject: str) -> Rule | None:
        """The rule that decides `tool` on `subject`: a deny from anywhere first,
        then an allow from the user file, a trusted project file or the session."""
        rules = self.rules()
        for r in rules:
            if r.effect == "deny" and r.matches(tool, subject):
                return r
        trusted = self.project_trusted()
        for r in rules:
            if r.effect == "allow" and r.matches(tool, subject):
                if r.source == "project" and not trusted:
                    continue
                return r
        return None

    # ---- editing -----------------------------------------------------------
    def add(self, text: str, effect: str, scope: str = "project") -> Rule:
        """Add a rule to the user or project file (or the session) and return it."""
        tool, pattern = parse(text)
        rule = Rule(tool, pattern, effect, scope)
        if scope == "session":
            if rule not in self.session:
                self.session.append(rule)
            return rule
        path = self.user_file if scope == "user" else self.project_file
        data = self._user if scope == "user" else self._project
        entries = [str(x) for x in data.get(effect) or []]
        if str(rule) not in entries:
            entries.append(str(rule))
        data[effect] = entries
        self._write(path, data)
        if scope == "project":  # the person wrote it: the file is theirs as it stands
            self.trust_project()
        return rule

    def remove(self, text: str) -> int:
        """Remove every rule written as `text` from the files and the session."""
        tool, pattern = parse(text)
        wanted = f"{tool}({pattern})"
        removed = 0
        for data, path in (
            (self._user, self.user_file),
            (self._project, self.project_file),
        ):
            changed = False
            for effect in ("allow", "deny"):
                before = [str(x) for x in data.get(effect) or []]
                after = [x for x in before if x != wanted]
                if len(after) != len(before):
                    data[effect] = after
                    removed += len(before) - len(after)
                    changed = True
            if changed:
                self._write(path, data)
                if path == self.project_file:
                    self.trust_project()
        kept = [r for r in self.session if str(r) != wanted]
        removed += len(self.session) - len(kept)
        self.session = kept
        return removed


# The rules in effect, set by wrencode at startup (None: no rules, always ask).
ACTIVE: Permissions | None = None
