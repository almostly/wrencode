"""Permission rules: what wrencode may do without asking, and what it must never do.

A rule is `tool(pattern)`: the tool is `bash`, `edit` or `write`, the pattern a
glob against the command (bash) or the workspace-relative path (edit, write),
where `*` matches anything and a trailing `:*` means "this prefix". Examples:

    bash(npm test)        exactly that command
    bash(git *)           any git command
    bash(pytest:*)        anything starting with pytest
    edit(src/*)           any file under src/
    write(.env)           that file
    mcp(github:*)         any tool of the MCP server named github
    fetch(docs.python.org/*)  any page on that host

Deny rules win over allow rules, over "allow all for this session" and over
--yes. Rules come from two files: user-wide `permissions.json` in the config
directory, which holds the person's own rules (`allow`/`deny`) and their rules
for one project (`projects[root]`, where `s` at a prompt saves), and the
project's `.wrencode/permissions.json`, which can be shared. A project's allow
rules are part of the repository, so they only take effect once the person has
seen and accepted them (their hash is then recorded under the user file's
`trusted`); its deny rules apply regardless, since they can only hold the
agent back. Editing the shared file from wrencode never grants that trust.

A bash rule is matched against every command of a command line separately
(`git status && curl x | sh` is three commands): a deny rule refuses when any
of them matches, an allow rule applies only when all of them do. A command
line with command substitution (`$(...)`, backticks, `<(...)`) is only ever
allowed by `bash(*)` or by a rule spelling it out exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass

TOOLS = ("bash", "edit", "write", "mcp", "fetch")
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
        return tool == self.tool and _glob(self.pattern, subject, tool)


def parse(text: str) -> tuple[str, str]:
    """Split `tool(pattern)` into its parts; raises ValueError for anything else."""
    m = _RULE.match(text.strip())
    if not m or m.group("tool") not in TOOLS or not m.group("pattern").strip():
        raise ValueError(
            f"not a rule: {text!r} (expected tool(pattern) with tool "
            f"{', '.join(TOOLS[:-1])} or {TOOLS[-1]})"
        )
    return m.group("tool"), m.group("pattern").strip()


def _glob(pattern: str, subject: str, tool: str = "") -> bool:
    subject = subject.strip()
    if pattern.endswith(":*"):  # a prefix; for a command, whole words of it
        prefix = pattern[:-2].strip()
        if tool == "bash":
            return subject == prefix or subject.startswith(prefix + " ")
        return subject.startswith(prefix)
    regex = "".join(
        ".*" if ch == "*" else "." if ch == "?" else re.escape(ch) for ch in pattern
    )
    return re.fullmatch(regex, subject, re.DOTALL) is not None


_OPERATORS = ("||", "&&", ";", "|", "&", "\n")
_SUBSTITUTION = ("$(", "`", "<(", ">(")


def commands(line: str) -> list[str]:
    """The separate commands of a shell command line: split at `;`, `&&`, `||`,
    `|`, `&` and newlines outside quotes. An empty list means the line can't be
    read command by command (it substitutes a command's output somewhere)."""
    if any(mark in line for mark in _SUBSTITUTION):
        return []
    out: list[str] = []
    cur: list[str] = []
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < len(line):
                cur.append(line[i + 1])
                i += 1
            elif ch == quote:
                quote = ""
            i += 1
            continue
        if ch == "\\" and i + 1 < len(line):
            cur.append(ch + line[i + 1])
            i += 2
            continue
        if ch in ("'", '"'):
            quote = ch
            cur.append(ch)
            i += 1
            continue
        op = next((o for o in _OPERATORS if line.startswith(o, i)), None)
        if op:
            if "".join(cur).strip():
                out.append("".join(cur).strip())
            cur = []
            i += len(op)
            continue
        cur.append(ch)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def suggest(tool: str, subject: str) -> str:
    """The rule offered at the prompt for "always allow this": the command's
    first two words as a prefix for bash (a command line of several commands,
    spelled out exactly), the file's directory for edits (the file itself when
    it sits at the project root)."""
    if tool == "bash":
        parts = commands(subject)
        if len(parts) != 1:
            return f"bash({' '.join(subject.split())})"
        words = parts[0].split()
        if len(words) <= 2:
            return f"bash({' '.join(words)})"
        return f"bash({' '.join(words[:2])}:*)"
    if tool == "mcp":  # server:tool -> that tool
        return f"mcp({subject.strip()})"
    if tool == "fetch":  # host/path -> that host
        return f"fetch({subject.split('/', 1)[0]}/*)"
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
        try:
            self._project_bytes = self.project_file.read_bytes()
        except OSError:
            self._project_bytes = b""
        try:
            data = json.loads(self._project_bytes)
        except ValueError:
            data = {}
        self._project = data if isinstance(data, dict) else {}

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

    def _local(self) -> dict:
        """The person's rules for this project, kept in their own file."""
        projects = self._user.get("projects") or {}
        data = (
            projects.get(str(self.project_root)) if isinstance(projects, dict) else {}
        )
        return data if isinstance(data, dict) else {}

    def rules(self) -> list[Rule]:
        """Every rule: user, then local (the person's, for this project), then
        the project's shared file, then the session."""
        return (
            self._rules(self._user, "user")
            + self._rules(self._local(), "local")
            + self._rules(self._project, "project")
            + list(self.session)
        )

    # ---- project trust -----------------------------------------------------
    def _project_hash(self) -> str:
        """The hash of the file as it was read (what was shown is what is trusted)."""
        return (
            hashlib.sha256(self._project_bytes).hexdigest()
            if self._project_bytes
            else ""
        )

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
        trusted = self.project_trusted()
        allows = [
            r
            for r in rules
            if r.effect == "allow" and (r.source != "project" or trusted)
        ]
        parts = commands(subject) if tool == "bash" else [subject]
        for r in rules:
            if r.effect == "deny" and any(
                r.matches(tool, p) for p in [subject, *parts]
            ):
                return r
        if not parts:  # command substitution: only an exact rule or bash(*) allows it
            allows = [r for r in allows if r.pattern == "*" or "*" not in r.pattern]
            parts = [subject]
        first: Rule | None = None
        for p in parts:
            r = next((r for r in allows if r.matches(tool, p)), None)
            if r is None:
                return None
            first = first or r
        return first

    # ---- editing -----------------------------------------------------------
    def add(self, text: str, effect: str, scope: str = "local") -> Rule:
        """Add a rule and return it. `scope` is where it goes: "user" (every
        project), "local" (this project, in the user's file), "project" (the
        shared file in the repository) or "session"."""
        tool, pattern = parse(text)
        rule = Rule(tool, pattern, effect, scope)
        if scope == "session":
            if rule not in self.session:
                self.session.append(rule)
            return rule
        if scope == "project":
            # The person's own edit keeps an accepted file accepted; it never
            # accepts rules the repository put there.
            was_trusted = self.project_trusted()
            self._append(self._project, effect, rule)
            self._write(self.project_file, self._project)
            self._project_bytes = self.project_file.read_bytes()
            if was_trusted:
                self.trust_project()
            return rule
        if scope == "local":
            projects = dict(self._user.get("projects") or {})
            local = dict(self._local())
            self._append(local, effect, rule)
            projects[str(self.project_root)] = local
            self._user["projects"] = projects
        else:
            self._append(self._user, effect, rule)
        self._write(self.user_file, self._user)
        return rule

    @staticmethod
    def _append(data: dict, effect: str, rule: Rule) -> None:
        entries = [str(x) for x in data.get(effect) or []]
        if str(rule) not in entries:
            entries.append(str(rule))
        data[effect] = entries

    def remove(self, text: str) -> int:
        """Remove every rule written as `text` from the files and the session."""
        tool, pattern = parse(text)
        wanted = f"{tool}({pattern})"

        def strip(data: dict) -> int:
            n = 0
            for effect in ("allow", "deny"):
                before = [str(x) for x in data.get(effect) or []]
                after = [x for x in before if x != wanted]
                if len(after) != len(before):
                    data[effect] = after
                    n += len(before) - len(after)
            return n

        removed = 0
        local = self._local()
        n = strip(self._user) + strip(local)
        if n:
            if local:
                projects = dict(self._user.get("projects") or {})
                projects[str(self.project_root)] = local
                self._user["projects"] = projects
            self._write(self.user_file, self._user)
            removed += n
        was_trusted = self.project_trusted()
        if n := strip(self._project):
            self._write(self.project_file, self._project)
            self._project_bytes = self.project_file.read_bytes()
            if was_trusted:  # fewer allow rules than were accepted: still accepted
                self.trust_project()
            removed += n
        kept = [r for r in self.session if str(r) != wanted]
        removed += len(self.session) - len(kept)
        self.session = kept
        return removed


# The rules in effect, set by wrencode at startup (None: no rules, always ask).
ACTIVE: Permissions | None = None
