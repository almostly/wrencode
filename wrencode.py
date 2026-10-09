#!/usr/bin/env python3
"""WrenCode — a minimal agentic coding assistant inspired by Harold Wren.

A lightweight alternative to Claude Code. This file is the agent loop: the tools,
the system prompt, the turn loop with its subagents and compaction, headless mode
and main(). The wrencode_*.py modules beside it are what it calls: the model
backends, backend/model configuration, the terminal UI, the Claude Agent SDK
backend and the synthesize subcommand.

Supports multiple inference backends: local Apple Silicon via MLX,
HuggingFace Transformers, Anthropic, OpenAI, OpenRouter, NanoGPT, and local proxy.
Provides a tool-calling agent loop with file read/write/edit, glob, grep,
and bash — enough to autonomously navigate and modify a codebase.

Copyright (c) 2026 Almostly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import ast
import contextlib
import difflib
import glob as globlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable

# The PyInstaller single-file binary ships no system CA trust store, so urllib's
# TLS verification fails out of the box ("CERTIFICATE_VERIFY_FAILED") on a clean
# machine. If certifi is bundled (it is in the binary build), point the SSL
# defaults at it before any request. Soft import keeps pip/uvx installs
# dependency-free; setdefault preserves any user-provided override.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# What a project's own .env may set: credentials for the backends, nothing else.
# Anything that steers the agent (BACKEND, WRENCODE_AUTO_APPROVE, a *_BASE_URL,
# WRENCODE_CONFIG_DIR, WRENCODE_WORKSPACE, ...) comes only from the real environment
# or the .env beside this script, so cloning a repository can't reconfigure wrencode.
DOTENV_PROJECT_KEYS = re.compile(r"^[A-Z0-9_]*_API_KEY$|^ANTHROPIC_WORKSPACE_ID$")
_DOTENV_IGNORED: list[
    str
] = []  # names a project .env tried to set; reported at startup


def load_dotenv(path: str, *, trusted: bool) -> list[str]:
    """Load KEY=VALUE lines from a .env file into the environment; real variables win.

    A trusted file (next to this script) may set anything. A project file may only
    set the names DOTENV_PROJECT_KEYS allows; the others are returned, not applied.
    """
    skipped: list[str] = []
    try:
        lines = pathlib.Path(path).read_text().splitlines()
    except OSError:
        return skipped
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key = key.strip()
        with contextlib.suppress(ValueError):  # malformed quoting: raw value
            value = shlex.split(value)[0] if value else value
        if not trusted and not DOTENV_PROJECT_KEYS.match(key):
            skipped.append(key)
            continue
        os.environ.setdefault(key, value)
    return skipped


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"), trusted=True)
if os.getcwd() != _SCRIPT_DIR:
    _DOTENV_IGNORED = load_dotenv(os.path.join(os.getcwd(), ".env"), trusted=False)

# The other modules read environment defaults at import, so they come after .env.
import wrencode_backends as backends
import wrencode_configure as configure
import wrencode_history as history
import wrencode_sandbox as sandbox
import wrencode_sdk as agent_sdk
import wrencode_synthesize as synthesize
import wrencode_ui as ui
from wrencode_ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW

# -----------------------------------------------------------------------------------------------
# Version, limits and per-run state
# -----------------------------------------------------------------------------------------------
WRENCODE_VERSION = "0.3.0"
# Project instruction files, in preference order per directory (see find_agents_files).
AGENTS_FILES = ("AGENTS.md", "CLAUDE.md")
MAX_AGENTS_MD_CHARS = 32_000

# The loaded local model (set in main / run_headless), shared with the task() tool.
# Everything that differs per agent lives in ui._AGENT_LOCAL, because parallel
# subagents run in threads: depth, tag ("1", "2", "1.2"...), the batch's cancel
# event, and last_action (the tool call shown, repeated when approval is needed).
_MLX_STATE: tuple[Any, Any] | None = None
# Set by run_headless(--json-schema): the final answer must come through the
# `respond` tool and match this JSON Schema. _STRUCTURED_RESULT holds it once accepted.
RESPOND_TOOL = "respond"
_OUTPUT_SCHEMA: dict[str, Any] | None = None
_STRUCTURED_RESULT: list[Any] = []
MAX_SUBAGENT_DEPTH = int(os.environ.get("WRENCODE_MAX_SUBAGENT_DEPTH", "2"))
# Task calls made in one reply run this many at a time; 1 runs them in order.
MAX_PARALLEL_SUBAGENTS = int(os.environ.get("WRENCODE_MAX_PARALLEL_SUBAGENTS", "4"))
# Auto-compaction: once the estimated prompt passes COMPACT_AT of the model's
# context window (backends.CONTEXT_TOKENS), older turns are summarized (0 disables it).
COMPACT_AT = float(os.environ.get("WRENCODE_COMPACT_AT", "0.75"))
MAX_READ_BYTES = int(os.environ.get("MAX_READ_BYTES", str(4 * 1024 * 1024)))
MAX_READ_LINES = int(os.environ.get("MAX_READ_LINES", "800"))
GREP_MAX = int(os.environ.get("GREP_MAX_MATCHES", "80"))
BASH_TIMEOUT = int(os.environ.get("BASH_TIMEOUT", "120"))
MAX_OUT = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "48000"))
TOOL_ERROR_REPEAT_LIMIT = int(os.environ.get("TOOL_ERROR_REPEAT_LIMIT", "3"))
_GLOB_SKIP: set[str] = {
    s
    for s in os.environ.get(
        "GLOB_SKIP_DIRS",
        ".git,node_modules,__pycache__,.venv,venv,dist,build,.mypy_cache,.pytest_cache,target",
    ).split(",")
    if s
}


# -----------------------------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------------------------
def workspace_root() -> pathlib.Path:
    """Return the resolved workspace root path from env or cwd."""
    if w := os.environ.get("WRENCODE_WORKSPACE"):
        return pathlib.Path(w).expanduser().resolve()
    return pathlib.Path(os.getcwd()).resolve()


def resolve_tool_path(raw: Any) -> pathlib.Path:
    """Resolve a raw path argument to an absolute Path within the workspace."""
    if not raw or not str(raw).strip():
        raise ValueError("path is required")
    p = pathlib.Path(str(raw).strip()).expanduser()
    root = workspace_root()
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if os.environ.get("WRENCODE_UNRESTRICTED_PATHS", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(
                f"path {raw!r} resolves outside workspace {root} "
                f"(set WRENCODE_UNRESTRICTED_PATHS=1)"
            ) from None
    return p


# -----------------------------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------------------------
def _require_str(args: dict[str, Any], key: str) -> str:
    """Require a non-empty string value from args dict by key."""
    val = args.get(key)
    if not val or not str(val).strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return str(val).strip()


def _optional_int(
    args: dict[str, Any], key: str, default: int | None = None
) -> int | None:
    """Return an optional integer from args dict, or default if absent."""
    val = args.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"'{key}' must be an integer, got {val!r}") from e


# -----------------------------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------------------------
def read(args: dict[str, Any]) -> str:
    """Read a file with line numbers or list a directory."""
    path = resolve_tool_path(_require_str(args, "path"))
    if path.is_dir():
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        return (
            "\n".join(f"  {e.name}{'/' if e.is_dir() else ''}" for e in entries)
            or "(empty)"
        )
    if not path.is_file():
        return f"error: not a file: {path}"
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        return f"error: file too large ({size} bytes, max {MAX_READ_BYTES})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    offset = _optional_int(args, "offset", 0) or 0
    if not (0 <= offset <= len(lines)):
        return f"error: offset {offset} out of range (file has {len(lines)} lines)"
    limit_val = _optional_int(args, "limit")
    cap = min(
        limit_val
        if (args.get("limit") and limit_val is not None)
        else len(lines) - offset,
        MAX_READ_LINES,
    )
    out = "".join(
        f"{offset + i + 1:4}| {line}"
        for i, line in enumerate(lines[offset : offset + cap])
    )
    if offset + cap < len(lines):
        out += f"\n... ({len(lines) - offset - cap} more lines; use offset/limit or raise MAX_READ_LINES)"
    return out


def write(args: dict[str, Any]) -> str:
    """Write content to a file, creating parent directories as needed."""
    path = resolve_tool_path(_require_str(args, "path"))
    if "content" not in args:
        return "error: 'content' is required (model response may have been truncated; raise MAX_TOKENS)"
    content = args["content"]
    approval = ui.confirm("write")
    if approval != "ok":
        return approval
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    return "ok"


def edit(args: dict[str, Any]) -> str:
    """Replace a unique string in a file with a new string."""
    path = resolve_tool_path(_require_str(args, "path"))
    # Don't strip `old`: it must match the file substring exactly, including
    # leading/trailing whitespace. Stripping it (as _require_str does) lets the
    # match start mid-indent while `new` carries its own indentation, which
    # duplicates the original leading whitespace in the result.
    old = str(args.get("old", ""))
    if not old.strip():
        raise ValueError("'old' is required and must be a non-empty string")
    new = str(args.get("new", ""))
    if not path.is_file():
        return f"error: not a file: {path}"
    if path.stat().st_size > MAX_READ_BYTES:
        return f"error: file too large (max {MAX_READ_BYTES} bytes)"
    text = path.read_text(encoding="utf-8", errors="replace")
    note = ""
    if old in text:
        count = text.count(old)
        if not args.get("all") and count > 1:
            return f"error: 'old' appears {count} times (add context to make it unique, or use all=true)"
        updated = (
            text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
        )
    elif (shifted := _reindented_edit(text, old, new)) is not None:
        updated, note = shifted
    else:
        return _not_found_error(text, old) + _elsewhere_hint(path, old)
    if updated == text:
        return "error: edit produced no change"
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            ast.parse(updated)
        except SyntaxError as exc:
            return f"error: edit would make invalid Python: {exc}"
    elif suffix == ".json":
        try:
            json.loads(updated)
        except json.JSONDecodeError as exc:
            return f"error: edit would make invalid JSON: {exc}"
    approval = ui.confirm("edit")
    if approval != "ok":
        return approval
    path.write_text(updated, encoding="utf-8")
    return f"ok ({note})" if note else "ok"


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindented_edit(text: str, old: str, new: str) -> tuple[str, str] | None:
    """Apply an edit whose `old` matches whole lines except for a uniform indent shift.

    Models often drop or add one indentation level when quoting a block (e.g. a
    method quoted at column 0). If exactly one block of lines matches after
    shifting every non-blank line by the same amount, replace it and shift `new`
    the same way. Returns (updated text, note) or None.
    """
    old_lines = old.strip("\n").split("\n")
    lines = text.split("\n")
    n = len(old_lines)
    hits: list[tuple[int, str, str]] = []  # (start line, add, remove)
    for i in range(len(lines) - n + 1):
        add = remove = None
        for fl, ol in zip(lines[i : i + n], old_lines):
            if not ol.strip():
                if fl.strip():
                    break
                continue
            if fl.lstrip() != ol.lstrip():
                break
            fi, oi = _indent(fl), _indent(ol)
            if fi.endswith(oi):  # file is indented deeper than `old`
                shift = (fi[: len(fi) - len(oi)], "")
            elif oi.endswith(fi):  # `old` is indented deeper than the file
                shift = ("", oi[: len(oi) - len(fi)])
            else:
                break
            if add is not None and shift != (add, remove):
                break
            add, remove = shift
        else:
            if add is not None and (add or remove):
                hits.append((i, add, remove or ""))
    if len(hits) != 1:
        return None
    i, add, remove = hits[0]
    shifted = []
    for line in new.strip("\n").split("\n"):
        if not line.strip():
            shifted.append(line)
        elif remove and line.startswith(remove):
            shifted.append(line[len(remove) :])
        else:
            shifted.append(add + line)
    updated = "\n".join(lines[:i] + shifted + lines[i + n :])
    how = f"added {len(add)}" if add else f"removed {len(remove)}"
    return (
        updated,
        f"matched lines {i + 1}-{i + n} after adjusting indentation ({how} chars)",
    )


def _not_found_error(text: str, old: str) -> str:
    """Explain a failed match and show the closest block of lines in the file."""
    msg = "error: 'old' text not found; it must match the file exactly, including indentation."
    old_lines = old.strip("\n").split("\n")
    lines = text.split("\n")
    n = max(1, len(old_lines))
    target = "\n".join(l.strip() for l in old_lines)
    best, best_i = 0.0, -1
    for i in range(max(1, len(lines) - n + 1)):
        window = "\n".join(l.strip() for l in lines[i : i + n])
        sm = difflib.SequenceMatcher(None, target, window)
        if (
            sm.real_quick_ratio() > best
            and sm.quick_ratio() > best
            and (r := sm.ratio()) > best
        ):
            best, best_i = r, i
    if best < 0.5:
        return msg + " Re-read the file and copy the text you want to replace."
    shown = "\n".join(
        f"{j + 1:>5}| {lines[j]}" for j in range(best_i, min(len(lines), best_i + n))
    )
    return f"{msg} Closest match ({best:.0%} similar), lines {best_i + 1}-{best_i + n}:\n{shown}"


def _elsewhere_hint(path: pathlib.Path, old: str) -> str:
    """If old's first line appears in other workspace files, say where.

    Catches edits aimed at the wrong file (e.g. a function that lives in a
    sibling module). Only same-suffix files are scanned, at most 500 of them.
    """
    first = next((l.strip() for l in old.split("\n") if l.strip()), "")
    if len(first) < 8:
        return ""
    root = workspace_root()
    found: list[str] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _GLOB_SKIP]
        for name in filenames:
            f = pathlib.Path(dirpath) / name
            if f.suffix != path.suffix or f == path:
                continue
            scanned += 1
            with contextlib.suppress(OSError):
                for n, line in enumerate(f.read_text(errors="replace").split("\n"), 1):
                    if line.strip() == first:
                        found.append(f"{f.relative_to(root)}:{n}")
                        break
            if scanned >= 500 or len(found) >= 3:
                break
        if scanned >= 500 or len(found) >= 3:
            break
    if not found:
        return ""
    return f"\nNote: `{first}` isn't in {path.name} but appears in {', '.join(found)}; did you mean that file?"


def glob(args: dict[str, Any]) -> str:
    """Find files matching a glob pattern, sorted by modification time."""
    if "pattern" in args and "pat" not in args:
        args["pat"] = args.pop("pattern")
    pat = _require_str(args, "pat")
    base = resolve_tool_path(args.get("path", "."))
    if not base.is_dir():
        return f"error: not a directory: {base}"
    files = [
        f
        for f in globlib.glob(str(base / pat), recursive=True)
        if os.path.isfile(f) and all(p not in _GLOB_SKIP for p in pathlib.Path(f).parts)
    ]
    return "\n".join(sorted(files, key=os.path.getmtime, reverse=True)) or "none"


def grep(args: dict[str, Any]) -> str:
    """Search files for a regex pattern using ripgrep."""
    pat = _require_str(args, "pat")
    target_raw = args.get("path", ".")
    try:
        target = resolve_tool_path(target_raw)
    except (ValueError, OSError, RuntimeError) as exc:  # bad path, or resolve() failed
        return f"error: invalid grep path {target_raw!r}: {exc}"
    if not target.exists():
        return f"error: grep path not found: {target}"
    search_dir = target if target.is_dir() else target.parent
    scope = "." if target.is_dir() else target.name
    rg = shutil.which("rg")
    grep_bin = shutil.which("grep")
    if not rg and not grep_bin:
        return "error: neither ripgrep (rg) nor grep is installed"
    tool = rg or grep_bin
    assert tool is not None  # guaranteed by the check above
    cmd = (
        [tool, "-n", "--color", "never", "--no-heading", "-e", pat, "--", scope]
        if rg
        else [tool, "-R", "-n", "-I", "--", pat, scope]
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(search_dir),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error: grep timed out (90s)"
    if proc.returncode not in (0, 1):
        return f"error: grep failed ({proc.returncode}): {(proc.stderr or '').strip()}"
    raw = proc.stdout.splitlines()
    body = "\n".join(raw[:GREP_MAX]) or "none"
    if len(raw) > GREP_MAX:
        body += f"\n... ({len(raw) - GREP_MAX} more; raise GREP_MAX_MATCHES)"
    return body


def get_response_cancellable(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: tuple[Any, Any] | None,
) -> str:
    """Run get_response in a worker thread so Escape can interrupt blocking calls."""
    if ui._agent_tag():  # a parallel subagent: the batch owner watches for Escape
        ui.check_cancelled()
        return backends.get_response(messages, system_prompt, mlx_state, tool_specs())
    if not sys.stdin.isatty():
        return backends.get_response(messages, system_prompt, mlx_state, tool_specs())

    result: list[str] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(
                backends.get_response(messages, system_prompt, mlx_state, tool_specs())
            )
        except BaseException as exc:  # propagate to caller
            error.append(exc)

    with ui.cancel_watch():
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            ui.check_cancelled()
            t.join(timeout=0.15)
    if error:
        raise error[0]
    return result[0]


def bash(args: dict[str, Any]) -> str:
    """Run a shell command with a timeout, streaming output to the terminal."""
    cmd = _require_str(args, "cmd")
    approval = ui.confirm("run")
    if approval != "ok":
        return approval
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd(),
    )
    output_lines: list[str] = []

    def reader() -> None:
        """Read subprocess stdout line by line and print to terminal."""
        with contextlib.suppress(Exception):
            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line)
                print(f"{DIM}│ {ui.visible(line.rstrip())}{RESET}", flush=True)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.monotonic() + BASH_TIMEOUT
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() > deadline:
            timed_out = True
            proc.kill()
            output_lines.append(f"\n(timed out after {BASH_TIMEOUT}s)\n")
            break
        time.sleep(0.05)
    t.join(timeout=2.0)
    if not timed_out:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)
    if proc.stdout is not None:
        proc.stdout.close()
    return "".join(output_lines).strip() or "(empty)"


def task(args: dict[str, Any]) -> str:
    """Run a subagent: a fresh agent loop over a self-contained subtask.

    The subagent shares the workspace and tool set but has its own (empty)
    message history, so the parent's context only grows by the returned result.
    Recursion is capped by MAX_SUBAGENT_DEPTH. For autonomous use run with
    --yes / WRENCODE_AUTO_APPROVE, else each sub-tool call still asks to confirm.
    """
    depth = ui._subagent_depth()
    if depth >= MAX_SUBAGENT_DEPTH:
        return f"error: max subagent depth ({MAX_SUBAGENT_DEPTH}) reached"
    prompt = _require_str(args, "prompt")
    ui._AGENT_LOCAL.depth = depth + 1
    print(f"{CYAN}↳ subagent:{RESET}{DIM} {prompt[:70]}{RESET}")
    sub: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        run_agent_turn(sub, build_system_prompt(), _MLX_STATE, max_iters=12)
    finally:
        ui._AGENT_LOCAL.depth = depth
    texts = [
        backends.flatten_content(m["content"]) for m in sub if m["role"] == "assistant"
    ]
    print(f"{CYAN}↳ subagent done{RESET}")
    return (texts[-1] if texts else "") or "(subagent produced no text output)"


RESPOND_DESCRIPTION = (
    "Give your final answer. Call this once, when the task is done, with arguments "
    "matching its schema; the answer is only accepted through this tool."
)


def _respond_schema() -> dict[str, Any] | None:
    """Return the respond tool's argument schema, or None when it isn't offered.

    Only the top-level agent gets it. A non-object output schema is wrapped as
    {"value": ...}, since tool arguments must be an object.
    """
    if _OUTPUT_SCHEMA is None or ui._subagent_depth():
        return None
    if _OUTPUT_SCHEMA.get("type") == "object":
        return _OUTPUT_SCHEMA
    return {
        "type": "object",
        "properties": {"value": _OUTPUT_SCHEMA},
        "required": ["value"],
    }


def _known_tool(name: Any) -> bool:
    return name in TOOLS or (name == RESPOND_TOOL and _respond_schema() is not None)


def respond(args: dict[str, Any]) -> str:
    """Record the final structured answer if it matches the output schema."""
    schema = _respond_schema()
    if schema is None or _OUTPUT_SCHEMA is None:
        return (
            "error: respond is only available to the top-level agent with --json-schema"
        )
    errors = validate_json(args, schema)
    if errors:
        listed = "\n".join(f"- {e}" for e in errors[:20])
        return f"error: the answer doesn't match the schema:\n{listed}\nFix these and call respond again."
    _STRUCTURED_RESULT[:] = [
        args if _OUTPUT_SCHEMA.get("type") == "object" else args["value"]
    ]
    return "ok: answer recorded"


_JSON_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _json_type_ok(value: Any, name: str) -> bool:
    if name in {"integer", "number"} and isinstance(value, bool):
        return False
    if name == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _JSON_TYPES.get(name, object))


def validate_json(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Check value against a JSON Schema subset; return readable errors (empty if valid).

    Supports type, enum, const, string length and pattern, numeric bounds,
    object properties/required/additionalProperties, array items and length,
    and anyOf/oneOf/allOf. Unknown keywords are ignored.
    """
    if not isinstance(schema, dict):
        return []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_json_type_ok(value, t) for t in types):
            return [
                f"{path}: expected {' or '.join(types)}, got {type(value).__name__}"
            ]
    errs: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {json.dumps(schema['enum'])}")
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must be {json.dumps(schema['const'])}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: doesn't match pattern {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, bad in (
            ("minimum", lambda v, b: v < b),
            ("maximum", lambda v, b: v > b),
            ("exclusiveMinimum", lambda v, b: v <= b),
            ("exclusiveMaximum", lambda v, b: v >= b),
        ):
            if key in schema and bad(value, schema[key]):
                errs.append(f"{path}: violates {key} {schema[key]}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errs.append(f"{path}: missing required property '{key}'")
        extra = schema.get("additionalProperties", True)
        for key, v in value.items():
            if key in props:
                errs += validate_json(v, props[key], f"{path}.{key}")
            elif extra is False:
                errs.append(f"{path}: unexpected property '{key}'")
            elif isinstance(extra, dict):
                errs += validate_json(v, extra, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                errs += validate_json(v, schema["items"], f"{path}[{i}]")
    if "allOf" in schema:
        for sub in schema["allOf"]:
            errs += validate_json(value, sub, path)
    if "anyOf" in schema and all(
        validate_json(value, sub, path) for sub in schema["anyOf"]
    ):
        errs.append(f"{path}: doesn't match any of the allowed shapes (anyOf)")
    if "oneOf" in schema:
        matches = sum(not validate_json(value, sub, path) for sub in schema["oneOf"])
        if matches != 1:
            errs.append(
                f"{path}: must match exactly one shape in oneOf, matched {matches}"
            )
    return errs


ToolFn = Callable[[dict[str, Any]], str]
ToolEntry = tuple[str, dict[str, str], ToolFn]

TOOLS: dict[str, ToolEntry] = {
    "read": (
        "Read file with line numbers, or list directory",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": ("Run shell command", {"cmd": "string"}, bash),
    "task": (
        (
            "Delegate a self-contained subtask to a fresh subagent (same tools, "
            "own context); returns only its final result. Several task calls in "
            "one reply run in parallel, so batch independent subtasks together"
        ),
        {"prompt": "string"},
        task,
    ),
}


def python(args: dict[str, Any]) -> str:
    """Run a model-written Python snippet in the sandbox (see wrencode_sandbox).

    The snippet can't reach the network, a shell or the environment, and sees the
    workspace read-only, so it runs without an approval prompt. The workspace
    tools are available inside it as functions that return Python values rather
    than tool text: glob() a list of workspace-relative paths, read() the file's
    text, grep() a list of "path:line:text" strings. A tool error is raised.
    """
    code = _require_str(args, "code")
    root = workspace_root()

    def failing(text: str) -> str:
        if text.startswith("error:"):
            raise RuntimeError(text.removeprefix("error:").strip())
        return text

    def read_text(path: str) -> str:
        p = resolve_tool_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        if p.stat().st_size > MAX_READ_BYTES:
            raise ValueError(
                f"file too large ({p.stat().st_size} bytes, max {MAX_READ_BYTES})"
            )
        return p.read_text(encoding="utf-8", errors="replace")

    def glob_paths(pat: str, path: str | None = None) -> list[str]:
        out = failing(glob({"pat": pat, **({"path": path} if path else {})}))
        return [os.path.relpath(f, root) for f in out.splitlines() if out != "none"]

    def grep_lines(pat: str, path: str | None = None) -> list[str]:
        out = failing(grep({"pat": pat, **({"path": path} if path else {})}))
        return [] if out == "none" else out.splitlines()

    functions = {"read": read_text, "glob": glob_paths, "grep": grep_lines}
    return sandbox.run(code, workspace=root, functions=functions)


# An eighth tool when pydantic-monty is installed (pip install 'wrencode[sandbox]').
if sandbox.available():
    TOOLS["python"] = (
        (
            "Run a Python snippet in a sandbox: no network, shell or environment, a "
            "standard-library subset: pathlib and re yes, os.path, os.walk and os.listdir no; the workspace "
            "read-only at /workspace (the working directory, so open('x.py') works). "
            "Functions available: glob(pat) -> list of workspace-relative paths, "
            "read(path) -> the file's text, grep(pat) -> list of 'path:line:text'. "
            "Each call is a fresh interpreter: nothing persists between calls, so do "
            "the whole job in one snippet. Printed output and the trailing "
            "expression's value come back"
        ),
        {"code": "string"},
        python,
    )


def normalize_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize common model arg aliases before dispatching a tool."""
    out = dict(args or {})
    if name == "bash" and not str(out.get("cmd", "")).strip():
        for alias in ("command", "shell", "script", "bash_command"):
            if str(out.get(alias, "")).strip():
                out["cmd"] = str(out[alias]).strip()
                break
    if name == "glob" and "pat" not in out and "pattern" in out:
        out["pat"] = out["pattern"]
    if name == "task" and not str(out.get("prompt", "")).strip():
        for alias in ("description", "subtask"):
            if str(out.get(alias, "")).strip():
                out["prompt"] = str(out[alias]).strip()
                break
    if name == "python" and not str(out.get("code", "")).strip():
        for alias in ("source", "script", "snippet"):
            if str(out.get(alias, "")).strip():
                out["code"] = str(out[alias])
                break
    return out


def _hidden_path_note(path: Any) -> str:
    """A warning for paths under a dot-directory or dotfile: hooks, workflows, rc files."""
    parts = pathlib.PurePath(str(path)).parts
    hidden = any(p.startswith(".") and p not in {".", ".."} for p in parts)
    return "  ⚠ hidden/config path" if hidden else ""


def format_tool_action(name: str, args: dict[str, Any]) -> str:
    """Human-readable summary of what a tool call will do."""
    args = normalize_tool_args(name, args)
    if name == "bash":
        cmd = str(args.get("cmd", "")).strip()
        return f"$ {cmd}" if cmd else "(empty shell command)"
    if name == "read":
        path = args.get("path", "?")
        offset = args.get("offset")
        limit = args.get("limit")
        extra = ""
        if offset is not None or limit is not None:
            extra = f"  offset={offset}, limit={limit}"
        return f"read {path}{extra}"
    if name == "write":
        path = args.get("path", "?")
        content = str(args.get("content", ""))
        lines = content.count("\n") + (1 if content else 0)
        preview = content[:160].replace("\n", "\\n")
        suffix = "..." if len(content) > 160 else ""
        return f"write {path}{_hidden_path_note(path)}  ({lines} lines)\n  {preview}{suffix}"
    if name == "edit":
        path = args.get("path", "?")
        old = str(args.get("old", ""))[:80].replace("\n", "\\n")
        new = str(args.get("new", ""))[:80].replace("\n", "\\n")
        return f"edit {path}{_hidden_path_note(path)}\n  - {old}\n  + {new}"
    if name == "glob":
        return f"glob {args.get('pat', args.get('pattern', '?'))}"
    if name == "grep":
        return f"grep {args.get('pat', '?')}"
    if name == "task":
        prompt = str(args.get("prompt", "")).strip()
        return f"task {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
    if name == "python":
        code = str(args.get("code", "")).strip()
        lines = code.split("\n")
        more = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        return f"python {lines[0][:160]}{more}"
    return f"{name}({json.dumps(args, ensure_ascii=False)[:200]})"


def print_tool_action(name: str, args: dict[str, Any]) -> None:
    """Print a tool call as plain text — no background boxes.

    Control characters are shown, not interpreted, so what the approval prompt
    displays is exactly what would run.
    """
    body = ui.visible(format_tool_action(name, args))
    ui._AGENT_LOCAL.last_action = body
    first, _, rest = body.partition("\n")
    print(f"{GREEN}⏺{RESET}{DIM} {first}{RESET}")
    for line in rest.split("\n"):
        if line.strip():
            print(f"{DIM}  {line}{RESET}")


def print_tool_result(result: str) -> None:
    """Print tool output with enough context to see what happened."""
    lines = ui.visible(result).split("\n")
    if ui._agent_tag():  # parallel agents: one line each, or the screen floods
        more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        print(f"{DIM}⎿ {lines[0][:160] or '(empty)'}{more}{RESET}")
        return
    print(f"{DIM}⎿ result{RESET}")
    if not result:
        print(f"{DIM}│ (empty){RESET}")
        return
    show = lines[:12]
    for line in show:
        print(f"{DIM}│ {line}{RESET}")
    if len(lines) > 12:
        print(f"{DIM}│ ... +{len(lines) - 12} more lines{RESET}")


def run_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a named tool with args, truncating output if it exceeds MAX_OUT."""
    if name == RESPOND_TOOL:
        return respond(args)
    try:
        result = TOOLS[name][2](normalize_tool_args(name, args))
        if len(result) > MAX_OUT:
            result = (
                result[:MAX_OUT]
                + f"\n... [truncated {len(result) - MAX_OUT} chars; raise MAX_TOOL_OUTPUT_CHARS]"
            )
        return result
    except Exception as e:  # any tool failure is reported to the model
        return f"error: {e}"


@contextlib.contextmanager
def thinking_spinner() -> Any:
    """Loader on the line below the user's input (style from /loader)."""
    if not sys.stdout.isatty() or ui._agent_tag():  # parallel agents share the screen
        yield
        return

    stop = threading.Event()
    step = 0
    context = ui.loader_context(backends.BACKEND, backends.MODEL)

    def animate() -> None:
        nonlocal step
        while not stop.is_set():
            bar = ui.loader_display(step, context)
            step += 1
            sys.stdout.write(f"\r{bar}")
            sys.stdout.flush()
            time.sleep(0.07)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.4)
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse all <tool_call> blocks from model output into structured dicts.

    The JSON object is extracted by brace matching (handles nesting), and the
    closing </tool_call> tag is optional: the mlx/transformers streaming loop
    stops at the JSON's closing brace before the tag is emitted, so requiring
    it would drop every local-model tool call.
    """
    calls: list[dict[str, Any]] = []
    pos = 0
    open_tag = "<tool_call>"
    close_tag = "</tool_call>"
    while (start := text.find(open_tag, pos)) != -1:
        brace = text.find("{", start)
        close = text.find(close_tag, start)
        if brace == -1:
            # malformed block with no JSON payload; skip and keep scanning
            pos = close + len(close_tag) if close != -1 else start + len(open_tag)
            continue
        payload: tuple[str, dict[str, Any]] | None = None
        with contextlib.suppress(Exception):
            obj, rel_end = json.JSONDecoder().raw_decode(text, brace)
            if isinstance(obj, dict):
                pos = rel_end
                if payload := _call_payload(obj):
                    calls.append(_tool_use(len(calls), *payload))
                continue
        if close != -1 and close > brace:
            with contextlib.suppress(Exception):
                payload = _call_payload(json.loads(text[brace:close]))
            pos = close + len(close_tag)
        else:
            break  # JSON may still be streaming
        if payload:
            calls.append(_tool_use(len(calls), *payload))
    return calls


def _call_payload(obj: Any) -> tuple[str, dict[str, Any]] | None:
    """Return (tool, args) from {"tool", "args"} or the Hermes/Qwen {"name", "arguments"} shape."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool", obj.get("name"))
    args = obj.get("args", obj.get("arguments", obj.get("parameters", {})))
    if isinstance(args, str):  # OpenAI-style arguments are a JSON string
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not _known_tool(name) or not isinstance(args, dict):
        return None
    return name, args


def _tool_use(index: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"call_{index}", "name": name, "input": args}


def _garbled_tool_call(text: str, native: bool) -> str:
    """Explain why a <tool_call> block in a reply wasn't run, or "" if there is none.

    Native backends parse tool calls server-side, so any call left in the text
    went unrun; on the XML path an unparsed block means bad JSON or an unknown tool.
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", text, re.DOTALL)
    if not m:
        return ""
    body = m.group(1)
    try:
        obj, _ = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as err:
        near = body[max(0, err.pos - 40) : err.pos + 20].replace("\n", "\\n")
        return f"invalid JSON at character {err.pos} ({err.msg}) near: {near}"
    if native:
        return (
            "it was written as text instead of through the function-calling interface"
        )
    name = obj.get("tool", obj.get("name")) if isinstance(obj, dict) else None
    return "" if _call_payload(obj) else f"unknown tool or bad arguments for {name!r}"


# -----------------------------------------------------------------------------------------------
# Tool definitions for the model
# -----------------------------------------------------------------------------------------------
_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "string?": "string",
    "number": "integer",
    "number?": "integer",
    "boolean": "boolean",
    "boolean?": "boolean",
}


def tool_specs() -> list[backends.ToolSpec]:
    """The tools offered this turn as (name, description, JSON schema).

    `respond` is included when a --json-schema answer is required of this agent.
    """
    specs: list[backends.ToolSpec] = []
    for name, (desc, params, _) in TOOLS.items():
        props = {k: {"type": _TYPE_MAP.get(v, "string")} for k, v in params.items()}
        req = [k for k, v in params.items() if not v.endswith("?")]
        specs.append(
            (name, desc, {"type": "object", "properties": props, "required": req})
        )
    if (respond_schema := _respond_schema()) is not None:
        specs.append((RESPOND_TOOL, RESPOND_DESCRIPTION, respond_schema))
    return specs


def _parse_response(
    response_text: str,
) -> tuple[str, list[backends.ToolCall], Any]:
    """Parse a raw API response into (display_text, tool_calls, raw_data).

    raw_data is the decoded JSON for native backends (used when appending to
    history); None for XML backends.
    """
    if backends.BACKEND in backends.NATIVE_TOOL_BACKENDS:
        data = json.loads(response_text)
        text, calls = backends._parse_native_response(data)
        return text, calls, data
    text = re.sub(
        r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL
    ).strip()
    calls = [
        backends.ToolCall(tc["id"], tc["name"], tc["input"])
        for tc in parse_tool_calls(response_text)
    ]
    return text, calls, None


# -----------------------------------------------------------------------------------------------
# History management
# -----------------------------------------------------------------------------------------------
# The Postgres store and the session the interactive loop is in, when the
# [history] extra is installed (see wrencode_history); otherwise history.json.
_STORE: history.Store | None = None
_SESSION_ID: int | None = None


def history_file_path() -> pathlib.Path:
    """Return the history file path from env override or user-level default."""
    if p := os.environ.get("WRENCODE_HISTORY_FILE"):
        return pathlib.Path(p).expanduser()
    return pathlib.Path.home() / ".wrencode" / "history.json"


def load_history() -> list[dict[str, Any]]:
    """Load the current session's messages from the store, else the JSON history file."""
    if _STORE is not None and _SESSION_ID is not None:
        return _STORE.load(_SESSION_ID)
    with contextlib.suppress(Exception), open(history_file_path()) as f:
        return list(json.load(f))
    return []


def save_history(messages: list[dict[str, Any]]) -> None:
    """Persist the conversation: the whole list, to the store or the JSON history file."""
    if _STORE is not None and _SESSION_ID is not None:
        try:
            _STORE.save(_SESSION_ID, messages)
        except Exception as err:  # the conversation is still in memory; say so
            print(
                f"{YELLOW}Could not save history to Postgres: {ui.visible(str(err))}{RESET}"
            )
        return
    with contextlib.suppress(Exception):
        p = history_file_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with open(fd, "w") as f:
            json.dump(messages, f)
        os.chmod(p, 0o600)  # a file created earlier with a wider mode


def _transcript(messages: list[dict[str, Any]], budget_chars: int) -> str:
    """Flatten messages into a plain transcript for summarizing.

    Each message is capped at 2000 chars, and if the whole still exceeds
    budget_chars the middle is cut, keeping the start (the task) and the end
    (the current state). Tool tags are defanged for NanoGPT's GLM models.
    """
    lines = []
    for m in messages:
        text = backends.flatten_content(m.get("content"))
        for c in m.get("tool_calls") or []:
            fn = c.get("function", {})
            text += f"\n[tool call] {fn.get('name')}({fn.get('arguments')})"
        if len(text) > 2000:
            text = f"{text[:2000]} ...[{len(text) - 2000} chars cut]"
        lines.append(f"{m['role']}: {text}")
    out = backends._defang_tool_tags("\n".join(lines))
    if len(out) > budget_chars:
        head = budget_chars // 4
        out = (
            out[:head]
            + "\n...[middle of the conversation omitted]...\n"
            + out[-(budget_chars - head) :]
        )
    return out


def compact_messages(
    messages: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Summarize conversation history to reduce context length (/compact)."""
    if not messages:
        return messages
    prompt = (
        "Summarize this conversation in 3-5 concise bullet points, "
        "preserving any file paths, code decisions, or unresolved tasks:\n\n"
        + _transcript(messages, backends.CONTEXT_TOKENS * 2)
    )
    summary = backends.complete(
        "You are a helpful assistant.",
        prompt,
        max_tokens=512,
        mlx_state=(model, tokenizer) if model else None,
    )
    return [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {
            "role": "assistant",
            "content": "Understood, I have the context from the summary.",
        },
    ]


_CONTEXT_ERROR = re.compile(
    r"context[_ ](length|size|window)|prompt is too long|input is too long"
    r"|too many (input )?tokens|maximum context",
    re.IGNORECASE,
)


def estimate_tokens(messages: list[dict[str, Any]], system_prompt: str) -> int:
    """Rough prompt size: about 4 characters per token of the serialized request."""
    chars = len(system_prompt) + sum(
        len(json.dumps(m, ensure_ascii=False)) for m in messages
    )
    return chars // 4


_COMPACTION_NOTE = "[Earlier conversation, compacted]"
_REQUEST_MARK = "\n\nLatest user request, verbatim:\n"
_CONTINUE_MARK = "\n\nContinue from where the conversation below leaves off."


def _is_compaction_note(m: dict[str, Any]) -> bool:
    return isinstance(m["content"], str) and m["content"].startswith(_COMPACTION_NOTE)


def _latest_request(messages: list[dict[str, Any]]) -> str:
    """Return the newest plain-text user message, unwrapping an earlier compaction note.

    Tool results are never plain strings, so a string user message is either
    something the user typed or a note from a previous compaction.
    """
    for m in reversed(messages):
        if m["role"] != "user" or not isinstance(m["content"], str):
            continue
        if not _is_compaction_note(m):
            return m["content"]
        _, found, rest = m["content"].partition(_REQUEST_MARK)
        return rest.removesuffix(_CONTINUE_MARK) if found else ""
    return ""


def auto_compact(
    messages: list[dict[str, Any]], mlx_state: tuple[Any, Any] | None
) -> None:
    """Summarize older messages in place, keeping recent ones verbatim.

    Safe mid-turn: the kept tail starts at an assistant message, so tool calls
    stay paired with their results, and the summary becomes the user message
    before it. The tail is as long as fits in about a quarter of the window.
    If the backend can't summarize, older messages are dropped with a note.
    """
    starts = [i for i, m in enumerate(messages) if i and m["role"] == "assistant"]
    if not starts:
        return
    cut = starts[-1]
    for i in reversed(starts):
        if estimate_tokens(messages[i:], "") > backends.CONTEXT_TOKENS // 4:
            break
        cut = i
    head, tail = messages[:cut], messages[cut:]
    if len(head) == 1 and _is_compaction_note(head[0]):
        return  # only the previous summary is left to fold in; nothing gained
    task = _latest_request(head)
    prompt = (
        "You are compacting the history of a coding agent's session so it can "
        "continue the task with less context. Write a concise summary covering: "
        "the user's request; files read, created, or changed and how; commands run "
        "and their key results; decisions made; and what remains to do. Keep exact "
        "file paths, names, and error messages.\n\n"
        + _transcript(head, backends.CONTEXT_TOKENS * 2)
    )
    try:
        summary = backends.complete(
            "You are a helpful assistant.", prompt, max_tokens=1500, mlx_state=mlx_state
        )
    except Exception as err:  # fall back to dropping history
        summary = f"(Earlier messages were dropped to fit the context window: {err})"
    note = f"{_COMPACTION_NOTE}\n{summary}"
    if task:
        note += f"{_REQUEST_MARK}{task}"
    note += _CONTINUE_MARK
    messages[:] = [{"role": "user", "content": note}, *tail]
    ui.print_system(f"⟳ Compacted {len(head)} older messages into a summary")


# -----------------------------------------------------------------------------------------------
# Workspace & system prompt
# -----------------------------------------------------------------------------------------------
def git_context() -> str:
    """Return a formatted git status string if inside a git repository."""
    with contextlib.suppress(Exception):
        r = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return f"\nGit status:\n{r.stdout.strip()}"
    return ""


def find_agents_files() -> list[pathlib.Path]:
    """Return the instruction files that apply to the workspace, outermost first.

    Looks in CONFIG_DIR (user-wide), then each directory from the git root down to
    the workspace; outside a git repo only the workspace itself. Each directory
    contributes AGENTS.md, or CLAUDE.md when it has no AGENTS.md. Nearer files come
    later in the prompt, so they read as taking precedence.
    """
    ws = workspace_root()
    chain = [ws, *ws.parents]
    top = next((i for i, d in enumerate(chain) if (d / ".git").exists()), 0)
    found: list[pathlib.Path] = []
    for d in [backends.CONFIG_DIR, *reversed(chain[: top + 1])]:
        for name in AGENTS_FILES:
            if (p := d / name).is_file():
                found.append(p)
                break
    return found


def agents_md_context() -> str:
    """Return AGENTS.md contents for the system prompt, capped at MAX_AGENTS_MD_CHARS."""
    parts: list[str] = []
    budget = MAX_AGENTS_MD_CHARS
    for p in find_agents_files():
        with contextlib.suppress(OSError):
            text = p.read_text(errors="replace").strip()[:budget]
            if text:
                budget -= len(text)
                parts.append(f"--- {p} ---\n{text}")
        if budget <= 0:
            break
    if not parts:
        return ""
    return (
        "\n\nProject instructions from AGENTS.md files. Follow them; "
        "later (nearer) files take precedence:\n\n" + "\n\n".join(parts)
    )


def build_system_prompt() -> str:
    """Build the system prompt with workspace context and tool definitions."""
    ws = workspace_root()
    path_rule = (
        "Paths are not restricted to the workspace."
        if os.environ.get("WRENCODE_UNRESTRICTED_PATHS", "").lower()
        in ("1", "true", "yes")
        else "Relative paths resolve under the workspace. Absolute paths must stay inside it."
    )
    if backends.BACKEND in backends.OPENAI_FORMAT_BACKENDS:
        # Tool schemas travel in the request's `tools` field; XML instructions here
        # would compete with native calling (and break NanoGPT's GLM models).
        tool_format = "Call tools through the native function-calling interface."
    else:
        tool_format = """To use a tool, format it EXACTLY like this:
<tool_call>{"tool": "name", "args": {"key": "value"}}</tool_call>

Examples:
<tool_call>{"tool": "read", "args": {"path": "file.py", "offset": 0, "limit": 20}}</tool_call>
<tool_call>{"tool": "glob", "args": {"pat": "*.py"}}</tool_call>"""
    python_line = ""
    if "python" in TOOLS:
        python_line = (
            "- python(code): Run a Python snippet in a sandbox: no network, shell or "
            "environment, a standard-library subset (pathlib and re yes; os.path, os.walk and os.listdir no), the "
            "workspace read-only at /workspace (the working directory). Functions: "
            "glob(pat) -> list of workspace-relative paths, read(path) -> the file's "
            "text, grep(pat) -> list of 'path:line:text'. Each call is a fresh "
            "interpreter, nothing persists between calls, so do the whole job in one "
            "snippet; print() what you want to see, a trailing expression's value "
            "is returned\n"
        )
    respond_line = ""
    if (respond_schema := _respond_schema()) is not None:
        respond_line = (
            f"- {RESPOND_TOOL}(...): {RESPOND_DESCRIPTION} Its arguments must match this "
            f"JSON Schema: {json.dumps(respond_schema)}\n\n"
        )
    return f"""You are a helpful coding assistant with tools to interact with the file system.
Workspace root: {ws}
Process cwd: {os.getcwd()}
{path_rule}{git_context()}

IMPORTANT: You MUST use the tools below for file operations.

Available tools:
- read(path, offset, limit): Read a file or list a directory
- write(path, content): Write to a file
- edit(path, old, new): Replace text in a file (old must be unique unless all=true)
- glob(pat): Find files matching pattern
- grep(pat): Search for text in files
- bash(cmd): Run a shell command
- task(prompt): Delegate a self-contained subtask to a fresh subagent; returns only its result. Several task calls in one reply run in parallel, so batch independent subtasks together
{python_line}
{respond_line}{tool_format}

When reading a file, always pass offset and limit. When you finish a task, summarize what you changed.
CRITICAL: You MUST use tools for file operations. Never say you can't access files!{agents_md_context()}"""


# -----------------------------------------------------------------------------------------------
# Agentic loop
# -----------------------------------------------------------------------------------------------
MAX_TRUNCATION_RETRIES = 2
# An identical failing tool call (same name and args) gets a hint on its 3rd
# try and stops the turn on its 5th, even if other calls happen in between.
REPEATED_CALL_HINT, REPEATED_CALL_STOP = 3, 5
REPEATED_CALL_NOTE = (
    "\n\n(This exact call has now failed {n} times. Repeating it won't work: "
    "change approach, e.g. re-read the file and copy the text exactly.)"
)
GARBLED_NUDGE = (
    "Your last message contained a tool call that couldn't be run: {why}. Nothing "
    "was executed. Send the call again as a proper tool call, with arguments as "
    "valid JSON: each string is one JSON string literal with newlines escaped as "
    '\\n and quotes as \\"; no + concatenation or Python syntax.'
)


RESPOND_NUDGE = (
    f"You haven't given your final answer. Call the {RESPOND_TOOL} tool with arguments "
    "matching its schema; a plain-text reply isn't accepted."
)
TRUNCATION_NUDGE = (
    "Your last response hit the output token limit before you called a tool or "
    "finished, so nothing happened. Continue in smaller steps: keep any reasoning "
    "brief, and create or edit one file per tool call."
)


def _track_error(
    result: str, last: str | None, count: int
) -> tuple[str | None, int, bool]:
    """Update repeated-error state; return (last_error, count, should_stop)."""
    if result.startswith("error:"):
        count = count + 1 if result == last else 1
        if count >= TOOL_ERROR_REPEAT_LIMIT:
            print(
                f"{YELLOW}Stopping: repeated identical tool error {count} times.{RESET}"
            )
            return result, count, True
        return result, count, False
    return None, 0, False


def run_agent_turn(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: tuple[Any, Any] | None,
    max_iters: int = 0,
) -> str:
    """Generate a response and execute any tool calls, repeating until no tools remain.

    max_iters > 0 caps the tool-calling rounds (used to bound subagents);
    0 means unlimited, preserving the interactive default. Returns why the turn
    ended: "done", "max_turns", "max_tokens", "tool_errors",
    "no_structured_output", "malformed_tool_call", or "cancelled".
    """
    iters = 0
    last_tool_error: str | None = None
    repeated_tool_error_count = 0
    retried_overflow = False
    truncations = 0
    respond_nudges = 0
    garbled_nudges = 0
    failed_calls: dict[str, int] = {}
    try:
        while True:
            if max_iters and iters >= max_iters:
                print(f"{YELLOW}(stopped after {max_iters} iterations){RESET}")
                return "max_turns"
            iters += 1
            if COMPACT_AT and estimate_tokens(messages, system_prompt) > (
                backends.CONTEXT_TOKENS * COMPACT_AT
            ):
                auto_compact(messages, mlx_state)
            try:
                with thinking_spinner():
                    response_text = get_response_cancellable(
                        messages, system_prompt, mlx_state
                    )
            except Exception as err:
                # The window guess was too big: compact once and retry the round.
                if retried_overflow or not _CONTEXT_ERROR.search(str(err)):
                    raise
                ui.print_system("⟳ Context limit hit, compacting and retrying")
                retried_overflow = True
                auto_compact(messages, mlx_state)
                iters -= 1
                continue
            retried_overflow = False
            display_text, tool_calls, raw_data = _parse_response(response_text)
            if display_text:
                ui.print_agent_message(display_text)
            if (
                not tool_calls
                and raw_data is not None
                and backends._is_truncated(raw_data)
            ):
                # Cut off before acting (often spent on reasoning): nudge, don't stop.
                truncations += 1
                if truncations > MAX_TRUNCATION_RETRIES:
                    return "max_tokens"
                messages.append(
                    {"role": "assistant", "content": display_text or "(cut off)"}
                )
                messages.append({"role": "user", "content": TRUNCATION_NUDGE})
                continue
            truncations = 0
            garbled = (
                ""
                if tool_calls
                else _garbled_tool_call(
                    display_text if raw_data is not None else response_text,
                    raw_data is not None,
                )
            )
            backends._append_assistant(messages, display_text, tool_calls, raw_data)
            if garbled:
                # A tool call the model meant to make but that couldn't run.
                garbled_nudges += 1
                if garbled_nudges > MAX_TRUNCATION_RETRIES:
                    return "malformed_tool_call"
                print(
                    f"{YELLOW}Unparseable tool call ({garbled}); asking for a resend.{RESET}"
                )
                messages.append(
                    {"role": "user", "content": GARBLED_NUDGE.format(why=garbled)}
                )
                continue
            if not tool_calls:
                if _respond_schema() is None or _STRUCTURED_RESULT:
                    return "done"
                # A structured answer is required but the model just stopped.
                respond_nudges += 1
                if respond_nudges > MAX_TRUNCATION_RETRIES:
                    return "no_structured_output"
                messages.append({"role": "user", "content": RESPOND_NUDGE})
                continue
            results: list[tuple[backends.ToolCall, str]] = []
            stop = False
            parallel = _run_parallel_tasks(tool_calls)
            for i, tc in enumerate(tool_calls):
                ui.check_cancelled()
                if i in parallel:
                    result = parallel[i]
                else:
                    print_tool_action(tc.name, tc.input)
                    result = run_tool(tc.name, tc.input)
                last_tool_error, repeated_tool_error_count, stop = _track_error(
                    result, last_tool_error, repeated_tool_error_count
                )
                if result.startswith("error:"):
                    # Same failing call again, even with other calls in between?
                    key = (
                        f"{tc.name}:{json.dumps(tc.input, sort_keys=True, default=str)}"
                    )
                    failed_calls[key] = failed_calls.get(key, 0) + 1
                    if failed_calls[key] >= REPEATED_CALL_STOP:
                        print(
                            f"{YELLOW}Stopping: the same failing {tc.name} call "
                            f"was made {failed_calls[key]} times.{RESET}"
                        )
                        stop = True
                    elif failed_calls[key] >= REPEATED_CALL_HINT:
                        result += REPEATED_CALL_NOTE.format(n=failed_calls[key])
                print_tool_result(result)
                results.append((tc, result))
                if stop:
                    break
            # Every tool call needs a result, or the next request is rejected.
            results += [
                (tc, "skipped: stopped after repeated errors")
                for tc in tool_calls[len(results) :]
            ]
            backends._append_tool_results(messages, results)
            if stop:
                return "tool_errors"
            if _STRUCTURED_RESULT and _respond_schema() is not None:
                return "done"
    except (ui.UserCancelled, KeyboardInterrupt):
        if ui._agent_tag():
            print(f"{YELLOW}cancelled{RESET}")
            return "cancelled"
        print(f"\n{YELLOW}Cancelled — back to prompt.{RESET}\n")
        return "cancelled"
    finally:
        if not ui._agent_tag():  # the batch owner clears the shared cancel state
            ui._CANCEL_REQUESTED.clear()
            ui._LISTENER_STOP.set()


def _parallel_subagents_enabled() -> bool:
    # In-process weights (mlx / transformers) can't serve two agents at once.
    return (
        MAX_PARALLEL_SUBAGENTS > 1
        and backends.BACKEND not in backends.LOCAL_ML_BACKENDS
    )


def _run_parallel_tasks(tool_calls: list[backends.ToolCall]) -> dict[int, str]:
    """Run a reply's task calls at once; return results by position in tool_calls.

    Returns {} (run everything in order) for fewer than two task calls, when
    parallelism is off, or at the depth limit, where task() reports the error.
    """
    positions = [i for i, tc in enumerate(tool_calls) if tc.name == "task"]
    if (
        len(positions) < 2
        or not _parallel_subagents_enabled()
        or ui._subagent_depth() >= MAX_SUBAGENT_DEPTH
    ):
        return {}
    for i in positions:
        print_tool_action("task", tool_calls[i].input)
    calls = [tool_calls[i] for i in positions]
    return dict(zip(positions, _run_tasks_concurrently(calls)))


def _run_tasks_concurrently(calls: list[backends.ToolCall]) -> list[str]:
    """Run task calls in threads, MAX_PARALLEL_SUBAGENTS at a time, results in order."""
    parent_depth = ui._subagent_depth()
    parent_tag = ui._agent_tag()
    parent_cancel = getattr(ui._AGENT_LOCAL, "cancel", None)
    cancel = threading.Event()
    slots = threading.Semaphore(MAX_PARALLEL_SUBAGENTS)
    results = ["cancelled: the parallel run was stopped"] * len(calls)

    def worker(n: int, tc: backends.ToolCall) -> None:
        with slots:
            ui._AGENT_LOCAL.depth = parent_depth
            ui._AGENT_LOCAL.tag = f"{parent_tag}.{n + 1}" if parent_tag else str(n + 1)
            ui._AGENT_LOCAL.cancel = cancel
            if cancel.is_set():
                return
            try:
                results[n] = run_tool(tc.name, tc.input)
            except BaseException as err:  # reported as the result
                results[n] = f"error: {err}"

    threads = [
        threading.Thread(target=worker, args=(n, tc), daemon=True)
        for n, tc in enumerate(calls)
    ]
    out: ui._AgentStdout | None = None
    if not isinstance(sys.stdout, ui._AgentStdout):
        out = ui._AgentStdout(sys.stdout)
        sys.stdout = out
    esc = None
    if not parent_tag:  # the top-level batch owns Escape-to-cancel
        esc = ui._EscWatch()
        ui.PARALLEL_ESC = esc
        ui._CANCEL_REQUESTED.clear()
        esc.start()
    print(f"{CYAN}↳ running {len(calls)} subagents in parallel{RESET}")
    try:
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            if ui._CANCEL_REQUESTED.is_set() or (
                parent_cancel is not None and parent_cancel.is_set()
            ):
                cancel.set()
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        cancel.set()
        raise
    finally:
        if cancel.is_set():  # give subagents a moment to stop at their next check
            deadline = time.monotonic() + 2
            for t in threads:
                t.join(timeout=max(0.0, deadline - time.monotonic()))
        if esc is not None:
            esc.stop()
            ui.PARALLEL_ESC = None
        if out is not None:
            out.release()
            sys.stdout = out._real
    if cancel.is_set():
        raise ui.UserCancelled()
    print(f"{CYAN}↳ {len(calls)} subagents done{RESET}")
    return results


# -----------------------------------------------------------------------------------------------
# Claude Agent SDK session
# -----------------------------------------------------------------------------------------------
# The backend itself is wrencode_sdk: Claude Code's own agent loop in a subprocess,
# with approvals routed through ui.confirm(). This is the one interactive session.
_AGENT_SDK_SESSION: agent_sdk.AgentSDKSession | None = None


def agent_sdk_session() -> agent_sdk.AgentSDKSession:
    """The interactive Agent SDK session, created on first use."""
    global _AGENT_SDK_SESSION
    if _AGENT_SDK_SESSION is None:
        _AGENT_SDK_SESSION = agent_sdk.AgentSDKSession(
            cwd=workspace_root(), instructions=agents_md_context()
        )
    return _AGENT_SDK_SESSION


def close_agent_sdk_session() -> None:
    global _AGENT_SDK_SESSION
    if _AGENT_SDK_SESSION is not None:
        _AGENT_SDK_SESSION.close()
        _AGENT_SDK_SESSION = None


# -----------------------------------------------------------------------------------------------
# Slash commands
# -----------------------------------------------------------------------------------------------
def handle_slash_command(
    cmd: str,
    messages: list[dict[str, Any]],
    mlx_state: tuple[Any, Any] | None,
) -> tuple[str | None, Any]:
    """Handle a slash command.

    Returns (action, mlx_state). mlx_state is _MLX_UNCHANGED unless the
    backend/model changed and the in-process model must be reloaded.
    """
    cmd = ui.SLASH_ALIASES.get(cmd, cmd)
    if backends.BACKEND == backends.AGENT_SDK_BACKEND and cmd in {"/clear", "/compact"}:
        session = agent_sdk_session()
        if cmd == "/clear":
            session.reset()
            ui.print_system("Cleared")
        else:
            session.run("/compact")  # Claude Code compacts its own context
        return "handled", configure._MLX_UNCHANGED
    if cmd in {"/quit", "exit"}:
        save_history(messages)
        return "quit", configure._MLX_UNCHANGED
    if cmd == "/clear":
        global _SESSION_ID
        ui.SESSION_AUTO_APPROVE = False  # "allow all" ends with the conversation
        messages.clear()
        if _STORE is not None:  # the old session stays in the store; start a new one
            _SESSION_ID = _STORE.new_session(
                str(workspace_root()), backends.BACKEND, backends.MODEL
            )
            ui.print_system(f"Cleared (new session #{_SESSION_ID})")
        else:
            save_history(messages)
            ui.print_system("Cleared")
        return "handled", configure._MLX_UNCHANGED
    if cmd in {"/sessions", "/resume", "/search", "/sync"} or cmd.startswith(
        ("/resume ", "/search ")
    ):
        _history_command(cmd, messages)
        return "handled", configure._MLX_UNCHANGED
    if cmd == "/compact":
        if (
            backends.BACKEND in backends.HOSTED_BACKENDS
            or backends.BACKEND == "ollama"
            or (backends.BACKEND in backends.LOCAL_ML_BACKENDS and mlx_state)
        ):
            ui.print_system("Compacting history...")
            model, tokenizer = mlx_state or (None, None)
            before = len(messages)
            messages[:] = compact_messages(messages, model, tokenizer)
            save_history(messages)
            ui.print_system(f"Compacted {before} → {len(messages)} messages")
        else:
            print(
                f"{YELLOW}/compact not available for backend '{backends.BACKEND}'{RESET}"
            )
        return "handled", configure._MLX_UNCHANGED
    if cmd in {"/backend", "/configure"}:
        return "handled", configure.switch_backend_runtime()
    if cmd == "/model" or cmd.startswith("/model "):
        model_id = cmd[7:].strip() if cmd.startswith("/model ") else ""
        return "handled", configure.switch_model_runtime(model_id)
    if cmd == "/help":
        for name, desc in ui.SLASH_COMMANDS.items():
            ui.print_system(f"{name:<12} {desc}")
        ui.print_system("Type / for suggestions: ↑↓ pick, Tab completes, Enter runs.")
        return "handled", configure._MLX_UNCHANGED
    return None, configure._MLX_UNCHANGED


def _arg_value(args: list[str], *names: str) -> str | None:
    """Return the value after the first of names in args (or --name=value), if any."""
    for i, a in enumerate(args):
        if a in names:
            nxt = args[i + 1] if i + 1 < len(args) else None
            return None if nxt is None or (nxt.startswith("-") and nxt != "-") else nxt
        for n in names:
            if n.startswith("--") and a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


VERIFY_ATTEMPTS = 3


def run_verify(cmd: str) -> tuple[bool, str]:
    """Run the --verify command in the workspace; return (passed, output tail)."""
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace_root(),
            capture_output=True,
            check=False,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out after 600s"
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0, f"exit code {r.returncode}\n{out[-4000:]}"


def run_headless(
    prompt: str,
    output_format: str = "text",
    max_turns: int = 0,
    schema: dict[str, Any] | None = None,
    verify: str = "",
) -> int:
    """Run one prompt without the interactive UI and return the exit code (wrencode -p).

    The UI goes to stderr so stdout carries only the final answer, or a JSON
    object with --output-format json. Saved history is neither loaded nor saved.
    Without --yes, writes and shell commands are declined rather than prompted.
    With a schema (--json-schema), the answer is a validated JSON value instead.
    With verify (--verify), that command must pass once the agent says it's done;
    on failure its output goes back to the agent, up to VERIFY_ATTEMPTS times.
    """
    global _MLX_STATE, _OUTPUT_SCHEMA
    ui.HEADLESS = True
    _OUTPUT_SCHEMA = schema
    _STRUCTURED_RESULT.clear()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    reason, error = "error", ""
    verified: bool | None = None
    verify_output = ""
    sdk: agent_sdk.AgentSDKSession | None = None
    cost_usd = 0.0
    with contextlib.redirect_stdout(sys.stderr):
        try:
            _warn_dotenv_ignored()
            configure.resolve_configuration()
            _MLX_STATE = backends.load_model()
            system_prompt = build_system_prompt()
            if backends.BACKEND == backends.AGENT_SDK_BACKEND:
                if schema is not None:
                    raise ValueError(
                        "--json-schema isn't supported on the claude-agent-sdk backend"
                    )
                sdk = agent_sdk.AgentSDKSession(
                    cwd=workspace_root(),
                    instructions=agents_md_context(),
                    persist=False,
                    max_turns=max_turns,
                )
            for attempt in range(VERIFY_ATTEMPTS if verify else 1):
                if sdk is not None:
                    turn = sdk.run(backends.flatten_content(messages[-1]["content"]))
                    messages.append({"role": "assistant", "content": turn.text})
                    cost_usd += turn.cost_usd
                    reason = "error" if turn.is_error else "done"
                    if turn.is_error:
                        error = turn.error
                else:
                    reason = run_agent_turn(
                        messages, system_prompt, _MLX_STATE, max_iters=max_turns
                    )
                if not verify or reason != "done":
                    break
                verified, verify_output = run_verify(verify)
                print(
                    f"{DIM}verify `{verify}`: {'passed' if verified else 'failed'}{RESET}"
                )
                if verified or attempt == VERIFY_ATTEMPTS - 1:
                    break
                _STRUCTURED_RESULT.clear()
                messages.append(
                    {
                        "role": "user",
                        "content": f"You said you're done, but the check `{verify}` failed "
                        f"({verify_output.splitlines()[0]}):\n\n{verify_output}\n\n"
                        "Fix the problem, then finish with a summary.",
                    }
                )
            if verified is False and reason == "done":
                reason = "verify_failed"
        except (
            SystemExit
        ):  # setup failed (no backend, key, or model); reason is on stderr
            error = "configuration error (see stderr)"
        except Exception as err:  # reported in the result
            error = str(err)
            print(f"{RED}Error: {ui.visible(error)}{RESET}")
        finally:
            if sdk is not None:
                sdk.close()
    texts = [
        backends.flatten_content(m["content"])
        for m in messages
        if m["role"] == "assistant"
    ]
    result = texts[-1].strip() if texts else ""
    is_error = reason != "done" or (schema is not None and not _STRUCTURED_RESULT)
    if verify and verified is None and reason == "done":  # never reached the check
        is_error = True
    if output_format == "json":
        out: dict[str, Any] = {
            "result": result,
            "is_error": is_error,
            "stop_reason": reason,
            "num_turns": len(texts),
            "backend": backends.BACKEND,
            "model": backends.MODEL,
        }
        if sdk is not None:
            out["cost_usd"] = round(cost_usd, 6)
        if schema is not None:
            out["structured_output"] = (
                _STRUCTURED_RESULT[0] if _STRUCTURED_RESULT else None
            )
        if verify:
            out["verified"] = verified
            if verified is False:
                out["verify_output"] = verify_output
        if error:
            out["error"] = error
        print(json.dumps(out, ensure_ascii=False))
    elif schema is not None:
        if _STRUCTURED_RESULT:
            print(json.dumps(_STRUCTURED_RESULT[0], ensure_ascii=False))
    elif result:
        print(result)
    return 1 if is_error else 0


def _history_command(cmd: str, messages: list[dict[str, Any]]) -> None:
    """/sessions, /resume <id>, /search <text> and /sync, over the Postgres history store."""
    global _SESSION_ID
    if _STORE is None:
        ui.print_system(
            "History is in history.json. For sessions and search, install the Postgres "
            "store: pip install 'wrencode[history]' (see README, History)."
        )
        return
    ws = str(workspace_root())
    word, _, arg = cmd.partition(" ")
    arg = arg.strip()
    if word == "/sessions":
        rows = _STORE.sessions(ws)
        for r in rows:
            mark = "›" if r["id"] == _SESSION_ID else " "
            when = r["updated_at"].strftime("%Y-%m-%d %H:%M")
            title = ui.visible(r["title"]) or "(empty)"
            ui.print_system(
                f"{mark} #{r['id']:<5} {when}  {r['chats']:>3} chats  {title}"
            )
        ui.print_system("/resume <id> continues one; /search <text> looks inside them.")
    elif word == "/sync":
        if _STORE.mirror is None:
            ui.print_system(
                "No mirror configured. Set WRENCODE_MIRROR_URL to a second Postgres to "
                "keep a copy of the history there."
            )
            return
        save_history(messages)
        queued, left = _STORE.sync(ws)
        if left:
            ui.print_system(
                f"Queued {queued} sessions for {_STORE.mirror.label}; {left} still "
                f"pending (the mirror is slow or unreachable; they retry in the background)"
            )
        else:
            ui.print_system(f"Mirrored {queued} sessions to {_STORE.mirror.label}")
    elif word == "/resume":
        if not arg.isdigit():
            ui.print_system("Usage: /resume <id>  (ids from /sessions)")
            return
        if not any(r["id"] == int(arg) for r in _STORE.sessions(ws, limit=1000)):
            ui.print_system(f"No session #{arg} for this workspace.")
            return
        save_history(messages)
        _SESSION_ID = int(arg)
        messages[:] = load_history()
        chats = sum(1 for m in messages if m.get("role") == "user")
        ui.print_system(f"Resumed session #{_SESSION_ID} ({chats} chats)")
    else:
        if not arg:
            ui.print_system("Usage: /search <text>")
            return
        hits = _STORE.search(ws, arg)
        if not hits:
            ui.print_system("No matches.")
        for h in hits:
            ui.print_system(
                f"#{h['session_id']:<5} {h['role']:<9} {ui.visible(h['text'])}"
            )


def _warn_dotenv_ignored() -> None:
    if _DOTENV_IGNORED:
        names = ", ".join(sorted(set(_DOTENV_IGNORED)))
        print(
            f"{YELLOW}Ignored from ./.env: {names}. A project's .env may only set "
            f"*_API_KEY and ANTHROPIC_WORKSPACE_ID; set the rest in your shell.{RESET}"
        )


def print_help() -> None:
    """Print CLI usage."""
    print("wrencode — a minimal agent harness for coding\n")
    print("Usage: wrencode [options]\n")
    print("Options:")
    print("-p, --print PROMPT   run one prompt headless and print the answer")
    print("                     (PROMPT '-' or omitted with piped stdin reads stdin)")
    print("--output-format F    with -p: text (default) or json")
    print("--max-turns N        with -p: cap tool-calling rounds")
    print(
        "--json-schema S      with -p: answer as JSON matching schema S (file or inline)"
    )
    print(
        "--verify CMD         with -p: CMD must pass when the agent finishes, else it retries"
    )
    print("--yes         auto-approve all writes/commands (WRENCODE_AUTO_APPROVE)")
    print("--uninstall   remove saved config and show how to delete wrencode")
    print("--version, -V print version and exit")
    print("--help, -h    show this help\n")
    print("Subcommands:")
    print("synthesize [files|dir]   fuse chat transcripts into one synthesis")
    print("synthesize               (no args) pick from this project's chat history")
    print("synthesize diff|log ...  diff = divergences only; log = decision timeline")
    print("synthesize --out FILE    write the result to FILE; --all skips the picker\n")
    print(
        "Slash commands: /backend /model /c /compact /sessions /resume /search /sync /q"
    )
    print(
        "Environment overrides: BACKEND, MODEL, and the backend's API key "
        "(e.g. ANTHROPIC_API_KEY) take precedence over saved config."
    )
    print(
        "Warning: --yes runs writes and shell commands without confirmation; "
        "use it only in a sandboxed workspace."
    )


def main() -> None:
    """Entry point — initialize the agent and run the interactive loop."""
    global _MLX_STATE
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print_help()
        return
    if "--version" in args or "-V" in args:
        print(f"wrencode {WRENCODE_VERSION}")
        return
    if "--uninstall" in args or (bool(args) and args[0] == "uninstall"):
        configure.uninstall()
        return
    if "--yes" in args or "--auto-approve" in args:
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"
    if args and args[0] == "synthesize":
        rest, out, paths, take_all = args[1:], None, [], False
        mode = "merge"
        if (
            rest and rest[0] in synthesize.SYNTH_MODES
        ):  # optional git-like submode keyword
            mode, rest = rest[0], rest[1:]
        i = 0
        while i < len(rest):
            if rest[i] in {"--out", "-o"} and i + 1 < len(rest):
                out = rest[i + 1]
                i += 2
            elif rest[i] == "--all":
                take_all = True
                i += 1
            else:
                paths.append(rest[i])
                i += 1
        configure.resolve_configuration()
        synthesize.run_synthesize(paths, out, interactive=not take_all, mode=mode)
        return

    os.environ.setdefault("WRENCODE_WORKSPACE", str(pathlib.Path.cwd().resolve()))
    piped = not sys.stdin.isatty()
    if {"-p", "--print"} & set(args) or any(a.startswith("--print=") for a in args):
        prompt = _arg_value(args, "-p", "--print")
        if prompt in {None, "-"}:
            prompt = sys.stdin.read() if piped else ""
        if not prompt.strip():
            print(f'{RED}No prompt: pass -p "..." or pipe one on stdin.{RESET}')
            raise SystemExit(2)
        fmt = _arg_value(args, "--output-format") or "text"
        if fmt not in {"text", "json"}:
            print(f"{RED}--output-format must be text or json.{RESET}")
            raise SystemExit(2)
        turns = _arg_value(args, "--max-turns") or "0"
        if not turns.isdigit():
            print(f"{RED}--max-turns must be a non-negative integer.{RESET}")
            raise SystemExit(2)
        schema = None
        if (raw := _arg_value(args, "--json-schema")) is not None:
            try:
                text = (
                    raw
                    if raw.lstrip().startswith("{")
                    else pathlib.Path(raw).read_text()
                )
                schema = json.loads(text)
            except (OSError, ValueError) as err:
                print(f"{RED}--json-schema: {err}{RESET}")
                raise SystemExit(2) from err
            if not isinstance(schema, dict):
                print(f"{RED}--json-schema must be a JSON object (a schema).{RESET}")
                raise SystemExit(2)
        verify = _arg_value(args, "--verify") or ""
        raise SystemExit(run_headless(prompt, fmt, int(turns), schema, verify))

    _warn_dotenv_ignored()
    configure.resolve_configuration()

    sys.stdout.write("\033]0;wrencode\007")  # set terminal tab/window title
    print(ui.render_banner(ui.colors_enabled()))
    print(f"{BOLD}wrencode{RESET} 🐦 | {DIM}{backends.BACKEND}:{backends.MODEL}{RESET}")
    mlx_state = backends.load_model()
    _MLX_STATE = mlx_state  # expose to the task() subagent tool
    system_prompt = build_system_prompt()
    for path in find_agents_files():
        print(f"{DIM}Loaded {path}{RESET}")
    global _STORE, _SESSION_ID
    _STORE = history.open_store()
    if _STORE is not None:
        ws = str(workspace_root())
        _SESSION_ID = _STORE.latest_session(ws) or _STORE.new_session(
            ws, backends.BACKEND, backends.MODEL
        )
    elif history.UNAVAILABLE_REASON:
        print(
            f"{YELLOW}Postgres history unavailable ({ui.visible(history.UNAVAILABLE_REASON)}); "
            f"using history.json{RESET}"
        )
    messages = load_history()
    if backends.BACKEND == backends.AGENT_SDK_BACKEND:
        if agent_sdk._load_agent_sdk_session_id(workspace_root()):
            print(
                f"{DIM}Resuming the Claude Agent SDK session (/clear starts fresh){RESET}"
            )
    elif messages:
        chats = sum(1 for m in messages if m.get("role") == "user")
        where = f"session #{_SESSION_ID} with " if _STORE is not None else ""
        print(f"{DIM}Restored {where}{chats} chats{RESET}")
    if _STORE is not None and _STORE.mirror is not None:
        print(
            f"{DIM}History mirrored to {_STORE.mirror.label} (/sync copies everything now){RESET}"
        )

    while True:
        try:
            user_input = ui.read_user_input()
            if not user_input:
                continue
            action, new_mlx = handle_slash_command(user_input, messages, mlx_state)
            if new_mlx is not configure._MLX_UNCHANGED:
                mlx_state = new_mlx
                _MLX_STATE = mlx_state
            if action == "quit":
                break
            if action == "handled":
                continue
            if backends.BACKEND == backends.AGENT_SDK_BACKEND:
                agent_sdk_session().run(user_input)
                continue
            messages.append({"role": "user", "content": user_input})
            run_agent_turn(messages, system_prompt, mlx_state)
            save_history(messages)
        except KeyboardInterrupt:
            save_history(messages)
            print(f"\n{DIM}(use /q to quit){RESET}")
            continue
        except EOFError:
            break
        except Exception as err:  # shown to the user; the session goes on
            msg = str(err)
            print(f"{RED}Error: {ui.visible(msg)}{RESET}")
            if backends.BACKEND == "ollama" and (
                "not found" in msg.lower() or "404" in msg
            ):
                print(
                    f"{YELLOW}Model '{backends.MODEL}' isn't pulled. "
                    f"Run: ollama pull {backends.MODEL}  (or `ollama list`).{RESET}"
                )
            if os.environ.get("WRENCODE_DEBUG"):
                traceback.print_exc()
            else:
                print(f"{DIM}(set WRENCODE_DEBUG=1 for the full traceback){RESET}")
    close_agent_sdk_session()
    if _STORE is not None:
        _STORE.close()


if __name__ == "__main__":
    main()
