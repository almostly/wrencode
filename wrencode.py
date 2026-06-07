#!/usr/bin/env python3
"""WrenCode — a minimal agentic coding assistant inspired by Harold Wren.

A lightweight alternative to Claude Code in a single Python file.

Supports multiple inference backends: local Apple Silicon via MLX,
HuggingFace Transformers, Anthropic, OpenAI, OpenRouter, and local proxy.
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

# flake8: noqa: E501, E203

import ast
import contextlib
import datetime
import getpass
import glob as globlib
import hashlib
import hmac
import json
import os
import pathlib
import platform
import re
import select
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

# The PyInstaller single-file binary ships no system CA trust store, so urllib's
# TLS verification fails out of the box ("CERTIFICATE_VERIFY_FAILED") on a clean
# machine. If certifi is bundled (it is in the binary build), point the SSL
# defaults at it before any request. Soft import keeps pip/uvx installs
# dependency-free; setdefault preserves any user-provided override.
try:
    import certifi  # ty: ignore[unresolved-import]

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# Load .env (next to this script, then the current directory); real env vars win.
for _dir in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    _env_path = os.path.join(_dir, ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                if _line.startswith("export "):
                    _line = _line[len("export ") :]
                _k, _v = _line.split("=", 1)
                try:
                    _v = shlex.split(_v)[0] if _v else _v
                except ValueError:
                    pass  # malformed quoting — use raw value
                os.environ.setdefault(_k.strip(), _v)

# -----------------------------------------------------------------------------------------------
# Backend configuration
# -----------------------------------------------------------------------------------------------
WRENCODE_VERSION = "0.1.5"

# Per-backend defaults. "kind" controls how a backend is treated:
#   api         - hosted HTTP API, needs an API key
#   local-proxy - Anthropic-compatible server already running on localhost
#   local-ml    - in-process model weights (mlx / transformers); source install only,
#                 since the standalone binary can't bundle the ML stack
BACKEND_SPECS: dict[str, dict[str, str]] = {
    "anthropic": {
        "kind": "api",
        "model": "claude-haiku-4-5-20251001",
        "key_env": "ANTHROPIC_API_KEY",
        "api_base": "https://api.anthropic.com/v1/messages",
        "label": "Anthropic Claude (API key)",
    },
    "openai": {
        "kind": "api",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "api_base": "https://api.openai.com/v1/chat/completions",
        "label": "OpenAI GPT (API key)",
    },
    "openrouter": {
        "kind": "api",
        "model": "anthropic/claude-3-haiku",
        "key_env": "OPENROUTER_API_KEY",
        "api_base": "https://openrouter.ai/api/v1/chat/completions",
        "label": "OpenRouter — any model (API key)",
    },
    "bedrock": {
        # AWS Bedrock via the model-agnostic Converse API — works across Claude,
        # Llama, Nova, Mistral, OpenAI GPT-OSS, etc. through one request/parse
        # path. Auth is AWS SigV4 (not a bearer key), so credentials come from
        # the AWS environment, not saved config.
        "kind": "aws",
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "label": "AWS Bedrock — any model via Converse (AWS credentials)",
    },
    "local": {
        "kind": "local-proxy",
        "model": "gpt-oss-20b",
        "key_env": "LOCAL_API_KEY",
        "label": "Local proxy (Anthropic-compatible server on localhost)",
    },
    "ollama": {
        "kind": "local-proxy",
        "model": "llama3.2",
        "label": "Ollama (local models via `ollama serve`)",
    },
    "transformers": {
        "kind": "local-ml",
        "model": "deburky/gpt-oss-claude-code",
        "label": "HuggingFace Transformers (CPU/GPU, source install)",
    },
    "mlx": {
        "kind": "local-ml",
        "model": "deburky/gpt-oss-claude-mlx",
        "label": "Apple Silicon via MLX (source install)",
    },
}

# Curated model lists for arrow-key pickers (OpenRouter/Ollama are fetched live).
BACKEND_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "o1-mini",
        "o1",
    ],
    "openrouter": [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.1-8b-instruct",
    ],
    "local": ["gpt-oss-20b"],
    "transformers": ["deburky/gpt-oss-claude-code"],
    "mlx": ["deburky/gpt-oss-claude-mlx"],
    "bedrock": [
        # Converse is model-agnostic, but wrencode is a TOOL-USING agent and
        # tool-call support varies by family. Verified end-to-end on AWS (text +
        # tool use): Claude and Amazon Nova — the recommended choices. The "us."
        # prefix is a cross-region inference profile; match it to your AWS_REGION's
        # geo, or type any custom Converse model id at the prompt.
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",  # verified (default)
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",  # verified
        "us.amazon.nova-pro-v1:0",  # verified
        # Experimental — chat works but tool use is unreliable, so the agent loop
        # may stall: Llama emits no tool_use; GPT-OSS is flaky on tools.
        "openai.gpt-oss-20b-1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
        # TODO: no verified tool-using non-Claude/Nova family yet. Mistral Large
        # 2407 was dropped (invalid Bedrock model id); revisit if a Converse
        # tool-capable Mistral/other id is confirmed.
    ],
}

CUSTOM_MODEL_OPTION = "— type a custom model id —"
_MLX_UNCHANGED = object()
API_BACKENDS: frozenset[str] = frozenset(
    name for name, spec in BACKEND_SPECS.items() if spec["kind"] == "api"
)
LOCAL_ML_BACKENDS: frozenset[str] = frozenset(
    name for name, spec in BACKEND_SPECS.items() if spec["kind"] == "local-ml"
)
# Backends whose responses use the Anthropic Messages format (content blocks,
# tool_use, usage). Bedrock speaks the model-agnostic Converse API instead, so
# it has its own format/parse path (see _bedrock_converse_call).
ANTHROPIC_FORMAT_BACKENDS: frozenset[str] = frozenset({"anthropic"})
# Backends that return JSON with native tool calls (vs. XML-in-text), parsed by
# _parse_native_response. Bedrock/Converse is native too.
NATIVE_TOOL_BACKENDS: frozenset[str] = frozenset({"anthropic", "openai", "bedrock"})
# Hosted backends reached over HTTP (vs. in-process local-ml weights). Bedrock
# is kind "aws" so it isn't in API_BACKENDS, but it's still a network call.
HOSTED_BACKENDS: frozenset[str] = API_BACKENDS | frozenset({"bedrock"})

CONFIG_DIR = pathlib.Path(
    os.environ.get("WRENCODE_CONFIG_DIR", "~/.wrencode")
).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.json"
OPENROUTER_MODELS_CACHE = CONFIG_DIR / "openrouter_models.json"

# Populated by apply_backend() once configuration is resolved (see resolve_configuration).
BACKEND = ""
MODEL = ""
API_KEY = ""
API_BASE = ""
AWS_REGION = ""
LOCAL_PORT = os.environ.get("LOCAL_PORT", "8082")

# Backend libraries are imported lazily inside load_model(); the rest of the
# module references them as globals. Declared here as Any so the module
# type-checks even when the heavy optional deps aren't installed.
load: Any = None  # mlx_lm.load
stream_generate: Any = None  # mlx_lm.generate.stream_generate
make_sampler: Any = None  # mlx_lm.sample_utils.make_sampler
torch: Any = None  # torch
AutoModelForCausalLM: Any = None  # transformers.AutoModelForCausalLM
AutoTokenizer: Any = None  # transformers.AutoTokenizer

# Subagent state: the loaded model (set in main) and a recursion-depth guard.
_MLX_STATE: Optional[tuple[Any, Any]] = None
_SUBAGENT_DEPTH = 0
MAX_SUBAGENT_DEPTH = int(os.environ.get("WRENCODE_MAX_SUBAGENT_DEPTH", "2"))


def apply_backend(backend: str, model: str = "", api_key: str = "") -> None:
    """Set the module-level backend globals from a backend name plus overrides.

    Precedence for each value: explicit environment variable > saved/chosen
    value > built-in default. Heavy backend imports are deferred to load_model().
    """
    global BACKEND, MODEL, API_KEY, API_BASE, AWS_REGION, LOCAL_PORT
    spec = BACKEND_SPECS[backend]
    BACKEND = backend
    MODEL = os.environ.get("MODEL") or model or spec["model"]
    if spec["kind"] == "api":
        API_KEY = os.environ.get(spec["key_env"]) or api_key or ""
        API_BASE = spec["api_base"]
    elif spec["kind"] == "aws":
        # Bedrock: creds + region come from the AWS environment at request
        # time (SigV4), so nothing is stored in API_KEY. API_BASE is built
        # per-request from region + model id in _bedrock_converse_call().
        AWS_REGION = _aws_region()
        API_KEY = ""
        API_BASE = ""
    elif backend == "ollama":
        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        API_KEY = "ollama"  # Ollama ignores the key; kept non-empty for the loader
        API_BASE = f"{base}/v1/chat/completions"
    elif backend == "local":
        LOCAL_PORT = os.environ.get("LOCAL_PORT", "8082")
        API_KEY = os.environ.get("LOCAL_API_KEY") or api_key or "local"
        API_BASE = f"http://localhost:{LOCAL_PORT}/v1/messages"
    else:  # local-ml (mlx / transformers): no key, weights loaded in-process
        API_KEY = ""
        API_BASE = ""


# -----------------------------------------------------------------------------------------------
# Constants & environment variables
# -----------------------------------------------------------------------------------------------
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
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
# Terminal colors
# -----------------------------------------------------------------------------------------------
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)
BRIGHT_CYAN = "\033[96m"
AGENT_TEXT = "\033[38;5;245m"  # muted grey for agent replies (Claude Code-style)
_COMPOSE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_LOADER_MODEL_MAX = 32


def _loader_context_text() -> str:
    b = BACKEND or "backend"
    if MODEL and len(MODEL) > _LOADER_MODEL_MAX:
        keep = _LOADER_MODEL_MAX - 1
        head = max(8, keep // 2)
        m = f"{MODEL[:head]}…{MODEL[-(keep - head) :]}"
    else:
        m = MODEL or "model"
    return f"{b} · {m} · waiting…"


def loader_display(step: int) -> str:
    """Render one animated loader frame for the given step."""
    sym = _COMPOSE_FRAMES[step % len(_COMPOSE_FRAMES)]
    text = _loader_context_text()
    if not colors_enabled():
        return f"{sym} {text}"
    return f"{BRIGHT_CYAN}{sym}{RESET} {DIM}{text}{RESET}"


def colors_enabled() -> bool:
    """Return True when ANSI styling should be applied.

    Precedence mirrors CPython's private _colorize.can_colorize:
    PYTHON_COLORS > NO_COLOR > FORCE_COLOR > TERM=dumb > isatty.
    """
    if (py_colors := os.environ.get("PYTHON_COLORS")) in {"0", "1"}:
        return py_colors == "1"
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def print_system(text: str, *, end: str = "\n") -> None:
    """Print slash-command / configure feedback in banner cyan."""
    s = f"{BOLD}{BRIGHT_CYAN}" if colors_enabled() else ""
    e = RESET if colors_enabled() else ""
    if sys.stdout.isatty():
        sys.stdout.write("\r")
    sys.stdout.write(f"{s}{text}{e}{end}")
    sys.stdout.flush()


_INPUT_HISTORY: list[str] = []


def format_input_line(text: str) -> str:
    """Render the ❯ prompt line; only the /command token is bold cyan while typing."""
    if not colors_enabled():
        return f"❯ {text}"
    prompt = f"{BRIGHT_CYAN}❯{RESET} "
    if not text.startswith("/"):
        return prompt + text
    cmd, _, rest = text.partition(" ")
    s = f"{BOLD}{BRIGHT_CYAN}"
    line = prompt + s + cmd + RESET
    if rest:
        line += " " + rest
    return line


def _read_input_char(fd: int) -> str:
    """Read one UTF-8 character from fd (raw/cbreak mode)."""
    first = os.read(fd, 1)
    if not first:
        return ""
    extra = 0
    lead = first[0]
    if lead & 0x80:
        if lead & 0xE0 == 0xC0:
            extra = 1
        elif lead & 0xF0 == 0xE0:
            extra = 2
        elif lead & 0xF8 == 0xF0:
            extra = 3
    if extra:
        first += os.read(fd, extra)
    return first.decode("utf-8", errors="replace")


def _redraw_input_line(text: str) -> None:
    sys.stdout.write("\r\033[K" + format_input_line(text))
    sys.stdout.flush()


def _read_tty_key(fd: int) -> str:
    """Read one key; arrow keys return up/down/left/right instead of escape junk."""
    ch = _read_input_char(fd)
    if not ch:
        return ""
    if ch == "\x1b":
        if not select.select([fd], [], [], 0.02)[0]:
            return "esc"
        seq = os.read(fd, 1)
        if seq != b"[":
            return "esc"
        if not select.select([fd], [], [], 0.02)[0]:
            return "esc"
        code = os.read(fd, 1)
        if code == b"A":
            return "up"
        if code == b"B":
            return "down"
        if code == b"C":
            return "right"
        if code == b"D":
            return "left"
        return "esc"
    if ch in "\r\n":
        return "enter"
    if ch in ("\x7f", "\x08"):
        return "backspace"
    if ch == "\x03":
        return "ctrl_c"
    if ch == "\x04":
        return "ctrl_d"
    return ch


def _read_tty_line(
    prompt: str,
    *,
    history: bool = False,
    redraw: Optional[Any] = None,
) -> str:
    """Read one line in cbreak mode; swallows arrow keys unless history=True."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []
    hist_idx = len(_INPUT_HISTORY)

    def _redraw() -> None:
        if redraw is not None:
            redraw("".join(buf))
        else:
            sys.stdout.write("\r\033[K" + prompt + "".join(buf))
            sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        sys.stdout.write(prompt)
        sys.stdout.flush()
        _redraw()
        while True:
            if not select.select([fd], [], [], None)[0]:
                continue
            key = _read_tty_key(fd)
            if not key:
                raise EOFError
            if key == "enter":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buf).strip()
            if key == "backspace":
                if buf:
                    buf.pop()
                    _redraw()
                continue
            if key == "ctrl_c":
                sys.stdout.write("\n")
                sys.stdout.flush()
                raise KeyboardInterrupt
            if key == "ctrl_d":
                if not buf:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    raise EOFError
                continue
            if key == "up" and history and _INPUT_HISTORY:
                if hist_idx > 0:
                    hist_idx -= 1
                    buf = list(_INPUT_HISTORY[hist_idx])
                    _redraw()
                continue
            if key == "down" and history:
                if hist_idx < len(_INPUT_HISTORY):
                    hist_idx += 1
                    buf = (
                        list(_INPUT_HISTORY[hist_idx])
                        if hist_idx < len(_INPUT_HISTORY)
                        else []
                    )
                    _redraw()
                continue
            if key in ("up", "down", "left", "right", "esc"):
                continue
            if len(key) == 1 and (key.isprintable() or key == "\t"):
                buf.append(key)
                _redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _remember_input(text: str) -> None:
    if text and (not _INPUT_HISTORY or _INPUT_HISTORY[-1] != text):
        _INPUT_HISTORY.append(text)


def _read_user_input_interactive() -> str:
    """TTY line editor with live slash-command coloring and history."""
    text = _read_tty_line("", history=True, redraw=_redraw_input_line)
    _remember_input(text)
    return text


def read_feedback_line() -> str:
    """Read decline feedback after choosing n in the approval picker."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        sys.stdout.write(f"\n{YELLOW}What should I do differently?{RESET}\n")
        sys.stdout.flush()
        return _read_tty_line(f"{BRIGHT_CYAN}❯{RESET} ")
    prompt = f"{YELLOW}What should I do differently?{RESET} "
    return input(prompt).strip()


def read_user_input() -> str:
    """Read one line from the ❯ prompt with live slash-command coloring."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _read_user_input_interactive()

    if colors_enabled():
        sys.stdout.write(f"{BRIGHT_CYAN}❯{RESET} ")
    else:
        sys.stdout.write("❯ ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.rstrip("\r\n").strip()


# Set by confirm() when the user chooses "allow all" for the rest of the session.
_SESSION_AUTO_APPROVE = False

# Escape (or Ctrl+C during a turn) sets this so the agent loop returns to the prompt.
_CANCEL_REQUESTED = threading.Event()
_LISTENER_STOP = threading.Event()


class UserCancelled(Exception):
    """Raised when the user presses Escape or Ctrl+C during an agent turn."""


WREN_BANNER = f"""{BRIGHT_CYAN}\u2588\u2588     \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588 \u2588\u2588\u2588    \u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588
\u2588\u2588     \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588      \u2588\u2588\u2588\u2588   \u2588\u2588 \u2588\u2588      \u2588\u2588    \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588
\u2588\u2588  \u2588  \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588   \u2588\u2588 \u2588\u2588  \u2588\u2588 \u2588\u2588      \u2588\u2588    \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588\u2588\u2588\u2588
\u2588\u2588 \u2588\u2588\u2588 \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588      \u2588\u2588  \u2588\u2588 \u2588\u2588 \u2588\u2588      \u2588\u2588    \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588
 \u2588\u2588\u2588 \u2588\u2588\u2588  \u2588\u2588   \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588\u2588 \u2588\u2588   \u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588
{RESET}"""


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
    args: dict[str, Any], key: str, default: Optional[int] = None
) -> Optional[int]:
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
    approval = confirm("write")
    if approval != "ok":
        return approval
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    return "ok"


def edit(args: dict[str, Any]) -> str:
    """Replace a unique string in a file with a new string."""
    path = resolve_tool_path(_require_str(args, "path"))
    old = _require_str(args, "old")
    new = str(args.get("new", ""))
    if not path.is_file():
        return f"error: not a file: {path}"
    if path.stat().st_size > MAX_READ_BYTES:
        return f"error: file too large (max {MAX_READ_BYTES} bytes)"
    text = path.read_text(encoding="utf-8", errors="replace")
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times (use all=true)"
    updated = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
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
    approval = confirm("edit")
    if approval != "ok":
        return approval
    path.write_text(updated, encoding="utf-8")
    return "ok"


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
    except Exception as exc:
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
        [tool, "-n", "--color", "never", "--no-heading", "-e", pat, scope]
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


def _cancel_listener() -> None:
    """Watch stdin for Escape while a blocking agent operation runs."""
    if not sys.stdin.isatty():
        return
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not _LISTENER_STOP.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = os.read(fd, 1).decode("utf-8", errors="replace")
                if ch == "\x1b":
                    _CANCEL_REQUESTED.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        pass


@contextlib.contextmanager
def cancel_watch() -> Any:
    """Enable Escape-to-cancel for the duration of a blocking operation."""
    if not sys.stdin.isatty():
        yield
        return
    _CANCEL_REQUESTED.clear()
    _LISTENER_STOP.clear()
    listener = threading.Thread(target=_cancel_listener, daemon=True)
    listener.start()
    try:
        yield
    finally:
        _LISTENER_STOP.set()
        listener.join(timeout=0.5)


def check_cancelled() -> None:
    """Raise UserCancelled if the user requested cancellation."""
    if _CANCEL_REQUESTED.is_set():
        raise UserCancelled()


def get_response_cancellable(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: Optional[tuple[Any, Any]],
) -> str:
    """Run get_response in a worker thread so Escape can interrupt blocking calls."""
    if not sys.stdin.isatty():
        return get_response(messages, system_prompt, mlx_state)

    result: list[str] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(get_response(messages, system_prompt, mlx_state))
        except BaseException as exc:  # noqa: BLE001 — propagate to caller
            error.append(exc)

    with cancel_watch():
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            check_cancelled()
            t.join(timeout=0.15)
    if error:
        raise error[0]
    return result[0]


def confirm(action: str = "") -> str:
    """Prompt for approval. Returns 'ok' or a cancellation message for the agent.

    Enter/y approves once; ``a`` approves all remaining actions this session;
    ``n`` declines and asks what to do differently. Auto-approve via
    WRENCODE_AUTO_APPROVE / --yes enables headless use and subagents.
    """
    global _SESSION_AUTO_APPROVE
    if (
        os.environ.get("WRENCODE_AUTO_APPROVE", "").lower() in ("1", "true", "yes")
        or _SESSION_AUTO_APPROVE
    ):
        label = action or "action"
        print(f"{DIM}⚠ {label} [auto-approved]{RESET}")
        return "ok"
    print(f"{DIM}  Enter/y   approve once{RESET}")
    print(f"{DIM}  a         allow all for this session{RESET}")
    print(f"{DIM}  n         decline{RESET}")
    while True:
        try:
            choice = input(f"{BLUE}❯{RESET} ").strip().lower()
        except KeyboardInterrupt:
            print()
            return "cancelled: user interrupted"
        if choice in ("", "y", "yes"):
            return "ok"
        if choice in ("a", "all"):
            _SESSION_AUTO_APPROVE = True
            print(f"{DIM}Auto-approving remaining actions this session.{RESET}")
            return "ok"
        if choice in ("n", "no"):
            try:
                feedback = read_feedback_line()
            except KeyboardInterrupt:
                print()
                return "cancelled: user interrupted"
            if feedback:
                return f"cancelled: user declined — {feedback}"
            return "cancelled: user declined without instructions"
        print(f"{DIM}Choose Enter, a, or n.{RESET}")


def bash(args: dict[str, Any]) -> str:
    """Run a shell command with a timeout, streaming output to the terminal."""
    cmd = _require_str(args, "cmd")
    approval = confirm("run")
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
                print(f"{DIM}│ {line.rstrip()}{RESET}", flush=True)

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
    return "".join(output_lines).strip() or "(empty)"


def task(args: dict[str, Any]) -> str:
    """Run a subagent: a fresh agent loop over a self-contained subtask.

    The subagent shares the workspace and tool set but has its own (empty)
    message history, so the parent's context only grows by the returned result.
    Recursion is capped by MAX_SUBAGENT_DEPTH. For autonomous use run with
    --yes / WRENCODE_AUTO_APPROVE, else each sub-tool call still asks to confirm.
    """
    global _SUBAGENT_DEPTH
    if _SUBAGENT_DEPTH >= MAX_SUBAGENT_DEPTH:
        return f"error: max subagent depth ({MAX_SUBAGENT_DEPTH}) reached"
    prompt = _require_str(args, "prompt")
    _SUBAGENT_DEPTH += 1
    print(f"{CYAN}  ↳ subagent:{RESET}{DIM} {prompt[:70]}{RESET}")
    sub: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        run_agent_turn(sub, build_system_prompt(), _MLX_STATE, max_iters=12)
    finally:
        _SUBAGENT_DEPTH -= 1
    texts = [flatten_content(m["content"]) for m in sub if m["role"] == "assistant"]
    print(f"{CYAN}  ↳ subagent done{RESET}")
    return (texts[-1] if texts else "") or "(subagent produced no text output)"


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
        "Delegate a self-contained subtask to a fresh subagent (same tools, "
        "own context); returns only its final result",
        {"prompt": "string"},
        task,
    ),
}


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
    return out


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
        return f"write {path}  ({lines} lines)\n  {preview}{suffix}"
    if name == "edit":
        path = args.get("path", "?")
        old = str(args.get("old", ""))[:80].replace("\n", "\\n")
        new = str(args.get("new", ""))[:80].replace("\n", "\\n")
        return f"edit {path}\n  - {old}\n  + {new}"
    if name == "glob":
        return f"glob {args.get('pat', args.get('pattern', '?'))}"
    if name == "grep":
        return f"grep {args.get('pat', '?')}"
    if name == "task":
        prompt = str(args.get("prompt", "")).strip()
        return f"task {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
    return f"{name}({json.dumps(args, ensure_ascii=False)[:200]})"


def print_tool_action(name: str, args: dict[str, Any]) -> None:
    """Print a tool call as plain text — no background boxes."""
    body = format_tool_action(name, args)
    first, _, rest = body.partition("\n")
    print(f"{GREEN}⏺{RESET}{DIM} {first}{RESET}")
    for line in rest.split("\n"):
        if line.strip():
            print(f"{DIM}  {line}{RESET}")


def print_tool_result(result: str) -> None:
    """Print tool output with enough context to see what happened."""
    lines = result.split("\n")
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
    try:
        result = TOOLS[name][2](normalize_tool_args(name, args))
        if len(result) > MAX_OUT:
            result = (
                result[:MAX_OUT]
                + f"\n... [truncated {len(result) - MAX_OUT} chars; raise MAX_TOOL_OUTPUT_CHARS]"
            )
        return result
    except Exception as e:
        return f"error: {e}"


# -----------------------------------------------------------------------------------------------
# Message formatting
# -----------------------------------------------------------------------------------------------
def flatten_content(content: Any) -> str:
    """Flatten a content list to plain string (Anthropic or Bedrock Converse blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block["text"])
        elif btype == "tool_use":
            parts.append(
                f'<tool_call>{{"tool": "{block["name"]}", "args": {json.dumps(block["input"])}}}</tool_call>'
            )
        elif btype == "tool_result":
            parts.append(f"Tool result: {block.get('content', '')}")
        # Bedrock Converse blocks have no "type" key.
        elif "text" in block:
            parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            parts.append(
                f'<tool_call>{{"tool": "{tu.get("name", "")}", "args": {json.dumps(tu.get("input", {}))}}}</tool_call>'
            )
        elif "toolResult" in block:
            tr = block["toolResult"]
            txt = " ".join(
                c.get("text", "") for c in tr.get("content", []) if isinstance(c, dict)
            )
            parts.append(f"Tool result: {txt}")
    return "\n".join(parts)


# Lightweight, language-agnostic code coloring — no pygments, keeps the binary lean.
_CODE_TOKEN = re.compile(
    r"(?P<comment>#[^\n]*|//[^\n]*)"
    r"|(?P<string>\"[^\"\n]*\"|'[^'\n]*'|`[^`\n]*`)"
    r"|(?P<num>\b\d[\d_.]*\b)"
    r"|(?P<kw>\b(?:def|class|return|import|from|if|elif|else|for|while|try|except|"
    r"finally|with|as|in|not|and|or|is|lambda|yield|async|await|pass|break|continue|"
    r"raise|global|nonlocal|assert|True|False|None|const|let|var|function|export|"
    r"default|new|this|fn|match|struct|enum|pub|use|mut|public|private|static|void)\b)"
)


def _highlight_code(code: str) -> str:
    """Apply light ANSI syntax coloring to a code block (best-effort, any language)."""

    def color(m: "re.Match[str]") -> str:
        g = m.lastgroup
        if g == "comment":
            return f"{DIM}{m.group()}{RESET}"
        if g == "string":
            return f"{GREEN}{m.group()}{RESET}"
        if g == "num":
            return f"{YELLOW}{m.group()}{RESET}"
        if g == "kw":
            return f"{BLUE}{m.group()}{RESET}"
        return m.group()

    return _CODE_TOKEN.sub(color, code)


def render_markdown(text: str) -> str:
    """Render fenced code blocks (lightly highlighted), inline code, and bold."""
    blocks: list[str] = []

    def stash(m: "re.Match[str]") -> str:
        lang = m.group(1) or ""
        body = _highlight_code(m.group(2).rstrip("\n"))
        head = f"{DIM}┌─ {lang}{RESET}\n" if lang else f"{DIM}┌─{RESET}\n"
        bordered = "\n".join(f"{DIM}│{RESET} {ln}" for ln in body.split("\n"))
        blocks.append(f"\n{head}{bordered}\n{DIM}└─{RESET}")
        return f"\x00B{len(blocks) - 1}\x00"

    text = re.sub(r"```(\w*)\n?(.*?)```", stash, text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", f"{CYAN}\\1{RESET}", text)
    text = re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)
    for i, b in enumerate(blocks):
        text = text.replace(f"\x00B{i}\x00", b)
    return text


def print_agent_message(text: str) -> None:
    """Print the agent response in muted grey text — no label, no box."""
    for line in render_markdown(text).split("\n"):
        print(f"{AGENT_TEXT}{line}{RESET}")
    print()


@contextlib.contextmanager
def thinking_spinner() -> Any:
    """Loader on the line below the user's input (style from /loader)."""
    if not sys.stdout.isatty():
        yield
        return

    stop = threading.Event()
    step = 0

    def animate() -> None:
        nonlocal step
        while not stop.is_set():
            bar = loader_display(step)
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


# -----------------------------------------------------------------------------------------------
# Token & output cleaning
# -----------------------------------------------------------------------------------------------
def strip_gptoss_tokens(text: str) -> str:
    """Strip GPT-OSS special tokens and channel markers from model output."""
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


def truncate_at_turn_leak(text: str) -> str:
    """Truncate text at the first sign of a leaked conversation turn marker."""
    return next(
        (
            text.split(m)[0].strip()
            for m in (
                "\nUser:",
                "\nSystem:",
                "\nHuman:",
                "\n\nUser:",
                "\n\nSystem:",
            )
            if m in text
        ),
        text,
    )


def _tool_call_complete(text: str) -> int:
    """Return end index of first complete <tool_call> block, or -1."""
    start = text.find("<tool_call>")
    if start == -1:
        return -1
    end_tag = text.find("</tool_call>", start)
    if end_tag != -1:
        return end_tag + len("</tool_call>")
    brace_start = text.find("{", start)
    if brace_start == -1:
        return -1
    depth, last = 0, -1
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last = i + 1
                break
    return last


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
        payload: Optional[dict[str, Any]] = None
        with contextlib.suppress(Exception):
            obj, rel_end = json.JSONDecoder().raw_decode(text, brace)
            if isinstance(obj, dict):
                payload = obj if obj.get("tool") in TOOLS else None
                pos = rel_end
                if payload:
                    calls.append(
                        {
                            "type": "tool_use",
                            "id": f"call_{len(calls)}",
                            "name": payload["tool"],
                            "input": payload.get("args", {}),
                        }
                    )
                continue
        if close != -1 and close > brace:
            payload = None
            with contextlib.suppress(Exception):
                d = json.loads(text[brace:close])
                if isinstance(d, dict):
                    payload = d
            pos = close + len(close_tag)
        else:
            break  # JSON may still be streaming
        if payload and payload.get("tool") in TOOLS:
            calls.append(
                {
                    "type": "tool_use",
                    "id": f"call_{len(calls)}",
                    "name": payload["tool"],
                    "input": payload.get("args", {}),
                }
            )
    return calls


# -----------------------------------------------------------------------------------------------
# Anthropic native tools
# -----------------------------------------------------------------------------------------------
_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "string?": "string",
    "number": "integer",
    "number?": "integer",
    "boolean": "boolean",
    "boolean?": "boolean",
}


def _build_tool_schemas(fmt: str) -> list[dict[str, Any]]:
    """Build tool definitions for 'anthropic', 'converse', or 'openai' format."""
    out = []
    for name, (desc, params, _) in TOOLS.items():
        props = {k: {"type": _TYPE_MAP.get(v, "string")} for k, v in params.items()}
        req = [k for k, v in params.items() if not v.endswith("?")]
        schema = {"type": "object", "properties": props, "required": req}
        if fmt == "anthropic":
            out.append({"name": name, "description": desc, "input_schema": schema})
        elif fmt == "converse":
            out.append({"toolSpec": {
                "name": name, "description": desc, "inputSchema": {"json": schema},
            }})
        else:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": schema,
                    },
                }
            )
    return out


def _anthropic_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }


def _openai_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}


def _warn_if_truncated(data: dict[str, Any]) -> None:
    """Print a stderr warning if the API truncated the response at max_tokens.

    A truncated `tool_use` may be missing required fields (e.g. `content` for
    `write`), so callers need to know to raise MAX_TOKENS rather than silently
    treating an empty operation as success.
    """
    if BACKEND == "bedrock":
        truncated = data.get("stopReason") == "max_tokens"
        out_tokens = data.get("usage", {}).get("outputTokens")
    elif BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        truncated = data.get("stop_reason") == "max_tokens"
        out_tokens = data.get("usage", {}).get("output_tokens")
    else:
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        truncated = finish == "length"
        out_tokens = data.get("usage", {}).get("completion_tokens")
    if truncated:
        print(
            f"{YELLOW}Warning: response truncated at MAX_TOKENS={MAX_TOKENS} "
            f"(output_tokens={out_tokens}). Tool calls may be incomplete — "
            f"raise MAX_TOKENS and retry.{RESET}",
            file=sys.stderr,
        )


def _log_usage_debug(data: dict[str, Any]) -> None:
    """When WRENCODE_DEBUG=1, print one stderr line per turn with token usage.

    Surfaces cache_creation_input_tokens / cache_read_input_tokens so prompt-
    cache hit rate is observable without external tooling.
    """
    if not os.environ.get("WRENCODE_DEBUG"):
        return
    u = data.get("usage", {})
    if BACKEND == "bedrock":
        print(
            f"{DIM}usage: in={u.get('inputTokens')} out={u.get('outputTokens')}{RESET}",
            file=sys.stderr,
        )
        return
    if BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        print(
            f"{DIM}usage: in={u.get('input_tokens')} out={u.get('output_tokens')} "
            f"cache_write={u.get('cache_creation_input_tokens', 0)} "
            f"cache_read={u.get('cache_read_input_tokens', 0)}{RESET}",
            file=sys.stderr,
        )
    else:
        print(
            f"{DIM}usage: in={u.get('prompt_tokens')} out={u.get('completion_tokens')}{RESET}",
            file=sys.stderr,
        )


def _parse_native_response(data: dict[str, Any]) -> tuple[str, list["ToolCall"]]:
    """Parse a native (Anthropic / Bedrock Converse / OpenAI) response into text + tool calls."""
    _warn_if_truncated(data)
    _log_usage_debug(data)
    if BACKEND == "bedrock":
        blocks = data.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(b["text"] for b in blocks if "text" in b).strip()
        calls = [
            ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"],
                     b["toolUse"].get("input", {}))
            for b in blocks if "toolUse" in b
        ]
        return text, calls
    if BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        blocks = data.get("content", [])
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text").strip()
        calls = [
            ToolCall(b["id"], b["name"], b.get("input", {}))
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        return text, calls
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    calls = []
    for tc in msg.get("tool_calls") or []:
        with contextlib.suppress(Exception):
            calls.append(
                ToolCall(
                    tc["id"],
                    tc["function"]["name"],
                    json.loads(tc["function"]["arguments"]),
                )
            )
    return text, calls


@dataclass
class ToolCall:
    """A single tool invocation parsed from a model response."""

    id: str
    name: str
    input: dict[str, Any]


def _parse_response(
    response_text: str,
) -> tuple[str, list[ToolCall], Any]:
    """Parse a raw API response into (display_text, tool_calls, raw_data).

    raw_data is the decoded JSON for native backends (used when appending to
    history); None for XML backends.
    """
    if BACKEND in NATIVE_TOOL_BACKENDS:
        data = json.loads(response_text)
        text, calls = _parse_native_response(data)
        return text, calls, data
    text = re.sub(
        r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL
    ).strip()
    calls = [
        ToolCall(tc["id"], tc["name"], tc["input"])
        for tc in parse_tool_calls(response_text)
    ]
    return text, calls, None


def _append_assistant(
    messages: list[dict[str, Any]],
    text: str,
    tool_calls: list[ToolCall],
    raw_data: Any,
) -> None:
    """Append the assistant turn to message history in the correct format."""
    if BACKEND == "bedrock":
        content = raw_data.get("output", {}).get("message", {}).get("content", [])
        # Keep only text + toolUse blocks; echoing other block types (e.g.
        # reasoningContent) back into the next Converse request can 400.
        kept = [b for b in content if "text" in b or "toolUse" in b]
        messages.append({"role": "assistant", "content": kept})
    elif BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        messages.append({"role": "assistant", "content": raw_data.get("content", [])})
    elif BACKEND == "openai":
        messages.append(raw_data["choices"][0]["message"])  # preserve tool_calls
    else:  # XML path
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        for tc in tool_calls:
            blocks.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
            )
        messages.append({"role": "assistant", "content": blocks})


def _append_tool_results(
    messages: list[dict[str, Any]],
    results: list[tuple[ToolCall, str]],
) -> None:
    """Append tool results to message history in the correct format."""
    if BACKEND == "openai":
        messages.extend(
            {"role": "tool", "tool_call_id": tc.id, "content": r} for tc, r in results
        )
    elif BACKEND == "bedrock":
        messages.append({
            "role": "user",
            "content": [
                {"toolResult": {
                    "toolUseId": tc.id,
                    "content": [{"text": r}],
                    "status": "success",
                }}
                for tc, r in results
            ],
        })
    else:  # anthropic + XML path both use tool_result blocks
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": r}
                    for tc, r in results
                ],
            }
        )


# -----------------------------------------------------------------------------------------------
# HTTP helper
# -----------------------------------------------------------------------------------------------
def _http_post_raw(url: str, data: bytes, headers: dict[str, str]) -> Any:
    """POST pre-encoded bytes to a URL and return the parsed JSON response."""
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP {e.code}: {e.read().decode()}") from e


def _http_post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
    """POST a JSON payload to a URL and return the parsed response."""
    return _http_post_raw(url, json.dumps(payload).encode(), headers)


# -----------------------------------------------------------------------------------------------
# AWS Bedrock: credentials + SigV4 request signing (stdlib only — no boto3)
# -----------------------------------------------------------------------------------------------
def _aws_region() -> str:
    """Resolve the AWS region: env, then ~/.aws/config for the active profile, else us-east-1."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region
    with contextlib.suppress(Exception):
        import configparser

        cfg = configparser.ConfigParser()
        cfg.read(os.path.expanduser("~/.aws/config"))
        profile = os.environ.get("AWS_PROFILE", "default")
        section = profile if profile == "default" else f"profile {profile}"
        if cfg.has_option(section, "region"):
            return cfg.get(section, "region")
    return "us-east-1"


def _aws_credentials() -> tuple[str, str, str]:
    """Resolve (access_key, secret_key, session_token) from env, then ~/.aws/credentials."""
    ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    token = os.environ.get("AWS_SESSION_TOKEN", "")
    if ak and sk:
        return ak, sk, token
    with contextlib.suppress(Exception):
        import configparser

        creds = configparser.ConfigParser()
        creds.read(os.path.expanduser("~/.aws/credentials"))
        profile = os.environ.get("AWS_PROFILE", "default")
        if creds.has_section(profile):
            ak = ak or creds.get(profile, "aws_access_key_id", fallback="")
            sk = sk or creds.get(profile, "aws_secret_access_key", fallback="")
            token = token or creds.get(profile, "aws_session_token", fallback="")
    return ak, sk, token


def _sigv4_authorization(
    method: str,
    canonical_uri: str,
    canonical_qs: str,
    headers: dict[str, str],
    payload_hash: str,
    service: str,
    region: str,
    amz_date: str,
    access_key: str,
    secret_key: str,
) -> tuple[str, str]:
    """Compute (Authorization header value, signed-headers list) for AWS SigV4.

    `headers` keys must be lowercase. Pure stdlib hashlib/hmac — this is the
    core algorithm, kept side-effect-free so it can be tested against AWS's
    published signing vectors.
    """
    datestamp = amz_date[:8]
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in sorted(headers))
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            canonical_qs,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k_date = _hmac(("AWS4" + secret_key).encode(), datestamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()
    auth = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return auth, signed_headers


def _sigv4_signed_headers(
    method: str, url: str, body: bytes, service: str, region: str
) -> dict[str, str]:
    """Build the full set of SigV4-signed request headers for a Bedrock call."""
    access_key, secret_key, token = _aws_credentials()
    if not (access_key and secret_key):
        raise Exception(
            "AWS credentials not found — set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY (and AWS_REGION)."
        )
    parsed = urllib.parse.urlsplit(url)
    payload_hash = hashlib.sha256(body).hexdigest()
    amz_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # SigV4 canonical URI: non-S3 services double-encode the (already
    # percent-encoded) path — a Bedrock model id's ':' goes %3A -> %253A.
    # Re-quoting the wire path encodes the '%' signs while leaving '/' intact.
    canonical_uri = urllib.parse.quote(parsed.path or "/", safe="/~")
    sign_these = {
        "host": parsed.netloc,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if token:
        sign_these["x-amz-security-token"] = token
    auth, _ = _sigv4_authorization(
        method,
        canonical_uri,
        parsed.query,
        sign_these,
        payload_hash,
        service,
        region,
        amz_date,
        access_key,
        secret_key,
    )
    headers = {
        "Content-Type": "application/json",
        "X-Amz-Date": amz_date,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": auth,
    }
    if token:
        headers["X-Amz-Security-Token"] = token
    return headers


def _bedrock_converse_call(body: dict[str, Any]) -> Any:
    """POST a Bedrock Converse body to the signed /converse endpoint, return JSON."""
    region = _aws_region()
    model_id = urllib.parse.quote(MODEL, safe="")
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"
    raw = json.dumps(body).encode()
    # The runtime host is bedrock-runtime.*, but the SigV4 signing service is "bedrock".
    headers = _sigv4_signed_headers("POST", url, raw, "bedrock", region)
    return _http_post_raw(url, raw, headers)


def _to_converse_message(m: dict[str, Any]) -> dict[str, Any]:
    """Normalize a stored message to Converse shape (string content -> [{text}])."""
    content = m["content"]
    if isinstance(content, str):
        content = [{"text": content}]
    return {"role": m["role"], "content": content}


# -----------------------------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------------------------
def get_response(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: Optional[tuple[Any, Any]],
) -> str:
    """Generate a response from the configured backend given the message history."""
    flat = [
        {"role": m["role"], "content": flatten_content(m["content"])} for m in messages
    ]

    # OpenAI - native function calling
    if BACKEND == "openai":
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
                "tools": _build_tool_schemas("openai"),
                "tool_choice": "auto",
            },
            _openai_headers(),
        )
        return json.dumps(data)  # return raw for agent loop to parse natively

    # OpenRouter / Ollama - OpenAI-compatible chat completions (no native tools)
    if BACKEND in {"openrouter", "ollama"}:
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + flat,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
            },
            _openai_headers(),
        )
        return str(data["choices"][0]["message"]["content"])

    # AWS Bedrock — model-agnostic Converse API (Claude, Llama, Nova, GPT-OSS…).
    if BACKEND == "bedrock":
        tools = _build_tool_schemas("converse")
        body: dict[str, Any] = {
            "messages": [_to_converse_message(m) for m in messages],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS, "temperature": 0.3},
        }
        if tools:
            body["toolConfig"] = {"tools": tools}
        data = _bedrock_converse_call(body)
        return json.dumps(data)  # return raw for agent loop to parse natively

    # Anthropic native tool use API.
    if BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        # Prompt caching: mark the (large, static) system block + last tool
        # schema as ephemeral. Anthropic caches everything up to each marker
        # for ~5 minutes; subsequent turns read at ~10% of normal input cost.
        # Two breakpoints of the four allowed per request.
        tools = _build_tool_schemas("anthropic")
        if tools:
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "tools": tools,
            },
            _anthropic_headers(),
        )
        return json.dumps(data)  # return raw for agent loop to parse natively

    # Local proxy - Anthropic messages API; tool calls returned as XML <tool_call> tags in text
    if BACKEND == "local":
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": system_prompt,
                "messages": flat,
                "max_tokens": MAX_TOKENS,
            },
            _anthropic_headers(),
        )
        text = "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
        return strip_gptoss_tokens(text)

    # Transformers (HuggingFace)
    if BACKEND == "transformers":
        assert mlx_state is not None  # load_model populates this for ML backends
        model, tokenizer = mlx_state
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}] + flat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=0.3,
                do_sample=True,
            )
        raw = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=False,
        )
        end = _tool_call_complete(raw)
        if end != -1:
            raw = raw[:end]
        return truncate_at_turn_leak(strip_gptoss_tokens(raw))

    # MLX (Apple Silicon)
    assert mlx_state is not None  # load_model populates this for ML backends
    model, tokenizer = mlx_state
    chat: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if c := flatten_content(m["content"]):
            chat.append({"role": m["role"], "content": c})
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
    out = ""
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        check_cancelled()
        out += chunk.text
        end = _tool_call_complete(out)
        if end != -1:
            out = out[:end]
            break
    if out.startswith(prompt):
        out = out[len(prompt) :].strip()
    return truncate_at_turn_leak(strip_gptoss_tokens(out))


# -----------------------------------------------------------------------------------------------
# History management
# -----------------------------------------------------------------------------------------------
def history_file_path() -> pathlib.Path:
    """Return the history file path from env override or user-level default."""
    if p := os.environ.get("WRENCODE_HISTORY_FILE"):
        return pathlib.Path(p).expanduser()
    return pathlib.Path.home() / ".wrencode" / "history.json"


def load_history() -> list[dict[str, Any]]:
    """Load conversation history from the JSON history file."""
    with contextlib.suppress(Exception):
        with open(history_file_path()) as f:
            return list(json.load(f))
    return []


def save_history(messages: list[dict[str, Any]]) -> None:
    """Persist conversation history to the JSON history file."""
    with contextlib.suppress(Exception):
        p = history_file_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(messages, f)


def compact_messages(
    messages: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Summarize conversation history to reduce context length."""
    if not messages:
        return messages
    history_text = "".join(
        f"{m['role']}: {flatten_content(m['content'])}\n" for m in messages
    )

    if BACKEND in HOSTED_BACKENDS:
        prompt = (
            "Summarize this conversation in 3-5 concise bullet points, "
            "preserving any file paths, code decisions, or unresolved tasks:\n\n"
            + history_text
        )
        if BACKEND == "bedrock":
            data = _bedrock_converse_call({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": "You are a helpful assistant."}],
                "inferenceConfig": {"maxTokens": 512},
            })
            summary = "\n".join(
                b["text"]
                for b in data.get("output", {}).get("message", {}).get("content", [])
                if "text" in b
            ).strip()
        elif BACKEND in ANTHROPIC_FORMAT_BACKENDS:
            data = _http_post(
                API_BASE,
                {
                    "model": MODEL,
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                },
                _anthropic_headers(),
            )
            summary = "\n".join(
                b["text"] for b in data.get("content", []) if b.get("type") == "text"
            ).strip()
        else:  # openai / openrouter
            data = _http_post(
                API_BASE,
                {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
                _openai_headers(),
            )
            summary = (data["choices"][0]["message"].get("content") or "").strip()
    else:
        # MLX / Transformers path
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Summarize this conversation in 3-5 bullet points:\n\n{history_text}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
        summary = "".join(
            c.text
            for c in stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=512, sampler=sampler
            )
        )
        if summary.startswith(prompt):
            summary = summary[len(prompt) :].strip()

    return [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {
            "role": "assistant",
            "content": "Understood, I have the context from the summary.",
        },
    ]


# -----------------------------------------------------------------------------------------------
# Workspace & system prompt
# -----------------------------------------------------------------------------------------------
def git_context() -> str:
    """Return a formatted git status string if inside a git repository."""
    with contextlib.suppress(Exception):
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return f"\nGit status:\n{r.stdout.strip()}"
    return ""


def build_system_prompt() -> str:
    """Build the system prompt with workspace context and tool definitions."""
    ws = workspace_root()
    path_rule = (
        "Paths are not restricted to the workspace."
        if os.environ.get("WRENCODE_UNRESTRICTED_PATHS", "").lower()
        in ("1", "true", "yes")
        else "Relative paths resolve under the workspace. Absolute paths must stay inside it."
    )
    return f"""You are a helpful coding assistant with tools to interact with the file system.
Workspace root: {ws}
Process cwd: {os.getcwd()}
{path_rule}{git_context()}

IMPORTANT: You MUST use tools by formatting them exactly as shown below.

Available tools:
- read(path, offset, limit): Read a file or list a directory
- write(path, content): Write to a file
- edit(path, old, new): Replace text in a file (old must be unique unless all=true)
- glob(pat): Find files matching pattern
- grep(pat): Search for text in files
- bash(cmd): Run a shell command
- task(prompt): Delegate a self-contained subtask to a fresh subagent; returns only its result

To use a tool, format it EXACTLY like this:
<tool_call>{{"tool": "name", "args": {{"key": "value"}}}}</tool_call>

Examples:
<tool_call>{{"tool": "read", "args": {{"path": "file.py", "offset": 0, "limit": 20}}}}</tool_call>
<tool_call>{{"tool": "glob", "args": {{"pat": "*.py"}}}}</tool_call>

When reading a file, always pass offset and limit. When you finish a task, summarize what you changed.
CRITICAL: You MUST use tools for file operations. Never say you can't access files!"""


# -----------------------------------------------------------------------------------------------
# Agentic loop
# -----------------------------------------------------------------------------------------------
def _track_error(
    result: str, last: Optional[str], count: int
) -> tuple[Optional[str], int, bool]:
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
    mlx_state: Optional[tuple[Any, Any]],
    max_iters: int = 0,
) -> None:
    """Generate a response and execute any tool calls, repeating until no tools remain.

    max_iters > 0 caps the tool-calling rounds (used to bound subagents);
    0 means unlimited, preserving the interactive default.
    """
    iters = 0
    last_tool_error: Optional[str] = None
    repeated_tool_error_count = 0
    try:
        while True:
            if max_iters and iters >= max_iters:
                print(f"{YELLOW}(stopped after {max_iters} iterations){RESET}")
                break
            iters += 1
            with thinking_spinner():
                response_text = get_response_cancellable(
                    messages, system_prompt, mlx_state
                )
            display_text, tool_calls, raw_data = _parse_response(response_text)
            if display_text:
                print_agent_message(display_text)
            _append_assistant(messages, display_text, tool_calls, raw_data)
            if not tool_calls:
                break
            results: list[tuple[ToolCall, str]] = []
            stop = False
            for tc in tool_calls:
                check_cancelled()
                print_tool_action(tc.name, tc.input)
                result = run_tool(tc.name, tc.input)
                print_tool_result(result)
                results.append((tc, result))
                last_tool_error, repeated_tool_error_count, stop = _track_error(
                    result, last_tool_error, repeated_tool_error_count
                )
                if stop:
                    break
            _append_tool_results(messages, results)
            if stop:
                break
    except (UserCancelled, KeyboardInterrupt):
        print(f"\n{YELLOW}Cancelled — back to prompt.{RESET}\n")
    finally:
        _CANCEL_REQUESTED.clear()
        _LISTENER_STOP.set()


# -----------------------------------------------------------------------------------------------
# Slash commands
# -----------------------------------------------------------------------------------------------
def handle_slash_command(
    cmd: str,
    messages: list[dict[str, Any]],
    mlx_state: Optional[tuple[Any, Any]],
) -> tuple[Optional[str], Any]:
    """Handle a slash command.

    Returns (action, mlx_state). mlx_state is _MLX_UNCHANGED unless the
    backend/model changed and the in-process model must be reloaded.
    """
    if cmd in {"/q", "exit"}:
        save_history(messages)
        return "quit", _MLX_UNCHANGED
    if cmd == "/c":
        global _SESSION_AUTO_APPROVE
        _SESSION_AUTO_APPROVE = False
        messages.clear()
        save_history(messages)
        print_system("Cleared")
        return "handled", _MLX_UNCHANGED
    if cmd == "/compact":
        if BACKEND in HOSTED_BACKENDS or (BACKEND in LOCAL_ML_BACKENDS and mlx_state):
            print_system("Compacting history...")
            model, tokenizer = mlx_state or (None, None)
            before = len(messages)
            messages[:] = compact_messages(messages, model, tokenizer)
            save_history(messages)
            print_system(f"Compacted {before} → {len(messages)} messages")
        else:
            print(f"{YELLOW}/compact not available for backend '{BACKEND}'{RESET}")
        return "handled", _MLX_UNCHANGED
    if cmd in {"/backend", "/configure"}:
        return "handled", switch_backend_runtime()
    if cmd == "/model" or cmd.startswith("/model "):
        model_id = cmd[7:].strip() if cmd.startswith("/model ") else ""
        return "handled", switch_model_runtime(model_id)
    if cmd == "/help":
        print_system("/c — clear  /compact — summarize  /q — quit")
        print_system("/backend — switch backend (↑↓)  /model — switch model (↑↓)")
        print_system("/model <id> — set model directly  /configure — same as /backend")
        return "handled", _MLX_UNCHANGED
    return None, _MLX_UNCHANGED


# -----------------------------------------------------------------------------------------------
# Backend selection & config persistence
# -----------------------------------------------------------------------------------------------
def is_frozen() -> bool:
    """Set true when running as a PyInstaller standalone binary."""
    return bool(getattr(sys, "frozen", False))


def load_config() -> dict[str, str]:
    """Load saved backend config from CONFIG_FILE, or {} if absent/unreadable."""
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(cfg: dict[str, str]) -> None:
    """Persist backend config to CONFIG_FILE with owner-only (0600) permissions."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.chmod(tmp, 0o600)
        tmp.replace(CONFIG_FILE)
    except OSError as err:
        print(f"{YELLOW}Could not save config to {CONFIG_FILE}: {err}{RESET}")


def available_backends() -> list[str]:
    """Backends offerable in the current runtime.

    The standalone binary can't bundle the ML stack, so local-ml backends
    (mlx/transformers) are only offered from a source install. MLX is further
    limited to Apple Silicon.
    """
    return [
        name
        for name, spec in BACKEND_SPECS.items()
        if not (
            spec["kind"] == "local-ml"
            and (
                is_frozen()
                or (
                    name == "mlx"
                    and not (
                        platform.system() == "Darwin" and platform.machine() == "arm64"
                    )
                )
            )
        )
    ]


def pick_from_list(
    title: str,
    options: list[str],
    *,
    labels: Optional[list[str]] = None,
    initial_index: int = 0,
) -> Optional[int]:
    """Pick one option from a numbered list."""
    if not options:
        print(f"{YELLOW}No options available.{RESET}")
        return None
    labels = labels if labels is not None else options
    initial_index = max(0, min(initial_index, len(options) - 1))
    print()
    if title:
        print_system(title)
        print()
    for i, label in enumerate(labels, 1):
        prefix = BOLD if i - 1 == initial_index else ""
        suffix = RESET if i - 1 == initial_index else ""
        print("  " + prefix + str(i) + ". " + label + suffix)
    default = str(initial_index + 1)
    while True:
        raw = input(f"{BLUE}❯{RESET} number [{default}]: ").strip() or default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"{RED}Enter a number between 1 and {len(options)}.{RESET}")


def fetch_openrouter_models() -> list[str]:
    """Fetch OpenRouter model ids, with a 24h local cache."""
    if OPENROUTER_MODELS_CACHE.exists():
        with contextlib.suppress(Exception):
            age = time.time() - OPENROUTER_MODELS_CACHE.stat().st_mtime
            if age < 86400:
                cached = json.loads(OPENROUTER_MODELS_CACHE.read_text())
                if isinstance(cached, list) and cached:
                    return [str(m) for m in cached]

    key = (
        os.environ.get("OPENROUTER_API_KEY")
        or API_KEY
        or load_config().get("api_key", "")
    )
    if not key:
        return list(BACKEND_MODELS.get("openrouter", []))

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        ids = sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
        if ids:
            OPENROUTER_MODELS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            OPENROUTER_MODELS_CACHE.write_text(json.dumps(ids, indent=2))
            os.chmod(OPENROUTER_MODELS_CACHE, 0o600)
        return ids or list(BACKEND_MODELS.get("openrouter", []))
    except Exception as err:
        print(f"{YELLOW}Could not fetch OpenRouter models: {err}{RESET}")
        return list(BACKEND_MODELS.get("openrouter", []))


def fetch_ollama_models() -> list[str]:
    """List models reported by a local Ollama server."""
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            data = json.load(resp)
        names = sorted(
            m.get("name", "") for m in data.get("models", []) if m.get("name")
        )
        return names or [BACKEND_SPECS["ollama"]["model"]]
    except Exception as err:
        print(f"{YELLOW}Could not reach Ollama at {base}: {err}{RESET}")
        return [BACKEND_SPECS["ollama"]["model"]]


def list_models_for_backend(backend: str) -> list[str]:
    """Return selectable models for a backend (includes a custom-id option)."""
    spec = BACKEND_SPECS[backend]
    if backend == "openrouter":
        models = fetch_openrouter_models()
    elif backend == "ollama":
        models = fetch_ollama_models()
    else:
        models = list(BACKEND_MODELS.get(backend, [spec["model"]]))

    if not models:
        models = [spec["model"]]
    models = list(dict.fromkeys(models))
    current = MODEL if backend == BACKEND else spec["model"]
    if current and current not in models:
        models.insert(0, current)
    if CUSTOM_MODEL_OPTION not in models:
        models.append(CUSTOM_MODEL_OPTION)
    return models


def _prompt_api_key_if_needed(
    backend: str, existing: dict[str, str], cfg: dict[str, str]
) -> None:
    """Prompt for an API key when switching to a hosted backend."""
    spec = BACKEND_SPECS[backend]
    if spec["kind"] == "aws":
        # Bedrock uses AWS environment credentials, not a saved key.
        ak, sk, _ = _aws_credentials()
        if ak and sk:
            print_system(f"✓ Using AWS credentials from environment @ {_aws_region()}")
        else:
            print(
                f"{YELLOW}Bedrock uses your AWS credentials — set AWS_ACCESS_KEY_ID, "
                f"AWS_SECRET_ACCESS_KEY, and AWS_REGION.{RESET}"
            )
        return
    if spec["kind"] != "api":
        return
    key_env = spec["key_env"]
    if os.environ.get(key_env):
        print_system(f"✓ Using {key_env} from environment")
        return
    saved_key = (
        existing.get("api_key", "") if existing.get("backend") == backend else ""
    )
    keep_hint = " (leave blank to keep saved key)" if saved_key else ""
    key = getpass.getpass(
        f"{BLUE}❯{RESET} {key_env}{keep_hint} (input hidden): "
    ).strip()
    if key:
        cfg["api_key"] = key
    elif saved_key:
        cfg["api_key"] = saved_key
        print_system(f"✓ Keeping saved {key_env}")
    else:
        print(f"{YELLOW}No key entered — set {key_env} or re-run with /backend.{RESET}")


def persist_backend_choice(cfg: dict[str, str]) -> None:
    """Merge cfg into saved config and apply module-level backend globals."""
    merged = {**load_config(), **cfg}
    save_config(merged)
    apply_backend(
        merged["backend"],
        merged.get("model", ""),
        merged.get("api_key", ""),
    )


def verify_api_key() -> tuple[str, str]:
    """Probe the current backend with API_KEY to see if the key actually works.

    Returns (state, detail): "ok" (accepted), "invalid" (rejected by the
    service), or "unknown" (couldn't reach it — network/timeout). Local
    backends and missing keys short-circuit to "ok"/"invalid" without a call.
    """
    spec = BACKEND_SPECS[BACKEND]
    if spec["kind"] == "aws":
        # Probe Bedrock's control plane (ListFoundationModels) with a signed
        # GET: confirms the AWS creds + region work without invoking a model.
        ak, sk, _ = _aws_credentials()
        if not (ak and sk):
            return ("invalid", "no AWS credentials")
        region = _aws_region()
        url = f"https://bedrock.{region}.amazonaws.com/foundation-models"
        try:
            headers = _sigv4_signed_headers("GET", url, b"", "bedrock", region)
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read(1)
            return ("ok", "")
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                return ("invalid", f"HTTP {err.code}")
            return ("unknown", f"HTTP {err.code}")
        except Exception as err:
            return ("unknown", str(err))
    if spec["kind"] != "api":
        return ("ok", "")
    if not API_KEY:
        return ("invalid", "no key")
    probes = {
        "anthropic": ("https://api.anthropic.com/v1/models", _anthropic_headers()),
        "openai": ("https://api.openai.com/v1/models", _openai_headers()),
        "openrouter": ("https://openrouter.ai/api/v1/key", _openai_headers()),
    }
    if BACKEND not in probes:
        return ("unknown", "")
    url, headers = probes[BACKEND]
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read(1)
        return ("ok", "")
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            return ("invalid", f"HTTP {err.code}")
        return ("unknown", f"HTTP {err.code}")
    except Exception as err:
        return ("unknown", str(err))


def pick_model_interactive(backend: str) -> Optional[str]:
    """Arrow-key model picker for a backend; returns model id or None."""
    models = list_models_for_backend(backend)
    labels = models[:]
    initial = models.index(MODEL) if MODEL in models else 0
    idx = pick_from_list(
        f"Choose model ({backend})",
        models,
        labels=labels,
        initial_index=initial,
    )
    if idx is None:
        return None
    choice = models[idx]
    if choice == CUSTOM_MODEL_OPTION:
        default = BACKEND_SPECS[backend]["model"]
        custom = input(f"{BLUE}❯{RESET} model id [{default}]: ").strip()
        return custom or default
    return choice


def try_reload_model() -> Any:
    """Load weights for local-ml backends; return None for API/proxy backends."""
    if BACKEND in LOCAL_ML_BACKENDS:
        try:
            return load_model()
        except SystemExit:
            return _MLX_UNCHANGED
    if BACKEND in API_BACKENDS and not API_KEY:
        key_env = BACKEND_SPECS[BACKEND]["key_env"]
        print(f"{RED}{key_env} not set — cannot use {BACKEND}.{RESET}")
        return _MLX_UNCHANGED
    return None


def switch_model_runtime(model_id: str = "") -> Any:
    """Switch model on the current backend; reload local weights if needed."""
    if model_id:
        model = model_id
    else:
        if not sys.stdin.isatty():
            print(f"{RED}/model needs an interactive terminal.{RESET}")
            return _MLX_UNCHANGED
        picked = pick_model_interactive(BACKEND)
        if picked is None:
            print_system("Cancelled.")
            return _MLX_UNCHANGED
        model = picked

    if os.environ.get("MODEL"):
        print(f"{YELLOW}MODEL env var overrides saved choice.{RESET}")

    persist_backend_choice({"backend": BACKEND, "model": model})
    print_system(f"✓ Model → {MODEL}")
    return try_reload_model()


def switch_backend_runtime() -> Any:
    """Switch backend (and model) mid-session; reload local weights if needed."""
    if not sys.stdin.isatty():
        print(f"{RED}/backend needs an interactive terminal.{RESET}")
        return _MLX_UNCHANGED

    existing = load_config()
    names = available_backends()
    labels = [
        f"{BACKEND_SPECS[n]['label']}  [{BACKEND_SPECS[n]['model']}]" for n in names
    ]
    initial = names.index(BACKEND) if BACKEND in names else 0
    idx = pick_from_list("Choose backend", names, labels=labels, initial_index=initial)
    if idx is None:
        print_system("Cancelled.")
        return _MLX_UNCHANGED

    backend = names[idx]
    model = pick_model_interactive(backend)
    if model is None:
        print_system("Cancelled.")
        return _MLX_UNCHANGED

    cfg: dict[str, str] = {"backend": backend, "model": model}
    _prompt_api_key_if_needed(backend, existing, cfg)
    persist_backend_choice(cfg)
    print_system(f"✓ Backend → {BACKEND}:{MODEL}")
    return try_reload_model()


def choose_backend_interactive() -> None:
    """Prompt the user to pick a backend, persist the choice, and apply it."""
    existing = load_config()
    names = available_backends()
    labels = [
        f"{BACKEND_SPECS[n]['label']}  [{BACKEND_SPECS[n]['model']}]" for n in names
    ]
    if not is_frozen():
        print_system("Local model backends need mlx-lm or transformers installed.")
        print()

    idx = pick_from_list("Choose backend", names, labels=labels, initial_index=0)
    if idx is None:
        print(f"{RED}Backend selection required.{RESET}")
        raise SystemExit(1)

    choice = names[idx]
    model = pick_model_interactive(choice)
    if model is None:
        print(f"{RED}Model selection required.{RESET}")
        raise SystemExit(1)

    cfg: dict[str, str] = {"backend": choice, "model": model}
    spec = BACKEND_SPECS[choice]
    _prompt_api_key_if_needed(choice, existing, cfg)
    persist_backend_choice(cfg)
    print_system(f"✓ Saved backend choice to {CONFIG_FILE}")

    # Verify the credential actually works before declaring the backend ready,
    # so a typo/revoked key (or missing AWS creds) surfaces here instead of
    # mid-chat. Re-prompt API keys on a hard rejection; Bedrock creds come from
    # the environment, so there's nothing to re-prompt. Don't block on a
    # transient network failure.
    cred_name = spec.get("key_env", "AWS credentials")
    for attempt in range(3):
        state, detail = verify_api_key()
        if state == "ok":
            print_system(f"✓ {choice}:{MODEL} ready")
            return
        if state == "unknown":
            print(f"{YELLOW}⚠ Couldn't verify {cred_name} ({detail}).{RESET}")
            return
        print(f"{RED}✗ {cred_name} was rejected ({detail}).{RESET}")
        if spec["kind"] != "api" or not sys.stdin.isatty() or attempt == 2:
            if spec["kind"] == "aws":
                print(
                    f"{DIM}Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (and AWS_REGION) and re-run.{RESET}"
                )
            return
        newkey = getpass.getpass(
            f"{BLUE}❯{RESET} re-enter {cred_name} (input hidden): "
        ).strip()
        if not newkey:
            return
        cfg["api_key"] = newkey
        persist_backend_choice(cfg)


def resolve_configuration() -> None:
    """Decide which backend to use: env override > saved config > interactive > error."""
    # 1. Explicit BACKEND env var — power users / CI. Unchanged from prior behaviour.
    env_backend = os.environ.get("BACKEND")
    if env_backend:
        if env_backend not in BACKEND_SPECS:
            valid = ", ".join(BACKEND_SPECS)
            print(f"{RED}Unknown BACKEND '{env_backend}'.{RESET} Valid: {valid}")
            raise SystemExit(1)
        apply_backend(env_backend)
        return

    # 2. A choice saved from a previous run.
    cfg = load_config()
    if cfg.get("backend") in BACKEND_SPECS:
        backend = cfg["backend"]
        spec = BACKEND_SPECS[backend]
        # A hosted backend with no usable key — env var unset and nothing
        # saved (e.g. configured in a dir whose .env supplied the key, so it
        # was never persisted) — would dead-end in load_model() with a
        # SystemExit. In a terminal, re-run the chooser so the user can pick a
        # backend and enter a key instead of the tool exiting immediately.
        key_missing = (
            spec["kind"] == "api"
            and not os.environ.get(spec["key_env"])
            and not cfg.get("api_key")
        )
        if key_missing and sys.stdin.isatty():
            print(
                f"{YELLOW}Saved backend '{backend}' has no API key "
                f"({spec['key_env']} is unset and none was saved).{RESET}"
            )
            choose_backend_interactive()
            return
        apply_backend(backend, cfg.get("model", ""), cfg.get("api_key", ""))
        return

    # 3. First run with a real terminal — ask the user.
    if sys.stdin.isatty():
        choose_backend_interactive()
        return

    # 4. Non-interactive with nothing configured — fail with guidance.
    print(f"{RED}No backend configured.{RESET}")
    print(
        "Set BACKEND=<name> (plus the matching API key), "
        "or run `wrencode` in a terminal to choose one."
    )
    raise SystemExit(1)


# -----------------------------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------------------------
def load_model() -> Optional[tuple[Any, Any]]:
    """Load model for the current backend and return mlx_state (or None for API backends)."""
    if BACKEND == "mlx":
        try:
            global load, stream_generate, make_sampler
            from mlx_lm import load  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
            from mlx_lm.generate import (  # ty: ignore[unresolved-import]
                stream_generate,  # type: ignore[import-not-found]
            )
            from mlx_lm.sample_utils import (  # ty: ignore[unresolved-import]
                make_sampler,  # type: ignore[import-not-found]
            )
        except ImportError:
            print(f"{RED}MLX backend needs mlx-lm:{RESET} pip install mlx-lm")
            print(f"{DIM}Or run `wrencode` to pick a hosted backend.{RESET}")
            raise SystemExit(1)
        print(f"{YELLOW}Loading model...{RESET}")
        model, tokenizer = load(MODEL)
        print_system(f"✓ Loaded: {getattr(model, 'name', MODEL)}")
        print()
        return (model, tokenizer)
    if BACKEND == "transformers":
        try:
            global torch, AutoModelForCausalLM, AutoTokenizer
            import torch  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
            from transformers import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError:
            print(
                f"{RED}transformers backend needs:{RESET} "
                "pip install transformers torch"
            )
            print(f"{DIM}Or run `wrencode` to pick a hosted backend.{RESET}")
            raise SystemExit(1)
        print(f"{YELLOW}Loading model via transformers...{RESET}")
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        # Load then move to the device. device_map= is for multi-device sharding
        # (needs accelerate, rejects a plain "mps"/"cpu" string in current transformers).
        _mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(
            _device
        )
        print_system(f"✓ Loaded on {_device}: {MODEL}")
        print()
        return (_mdl, _tok)
    if BACKEND == "local":
        print_system(f"Local proxy at {API_BASE}")
        return None
    if BACKEND == "ollama":
        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
                installed = {m.get("name", "") for m in json.load(r).get("models", [])}
            want = MODEL if ":" in MODEL else f"{MODEL}:latest"
            if not (
                MODEL in installed
                or want in installed
                or any(n.split(":")[0] == MODEL for n in installed)
            ):
                have = ", ".join(sorted(installed)) or "none"
                print(f"{YELLOW}⚠ Model '{MODEL}' isn't pulled into Ollama.{RESET}")
                print(f"{DIM}  Run: ollama pull {MODEL}   (installed: {have}){RESET}")
        except Exception:
            print(
                f"{YELLOW}⚠ Couldn't reach Ollama at {base} — is `ollama serve` running?{RESET}"
            )
        print_system(f"{BACKEND} ({MODEL})")
        return None
    # Bedrock — uses AWS environment credentials (SigV4), not an API key.
    if BACKEND == "bedrock":
        ak, sk, _ = _aws_credentials()
        if not (ak and sk):
            print(f"{RED}AWS credentials not found.{RESET}")
            print(
                f"{DIM}Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and "
                f"AWS_REGION) — Bedrock uses your AWS environment.{RESET}"
            )
            raise SystemExit(1)
        print_system(f"bedrock ({MODEL}) @ {_aws_region()}")
        return None
    # Hosted API backends — all require a key.
    if not API_KEY:
        key_env = BACKEND_SPECS[BACKEND]["key_env"]
        print(f"{RED}{key_env} not set.{RESET}")
        print(
            f"{DIM}Set {key_env}, or run `wrencode` in a terminal to enter a key.{RESET}"
        )
        raise SystemExit(1)
    print_system(f"{BACKEND} ({MODEL})")
    return None


# -----------------------------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------------------------
def uninstall() -> None:
    """Delete saved state and print how to remove the program for this install."""
    if CONFIG_DIR.exists():
        try:
            shutil.rmtree(CONFIG_DIR)
            print_system(f"✓ Removed saved config {CONFIG_DIR}")
        except OSError as err:
            print(f"{YELLOW}Could not remove {CONFIG_DIR}: {err}{RESET}")
    else:
        print_system(f"No saved config at {CONFIG_DIR}")

    print_system("To remove the program itself:")
    if is_frozen():
        # install.sh drops a standalone binary; sys.executable is that file.
        print(f"rm {sys.executable}")
    else:
        print(f"{DIM}uv tool install:{RESET} uv tool uninstall wrencode")
        print(f"{DIM}pip install:{RESET} pip uninstall wrencode")
        print(f"{DIM}uvx cache:{RESET} uv cache clean wrencode")


def print_help() -> None:
    """Print CLI usage."""
    print("wrencode — a minimal agentic coding assistant\n")
    print("Usage: wrencode [options]\n")
    print("Options:")
    print("--yes         auto-approve all writes/commands (WRENCODE_AUTO_APPROVE)")
    print("--uninstall   remove saved config and show how to delete wrencode")
    print("--version, -V print version and exit")
    print("--help, -h    show this help\n")
    print("Slash commands: /backend /model /c /compact /q  (see /help in session)")
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
        uninstall()
        return
    if "--yes" in args or "--auto-approve" in args:
        os.environ["WRENCODE_AUTO_APPROVE"] = "1"

    os.environ.setdefault("WRENCODE_WORKSPACE", str(pathlib.Path.cwd().resolve()))
    resolve_configuration()

    sys.stdout.write("\033]0;wrencode\007")  # set terminal tab/window title
    print(WREN_BANNER)
    print(f"{BOLD}wrencode{RESET} 🐦 | {DIM}{BACKEND}:{MODEL}{RESET}")
    mlx_state = load_model()
    _MLX_STATE = mlx_state  # expose to the task() subagent tool
    system_prompt = build_system_prompt()
    messages = load_history()
    if messages:
        chats = sum(1 for m in messages if m.get("role") == "user")
        print(f"{DIM}Restored {chats} chats{RESET}")

    while True:
        try:
            user_input = read_user_input()
            if not user_input:
                continue
            action, new_mlx = handle_slash_command(user_input, messages, mlx_state)
            if new_mlx is not _MLX_UNCHANGED:
                mlx_state = new_mlx
                _MLX_STATE = mlx_state
            if action == "quit":
                break
            if action == "handled":
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
        except Exception as err:
            msg = str(err)
            print(f"{RED}Error: {msg}{RESET}")
            if BACKEND == "ollama" and ("not found" in msg.lower() or "404" in msg):
                print(
                    f"{YELLOW}Model '{MODEL}' isn't pulled. "
                    f"Run: ollama pull {MODEL}  (or `ollama list`).{RESET}"
                )
            if os.environ.get("WRENCODE_DEBUG"):
                traceback.print_exc()
            else:
                print(f"{DIM}(set WRENCODE_DEBUG=1 for the full traceback){RESET}")


if __name__ == "__main__":
    main()
