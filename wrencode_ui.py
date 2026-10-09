"""Terminal UI for wrencode: colors, input, approvals, Escape-to-cancel, output tagging.

Everything that talks to the person at the keyboard lives here, plus the per-thread
agent identity (_AGENT_LOCAL) that tags a parallel subagent's output and approvals.
This module imports nothing else from wrencode.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import sys
import threading
from collections.abc import Callable
from typing import Any

import wrencode_permissions as permissions

_AGENT_LOCAL = threading.local()


def _subagent_depth() -> int:
    return getattr(_AGENT_LOCAL, "depth", 0)


def _agent_tag() -> str:
    """This thread's parallel-subagent tag, or '' outside a parallel batch."""
    return getattr(_AGENT_LOCAL, "tag", "")


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


def _light_background() -> bool:
    """Whether the terminal is light: WRENCODE_THEME=light|dark wins, then the
    COLORFGBG hint some terminals export ("15;0" is white on black)."""
    theme = os.environ.get("WRENCODE_THEME", "").lower()
    if theme in ("light", "dark"):
        return theme == "light"
    fgbg = os.environ.get("COLORFGBG", "")
    if ";" in fgbg:
        bg = fgbg.rsplit(";", 1)[-1]
        return bg.isdigit() and (int(bg) in (7, 15) or int(bg) >= 231)
    return False


LIGHT = _light_background()
# The assistant's prose: a shade off the user's text. Fenced code: a tint of its
# own so code reads apart from prose. Both chosen for the background in use.
AGENT_TEXT = "\033[38;5;236m" if LIGHT else "\033[38;5;252m"
CODE_TEXT = "\033[38;5;94m" if LIGHT else "\033[38;5;223m"
AGENT_MARK = f"{BRIGHT_CYAN}●{RESET}"  # opens every assistant reply
TOOL_MARK = f"{GREEN}●{RESET}"  # opens every tool call: same dot, its own color
_COMPOSE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_LOADER_MODEL_MAX = 32


def loader_context(backend: str, model: str, activity: str = "thinking") -> str:
    """The loader's label: what is happening and which model is doing it."""
    if model and len(model) > _LOADER_MODEL_MAX:
        keep = _LOADER_MODEL_MAX - 1
        head = max(8, keep // 2)
        m = f"{model[:head]}…{model[-(keep - head) :]}"
    else:
        m = model or backend or "model"
    return f"{activity} · {m}"


def loader_display(step: int, context: str, seconds: int = -1) -> str:
    """Render one animated loader frame: the context, how long it has been, and
    the way out (seconds < 0 leaves both off)."""
    sym = _COMPOSE_FRAMES[step % len(_COMPOSE_FRAMES)]
    tail = f" · {seconds}s · esc to cancel" if seconds >= 0 else ""
    if not colors_enabled():
        return f"{sym} {context}{tail}"
    return f"{BRIGHT_CYAN}{sym}{RESET} {DIM}{context}{tail}{RESET}"


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


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keeps \t and \n


def visible(text: str) -> str:
    """Show control characters as ^X (Escape as ^[) instead of letting the terminal act on them.

    Model output, file contents and command output are untrusted; a carriage
    return or escape sequence in them could redraw the screen or hide part of a
    command from the approval prompt.
    """
    return _CONTROL_CHARS.sub(
        lambda m: "^?" if m.group() == "\x7f" else f"^{chr(ord(m.group()) ^ 0x40)}",
        text,
    )


# What a failure means and what to do about it, for the usual ones. The raw
# message stays available under WRENCODE_DEBUG.
_ERROR_HINTS: tuple[tuple[str, str], ...] = (
    (
        r"HTTP 401|invalid x-api-key|authentication_error",
        "The API key was rejected. Run /configure to enter a new one.",
    ),
    (
        r"HTTP 403",
        "The key does not have access to this model or workspace. Check it in the provider's console, or run /configure.",
    ),
    (
        r"HTTP 404|not_found_error",
        "The model id was not found for this backend. Run /model to pick one.",
    ),
    (
        r"HTTP 429|rate_limit",
        "The provider is rate limiting this key. Wait a moment and send again.",
    ),
    (r"HTTP 529|overloaded", "The provider is overloaded. Try again shortly."),
    (r"HTTP 5\d\d", "The provider had an internal error. Try again shortly."),
    (
        r"context_length|too many tokens|prompt is too long|maximum context",
        "The conversation no longer fits the model's window. /compact summarizes it, /clear starts over.",
    ),
    (
        r"credit|billing|insufficient_quota",
        "The account is out of credit. Top it up in the provider's console.",
    ),
    (
        r"Connection refused|Name or service not known|nodename nor servname|Temporary failure in name resolution",
        "Could not reach the provider: check the network, proxy, or the local server's address.",
    ),
    (
        r"timed out|TimeoutError",
        "The request timed out. Try again; WRENCODE_HTTP_TIMEOUT raises the limit.",
    ),
    (
        r"CERTIFICATE_VERIFY_FAILED|SSL",
        "TLS verification failed, usually a proxy in the way: point SSL_CERT_FILE at its certificate bundle.",
    ),
)


def explain_error(message: str) -> str:
    """One sentence on what went wrong and the next step, or '' for an unknown error."""
    for pattern, hint in _ERROR_HINTS:
        if re.search(pattern, message, re.IGNORECASE):
            return hint
    return ""


def print_error(message: str) -> None:
    """Report a failure as a sentence and a next step; the raw text follows
    when it adds something, or under WRENCODE_DEBUG in full."""
    message = visible(message)
    hint = explain_error(message)
    if not hint:
        print(f"{RED}Error: {message}{RESET}")
        return
    print(f"{RED}Error: {hint}{RESET}")
    if os.environ.get("WRENCODE_DEBUG"):
        print(f"{DIM}{message}{RESET}")
    else:
        head = message.split("\n", 1)[0]
        print(f"{DIM}{head[:160]}{'…' if len(head) > 160 else ''}{RESET}")


def print_system(text: str, *, end: str = "\n") -> None:
    """Print slash-command / configure feedback: plain text, so it reads as a reply
    rather than a banner (the ❯ prompt and tool lines carry the color)."""
    if sys.stdout.isatty():
        sys.stdout.write("\r")
    sys.stdout.write(f"{text}{end}")
    sys.stdout.flush()


_INPUT_HISTORY: list[str] = []

# Slash commands shown in the completion menu and /help, in display order.
SLASH_COMMANDS: dict[str, str] = {
    "/model": "switch model (or /model <id>)",
    "/backend": "switch backend and model",
    "/configure": "same as /backend",
    "/compact": "summarize history to free context",
    "/clear": "clear the conversation (starts a new session)",
    "/sessions": "list this project's conversations",
    "/resume": "continue an earlier session: pick one, or /resume <id>",
    "/search": "find a past session by its words: /search <text>",
    "/sync": "copy this project's history to the mirror now",
    "/usage": "token usage and spend: this turn and the session",
    "/permissions": "rules that allow or deny actions without asking",
    "/mcp": "MCP servers and their tools (/mcp reload reconnects)",
    "/help": "list commands",
    "/quit": "save history and exit",
}
# Short aliases: accepted and highlighted, but kept out of the menu.
SLASH_ALIASES: dict[str, str] = {"/c": "/clear", "/q": "/quit", "/exit": "/quit"}
MAX_SLASH_MENU = 14


def slash_matches(text: str) -> list[str]:
    """Commands that complete the /prefix being typed (empty once args start)."""
    if not text.startswith("/") or " " in text:
        return []
    return [c for c in SLASH_COMMANDS if c.startswith(text)][:MAX_SLASH_MENU]


def _is_slash_command(token: str) -> bool:
    return token in SLASH_COMMANDS or token in SLASH_ALIASES


def format_input_line(text: str) -> str:
    """Render the ❯ prompt line; a known /command (or its prefix) is bold cyan."""
    if not colors_enabled():
        return f"❯ {text}"
    prompt = f"{BRIGHT_CYAN}❯{RESET} "
    cmd, _, rest = text.partition(" ")
    if not (_is_slash_command(cmd) or (not rest and slash_matches(cmd))):
        return prompt + text
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LAST_CURSOR_ROW = 0  # the input block's row the cursor was left on by the last draw


def _cols() -> int:
    import shutil

    return max(shutil.get_terminal_size().columns, 20)


def _rows_of(length: int, width: int) -> int:
    """How many terminal rows `length` cells occupy (a full row stays one row)."""
    return max(1, -(-length // width))


def _redraw_input_line(
    text: str,
    matches: list[str] | None = None,
    sel: int = 0,
    hint: str = "",
    cursor: int | None = None,
    prompt: str | None = None,
) -> None:
    """Redraw the input, which may span lines, plus a completion menu or a hint
    below it, and put the cursor back at `cursor` (the end by default).

    The first line carries the prompt (the ❯ by default), continuation lines a
    two-space margin. Rows are counted with wrapping so the cursor lands right
    on long lines too; `_LAST_CURSOR_ROW` remembers where it was left so the
    next draw can climb back to the top of the block before clearing it.
    """
    global _LAST_CURSOR_ROW
    width = _cols()
    lines = text.split("\n")
    first = format_input_line(lines[0]) if prompt is None else prompt + lines[0]
    rendered = [first] + [f"  {ln}" for ln in lines[1:]]
    cur = len(text) if cursor is None else cursor
    # Which logical line the cursor is on, and its offset in it.
    line_no, offset = 0, cur
    for ln in lines:
        if offset <= len(ln):
            break
        offset -= len(ln) + 1
        line_no += 1
    rows_before = sum(_rows_of(2 + len(ln), width) for ln in lines[:line_no])
    cur_row, cur_col = divmod(2 + offset, width)
    if cur_col == 0 and offset and (2 + offset) % width == 0:
        cur_row, cur_col = cur_row - 1, width  # sits at the row's edge, pending wrap
    cursor_row = rows_before + cur_row
    total_rows = sum(_rows_of(2 + len(ln), width) for ln in lines)

    out = (f"\033[{_LAST_CURSOR_ROW}A" if _LAST_CURSOR_ROW else "") + "\r\033[J"
    out += "\n".join(rendered)
    below = len(matches or [])
    for i, cmd in enumerate(matches or []):
        desc = SLASH_COMMANDS.get(cmd, "")
        if not colors_enabled():
            out += f"\n{'>' if i == sel else ' '} {cmd:<12} {desc}"
        elif i == sel:
            out += f"\n  {BOLD}{BRIGHT_CYAN}{cmd:<12}{RESET} {desc}"
        else:
            out += f"\n  {DIM}{cmd:<12} {desc}{RESET}"
    if hint and not matches:
        out += f"\n  {DIM}{hint}{RESET}" if colors_enabled() else f"\n  {hint}"
        below = 1
    up = (total_rows - 1 - cursor_row) + below
    if up:
        out += f"\033[{up}A"
    out += "\r" + (f"\033[{cur_col}C" if cur_col else "")
    _LAST_CURSOR_ROW = cursor_row
    sys.stdout.write(out)
    sys.stdout.flush()


def _read_tty_key(fd: int) -> str:
    """Read one key: printable text, or a name for the editing keys (up, down,
    left, right, home, end, delete, backspace, enter, alt_enter, esc, ctrl_*),
    or 'paste:' followed by a bracketed paste's text."""
    ch = _read_input_char(fd)
    if not ch:
        return ""
    if ch == "\x1b":
        if not select.select([fd], [], [], 0.02)[0]:
            return "esc"
        seq = os.read(fd, 1)
        if seq in (b"\r", b"\n"):
            return "alt_enter"
        if seq not in (b"[", b"O"):
            return "esc"
        if not select.select([fd], [], [], 0.02)[0]:
            return "esc"
        code = os.read(fd, 1)
        while code.isdigit() or code == b";":  # ESC [ 1 ~, ESC [ 3 ~, ESC [ 1 ; 5 C
            if not select.select([fd], [], [], 0.02)[0]:
                return "esc"
            code += os.read(fd, 1)
        if code == b"200~":  # bracketed paste: everything up to ESC [ 201 ~
            return "paste:" + _read_paste(fd)
        return _CSI_KEYS.get(code, "esc")
    if ch in "\r\n":
        return "enter"
    if ch in ("\x7f", "\x08"):
        return "backspace"
    return _CTRL_KEYS.get(ch, ch)


def _read_paste(fd: int) -> str:
    buf = b""
    end = b"\x1b[201~"
    while not buf.endswith(end):
        if not select.select([fd], [], [], 2.0)[0]:
            break
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        buf += chunk
    text = buf.removesuffix(end)
    return (
        text.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    )


_CSI_KEYS = {
    b"A": "up",
    b"B": "down",
    b"C": "right",
    b"D": "left",
    b"H": "home",
    b"F": "end",
    b"1~": "home",
    b"4~": "end",
    b"7~": "home",
    b"8~": "end",
    b"3~": "delete",
}
_CTRL_KEYS = {
    "\x03": "ctrl_c",
    "\x04": "ctrl_d",
    "\x01": "home",
    "\x05": "end",
    "\x17": "ctrl_w",
    "\x15": "ctrl_u",
    "\x0b": "ctrl_k",
}


class LineEditor:
    """The input buffer and what each key does to it; no terminal of its own.

    The text can span lines: a backslash before Enter, Alt+Enter, or a pasted
    newline continues on the next line, plain Enter submits. ↑ and ↓ move
    between lines, or through the input history from the first and last line.
    `apply(key)` returns the submitted text, or None to keep editing; it raises
    KeyboardInterrupt on Ctrl+C and EOFError on Ctrl+D with nothing typed.
    """

    def __init__(self, *, history: bool = False, complete: bool = False) -> None:
        self.buf: list[str] = []
        self.cur = 0
        self.sel = 0
        self.history = history
        self.complete = complete
        self.hist_idx = len(_INPUT_HISTORY)

    @property
    def text(self) -> str:
        return "".join(self.buf)

    def matches(self) -> list[str]:
        return (
            slash_matches(self.text) if self.complete and "\n" not in self.text else []
        )

    def set(self, text: str) -> None:
        self.buf = list(text)
        self.cur = len(self.buf)

    def _insert(self, text: str) -> None:
        self.buf[self.cur : self.cur] = list(text)
        self.cur += len(text)
        self.sel = 0

    def _line_bounds(self) -> tuple[int, int]:
        """Start and end offsets of the line the cursor is on."""
        start = self.text.rfind("\n", 0, self.cur) + 1
        end = self.text.find("\n", self.cur)
        return start, (len(self.buf) if end < 0 else end)

    def apply(self, key: str) -> str | None:
        matches = self.matches()
        if key == "enter":
            if self.buf and self.cur == len(self.buf) and self.buf[-1] == "\\":
                self.buf[-1] = "\n"  # a backslash before Enter continues the line
                return None
            text = self.text.strip()
            if matches and not _is_slash_command(text):
                text = matches[min(self.sel, len(matches) - 1)]
            return text
        if key == "alt_enter":
            self._insert("\n")
            return None
        if key.startswith("paste:"):
            self._insert(key[6:])
            return None
        if matches and key in {"\t", "right"}:
            self.set(matches[min(self.sel, len(matches) - 1)])
            self.sel = 0
            return None
        if matches and key in {"up", "down"}:
            step = -1 if key == "up" else 1
            self.sel = (min(self.sel, len(matches) - 1) + step) % len(matches)
            return None
        if key == "backspace":
            if self.cur:
                del self.buf[self.cur - 1]
                self.cur -= 1
                self.sel = 0
            return None
        if key == "delete":
            if self.cur < len(self.buf):
                del self.buf[self.cur]
            return None
        if key == "left":
            self.cur = max(self.cur - 1, 0)
            return None
        if key == "right":
            self.cur = min(self.cur + 1, len(self.buf))
            return None
        if key in {"home", "end"}:
            start, end = self._line_bounds()
            self.cur = start if key == "home" else end
            return None
        if key == "ctrl_w":  # delete the word before the cursor
            start = self.cur
            while start and self.buf[start - 1] == " ":
                start -= 1
            while start and self.buf[start - 1] not in " \n":
                start -= 1
            del self.buf[start : self.cur]
            self.cur = start
            return None
        if key == "ctrl_u":  # delete to the start of the line
            start, _ = self._line_bounds()
            del self.buf[start : self.cur]
            self.cur = start
            return None
        if key == "ctrl_k":  # delete to the end of the line
            _, end = self._line_bounds()
            del self.buf[self.cur : end]
            return None
        if key == "ctrl_c":
            raise KeyboardInterrupt
        if key == "ctrl_d":
            if not self.buf:
                raise EOFError
            return None
        if key in {"up", "down"}:
            start, end = self._line_bounds()
            if key == "up" and start > 0:  # a line above: move to it
                col = self.cur - start
                prev_start = self.text.rfind("\n", 0, start - 1) + 1
                self.cur = min(prev_start + col, start - 1)
                return None
            if key == "down" and end < len(self.buf):
                col = self.cur - start
                next_end = self.text.find("\n", end + 1)
                next_end = len(self.buf) if next_end < 0 else next_end
                self.cur = min(end + 1 + col, next_end)
                return None
            if not self.history:
                return None
            if key == "up" and self.hist_idx > 0:
                self.hist_idx -= 1
                self.set(_INPUT_HISTORY[self.hist_idx])
            elif key == "down" and self.hist_idx < len(_INPUT_HISTORY):
                self.hist_idx += 1
                self.set(
                    _INPUT_HISTORY[self.hist_idx]
                    if self.hist_idx < len(_INPUT_HISTORY)
                    else ""
                )
            return None
        if key == "esc":
            return None
        if len(key) == 1 and (key.isprintable() or key == "\t"):
            self._insert(key)
        return None


def _read_tty_line(
    prompt: str,
    *,
    history: bool = False,
    redraw: Any | None = None,
    complete: bool = False,
    hint: Callable[[str], str] | None = None,
) -> str:
    """Read input in cbreak mode with the LineEditor's keys; see LineEditor.

    With complete=True, typing a /prefix shows matching slash commands below
    the line: ↑↓ pick, Tab or → fills in, Enter runs the highlighted one.
    `hint(text)` returns a dim line to show under what's typed (empty for none);
    it is asked again on every keystroke, so it should be cheap. Pastes keep
    their line breaks (bracketed paste is enabled while reading).
    """
    global _LAST_CURSOR_ROW
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    editor = LineEditor(history=history, complete=complete)

    def _hint() -> str:
        text = editor.text
        if hint is None or not text or text.startswith("/"):
            return ""
        return hint(text)

    def _redraw() -> None:
        if redraw is not None:
            redraw(editor.text)
        elif complete:
            _redraw_input_line(
                editor.text, editor.matches(), editor.sel, _hint(), editor.cur
            )
        else:
            _redraw_input_line(editor.text, cursor=editor.cur, prompt=prompt)

    _LAST_CURSOR_ROW = 0
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[?2004h")  # bracketed paste on
        sys.stdout.flush()
        _redraw()
        while True:
            if not select.select([fd], [], [], None)[0]:
                continue
            key = _read_tty_key(fd)
            if not key:
                raise EOFError
            try:
                submitted = editor.apply(key)
            except (KeyboardInterrupt, EOFError):
                sys.stdout.write("\n")
                sys.stdout.flush()
                raise
            if submitted is not None:
                # Leave the final text on screen without the menu or hint.
                if redraw is None:
                    _redraw_input_line(
                        submitted if complete else editor.text,
                        prompt=None if complete else prompt,
                    )
                sys.stdout.write("\n")
                sys.stdout.flush()
                return submitted
            _redraw()
    finally:
        sys.stdout.write("\033[?2004l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _remember_input(text: str) -> None:
    if text and (not _INPUT_HISTORY or _INPUT_HISTORY[-1] != text):
        _INPUT_HISTORY.append(text)


def _read_user_input_interactive(hint: Callable[[str], str] | None = None) -> str:
    """TTY line editor with live slash-command coloring and history."""
    text = _read_tty_line("", history=True, complete=True, hint=hint)
    _remember_input(text)
    return text


def ask_line(prompt: str) -> str:
    """Read one short answer at a plain prompt (y/n questions at startup)."""
    try:
        return input(
            f"{BRIGHT_CYAN}❯{RESET} {prompt}" if colors_enabled() else f"❯ {prompt}"
        )
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def read_feedback_line() -> str:
    """Read decline feedback after choosing n in the approval picker."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        sys.stdout.write(f"\n{YELLOW}What should I do differently?{RESET}\n")
        sys.stdout.flush()
        return _read_tty_line(f"{BRIGHT_CYAN}❯{RESET} ")
    prompt = f"{YELLOW}What should I do differently?{RESET} "
    return input(prompt).strip()


def read_user_input(hint: Callable[[str], str] | None = None) -> str:
    """Read one line from the ❯ prompt with live slash-command coloring.

    On a terminal, `hint(text)` supplies a dim line shown under the input as it
    is typed (see _read_tty_line); it is ignored when stdin is a pipe.
    """
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _read_user_input_interactive(hint)

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
SESSION_AUTO_APPROVE: bool = False
# Set by run_headless(): there's no one to ask, so confirm() declines instead of prompting.
HEADLESS: bool = False

# Escape (or Ctrl+C during a turn) sets this so the agent loop returns to the prompt.
_CANCEL_REQUESTED = threading.Event()
_LISTENER_STOP = threading.Event()


class UserCancelled(Exception):
    """Raised when the user presses Escape or Ctrl+C during an agent turn."""


# "WRENCODE" in the ANSI Shadow figlet font. Block faces get a sky-to-deep-blue
# gradient down the rows and the box-drawing shadow a darker blue (256-color).
_BANNER_ROWS = (
    "██╗    ██╗██████╗ ███████╗███╗   ██╗ ██████╗ ██████╗ ██████╗ ███████╗",
    "██║    ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝",
    "██║ █╗ ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║   ██║██║  ██║█████╗",
    "██║███╗██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██║   ██║██║  ██║██╔══╝",
    "╚███╔███╔╝██║  ██║███████╗██║ ╚████║╚██████╗╚██████╔╝██████╔╝███████╗",
    " ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
)
_BANNER_FACES = (117, 111, 75, 69, 33, 27)
_BANNER_SHADOW = 25


def render_banner(color: bool) -> str:
    """Return the startup banner, colored when color is True."""
    if not color:
        return "\n".join(_BANNER_ROWS)
    shade = f"\033[38;5;{_BANNER_SHADOW}m"
    lines = []
    for row, face in zip(_BANNER_ROWS, _BANNER_FACES):
        tint = f"\033[38;5;{face}m"
        runs = re.sub(
            r"█+|[^█ ]+",
            lambda m, tint=tint: (tint if m[0][0] == "█" else shade) + m[0],
            row,
        )
        lines.append(runs + RESET)
    return "\n".join(lines)


def _cancel_listener() -> None:
    """Watch stdin for Escape while a blocking agent operation runs."""
    if not sys.stdin.isatty():
        return
    # Without termios or a raw-capable stdin we only lose Escape-to-cancel.
    with contextlib.suppress(Exception):
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
    """Raise UserCancelled if the user (or this agent's parallel batch) cancelled."""
    batch = getattr(_AGENT_LOCAL, "cancel", None)
    if _CANCEL_REQUESTED.is_set() or (batch is not None and batch.is_set()):
        raise UserCancelled()


def confirm(action: str = "", question: str = "", subject: str = "") -> str:
    """Prompt for approval. Returns 'ok' or a cancellation message for the agent.

    `question` is the one line asked ("Apply to app.py?"); `action` names the
    tool (bash, edit, write) and `subject` what it acts on (the command or the
    path), which the permission rules are matched against: a deny rule refuses
    without asking, an allow rule approves without asking. Otherwise Enter/y
    approves once; ``a`` approves all remaining actions this session; ``s``
    saves a rule so this kind of action never asks again; ``n`` declines and
    asks what to do differently. Auto-approve via WRENCODE_AUTO_APPROVE / --yes
    enables headless use and subagents.
    """
    rules = permissions.ACTIVE
    rule = rules.check(action, subject) if rules is not None and subject else None
    label = f"{action} {subject}".strip()[:120] if subject else (action or "action")
    if rule is not None and rule.effect == "deny":
        print(f"{YELLOW}⊘ {label} [denied by rule {rule}]{RESET}")
        return (
            f"cancelled: denied by the permission rule {rule} ({rule.source}). "
            "Do what you can without it and say what is left to do."
        )
    if rule is not None:
        print(f"{DIM}✓ {label} [allowed by rule {rule}]{RESET}")
        return "ok"
    if (
        os.environ.get("WRENCODE_AUTO_APPROVE", "").lower() in ("1", "true", "yes")
        or SESSION_AUTO_APPROVE
    ):
        label = action or "action"
        print(f"{DIM}⚠ {label} [auto-approved]{RESET}")
        return "ok"
    if HEADLESS:
        print(f"{DIM}⚠ {action or 'action'} [declined: headless without --yes]{RESET}")
        return (
            "cancelled: running headless without --yes, so this action can't be "
            "approved. Do what you can without it and say what is left to do."
        )
    if _agent_tag():
        return _confirm_from_subagent(action, question, subject)
    return _confirm_prompt(question, action, subject)


# One approval prompt at a time across parallel subagents.
_APPROVAL_LOCK = threading.Lock()
# The Escape listener of the running parallel batch, paused while a prompt reads stdin.
PARALLEL_ESC: _EscWatch | None = None


def _confirm_from_subagent(action: str, question: str = "", subject: str = "") -> str:
    """Ask for approval on behalf of a parallel subagent.

    Prompts take turns, the Escape listener lets go of stdin, and the other
    agents' output is held until the answer is in, so the prompt stays readable.
    """
    with _APPROVAL_LOCK:
        check_cancelled()
        out = sys.stdout if isinstance(sys.stdout, _AgentStdout) else None
        esc = PARALLEL_ESC
        if out is not None:
            out.hold()
        if esc is not None:
            esc.stop()
        try:
            detail = getattr(_AGENT_LOCAL, "last_action", "") or action
            print(f"{YELLOW}[{_agent_tag()}] needs approval:{RESET} {detail}")
            return _confirm_prompt(question, action, subject)
        finally:
            if esc is not None:
                esc.start()
            if out is not None:
                out.release()


def _confirm_prompt(question: str = "", action: str = "", subject: str = "") -> str:
    """The interactive approve / allow-all / save-rule / decline prompt: one line, then ❯."""
    global SESSION_AUTO_APPROVE
    ask = question or "Allow this?"
    offer = (
        permissions.suggest(action, subject)
        if permissions.ACTIVE is not None and action in permissions.TOOLS and subject
        else ""
    )
    keys = "Enter yes · a always · " + (f"s allow {offer} · " if offer else "") + "n no"
    print(
        f"{BOLD}{ask}{RESET}  {DIM}{keys}{RESET}"
        if colors_enabled()
        else f"{ask}  {keys}"
    )
    while True:
        try:
            choice = input(f"{BLUE}❯{RESET} ").strip().lower()
        except KeyboardInterrupt:
            print()
            return "cancelled: user interrupted"
        if choice in ("", "y", "yes"):
            return "ok"
        if choice in ("a", "all"):
            SESSION_AUTO_APPROVE = True
            print(f"{DIM}Auto-approving remaining actions this session.{RESET}")
            return "ok"
        if choice in ("s", "save") and offer and permissions.ACTIVE is not None:
            rule = permissions.ACTIVE.add(offer, "allow", "project")
            print(
                f"{DIM}Saved {rule} to {permissions.PROJECT_FILE}; it won't ask again "
                f"(/permissions lists and removes rules).{RESET}"
            )
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
        print(f"{DIM}Choose Enter, a, {'s, ' if offer else ''}or n.{RESET}")


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


def _highlight_code(code: str, base: str = "") -> str:
    """Apply light ANSI syntax coloring to a code block (best-effort, any language).

    `base` is the color the rest of the code is drawn in; it is restored after
    every token so the block keeps its tint.
    """

    def color(m: re.Match[str]) -> str:
        g = m.lastgroup
        if g == "comment":
            return f"{DIM}{m.group()}{RESET}{base}"
        if g == "string":
            return f"{GREEN}{m.group()}{RESET}{base}"
        if g == "num":
            return f"{YELLOW}{m.group()}{RESET}{base}"
        if g == "kw":
            return f"{BLUE}{m.group()}{RESET}{base}"
        return m.group()

    return _CODE_TOKEN.sub(color, code)


def print_diff(label: str, before: str, after: str, limit: int = 40) -> None:
    """Show what a change does to a file: a unified diff with two lines of context,
    removed lines red, added lines green, each hunk headed by `label:line`."""
    import difflib

    out: list[str] = []
    for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=2
    ):
        if line.startswith(("---", "+++")):
            continue
        shown = visible(line)
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)", line)
            out.append(f"{DIM}@@ {label}:{m.group(1) if m else '?'}{RESET}")
        elif line.startswith("+"):
            out.append(f"{GREEN}{shown}{RESET}")
        elif line.startswith("-"):
            out.append(f"{RED}{shown}{RESET}")
        else:
            out.append(f"{DIM}{shown}{RESET}")
    for line in out[:limit]:
        print(f"    {line}")
    if len(out) > limit:
        print(f"    {DIM}… +{len(out) - limit} more lines{RESET}")


def render_markdown(text: str) -> str:
    """Render fenced code blocks (tinted and lightly highlighted), inline code,
    bold, and headings; `prose` is the color to return to after a span."""
    blocks: list[str] = []

    def stash(m: re.Match[str]) -> str:
        lang = m.group(1) or ""
        body = _highlight_code(m.group(2).rstrip("\n"), CODE_TEXT)
        head = f"{DIM}┌─ {lang}{RESET}\n" if lang else f"{DIM}┌─{RESET}\n"
        bordered = "\n".join(
            f"{DIM}│{RESET} {CODE_TEXT}{ln}{RESET}" for ln in body.split("\n")
        )
        blocks.append(f"\n{head}{bordered}\n{DIM}└─{RESET}")
        return f"\x00B{len(blocks) - 1}\x00"

    text = re.sub(r"```(\w*)\n?(.*?)```", stash, text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", f"{CYAN}\\1{RESET}", text)
    text = re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)
    text = re.sub(r"^#{1,6} +(.+)$", f"{BOLD}\\1{RESET}", text, flags=re.MULTILINE)
    for i, b in enumerate(blocks):
        text = text.replace(f"\x00B{i}\x00", b)
    return re.sub(r"\n{3,}", "\n\n", text)  # one blank line around a block, not two


class StreamPrinter:
    """Print a reply as it streams, rendered line by line.

    The first piece opens the reply with the dot, like print_agent_message.
    Text is shown as it arrives; once a line is complete it is rendered
    (inline code, bold, headings), redrawn in place when it fits on one row.
    Fenced code is drawn with the gutter and tint as its lines complete.
    `on_first` runs before the first character is printed (to stop a spinner).
    """

    def __init__(self, on_first: Callable[[], None] | None = None) -> None:
        self.on_first = on_first
        self.started = False
        self._line = ""  # the line in progress, raw
        self._shown = 0  # how much of it is already on screen
        self._fence = False
        self._first_line = True
        self._cur_lead = "  "  # what the line in progress was opened with

    def _lead(self) -> str:
        self._cur_lead = f"{AGENT_MARK} " if self._first_line else "  "
        self._first_line = False
        return self._cur_lead

    def feed(self, text: str) -> None:
        if not text:
            return
        if not self.started:
            self.started = True
            if self.on_first is not None:
                self.on_first()
        self._line += visible(text)
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            self._finish_line(line)
            self._shown = 0
        self._show_partial()

    def _show_partial(self) -> None:
        if self._shown == 0 and (self._line or not self._fence):
            if not self._line:
                return
            sys.stdout.write(
                self._lead() + ("" if not self._fence else f"{DIM}│{RESET} ")
            )
        chunk = self._line[self._shown :]
        if chunk:
            color = CODE_TEXT if self._fence else AGENT_TEXT
            sys.stdout.write(f"{color}{chunk}{RESET}")
            self._shown = len(self._line)
        sys.stdout.flush()

    def _finish_line(self, line: str) -> None:
        """Render a completed line, replacing what streamed if it fits one row."""
        lead = self._lead() if self._shown == 0 else None
        if line.startswith("```"):
            self._fence = not self._fence
            lang = line[3:].strip()
            bar = (
                f"{DIM}┌─ {lang}{RESET}"
                if self._fence and lang
                else (f"{DIM}┌─{RESET}" if self._fence else f"{DIM}└─{RESET}")
            )
            self._replace(lead, bar, line)
            return
        if self._fence:
            body = f"{DIM}│{RESET} {CODE_TEXT}{_highlight_code(line, CODE_TEXT)}{RESET}"
            self._replace(lead, body, line)
            return
        rendered = render_markdown(line).replace(RESET, f"{RESET}{AGENT_TEXT}")
        self._replace(lead, f"{AGENT_TEXT}{rendered}{RESET}", line)

    def _replace(self, lead: str | None, rendered: str, raw: str) -> None:
        import shutil

        if lead is None:  # part of the raw line is on screen already
            width = shutil.get_terminal_size().columns
            if len(raw) + 2 < width and _ANSI_RE.sub("", rendered) != raw:
                sys.stdout.write("\r\033[K" + self._cur_lead + rendered + "\n")
            else:
                rest = raw[self._shown :]
                color = CODE_TEXT if self._fence else AGENT_TEXT
                sys.stdout.write(f"{color}{rest}{RESET}\n")
        else:
            sys.stdout.write(lead + rendered + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        """End the reply: finish the last line and leave a blank one."""
        if not self.started:
            return
        if self._line:
            self._finish_line(self._line)
            self._line = ""
        elif self._shown:
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()


def print_agent_message(text: str) -> None:
    """Print the agent's reply: a cyan dot opens it, the text hangs under it.

    Prose is near-white, inline code cyan, fenced code warm with a gutter, so
    what the model says, what it quotes and what it wrote stand apart; the
    user's own line keeps the ❯ prompt.
    """
    rendered = render_markdown(visible(text)).replace(RESET, f"{RESET}{AGENT_TEXT}")
    for i, line in enumerate(rendered.split("\n")):
        lead = f"{AGENT_MARK} " if i == 0 else "  "
        print(f"{lead}{AGENT_TEXT}{line}{RESET}")
    print()


# -----------------------------------------------------------------------------------------------
# Parallel subagents: tagged output and a pausable Escape listener
# -----------------------------------------------------------------------------------------------
# When one reply makes several task calls, wrencode runs the subagents at the same
# time, each in its own thread. Their output is tagged [1], [2]... and written a
# whole line at a time; approvals take turns (_confirm_from_subagent).
class _AgentStdout:
    """Stdout for a parallel batch: each subagent's lines are tagged and kept whole."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self._lock = threading.Lock()
        self._partial: dict[int, str] = {}
        self._owner: int | None = None  # thread at an approval prompt
        self._held: list[str] = []

    def write(self, text: str) -> int:
        tag = _agent_tag()
        me = threading.get_ident()
        with self._lock:
            if not tag or me == self._owner:
                self._real.write(text)
                return len(text)
            *lines, rest = (self._partial.get(me, "") + text).split("\n")
            self._partial[me] = rest
            for line in lines:
                tagged = f"{DIM}[{tag}]{RESET} {line.replace(chr(13), '')}\n"
                if self._owner is None:
                    self._real.write(tagged)
                else:
                    self._held.append(tagged)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._real.flush()

    def hold(self) -> None:
        """Pass this thread's writes straight through; hold everyone else's."""
        with self._lock:
            self._owner = threading.get_ident()

    def release(self) -> None:
        with self._lock:
            self._owner = None
            for line in self._held:
                self._real.write(line)
            self._held.clear()
            self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _EscWatch:
    """Escape-to-cancel listener that can pause while an approval prompt reads stdin."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or not sys.stdin.isatty() or HEADLESS:
            return
        _LISTENER_STOP.clear()
        self._thread = threading.Thread(target=_cancel_listener, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        _LISTENER_STOP.set()
        self._thread.join(timeout=0.5)
        self._thread = None


def _clip(text: str, width: int) -> str:
    """Cut `text` to `width` visible cells, keeping its color codes and ending in …."""
    out, seen = [], 0
    for part in re.split(r"(\x1b\[[0-9;]*m)", text):
        if part.startswith("\x1b"):
            out.append(part)
            continue
        room = width - seen
        if len(part) > room:
            out.append(part[: max(room - 1, 0)] + "…")
            seen = width
            break
        out.append(part)
        seen += len(part)
    return "".join(out)


def _pick_with_arrows(title: str, labels: list[str], initial: int) -> int | None:
    """The arrow-key menu behind pick_from_list: ↑↓ move, Enter picks, Esc or q
    cancels, a digit jumps to that entry."""
    import shutil
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sel = initial
    typed = ""
    width = (
        shutil.get_terminal_size().columns - 3
    )  # a wrapped row would break the redraw
    rows = [_clip(label, width) for label in labels]
    if title:
        print(f"{BOLD}{title}{RESET}  {DIM}↑↓ Enter · Esc cancels{RESET}")

    def draw(first: bool) -> None:
        out = "" if first else f"\033[{len(rows)}A"
        for i, label in enumerate(rows):
            out += "\r\033[K"
            if i == sel:
                out += f"{BRIGHT_CYAN}❯{RESET} {BOLD}{label}{RESET}\n"
            else:
                out += f"  {label}\n"
        sys.stdout.write(out)
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        draw(True)
        while True:
            key = _read_tty_key(fd)
            if key in ("", "esc", "q", "ctrl_c", "ctrl_d"):
                return None
            if key == "enter":
                return sel
            if key in ("up", "down"):
                sel = (sel + (-1 if key == "up" else 1)) % len(labels)
                typed = ""
            elif key.isdigit():
                typed += key
                if typed.isdigit() and 1 <= int(typed) <= len(labels):
                    sel = int(typed) - 1
                if len(typed) >= len(str(len(labels))):
                    typed = ""
            else:
                continue
            draw(False)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def pick_from_list(
    title: str,
    options: list[str],
    *,
    labels: list[str] | None = None,
    initial_index: int = 0,
) -> int | None:
    """Pick one option: arrow keys and Enter on a terminal (Esc cancels, a
    number jumps), a numbered prompt otherwise."""
    if not options:
        print(f"{YELLOW}No options available.{RESET}")
        return None
    labels = labels if labels is not None else options
    initial_index = max(0, min(initial_index, len(options) - 1))
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _pick_with_arrows(title, labels, initial_index)
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
