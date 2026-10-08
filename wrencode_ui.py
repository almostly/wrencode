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
from typing import Any

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
AGENT_TEXT = "\033[38;5;245m"  # muted grey for agent replies (Claude Code-style)
_COMPOSE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_LOADER_MODEL_MAX = 32


def loader_context(backend: str, model: str) -> str:
    """The loader's label: the backend and a shortened model id."""
    b = backend or "backend"
    if model and len(model) > _LOADER_MODEL_MAX:
        keep = _LOADER_MODEL_MAX - 1
        head = max(8, keep // 2)
        m = f"{model[:head]}…{model[-(keep - head) :]}"
    else:
        m = model or "model"
    return f"{b} · {m} · waiting…"


def loader_display(step: int, context: str) -> str:
    """Render one animated loader frame for the given step."""
    sym = _COMPOSE_FRAMES[step % len(_COMPOSE_FRAMES)]
    if not colors_enabled():
        return f"{sym} {context}"
    return f"{BRIGHT_CYAN}{sym}{RESET} {DIM}{context}{RESET}"


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


def print_system(text: str, *, end: str = "\n") -> None:
    """Print slash-command / configure feedback in banner cyan."""
    s = f"{BOLD}{BRIGHT_CYAN}" if colors_enabled() else ""
    e = RESET if colors_enabled() else ""
    if sys.stdout.isatty():
        sys.stdout.write("\r")
    sys.stdout.write(f"{s}{text}{e}{end}")
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
    "/resume": "continue one: /resume <id>",
    "/search": "search past conversations: /search <text>",
    "/help": "list commands",
    "/quit": "save history and exit",
}
# Short aliases: accepted and highlighted, but kept out of the menu.
SLASH_ALIASES: dict[str, str] = {"/c": "/clear", "/q": "/quit", "/exit": "/quit"}
MAX_SLASH_MENU = 10


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


def _redraw_input_line(
    text: str, matches: list[str] | None = None, sel: int = 0
) -> None:
    """Redraw the prompt line plus a completion menu below it, cursor kept on the line."""
    line = format_input_line(text)
    out = "\r\033[J" + line
    for i, cmd in enumerate(matches or []):
        desc = SLASH_COMMANDS.get(cmd, "")
        if not colors_enabled():
            out += f"\n{'>' if i == sel else ' '} {cmd:<12} {desc}"
        elif i == sel:
            out += f"\n  {BOLD}{BRIGHT_CYAN}{cmd:<12}{RESET} {desc}"
        else:
            out += f"\n  {DIM}{cmd:<12} {desc}{RESET}"
    if matches:
        col = 2 + len(text)  # "❯ " is two cells
        out += f"\033[{len(matches)}A\r" + (f"\033[{col}C" if col else "")
    sys.stdout.write(out)
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
    redraw: Any | None = None,
    complete: bool = False,
) -> str:
    """Read one line in cbreak mode; swallows arrow keys unless history=True.

    With complete=True, typing a /prefix shows matching slash commands below
    the line: ↑↓ pick, Tab or → fills in, Enter runs the highlighted one.
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []
    hist_idx = len(_INPUT_HISTORY)
    sel = 0

    def _matches() -> list[str]:
        return slash_matches("".join(buf)) if complete else []

    def _redraw() -> None:
        if complete:
            _redraw_input_line("".join(buf), _matches(), sel)
        elif redraw is not None:
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
            matches = _matches()
            if key == "enter":
                text = "".join(buf).strip()
                if matches and not _is_slash_command(text):
                    text = matches[min(sel, len(matches) - 1)]
                if complete:  # clear the menu, leave the final line
                    _redraw_input_line(text)
                sys.stdout.write("\n")
                sys.stdout.flush()
                return text
            if matches and key in {"\t", "right"}:
                buf = list(matches[min(sel, len(matches) - 1)])
                sel = 0
                _redraw()
                continue
            if matches and key in {"up", "down"}:
                step = -1 if key == "up" else 1
                sel = (min(sel, len(matches) - 1) + step) % len(matches)
                _redraw()
                continue
            if key == "backspace":
                if buf:
                    buf.pop()
                    sel = 0
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
                sel = 0
                _redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _remember_input(text: str) -> None:
    if text and (not _INPUT_HISTORY or _INPUT_HISTORY[-1] != text):
        _INPUT_HISTORY.append(text)


def _read_user_input_interactive() -> str:
    """TTY line editor with live slash-command coloring and history."""
    text = _read_tty_line("", history=True, complete=True)
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


def confirm(action: str = "") -> str:
    """Prompt for approval. Returns 'ok' or a cancellation message for the agent.

    Enter/y approves once; ``a`` approves all remaining actions this session;
    ``n`` declines and asks what to do differently. Auto-approve via
    WRENCODE_AUTO_APPROVE / --yes enables headless use and subagents.
    """
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
        return _confirm_from_subagent(action)
    return _confirm_prompt()


# One approval prompt at a time across parallel subagents.
_APPROVAL_LOCK = threading.Lock()
# The Escape listener of the running parallel batch, paused while a prompt reads stdin.
PARALLEL_ESC: _EscWatch | None = None


def _confirm_from_subagent(action: str) -> str:
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
            return _confirm_prompt()
        finally:
            if esc is not None:
                esc.start()
            if out is not None:
                out.release()


def _confirm_prompt() -> str:
    """The interactive approve / allow-all / decline prompt."""
    global SESSION_AUTO_APPROVE
    print(f"{DIM}Enter/y   approve once{RESET}")
    print(f"{DIM}a         allow all for this session{RESET}")
    print(f"{DIM}n         decline{RESET}")
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

    def color(m: re.Match[str]) -> str:
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

    def stash(m: re.Match[str]) -> str:
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
    for line in render_markdown(visible(text)).split("\n"):
        print(f"{AGENT_TEXT}{line}{RESET}")
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


def pick_from_list(
    title: str,
    options: list[str],
    *,
    labels: list[str] | None = None,
    initial_index: int = 0,
) -> int | None:
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
