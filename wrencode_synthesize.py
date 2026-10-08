"""`wrencode synthesize`: fuse agent chat transcripts into one synthesis
(normalize -> extract -> reconcile), with model-agnostic transcript adapters.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any

import wrencode_backends as backends
import wrencode_ui as ui
from wrencode_ui import BLUE, BOLD, DIM, RED, RESET, YELLOW

# -----------------------------------------------------------------------------------------------
# synthesize — fuse multiple agent chat transcripts into one provenance-cited synthesis.
# A semantic git-merge for chats: extract each chat's decisions, then reconcile them into
# Reinforced (≥2 agree) / Unique / ⚠ Conflict, every claim carrying a chat-id citation.
# -----------------------------------------------------------------------------------------------
SYNTH_EXTRACT_SYS = (
    "You analyze ONE coding-agent chat transcript and extract its substantive "
    "engineering content. Return ONLY a JSON object, no prose, with this exact "
    'shape: {"decisions": [{"subject": str, "conclusion": str}], '
    '"problems_solved": [{"problem": str, "resolution": str}], '
    '"files_touched": [str], "open_questions": [str]}. '
    "A decision's `subject` is a short stable slug for WHAT was decided "
    "('storage-format', 'auth-strategy') — never the answer; `conclusion` is the "
    "chosen answer ('Iceberg'). Two chats deciding the same subject share the "
    "same slug. Keep each value one terse phrase. Capture conclusions, not "
    "narration. Do NOT continue the conversation — extract from a finished log."
)
SYNTH_RECONCILE_SYS = (
    "You MERGE structured facts from multiple coding-agent chats, like a semantic "
    "git merge. You are given a JSON array of per-chat fact sets, each tagged with "
    "its chat id. Align decisions by their `subject` slug: same subject + same "
    "conclusion → reinforced; same subject + different conclusion → CONFLICT; a "
    "subject only one chat has → unique. Produce Markdown with EXACTLY:\n\n"
    "## Reinforced decisions\nSubjects ≥2 chats agree on; cite ids e.g. `(a1b2, c3d4)`.\n\n"
    "## Unique contributions\nDecisions/findings only one chat has; cite the source.\n\n"
    "## ⚠ Conflicts\nWhere chats CONTRADICT on the same entity (different root "
    "cause, reversed decision, incompatible approach). Show both sides with "
    "citations. If a later chat overrode an earlier conclusion, mark it resolved "
    "and name the winner. If genuinely none, write 'None detected.'\n\n"
    "## Open questions\nUnresolved items across all chats, cited.\n\n"
    "Every claim MUST carry a chat-id citation. Never invent agreement."
)
SYNTH_DIFF_SYS = (
    "You compare structured facts from multiple coding-agent chats and report ONLY "
    "where they DIVERGE, like `git diff`. Given a JSON array of per-chat fact sets "
    "(each tagged with its chat id), align decisions by their `subject` slug (same "
    "subject + different conclusion = a conflict). Produce Markdown with EXACTLY:\n\n"
    "## ⚠ Conflicts\nDirect contradictions on the same subject/entity (different root cause, "
    "reversed decision, incompatible approach); show both sides and cite ids. If a "
    "later chat overrode an earlier one, mark it resolved and name the winner.\n\n"
    "## Only in one chat\nDecisions/findings present in just one chat, grouped by id.\n\n"
    "Omit everything the chats agree on. Every claim cites a chat id. If there is no "
    "divergence at all, write 'No divergence — the chats agree.'"
)
SYNTH_LOG_SYS = (
    "You build a CHRONOLOGICAL decision log from multiple coding-agent chats, like "
    "`git log`. The JSON array of per-chat fact sets is ordered OLDEST→NEWEST. Emit a "
    "single Markdown timeline, oldest first, one bullet per decision or resolved "
    "problem:\n\n- **<chat id>** — <what was decided/resolved>\n\n"
    "Keep the listed order within a chat. When a later chat reverses or supersedes an "
    "earlier decision, add '↳ supersedes <id>: …'. Cite the chat id on every line. "
    "End with '## Net state' summarizing where things landed."
)
SYNTH_MODES = {
    "merge": SYNTH_RECONCILE_SYS,
    "diff": SYNTH_DIFF_SYS,
    "log": SYNTH_LOG_SYS,
}


def _synth_complete(
    system_prompt: str,
    user_text: str,
    prefill: str = "",
    mlx_state: tuple[Any, Any] | None = None,
) -> str:
    """One-shot completion on the configured backend, deterministic, no agent tools.

    `prefill` forces the reply to continue from given text (used for JSON); the
    Anthropic and Bedrock APIs honor it, the others ignore it.
    """
    return backends.complete(
        system_prompt, user_text, temperature=0.0, prefill=prefill, mlx_state=mlx_state
    )


def _parse_claude_code_jsonl(raw: str) -> list[dict[str, str]] | None:
    """Extract user prompts + assistant text from a Claude Code JSONL log, or None.

    Drops tool calls, tool results, and thinking — only the user↔assistant signal
    survives. Returns None when the input isn't Claude Code JSONL.
    """
    turns: list[dict[str, str]] = []
    looks_jsonl = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            return None  # a non-JSON line means this isn't a JSONL transcript
        if not isinstance(o, dict) or "type" not in o:
            return None
        looks_jsonl = True
        m = o.get("message")
        if not isinstance(m, dict):
            continue
        role, content = o.get("type"), m.get("content")
        if role == "user" and isinstance(content, str):
            t = content.strip()
            if t and not t.startswith("<"):  # skip injected/system-ish prompts
                turns.append({"role": "user", "text": t})
        elif role == "assistant" and isinstance(content, list):
            text = "\n".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text:
                turns.append({"role": "assistant", "text": text})
    return turns if looks_jsonl else None


# Common key names different agents use for a message's role and its content. The
# generic adapters sniff these so wrencode stays tool-agnostic (Codex, OpenAI-style
# logs, ChatGPT exports, …) without a hand-written schema per tool.
_ROLE_KEYS = ("role", "type", "sender", "author")
_CONTENT_KEYS = ("content", "text", "message", "body", "parts")
_USER_ROLES = {"user", "human", "prompt", "you"}
_ASSISTANT_ROLES = {"assistant", "ai", "model", "bot", "agent", "gpt", "claude"}


def _coerce_role(o: dict[str, Any]) -> str | None:
    """Map any of the common role fields onto 'user'/'assistant', else None."""
    for k in _ROLE_KEYS:
        v = o.get(k)
        if isinstance(v, str):
            r = v.lower()
            if r in _USER_ROLES:
                return "user"
            if r in _ASSISTANT_ROLES:
                return "assistant"
    return None


def _coerce_text(v: Any) -> str:
    """Flatten a content value (str / list of blocks / nested dict) to plain text."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for b in v:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                t = b.get("text") or b.get("content") or b.get("value")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    if isinstance(v, dict):
        return _coerce_text(v.get("text") or v.get("content") or "")
    return ""


def _record_to_turn(o: Any) -> dict[str, str] | None:
    """Convert one generic message record to a {role, text} turn, or None to skip."""
    if not isinstance(o, dict):
        return None
    m = o.get("message") if isinstance(o.get("message"), dict) else o
    role = _coerce_role(m) or _coerce_role(o)
    if role not in {"user", "assistant"}:
        return None
    text = ""
    for k in _CONTENT_KEYS:
        if k in m:
            text = _coerce_text(m[k])
            if text:
                break
    if not text or (role == "user" and text.startswith("<")):
        return None
    return {"role": role, "text": text}


def _adapt_generic_jsonl(raw: str) -> list[dict[str, str]] | None:
    """Adapt a JSONL log where each line is a message-ish dict (Codex/OpenAI-style)."""
    turns: list[dict[str, str]] = []
    saw = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(o, dict):
            return None
        saw = True
        t = _record_to_turn(o)
        if t:
            turns.append(t)
    return turns if saw and turns else None


def _adapt_messages_json(raw: str) -> list[dict[str, str]] | None:
    """Adapt a single JSON value: a list of messages or {messages|conversation:[…]}."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    seq: Any = None
    if isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        for k in ("messages", "conversation", "turns", "history", "chat"):
            if isinstance(data.get(k), list):
                seq = data[k]
                break
    if not isinstance(seq, list):
        return None
    turns = [t for t in (_record_to_turn(o) for o in seq) if t]
    return turns or None


# Tried in order; the precise Claude Code adapter wins before the generic sniffers
# (it correctly drops thinking/tool blocks that the generic one would keep).
SYNTH_ADAPTERS: list[tuple[str, Any]] = [
    ("claude-code", _parse_claude_code_jsonl),
    ("jsonl", _adapt_generic_jsonl),
    ("messages-json", _adapt_messages_json),
]


def _chat_ids(paths: list[str]) -> list[str]:
    """Short citation ids for the selected transcripts, unique within this run.

    The first eight characters of the file name (a Claude Code session id prefix),
    unless two transcripts share them: those get their whole name, and a numeric
    suffix if even that collides.
    """
    stems = [pathlib.Path(p).stem or "chat" for p in paths]
    short = [s[:8] for s in stems]
    ids = [stems[i] if short.count(c) > 1 else c for i, c in enumerate(short)]
    seen: dict[str, int] = {}
    for i, cid in enumerate(ids):
        n = seen.get(cid, 0)
        seen[cid] = n + 1
        if n:
            ids[i] = f"{cid}-{n + 1}"
    return ids


def _synth_normalize(path: str, cid: str | None = None) -> dict[str, Any]:
    """Load a transcript into the common {id, source, turns:[{role,text}]} shape."""
    raw = pathlib.Path(path).read_text(errors="replace")
    turns: list[dict[str, str]] | None = None
    source = "text"
    for name, adapt in SYNTH_ADAPTERS:
        turns = adapt(raw)
        if turns:
            source = name
            break
    if not turns:  # unknown format → treat the whole file as one block
        turns = [{"role": "user", "text": raw}]
        source = "text"
    if cid is None:
        cid = _chat_ids([path])[0]
    return {"id": cid, "source": source, "turns": turns}


def _transcript_budget() -> int:
    """Characters of transcript to send for extraction: most of the context window
    (about 4 characters per token), leaving room for the instructions and the answer."""
    return int(backends.CONTEXT_TOKENS * 4 * 0.6)


def _synth_render(chat: dict[str, Any], max_chars: int | None = None) -> str:
    """Flatten a normalized chat to a tagged transcript string that fits the budget.

    A transcript over budget keeps its start (the task) and, mostly, its end: the
    latest decisions are the ones that override earlier ones.
    """
    s = "\n\n".join(f"[{t['role'].upper()}] {t['text']}" for t in chat["turns"])
    limit = _transcript_budget() if max_chars is None else max_chars
    if len(s) <= limit:
        return s
    head = limit // 4
    tail = limit - head
    return f"{s[:head]}\n\n[... {len(s) - head - tail} characters omitted ...]\n\n{s[-tail:]}"


def _json_slice(s: str) -> str:
    """Return the outermost {...} span of s (tolerates prose around the JSON)."""
    i, j = s.find("{"), s.rfind("}")
    return s[i : j + 1] if i != -1 and j > i else s


_FACT_KEYS = ("decisions", "problems_solved", "files_touched", "open_questions")


def _normalize_facts(facts: Any) -> dict[str, Any]:
    """Keep only the expected fact lists, each a list, so reconcile gets a clean shape."""
    src = facts if isinstance(facts, dict) else {}
    return {k: v if isinstance(v := src.get(k), list) else [] for k in _FACT_KEYS}


def _synth_extract(
    chat: dict[str, Any], mlx_state: tuple[Any, Any] | None = None
) -> dict[str, Any]:
    """Extract one chat's structured facts, tagged with its chat id (provenance).

    When the model doesn't answer with JSON, the facts are empty and `_parse_error`
    carries the start of what it said instead.
    """
    user = (
        f"<transcript chat_id={chat['id']}>\n{_synth_render(chat)}\n</transcript>\n\n"
        "Extract the JSON object described in your instructions from the transcript "
        "above. Do NOT continue the conversation; output only JSON."
    )
    raw = _synth_complete(SYNTH_EXTRACT_SYS, user, prefill="{", mlx_state=mlx_state)
    try:
        parsed = json.loads(_json_slice(raw))
        if not isinstance(parsed, dict):
            raise TypeError("not an object")
        facts = _normalize_facts(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        facts = _normalize_facts({})
        facts["_parse_error"] = raw[:200]
    facts["chat"] = chat["id"]
    return facts


def _synth_reconcile(
    fact_sets: list[dict[str, Any]],
    mode: str = "merge",
    mlx_state: tuple[Any, Any] | None = None,
) -> str:
    """Fuse per-chat fact sets into Markdown — merge (default), diff, or log."""
    return _synth_complete(
        SYNTH_MODES.get(mode, SYNTH_RECONCILE_SYS),
        "Fact sets:\n\n" + json.dumps(fact_sets, indent=2),
        mlx_state=mlx_state,
    )


def _claude_project_dir() -> pathlib.Path | None:
    """Return this workspace's Claude Code transcript dir (~/.claude/projects/...) if any."""
    cwd = os.environ.get("WRENCODE_WORKSPACE") or str(pathlib.Path.cwd().resolve())
    encoded = cwd.replace("/", "-")  # Claude Code encodes the cwd path with dashes
    d = pathlib.Path.home() / ".claude" / "projects" / encoded
    return d if d.is_dir() else None


def _discover_transcripts(dirs: list[str]) -> list[str]:
    """Find *.jsonl transcripts in the given directories, newest first."""
    found: list[str] = []
    for d in dirs:
        found += [str(x) for x in pathlib.Path(d).glob("*.jsonl")]
    found.sort(key=lambda f: pathlib.Path(f).stat().st_mtime, reverse=True)
    return found


def _ago(seconds: float) -> str:
    """Compact relative age: 5m, 3h, 2d."""
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _first_user_prompt(path: str) -> str:
    """Peek at a transcript's first real user prompt for use as a picker label."""
    try:
        with open(path) as fh:
            for _ in range(400):
                line = fh.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn = _record_to_turn(o)
                if turn and turn["role"] == "user":
                    return ui.visible(" ".join(turn["text"].split()))
    except OSError:
        pass
    return "(no prompt found)"


def _transcript_label(path: str) -> str:
    """One-line picker label: id, size, age, first-prompt snippet."""
    p = pathlib.Path(path)
    size = p.stat().st_size
    human = f"{size / 1024:.0f}K" if size < (1 << 20) else f"{size / (1 << 20):.1f}M"
    age = _ago(time.time() - p.stat().st_mtime)
    return f"{p.stem[:8]}  {human:>6}  {age:>4} ago  {_first_user_prompt(path)[:64]}"


def _parse_selection(raw: str, n: int) -> list[int]:
    """Parse '1 3 5', '1-3', '1,2', or 'all' into sorted 0-based indices within [1,n]."""
    raw = raw.strip().lower()
    if raw in {"all", "*", "a"}:
        return list(range(n))
    out: list[int] = []
    for tok in raw.replace(",", " ").split():
        a, sep, b = tok.partition("-")
        if sep and a.isdigit() and b.isdigit():
            out += range(int(a), int(b) + 1)
        elif tok.isdigit():
            out.append(int(tok))
    return sorted({i - 1 for i in out if 1 <= i <= n})


def _synth_pick_numbered(candidates: list[str]) -> list[str]:
    """Fallback picker: numbered list, accepts numbers / ranges / 'all'."""
    print()
    ui.print_system(
        "Select transcripts to synthesize (numbers, ranges like 1-3, or 'all'):"
    )
    print()
    for i, path in enumerate(candidates, 1):
        print("  " + str(i) + ". " + _transcript_label(path))
    while True:
        raw = input(f"{BLUE}❯{RESET} selection [all]: ").strip() or "all"
        idx = _parse_selection(raw, len(candidates))
        if idx:
            return [candidates[i] for i in idx]
        print(f"{RED}Enter numbers 1-{len(candidates)}, a range, or 'all'.{RESET}")


def _synth_pick_tty(candidates: list[str]) -> list[str]:
    """Arrow-key checkbox picker: ↑/↓ move, space toggle, a all, enter confirm, esc cancel."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    labels = [_transcript_label(p) for p in candidates]
    n = len(candidates)
    selected = [True] * n  # default: everything on
    cursor = 0
    header = (
        f"{BOLD}Select transcripts{RESET}  "
        f"{DIM}↑/↓ move · space toggle · a all · enter confirm · esc cancel{RESET}"
    )

    def render(first: bool) -> None:
        if not first:
            sys.stdout.write(f"\033[{n + 2}A")  # back to the top of the block
        sys.stdout.write("\r\033[J" + header + "\r\n\r\n")
        for i, lab in enumerate(labels):
            box = "◉" if selected[i] else "○"
            row = f"{'›' if i == cursor else ' '} {box} {lab}"
            sys.stdout.write((f"{BOLD}{row}{RESET}" if i == cursor else row) + "\r\n")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        render(True)
        while True:
            key = ui._read_tty_key(fd)
            if key == "up":
                cursor = (cursor - 1) % n
            elif key == "down":
                cursor = (cursor + 1) % n
            elif key == " ":
                selected[cursor] = not selected[cursor]
            elif key in ("a", "A"):
                turn_on = not all(selected)
                selected = [turn_on] * n
            elif key == "enter":
                break
            elif key in ("esc", "ctrl_c", "ctrl_d"):
                selected = [False] * n
                break
            render(False)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()
    chosen = [candidates[i] for i, on in enumerate(selected) if on]
    if not chosen:
        raise SystemExit("synthesize: no transcripts selected")
    return chosen


def _synth_pick(candidates: list[str]) -> list[str]:
    """Pick transcripts interactively — arrow-key checkboxes on a tty, else numbered."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _synth_pick_tty(candidates)
        except (ImportError, OSError):
            pass
    return _synth_pick_numbered(candidates)


def run_synthesize(
    paths: list[str],
    out: str | None = None,
    *,
    interactive: bool = True,
    mode: str = "merge",
) -> None:
    """Fuse agent chat transcripts into one synthesis (normalize → extract → reconcile).

    Inputs may be explicit files, directories to scan, or nothing (defaults to this
    project's Claude Code history). When scanning, an interactive picker lets the
    user choose which transcripts to fuse; --all (interactive=False) takes them all.
    `mode` selects the output: 'merge' (full synthesis), 'diff' (divergences only),
    or 'log' (chronological decision timeline).
    """
    if mode not in SYNTH_MODES:
        raise SystemExit(f"synthesize: unknown mode '{mode}' (merge, diff, or log)")
    file_args = [p for p in paths if pathlib.Path(p).is_file()]
    dir_args = [p for p in paths if pathlib.Path(p).is_dir()]
    for p in paths:
        if p not in file_args and p not in dir_args:
            print(f"{YELLOW}skip (not found): {p}{RESET}")

    if file_args and not dir_args:
        selected = file_args  # user named specific files — honor them as-is
    else:
        search = dir_args
        if not search and not paths:  # only auto-discover when given nothing at all
            default = _claude_project_dir()
            if default:
                search = [str(default)]
                ui.print_system(f"No paths given — listing transcripts in {default}")
        candidates = list(dict.fromkeys(file_args + _discover_transcripts(search)))
        if not candidates:
            raise SystemExit(
                "synthesize: no transcripts found — pass files or a directory, or "
                "run from a project that has Claude Code chat history"
            )
        selected = (
            _synth_pick(candidates)
            if interactive and sys.stdin.isatty()
            else candidates
        )

    if len(selected) == 1:
        print(
            f"{DIM}Only one transcript — degrades to a structured summary "
            f"(fusion needs ≥2).{RESET}"
        )
    if mode == "log":  # a timeline reads oldest → newest
        selected = sorted(selected, key=lambda f: pathlib.Path(f).stat().st_mtime)
    mlx_state = (
        backends.load_model()
        if backends.BACKEND in backends.LOCAL_ML_BACKENDS
        else None
    )
    fact_sets = []
    for p, cid in zip(selected, _chat_ids(selected)):
        chat = _synth_normalize(p, cid)
        ui.print_system(
            f"normalize {chat['id']} ({chat['source']}): {len(chat['turns'])} turns"
        )
        facts = _synth_extract(chat, mlx_state)
        if "_parse_error" in facts:
            print(
                f"{YELLOW}extract   {chat['id']}: the model didn't answer with JSON, "
                f"so this chat contributes nothing{RESET}"
            )
        else:
            ui.print_system(
                f"extract   {chat['id']}: {len(facts['decisions'])} decisions, "
                f"{len(facts['problems_solved'])} problems"
            )
        fact_sets.append(facts)
    ui.print_system(f"reconcile ({mode}): fusing…")
    doc = _synth_reconcile(fact_sets, mode, mlx_state)
    if out:
        pathlib.Path(out).write_text(doc)
        ui.print_system(f"✓ wrote synthesis to {out}")
    else:
        print("\n" + (ui.render_markdown(doc) if sys.stdout.isatty() else doc))
