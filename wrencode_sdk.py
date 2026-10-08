"""The claude-agent-sdk backend: Claude Code's own agent loop driven through the
Claude Agent SDK, with wrencode's approvals, Escape-to-interrupt and cost display.
"""

import contextlib
import json
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import wrencode_backends as backends
import wrencode_ui as ui
from wrencode_ui import RESET, DIM, GREEN, YELLOW, RED


def _agent_sdk_sessions_file() -> pathlib.Path:
    return backends.CONFIG_DIR / "agent_sdk_sessions.json"


def _load_agent_sdk_session_id(cwd: pathlib.Path) -> str:
    """The last SDK session id saved for the workspace `cwd`, or ''."""
    with contextlib.suppress(Exception):
        data = json.loads(_agent_sdk_sessions_file().read_text())
        return str(data.get(str(cwd), ""))
    return ""


def _save_agent_sdk_session_id(cwd: pathlib.Path, session_id: str) -> None:
    path = _agent_sdk_sessions_file()
    data: dict[str, str] = {}
    with contextlib.suppress(Exception):
        data = dict(json.loads(path.read_text()))
    key = str(cwd)
    if session_id:
        data[key] = session_id
    else:
        data.pop(key, None)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        os.chmod(path, 0o600)


def _agent_sdk_env() -> dict[str, str]:
    """Environment for the SDK's Claude Code process: API key auth only."""
    env = {
        "ANTHROPIC_API_KEY": backends.API_KEY,
        # Blank the subscription credentials so they can't take over billing.
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_AUTH_TOKEN": "",
    }
    if backends.ANTHROPIC_WORKSPACE_ID:
        header = f"anthropic-workspace-id: {backends.ANTHROPIC_WORKSPACE_ID}"
        extra = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").strip()
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"{extra}\n{header}" if extra else header
    return env


def _short(text: Any, limit: int = 80) -> str:
    flat = str(text).replace("\n", "\\n")
    return flat[:limit] + ("..." if len(flat) > limit else "")


def format_sdk_tool_action(name: str, inp: dict[str, Any]) -> str:
    """Human-readable summary of a Claude Code tool call."""
    if name == "Bash":
        return f"$ {str(inp.get('command', '')).strip()}"
    if name == "Edit":
        return (
            f"Edit {inp.get('file_path', '?')}\n"
            f"- {_short(inp.get('old_string', ''))}\n"
            f"+ {_short(inp.get('new_string', ''))}"
        )
    if name == "Write":
        content = str(inp.get("content", ""))
        lines = content.count("\n") + (1 if content else 0)
        return f"Write {inp.get('file_path', '?')}  ({lines} lines)\n{_short(content, 160)}"
    for field in ("file_path", "notebook_path", "path", "pattern", "url", "query"):
        if inp.get(field):
            return f"{name} {inp[field]}"
    if inp.get("description"):
        return f"{name}: {_short(inp['description'], 160)}"
    return f"{name}({json.dumps(inp, ensure_ascii=False)[:160]})"


def _sdk_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


class _Spinner:
    """The thinking loader, startable and stoppable between streamed messages."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not sys.stdout.isatty() or ui.HEADLESS:
            return
        self._stop.clear()
        context = ui.loader_context(backends.BACKEND, backends.MODEL)

        def animate() -> None:
            step = 0
            while not self._stop.is_set():
                sys.stdout.write(f"\r{ui.loader_display(step, context)}")
                sys.stdout.flush()
                step += 1
                time.sleep(0.07)

        self._thread = threading.Thread(target=animate, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.4)
        self._thread = None
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


@dataclass
class AgentSDKTurn:
    """Outcome of one prompt sent through the Agent SDK."""

    text: str = ""
    is_error: bool = False
    error: str = ""
    cost_usd: float = 0.0
    num_turns: int = 0


class AgentSDKSession:
    """One Claude Code session driven through the Agent SDK, reused across prompts."""

    def __init__(
        self,
        *,
        cwd: pathlib.Path,
        instructions: str = "",
        persist: bool = True,
        max_turns: int = 0,
    ) -> None:
        """`cwd` is the workspace; `instructions` (AGENTS.md text) extend Claude Code's
        system prompt. With `persist`, the session resumes per workspace across runs."""
        import asyncio

        self._loop = asyncio.new_event_loop()
        self._client: Any = None
        self._model = ""
        self._cwd = cwd
        self._instructions = instructions
        self._persist = persist
        self._max_turns = max_turns
        self.session_id = _load_agent_sdk_session_id(cwd) if persist else ""
        self._reported_cost = 0.0  # the SDK's cost total for the current client
        self._spinner = _Spinner()
        self._esc = ui._EscWatch()

    def _options(self, resume: str) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk.types import SystemPromptPreset

        system_prompt = SystemPromptPreset(type="preset", preset="claude_code")
        if extra := self._instructions.strip():
            system_prompt["append"] = extra
        return ClaudeAgentOptions(
            model=backends.MODEL or None,
            cwd=str(self._cwd),
            system_prompt=system_prompt,
            # Isolated from ~/.claude hooks, plugins and MCP servers; project
            # instructions come from AGENTS.md/CLAUDE.md via the append above.
            setting_sources=[],
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            env=_agent_sdk_env(),
            resume=resume or None,
            max_turns=self._max_turns or None,
            effort=backends.CLAUDE_EFFORT,
            stderr=self._on_stderr,
        )

    @staticmethod
    def _on_stderr(line: str) -> None:
        if os.environ.get("WRENCODE_DEBUG"):
            print(f"{DIM}[claude] {line.rstrip()}{RESET}", file=sys.stderr)

    async def _ensure_client(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        if self._client is not None:
            if self._model != backends.MODEL:
                await self._client.set_model(backends.MODEL or None)
                self._model = backends.MODEL
            return
        client = ClaudeSDKClient(self._options(self.session_id))
        try:
            await client.connect()
        except Exception:
            if not self.session_id:
                raise
            # A saved session that no longer exists: start a fresh one.
            self.session_id = ""
            client = ClaudeSDKClient(self._options(""))
            await client.connect()
        self._client = client
        self._model = backends.MODEL
        self._reported_cost = 0.0

    async def _disconnect(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.disconnect()
            self._client = None

    def reset(self) -> None:
        """Forget the conversation: the next prompt starts a new session."""
        self._loop.run_until_complete(self._disconnect())
        self.session_id = ""
        if self._persist:
            _save_agent_sdk_session_id(self._cwd, "")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._loop.run_until_complete(self._disconnect())
        self._loop.close()

    async def _can_use_tool(self, name: str, inp: dict[str, Any], context: Any) -> Any:
        import asyncio

        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        self._spinner.stop()
        self._esc.stop()  # confirm() reads stdin; the Escape listener must let go
        try:
            body = format_sdk_tool_action(name, inp)
            first, _, rest = body.partition("\n")
            print(f"{YELLOW}?{RESET} {first}")
            for line in rest.split("\n"):
                if line.strip():
                    print(f"{DIM}  {line}{RESET}")
            verdict = await asyncio.to_thread(ui.confirm, first)
        finally:
            self._esc.start()
        if verdict == "ok":
            return PermissionResultAllow(updated_input=inp)
        return PermissionResultDeny(message=verdict)

    def run(self, prompt: str) -> AgentSDKTurn:
        import asyncio

        ui._CANCEL_REQUESTED.clear()
        try:
            return self._loop.run_until_complete(self._run(prompt))
        except KeyboardInterrupt:
            # The turn is abandoned mid-stream. Drop the client so leftover
            # messages can't leak into the next prompt; the session id stays,
            # so the next prompt resumes the same conversation.
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            with contextlib.suppress(BaseException):
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            with contextlib.suppress(BaseException):
                self._loop.run_until_complete(self._disconnect())
            raise
        finally:
            self._spinner.stop()
            self._esc.stop()

    async def _watch_cancel(self) -> None:
        import asyncio

        while not ui._CANCEL_REQUESTED.is_set():
            await asyncio.sleep(0.1)
        self._spinner.stop()
        print(f"{YELLOW}Interrupted{RESET}")
        with contextlib.suppress(Exception):
            await self._client.interrupt()

    async def _run(self, prompt: str) -> AgentSDKTurn:
        import asyncio

        await self._ensure_client()
        turn = AgentSDKTurn()
        await self._client.query(prompt)
        self._esc.start()
        self._spinner.start()
        watcher = asyncio.ensure_future(self._watch_cancel())
        try:
            # receive_response() ends on its own at the ResultMessage; the SDK
            # docs advise against break here (asyncio cleanup issues).
            async for msg in self._client.receive_response():
                self._spinner.stop()
                if not self._handle_message(msg, turn):
                    self._spinner.start()
        finally:
            watcher.cancel()
            self._spinner.stop()
            self._esc.stop()
        return turn

    def _handle_message(self, msg: Any, turn: AgentSDKTurn) -> bool:
        """Print one streamed message; return True once the turn's result arrived."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        if isinstance(msg, SystemMessage):
            data = msg.data or {}
            if msg.subtype == "init":
                source = data.get("apiKeySource")
                if source and source != "ANTHROPIC_API_KEY":
                    print(
                        f"{YELLOW}Warning: the Agent SDK is authenticating with "
                        f"{source}, not ANTHROPIC_API_KEY, so usage may not bill "
                        f"to your API credits.{RESET}"
                    )
                self._remember_session(str(data.get("session_id", "")))
            elif msg.subtype == "compact_boundary":
                ui.print_system("Compacted conversation")
            return False
        if isinstance(msg, AssistantMessage):
            nested = bool(msg.parent_tool_use_id)  # a subagent working
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip() and not nested:
                    turn.text = block.text
                    if not msg.error:  # an API error is reported with the result
                        ui.print_agent_message(block.text)
                elif isinstance(block, ToolUseBlock):
                    first = format_sdk_tool_action(block.name, block.input).split("\n")[
                        0
                    ]
                    mark = f"{DIM}↳" if nested else f"{GREEN}⏺"  # ↳ = subagent
                    print(f"{mark}{RESET}{DIM} {first}{RESET}")
            if msg.error:
                turn.error = turn.text or str(msg.error)
            return False
        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    lines = _sdk_result_text(block.content).strip().split("\n")
                    color = RED if block.is_error else DIM
                    head = lines[0][:200] if lines and lines[0] else "(empty)"
                    more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
                    print(f"{color}  ⎿ {head}{more}{RESET}")
            return False
        if isinstance(msg, ResultMessage):
            self._remember_session(msg.session_id)
            total = float(msg.total_cost_usd or 0.0)
            turn.cost_usd = max(total - self._reported_cost, 0.0)
            self._reported_cost = total
            turn.num_turns = msg.num_turns
            turn.is_error = bool(msg.is_error)
            if msg.result and not turn.text:
                turn.text = msg.result
            if turn.is_error:
                turn.error = msg.result or turn.error or msg.subtype
                print(f"{RED}Error: {turn.error}{RESET}")
                if "workspace" in turn.error.lower():
                    print(
                        f"{YELLOW}Your API key spans several workspaces. Run "
                        f"/configure and enter the workspace id from the Console "
                        f"(Settings, Workspaces), or set ANTHROPIC_WORKSPACE_ID.{RESET}"
                    )
            print(
                f"{DIM}${turn.cost_usd:.4f} this turn · ${total:.4f} this session{RESET}"
            )
            return True
        return False

    def _remember_session(self, session_id: str) -> None:
        if session_id and session_id != self.session_id:
            self.session_id = session_id
            if self._persist:
                _save_agent_sdk_session_id(self._cwd, session_id)
