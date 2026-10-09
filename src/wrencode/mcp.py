"""MCP client: tools from Model Context Protocol servers join the tool list.

Servers are declared in `.wrencode/mcp.json` in the project (or Claude Code's
`.mcp.json`, same format) and `mcp.json` in the config directory:

    {"mcpServers": {
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                   "env": {"GITHUB_TOKEN": "..."}},
        "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ..."}}
    }}

A `command` entry is run as a subprocess and spoken to over stdio (JSON-RPC,
one message per line); a `url` entry is Streamable HTTP (JSON-RPC POSTs whose
replies may be JSON or server-sent events). Each server's tools are offered to
the model as `mcp__<server>__<tool>`; calling one asks for approval unless the
server marks it read-only, and permission rules `mcp(server:tool)` apply.

A project's file is part of the repository and its servers run commands on
this machine, so, like permission rules, they are shown once and only start
after they are accepted; their hash is kept under `trusted` in the user file.
A command is looked up on wrencode's own PATH (a server's `env` can't redirect
it), the backend API keys and WRENCODE_* settings are kept out of a server's
environment unless its `env` passes them, and a project server's `env` can't
set the loader variables (LD_PRELOAD, NODE_OPTIONS, PYTHONPATH, ...) that
would run code before the command does. HTTP servers are not followed across
redirects, so an Authorization header only ever reaches the URL configured.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import __version__

PROTOCOL_VERSION = "2025-06-18"
PROJECT_FILES = (pathlib.Path(".wrencode") / "mcp.json", pathlib.Path(".mcp.json"))
CONNECT_TIMEOUT = float(os.environ.get("WRENCODE_MCP_CONNECT_TIMEOUT", "20"))
CALL_TIMEOUT = float(os.environ.get("WRENCODE_MCP_TIMEOUT", "120"))
_NAME_OK = re.compile(r"[^A-Za-z0-9_-]")
# Kept out of a server's environment unless its own `env` sets them.
HIDDEN_ENV = re.compile(r"^(?:[A-Z0-9_]*_API_KEY|WRENCODE_[A-Z0-9_]*)$")
# What a project's server may not set: they run code before the command does.
LOADER_ENV = frozenset(
    {
        "PATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "NODE_OPTIONS",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PERL5OPT",
        "RUBYOPT",
        "BASH_ENV",
        "ENV",
    }
)


def tool_name(server: str, tool: str) -> str:
    """`mcp__<server>__<tool>`, in the characters the model APIs accept."""
    s = _NAME_OK.sub("_", server)
    t = _NAME_OK.sub("_", tool)
    return f"mcp__{s}__{t}"[:64]


class MCPError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Transports: both speak JSON-RPC 2.0, request() blocks for the matching reply.
# ---------------------------------------------------------------------------
class _Stdio:
    """A server started as a subprocess; messages are lines on its stdin/stdout."""

    def __init__(
        self, command: str, args: list[str], env: dict[str, str], cwd: str
    ) -> None:
        exe = shutil.which(command) if os.sep not in command else command
        if not exe:
            raise MCPError(f"command not found: {command}")
        inherited = {k: v for k, v in os.environ.items() if not HIDDEN_ENV.match(k)}
        self.proc = subprocess.Popen(
            [exe, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**inherited, **env},
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, list[Any]]] = {}
        self._next = 1
        self.notifications: list[dict[str, Any]] = []
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                with self._lock:
                    slot = (
                        self._pending.pop(int(msg["id"]), None)
                        if str(msg["id"]).isdigit()
                        else None
                    )
                if slot is not None:
                    slot[1].append(msg)
                    slot[0].set()
            elif "id" in msg and "method" in msg:  # a request from the server
                self._answer(msg)
            else:
                self.notifications.append(msg)
        # the server is gone: wake every waiter with an error
        with self._lock:
            for event, box in self._pending.values():
                box.append({"error": {"message": "server exited"}})
                event.set()
            self._pending.clear()

    def _answer(self, msg: dict[str, Any]) -> None:
        """Keep the session alive: ping gets an empty result, anything else
        (sampling, roots) a method-not-found error."""
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"]}
        if msg["method"] == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {
                "code": -32601,
                "message": f"{msg['method']} not supported",
            }
        with contextlib.suppress(MCPError):
            self._send(reply)

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as err:
            raise MCPError(f"server not reachable: {err}") from err

    def request(
        self, method: str, params: dict[str, Any] | None, timeout: float
    ) -> Any:
        event, box = threading.Event(), []
        with self._lock:
            rid = self._next
            self._next += 1
            self._pending[rid] = (event, box)
        self._send(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        )
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise MCPError(f"{method} timed out after {timeout:.0f}s")
        reply = box[0]
        if "error" in reply:
            err = reply["error"]
            raise MCPError(
                str(err.get("message", err)) if isinstance(err, dict) else str(err)
            )
        return reply.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            if self.proc.stdin:
                self.proc.stdin.close()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the Authorization header somewhere else: refuse it."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class _Http:
    """Streamable HTTP: JSON-RPC over POST; a reply is JSON or an SSE stream."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self.session: str | None = None
        self._next = 1
        self._lock = threading.Lock()

    def _post(
        self, msg: dict[str, Any], timeout: float
    ) -> tuple[int, str, bytes, dict[str, str]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(
            self.url, data=json.dumps(msg).encode(), headers=headers
        )
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return (
                    resp.status,
                    resp.headers.get("Content-Type", ""),
                    resp.read(),
                    dict(resp.headers),
                )
        except urllib.error.HTTPError as err:
            if 300 <= err.code < 400:
                raise MCPError(
                    f"{self.url} redirects to {err.headers.get('Location', '?')}; "
                    "configure that URL instead (redirects are not followed)"
                ) from err
            body = err.read().decode(errors="replace")[:300]
            raise MCPError(f"HTTP {err.code} from {self.url}: {body}") from err
        except (urllib.error.URLError, OSError) as err:
            raise MCPError(
                f"could not reach {self.url}: {getattr(err, 'reason', err)}"
            ) from err

    def request(
        self, method: str, params: dict[str, Any] | None, timeout: float
    ) -> Any:
        with self._lock:
            rid = self._next
            self._next += 1
        status, ctype, body, headers = self._post(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}},
            timeout,
        )
        for k, v in headers.items():
            if k.lower() == "mcp-session-id":
                self.session = v
        reply: dict[str, Any] | None = None
        if "text/event-stream" in ctype:
            for line in body.decode("utf-8", errors="replace").splitlines():
                if line.startswith("data:"):
                    with contextlib.suppress(ValueError):
                        msg = json.loads(line[5:].strip())
                        if isinstance(msg, dict) and msg.get("id") == rid:
                            reply = msg
        else:
            with contextlib.suppress(ValueError):
                parsed = json.loads(body or b"null")
                if isinstance(parsed, dict):
                    reply = parsed
        if reply is None:
            raise MCPError(f"no reply to {method} (HTTP {status})")
        if "error" in reply:
            err = reply["error"]
            raise MCPError(
                str(err.get("message", err)) if isinstance(err, dict) else str(err)
            )
        return reply.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        with contextlib.suppress(MCPError):
            self._post({"jsonrpc": "2.0", "method": method, "params": params or {}}, 10)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# A server and its tools
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    server: str
    name: str
    description: str
    schema: dict[str, Any]
    read_only: bool = False

    @property
    def full_name(self) -> str:
        return tool_name(self.server, self.name)


@dataclass
class Server:
    name: str
    spec: dict[str, Any]
    source: str  # "user" or "project"
    transport: Any = None
    tools: list[Tool] = field(default_factory=list)
    error: str = ""
    info: str = ""

    def connect(self, cwd: str) -> None:
        """Start or reach the server, then fetch its tools; errors are kept, not raised."""
        try:
            if self.spec.get("url"):
                self.transport = _Http(
                    str(self.spec["url"]), dict(self.spec.get("headers") or {})
                )
            elif self.spec.get("command"):
                env = {str(k): str(v) for k, v in (self.spec.get("env") or {}).items()}
                if self.source == "project":
                    env = {k: v for k, v in env.items() if k not in LOADER_ENV}
                self.transport = _Stdio(
                    str(self.spec["command"]),
                    [str(a) for a in self.spec.get("args") or []],
                    env,
                    cwd,
                )
            else:
                raise MCPError("needs a command or a url")
            result = self.transport.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "wrencode", "version": __version__},
                },
                CONNECT_TIMEOUT,
            )
            info = (result or {}).get("serverInfo") or {}
            self.info = f"{info.get('name', '')} {info.get('version', '')}".strip()
            self.transport.notify("notifications/initialized")
            self.tools = self._list_tools()
        except (MCPError, OSError, ValueError) as err:
            self.error = str(err)
            self.close()

    def _list_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        cursor = None
        for _ in range(50):
            params = {"cursor": cursor} if cursor else {}
            result = self.transport.request("tools/list", params, CONNECT_TIMEOUT) or {}
            for t in result.get("tools") or []:
                if not isinstance(t, dict) or not t.get("name"):
                    continue
                schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    schema = {"type": "object", "properties": {}}
                notes = t.get("annotations") or {}
                tools.append(
                    Tool(
                        self.name,
                        str(t["name"]),
                        str(t.get("description") or ""),
                        schema,
                        bool(isinstance(notes, dict) and notes.get("readOnlyHint")),
                    )
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call(self, tool: str, arguments: dict[str, Any]) -> str:
        """Call a tool and flatten its content to text; an error result is 'error: ...'."""
        if self.transport is None:
            return f"error: MCP server {self.name} is not connected ({self.error})"
        try:
            result = (
                self.transport.request(
                    "tools/call", {"name": tool, "arguments": arguments}, CALL_TIMEOUT
                )
                or {}
            )
        except MCPError as err:
            return f"error: {err}"
        parts: list[str] = []
        for block in result.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(str(block.get("text", "")))
            elif kind == "image":
                parts.append(f"[image {block.get('mimeType', '')}]")
            elif kind == "resource":
                res = block.get("resource") or {}
                parts.append(str(res.get("text") or f"[resource {res.get('uri', '')}]"))
            else:
                parts.append(json.dumps(block)[:500])
        if "structuredContent" in result and not parts:
            parts.append(json.dumps(result["structuredContent"]))
        text = "\n".join(p for p in parts if p).strip() or "(empty)"
        return f"error: {text}" if result.get("isError") else text

    def close(self) -> None:
        if self.transport is not None:
            with contextlib.suppress(Exception):
                self.transport.close()
            self.transport = None


# ---------------------------------------------------------------------------
# Configuration and the set of servers in use
# ---------------------------------------------------------------------------
def _parse(raw: bytes | str) -> dict[str, Any]:
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return {}
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    return servers if isinstance(servers, dict) else {}


def _read(path: pathlib.Path) -> dict[str, Any]:
    try:
        return _parse(path.read_bytes())
    except OSError:
        return {}


class Registry:
    """The servers from both files, connected on demand, with project trust."""

    def __init__(self, user_file: pathlib.Path, project_root: pathlib.Path) -> None:
        self.user_file = user_file
        self.project_root = project_root
        self.servers: list[Server] = []
        self.project_file = next(
            (project_root / p for p in PROJECT_FILES if (project_root / p).is_file()),
            None,
        )
        self._user_data: dict[str, Any] = {}
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(user_file.read_text())
            if isinstance(loaded, dict):
                self._user_data = loaded
        self._project_bytes = b""
        if self.project_file is not None:
            with contextlib.suppress(OSError):
                self._project_bytes = self.project_file.read_bytes()
        self.project_servers = _parse(self._project_bytes)
        atexit.register(self.close)

    def _project_hash(self) -> str:
        """The hash of the file as it was read: what was shown is what is trusted."""
        if not self._project_bytes:
            return ""
        return hashlib.sha256(self._project_bytes).hexdigest()

    def project_trusted(self) -> bool:
        if not self.project_servers:
            return True
        trusted = self._user_data.get("trusted") or {}
        return trusted.get(str(self.project_root)) == self._project_hash()

    def trust_project(self) -> None:
        trusted = dict(self._user_data.get("trusted") or {})
        trusted[str(self.project_root)] = self._project_hash()
        self._user_data["trusted"] = trusted
        self.user_file.parent.mkdir(parents=True, exist_ok=True)
        self.user_file.write_text(json.dumps(self._user_data, indent=2) + "\n")
        os.chmod(self.user_file, 0o600)

    def connect_all(self) -> list[Server]:
        """Connect every configured server (project ones only when trusted), in parallel."""
        self.close()
        specs: list[tuple[str, dict[str, Any], str]] = [
            (name, spec, "user")
            for name, spec in (_read(self.user_file)).items()
            if isinstance(spec, dict)
        ]
        if self.project_trusted():
            specs += [
                (name, spec, "project")
                for name, spec in self.project_servers.items()
                if isinstance(spec, dict) and name not in {s[0] for s in specs}
            ]
        self.servers = [Server(name, spec, source) for name, spec, source in specs]
        threads = [
            threading.Thread(
                target=s.connect, args=(str(self.project_root),), daemon=True
            )
            for s in self.servers
        ]
        for t in threads:
            t.start()
        deadline = time.monotonic() + CONNECT_TIMEOUT + 5
        for t in threads:
            t.join(max(deadline - time.monotonic(), 0))
        for s, t in zip(self.servers, threads, strict=True):
            if t.is_alive():  # still connecting or listing tools: not usable
                s.error = s.error or "did not answer in time"
                s.tools = []
                s.close()
            elif s.transport is None and not s.error:
                s.error = "did not answer in time"
        return self.servers

    def tools(self) -> list[Tool]:
        return [t for s in self.servers for t in s.tools]

    def server(self, name: str) -> Server | None:
        return next((s for s in self.servers if s.name == name), None)

    def close(self) -> None:
        for s in self.servers:
            s.close()


# The registry in use, set by wrencode at startup (None: no MCP servers).
ACTIVE: Registry | None = None
