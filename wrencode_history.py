"""Conversation history in Postgres.

Either a server you point at with WRENCODE_DATABASE_URL, or an embedded PGlite
(Postgres compiled to WebAssembly, run by Node.js) kept under ~/.wrencode/pglite.
Sessions are per workspace. A session holds the current message list exactly as
the backend format needs it, replaced whole on each save: the same snapshot
semantics as the history.json file it supersedes, which stays the fallback when
psycopg (or Node, for the embedded engine) isn't available.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import time
from collections.abc import Generator
from typing import Any

import wrencode_backends as backends
import wrencode_ui as ui

# The Postgres client, or None when the [history] extra isn't installed.
psycopg: Any = None
with contextlib.suppress(ImportError):
    import psycopg

DATABASE_URL = os.environ.get("WRENCODE_DATABASE_URL", "")
# PGlite and its wire-protocol server, pinned; installed once into ~/.wrencode/pglite.
PGLITE_PACKAGES = {
    "@electric-sql/pglite": "0.3.3",
    "@electric-sql/pglite-socket": "0.0.8",
}
PGLITE_START_TIMEOUT = float(os.environ.get("WRENCODE_PGLITE_START_TIMEOUT", "60"))
# Why open_store() returned None, for the startup note when Postgres was intended.
UNAVAILABLE_REASON = ""

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         BIGSERIAL PRIMARY KEY,
    workspace  TEXT NOT NULL,
    backend    TEXT NOT NULL,
    model      TEXT NOT NULL,
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_workspace_idx ON sessions (workspace, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id         BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    message    JSONB NOT NULL,
    text       TEXT NOT NULL,
    UNIQUE (session_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_search_idx
    ON messages USING GIN (to_tsvector('simple', text));
"""

SERVER_JS = r"""
// PGlite behind a Postgres wire-protocol socket, written by wrencode_history.py.
const { PGlite } = require('@electric-sql/pglite');
const { PGLiteSocketServer } = require('@electric-sql/pglite-socket');
const fs = require('fs');
const [dataDir, socketPath] = process.argv.slice(2);

async function main() {
  const db = new PGlite({ dataDir });
  await db.waitReady;
  if (fs.existsSync(socketPath)) fs.unlinkSync(socketPath);
  const server = new PGLiteSocketServer({ db, path: socketPath });
  await server.start();
  fs.chmodSync(socketPath, 0o600);
  console.log('ready');
  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    setTimeout(() => process.exit(0), 2000).unref(); // never outlive the parent
    try { await server.stop(); await db.close(); } finally { process.exit(0); }
  };
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
  // wrencode holds our stdin; when it exits, for any reason, so do we.
  process.stdin.resume();
  process.stdin.on('end', stop);
  process.stdin.on('close', stop);
}

main().catch((err) => { console.error(err); process.exit(1); });
"""


def available() -> bool:
    """True when psycopg is installed, so Postgres history can be attempted."""
    return psycopg is not None


def _connectable(path: pathlib.Path) -> bool:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(0.5)
    try:
        s.connect(str(path))
    except OSError:
        return False
    else:
        return True
    finally:
        s.close()


class EmbeddedPGlite:
    """PGlite serving a persistent data directory over a Unix socket, started on demand.

    The socket lives in an owner-only directory, named the way libpq expects, so the
    DSN is just the directory. If another wrencode already serves the database, its
    socket is tried instead of starting a second server; PGlite is single-user, so
    that connection only succeeds once the first wrencode has let go.
    """

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = root if root is not None else backends.CONFIG_DIR / "pglite"
        self.data = self.root / "data"
        self.run = self.root / "run"
        self.socket = self.run / ".s.PGSQL.5432"
        self.proc: subprocess.Popen[bytes] | None = None

    def dsn(self) -> str:
        return f"host={self.run} dbname=postgres user=postgres password=postgres"

    def start(self) -> str:
        """Make the database reachable and return its DSN."""
        if _connectable(self.socket):
            return self.dsn()
        node = shutil.which("node")
        if not node:
            raise RuntimeError(
                "Node.js is needed for the embedded Postgres (PGlite); install it, or "
                "set WRENCODE_DATABASE_URL to use a Postgres server"
            )
        self._install()
        for d in (self.data, self.run):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
        with contextlib.suppress(OSError):
            self.socket.unlink()  # a stale socket from a server that is gone
        self.proc = subprocess.Popen(
            [
                node,
                str(self.root / "pglite_server.js"),
                str(self.data),
                str(self.socket),
            ],
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        atexit.register(self.stop)
        deadline = time.monotonic() + PGLITE_START_TIMEOUT
        while time.monotonic() < deadline:
            if _connectable(self.socket):
                return self.dsn()
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() if self.proc.stderr else b"").decode()
                raise RuntimeError(
                    f"PGlite exited: {err.strip()[-400:] or self.proc.returncode}"
                )
            time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"PGlite did not start within {PGLITE_START_TIMEOUT:.0f}s")

    def _install(self) -> None:
        """Write the pinned package.json and server script; npm install on first use."""
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        (self.root / "package.json").write_text(
            json.dumps(
                {
                    "name": "wrencode-pglite",
                    "private": True,
                    "dependencies": PGLITE_PACKAGES,
                },
                indent=2,
            )
        )
        (self.root / "pglite_server.js").write_text(SERVER_JS.lstrip())
        if all((self.root / "node_modules" / p).is_dir() for p in PGLITE_PACKAGES):
            return
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm is needed to install PGlite the first time")
        ui.print_system(
            f"Installing PGlite (Postgres in WebAssembly) into {self.root}, once…"
        )
        result = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund", "--loglevel=error"],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"npm install failed: {result.stderr.strip()[-400:]}")

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            if proc.stdin:
                proc.stdin.close()  # the server exits when its stdin closes
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        if proc.stderr:
            proc.stderr.close()


class Store:
    """Sessions and their messages in Postgres, over one connection kept for the store's life.

    PGlite runs a single backend and only its first connection is fully set up (later
    ones get no parameter status, which crashes psycopg's C loaders), so the store
    holds one connection and reconnects only when it drops. A second wrencode that
    can't get the connection within a few seconds falls back to history.json.
    """

    def __init__(self, dsn: str, server: EmbeddedPGlite | None = None) -> None:
        self.dsn = dsn
        self.server = server
        self._connection: Any = None

    def _connect(self) -> Any:
        if self._connection is None or self._connection.closed:
            # No server-side prepared statements: PGlite keeps them across connections,
            # so one prepared before a reconnect would collide afterwards.
            self._connection = psycopg.connect(
                self.dsn, prepare_threshold=None, connect_timeout=5
            )
        return self._connection

    @contextlib.contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except (
            psycopg.OperationalError
        ):  # the connection is gone; the next call reconnects
            self._connection = None
            raise
        except BaseException:
            conn.rollback()
            raise

    @staticmethod
    def _run(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
        return conn.execute(sql, params)

    def init_schema(self) -> None:
        with self._conn() as conn:
            for statement in SCHEMA.split(";"):  # one at a time: see _run
                if statement.strip():
                    self._run(conn, statement)

    def latest_session(self, workspace: str) -> int | None:
        with self._conn() as conn:
            row = self._run(
                conn,
                "SELECT id FROM sessions WHERE workspace = %s ORDER BY updated_at DESC LIMIT 1",
                (workspace,),
            ).fetchone()
        return int(row[0]) if row else None

    def new_session(self, workspace: str, backend: str, model: str) -> int:
        with self._conn() as conn:
            row = self._run(
                conn,
                "INSERT INTO sessions (workspace, backend, model) VALUES (%s, %s, %s) RETURNING id",
                (workspace, backend, model),
            ).fetchone()
        return int(row[0])

    def load(self, session_id: int) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = self._run(
                conn,
                "SELECT message FROM messages WHERE session_id = %s ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [dict(r[0]) for r in rows]

    def save(self, session_id: int, messages: list[dict[str, Any]]) -> None:
        """Replace the session's messages with `messages` (the loop's whole list)."""
        rows = [
            (
                session_id,
                i,
                str(m.get("role", "")),
                json.dumps(m),
                backends.flatten_content(m.get("content") or ""),
            )
            for i, m in enumerate(messages)
        ]
        title = next(
            (text for _, _, role, _, text in rows if role == "user" and text.strip()),
            "",
        )
        with self._conn() as conn:
            self._run(conn, "DELETE FROM messages WHERE session_id = %s", (session_id,))
            for row in rows:
                self._run(
                    conn,
                    "INSERT INTO messages (session_id, seq, role, message, text) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s)",
                    row,
                )
            self._run(
                conn,
                "UPDATE sessions SET updated_at = now(), title = COALESCE(title, NULLIF(%s, '')) WHERE id = %s",
                (" ".join(title.split())[:80], session_id),
            )

    def sessions(self, workspace: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = self._run(
                conn,
                """
                SELECT s.id, s.title, s.model, s.updated_at,
                       (SELECT count(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'user')
                FROM sessions s WHERE s.workspace = %s ORDER BY s.updated_at DESC LIMIT %s
                """,
                (workspace, limit),
            ).fetchall()
        return [
            {
                "id": int(r[0]),
                "title": r[1] or "",
                "model": r[2],
                "updated_at": r[3],
                "chats": int(r[4]),
            }
            for r in rows
        ]

    def search(
        self, workspace: str, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Messages in this workspace's sessions matching `query` (full-text, newest first)."""
        with self._conn() as conn:
            rows = self._run(
                conn,
                """
                SELECT m.session_id, m.role, left(m.text, 160)
                FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE s.workspace = %s
                  AND to_tsvector('simple', m.text) @@ plainto_tsquery('simple', %s)
                ORDER BY m.id DESC LIMIT %s
                """,
                (workspace, query, limit),
            ).fetchall()
        return [{"session_id": int(r[0]), "role": r[1], "text": r[2]} for r in rows]

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(Exception):
                self._connection.close()
            self._connection = None
        if self.server is not None:
            self.server.stop()


def open_store() -> Store | None:
    """The history store, or None when history.json should be used instead.

    None without psycopg (the default install). With psycopg but no reachable
    database, UNAVAILABLE_REASON says why, for a note at startup.
    """
    global UNAVAILABLE_REASON
    UNAVAILABLE_REASON = ""
    if psycopg is None:
        return None
    server: EmbeddedPGlite | None = None
    try:
        if DATABASE_URL:
            store = Store(DATABASE_URL)
        else:
            server = EmbeddedPGlite()
            store = Store(server.start(), server)
        store.init_schema()
    except Exception as err:  # fall back to the JSON file, but say why
        if server is not None:
            server.stop()
        UNAVAILABLE_REASON = str(err)
        return None
    return store
