"""A sandbox for model-written Python, on pydantic-monty.

The `python` tool in wrencode.py runs the model's snippet here: a fresh interpreter
per call, no network, no shell, no environment variables, a read-only view of the
workspace at /workspace, and whatever host functions the tool hands in (read, glob,
grep). pydantic-monty is optional and soft-imported, so `available()` says whether
the tool can be offered at all. This module imports nothing else from wrencode.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import pathlib
import threading
from collections.abc import Callable
from typing import Any

# The module, or None when the extra isn't installed (then the tool isn't offered).
pydantic_monty: Any = None
with contextlib.suppress(ImportError):
    import pydantic_monty

# Where the workspace appears inside the sandbox; also its working directory.
MOUNT_PATH = "/workspace"
# Each run is bounded in time and memory; the model sees a TimeoutError or
# MemoryError and can try again with less.
TIMEOUT = float(os.environ.get("WRENCODE_SANDBOX_TIMEOUT", "30"))
MEMORY_MB = int(os.environ.get("WRENCODE_SANDBOX_MEMORY_MB", "256"))
# Printed output is capped here; run_tool() truncates the tool result further.
MAX_PRINT_BYTES = 1_000_000

_LOCK = threading.Lock()
_POOL: Any = None
_MOUNTS: dict[str, Any] = {}


def available() -> bool:
    """True when pydantic-monty is installed, so the `python` tool can be offered."""
    return pydantic_monty is not None


def _pool() -> Any:
    """The worker pool, started on first use and closed at exit."""
    global _POOL
    with _LOCK:
        if _POOL is None:
            _POOL = pydantic_monty.Monty(min_processes=1).__enter__()
            atexit.register(close)
        return _POOL


def _mount(workspace: pathlib.Path) -> Any:
    """The read-only mount of `workspace`; one per directory, reused across runs."""
    key = str(workspace)
    with _LOCK:
        if key not in _MOUNTS:
            _MOUNTS[key] = pydantic_monty.MountDir(
                host_path=workspace, virtual_path=MOUNT_PATH, mode="read-only"
            )
        return _MOUNTS[key]


def close() -> None:
    """Stop the workers and release the mounts. Safe to call more than once."""
    global _POOL
    with _LOCK:
        if _POOL is not None:
            _POOL.__exit__(None, None, None)
            _POOL = None
        for mount in _MOUNTS.values():
            mount.close()
        _MOUNTS.clear()


def run(
    code: str, *, workspace: pathlib.Path, functions: dict[str, Callable[..., Any]]
) -> str:
    """Run `code` in a fresh sandbox and return what the model should see.

    That is the printed output followed by the value of a trailing expression
    (as a repr), "(no output)" when there is neither, or an "error: ..." text
    with the traceback and whatever was printed before it. `functions` are host
    callables the snippet can call by name; the result of each crosses back as
    a plain value.
    """
    if pydantic_monty is None:
        return "error: the python tool needs pydantic-monty (pip install 'wrencode[sandbox]')"
    printed = pydantic_monty.CollectString(max_bytes=MAX_PRINT_BYTES)
    limits: dict[str, Any] = {
        "max_feed_duration_secs": TIMEOUT,
        "max_memory": MEMORY_MB * 1024 * 1024,
    }
    try:
        with _pool().checkout(script_name="snippet.py", limits=limits) as session:
            value = session.feed_run(
                code,
                external_lookup=dict(functions),
                print_callback=printed,
                mount=_mount(workspace),
                cwd=MOUNT_PATH,
            )
    except pydantic_monty.MontyCrashedError as err:
        return _result(printed.output, error=f"the sandbox stopped: {err}")
    except pydantic_monty.MontyError as err:  # syntax, runtime, conversion
        display = getattr(err, "display", None)
        return _result(
            printed.output, error=display("traceback") if display else str(err)
        )
    return _result(printed.output, value=value)


def _result(printed: str, *, value: Any = None, error: str | None = None) -> str:
    output = printed.rstrip("\n")
    if error is not None:  # "error:" first, so the loop's repeat detection sees it
        text = f"error: {error.strip()}"
        return f"{text}\n--- printed before the error ---\n{output}" if output else text
    if value is not None:
        output = f"{output}\n{value!r}" if output else repr(value)
    return output or "(no output)"
