"""Model backends for wrencode: backend tables and state, HTTP, request/response formats.

get_response() sends one turn to the configured backend and returns its raw reply;
_summarize() is the no-tools one-shot used for compaction. apply_backend() sets the
module-level state (BACKEND, MODEL, API_KEY, ...) that the rest of the file reads.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import hmac
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

import wrencode_ui as ui
from wrencode_ui import DIM, RED, RESET, YELLOW

# Per-backend defaults. "kind" controls how a backend is treated:
#   api         - hosted HTTP API, needs an API key
#   local-proxy - Anthropic-compatible server already running on localhost
#   local-ml    - in-process model weights (mlx / transformers); source install only,
#                 since the standalone binary can't bundle the ML stack
#   agent-sdk   - Claude Agent SDK: Claude Code's own agent loop and tools, billed
#                 to an Anthropic API key; source install with claude-agent-sdk
BACKEND_SPECS: dict[str, dict[str, str]] = {
    "anthropic": {
        "kind": "api",
        "model": "claude-opus-5-5",
        "key_env": "ANTHROPIC_API_KEY",
        "api_base": "https://api.anthropic.com/v1/messages",
        "label": "Anthropic Claude (API key)",
    },
    "claude-agent-sdk": {
        "kind": "agent-sdk",
        "model": "claude-opus-5-5",
        "key_env": "ANTHROPIC_API_KEY",
        # One-shot helpers (/compact summaries, synthesize) use the Messages API.
        "api_base": "https://api.anthropic.com/v1/messages",
        "label": "Claude Agent SDK — Claude Code's agent loop (API key, source install)",
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
    "nanogpt": {
        "kind": "api",
        "model": "z-ai/glm-5.3-flash",
        "key_env": "NANOGPT_API_KEY",
        "api_base": "https://nano-gpt.com/api/v1/chat/completions",
        "label": "NanoGPT — any model (API key)",
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
    "openai-compatible": {
        # Any server speaking OpenAI chat completions: vLLM, llama.cpp's
        # llama-server, Hugging Face Inference Providers, LM Studio, etc.
        # URL from OPENAI_COMPATIBLE_BASE_URL; an empty model means "use the
        # server's only model" (resolved in load_model).
        "kind": "local-proxy",
        "model": "",
        "key_env": "OPENAI_COMPATIBLE_API_KEY",
        "label": "OpenAI-compatible server (vLLM, llama.cpp, Hugging Face, ...)",
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

# Offline/fallback model lists used only when a live /models fetch fails.
# Anthropic, OpenAI, OpenRouter, NanoGPT, and Ollama are fetched from the
# provider when a key (or local server) is available.
BACKEND_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    ],
    "openai": [
        "gpt-4o-mini",
    ],
    "openrouter": [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.1-8b-instruct",
    ],
    "nanogpt": [
        "z-ai/glm-5.3-flash",
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash-uncensored",
        "z-ai/glm-5.3-uncensored",
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
API_BACKENDS: frozenset[str] = frozenset(
    name for name, spec in BACKEND_SPECS.items() if spec["kind"] == "api"
)
LOCAL_ML_BACKENDS: frozenset[str] = frozenset(
    name for name, spec in BACKEND_SPECS.items() if spec["kind"] == "local-ml"
)
# Backends whose responses use the Anthropic Messages format (content blocks,
# tool_use, usage). Bedrock speaks the model-agnostic Converse API instead, so
# it has its own format/parse path (see _bedrock_converse_call).
ANTHROPIC_FORMAT_BACKENDS: frozenset[str] = frozenset({"anthropic", "claude-agent-sdk"})
AGENT_SDK_BACKEND = "claude-agent-sdk"
# Backends authenticated with an Anthropic API key (and optional workspace id).
ANTHROPIC_KEY_BACKENDS: frozenset[str] = frozenset({"anthropic", AGENT_SDK_BACKEND})
# Backend kinds that need an API key prompted, saved, and verified.
KEYED_KINDS: frozenset[str] = frozenset({"api", "agent-sdk"})
# Backends that return JSON with native tool calls (vs. XML-in-text), parsed by
# _parse_native_response. Bedrock/Converse is native too.
NATIVE_TOOL_BACKENDS: frozenset[str] = frozenset(
    {"anthropic", "openai", "nanogpt", "openai-compatible", "bedrock"}
)
# Native-tool backends speaking OpenAI chat completions (tool_calls / role "tool").
# NanoGPT must use this path: its GLM models reserve <tool_call> as a template
# token, so the XML-in-text tool prompt makes the upstream request fail (503).
OPENAI_FORMAT_BACKENDS: frozenset[str] = frozenset(
    {"openai", "nanogpt", "openai-compatible"}
)
# Hosted backends reached over HTTP (vs. in-process local-ml weights). Bedrock
# is kind "aws" so it isn't in API_BACKENDS, but it's still a network call, and
# openai-compatible servers are often hosted (Hugging Face) or remote.
HOSTED_BACKENDS: frozenset[str] = API_BACKENDS | frozenset(
    {"bedrock", "openai-compatible"}
)

CONFIG_DIR = pathlib.Path(
    os.environ.get("WRENCODE_CONFIG_DIR", "~/.wrencode")
).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.json"
OPENROUTER_MODELS_CACHE = CONFIG_DIR / "openrouter_models.json"
ANTHROPIC_MODELS_CACHE = CONFIG_DIR / "anthropic_models.json"
OPENAI_MODELS_CACHE = CONFIG_DIR / "openai_models.json"
NANOGPT_MODELS_CACHE = CONFIG_DIR / "nanogpt_models.json"

# Populated by apply_backend() once configuration is resolved (see resolve_configuration).
BACKEND: str = ""
MODEL: str = ""
API_KEY: str = ""
API_BASE: str = ""
AWS_REGION: str = ""
# Multi-workspace Anthropic API keys need this on every request
# (anthropic-workspace-id). Workspace-scoped keys can leave it empty.
ANTHROPIC_WORKSPACE_ID: str = ""
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


def _openai_compatible_base() -> str:
    """Return the openai-compatible server's base URL (ending in /v1, usually)."""
    base = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
    return base.rstrip("/").removesuffix("/chat/completions")


def _env_api_key(backend: str) -> str:
    """The backend's key from the environment (or .env), unless /configure saved
    a key that should override it (config "api_key_overrides_env")."""
    key_env = BACKEND_SPECS[backend].get("key_env", "")
    if not key_env:
        return ""
    cfg = load_config()
    if (
        cfg.get("backend") == backend
        and cfg.get("api_key_overrides_env")
        and cfg.get("api_key")
    ):
        return ""
    return os.environ.get(key_env, "")


def apply_backend(
    backend: str,
    model: str = "",
    api_key: str = "",
    anthropic_workspace_id: str = "",
) -> None:
    """Set the module-level backend globals from a backend name plus overrides.

    Precedence for each value: explicit environment variable > saved/chosen
    value > built-in default. The one exception is an API key entered in
    /configure to replace an environment key (see _env_api_key). Heavy backend
    imports are deferred to load_model().
    """
    global BACKEND, MODEL, API_KEY, API_BASE, AWS_REGION, LOCAL_PORT
    global ANTHROPIC_WORKSPACE_ID
    spec = BACKEND_SPECS[backend]
    BACKEND = backend
    MODEL = os.environ.get("MODEL") or model or spec["model"]
    ANTHROPIC_WORKSPACE_ID = (
        (os.environ.get("ANTHROPIC_WORKSPACE_ID") or anthropic_workspace_id or "")
        if backend in ANTHROPIC_KEY_BACKENDS
        else ""
    )
    if spec["kind"] in KEYED_KINDS:
        API_KEY = _env_api_key(backend) or api_key or ""
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
    elif backend == "openai-compatible":
        # vLLM and llama-server accept any key unless started with one.
        API_KEY = _env_api_key(backend) or api_key or "EMPTY"
        API_BASE = f"{_openai_compatible_base()}/chat/completions"
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
# Claude models think before answering and that counts toward max_tokens, so the
# Claude backends get more room unless MAX_TOKENS is set explicitly.
CLAUDE_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16000"))
# Optional reasoning effort for Claude models: low, medium, high, xhigh, or max.
# Unset or unrecognized leaves the model's default (medium on Claude Opus 5.5).
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
_EFFORT_LEVELS: dict[str, EffortLevel] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
CLAUDE_EFFORT: EffortLevel | None = _EFFORT_LEVELS.get(
    os.environ.get("WRENCODE_EFFORT", "").strip().lower()
)
HTTP_TIMEOUT = float(os.environ.get("WRENCODE_HTTP_TIMEOUT", "600"))
# The model's context window, in tokens: drives auto-compaction and how much of a
# transcript `synthesize` sends. Set it for local models with small windows.
CONTEXT_TOKENS = int(os.environ.get("WRENCODE_CONTEXT_TOKENS", "128000"))
HTTP_RETRIES = int(os.environ.get("WRENCODE_HTTP_RETRIES", "2"))


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


ToolSpec = tuple[str, str, dict[str, Any]]  # (name, description, JSON schema)


def _build_tool_schemas(fmt: str, specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Wrap tool specs as 'anthropic', 'converse', or 'openai' tool definitions."""
    out = []
    for name, desc, schema in specs:
        if fmt == "anthropic":
            out.append({"name": name, "description": desc, "input_schema": schema})
        elif fmt == "converse":
            out.append(
                {
                    "toolSpec": {
                        "name": name,
                        "description": desc,
                        "inputSchema": {"json": schema},
                    }
                }
            )
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


def _claude_output_config() -> dict[str, Any]:
    """Request fields for WRENCODE_EFFORT, or nothing when it's unset."""
    return {"output_config": {"effort": CLAUDE_EFFORT}} if CLAUDE_EFFORT else {}


def _anthropic_headers(*, api_key: str = "", workspace_id: str = "") -> dict[str, str]:
    """Build Anthropic request headers, including workspace id when required.

    Multi-workspace (identity-linked) API keys must send anthropic-workspace-id
    on every request; workspace-scoped keys may omit it.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key or API_KEY,
        "anthropic-version": "2023-06-01",
    }
    ws = workspace_id or ANTHROPIC_WORKSPACE_ID
    if ws:
        headers["anthropic-workspace-id"] = ws
    return headers


def _openai_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}


def _is_truncated(data: dict[str, Any]) -> bool:
    """Return True if a native API response stopped at max_tokens."""
    if BACKEND == "bedrock":
        return data.get("stopReason") == "max_tokens"
    if BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        return data.get("stop_reason") == "max_tokens"
    return (data.get("choices") or [{}])[0].get("finish_reason") == "length"


def _warn_if_truncated(data: dict[str, Any]) -> None:
    """Print a stderr warning if the API truncated the response at max_tokens.

    A truncated `tool_use` may be missing required fields (e.g. `content` for
    `write`), so callers need to know to raise MAX_TOKENS rather than silently
    treating an empty operation as success.
    """
    u = data.get("usage", {})
    out_tokens = u.get(
        "outputTokens", u.get("output_tokens", u.get("completion_tokens"))
    )
    if _is_truncated(data):
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


def _parse_native_response(data: dict[str, Any]) -> tuple[str, list[ToolCall]]:
    """Parse a native (Anthropic / Bedrock Converse / OpenAI) response into text + tool calls."""
    _warn_if_truncated(data)
    _log_usage_debug(data)
    if BACKEND == "bedrock":
        blocks = data.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(b["text"] for b in blocks if "text" in b).strip()
        calls = [
            ToolCall(
                b["toolUse"]["toolUseId"],
                b["toolUse"]["name"],
                b["toolUse"].get("input", {}),
            )
            for b in blocks
            if "toolUse" in b
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
    elif BACKEND in OPENAI_FORMAT_BACKENDS:
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
    if BACKEND in OPENAI_FORMAT_BACKENDS:
        messages.extend(
            {"role": "tool", "tool_call_id": tc.id, "content": r} for tc, r in results
        )
    elif BACKEND == "bedrock":
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tc.id,
                            "content": [{"text": r}],
                            "status": "success",
                        }
                    }
                    for tc, r in results
                ],
            }
        )
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
    """POST pre-encoded bytes to a URL and return the parsed JSON response.

    Responses aren't streamed, so the timeout must cover a whole generation.
    Rate limits and server errors (429/5xx) are retried with backoff, and so are
    network failures: refused or dropped connections and timeouts. (A Bedrock
    request is signed once, so a retry after a long timeout may be refused as
    expired; that surfaces as an HTTP 403 rather than a hang.)
    """
    req = urllib.request.Request(url, data=data, headers=headers)
    for attempt in range(HTTP_RETRIES + 1):
        wait = 2 ** (attempt + 1)
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = ui.visible(e.read().decode(errors="replace"))
            if e.code in {429, 500, 502, 503, 504} and attempt < HTTP_RETRIES:
                print(
                    f"{YELLOW}HTTP {e.code}, retrying in {wait}s{RESET}",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            hint = ""
            if e.code == 400 and "workspace" in body.lower():
                hint = (
                    " Set ANTHROPIC_WORKSPACE_ID (or re-run /configure and enter a "
                    "workspace id from Settings → Workspaces)."
                )
            raise RuntimeError(f"HTTP {e.code}: {body}{hint}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            # urlopen wraps connect errors in URLError; a dropped connection while
            # waiting (RemoteDisconnected) or a read timeout comes through raw.
            if attempt == HTTP_RETRIES:
                raise
            reason = getattr(e, "reason", None) or e
            print(
                f"{YELLOW}Network error ({reason}), retrying in {wait}s{RESET}",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


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
    canonical_request = (
        f"{method}\n{canonical_uri}\n{canonical_qs}\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
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
        raise RuntimeError(
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


def _defang_tool_tags(text: str) -> str:
    return text.replace("<tool_call>", "<tool-call>").replace(
        "</tool_call>", "</tool-call>"
    )


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce history into OpenAI chat format.

    History is shared across backends (~/.wrencode/history.json), so turns saved by
    Anthropic/Bedrock backends carry content-block lists that OpenAI-compatible APIs
    reject. tool_use/toolUse blocks become assistant tool_calls and their results
    become role "tool" messages; other blocks are flattened to text. Native OpenAI
    turns pass through. Tool calls left without a result are dropped, since the API
    requires every tool_call id to be answered. Literal <tool_call> tags left in text
    (from XML-backend turns) are defanged: GLM models treat them as template tokens
    and NanoGPT fails the request with a 503.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        msg = m
        if isinstance(content, str) and "tool_call>" in content:
            msg = {**m, "content": _defang_tool_tags(content)}
        if content is None or isinstance(content, str):
            out.append(msg)
            continue
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" or "toolUse" in block:
                tu = block.get("toolUse", block)
                calls.append(
                    {
                        "id": tu.get("id") or tu.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                )
            elif block.get("type") == "tool_result" or "toolResult" in block:
                tr = block.get("toolResult", block)
                body = tr.get("content", "")
                if not isinstance(body, str):
                    body = flatten_content(body)
                body = _defang_tool_tags(body)
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id")
                        or tr.get("toolUseId", ""),
                        "content": body,
                    }
                )
            else:
                texts.append(flatten_content([block]))
        text = _defang_tool_tags("\n".join(t for t in texts if t))
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            out.extend(results)
            if text:
                out.append({"role": m["role"], "content": text})

    # Keep only tool_calls answered by the tool messages directly after them, and
    # only tool messages answering the assistant turn before them.
    fixed: list[dict[str, Any]] = []
    i = 0
    while i < len(out):
        m = out[i]
        if m["role"] == "tool":  # orphan: no preceding assistant tool_calls
            fixed.append({"role": "user", "content": f"Tool result: {m['content']}"})
            i += 1
            continue
        if m["role"] == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < len(out) and out[j]["role"] == "tool":
                j += 1
            answered = {t["tool_call_id"] for t in out[i + 1 : j]}
            calls = [c for c in m["tool_calls"] if c["id"] in answered]
            kept = {c["id"] for c in calls}
            msg = {k: v for k, v in m.items() if k != "tool_calls"}
            if calls:
                msg["tool_calls"] = calls
            fixed.append(msg)
            for t in out[i + 1 : j]:
                if t["tool_call_id"] in kept:
                    fixed.append(t)
                else:
                    fixed.append(
                        {"role": "user", "content": f"Tool result: {t['content']}"}
                    )
            i = j
            continue
        fixed.append(m)
        i += 1
    return fixed


# -----------------------------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------------------------
def get_response(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: tuple[Any, Any] | None,
    tools: list[ToolSpec] | None = None,
) -> str:
    """Generate a response from the configured backend given the message history.

    `tools` are offered to backends with native tool calling; the XML-in-text
    backends get their tool instructions from the system prompt instead.
    """
    flat = [
        {"role": m["role"], "content": flatten_content(m["content"])} for m in messages
    ]

    # OpenAI - native function calling
    if BACKEND in OPENAI_FORMAT_BACKENDS:
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *_to_openai_messages(messages),
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
                "tools": _build_tool_schemas("openai", tools or []),
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
                "messages": [{"role": "system", "content": system_prompt}, *flat],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
            },
            _openai_headers(),
        )
        return str(data["choices"][0]["message"]["content"])

    # AWS Bedrock — model-agnostic Converse API (Claude, Llama, Nova, GPT-OSS…).
    if BACKEND == "bedrock":
        defs = _build_tool_schemas("converse", tools or [])
        body: dict[str, Any] = {
            "messages": [_to_converse_message(m) for m in messages],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS, "temperature": 0.3},
        }
        if defs:
            body["toolConfig"] = {"tools": defs}
        data = _bedrock_converse_call(body)
        return json.dumps(data)  # return raw for agent loop to parse natively

    # Anthropic native tool use API.
    if BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        # Prompt caching, three of the four breakpoints allowed per request:
        # explicit markers on the last tool schema and the static system block
        # give the shared prefix a guaranteed read point, and the top-level
        # cache_control (automatic caching) moves a breakpoint to the end of
        # the growing conversation, so each agent step re-reads the whole
        # history from cache instead of paying full input price for it.
        # Entries live ~5 minutes and refresh on every read.
        defs = _build_tool_schemas("anthropic", tools or [])
        if defs:
            defs[-1] = {**defs[-1], "cache_control": {"type": "ephemeral"}}
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
                "max_tokens": CLAUDE_MAX_TOKENS,
                "tools": defs,
                "cache_control": {"type": "ephemeral"},
                **_claude_output_config(),
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

    # In-process weights (transformers / mlx)
    assert mlx_state is not None  # load_model populates this for ML backends
    chat: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if c := flatten_content(m["content"]):
            chat.append({"role": m["role"], "content": c})
    return _generate_local(chat, mlx_state, MAX_TOKENS)


def _generate_local(
    chat: list[dict[str, str]],
    mlx_state: tuple[Any, Any],
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    """Run one chat through the in-process model (transformers or mlx); return its text.

    Generation stops once a <tool_call> block is complete, and anything after a
    leaked next turn is dropped, so the agent loop can parse the reply as-is.
    """
    model, tokenizer = mlx_state
    if BACKEND == "transformers":
        inputs = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        sampling = (
            {"do_sample": True, "temperature": temperature} if temperature > 0 else {}
        )
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_tokens, **sampling)
        raw = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False
        )
        end = _tool_call_complete(raw)
        if end != -1:
            raw = raw[:end]
        return truncate_at_turn_leak(strip_gptoss_tokens(raw))
    # MLX (Apple Silicon)
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(
        temp=temperature, top_p=0.95, min_p=0.0, min_tokens_to_keep=1
    )
    out = ""
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
    ):
        ui.check_cancelled()
        out += chunk.text
        end = _tool_call_complete(out)
        if end != -1:
            out = out[:end]
            break
    if out.startswith(prompt):
        out = out[len(prompt) :].strip()
    return truncate_at_turn_leak(strip_gptoss_tokens(out))


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def complete(
    system_prompt: str,
    user_text: str,
    *,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    prefill: str = "",
    mlx_state: tuple[Any, Any] | None = None,
) -> str:
    """A one-shot completion on the configured backend, with no tools: the reply's text.

    Used for compaction summaries and by `synthesize`. `prefill` is text the reply
    must continue from (it forces JSON, for instance): the Anthropic and Bedrock
    APIs honor it and it is returned as part of the result; other backends ignore
    it. `temperature` applies where the API takes one. `max_tokens` defaults to
    the backend's usual cap.
    """
    if max_tokens is None:
        max_tokens = (
            CLAUDE_MAX_TOKENS if BACKEND in ANTHROPIC_FORMAT_BACKENDS else MAX_TOKENS
        )
    if BACKEND == "bedrock":
        cmsgs: list[dict[str, Any]] = [
            {"role": "user", "content": [{"text": user_text}]}
        ]
        if prefill:
            cmsgs.append({"role": "assistant", "content": [{"text": prefill}]})
        data = _bedrock_converse_call(
            {
                "messages": cmsgs,
                "system": [{"text": system_prompt}],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            }
        )
        text, _ = _parse_native_response(data)
        return (prefill + text).strip()
    if BACKEND in ANTHROPIC_FORMAT_BACKENDS or BACKEND == "local":
        amsgs: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
        if prefill and BACKEND != "local":
            amsgs.append({"role": "assistant", "content": prefill})
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": system_prompt,
                "messages": amsgs,
                "max_tokens": max_tokens,
            },
            _anthropic_headers(),
        )
        if BACKEND == "local":  # a proxied local model: plain text, maybe tool tags
            text = "".join(
                b["text"] for b in data.get("content", []) if b.get("type") == "text"
            )
            return _TOOL_CALL_BLOCK.sub("", strip_gptoss_tokens(text)).strip()
        text, _ = _parse_native_response(data)
        return (prefill + text).strip()
    if BACKEND in OPENAI_FORMAT_BACKENDS or BACKEND in {"openrouter", "ollama"}:
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            _openai_headers(),
        )
        text, _ = _parse_native_response(data)
        return text.strip()
    if BACKEND in LOCAL_ML_BACKENDS and mlx_state is not None:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        text = _generate_local(chat, mlx_state, max_tokens, temperature)
        return _TOOL_CALL_BLOCK.sub("", text).strip()
    raise RuntimeError(f"one-shot completion isn't supported for backend '{BACKEND}'")


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


def _list_openai_compatible_models() -> tuple[list[str], str]:
    """Return (model ids, "") from the server's GET /models, or ([], why it failed)."""
    key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        req = urllib.request.Request(
            f"{_openai_compatible_base()}/models", headers=headers
        )
        # Generous timeout: serverless hosts (Modal, etc.) may cold-start here.
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        return sorted(m["id"] for m in data.get("data", []) if m.get("id")), ""
    except urllib.error.HTTPError as err:
        if err.code in {401, 403}:
            return (
                [],
                f"the server rejected the key (HTTP {err.code}); check OPENAI_COMPATIBLE_API_KEY",
            )
        return [], f"HTTP {err.code}"
    except (OSError, ValueError) as err:  # connection, HTTP, or bad JSON
        return [], f"{err} — is the server running?"


# -----------------------------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------------------------
def load_model() -> tuple[Any, Any] | None:
    """Load model for the current backend and return mlx_state (or None for API backends)."""
    global MODEL
    if BACKEND == AGENT_SDK_BACKEND:
        if sys.version_info < (3, 10):
            print(f"{RED}The Claude Agent SDK needs Python 3.10 or newer.{RESET}")
            raise SystemExit(1)
        import importlib.util

        if importlib.util.find_spec("claude_agent_sdk") is None:
            print(
                f"{RED}This backend needs the Claude Agent SDK:{RESET} pip install claude-agent-sdk"
            )
            print(f"{DIM}Or pick another backend with /backend.{RESET}")
            raise SystemExit(1)
        if not API_KEY:
            print(f"{RED}ANTHROPIC_API_KEY not set — run /configure.{RESET}")
            raise SystemExit(1)
        return None
    if BACKEND == "mlx":
        try:
            global load, stream_generate, make_sampler
            from mlx_lm import load
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            print(f"{RED}MLX backend needs mlx-lm:{RESET} pip install mlx-lm")
            print(f"{DIM}Or run `wrencode` to pick a hosted backend.{RESET}")
            raise SystemExit(1) from None
        print(f"{YELLOW}Loading model...{RESET}")
        model, tokenizer = load(MODEL)
        ui.print_system(f"✓ Loaded: {getattr(model, 'name', MODEL)}")
        print()
        return (model, tokenizer)
    if BACKEND == "transformers":
        try:
            global torch, AutoModelForCausalLM, AutoTokenizer
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError:
            print(
                f"{RED}transformers backend needs:{RESET} "
                "pip install transformers torch"
            )
            print(f"{DIM}Or run `wrencode` to pick a hosted backend.{RESET}")
            raise SystemExit(1) from None
        print(f"{YELLOW}Loading model via transformers...{RESET}")
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        # Load then move to the device. device_map= is for multi-device sharding
        # (needs accelerate, rejects a plain "mps"/"cpu" string in current transformers).
        _mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(
            _device
        )
        ui.print_system(f"✓ Loaded on {_device}: {MODEL}")
        print()
        return (_mdl, _tok)
    if BACKEND == "local":
        ui.print_system(f"Local proxy at {API_BASE}")
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
        except (OSError, ValueError):  # connection, HTTP, or bad JSON
            print(
                f"{YELLOW}⚠ Couldn't reach Ollama at {base} — is `ollama serve` running?{RESET}"
            )
        ui.print_system(f"{BACKEND} ({MODEL})")
        return None
    if BACKEND == "openai-compatible":
        base = _openai_compatible_base()
        served, why = _list_openai_compatible_models()
        if not served:
            print(f"{YELLOW}⚠ Couldn't list models at {base}: {why}{RESET}")
        elif not MODEL and len(served) == 1:
            MODEL = served[0]
        if not MODEL:
            print(f"{RED}No model set for {base}.{RESET}")
            if served:
                print(f"{DIM}Set MODEL to one of: {', '.join(served[:10])}{RESET}")
            raise SystemExit(1)
        ui.print_system(f"{BACKEND} ({MODEL}) @ {base}")
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
        ui.print_system(f"bedrock ({MODEL}) @ {_aws_region()}")
        return None
    # Hosted API backends — all require a key.
    if not API_KEY:
        key_env = BACKEND_SPECS[BACKEND]["key_env"]
        print(f"{RED}{key_env} not set.{RESET}")
        print(
            f"{DIM}Set {key_env}, or run `wrencode` in a terminal to enter a key.{RESET}"
        )
        raise SystemExit(1)
    ui.print_system(f"{BACKEND} ({MODEL})")
    return None
