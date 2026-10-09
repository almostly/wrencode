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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from . import ui
from .ui import DIM, RED, RESET, YELLOW

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
# Backends whose replies can be streamed over server-sent events: the Anthropic
# Messages API and OpenAI-style chat completions. Bedrock's Converse stream is
# a binary event stream and the local proxy is left whole.
STREAMING_BACKENDS: frozenset[str] = frozenset(
    {"anthropic", "openai", "nanogpt", "openai-compatible", "openrouter", "ollama"}
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
# Stream replies token by token where the API can (Anthropic and OpenAI-format
# backends); WRENCODE_STREAM=0 waits for whole replies instead.
STREAM = os.environ.get("WRENCODE_STREAM", "1").lower() not in ("0", "false", "no")
# Anthropic's server-side web search, offered to Claude on the anthropic backend;
# WRENCODE_WEB_SEARCH=0 leaves it out. Each search is billed on top of tokens.
WEB_SEARCH = os.environ.get("WRENCODE_WEB_SEARCH", "1").lower() not in (
    "0",
    "false",
    "no",
)
WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search_20260209", "name": "web_search"}
WEB_SEARCH_USD = 0.01  # $10 per 1,000 searches


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


@dataclass(frozen=True)
class Price:
    """USD per million tokens: input, output, prompt-cache read, prompt-cache write.

    `long` is the rate card used once the prompt passes 100K tokens, for models
    that charge more there (Claude Haiku 5.5); `source` says where the figures
    came from, for /usage.
    """

    input: float
    output: float
    cache_read: float
    cache_write: float
    source: str = "built-in"
    long: tuple[float, float] | None = None

    def at(self, prompt: int) -> tuple[float, float, float, float]:
        """The four rates that apply to a request with `prompt` input tokens."""
        if self.long and prompt > 100_000:
            inp, out = self.long
            scale = inp / self.input if self.input else 1.0
            return inp, out, self.cache_read * scale, self.cache_write * scale
        return self.input, self.output, self.cache_read, self.cache_write

    def label(self) -> str:
        """'$2/$10 per MTok', the way price lists put it."""
        return f"${_rate(self.input)}/${_rate(self.output)} per MTok"


def _rate(usd: float) -> str:
    """A per-million rate: 2, 0.10, 0.075."""
    if usd == int(usd):
        return str(int(usd))
    text = f"{usd:.4f}".rstrip("0")
    return text if len(text.split(".")[1]) >= 2 else text + "0"


def _p(
    inp: float,
    out: float,
    read: float | None = None,
    long: tuple[float, float] | None = None,
) -> Price:
    """A Claude-style price: cache reads 0.1x input unless given, writes 1.25x."""
    return Price(inp, out, inp * 0.1 if read is None else read, inp * 1.25, long=long)


# Prices the model APIs don't report (Anthropic's /v1/models has none), USD per
# million tokens, checked October 2026. Matched as substrings of the lowercased
# model id with '.' read as '-', so the same row covers claude-sonnet-4-5-20250929
# (Anthropic), anthropic/claude-sonnet-4.5 (OpenRouter) and
# us.anthropic.claude-sonnet-4-5-20250929-v1:0 (Bedrock). Order matters: a more
# specific id goes before the prefix it extends. WRENCODE_PRICE overrides.
PRICE_TABLE: tuple[tuple[str, Price], ...] = (
    ("claude-fable-5-1", _p(10, 50, 0.25)),
    ("claude-mythos-5-1", _p(10, 50, 0.25)),
    ("claude-fable-5", _p(10, 50, 1.0)),
    ("claude-mythos-5", _p(10, 50, 1.0)),
    ("claude-opus-5-5", _p(4, 20, 0.20)),
    ("claude-opus-5", _p(5, 25)),
    ("claude-sonnet-5-5", _p(2, 10)),
    ("claude-sonnet-5", _p(2, 10)),
    ("claude-haiku-5-5", _p(0.10, 0.50, long=(0.50, 2.50))),
    ("claude-opus-4-1", _p(15, 75)),
    ("claude-opus-4-5", _p(5, 25)),
    ("claude-opus-4-6", _p(5, 25)),
    ("claude-opus-4-7", _p(5, 25)),
    ("claude-opus-4-8", _p(5, 25)),
    ("claude-opus-4", _p(15, 75)),
    ("claude-sonnet-4", _p(3, 15)),
    ("claude-haiku-4-5", _p(1, 5)),
    ("claude-3-7-sonnet", _p(3, 15)),
    ("claude-3-5-sonnet", _p(3, 15)),
    ("claude-3-5-haiku", _p(0.80, 4)),
    ("claude-3-opus", _p(15, 75)),
    ("claude-3-haiku", _p(0.25, 1.25)),
    ("gpt-4o-mini", Price(0.15, 0.60, 0.075, 0)),
    ("gpt-4o", Price(2.50, 10, 1.25, 0)),
    ("gpt-4-1-nano", Price(0.10, 0.40, 0.025, 0)),
    ("gpt-4-1-mini", Price(0.40, 1.60, 0.10, 0)),
    ("gpt-4-1", Price(2, 8, 0.50, 0)),
    ("gpt-5-nano", Price(0.05, 0.40, 0.005, 0)),
    ("gpt-5-mini", Price(0.25, 2, 0.025, 0)),
    ("gpt-5", Price(1.25, 10, 0.125, 0)),
    ("nova-micro", Price(0.035, 0.14, 0.00875, 0)),
    ("nova-lite", Price(0.06, 0.24, 0.015, 0)),
    ("nova-pro", Price(0.80, 3.20, 0.20, 0)),
)
# Prices /configure saved from a provider catalog that lists them (OpenRouter,
# NanoGPT): {"<backend>/<model id>": [input, output, cache_read, cache_write]}.
PRICES_FILE = CONFIG_DIR / "prices.json"
_PRICES_FILE_CACHE: tuple[float, dict[str, Any]] = (-1.0, {})


def _catalog_prices() -> dict[str, Any]:
    """The saved catalog prices, re-read when the file changes."""
    global _PRICES_FILE_CACHE
    try:
        mtime = PRICES_FILE.stat().st_mtime
    except OSError:
        return {}
    if mtime != _PRICES_FILE_CACHE[0]:
        try:
            data = json.loads(PRICES_FILE.read_text())
        except (OSError, ValueError):
            data = {}
        _PRICES_FILE_CACHE = (mtime, data if isinstance(data, dict) else {})
    return _PRICES_FILE_CACHE[1]


def _price_from_rates(rates: Any, source: str) -> Price | None:
    """Price from an [input, output, cache_read, cache_write] list (the last two optional)."""
    try:
        nums = [float(x) for x in rates]
    except (TypeError, ValueError):
        return None
    if len(nums) < 2 or len(nums) > 4 or any(n < 0 for n in nums):
        return None
    inp, out = nums[0], nums[1]
    read = nums[2] if len(nums) > 2 else inp
    write = nums[3] if len(nums) > 3 else inp
    return Price(inp, out, read, write, source=source)


def price_for(model: str = "", backend: str = "") -> Price | None:
    """The price of `model` (default: the active one), or None when it isn't known.

    WRENCODE_PRICE="input,output[,cache_read[,cache_write]]" (USD per million
    tokens) wins, then the catalog prices /configure saved, then the built-in
    table. Local models and proxies have no price unless WRENCODE_PRICE says so.
    """
    model = model or MODEL
    backend = backend or BACKEND
    if env := os.environ.get("WRENCODE_PRICE", "").strip():
        return _price_from_rates(env.split(","), "WRENCODE_PRICE")
    saved = _catalog_prices().get(f"{backend}/{model}")
    if saved and (price := _price_from_rates(saved, f"{backend} catalog")):
        return price
    if backend in LOCAL_ML_BACKENDS or backend == "local":
        return None
    needle = model.lower().replace(".", "-")
    for key, price in PRICE_TABLE:
        if key in needle:
            return price
    return None


def call_cost(
    price: Price | None, uncached: int, cache_read: int, cache_write: int, out: int
) -> float | None:
    """What one model call cost in USD, or None without a price."""
    if price is None:
        return None
    inp, outp, read, write = price.at(uncached + cache_read + cache_write)
    return (uncached * inp + cache_read * read + cache_write * write + out * outp) / 1e6


@dataclass
class Usage:
    """Token counts the backend reported: the current turn and the whole session.

    `uncached` are input tokens charged in full, `cache_read` the ones served from
    the prompt cache, `cache_write` the ones written to it. `prompt` is the size
    of the latest request's prompt: what the context window currently holds.
    Costs are USD from price_for(); `unpriced_calls` counts the calls made
    without a known price, so a total is only shown when it is complete.
    `*_seconds` is the wall time of the timed requests, for the output rate:
    tokens per second over the whole request, so the time to the first token
    and the network are in it. `prev_rate` is the previous turn's rate, for the
    trend arrow.
    """

    turn_uncached: int = 0
    turn_cache_read: int = 0
    turn_cache_write: int = 0
    turn_out: int = 0
    turn_calls: int = 0
    turn_cost: float = 0.0
    turn_unpriced: int = 0
    turn_seconds: float = 0.0
    turn_searches: int = 0
    session_uncached: int = 0
    session_cache_read: int = 0
    session_cache_write: int = 0
    session_out: int = 0
    session_calls: int = 0
    session_cost: float = 0.0
    unpriced_calls: int = 0
    session_seconds: float = 0.0
    session_searches: int = 0
    prev_rate: float = 0.0
    call_started: float = 0.0  # time.monotonic() when the current request began
    prompt: int = 0
    last_cached: int = 0  # the latest request's cache reads + writes
    last_out: int = 0

    def begin_turn(self) -> None:
        if self.turn_seconds:
            self.prev_rate = self.turn_rate()
        self.turn_uncached = self.turn_cache_read = self.turn_cache_write = 0
        self.turn_out = self.turn_calls = self.turn_unpriced = self.turn_searches = 0
        self.turn_cost = self.turn_seconds = 0.0

    def begin_call(self) -> None:
        self.call_started = time.monotonic()

    def turn_rate(self) -> float:
        """Output tokens per second over this turn's timed requests, 0 when unknown."""
        return self.turn_out / self.turn_seconds if self.turn_seconds else 0.0

    def session_rate(self) -> float:
        return self.session_out / self.session_seconds if self.session_seconds else 0.0

    def trend(self) -> str:
        """↗ ↘ or → against the previous turn's rate; '' without one to compare."""
        rate = self.turn_rate()
        if not rate or not self.prev_rate:
            return ""
        change = rate / self.prev_rate - 1
        if change >= 0.1:
            return "↗"
        if change <= -0.1:
            return "↘"
        return "→"

    def add(
        self,
        uncached: int,
        cache_read: int,
        cache_write: int,
        out: int,
        cost: float | None = None,
        seconds: float = 0.0,
        searches: int = 0,
    ) -> None:
        self.turn_searches += searches
        self.session_searches += searches
        self.turn_uncached += uncached
        self.turn_cache_read += cache_read
        self.turn_cache_write += cache_write
        self.turn_out += out
        self.turn_calls += 1
        self.session_uncached += uncached
        self.session_cache_read += cache_read
        self.session_cache_write += cache_write
        self.session_out += out
        self.session_calls += 1
        if cost is None:
            self.turn_unpriced += 1
            self.unpriced_calls += 1
        else:
            self.turn_cost += cost
            self.session_cost += cost
        if seconds > 0:
            self.turn_seconds += seconds
            self.session_seconds += seconds
        self.prompt = uncached + cache_read + cache_write
        self.last_cached = cache_read + cache_write
        self.last_out = out

    def as_dict(self) -> dict[str, Any]:
        """Session totals, for the headless JSON result."""
        d: dict[str, Any] = {
            "input_tokens": self.session_uncached
            + self.session_cache_read
            + self.session_cache_write,
            "cache_read_tokens": self.session_cache_read,
            "cache_write_tokens": self.session_cache_write,
            "output_tokens": self.session_out,
            "model_calls": self.session_calls,
            "context_tokens": self.prompt,
        }
        if self.session_calls and not self.unpriced_calls:
            d["cost_usd"] = round(self.session_cost, 6)
        if self.session_seconds:
            d["output_tokens_per_second"] = round(self.session_rate(), 1)
        if self.session_searches:
            d["web_searches"] = self.session_searches
        return d


USAGE = Usage()


def _record_usage(data: dict[str, Any]) -> None:
    """Add a native response's usage block to USAGE, whatever the backend's format."""
    u = data.get("usage") or {}
    if not isinstance(u, dict) or not u:
        return
    if BACKEND == "bedrock":
        uncached = int(u.get("inputTokens") or 0)
        cache_read = int(u.get("cacheReadInputTokens") or 0)
        cache_write = int(u.get("cacheWriteInputTokens") or 0)
        out = int(u.get("outputTokens") or 0)
    elif BACKEND in ANTHROPIC_FORMAT_BACKENDS:
        uncached = int(u.get("input_tokens") or 0)  # excludes the cached tokens
        cache_read = int(u.get("cache_read_input_tokens") or 0)
        cache_write = int(u.get("cache_creation_input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        server = u.get("server_tool_use") or {}
        searches = (
            int(server.get("web_search_requests") or 0)
            if isinstance(server, dict)
            else 0
        )
    else:  # OpenAI chat format: prompt_tokens includes the cached ones
        details = u.get("prompt_tokens_details") or {}
        cache_read = (
            int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        )
        uncached = max(int(u.get("prompt_tokens") or 0) - cache_read, 0)
        cache_write = 0
        out = int(u.get("completion_tokens") or 0)
    searches = locals().get("searches", 0)
    cost = call_cost(price_for(), uncached, cache_read, cache_write, out)
    if cost is not None and searches:
        cost += searches * WEB_SEARCH_USD
    seconds = time.monotonic() - USAGE.call_started if USAGE.call_started else 0.0
    USAGE.call_started = 0.0
    USAGE.add(uncached, cache_read, cache_write, out, cost, seconds, searches)


def _count(n: int) -> str:
    if n < 1000:
        return str(n)
    return f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:.1f}k"


def _money(usd: float) -> str:
    """Dollars at a precision that keeps small amounts visible: $0.0042, $0.21, $12."""
    if usd == 0:
        return "$0"
    if usd < 0.0001:
        return "<$0.0001"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1:
        return f"${usd:.3f}"
    if usd < 100:
        return f"${usd:.2f}"
    return f"${usd:,.0f}"


def context_fill() -> float:
    """How full the context window is after the latest request, 0.0 to 1.0."""
    return min(USAGE.prompt / CONTEXT_TOKENS, 1.0) if CONTEXT_TOKENS else 0.0


def _speed(rate: float) -> str:
    """'42 tok/s', one decimal under ten."""
    return f"{rate:.1f} tok/s" if rate < 10 else f"{rate:.0f} tok/s"


def _pace() -> str:
    """This turn's output rate with the trend arrow, or '' when no request was timed."""
    u = USAGE
    if not u.turn_seconds:
        return ""
    arrow = f" {u.trend()}" if u.trend() else ""
    return f"  {_speed(u.turn_rate())}{arrow}"


def _spend() -> str:
    """The turn's cost with the session total, or '' once a call had no price."""
    u = USAGE
    if u.unpriced_calls:
        return ""
    if u.session_calls == u.turn_calls:
        return f"  {_money(u.session_cost)}"
    return f"  {_money(u.turn_cost)} · total {_money(u.session_cost)}"


def usage_line(warn_at: float = 0.0) -> str:
    """The compact line after a turn: tokens up and down, cache hit rate, context meter, cost.

    `warn_at` colors the meter once the context fill reaches it (the auto-compaction
    threshold), when colors are on.
    """
    u = USAGE
    turn_in = u.turn_uncached + u.turn_cache_read + u.turn_cache_write
    fill = context_fill()
    cells = min(10, round(fill * 10))
    bar = "▰" * cells + "▱" * (10 - cells)
    if warn_at and fill >= warn_at and ui.colors_enabled():
        bar = f"{YELLOW}{bar}{RESET}{DIM}"
    return (
        f"↑ {_count(turn_in)}  ↓ {_count(u.turn_out)}{_pace()}  "
        f"{bar} {100 * fill:.0f}%{_spend()}"
    )


def price_note() -> str:
    """Where the active model's price comes from, or how to set one."""
    price = price_for()
    if price is None:
        return (
            f"price: unknown for {MODEL}; set WRENCODE_PRICE=input,output[,cache_read"
            f"[,cache_write]] in USD per million tokens"
        )
    note = (
        f"price: {price.label()} (cache read ${_rate(price.cache_read)}, "
        f"write ${_rate(price.cache_write)}; {price.source})"
    )
    if price.long:
        note += (
            f"; ${_rate(price.long[0])}/${_rate(price.long[1])} "
            "once the prompt passes 100k"
        )
    return note


def usage_report() -> list[str]:
    """The full numbers for /usage: this turn and the session, then the context and price."""
    u = USAGE
    rows = [
        (
            "this turn",
            u.turn_uncached,
            u.turn_cache_read,
            u.turn_cache_write,
            u.turn_out,
            u.turn_calls,
            "$?" if u.turn_unpriced else _money(u.turn_cost),
        ),
        (
            "session",
            u.session_uncached,
            u.session_cache_read,
            u.session_cache_write,
            u.session_out,
            u.session_calls,
            "$?" if u.unpriced_calls else _money(u.session_cost),
        ),
    ]
    head = f"{'':<10}{'input':>8}{'cached':>8}{'written':>8}{'output':>8}{'calls':>6}{'cost':>10}"
    lines = [head]
    for name, unc, read, write, out, calls, cost in rows:
        lines.append(
            f"{name:<10}{_count(unc + read + write):>8}{_count(read):>8}"
            f"{_count(write):>8}{_count(out):>8}{calls:>6}{cost:>10}"
        )
    lines.append(
        f"context: {_count(u.prompt)} of {_count(CONTEXT_TOKENS)} ({100 * context_fill():.0f}%); "
        f"input = uncached + cached (read) + written"
    )
    lines.append(price_note())
    if u.session_searches:
        lines.append(
            f"web searches: {u.turn_searches} this turn, {u.session_searches} this session "
            f"(${WEB_SEARCH_USD * 1000:.0f} per 1,000)"
        )
    if u.turn_seconds or u.session_seconds:
        speed = f"speed: {_speed(u.session_rate())} this session"
        if u.turn_seconds:
            speed = f"speed: {_speed(u.turn_rate())} this turn"
            if u.trend():
                speed += f" ({u.trend()} from {_speed(u.prev_rate)} last turn)"
            speed += f"; {_speed(u.session_rate())} this session"
        lines.append(speed + "; output tokens over the request's wall time")
    return lines


def usage_title() -> str:
    """The terminal title: the context fill, the session's tokens and spend at a glance."""
    u = USAGE
    session_in = u.session_uncached + u.session_cache_read + u.session_cache_write
    title = f"wrencode · ctx {100 * context_fill():.0f}% · ↑{_count(session_in)} ↓{_count(u.session_out)}"
    if not u.unpriced_calls:
        title += f" · {_money(u.session_cost)}"
    if u.turn_seconds:
        title += f" · {_speed(u.turn_rate())}" + (f" {u.trend()}" if u.trend() else "")
    return title


def send_estimate(typed_chars: int, context_tokens: int = 0) -> str:
    """'≈ $0.0031 input' for sending what's typed so far, or '' without a price.

    The next request carries the previous prompt again (the part the backend
    cached last time is assumed to hit the cache at the read rate; the rest and
    anything new at the write rate once caching is in use, the plain input
    rate otherwise), plus the model's last reply and the typed text at about
    four characters per token. Output can't be known ahead, so it is not counted.
    `context_tokens` is the estimated prompt size before the first call.
    """
    price = price_for()
    if price is None:
        return ""
    u = USAGE
    reused = u.prompt or max(context_tokens, 0)
    cached = min(u.last_cached, reused) if u.prompt else 0
    typed = typed_chars // 4 + 1
    inp, _, read, write = price.at(reused + u.last_out + typed)
    fresh_rate = (write or inp) if cached else inp
    cost = (cached * read + (reused - cached + u.last_out + typed) * fresh_rate) / 1e6
    return f"≈ {_money(cost)} input"


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
    _record_usage(data)
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


def _http_stream(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> Iterator[dict[str, Any]]:
    """POST a JSON payload and yield the JSON objects of its server-sent events.

    The request is retried like _http_post_raw until the first byte arrives;
    after that an error ends the stream. Lines other than `data:` (event
    names, comments, keep-alives) and the `[DONE]` marker are skipped.
    """
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers
    )
    for attempt in range(HTTP_RETRIES + 1):
        wait = 2 ** (attempt + 1)
        try:
            resp = urllib.request.urlopen(req, timeout=HTTP_TIMEOUT)
        except urllib.error.HTTPError as e:
            body = ui.visible(e.read().decode(errors="replace"))
            if e.code in {429, 500, 502, 503, 504} and attempt < HTTP_RETRIES:
                print(
                    f"{YELLOW}HTTP {e.code}, retrying in {wait}s{RESET}",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt == HTTP_RETRIES:
                raise
            reason = getattr(e, "reason", None) or e
            print(
                f"{YELLOW}Network error ({reason}), retrying in {wait}s{RESET}",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue
        with resp:
            for raw in resp:
                ui.check_cancelled()  # Escape: stop reading, closing the connection
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event
        return
    raise AssertionError("unreachable")


def _stream_anthropic(
    events: Iterator[dict[str, Any]],
    on_text: Callable[[str], None],
    on_block: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Assemble a streamed Messages reply into the shape a plain request returns.

    Text deltas go to `on_text` as they arrive. Every block is kept whole, with
    tool inputs parsed from their JSON deltas and thinking blocks carrying their
    signature, so the message can go back into history unchanged. Server-side
    tool blocks (a web search and its results) are passed to `on_block` as each
    completes, so they can be shown in order.
    """
    message: dict[str, Any] = {"role": "assistant", "content": []}
    blocks: list[dict[str, Any]] = message["content"]
    partial: dict[int, str] = {}
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message.update(event.get("message") or {})
            message["content"] = blocks
        elif kind == "content_block_start":
            block = dict(event.get("content_block") or {})
            if block.get("type") in ("tool_use", "server_tool_use"):
                partial[int(event.get("index", len(blocks)))] = ""
            blocks.append(block)
        elif kind == "content_block_delta":
            index = int(event.get("index", len(blocks) - 1))
            while index >= len(blocks):
                blocks.append({"type": "text", "text": ""})
            block = blocks[index]
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = str(delta.get("text", ""))
                block["text"] = block.get("text", "") + text
                if text:
                    on_text(text)
            elif dtype == "input_json_delta":
                partial[index] = partial.get(index, "") + str(
                    delta.get("partial_json", "")
                )
            elif dtype == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + str(
                    delta.get("thinking", "")
                )
            elif dtype == "signature_delta":
                block["signature"] = str(delta.get("signature", ""))
        elif kind == "content_block_stop":
            index = int(event.get("index", -1))
            if index in partial:
                raw = partial.pop(index)
                if raw.strip():  # otherwise the input came whole with the block start
                    try:
                        blocks[index]["input"] = json.loads(raw)
                    except ValueError:
                        blocks[index]["input"] = {}
                elif not isinstance(blocks[index].get("input"), dict):
                    blocks[index]["input"] = {}
            if (
                on_block is not None
                and 0 <= index < len(blocks)
                and str(blocks[index].get("type", "")) in SERVER_BLOCKS
            ):
                on_block(blocks[index])
        elif kind == "message_delta":
            message.update({k: v for k, v in (event.get("delta") or {}).items()})
            usage = event.get("usage") or {}
            message["usage"] = {**message.get("usage", {}), **usage}
        elif kind == "error":
            err = event.get("error") or {}
            raise RuntimeError(
                f"stream error: {err.get('type', '')}: {err.get('message', '')}"
            )
    for index, raw in partial.items():  # a stream cut before the block closed
        if raw.strip():
            with contextlib.suppress(ValueError):
                blocks[index]["input"] = json.loads(raw)
    # A text block opened but never written to (around a tool call, or a cut
    # stream) must not go back to the API: it rejects empty text blocks.
    message["content"] = [
        b for b in blocks if b.get("type") != "text" or str(b.get("text", "")).strip()
    ]
    return message


# Blocks a server-side tool leaves in a reply: what Claude searched and what came back.
SERVER_BLOCKS: frozenset[str] = frozenset({"server_tool_use", "web_search_tool_result"})


def describe_server_block(block: dict[str, Any]) -> str:
    """One line for a server-side tool block: the search, or how many results;
    '' for the code the search runs to filter its results, which is noise."""
    kind = block.get("type")
    if kind == "server_tool_use":
        if block.get("name") != "web_search":
            return ""
        query = (block.get("input") or {}).get("query", "")
        return f"web_search {json.dumps(query, ensure_ascii=False)}"
    content = block.get("content")
    if isinstance(content, dict):  # an error
        return f"search failed: {content.get('error_code', 'error')}"
    results = [c for c in (content or []) if isinstance(c, dict) and c.get("url")]
    titles = ", ".join(str(c.get("title") or c["url"])[:40] for c in results[:3])
    more = f" (+{len(results) - 3})" if len(results) > 3 else ""
    return f"{len(results)} results: {titles}{more}" if results else "no results"


def _stream_openai(
    events: Iterator[dict[str, Any]], on_text: Callable[[str], None]
) -> dict[str, Any]:
    """Assemble streamed chat-completion chunks into one choice with its usage."""
    text = ""
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage: dict[str, Any] = {}
    for chunk in events:
        if chunk.get("error"):
            err = chunk["error"]
            raise RuntimeError(
                f"stream error: {err.get('message', err) if isinstance(err, dict) else err}"
            )
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                text += piece
                on_text(piece)
            for tc in delta.get("tool_calls") or []:
                if "index" in tc:
                    i = int(tc["index"])
                elif tc.get("id") and all(c["id"] != tc["id"] for c in calls.values()):
                    i = len(calls)  # servers that send whole calls without an index
                else:
                    i = max(calls, default=0)
                call = calls.setdefault(
                    i,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    call["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    call["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


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
def _openai_request(
    payload: dict[str, Any], on_text: Callable[[str], None] | None
) -> dict[str, Any]:
    """A chat-completion request, streamed when a text callback is given."""
    if on_text is None or not STREAM:
        return _http_post(API_BASE, payload, _openai_headers())
    streamed = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    return _stream_openai(_http_stream(API_BASE, streamed, _openai_headers()), on_text)


def get_response(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: tuple[Any, Any] | None,
    tools: list[ToolSpec] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_block: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Generate a response from the configured backend given the message history.

    `tools` are offered to backends with native tool calling; the XML-in-text
    backends get their tool instructions from the system prompt instead.
    With `on_text`, hosted Anthropic and OpenAI-format backends stream: each
    piece of reply text is passed to it as it arrives, and the reply is still
    returned whole afterwards, in the same shape as an unstreamed one.
    """
    flat = [
        {"role": m["role"], "content": flatten_content(m["content"])} for m in messages
    ]
    USAGE.begin_call()  # _record_usage reads the clock when the reply is parsed

    # OpenAI - native function calling
    if BACKEND in OPENAI_FORMAT_BACKENDS:
        data = _openai_request(
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
            on_text,
        )
        return json.dumps(data)  # return raw for agent loop to parse natively

    # OpenRouter / Ollama - OpenAI-compatible chat completions (no native tools)
    if BACKEND in {"openrouter", "ollama"}:
        data = _openai_request(
            {
                "model": MODEL,
                "messages": [{"role": "system", "content": system_prompt}, *flat],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
            },
            on_text,
        )
        _record_usage(data)
        return str(data["choices"][0]["message"]["content"] or "")

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
        if WEB_SEARCH and BACKEND == "anthropic" and tools:
            defs = [
                dict(WEB_SEARCH_TOOL),
                *defs,
            ]  # first, so the cache marker stays last
        payload = {
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
        }
        if on_text is not None and STREAM:
            data = _stream_anthropic(
                _http_stream(
                    API_BASE, {**payload, "stream": True}, _anthropic_headers()
                ),
                on_text,
                on_block,
            )
        else:
            data = _http_post(API_BASE, payload, _anthropic_headers())
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
        return None  # the status line under the banner names the model
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
    return None  # the status line under the banner names the model
