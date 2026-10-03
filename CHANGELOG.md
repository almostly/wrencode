## 0.2.0 (2026-10-03)

### Feat

- --verify checks a headless run before calling it done
- typed answers with --json-schema in headless mode
- ANSI Shadow startup banner in a blue gradient
- compact long sessions automatically
- add openai-compatible backend for vLLM, llama.cpp, and Hugging Face
- add headless -p/--print mode
- load AGENTS.md project instructions into the system prompt
- **nanogpt**: add NanoGPT backend with native OpenAI-format tool calls

### Fix

- resend garbled tool calls instead of treating them as the answer
- **edit**: point to the file that has the text when it's the wrong one
- stop on identical failing tool calls even when interleaved
- **edit**: accept a uniformly shifted indent and show the closest match
- **openai-compatible**: say when the server rejects the API key
- recover when a response is cut off before acting
- wait up to 600s for model responses and retry 429/5xx
- report headless setup failures in the JSON result

## 0.1.5.1 (2026-06-08)

### Fix

- **edit**: match `old` exactly so indentation isn't duplicated

## 0.1.5 (2026-06-07)

### Feat

- **synthesize**: add `wrencode synthesize`, a semantic git-merge of chat transcripts
- **synthesize**: model-agnostic adapters, arrow-key picker, git-like submodes
- **synthesize**: give decisions a stable subject key for sharper conflict detection
- **bedrock**: add AWS Bedrock backend via the Converse API (stdlib SigV4)

### Fix

- **bedrock**: use SigV4 service name `bedrock`, not `bedrock-runtime`; refresh model list
- **bedrock**: scope model list to verified tool-capable models
- `colors_enabled` follows CPython `can_colorize` precedence

### Refactor

- make `wrencode.py` clean under the ty type checker

## 0.1.4.7 (2026-06-01)

### Feat

- validate the API key on startup; add uninstall

### Fix

- immediate exit when no `.env` is present

## 0.1.4.6 (2026-05-30)

### Fix

- bundle certifi so the standalone binary can do TLS

## 0.1.4.5 (2026-05-29)

### Perf

- enable Anthropic prompt caching on system prompt and tools

## 0.1.4.4 (2026-05-28)

### Fix

- guard against `MAX_TOKENS`-truncated `tool_use` blocks

## 0.1.4.3 (2026-05-28)

### Feat

- cleaner CLI layout, context loader, and approval flow

### Fix

- `available_backends` precedence
- `read_confirm_choice` regression

### Refactor

- unify `run_agent_turn` with a `ToolCall` dataclass and `_append_*` helpers
- collapse remaining backend duplication; inline loaders, merge tool schemas, factor headers
- use the stdlib where it does the work (`shlex`, `raw_decode`, `json.load`)

## 0.1.4.2 (2026-05-27)

### Feat

- safety guardrails for edit/grep/parser loops
- stdlib `unittest` suite and CI workflow

## 0.1.4.1 (2026-05-25)

### Feat

- syntax-highlighted code blocks, inline code, cleaner prompt, terminal title
- friendlier failures: Ollama model pre-flight, actionable 404 hint, traceback behind `WRENCODE_DEBUG`
- default mlx to `gpt-oss-claude-mlx`; link HF models

## 0.1.4 (2026-05-24)

### Feat

- `task` subagent tool (sequential, in-process)
- headless auto-approve mode (`WRENCODE_AUTO_APPROVE` / `--yes`)
- `ollama` backend
- curl installer and first-run backend chooser
- PyPI packaging and publish workflow

### Fix

- load transformers models with `.to(device)`, not `device_map`
- parse local-model tool calls without a closing tag
- startup and `.env` loading bugs

### Refactor

- extract typed frozenset backend groups

## 0.1.3 (2026-05-06)

### Feat

- grep fallback when ripgrep is missing

## 0.1.2 (2026-05-01)

### Fix

- use `build_prefix` in the release matrix to avoid the arch flag on Linux

## v0.1.1 (2026-05-01)

### Fix

- user-level history file; drop the macos-13 runner

## v0.1.0 (2026-05-01)

### Feat

- single-file agent loop with read/write/edit/glob/grep/bash tools
- native tool calling for Anthropic and OpenAI
- `/compact` for OpenAI and Anthropic models
- binary release workflow
