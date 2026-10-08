# 🐦 WrenCode

A minimal agent harness for coding. The agent loop is one readable Python file.

Named after Harold Wren - the alias of a genius who built a superintelligent AI and operated quietly in the background.

-----

## What it is

WrenCode is a coding agent harness: everything around the model that turns it into an agent. It runs the tool-calling loop, executes tools, builds the system prompt, and manages context, locally or via API, giving an LLM the ability to read, write, and edit files, search codebases, and run shell commands - enough to autonomously navigate and modify a real project.

Where Claude Code is the batteries-included harness, WrenCode is the **"understand and own your agent" harness**: the entire agent loop fits in one readable file, runs against local or hosted models, and is yours to hack.

## Code layout

Read `wrencode.py` top to bottom to understand the agent; the files beside it are what it calls.

|File                    |What's in it                                                      |
|------------------------|------------------------------------------------------------------|
|`wrencode.py`           |The harness: the seven tools, the system prompt, the turn loop, parallel subagents, context compaction, headless mode, `main()`|
|`wrencode_backends.py`  |Talking to models: backend tables and state, HTTP with retries, request/response formats (Anthropic, OpenAI, Bedrock Converse, local), `get_response()`|
|`wrencode_configure.py` |Picking a backend and model: the first-run chooser, `/configure` and `/model`, API-key prompts and verification, model lists, saved config|
|`wrencode_ui.py`        |The terminal: colors, input with slash-command completion, approvals, Escape-to-cancel, tagged output from parallel subagents|
|`wrencode_sdk.py`       |The `claude-agent-sdk` backend                                    |
|`wrencode_synthesize.py`|The `synthesize` subcommand                                       |

Each module imports only the ones below it in this table's dependency order (`wrencode.py` → backends/configure/sdk/synthesize → ui), so the loop can be read without the rest.

## Backends

On first run WrenCode asks you to pick a backend and saves the choice to
`~/.wrencode/config.json`. Run `wrencode --configure` any time to change it.
Set `BACKEND` (and the matching API key) in the environment to override the
saved choice, e.g. for CI.

|Backend       |Description                             |Availability          |
|--------------|----------------------------------------|----------------------|
|`anthropic`   |Claude via Anthropic API                |binary + source       |
|`claude-agent-sdk`|Claude Code's agent loop and tools via the Claude Agent SDK|source install, Python 3.10+|
|`openai`      |GPT models via OpenAI API               |binary + source       |
|`openrouter`  |Any model via OpenRouter                |binary + source       |
|`nanogpt`     |Any model via NanoGPT                   |binary + source       |
|`ollama`      |Local models via a running `ollama serve`|binary + source     |
|`openai-compatible`|vLLM, llama.cpp, Hugging Face, any OpenAI-compatible server|binary + source|
|`bedrock`     |Any model via AWS Bedrock Converse (AWS credentials)|binary + source|
|`local`       |Local proxy via Anthropic-compatible API|binary + source       |
|`transformers`|HuggingFace Transformers (CPU/MPS/GPU)  |source install only   |
|`mlx`         |Apple Silicon via MLX                   |source install, macOS |

The standalone binary can't bundle the heavy ML stack, so the local-weights
backends (`mlx`, `transformers`) are only offered when running from source.

The default local models are
[`deburky/gpt-oss-claude-code`](https://huggingface.co/deburky/gpt-oss-claude-code)
(transformers) and
[`deburky/gpt-oss-claude-mlx`](https://huggingface.co/deburky/gpt-oss-claude-mlx)
(MLX) — override either with `MODEL=...`.

### Claude Agent SDK

The `claude-agent-sdk` backend hands each prompt to the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), which
runs Claude Code's own agent loop, tools and subagents. WrenCode shows the
stream, asks before edits and commands, and prints the cost of each turn.
Install the extra, then pick the backend in `/configure`:

```bash
pip install 'wrencode[agent-sdk]'   # or: pip install claude-agent-sdk
```

It always authenticates with `ANTHROPIC_API_KEY`, so usage bills to your
Console credits, including the monthly API credits that come with Max and Team
plans. Subscription logins are never used. Multi-workspace keys need
`ANTHROPIC_WORKSPACE_ID`. The conversation resumes per project across restarts;
`/clear` starts a new one. Your `~/.claude` hooks, plugins and MCP servers are
not loaded; project instructions come from `AGENTS.md` / `CLAUDE.md`.

To run several prompts in parallel, each as its own agent with a fresh
context, see `examples/agent_sdk_swarm.py`. It prints every answer and the
total cost.

### OpenAI-compatible servers

`openai-compatible` talks to any server that implements OpenAI chat completions,
using native tool calls. Point it at the server with `OPENAI_COMPATIBLE_BASE_URL`
(default `http://localhost:8000/v1`). If the server serves exactly one model,
WrenCode uses it; otherwise set `MODEL`.

```bash
# vLLM (tool calling needs these flags; pick the parser for your model)
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
BACKEND=openai-compatible wrencode

# llama.cpp (--jinja enables tool calling)
llama-server -m qwen2.5-coder-7b-instruct-q4_k_m.gguf --jinja --port 8080
BACKEND=openai-compatible OPENAI_COMPATIBLE_BASE_URL=http://localhost:8080/v1 wrencode

# Hugging Face Inference Providers
BACKEND=openai-compatible OPENAI_COMPATIBLE_BASE_URL=https://router.huggingface.co/v1 \
  OPENAI_COMPATIBLE_API_KEY=$HF_TOKEN MODEL=Qwen/Qwen2.5-Coder-32B-Instruct wrencode
```

## Tools

The agent has access to seven tools:

- **read** - read a file with line numbers, or list a directory
- **write** - write content to a file
- **edit** - replace a unique string in a file. If the text only matches with
  its indentation shifted by a consistent amount (a common slip when quoting a
  method), the edit is applied with the replacement shifted to match; otherwise
  the error shows the closest lines in the file
- **glob** - find files by pattern, sorted by modification time
- **grep** - search files for a regex pattern using `rg` when available, falling back to `grep`
- **bash** - run a shell command with timeout and streaming output
- **task** - delegate a self-contained subtask to a fresh subagent (its own context, same tools) that returns only its final result

All file operations are sandboxed to the workspace root by default.

### Subagents

The `task` tool runs a nested agent loop on a fresh message history, so the
parent's context only grows by the returned summary — useful for context-heavy
subtasks.

Task calls made in the same reply run in parallel, up to
`WRENCODE_MAX_PARALLEL_SUBAGENTS` at a time (default 4), on every backend
except the in-process `mlx` and `transformers` ones. Ask for it in the prompt,
for example "analyze each file in docs/ with its own subagent, in parallel".
Each subagent's output is tagged `[1]`, `[2]`, and so on; approval prompts
take turns and pause the other agents' output until you answer. Escape stops
the whole batch. Recursion is capped by `WRENCODE_MAX_SUBAGENT_DEPTH` (default 2), and
each subagent round is bounded. For autonomous subagent runs, enable
`--yes` / `WRENCODE_AUTO_APPROVE` so sub-tool calls don't block on confirmation.

## Project instructions (AGENTS.md)

WrenCode reads [`AGENTS.md`](https://agents.md) files and adds them to the
system prompt, so conventions you've written for other agents apply here too.
It looks in `~/.wrencode/`, then in every directory from the git root down to
the workspace (outside a git repo, only the workspace). A directory without an
`AGENTS.md` falls back to `CLAUDE.md`. Files closer to the workspace come later
and take precedence. The total is capped at 32,000 characters, and the files
loaded are listed at startup.

## Headless mode

`-p` / `--print` runs a single prompt without the interactive UI, for scripts,
CI, and evals:

```bash
wrencode -p "Why is test_parse failing?"
git diff | wrencode -p "Review this diff"             # prompt from stdin
wrencode --yes -p "Fix the lint errors" --max-turns 20
wrencode -p "List the TODOs" --output-format json | jq -r .result
wrencode --yes -p "Make the tests pass" --verify "python3 -m unittest -q"
```

- stdout carries only the final answer (or one JSON object with
  `--output-format json`: `result`, `is_error`, `stop_reason`, `num_turns`,
  `backend`, `model`); progress and tool output go to stderr.
- Each run starts from a fresh history and doesn't touch the saved one.
- Without `--yes`, writes and shell commands are declined (the model is told
  why) instead of waiting for approval. Read-only tools always work.
- The exit code is `0` when the agent finishes, `1` if it errors, hits
  `--max-turns`, or stops on repeated tool errors, and `2` for bad arguments.
- `--verify CMD` checks the agent's claim of being done: WrenCode runs `CMD`
  in the workspace when the agent finishes, and if it fails, sends the output
  back and lets the agent continue (up to 3 attempts in all). The result says
  `verified: true/false`, and a final failure exits `1` with
  `stop_reason: "verify_failed"`. `--max-turns` applies to each attempt.

### Structured output

`--json-schema` makes the answer a JSON value that matches a schema, given as
a file or inline:

```bash
wrencode -p "Review this repo for bugs" --json-schema bugs.schema.json
wrencode -p "Is the build green?" --json-schema '{"type": "object", "properties": {"green": {"type": "boolean"}}, "required": ["green"]}'
```

The agent gets a `respond` tool whose arguments are your schema, and the run
ends when it calls `respond` with a valid answer. If the answer doesn't match,
the validation errors go back to the model so it can fix them; if it never
calls `respond`, the run fails with `stop_reason: "no_structured_output"`.
stdout is the JSON value (or, with `--output-format json`, the usual object
with a `structured_output` field). Validation is built in and covers the
common keywords: `type`, `enum`, `const`, `properties`, `required`,
`additionalProperties`, `items`, length and numeric bounds, `pattern`, and
`anyOf`/`oneOf`/`allOf`.

## Synthesize

`wrencode synthesize` fuses several agent chat transcripts into one document: a
semantic git-merge for conversations. Each transcript is normalized to user and
assistant turns, the model extracts its decisions, problems solved, files touched
and open questions, and those fact sets are then reconciled across chats. Every
claim cites the chat it came from, by the first eight characters of the file name.

```bash
wrencode synthesize                          # pick from this project's Claude Code history
wrencode synthesize a.jsonl b.jsonl          # fuse these transcripts
wrencode synthesize ~/.claude/projects/-home-me-app --all   # a whole directory, no picker
wrencode synthesize diff a.jsonl b.jsonl     # only where the chats diverge
wrencode synthesize log --out DECISIONS.md   # a decision timeline, written to a file
```

Three modes, chosen by the first word after `synthesize`:

- **merge** (default) — `Reinforced decisions` that two or more chats agree on,
  `Unique contributions`, `⚠ Conflicts` where chats contradict each other (a
  later chat that overrode an earlier one is marked resolved), and
  `Open questions`.
- **diff** — only the divergences: conflicts and what appears in just one chat,
  like `git diff`.
- **log** — one chronological timeline of decisions, oldest first, noting where
  a later chat supersedes an earlier one, and ending with the net state.

Inputs can be files, directories (every `*.jsonl` inside, newest first), or
nothing, which lists this project's Claude Code transcripts from
`~/.claude/projects/`. With a directory or no paths, an interactive picker lets
you choose: `↑`/`↓` move, space toggles, `a` selects all, Enter confirms, Esc
cancels. `--all` skips the picker. Transcripts are recognized in three formats:
Claude Code JSONL (tool calls, tool results and thinking are dropped), generic
JSONL message logs (Codex/OpenAI style), and a JSON list of messages or a
`{"messages": [...]}` object. Anything else is read as one block of text. The
synthesis uses the configured backend; local models are loaded on demand. With a
single transcript the result degrades to a structured summary.

## Context management

Long sessions are compacted automatically. Before each model call WrenCode
estimates the prompt size (about 4 characters per token), and once it passes
`WRENCODE_COMPACT_AT` (default 75%) of `WRENCODE_CONTEXT_TOKENS` (default
128,000) it has the model summarize the older messages: the request, files
touched, commands and results, decisions, and what's left to do. The most recent
messages, about a quarter of the window, are kept verbatim, along with the
user's latest request, so it works mid-task, between tool calls. If a request still
fails with a context-length error, WrenCode compacts and retries once.

Set `WRENCODE_CONTEXT_TOKENS` to your model's window, especially for local
models with small ones. `/compact` summarizes on demand.

## Installation

### Option 1: Standalone binary (recommended)

Run the guided installer:

```bash
curl -fsSL https://raw.githubusercontent.com/almostly/wrencode/main/install.sh | sh
```

It detects your OS/arch, downloads the matching binary from the latest GitHub
Release, and installs it to `~/.local/bin` (no sudo). Override the location with
`WRENCODE_INSTALL_DIR`, or pin a release with `WRENCODE_VERSION`:

```bash
curl -fsSL https://raw.githubusercontent.com/almostly/wrencode/main/install.sh \
  | WRENCODE_INSTALL_DIR=/usr/local/bin WRENCODE_VERSION=0.1.3 sh
```

Or download and run it locally:

```bash
curl -fsSL https://raw.githubusercontent.com/almostly/wrencode/main/install.sh -o install.sh
chmod +x install.sh
./install.sh
```

Manual install (fallback): download the right binary from GitHub Releases, make it executable, and move it into your `PATH`.

Windows: download `wrencode-windows-x64.exe` from GitHub Releases and put it on your `PATH`.

macOS Apple Silicon:

```bash
curl -L https://github.com/almostly/wrencode/releases/latest/download/wrencode-macos-arm64 -o wrencode
chmod +x wrencode
sudo mv wrencode /usr/local/bin/wrencode
```

macOS Intel:

```bash
curl -L https://github.com/almostly/wrencode/releases/latest/download/wrencode-macos-x64 -o wrencode
chmod +x wrencode
sudo mv wrencode /usr/local/bin/wrencode
```

Linux x64:

```bash
curl -L https://github.com/almostly/wrencode/releases/latest/download/wrencode-linux-x64 -o wrencode
chmod +x wrencode
sudo mv wrencode /usr/local/bin/wrencode
```

### Option 2: Run from source

Standard library only (except the backend you choose). `wrencode.py` runs with the
`wrencode_*.py` modules next to it.

```bash
git clone https://github.com/almostly/wrencode
cd wrencode
```

For MLX (Mac Silicon):

```bash
pip install mlx-lm
```

For Anthropic:

```bash
pip install anthropic  # not required - uses urllib directly
export ANTHROPIC_API_KEY=your_key
```

For OpenAI:

```bash
export OPENAI_API_KEY=your_key
```

For OpenRouter:

```bash
export OPENROUTER_API_KEY=your_key
```

For NanoGPT:

```bash
export NANOGPT_API_KEY=your_key
```

For HuggingFace Transformers:

```bash
pip install transformers torch
```

## Usage

```bash
# Standalone binary — prompts for a backend on first run
wrencode

# Re-pick the backend at any time
wrencode --configure

# Or from source — also prompts on first run
python3 wrencode.py

# Anthropic Claude (model list is fetched live from the API during /configure)
BACKEND=anthropic python3 wrencode.py
# Multi-workspace Anthropic keys also need a workspace id:
# ANTHROPIC_WORKSPACE_ID=wrkspc_... BACKEND=anthropic python3 wrencode.py

# OpenAI (model list fetched live from the API during /configure)
BACKEND=openai MODEL=gpt-4o python3 wrencode.py

# OpenRouter
BACKEND=openrouter MODEL=anthropic/claude-3-haiku python3 wrencode.py

# NanoGPT
BACKEND=nanogpt MODEL=z-ai/glm-5.3-flash-uncensored python3 wrencode.py

# Ollama (needs `ollama serve` running and the model pulled)
BACKEND=ollama MODEL=llama3.2 python3 wrencode.py

# HuggingFace model
BACKEND=transformers MODEL=deburky/gpt-oss-claude-code python3 wrencode.py

# Local proxy
BACKEND=local LOCAL_PORT=8082 python3 wrencode.py
```

## Developing

```bash
python3 -m unittest -q test_wrencode   # the test suite (stdlib unittest)
uvx ruff check . && uvx ruff format .  # lint and format; the rule set is in pyproject.toml
uvx ty check wrencode*.py              # type check
```

A deliberate catch-all `except Exception` carries a `# noqa: BLE001` with its reason;
everything else is kept clean under the pinned rules.

## Releasing

Versions and [`CHANGELOG.md`](CHANGELOG.md) are managed with
[commitizen](https://commitizen-tools.github.io/commitizen/), so write commit
messages as [conventional commits](https://www.conventionalcommits.org/)
(`feat: ...`, `fix(edit): ...`, `refactor: ...`). To cut a release:

```bash
uvx --from commitizen cz bump      # bumps WRENCODE_VERSION, updates CHANGELOG.md, tags
git push origin main --tags
```

Preview the next changelog entry with `uvx --from commitizen cz changelog --dry-run`.

Binaries are built automatically by GitHub Actions when a version tag is pushed.

This publishes release assets:
- `wrencode-linux-x64`
- `wrencode-macos-x64`
- `wrencode-macos-arm64`
- `wrencode-windows-x64.exe`

## Slash Commands

|Command       |Description                                   |
|--------------|----------------------------------------------|
|`/help`       |Show available commands                       |
|`/model`      |Switch model, or `/model <id>` to set it directly|
|`/backend`, `/configure`|Switch backend, model and API key   |
|`/clear` or `/c`|Clear conversation history                  |
|`/compact`    |Summarize history to reduce context          |
|`/quit`, `/q` or `/exit`|Quit                                |

Type `/` to see matching commands: ↑↓ pick, Tab completes, Enter runs.

## Environment Variables

|Variable                     |Default                |Description                       |
|-----------------------------|-----------------------|----------------------------------|
|`BACKEND`                    |chooser/saved config   |Override the saved inference backend|
|`MODEL`                      |backend-dependent      |Model path or ID                  |
|`WRENCODE_CONFIG_DIR`        |`~/.wrencode`          |Dir for `config.json` (saved backend/key)|
|`WRENCODE_WORKSPACE`         |cwd                    |Root directory for file operations|
|`WRENCODE_HISTORY_FILE`      |`~/.wrencode/history.json`|Conversation history file path |
|`WRENCODE_UNRESTRICTED_PATHS`|`0`                    |Allow paths outside workspace     |
|`WRENCODE_AUTO_APPROVE`      |`0`                    |Skip y/N confirmation for writes/commands (headless; also `--yes`)|
|`WRENCODE_MAX_SUBAGENT_DEPTH`|`2`                    |Max nested subagent recursion depth (`task` tool)|
|`WRENCODE_MAX_PARALLEL_SUBAGENTS`|`4`                |Subagents run at once from one reply; `1` runs them in order|
|`MAX_TOKENS`                 |`8192`, `16000` for Claude|Max tokens per response        |
|`WRENCODE_EFFORT`            |-                      |Claude reasoning effort: `low`, `medium`, `high`, `xhigh`, `max`|
|`WRENCODE_HTTP_TIMEOUT`      |`600`                  |Seconds to wait for a model response|
|`WRENCODE_HTTP_RETRIES`      |`2`                    |Retries on HTTP 429/5xx and network errors, with backoff|
|`WRENCODE_CONTEXT_TOKENS`    |`128000`               |Model context window, for auto-compaction|
|`WRENCODE_COMPACT_AT`        |`0.75`                 |Compact at this fraction of the window (`0` disables)|
|`MAX_READ_BYTES`             |`4MB`                  |Max file size to read             |
|`MAX_READ_LINES`             |`800`                  |Max lines returned per read       |
|`GREP_MAX_MATCHES`           |`80`                   |Max grep results                  |
|`BASH_TIMEOUT`               |`120`                  |Shell command timeout in seconds  |
|`MAX_TOOL_OUTPUT_CHARS`      |`48000`                |Max tool output before truncation |
|`GLOB_SKIP_DIRS`             |`.git,node_modules,...`|Directories to skip in glob       |
|`OPENROUTER_API_KEY`         |-                      |OpenRouter API key                |
|`NANOGPT_API_KEY`            |-                      |NanoGPT API key                   |
|`OPENAI_API_KEY`             |-                      |OpenAI API key                    |
|`ANTHROPIC_API_KEY`          |-                      |Anthropic API key                 |
|`ANTHROPIC_WORKSPACE_ID`     |-                      |Anthropic workspace id (`wrkspc_…`); required for multi-workspace keys|
|`LOCAL_API_KEY`              |`local`                |Local proxy API key               |
|`LOCAL_PORT`                 |`8082`                 |Local proxy port                  |
|`OLLAMA_HOST`                |`http://localhost:11434`|Ollama server base URL           |
|`OPENAI_COMPATIBLE_BASE_URL` |`http://localhost:8000/v1`|OpenAI-compatible server base URL|
|`OPENAI_COMPATIBLE_API_KEY`  |-                      |Key for that server, if it needs one|

## History

Conversation history is persisted to `~/.wrencode/history.json` by default. It is restored automatically on next launch.

To override the history file location, set `WRENCODE_HISTORY_FILE` to a custom path.

To clear history: use `/c` in the session, or delete `~/.wrencode/history.json` (or your override path).

## License

MIT - Copyright 2026 Almostly.

-----

<p align="center">
  <img src="assets/almostly-badge.svg" alt="Almostly" />
</p>
