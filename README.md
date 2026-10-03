# 🐦 WrenCode

A minimal agent harness for coding, in a single Python file.

Named after Harold Wren - the alias of a genius who built a superintelligent AI and operated quietly in the background.

-----

## What it is

WrenCode is a coding agent harness: everything around the model that turns it into an agent. It runs the tool-calling loop, executes tools, builds the system prompt, and manages context, locally or via API, giving an LLM the ability to read, write, and edit files, search codebases, and run shell commands - enough to autonomously navigate and modify a real project.

Where Claude Code is the batteries-included harness, WrenCode is the **"understand and own your agent" harness**: the entire agent loop fits in one readable file, runs against local or hosted models, and is yours to hack.

## Backends

On first run WrenCode asks you to pick a backend and saves the choice to
`~/.wrencode/config.json`. Run `wrencode --configure` any time to change it.
Set `BACKEND` (and the matching API key) in the environment to override the
saved choice, e.g. for CI.

|Backend       |Description                             |Availability          |
|--------------|----------------------------------------|----------------------|
|`anthropic`   |Claude via Anthropic API                |binary + source       |
|`openai`      |GPT models via OpenAI API               |binary + source       |
|`openrouter`  |Any model via OpenRouter                |binary + source       |
|`nanogpt`     |Any model via NanoGPT                   |binary + source       |
|`ollama`      |Local models via a running `ollama serve`|binary + source     |
|`openai-compatible`|vLLM, llama.cpp, Hugging Face, any OpenAI-compatible server|binary + source|
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
- **edit** - replace a unique string in a file
- **glob** - find files by pattern, sorted by modification time
- **grep** - search files for a regex pattern using `rg` when available, falling back to `grep`
- **bash** - run a shell command with timeout and streaming output
- **task** - delegate a self-contained subtask to a fresh subagent (its own context, same tools) that returns only its final result

All file operations are sandboxed to the workspace root by default.

### Subagents

The `task` tool runs a nested agent loop on a fresh message history, so the
parent's context only grows by the returned summary — useful for context-heavy
subtasks. Recursion is capped by `WRENCODE_MAX_SUBAGENT_DEPTH` (default 2), and
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
```

- stdout carries only the final answer (or one JSON object with
  `--output-format json`: `result`, `is_error`, `stop_reason`, `num_turns`,
  `backend`, `model`); progress and tool output go to stderr.
- Each run starts from a fresh history and doesn't touch the saved one.
- Without `--yes`, writes and shell commands are declined (the model is told
  why) instead of waiting for approval. Read-only tools always work.
- The exit code is `0` when the agent finishes, `1` if it errors, hits
  `--max-turns`, or stops on repeated tool errors, and `2` for bad arguments.

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

Single file, standard library only (except the backend you choose).

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

# Anthropic Claude
BACKEND=anthropic python3 wrencode.py

# OpenAI
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

## Slash Commands

|Command       |Description                                   |
|--------------|----------------------------------------------|
|`/help`       |Show available commands                       |
|`/c`          |Clear conversation history                    |
|`/compact`    |Summarize history to reduce context          |
|`/q` or `exit`|Quit                                          |

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
|`MAX_TOKENS`                 |`8192`                 |Max tokens per response           |
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
