# 🐦 WrenCode

A minimal agentic coding assistant in a single Python file.

Named after Harold Wren - the alias of a genius who built a superintelligent AI and operated quietly in the background.

-----

## What it is

WrenCode is a lightweight alternative to Claude Code. It runs a tool-calling agent loop locally or via API, giving an LLM the ability to read, write, and edit files, search codebases, and run shell commands - enough to autonomously navigate and modify a real project.

Where Claude Code is the batteries-included tool, WrenCode is the **"understand and own your agent" tool**: the entire agent loop fits in one readable file, runs against local or hosted models, and is yours to hack.

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
|`ollama`      |Local models via a running `ollama serve`|binary + source     |
|`local`       |Local proxy via Anthropic-compatible API|binary + source       |
|`transformers`|HuggingFace Transformers (CPU/MPS/GPU)  |source install only   |
|`mlx`         |Apple Silicon via MLX                   |source install, macOS |

The standalone binary can't bundle the heavy ML stack, so the local-weights
backends (`mlx`, `transformers`) are only offered when running from source.

## Tools

The agent has access to six tools:

- **read** - read a file with line numbers, or list a directory
- **write** - write content to a file
- **edit** - replace a unique string in a file
- **glob** - find files by pattern, sorted by modification time
- **grep** - search files for a regex pattern using `rg` when available, falling back to `grep`
- **bash** - run a shell command with timeout and streaming output

All file operations are sandboxed to the workspace root by default.

## Installation

### Option 1: Standalone binary (recommended)

Run the guided installer:

```bash
curl -fsSL https://raw.githubusercontent.com/deburky/wrencode/main/install.sh | sh
```

It detects your OS/arch, downloads the matching binary from the latest GitHub
Release, and installs it to `~/.local/bin` (no sudo). Override the location with
`WRENCODE_INSTALL_DIR`, or pin a release with `WRENCODE_VERSION`:

```bash
curl -fsSL https://raw.githubusercontent.com/deburky/wrencode/main/install.sh \
  | WRENCODE_INSTALL_DIR=/usr/local/bin WRENCODE_VERSION=0.1.3 sh
```

Or download and run it locally:

```bash
curl -fsSL https://raw.githubusercontent.com/deburky/wrencode/main/install.sh -o install.sh
chmod +x install.sh
./install.sh
```

Manual install (fallback): download the right binary from GitHub Releases, make it executable, and move it into your `PATH`.

macOS Apple Silicon:

```bash
curl -L https://github.com/deburky/wrencode/releases/latest/download/wrencode-macos-arm64 -o wrencode
chmod +x wrencode
sudo mv wrencode /usr/local/bin/wrencode
```

macOS Intel:

```bash
curl -L https://github.com/deburky/wrencode/releases/latest/download/wrencode-macos-x64 -o wrencode
chmod +x wrencode
sudo mv wrencode /usr/local/bin/wrencode
```

Linux x64:

```bash
curl -L https://github.com/deburky/wrencode/releases/latest/download/wrencode-linux-x64 -o wrencode
chmod +x wrencode
sudo mv wrencode /usr/local/bin/wrencode
```

### Option 2: Run from source

Single file, standard library only (except the backend you choose).

```bash
git clone https://github.com/deburky/wrencode
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

# Ollama (needs `ollama serve` running and the model pulled)
BACKEND=ollama MODEL=llama3.2 python3 wrencode.py

# HuggingFace model
BACKEND=transformers MODEL=deburky/gpt-oss-claude-code python3 wrencode.py

# Local proxy
BACKEND=local LOCAL_PORT=8082 python3 wrencode.py
```

## Releasing binaries

Binaries are built automatically by GitHub Actions when you push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This publishes release assets:
- `wrencode-linux-x64`
- `wrencode-macos-x64`
- `wrencode-macos-arm64`

## Slash Commands

|Command       |Description                                   |
|--------------|----------------------------------------------|
|`/help`       |Show available commands                       |
|`/c`          |Clear conversation history                    |
|`/compact`    |Summarize history to reduce context (mlx only)|
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
|`MAX_TOKENS`                 |`4096`                 |Max tokens per response           |
|`MAX_READ_BYTES`             |`4MB`                  |Max file size to read             |
|`MAX_READ_LINES`             |`800`                  |Max lines returned per read       |
|`GREP_MAX_MATCHES`           |`80`                   |Max grep results                  |
|`BASH_TIMEOUT`               |`120`                  |Shell command timeout in seconds  |
|`MAX_TOOL_OUTPUT_CHARS`      |`48000`                |Max tool output before truncation |
|`GLOB_SKIP_DIRS`             |`.git,node_modules,...`|Directories to skip in glob       |
|`OPENROUTER_API_KEY`         |-                      |OpenRouter API key                |
|`OPENAI_API_KEY`             |-                      |OpenAI API key                    |
|`ANTHROPIC_API_KEY`          |-                      |Anthropic API key                 |
|`LOCAL_API_KEY`              |`local`                |Local proxy API key               |
|`LOCAL_PORT`                 |`8082`                 |Local proxy port                  |
|`OLLAMA_HOST`                |`http://localhost:11434`|Ollama server base URL           |

## History

Conversation history is persisted to `~/.wrencode/history.json` by default. It is restored automatically on next launch.

To override the history file location, set `WRENCODE_HISTORY_FILE` to a custom path.

To clear history: use `/c` in the session, or delete `~/.wrencode/history.json` (or your override path).

## License

MIT - Copyright 2026 Denis Burakov.
