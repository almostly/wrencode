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
|`wrencode_sandbox.py`   |The `python` tool's sandbox, on pydantic-monty                    |
|`wrencode_history.py`   |Conversation history in Postgres: a server or embedded PGlite     |
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
- **python** - run a snippet of Python in a sandbox, with `pydantic-monty` installed (see below)

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

### Python sandbox

With [pydantic-monty](https://github.com/pydantic/monty) installed, the agent gets
an eighth tool, `python`. It runs a snippet of model-written code in Monty, a
Python interpreter built as a sandbox: each call is a fresh interpreter with no
network, no shell, no environment variables, and a read-only view of the
workspace at `/workspace`, which is also the working directory, so
`open("foo.py")` works. wrencode's `read(path)`, `glob(pat)` and `grep(pat)` are
callable inside the snippet. Printed output and the value of a trailing
expression come back to the model; an exception comes back as its traceback.
Since the snippet can't change anything, it runs without an approval prompt;
changes still go through `write` and `edit`.

```bash
pip install 'wrencode[sandbox]'   # or: pip install pydantic-monty
```

Monty runs a subset of Python: no class inheritance, generators or third-party
packages, and a curated standard library (`json`, `re`, `math`, `datetime`,
`pathlib`, ...). `WRENCODE_SANDBOX_TIMEOUT` (default 30 seconds) and
`WRENCODE_SANDBOX_MEMORY_MB` (default 256) bound each run. The standalone binary
doesn't bundle Monty, so the tool is a source-install feature.

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
claim cites the chat it came from: the first eight characters of the file name,
or more when two names would clash, so ids are unique within a run.

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
`{"messages": [...]}` object. Anything else is read as one block of text.

The synthesis uses the configured backend at temperature 0; local models are
loaded on demand. A transcript longer than the model's context window
(`WRENCODE_CONTEXT_TOKENS`) is sent with its start and, mostly, its end, since the
latest decisions override earlier ones. A chat whose extraction doesn't come back
as JSON is reported and contributes nothing. With a single transcript the result
degrades to a structured summary.

## Token usage and spend

After each turn wrencode prints one dim line with what the backend reported:

```
↑ 1.6k  ↓ 108  42 tok/s ↗  ▰▱▱▱▱▱▱▱▱▱ 1%  $0.0042 · total $0.21
```

Up is the turn's input tokens, down its output, then the speed: output tokens
per second over the turn's requests, with an arrow against the previous turn
(`↗` at least a tenth faster, `↘` a tenth slower, `→` about the same). The rate is measured over the whole request, so the network and the wait
for the first token are in it; a drop usually means the provider is busy. The
meter is the context fill: the size of the latest request against the model's
window, which is what the next turn starts from and what auto-compaction watches.
It turns yellow at the compaction threshold. Then the turn's cost and, after the
first turn, the session total. The terminal tab title shows the context fill, the
session totals, the spend and the speed. `/usage` prints the full numbers:

```
             input  cached written  output calls      cost
this turn     1.6k    1.6k      54     108     1   $0.0042
session       4.7k    3.9k    1.6k     236     3    $0.213
context: 1.6k of 128k (1%); input = uncached + cached (read) + written
price: $2/$10 per MTok (cache read $0.20, write $2.50; built-in)
speed: 42 tok/s this turn (↗ from 36 tok/s last turn); 39 tok/s this session; output tokens over the request's wall time
```

While you type, a dim line under the prompt estimates what sending the message
will cost in input tokens: the context the model already holds (at the cache-read
rate for the part it served from the cache last time), its last reply, and your
text at about four characters per token. Output can't be known ahead, so it isn't
counted.

```
❯ fix the bug in the parser
  ≈ $0.0031 input
```

**Where the prices come from.** Each call is priced at the four rates in effect
for the model: input, output, cache read and cache write, in USD per million
tokens. Anthropic's Models API lists models but not prices, so Claude models (and
the OpenAI, Amazon Nova and Bedrock ids wrencode knows) come from a built-in
table, checked October 2026; `/usage` says `built-in` and the startup line shows
the rate. OpenRouter and NanoGPT list prices in their model catalogs, and
`/configure` saves those to `~/.wrencode/prices.json` when it fetches the model
list, keyed `backend/model id`; you can add your own entries there as
`[input, output, cache_read, cache_write]`. `WRENCODE_PRICE=input,output[,cache_read[,cache_write]]`
overrides everything for the current model (`WRENCODE_PRICE=0,0` marks a local
model free). Without a known price nothing is shown, `/usage` prints `$?` and how
to set one, and no estimate appears while typing. Claude Haiku 5.5's higher rate
card above 100K-token prompts is applied per call. The model picker in
`/configure` and `/model` shows the price beside each model it knows.

Headless runs print the line to stderr and add a `usage` object to the
`--output-format json` result, with `cost_usd` when every call had a known
price and `output_tokens_per_second` for the run. Set `WRENCODE_SHOW_USAGE=0` to turn the line and the typing estimate off.
Backends that report no usage (local models, the local proxy) print nothing.

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

## Security

wrencode reads untrusted files and runs commands on your machine, so the goal is
narrower than "can't be attacked": nothing changes outside the sandbox without you
seeing and approving the real action, and opening an untrusted repository is safe.

- **Approvals show what will run.** Every write, edit and shell command asks first.
  Control characters and escape sequences in a command, a file or a model reply are
  displayed as `^[`, `^M` and so on, never interpreted, so nothing can redraw the
  screen or hide part of a command. Writes under a hidden path (`.git/hooks`,
  `.github/workflows`, dotfiles) are flagged. `--yes` turns the prompts off; use it
  only in a sandbox you can throw away.
- **File tools stay in the workspace.** Paths are resolved (symlinks followed) and
  must land inside the workspace root unless `WRENCODE_UNRESTRICTED_PATHS=1`.
  `grep` passes the pattern and path as arguments, never as flags.
- **A project's `.env` can't reconfigure the agent.** It may set `*_API_KEY` and
  `ANTHROPIC_WORKSPACE_ID` only. The backend, any server URL, auto-approve, and the
  config, history and workspace locations come from your shell or the `.env` beside
  `wrencode.py`; names a project `.env` tried to set are reported at startup.
- **`AGENTS.md` / `CLAUDE.md` are prompt input.** A repository's instructions go into
  the system prompt by design, which means a repository can steer the agent. The
  approval prompts are the control; the files loaded are listed at startup.
- **Keys go only to their backend.** Fixed hosts for Anthropic, OpenAI, OpenRouter,
  NanoGPT and Bedrock; the URL you configured for openai-compatible, Ollama and the
  local proxy. Saved keys, the conversation history and model caches are owner-only
  files (`0600`) under `~/.wrencode`; the embedded Postgres data and socket
  directories are owner-only (`0700`). The Agent SDK backend runs with the API key
  only, subscription credentials blanked.
- **The `python` tool is sandboxed** in pydantic-monty: no network, shell or
  environment, a read-only workspace, and time and memory limits.
- **Releases are verifiable.** Each binary is published with a SHA-256 checksum that
  `install.sh` checks. Hosted backends are reached over TLS with certificate
  verification. wrencode has no runtime dependencies beyond the standard library.

Please report security issues privately through the repository's GitHub security
advisories rather than in a public issue.

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

For the `python` sandbox tool:

```bash
pip install pydantic-monty
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

There are no `# noqa` markers: a rule the design contradicts is turned off in
`pyproject.toml` with its reason, and a test keeps it that way.

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
|`/clear` or `/c`|Clear the conversation (a new session with Postgres history)|
|`/sessions`   |List this project's conversations (Postgres history)|
|`/resume [id]`|Continue an earlier session: a picker, or by id|
|`/search <text>`|Find past sessions by their words, then `/resume` one|
|`/sync`       |Copy this project's history to the mirror now (Postgres history)|
|`/usage`      |Token usage and spend for this turn and the session, and the price in effect|
|`/compact`    |Summarize history to reduce context          |
|`/quit`, `/q` or `/exit`|Quit                                |

Type `/` to see matching commands: ↑↓ pick, Tab completes, Enter runs.

## In the terminal

What a session looks like, and the keys that drive it.

- **Who is speaking.** Your line keeps the `❯` prompt. The model's reply opens
  with a cyan dot; a tool call opens with a green dot, and what it returned
  hangs under it after `⎿`: one line when that says it all (`ok`, `3 lines
  read`, an error), a short excerpt otherwise. Code the model quotes is drawn
  in its own tint with a gutter.
- **Approvals.** Before a write, edit or shell command runs, you see exactly
  what it does: a unified diff of the file with a few lines of context, red
  and green, headed by the file and line; then one question, `Apply to
  app.py?  Enter yes · a always · n no`. `n` asks what to do differently.
- **Waiting.** The spinner says what is happening (`thinking`, `running
  python`), how long it has been, and that Escape cancels the turn. On the
  Anthropic and OpenAI-style backends the reply then streams in as the model
  writes it, rendered line by line; Escape stops it mid-sentence. The usage
  line still counts the whole request. `WRENCODE_STREAM=0` waits for whole
  replies instead; headless runs and Bedrock always do.
- **The prompt.** `←` `→` move, `Home`/`End` or `Ctrl-A`/`Ctrl-E` jump,
  `Ctrl-W` deletes the word before the cursor, `Ctrl-U` to the start of the
  line, `Ctrl-K` to the end, `↑`/`↓` walk the input history. While you type,
  a dim line estimates the input cost of sending the message.
- **Pickers.** Models and sessions are picked with `↑`/`↓` and Enter; Escape
  cancels, a number jumps. Without a terminal they fall back to a numbered
  prompt.
- **Errors.** A failed request is reported as a sentence and a next step
  (`The API key was rejected. Run /configure to enter a new one.`), with the
  raw message under it; `WRENCODE_DEBUG=1` prints it in full.
- **Light terminals.** The prose and code tints are chosen for a dark
  background. Set `WRENCODE_THEME=light` on a light one (terminals that
  export `COLORFGBG` are detected).

## Environment Variables

|Variable                     |Default                |Description                       |
|-----------------------------|-----------------------|----------------------------------|
|`BACKEND`                    |chooser/saved config   |Override the saved inference backend|
|`MODEL`                      |backend-dependent      |Model path or ID                  |
|`WRENCODE_CONFIG_DIR`        |`~/.wrencode`          |Dir for `config.json` (saved backend/key)|
|`WRENCODE_WORKSPACE`         |cwd                    |Root directory for file operations|
|`WRENCODE_HISTORY_FILE`      |`<config dir>/history.json`|Conversation history file, without the Postgres store|
|`WRENCODE_DATABASE_URL`      |-                      |Postgres URL for history; unset, embedded PGlite is used|
|`WRENCODE_MIRROR_URL`        |-                      |A second Postgres that receives a copy of every saved session|
|`WRENCODE_PGLITE_START_TIMEOUT`|`60`                 |Seconds to wait for the embedded PGlite to start|
|`WRENCODE_UNRESTRICTED_PATHS`|`0`                    |Allow paths outside workspace     |
|`WRENCODE_AUTO_APPROVE`      |`0`                    |Skip y/N confirmation for writes/commands (headless; also `--yes`)|
|`WRENCODE_MAX_SUBAGENT_DEPTH`|`2`                    |Max nested subagent recursion depth (`task` tool)|
|`WRENCODE_MAX_PARALLEL_SUBAGENTS`|`4`                |Subagents run at once from one reply; `1` runs them in order|
|`WRENCODE_SANDBOX_TIMEOUT`   |`30`                   |Seconds a `python` tool snippet may run|
|`WRENCODE_SANDBOX_MEMORY_MB` |`256`                  |Memory a `python` tool snippet may use|
|`MAX_TOKENS`                 |`8192`, `16000` for Claude|Max tokens per response        |
|`WRENCODE_EFFORT`            |-                      |Claude reasoning effort: `low`, `medium`, `high`, `xhigh`, `max`|
|`WRENCODE_SHOW_USAGE`        |`1`                    |Print the usage line after each turn and the cost estimate while typing|
|`WRENCODE_THEME`             |auto                   |`light` or `dark`: picks the prose and code tints (auto reads `COLORFGBG`)|
|`WRENCODE_STREAM`            |`1`                    |Stream replies as they are written (Anthropic and OpenAI-style backends)|
|`WRENCODE_PRICE`             |-                      |Price of the current model, USD per million tokens: `input,output[,cache_read[,cache_write]]`|
|`WRENCODE_HTTP_TIMEOUT`      |`600`                  |Seconds to wait for a model response|
|`WRENCODE_HTTP_RETRIES`      |`2`                    |Retries on HTTP 429/5xx and network errors, with backoff|
|`WRENCODE_CONTEXT_TOKENS`    |`128000`               |Model context window, for auto-compaction and `synthesize`|
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

By default the conversation is saved to `~/.wrencode/history.json` and restored on
the next launch (`WRENCODE_HISTORY_FILE` moves it; `/c` clears it).

With the `history` extra, conversations live in Postgres instead, as sessions per
project: wrencode resumes the project's latest session on launch, `/clear` starts a
new one and keeps the old, `/sessions` lists them, `/resume` picks one to continue (or `/resume <id>`) and
`/search <text>` looks inside all of them (Postgres full-text search).

```bash
pip install 'wrencode[history]'   # psycopg; Node.js is needed for the embedded engine
```

Two engines, both real Postgres:

- **Embedded PGlite** (the default): Postgres compiled to WebAssembly, run by Node.js
  with a persistent data directory under `~/.wrencode/pglite`. The first launch runs
  `npm install` there for the pinned `@electric-sql/pglite` packages; after that it
  starts in about a second, serves a Unix socket in an owner-only directory, and
  stops when wrencode exits. It is a single-user database: one wrencode at a time
  holds it, and a second one started meanwhile says so and uses `history.json`
  for that run.
- **A Postgres server**: set `WRENCODE_DATABASE_URL=postgres://user:pass@host/db`
  and the same schema is created there.

Each session stores its message list exactly as the backend format needs it
(JSONB), replaced whole on every save, so switching backends mid-history behaves as
it always has. Headless runs (`-p`) never read or write history.

### Mirroring to another Postgres

PGlite has a real write-ahead log but, as a single-user engine, no replication
protocol: nothing can subscribe to it. wrencode replicates at the application level
instead, which is exact because it owns every write and saves each session whole.
Set `WRENCODE_MIRROR_URL=postgres://user:pass@host/db` and every saved session is
copied there, matched by a stable session uid, from a background thread so a slow or
unreachable mirror never holds up the loop. Only the latest snapshot per session is
kept pending; an outage is reported once, retried with backoff, and `/sync` queues
the whole project's history again and waits for it. The mirror has the same schema,
so it can serve as `WRENCODE_DATABASE_URL` for another machine. Like the other
settings that steer wrencode, the mirror URL is read from your shell, never from a
project's `.env`.

## License

MIT - Copyright 2026 Almostly.

-----

<p align="center">
  <img src="assets/almostly-badge.svg" alt="Almostly" />
</p>
