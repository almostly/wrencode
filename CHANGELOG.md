## Unreleased

### Fix

- web search on the Anthropic backend sends the search tool version the model accepts: the current one on Opus and Sonnet 4.6 and later and Fable, the basic one on Haiku, which refused the current one with a 400
- colors are 24-bit only where the terminal takes them (`COLORTERM`, or iTerm2, WezTerm, Ghostty, VS Code's terminal; `WRENCODE_TRUECOLOR=1|0` overrides) and the nearest of 256 elsewhere, so Terminal.app and a plain tmux show the palette instead of dropping it

## 0.3.2 (2026-10-09)

### Feat

- **colors**: wrencode draws in Baseline (github.com/xRiskLab/vscode-themes), dark or light to match the terminal's background, which is read from `COLORFGBG` or asked of the terminal itself (when neither says, the first run asks once; `/theme light|dark|auto` changes the saved answer): prose in the editor text color, code the way the editor colors it (identifiers blue, keywords red, strings, calls and attributes orange, constants magenta, comments muted), the accent blue on the assistant's dot, the prompt and the banner. `WRENCODE_THEME=light|dark` forces a variant, `ansi` keeps the terminal's own colors, and a Zed theme file (`path.json#Name`) replaces the palette

### Fix

- replies were barely readable on a light terminal that does not say so: the prose was a light grey chosen for a dark background. The background is now asked of the terminal, and the light palette used on a light one

## 0.3.1 (2026-10-09)

### Feat

- **modules**: `wrencode.py` is the agent loop; backends, configure, ui, history, sandbox, permissions, mcp, web and synthesize are sibling modules with one-way dependencies
- **python**: a `python` tool runs model-written snippets in the pydantic-monty sandbox (no network, shell or environment; the workspace read-only) with `glob`, `read` and `grep` as functions; `pip install 'wrencode[sandbox]'`
- **history**: conversations in Postgres, a server via `WRENCODE_DATABASE_URL` or embedded PGlite; `/sessions`, `/resume` (a picker, or by id), `/search` finds sessions by their words, `/clear` starts a new one; `pip install 'wrencode[history]'`
- **history**: every saved session mirrored to a second Postgres (`WRENCODE_MIRROR_URL`) with a retry queue and `/sync`
- **usage**: a compact line after each turn: tokens up and down, output speed with an arrow against the previous turn, a context meter that warns at the compaction threshold, the turn's cost and the session total; `/usage` for the numbers; the terminal title carries the totals; headless JSON gains `usage` and `cost_usd`
- **pricing**: a built-in price table for Claude, GPT and Nova models, OpenRouter and NanoGPT catalog prices saved by `/configure`, `WRENCODE_PRICE` to override; the model picker shows prices; a cost estimate under the prompt while typing
- **streaming**: replies stream as the model writes them on the Anthropic and OpenAI-style backends, rendered line by line; Escape stops mid-sentence; `WRENCODE_STREAM=0` waits for whole replies
- **input**: messages can span lines: paste keeps line breaks, a backslash before Enter or Alt-Enter continues; `←` `→` Home End Ctrl-A/E/W/U/K and Delete edit in place; `↑` `↓` move between lines and through history
- **permissions**: rules `bash(git *)`, `edit(src/*)`, `write(.env)`, `mcp(server:tool)`, `fetch(host/*)` allow or deny without asking; deny wins over `a` and `--yes`; `s` at a prompt saves a rule; `/permissions` lists and edits them; a project's allow rules apply only after they are accepted once
- **mcp**: tools from MCP servers (`.wrencode/mcp.json`, Claude Code's `.mcp.json`, or `~/.wrencode/mcp.json`; stdio or Streamable HTTP) join the tool list as `mcp__server__tool`, with approval unless read-only; `/mcp` shows status; a project's servers start only after they are accepted once; stdlib only
- **web**: a `fetch(url)` tool on every backend reads a page as text with approval; Anthropic's server-side web search on the Anthropic backend, shown in the transcript and billed on the usage line (`WRENCODE_WEB_SEARCH=0` turns it off)
- **ui**: approvals show a unified diff with context and one question line; tool results fold to one line when that says it all; the spinner says what is happening, for how long, and that Escape cancels; one status line at startup; arrow-key pickers for models and sessions; errors as a sentence and a next step; the assistant's reply opens with a dot, code is tinted apart from prose; `WRENCODE_THEME=light` for light terminals
- **synthesize**: one shared completion primitive, unique chat ids, window-aware transcripts; documented
- **security**: a project `.env` may set only `*_API_KEY` and `ANTHROPIC_WORKSPACE_ID` (the names it set are reported); control characters, C1 controls and bidi overrides in commands, files, URLs and replies are shown, never interpreted; `grep` passes its pattern as an argument; writes under hidden paths are flagged on the resolved path; `glob` stays in the workspace; `fetch` reaches public addresses only and follows redirects on the same host only; MCP servers run without the backend keys, a project's without loader variables, and an HTTP server's headers never follow a redirect; releases ship SHA-256 checksums verified by install.sh; CI runs with read-only permissions

### Fix

- a bash rule must cover every command of a command line (`bash(git *)` no longer allows `git status && curl x | sh`), a prefix ends at a word, and a deny rule holds for read-only MCP tools too
- `s` at a prompt saves the rule for this project in your own file; editing the shared project file never accepts rules the repository put there; trust is recorded for the bytes that were shown
- Escape ends a streamed reply at once: nothing more is printed and the connection closes; Ctrl/Shift-arrow keys are read as arrows, not as typed text
- a reply whose text block stayed empty around a tool call goes back to the API without it; OpenAI-style servers that stream whole tool calls without an index get each call
- a streamed code line that wrapped is redrawn from its first row; `/mcp reload` stops the old servers; a server's `ping` is answered; a server still connecting at the deadline is reported, not half-registered; tools whose names clash are skipped with a note
- a fetched page without `</head>` is read in full; a gzip body is capped at the size limit; the fetch rule subject is the lower-cased host without a default port, with the query
- a mirror URL with a bad port falls back cleanly; the mirror's idle flag can't be set with a snapshot pending; the embedded Postgres socket directory in a shared temp dir must be an owner-only directory of yours

- the embedded Postgres works under a long config path (the socket moves to a short directory), and Node's error text loses its color codes
- `history.json` follows `WRENCODE_CONFIG_DIR`
- a shell command's output is not repeated under its call; a streamed line that wrapped is still rendered once complete; a server tool's whole input survives streaming
- the certifi test holds with a pre-set `SSL_CERT_FILE`; the bash tool closes its pipe
- no `noqa` markers: rules the design contradicts are off in pyproject with reasons, and a test enforces it

## 0.3.0 (2026-10-08)

### Feat

- **claude-agent-sdk**: new backend running Claude Code's agent loop and tools, billed to ANTHROPIC_API_KEY, with approvals, Escape to interrupt, per-turn cost and per-project session resume
- run task subagents from one reply in parallel (WRENCODE_MAX_PARALLEL_SUBAGENTS), with tagged output and approvals that take turns
- show matching slash commands while typing /, with arrow keys, Tab completion and highlighting of real commands only; /clear, /quit and /exit aliases
- **configure**: replace a key that comes from .env; the saved key keeps winning on later launches
- WRENCODE_EFFORT sets Claude reasoning effort
- **examples**: run several prompts in parallel on the claude-agent-sdk backend

### Fix

- fetch Anthropic and OpenAI model lists live, and send anthropic-workspace-id for multi-workspace Anthropic keys
- prompt caching covers the growing conversation, not only the system prompt and tools
- Claude backends default to claude-opus-5-5 with 16000 max tokens
- ty passes without ignore comments
- make wrencode.py executable to match its shebang
- retry dropped connections and timeouts, not just 429/5xx

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
