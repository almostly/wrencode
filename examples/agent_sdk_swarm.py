"""Run several prompts in parallel through WrenCode's claude-agent-sdk backend.

Each prompt runs as its own headless ``wrencode -p`` process, so every agent gets
Claude Code's loop and tools with a fresh context, and all of them bill to the
same Anthropic API key. Results and the total cost are collected at the end.

Set it up once:

    pip install claude-agent-sdk
    export ANTHROPIC_API_KEY=sk-ant-...
    export ANTHROPIC_WORKSPACE_ID=wrkspc_...   # only for multi-workspace keys

Then pass prompts as arguments, or one per line in a file:

    python examples/agent_sdk_swarm.py "Summarize README.md" "List the TODOs in src/wrencode/app.py"
    python examples/agent_sdk_swarm.py --tasks tasks.txt --workers 4 --out results.json
    MODEL=claude-sonnet-5-5 WRENCODE_EFFORT=low python examples/agent_sdk_swarm.py ...

Without --yes the agents can read and search, but every edit or command that
needs approval is declined, which keeps parallel runs safe on one checkout.
Pass --yes only when the tasks touch different files, or run each task in its
own git worktree with --cwd.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import sys
import time

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"  # the wrencode package
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_task(prompt: str, args: argparse.Namespace) -> dict:
    """Run one prompt through ``wrencode -p`` and return its JSON result."""
    cmd = [sys.executable, "-m", "wrencode", "-p", prompt, "--output-format", "json"]
    if args.max_turns:
        cmd += ["--max-turns", str(args.max_turns)]
    if args.yes:
        cmd.append("--yes")
    env = {**os.environ, "BACKEND": "claude-agent-sdk", "PYTHONPATH": str(SRC)}
    started = time.monotonic()
    proc = subprocess.run(  # fixed argv, no shell
        cmd, cwd=args.cwd, env=env, capture_output=True, text=True, check=False
    )
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        result = {"result": "", "is_error": True, "error": "no JSON output"}
    result["prompt"] = prompt
    result["seconds"] = round(time.monotonic() - started, 1)
    if result.get("is_error"):
        # The agent's UI went to stderr; keep its tail to explain the failure.
        plain = ANSI.sub("", proc.stderr).strip()
        result["stderr_tail"] = plain.splitlines()[-5:]
    return result


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompts)
    if args.tasks:
        lines = pathlib.Path(args.tasks).read_text().splitlines()
        prompts += [line.strip() for line in lines if line.strip()]
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompts", nargs="*", help="prompts to run in parallel")
    parser.add_argument("--tasks", help="file with one prompt per line")
    parser.add_argument("--workers", type=int, default=3, help="agents at once")
    parser.add_argument("--max-turns", type=int, default=0, help="cap per agent")
    parser.add_argument("--cwd", default=None, help="directory the agents work in")
    parser.add_argument("--yes", action="store_true", help="auto-approve edits")
    parser.add_argument("--out", help="write all results to this JSON file")
    args = parser.parse_args()

    prompts = load_prompts(args)
    if not prompts:
        parser.error("give prompts as arguments or with --tasks")

    print(f"Running {len(prompts)} agents, {args.workers} at a time...")
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, p, args): p for p in prompts}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results.append(r)
            mark = "✗" if r.get("is_error") else "✓"
            cost = r.get("cost_usd", 0.0)
            print(f"{mark} ${cost:.4f} {r['seconds']}s  {r['prompt'][:70]}")

    order = {p: i for i, p in enumerate(prompts)}
    results.sort(key=lambda r: order[r["prompt"]])
    for r in results:
        print(f"\n## {r['prompt']}\n")
        print(r.get("result") or r.get("error", ""))
        for line in r.get("stderr_tail", []):
            print(f"| {line}")

    total = sum(r.get("cost_usd", 0.0) for r in results)
    failed = sum(1 for r in results if r.get("is_error"))
    print(f"\nTotal ${total:.4f} · {len(results) - failed} ok, {failed} failed")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
