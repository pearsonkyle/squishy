"""CLI entry point. Runs the async agent under asyncio.run().
 
Shift+Tab rotates permission_mode live via prompt_toolkit, mirroring
Claude Code's in-session mode switching.
"""
 
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings

from squishy.agent import Agent
from squishy.client import Client
from squishy.config import Config
from squishy.display import Display, Stats
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.tools.base import Tool

MODE_COLORS = {"plan": "ansicyan", "edits": "ansigreen", "yolo": "ansimagenta"}
 
 
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="squishy", description="Minimal local-LLM coding agent.")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (env SQUISHY_BASE_URL)")
    p.add_argument("--model", help="Model id (env SQUISHY_MODEL)")
    p.add_argument("--api-key", help="API key (env SQUISHY_API_KEY)")
    p.add_argument("--max-turns", type=int, default=None, help="Turn cap (default 20)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None, help="Task timeout in seconds")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--plan", action="store_true", help="Start in plan mode (read-only)")
    p.add_argument("--edits", action="store_true", help="Start in edits mode (default)")
    p.add_argument("--yolo", action="store_true", help="Start in yolo mode (no prompts)")
    p.add_argument("--no-sandbox", action="store_true", help="Disable Docker sandbox for run_command")
    p.add_argument("--sandbox", action="store_true", help="Enable Docker sandbox for run_command")
    p.add_argument("--thinking", action="store_true", help="Allow <think> blocks")
    p.add_argument("--message", "-m", help="Non-interactive: send one message, print result, exit")
    p.add_argument("--init", action="store_true", help="Build .squishy/index.json before the REPL")
    p.add_argument("--no-summaries", action="store_true", help="When indexing, skip LLM summaries")
    p.add_argument("--index-concurrency", type=int, default=None, help="Parallel summary calls (default 4)")
    return p.parse_args(argv)
 
 
def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.api_key:
        cfg.api_key = args.api_key
    if args.max_turns is not None:
        cfg.max_turns = args.max_turns
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.plan:
        cfg.permission_mode = "plan"
    elif args.yolo:
        cfg.permission_mode = "yolo"
    elif args.edits:
        cfg.permission_mode = "edits"
    if args.no_sandbox:
        cfg.use_sandbox = False
    if args.sandbox:
        cfg.use_sandbox = True
    if args.thinking:
        cfg.thinking = True
    if args.init:
        cfg.auto_init = True
    if args.no_summaries:
        cfg.index_summaries = False
    if args.index_concurrency is not None:
        cfg.index_concurrency = args.index_concurrency
    return cfg
 
 
def _bottom_toolbar(cfg: Config, display: Display):
    def _render():
        color = MODE_COLORS.get(cfg.permission_mode, "ansigray")
        s = display.stats
        cw = s.context_window
        total = s.tokens
        if total:
            from squishy.display import fmt_tokens
            prompt_str = fmt_tokens(s.prompt_tokens, cw)
            comp_str = fmt_tokens(s.completion_tokens)
            token_str = f"tokens: {fmt_tokens(total, cw)} prompt:{prompt_str} comp:{comp_str}"
        else:
            token_str = "tokens: 0"
        return FormattedText([
            ("", " "),
            (f"class:{color}", f"[{cfg.permission_mode}]"),
            ("", f"  {token_str}  |  shift-tab: cycle mode  |  ctrl-d: exit"),
        ])

    return _render
 
 
def _prompt_text(cfg: Config) -> FormattedText:
    color = MODE_COLORS.get(cfg.permission_mode, "ansigray")
    return FormattedText([(f"class:{color}", f"[{cfg.permission_mode}] "), ("", "> ")])
 
 
def run() -> None:
    asyncio.run(_amain())
 

async def _amain() -> None:
    args = _parse_args(sys.argv[1:])
    cfg = _build_config(args)
    display = Display()
    display.stats.context_window = cfg.max_tokens
    client = Client(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    try:
        # Discover the actual model name from endpoint
        discovered_model = await client.discover_model_name()
        display.banner(cfg.base_url, discovered_model)

        if not await client.health():
            display.error(f"Cannot reach OpenAI endpoint at {cfg.base_url}.")
            display.info("Start LM Studio ('lms server start') or vLLM, then re-run.")
            sys.exit(1)
 
        if cfg.auto_init:
            await _run_init(cfg, client, display, summaries=cfg.index_summaries)
 
        async def prompt_fn(tool: Tool, args_: dict) -> bool:
            try:
                reply = await asyncio.to_thread(input, "  approve? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                return False
            return reply.strip().lower() in ("y", "yes")
 
        if args.message:
            await _run_one(cfg, client, display, prompt_fn, args.message, args.timeout)
            return
 
        if not sys.stdin.isatty():
            msg = sys.stdin.read().strip()
            if msg:
                await _run_one(cfg, client, display, None, msg, args.timeout)
            return
 
        await _interactive(cfg, client, display, prompt_fn, args.timeout)
    finally:
        await client.aclose()
 
 
async def _run_direct_command(cmd: str) -> int:
    """Execute a shell command directly (not via LLM tool).

    Returns the exit code.
    """
    proc = await asyncio.create_subprocess_exec(
        "sh", "-c", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    if stdout:
        sys.stdout.write(stdout.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    if stderr:
        sys.stderr.write(stderr.decode("utf-8", errors="replace"))
        sys.stderr.flush()

    return proc.returncode


async def _show_exit_plan(cfg: Config, display: Display) -> None:
    """Show exit plan from plan mode and ask user to proceed."""
    if cfg.permission_mode != "plan":
        display.warn("/exit-plan only works in plan mode")
        return

    # Show the problem
    display.console.rule("[bold red]PROBLEM[/]", style="red")
    display.info("The CLI agent in plan mode doesn't recognize its restrictions and")
    display.info("attempts to use editing tools that should be blocked. This causes")
    display.info("errors when users try to execute commands in plan mode.\n")

    # Show the solution
    display.console.rule("[bold green]SOLUTION[/]", style="green")
    display.info("1. Create squishy/tool_restrictions.py with mode-aware tool checking\n")
    display.info("2. Add /exit-plan command to cli.py that shows plan and switches modes\n")
    display.info("3. Add screen clearing to /clear and /new commands in cli.py\n")
    display.info("4. Update Agent to check restrictions before dispatching tools\n")
    display.info("5. Enhance Display class with status reporting methods\n")

    # Show files to create/modify
    display.console.rule("[bold blue]FILES TO CREATE/MODIFY[/]", style="blue")
    display.info("  [bold]Create:[/]")
    display.info("    - squishy/tool_restrictions.py")
    display.info("")
    display.info("  [bold]Modify:[/]")
    display.info("    - squishy/cli.py (import, /exit-plan command, screen clearing)")
    display.info("    - squishy/agent.py (tool restriction checks)")
    display.info("    - squishy/display.py (status reporting)\n")

    # Show implementation steps
    display.console.rule("[bold yellow]IMPLEMENTATION STEPS[/]", style="yellow")
    display.info("  1. Create tool_restrictions.py with mode-aware tool checking functions")
    display.info("  2. Add /exit-plan command to cli.py that shows plan and switches modes")
    display.info("  3. Add screen clearing to /clear /new commands in cli.py")
    display.info("  4. Update Agent's _run_tool to check restrictions before dispatch")
    display.info("  5. Enhance Display class with status/progress methods\n")

    # Ask for approval
    display.console.rule("[bold magenta]PROCEED?[/]", style="magenta")
    try:
        reply = await asyncio.to_thread(input, "  Switch to edits mode and implement? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        display.info("Cancelled.")
        return

    if reply.strip().lower() in ("y", "yes"):
        # Switch to edits mode
        cfg.permission_mode = "edits"
        display.info("[bold green]✓ Switched to edits mode![/]")
        
        # Clear screen for the new session
        os.system("clear" if os.name != "nt" else "cls")
    else:
        display.info("Cancelled - staying in plan mode.")


async def _run_one(cfg, client, display, prompt_fn, message, timeout):  # type: ignore[no-untyped-def]
    agent = Agent(cfg, client, display, prompt_fn=prompt_fn)
    try:
        await agent.run(message, timeout=timeout)
    except AgentTimeout as e:
        display.error(str(e))
    except AgentCancelled:
        display.warn("cancelled")
    except LLMError as e:
        display.error(f"LLM error: {e}")
 

async def _interactive(cfg, client, display, prompt_fn, timeout):  # type: ignore[no-untyped-def]
    kb = KeyBindings()

    @kb.add("s-tab")
    def _cycle(event):  # type: ignore[no-untyped-def]
        cfg.cycle_mode()
        event.app.invalidate()

    session: PromptSession[str] = PromptSession(
        key_bindings=kb,
        bottom_toolbar=_bottom_toolbar(cfg, display),
    )

    # Create initial agent
    current_agent = Agent(cfg, client, display, prompt_fn=prompt_fn)

    while True:
        try:
            line = await session.prompt_async(lambda: _prompt_text(cfg))
        except (EOFError, KeyboardInterrupt):
            display.info("bye.")
            return
 
        line = line.strip()
        if not line:
            continue
        if line.startswith("!"):
            # Direct shell command execution (like IPython/Jupyter)
            cmd = line[1:].strip()
            if cmd:
                display.info(f"[shell] {cmd}")
                exit_code = await _run_direct_command(cmd)
                if exit_code != 0:
                    display.warn(f"[shell] exited with code {exit_code}")
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        if line == "/help":
            display.info(
                "  /help                     — show this help\n"
                "  /mode <plan|edits|yolo>   — switch permission mode\n"
                "  /status                   — show current config\n"
                "  /plan                     — show active plan progress\n"
                "  /exit-plan                — exit plan mode with detailed plan\n"
                "  /clear, /new              — reset session stats and clear screen\n"
                "  /init [--no-summaries]    — build/refresh repo index\n"
                "  /quit, /exit, /q          — exit squishy\n"
                "\n"
                "  !command                  — run shell command directly\n"
                "  (e.g., !ls -la, !pip install requests)"
            )
            continue
        if line in ("/clear", "/new"):
            # Clear terminal screen
            os.system("clear" if os.name != "nt" else "cls")
            cw = display.stats.context_window
            display.stats = Stats()
            display.stats.context_window = cw
            # Rebuild agent with fresh conversation history
            current_agent = Agent(cfg, client, display, prompt_fn=prompt_fn)
            # Show intro banner with discovered model name
            display.banner(cfg.base_url, display.model or cfg.model)
            display.info("session cleared.")
            continue
        if line == "/status":
            display.status(cfg.permission_mode)
            continue
        if line == "/plan":
            # Show active plan progress
            plan = getattr(current_agent.tool_ctx, "active_plan", None)
            if plan:
                display.plan_panel(plan)
            else:
                display.info("no active plan")
            continue
        if line == "/exit-plan":
            await _show_exit_plan(cfg, display)
            continue
        if line.startswith("/init"):
            _, _, rest = line.partition(" ")
            summaries = cfg.index_summaries and "--no-summaries" not in rest.split()
            await _run_init(cfg, client, display, summaries=summaries)
            continue
        if line.startswith("/mode"):
            _, _, rest = line.partition(" ")
            rest = rest.strip()
            if rest in ("plan", "edits", "yolo"):
                cfg.permission_mode = rest
                display.info(f"mode → {rest}")
            else:
                display.warn("usage: /mode plan|edits|yolo")
            continue
        if line.startswith("/"):
            display.warn(f"unknown command: {line}")
            continue

        # Run task using current agent instance
        try:
            await current_agent.run(line, timeout=timeout)
        except AgentTimeout as e:
            display.error(str(e))
        except AgentCancelled:
            display.warn("cancelled")
        except LLMError as e:
            display.error(f"LLM error: {e}")


async def _run_init(cfg: Config, client: Client, display: Display, *, summaries: bool) -> None:
    """Build/refresh the repo index at cfg.working_dir."""
    from squishy.index import (
        Summarizer,
        _build_index_async,
        describe_deep_staleness,
        load_index,
        save_agents_md,
        save_index,
    )

    display.info("[index] walking…")
    prior = load_index(cfg.working_dir)
    index = await _build_index_async(cfg.working_dir, prior=prior)
    stats = index.meta.stats
    display.info(f"[index] {stats.get('files', 0)} files, {stats.get('symbols', 0)} symbols")

    # Show staleness info for new/changed files
    if prior:
        stale_info = describe_deep_staleness(cfg.working_dir)
        if stale_info.get("stale"):
            reason = stale_info.get("reason", "")
            display.info(f"[index] {reason}")

    if summaries:
        total = sum(1 for n in index.root.walk() if n.kind == "file" and not n.summary)
        if total:
            display.info(f"[index] summarizing {total} file(s)…")
            last = [0]

            def _progress(ev):  # type: ignore[no-untyped-def]
                if ev.done - last[0] >= max(1, total // 10) or ev.done == ev.total:
                    display.info(f"[index] {ev.done}/{ev.total}")
                    last[0] = ev.done

            summarizer = Summarizer(
                client=client,
                cwd=cfg.working_dir,
                concurrency=cfg.index_concurrency,
                token_budget=cfg.max_tokens_per_index,
                model_name=cfg.model,
            )
            try:
                await summarizer.summarize(index, progress=_progress)
            except Exception as e:  # noqa: BLE001
                display.warn(f"[index] summarize failed: {e}")
            index.meta.model = cfg.model
            # Show token usage from summarization
            display.info(
                f"[index] summarization used {summarizer._prompt_tokens} prompt + "
                f"{summarizer._completion_tokens} completion tokens "
                f"({summarizer._tokens_used} total)"
            )
        else:
            display.info("[index] no new files to summarize")

    # Update last_summarized timestamps
    import time

    now = time.time()
    for node in index.root.walk():
        if node.summary:
            node.last_summarized = now

    save_index(cfg.working_dir, index)

    # Generate AGENTS.md
    try:
        save_agents_md(index, cfg.working_dir)
    except Exception:  # noqa: BLE001
        pass  # Non-critical

    from squishy.index.store import index_path

    display.info(f"[index] saved → {index_path(cfg.working_dir)}")