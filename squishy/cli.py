"""CLI entry point. Runs the async agent under asyncio.run().
 
Shift+Tab rotates permission_mode live via prompt_toolkit, mirroring
Claude Code's in-session mode switching.
"""
 
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys

from dotenv import load_dotenv

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings

from squishy.agent import Agent
from squishy.client import Client
from squishy.config import Config
from squishy.display import Display, MODE_COLORS, Stats
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.file_browser import format_reference_list, inject_references
from squishy.plan_state import PlanState
from squishy.session import (
    create_session,
    export_training_to_file,
    list_sessions,
    load_messages,
    load_session,
)
from squishy.tools.base import Tool

EXECUTE_APPROVED_PLAN_PROMPT = "Execute the approved plan."
 
 
def _parse_args(argv: list[str]) -> argparse.Namespace:
    default_turns = Config().max_turns
    p = argparse.ArgumentParser(prog="squishy", description="Minimal local-LLM coding agent.")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (env SQUISHY_BASE_URL)")
    p.add_argument("--model", help="Model id (env SQUISHY_MODEL)")
    p.add_argument("--api-key", help="API key (env SQUISHY_API_KEY)")
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=f"Turn cap (uses config default: {default_turns})",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None, help="Task timeout in seconds")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--plan", action="store_true", help="Start in plan mode (default, read-only)")
    p.add_argument("--edits", action="store_true", help="Start in edits mode")
    p.add_argument("--yolo", action="store_true", help="Start in yolo mode (no prompts)")
    p.add_argument("--no-sandbox", action="store_true", help="Disable Docker sandbox for run_command")
    p.add_argument("--sandbox", action="store_true", help="Enable Docker sandbox for run_command")
    p.add_argument("--thinking", action="store_true", help="Allow <think> blocks")
    p.add_argument("--message", "-m", help="Non-interactive: send one message, print result, exit")
    p.add_argument("--init", action="store_true", help="Build .squishy/index.json before the REPL")
    p.add_argument("--no-summaries", action="store_true", help="When indexing, skip LLM summaries")
    p.add_argument("--index-concurrency", type=int, default=None, help="Parallel summary calls (default 4)")
    p.add_argument("--resume", metavar="UUID", help="Resume a previous session by UUID")
    p.add_argument("--session-dir", help="Session storage directory (env SQUISHY_SESSION_DIR)")
    p.add_argument("--no-sessions", action="store_true", help="Disable session persistence")
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
    if args.session_dir:
        cfg.session_dir = args.session_dir
    if args.no_sessions:
        cfg.save_sessions = False
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
            ("", f"  {token_str}  |  shift-tab: cycle mode  |  ctrl-j: newline  |  ctrl-d: exit"),
        ])

    return _render
 
 
def _prompt_text(cfg: Config) -> FormattedText:
    color = MODE_COLORS.get(cfg.permission_mode, "ansigray")
    return FormattedText([(f"class:{color}", f"[{cfg.permission_mode}] "), ("", "> ")])
 
 
def run() -> None:
    asyncio.run(_amain())
 

async def _amain() -> None:
    # Load .env file before any Config instantiation so env vars are available.
    load_dotenv()

    args = _parse_args(sys.argv[1:])
    cfg = _build_config(args)
    display = Display()
    client = Client(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        thinking=cfg.thinking,
    )

    try:
        # Discover the actual model name from endpoint (also discovers context_window)
        discovered_model = await client.discover_model_name()
        display.banner(cfg.base_url, discovered_model)
        # Use the discovered context window so the % usage display is meaningful.
        # Falls back to 0 (no % shown) for endpoints that don't expose it.
        display.stats.context_window = client.context_window

        if cfg.auto_init:
            await _run_init(cfg, client, display, summaries=cfg.index_summaries)

        # Initialize MCP servers (non-blocking).
        try:
            from squishy.mcp.tools import initialize_mcp
            mcp_errors = await asyncio.to_thread(initialize_mcp)
            for server, mcp_err in mcp_errors.items():
                if mcp_err:
                    display.warn(f"[mcp] {server}: {mcp_err}")
                else:
                    display.info(f"[mcp] {server}: connected")
        except Exception as e:
            display.warn(f"[mcp] init failed: {e}")

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
 
        await _interactive(cfg, client, display, prompt_fn, args.timeout, resume_id=args.resume)
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


async def _show_exit_plan(cfg: Config, display: Display, plan: dict | None) -> None:
    """Show the active plan and offer to switch into edits mode."""
    if cfg.permission_mode != "plan":
        display.warn("/exit-plan only works in plan mode")
        return
    if not plan:
        display.info("no active plan — ask the agent to produce one first")
        return

    display.plan_panel(plan)
    await _prompt_switch_to_edits(
        cfg,
        display,
        prompt_text="  Switch to edits mode and execute the plan? [Y/n] ",
        success_text="[bold green]✓ Switched to edits mode[/]",
    )


async def _prompt_switch_to_edits(
    cfg: Config,
    display: Display,
    *,
    prompt_text: str,
    success_text: str,
) -> None:
    try:
        reply = await asyncio.to_thread(input, prompt_text)
    except (EOFError, KeyboardInterrupt):
        display.info("Cancelled.")
        return
    if reply.strip().lower() in ("", "y", "yes"):
        cfg.permission_mode = "edits"
        display.info(success_text)
    else:
        display.info("Staying in plan mode.")


async def _auto_execute_plan(agent: Agent, cfg: Config, display: Display, timeout: float | None) -> None:
    """If a plan was just approved, auto-switch to edits mode and execute it."""
    plan = agent.tool_ctx.plan
    if not (
        cfg.permission_mode == "plan"
        and plan is not None
        and plan.approved
        and not agent.tool_ctx.plan_switch_prompted
    ):
        return
    agent.tool_ctx.plan_switch_prompted = True
    cfg.permission_mode = "edits"
    display.info("[bold green]✓ Switched to edits mode[/]")
    try:
        await agent.run(EXECUTE_APPROVED_PLAN_PROMPT, timeout=timeout)
    except AgentTimeout as e:
        display.error(str(e))
    except AgentCancelled:
        display.warn("cancelled")
    except LLMError as e:
        display.error(f"LLM error: {e}")


def _create_session_for_agent(cfg: Config, model_name: str) -> str | None:
    """Create a session and return its ID, or None if disabled."""
    if not cfg.save_sessions:
        return None
    try:
        from squishy.tools import openai_schemas
        tools = openai_schemas(cfg.permission_mode)
        sess = create_session(
            model=model_name,
            working_dir=cfg.working_dir,
            mode=cfg.permission_mode,
            tools=tools,
            root=cfg.session_dir,
        )
        return sess.id
    except Exception:  # noqa: BLE001
        return None


async def _run_one(cfg, client, display, prompt_fn, message, timeout):  # type: ignore[no-untyped-def]
    session_id = _create_session_for_agent(cfg, cfg.model)
    agent = Agent(cfg, client, display, prompt_fn=prompt_fn, session_id=session_id)
    try:
        # Inject file references before running
        message_with_files, references = inject_references(message, cfg.working_dir)
        if references:
            display.info(format_reference_list(references))
        await agent.run(message_with_files, timeout=timeout)
    except AgentTimeout as e:
        display.error(str(e))
        return
    except AgentCancelled:
        display.warn("cancelled")
        return
    except LLMError as e:
        display.error(f"LLM error: {e}")
        return

    await _auto_execute_plan(agent, cfg, display, timeout)
 

async def _interactive(cfg, client, display, prompt_fn, timeout, *, resume_id: str | None = None):  # type: ignore[no-untyped-def]
    kb = KeyBindings()

    @kb.add("s-tab")
    def _cycle(event):  # type: ignore[no-untyped-def]
        cfg.cycle_mode()
        event.app.invalidate()

    session: PromptSession[str] = PromptSession(
        key_bindings=kb,
        bottom_toolbar=_bottom_toolbar(cfg, display),
    )

    # Resume or create initial agent.
    if resume_id:
        try:
            prev_messages = load_messages(resume_id, root=cfg.session_dir)
            current_agent = Agent(cfg, client, display, prompt_fn=prompt_fn, session_id=resume_id)
            # Replace the fresh messages with the loaded ones.
            current_agent.messages = prev_messages
            current_agent._last_persisted_idx = len(prev_messages)
            display.info(f"[session] resumed {resume_id[:12]}… ({len(prev_messages)} messages)")
        except Exception as e:  # noqa: BLE001
            display.error(f"failed to resume session {resume_id}: {e}")
            return
    else:
        session_id = _create_session_for_agent(cfg, display.model or cfg.model)
        current_agent = Agent(cfg, client, display, prompt_fn=prompt_fn, session_id=session_id)
        if session_id:
            display.info(f"[session] {session_id[:12]}…")

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
                "  /mcp [list|reload|add|remove] — manage MCP servers\n"
                "  /session                  — show current session UUID\n"
                "  /sessions                 — list recent sessions\n"
                "  /export [UUID]            — export session as training JSONL\n"
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
            # Rebuild agent with fresh conversation history and new session.
            session_id = _create_session_for_agent(cfg, display.model or cfg.model)
            current_agent = Agent(cfg, client, display, prompt_fn=prompt_fn, session_id=session_id)
            # Show intro banner with discovered model name
            display.banner(cfg.base_url, display.model or cfg.model)
            display.info("session cleared.")
            if session_id:
                display.info(f"[session] {session_id[:12]}…")
            continue
        if line == "/status":
            display.status(cfg.permission_mode)
            continue
        if line == "/plan":
            # Show active plan progress
            plan = current_agent.tool_ctx.plan
            if plan:
                display.plan_panel(plan.to_dict())
            else:
                display.info("no active plan")
            continue
        if line == "/exit-plan":
            plan = current_agent.tool_ctx.plan
            await _show_exit_plan(cfg, display, plan.to_dict() if plan else None)
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
        if line.startswith("/mcp"):
            _, _, mcp_rest = line.partition(" ")
            await _handle_mcp_command(mcp_rest.strip(), display)
            continue
        if line == "/session":
            sid = current_agent.session_id
            if sid:
                display.info(f"[session] {sid}")
            else:
                display.info("[session] not active (sessions disabled)")
            continue
        if line == "/sessions":
            sessions = list_sessions(limit=20, working_dir=cfg.working_dir, root=cfg.session_dir)
            if not sessions:
                display.info("no sessions found")
            else:
                lines = []
                for s in sessions:
                    short_id = s.id[:12]
                    date = s.updated_at[:19] if s.updated_at else "?"
                    lines.append(
                        f"  {short_id}  {date}  {s.mode:<6}  "
                        f"turns={s.turns}  tokens={s.tokens}  {s.status}"
                    )
                display.info("\n".join(lines))
            continue
        if line.startswith("/export"):
            _, _, rest = line.partition(" ")
            export_id = rest.strip() or (current_agent.session_id or "")
            if not export_id:
                display.warn("no session to export (provide UUID or start a session)")
                continue
            # Resolve short IDs by prefix match.
            if len(export_id) < 32:
                all_sessions = list_sessions(limit=100, root=cfg.session_dir)
                matches = [s for s in all_sessions if s.id.startswith(export_id)]
                if len(matches) == 1:
                    export_id = matches[0].id
                elif len(matches) > 1:
                    display.warn(f"ambiguous prefix '{export_id}' — matches {len(matches)} sessions")
                    continue
                else:
                    display.warn(f"no session matching '{export_id}'")
                    continue
            try:
                from pathlib import Path
                out_path = Path(cfg.session_dir) / f"{export_id[:12]}_training.jsonl"
                export_training_to_file(export_id, out_path, root=cfg.session_dir)
                display.info(f"[export] {out_path}")
            except Exception as e:  # noqa: BLE001
                display.error(f"export failed: {e}")
            continue
        if line.startswith("/"):
            display.warn(f"unknown command: {line}")
            continue

        # Run task using current agent instance
        try:
            # Inject file references before running
            message_with_files, references = inject_references(line, cfg.working_dir)
            if references:
                display.info(format_reference_list(references))
            await current_agent.run(message_with_files, timeout=timeout)
        except AgentTimeout as e:
            display.error(str(e))
        except AgentCancelled:
            display.warn("cancelled")
        except LLMError as e:
            display.error(f"LLM error: {e}")
            continue

        await _auto_execute_plan(current_agent, cfg, display, timeout)


async def _handle_mcp_command(rest: str, display: Display) -> None:
    """Handle /mcp slash command."""
    from squishy.mcp.tools import reload_mcp, get_connect_errors
    from squishy.mcp.client import get_mcp_manager
    from squishy.mcp.config import add_server_to_user_config, remove_server_from_user_config

    parts = rest.split() if rest else []
    subcmd = parts[0].lower() if parts else "list"

    if subcmd == "reload":
        display.info("[mcp] reloading...")
        errors = await asyncio.to_thread(reload_mcp)
        for server, err in errors.items():
            if err:
                display.warn(f"  {server}: {err}")
            else:
                display.info(f"  {server}: connected")
    elif subcmd == "add":
        if len(parts) < 3:
            display.warn("usage: /mcp add <name> <command> [args...]")
            return
        name = parts[1]
        command = parts[2]
        cmd_args = parts[3:]
        raw: dict = {"type": "stdio", "command": command}
        if cmd_args:
            raw["args"] = cmd_args
        add_server_to_user_config(name, raw)
        display.info(f"[mcp] added '{name}' — run /mcp reload to connect")
    elif subcmd == "remove":
        if len(parts) < 2:
            display.warn("usage: /mcp remove <name>")
            return
        name = parts[1]
        if remove_server_from_user_config(name):
            display.info(f"[mcp] removed '{name}'")
            await asyncio.to_thread(reload_mcp)
        else:
            display.warn(f"[mcp] server '{name}' not found")
    else:
        # Default: list servers and tools
        mgr = get_mcp_manager()
        servers = mgr.list_servers()
        if not servers:
            display.info("[mcp] no servers configured")
            display.info("  add servers in ~/.squishy/mcp.json or .mcp.json")
            return
        display.info(f"[mcp] {len(servers)} server(s):")
        total_tools = 0
        for client in servers:
            display.info(f"  {client.status_line()}")
            for tool in client._tools:
                display.info(f"    - {tool.qualified_name}: {tool.description[:60]}")
                total_tools += 1
        if total_tools:
            display.info(f"  total: {total_tools} MCP tool(s)")


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
    with contextlib.suppress(Exception):
        save_agents_md(index, cfg.working_dir)

    from squishy.index.store import index_path

    display.info(f"[index] saved → {index_path(cfg.working_dir)}")
