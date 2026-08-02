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
from squishy.async_input import ModeCycler
from squishy.client import Client
from squishy.config import Config
from squishy.display import MODE_COLORS, Display, Stats
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.file_browser import format_reference_list, inject_references_with_missing
from squishy.session import (
    create_session,
    export_training_to_file,
    list_sessions,
    load_messages,
    restore_for_replay,
)
from squishy.tool_restrictions import TOOL_PROFILES
from squishy.tools.base import Tool


# Slash commands that take no arguments — typing extra text is almost
# always a typo (e.g. ``/clear all``) that we silently swallowed before.
_NO_ARG_SLASH_CMDS: frozenset[str] = frozenset({
    "/quit", "/exit", "/q",
    "/help",
    "/clear", "/new",
    "/status",
    "/session",
    "/sessions",
})


def _slash_extra_args(line: str) -> tuple[str, str]:
    """Split ``"/cmd extra args"`` into ``("/cmd", "extra args")``.

    Returns ``("", "")`` when ``line`` is not a slash command. The
    second element is empty when no args were supplied.
    """
    if not line.startswith("/"):
        return ("", "")
    head, _, rest = line.partition(" ")
    return (head, rest.strip())
 
 
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
    p.add_argument("--edits", action="store_true", help="Start in edits mode")
    p.add_argument("--yolo", action="store_true", help="Start in yolo mode (no prompts)")
    p.add_argument(
        "--tools", dest="tool_profile", choices=sorted(TOOL_PROFILES), default="standard",
        help="Tool profile: standard (all tools the mode allows) or minimal "
             "(shell + file primitives only)",
    )
    p.add_argument("--no-sandbox", action="store_true", help="Disable Docker sandbox for run_command")
    p.add_argument("--sandbox", action="store_true", help="Enable Docker sandbox for run_command")
    p.add_argument("--thinking", action="store_true", help="Allow <think> blocks")
    p.add_argument(
        "--context-window", type=int, default=None,
        help="Context window in tokens (default: auto-detect, else assume 32768). "
             "Set this when your endpoint doesn't advertise context_length.",
    )
    p.add_argument("--message", "-m", help="Non-interactive: send one message, print result, exit")
    p.add_argument("--init", action="store_true", help="Build .squishy/index.json before the REPL")
    p.add_argument("--no-summaries", action="store_true", help="When indexing, skip LLM summaries")
    p.add_argument("--index-concurrency", type=int, default=None, help="Parallel summary calls (default 4)")
    p.add_argument("--resume", metavar="UUID", help="Resume a previous session by UUID")
    p.add_argument("--session-dir", help="Session storage directory (env SQUISHY_SESSION_DIR)")
    p.add_argument("--no-sessions", action="store_true", help="Disable session persistence")
    return p.parse_args(argv)
 
 
def _user_configured_model(args: argparse.Namespace) -> bool:
    """Return True if the user explicitly chose a model (via --model or env).

    When False we let endpoint discovery fill in cfg.model so the banner
    matches whatever requests are actually routed to. When True we keep the
    user's choice (e.g. SQUISHY_MODEL from a .env) so a multi-model endpoint
    doesn't make the banner contradict the requests being sent.
    """
    if args.model:
        return True
    return bool(os.environ.get("SQUISHY_MODEL"))


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
    elif args.yolo:
        cfg.permission_mode = "yolo"
    elif args.edits:
        cfg.permission_mode = "edits"
    cfg.tool_profile = getattr(args, "tool_profile", "standard")
    if args.no_sandbox:
        cfg.use_sandbox = False
    if args.sandbox:
        cfg.use_sandbox = True
    if args.thinking:
        cfg.thinking = True
    if args.context_window is not None:
        cfg.context_window = args.context_window
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
    display.set_mode(cfg.permission_mode)
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
        # Discover the endpoint's model list (also picks up context_window).
        # Only adopt the discovered name when the user hasn't explicitly
        # configured a model — otherwise the banner would show a different
        # model than the one requests are routed to (e.g. when a .env file
        # sets SQUISHY_MODEL but the endpoint has multiple models loaded).
        # Bounded at 10s so a dead endpoint can't strand the user on a
        # 120s blank-screen wait — the banner shows up either way.
        discovered_model = await client.discover_model_name()
        if discovered_model == cfg.model and cfg.model and not _user_configured_model(args):
            # Discovery returned the fallback (configured model). Tell the
            # user the endpoint isn't responding so they can ^C instead of
            # waiting on a dead URL.
            display.warn(
                f"endpoint {cfg.base_url} did not respond to model discovery "
                "— it may be unreachable. Check the URL or hit ^C to exit."
            )
        if not _user_configured_model(args):
            cfg.model = discovered_model
            client.model = discovered_model
        display.banner(cfg.base_url, cfg.model)
        # Show % usage against the window the agent actually budgets against:
        # explicit override → endpoint-advertised → assumed default.
        display.stats.context_window = (
            cfg.context_window
            or client.context_window
            or cfg.assumed_context_window
        )

        if cfg.auto_init:
            # A failed --init build must not abort startup — the REPL/one-shot
            # can still run (recall just won't be available).
            try:
                await _run_init(cfg, client, display, summaries=cfg.index_summaries)
            except (KeyboardInterrupt, asyncio.CancelledError):
                display.warn("[index] --init cancelled")
            except Exception as e:  # noqa: BLE001
                display.error(f"[index] --init failed: {e}")

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

        # Single mode cycler shared across this whole interactive session.
        # It's started around each agent.run() call so shift-tab works while
        # the model is busy. paused() temporarily releases stdin so the
        # approval prompt's reader can take over cleanly.
        mode_cycler = ModeCycler(
            on_cycle=lambda: display.mode_changed(cfg.cycle_mode())
        )
        # Dedicated PromptSession for approval prompts. Using
        # prompt_toolkit instead of asyncio.to_thread(input) so Ctrl+C
        # raises cleanly and stdin/terminal state is restored properly —
        # the previous to_thread(input) path left the input thread
        # blocked on stdin while the mode cycler resumed cbreak mode,
        # which deadlocked the terminal.
        approval_session: PromptSession[str] = PromptSession()

        async def prompt_fn(tool: Tool, args_: dict):
            label = "  approve? [y/N, ^C cancels] "
            # Make sure any in-flight streaming markdown is finalised before
            # we hand the terminal to prompt_toolkit, otherwise the live
            # region and the prompt fight for the same screen rows.
            display.flush_streaming_text()
            with mode_cycler.paused():
                try:
                    reply = await approval_session.prompt_async(label)
                except EOFError:
                    # Ctrl+D — same as a polite decline.
                    display.info("declined.")
                    return False
                # Ctrl+C is intentionally *not* caught here. We want it to
                # propagate up through the agent loop so the whole turn
                # is cancelled and the user lands
                # back at the REPL prompt — instead of the agent silently
                # treating it as "n" and continuing to chug.
            stripped = (reply or "").strip()
            lowered = stripped.lower()
            return lowered in ("y", "yes")

 
        if args.message:
            # -m is a one-shot ("send one message, print result, exit"). Only
            # wire the interactive approval prompt when stdin is a real TTY —
            # otherwise (piped/CI/non-TTY) a shell approval would block forever
            # on a prompt that can never be answered. Non-TTY falls back to
            # auto-approve, matching the stdin-pipe branch below.
            interactive = sys.stdin.isatty()
            await _run_one(
                cfg, client, display,
                prompt_fn if interactive else None,
                args.message, args.timeout,
                mode_cycler if interactive else None,
            )
            return

        if not sys.stdin.isatty():
            msg = sys.stdin.read().strip()
            if msg:
                await _run_one(cfg, client, display, None, msg, args.timeout, None)
            else:
                # Silent exit was confusing — explain why nothing happened.
                display.warn(
                    "no input on stdin and no -m message; nothing to do"
                )
            return

        await _interactive(
            cfg, client, display, prompt_fn, args.timeout,
            resume_id=args.resume, mode_cycler=mode_cycler,
        )
    finally:
        await client.aclose()
 
 
async def _run_direct_command(cmd: str, timeout: float = 120.0) -> int:
    """Execute a shell command directly (not via LLM tool).

    Returns the exit code. Raises ``KeyboardInterrupt`` if the user
    cancels with Ctrl-C; the child process is killed in that case.
    Long-running commands are killed after ``timeout`` seconds to
    keep the REPL responsive.
    """
    proc = await asyncio.create_subprocess_exec(
        "sh", "-c", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        sys.stderr.write(
            f"[shell] command exceeded {timeout:.0f}s timeout; killed\n"
        )
        sys.stderr.flush()
        return 124  # GNU timeout(1) convention.
    except (asyncio.CancelledError, KeyboardInterrupt):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        # Re-raise CancelledError (not KeyboardInterrupt) so pytest and
        # other framework-level Ctrl-C handlers behave normally; the
        # REPL caller catches both shapes via the (AgentCancelled,
        # KeyboardInterrupt) tuple.
        raise asyncio.CancelledError

    if stdout:
        sys.stdout.write(stdout.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    if stderr:
        sys.stderr.write(stderr.decode("utf-8", errors="replace"))
        sys.stderr.flush()

    return proc.returncode if proc.returncode is not None else 1


def _create_session_for_agent(
    cfg: Config, model_name: str, display: Display | None = None,
) -> str | None:
    """Create a session and return its ID, or None if disabled.

    Surfaces failures to the display (when provided) instead of swallowing
    them silently — a broken session_dir is a real configuration issue
    the user should hear about.
    """
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
    except Exception as e:  # noqa: BLE001
        if display is not None:
            display.warn(f"[session] failed to create session: {e}")
        return None


async def _run_one(cfg, client, display, prompt_fn, message, timeout, mode_cycler=None):  # type: ignore[no-untyped-def]
    session_id = _create_session_for_agent(cfg, cfg.model, display)
    agent = Agent(cfg, client, display, prompt_fn=prompt_fn, session_id=session_id)
    cycler = mode_cycler or _NullModeCycler()
    try:
        # Inject file references before running
        message_with_files, references, missing = inject_references_with_missing(
            message, cfg.working_dir,
        )
        if references:
            display.info(format_reference_list(references))
        for missing_path in missing:
            display.warn(f"@{missing_path}: file not found (skipped)")
        async with cycler:
            await agent.run(message_with_files, timeout=timeout)
    except AgentTimeout as e:
        display.error(str(e))
        return
    except (AgentCancelled, KeyboardInterrupt):
        display.flush_streaming_text()
        display.warn("cancelled")
        return
    except LLMError as e:
        display.error(f"LLM error: {e}")
        return


class _NullModeCycler:
    """No-op stand-in used when the mode cycler isn't available
    (non-TTY, headless message piped via stdin, etc.)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None
 

async def _interactive(cfg, client, display, prompt_fn, timeout, *, resume_id: str | None = None, mode_cycler=None):  # type: ignore[no-untyped-def]
    kb = KeyBindings()
    cycler = mode_cycler or _NullModeCycler()

    @kb.add("s-tab")
    def _cycle(event):  # type: ignore[no-untyped-def]
        new_mode = cfg.cycle_mode()
        # Print the change so the user sees it inline — silently swapping
        # edits→yolo while a tool is queued is the easiest way to give
        # the agent unintended write permissions. ``mode_changed`` calls
        # ``set_mode`` internally so we don't double-set.
        display.mode_changed(new_mode)
        event.app.invalidate()

    session: PromptSession[str] = PromptSession(
        key_bindings=kb,
        bottom_toolbar=_bottom_toolbar(cfg, display),
    )

    # Resume or create initial agent.
    current_agent = None
    if resume_id:
        try:
            # Restore stored dict-args tool_calls back to the OpenAI wire
            # format (JSON-string arguments) so the resumed transcript is valid
            # to send to the endpoint.
            prev_messages = restore_for_replay(
                load_messages(resume_id, root=cfg.session_dir)
            )
            # Build WITHOUT the session id so __post_init__ doesn't append a
            # fresh system prompt into the resumed session's on-disk log; wire
            # the id up only after the loaded transcript replaces messages.
            current_agent = Agent(cfg, client, display, prompt_fn=prompt_fn, session_id=None)
            current_agent.messages = prev_messages
            current_agent.session_id = resume_id
            current_agent._last_persisted_idx = len(prev_messages)
            current_agent._full_log_idx = len(prev_messages)
            display.info(f"[session] resumed {resume_id[:12]}… ({len(prev_messages)} messages)")
        except Exception as e:  # noqa: BLE001
            display.error(f"failed to resume session {resume_id}: {e}")
            display.info("starting a fresh session instead.")
            resume_id = None
    if current_agent is None:
        session_id = _create_session_for_agent(cfg, display.model or cfg.model, display)
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
            if not cmd:
                display.warn("usage: !<command>  (e.g. !ls -la)")
                continue
            display.info(f"[shell] {cmd}")
            try:
                exit_code = await _run_direct_command(cmd)
            except (KeyboardInterrupt, asyncio.CancelledError):
                display.warn("[shell] interrupted")
                continue
            if exit_code != 0:
                display.warn(f"[shell] exited with code {exit_code}")
            continue
        # Warn about unexpected args on no-arg slash commands so a typo
        # like ``/clear cache`` doesn't silently fall through to "unknown
        # command" or get treated as an LLM prompt.
        head, extra = _slash_extra_args(line)
        if head in _NO_ARG_SLASH_CMDS and extra:
            display.warn(f"{head} takes no arguments (got: {extra!r}); ignoring")
            line = head
        if line in ("/quit", "/exit", "/q"):
            return
        if line == "/help":
            display.info(
                "  /help                     — show this help\n"
                "  /mode <edits|yolo>        — switch permission mode\n"
                "  /status                   — show current config\n"
                "  /clear, /new              — reset session stats and clear screen\n"
                "  /init [--no-summaries]    — build/refresh repo index + code graph\n"
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
            print("\033[H\033[2J", end="", flush=True)
            cw = display.stats.context_window
            display.stats = Stats()
            display.stats.context_window = cw
            # Rebuild agent with fresh conversation history and new session.
            session_id = _create_session_for_agent(cfg, display.model or cfg.model, display)
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
        if line.startswith("/init"):
            _, _, rest = line.partition(" ")
            summaries = cfg.index_summaries and "--no-summaries" not in rest.split()
            # Fault-isolate: a mid-build indexing failure must not kill the
            # whole REPL. Ctrl+C returns cleanly to the prompt.
            try:
                await _run_init(cfg, client, display, summaries=summaries)
            except (KeyboardInterrupt, asyncio.CancelledError):
                display.warn("[index] /init cancelled")
            except Exception as e:  # noqa: BLE001
                display.error(f"[index] /init failed: {e}")
            continue
        if line.startswith("/mode"):
            _, _, rest = line.partition(" ")
            rest = rest.strip()
            if rest in ("edits", "yolo"):
                cfg.permission_mode = rest
                display.set_mode(rest)
                display.info(f"mode → {rest}")
            else:
                display.warn("usage: /mode edits|yolo")
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
            if not cfg.save_sessions:
                display.warn(
                    "session saving is disabled (--no-sessions); "
                    "nothing to export"
                )
                continue
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
        if line == "/":
            display.warn("empty slash command — type /help for the list")
            continue
        if line.startswith("/"):
            # Strip args so the suggestion focuses on the command itself.
            cmd_only = line.split(maxsplit=1)[0]
            display.warn(
                f"unknown command: {cmd_only} — type /help for the list"
            )
            continue

        # Run task using current agent instance
        try:
            # Inject file references before running
            message_with_files, references, missing = inject_references_with_missing(
                line, cfg.working_dir,
            )
            if references:
                display.info(format_reference_list(references))
            for missing_path in missing:
                display.warn(f"@{missing_path}: file not found (skipped)")
            # Activate the mode cycler so shift-tab works while the model
            # is busy. The cycler is a no-op on non-TTY platforms.
            async with cycler:
                await current_agent.run(message_with_files, timeout=timeout)
        except AgentTimeout as e:
            display.error(str(e))
        except (AgentCancelled, KeyboardInterrupt):
            # Ctrl+C inside agent.run lands here once asyncio cancels the
            # task. Always make sure the streamed display is closed so the
            # next REPL prompt doesn't draw on top of a half-rendered
            # markdown live region.
            display.flush_streaming_text()
            display.warn("cancelled")
            continue
        except LLMError as e:
            display.error(f"LLM error: {e}")
            continue


async def _handle_mcp_command(rest: str, display: Display) -> None:
    """Handle /mcp slash command."""
    from squishy.mcp.client import get_mcp_manager
    from squishy.mcp.config import add_server_to_user_config, remove_server_from_user_config
    from squishy.mcp.tools import reload_mcp

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

    # The call/import/inherit graph, built from the same tree in the same
    # command. Kept separate from the index because it answers different
    # questions (who calls this, what breaks if I change it) and because it is
    # Python-only, while the index covers every language.
    display.info("[graph] building…")
    try:
        from squishy.graph import build_repo_graph, graph_path

        graph = build_repo_graph(cfg.working_dir)
        s = graph.stats()
        display.info(
            f"[graph] {s['nodes']} nodes, {s['edges']} edges → "
            f"{graph_path(cfg.working_dir)}"
        )
    except Exception as e:  # noqa: BLE001
        # A repo with no Python, or one that fails to parse, still gets a
        # working index — the graph is an enhancement, not a prerequisite.
        display.warn(f"[graph] skipped: {e}")
