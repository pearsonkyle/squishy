"""Terminal output — turn headers, tool status lines, diff previews, summary box."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from squishy.plan_state import STATUS_ICONS
# Re-exported so existing `from squishy.display import estimate_tokens`
# imports keep working; the implementation now lives in squishy.tokens.
from squishy.tokens import estimate_tokens as estimate_tokens
from squishy.tool_restrictions import get_allowed_tools

MODE_COLORS = {"plan": "ansicyan", "edits": "ansigreen", "yolo": "ansimagenta", "bench": "ansiyellow"}


def fmt_tokens(count: int, context_window: int = 0) -> str:
    """Format a token count with K notation and optional window percentage.

    Examples:
        fmt_tokens(500)         -> "500"
        fmt_tokens(1234)        -> "1.2K"
        fmt_tokens(12345)       -> "12.3K"
        fmt_tokens(1234, 8192)  -> "1.2K (15%)"
    """
    label = f"{count / 1000:.1f}K" if count >= 1000 else str(count)
    if context_window > 0:
        pct = count * 100 // context_window
        return f"{label} ({pct}%)"
    return label


ICONS = {
    "read_file": "[cyan]📖[/]",
    "write_file": "[green]✎[/]",
    "edit_file": "[yellow]✏️[/]",
    "list_directory": "[cyan]📁[/]",
    "search_files": "[cyan]🔍[/]",
    "run_command": "[magenta]🔧[/]",
    "plan_task": "[cyan]📋[/]",
    "update_plan": "[cyan]📊[/]",
    "finish_plan": "[green]🏁[/]",
    "get_plan": "[cyan]📋[/]",
    "save_note": "[cyan]📝[/]",
    "recall": "[cyan]🔎[/]",
    "glob_files": "[cyan]🔍[/]",
}


@dataclass
class Stats:
    files_created: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    commands_run: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_window: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Display:
    def __init__(self) -> None:
        self.console = Console()
        self.stats = Stats()
        self.model: str = ""
        self.mode: str = ""
        # Streaming markdown state
        self._stream_buffer: str = ""
        self._live_render: Markdown | None = None
        self._live: Live | None = None
        self._use_live: bool = False
        # Thinking spinner — owns the same Live channel as streaming text,
        # so it MUST be stopped before streaming_text_chunk starts a new
        # Live or the two will fight for the same screen rows.
        self._spinner: Live | None = None

    def set_mode(self, mode: str) -> None:
        """Record the current permission mode so it can be shown alongside
        live output (turn headers, mode change notifications, etc.)."""
        self.mode = mode

    def mode_tag(self, mode: str | None = None) -> str:
        """Render a colorized [mode] tag suitable for inline output."""
        m = mode or self.mode
        if not m:
            return ""
        color = MODE_COLORS.get(m, "ansigray")
        return f"[{color}]\\[{m}][/]"

    def banner(self, base_url: str, model: str) -> None:
        self.model = model
        self.console.print(
            Panel.fit(
                f"[bold]squishy[/] — local coding agent\n"
                f"endpoint: [cyan]{base_url}[/]\n"
                f"model:    [cyan]{model}[/]",
                border_style="blue",
            )
        )

    def turn_header(
        self, turn: int, max_turns: int, tool_name: str, brief: str,
        mode: str | None = None,
    ) -> None:
        icon = ICONS.get(tool_name, "•")
        tag = self.mode_tag(mode)
        prefix = f"{tag} " if tag else ""
        # ``brief`` is built from tool args (paths, queries) and may
        # carry ``[`` characters — escape so it can't break the dim tag.
        self.console.print(
            f"{prefix}[dim]\\[Turn {turn}/{max_turns}][/] {icon} {rich_escape(tool_name)} "
            f"[dim]{rich_escape(brief)}[/]"
        )

    def mode_changed(self, mode: str) -> None:
        """Inline notification that the user cycled the permission mode.

        Escalations into ``edits`` or ``yolo`` get an extra warning line so
        the user notices when shift-tab silently grants write permissions
        — the most common source of "the agent edited a file I didn't
        expect" complaints.
        """
        prev = self.mode
        self.set_mode(mode)
        color = MODE_COLORS.get(mode, "ansigray")
        self.console.print(f"  [{color}]◆ mode → {mode}[/]")
        if prev == "plan" and mode in ("edits", "yolo"):
            self.console.print(
                f"  [yellow]⚠ write tools now allowed — agent can modify files in this mode[/]"
            )
        elif mode == "yolo":
            self.console.print(
                f"  [yellow]⚠ yolo mode: shell commands run without per-call approval[/]"
            )

    def command_line(self, command: str) -> None:
        """Show the full shell command on its own line (markup-safe)."""
        t = Text(f"! {command}")
        t.stylize("yellow")
        self.console.print(t)
 
    def tool_result(self, success: bool, display: str, duration_ms: float) -> None:
        mark = "[green]✓[/]" if success else "[red]✗[/]"
        # ``display`` is built from tool output (filenames, error
        # messages, command summaries) which routinely contain ``[``
        # characters Rich would interpret as markup.  Escape so a path
        # like ``[abc].py`` doesn't crash or render as a broken style.
        self.console.print(
            f"  {mark} {rich_escape(display)} [dim]({duration_ms:.1f}ms)[/]"
        )

    def command_output(self, data: dict[str, object]) -> None:
        """Show a compact preview of command stdout/stderr."""
        max_lines = 30
        stdout = str(data.get("stdout", "")).rstrip()
        stderr = str(data.get("stderr", "")).rstrip()
        if stdout:
            lines = stdout.splitlines()
            for line in lines[:max_lines]:
                self.console.print(Text(f"    {line}"))
            if len(lines) > max_lines:
                self.console.print(f"    [dim]… {len(lines) - max_lines} more lines[/]")
        if stderr:
            lines = stderr.splitlines()
            for line in lines[:10]:
                self.console.print(Text(f"    {line}", style="dim red"))
            if len(lines) > 10:
                self.console.print(f"    [dim red]… {len(lines) - 10} more lines[/]")
 
    def edit_diff(self, path: str, old: str, new: str) -> None:
        diff = list(
            difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm="", n=2
            )
        )
        for line in diff[:12]:
            # Diff lines come from arbitrary file content — escape Rich
            # markup so e.g. ``+ list[int]`` doesn't blow up the parser.
            safe = rich_escape(line)
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(f"  [green]{safe}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(f"  [red]{safe}[/]")
            elif line.startswith("@@"):
                self.console.print(f"  [dim]{safe}[/]")

    def write_preview(self, path: str, content: str) -> None:
        lines = content.splitlines()
        snippet = lines if len(lines) <= 6 else lines[:3] + ["  ..."] + lines[-3:]
        for line in snippet:
            # Preview shows arbitrary file content — escape Rich markup
            # so brackets in code don't crash the renderer.
            self.console.print(f"  [dim]│[/] {rich_escape(line)}")
 
    def streaming_text_chunk(self, s: str) -> None:
        """Accumulate text chunks and render as streaming markdown.

        Uses Rich Live for smooth incremental rendering that updates
        in place rather than printing each chunk below previous output.
        """
        if not s:
            return
        # First chunk supersedes the thinking spinner (if any) — both
        # share the Live channel and only one can own the screen rows.
        if self._spinner is not None:
            self.stop_thinking()
        self._stream_buffer += s

        new_render = Markdown(self._stream_buffer)
        self._live_render = new_render
        if not self._use_live:
            self._use_live = True
            self._live = Live(
                new_render,
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start()
        elif self._live is not None:
            # rich.markdown.Markdown has no .update(); swap the renderable
            # on the Live object instead.
            self._live.update(new_render, refresh=True)

    def flush_streaming_text(self) -> None:
        """Finalize streaming text output.

        Stops the transient Live display and prints the final rendered markdown
        once as a permanent output. Always clears the buffer so successive
        turns don't re-render previously streamed text.
        """
        # A spinner shares the Live channel with streaming text — drop it
        # first so a flush in the middle of a wait doesn't leave the
        # spinner thread refreshing into a stopped console.
        self.stop_thinking()
        final_render = self._live_render
        if self._live is not None:
            self._live.stop()
        # transient=True clears the live area on stop, so print the final
        # frame once for permanent display.  Use ``self._stream_buffer``
        # (not ``.strip()``) so streams that ended on whitespace — which
        # the user *did* see flicker on screen — still get a permanent
        # frame instead of vanishing when Live wipes the area.
        if final_render is not None and self._stream_buffer:
            self.console.print(final_render)
        self._live = None
        self._live_render = None
        self._stream_buffer = ""
        self._use_live = False

    def reset_streaming(self) -> None:
        """Drop any in-flight streaming state WITHOUT printing.

        Used by the client retry path: when a stream drops mid-flight and
        tenacity restarts ``complete()``, the second attempt's chunks
        would otherwise be appended to the partial first attempt's
        buffer, producing garbled output. Call this before the retry
        runs so the screen and buffer are both clean.
        """
        self.stop_thinking()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001
                pass
        self._live = None
        self._live_render = None
        self._stream_buffer = ""
        self._use_live = False

    def start_thinking(self, label: str = "thinking") -> None:
        """Start a spinner while waiting for the first stream chunk.

        Idempotent — calling twice is a no-op. Automatically stopped by
        the first ``streaming_text_chunk`` (so a model that streams text
        immediately doesn't fight with the spinner) and by
        ``flush_streaming_text`` / ``reset_streaming``.
        """
        if self._spinner is not None or self._live is not None:
            return
        spinner = Spinner("dots", text=Text(f" {label}…", style="dim"))
        self._spinner = Live(
            spinner,
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        try:
            self._spinner.start()
        except Exception:  # noqa: BLE001
            # Headless console / no TTY — silently downgrade.
            self._spinner = None

    def stop_thinking(self) -> None:
        """Stop the thinking spinner if one is running. Safe to call twice."""
        if self._spinner is None:
            return
        try:
            self._spinner.stop()
        except Exception:  # noqa: BLE001
            pass
        self._spinner = None

    def nudge(self, content: str) -> None:
        """Display a system nudge so the user can see what the harness is
        telling the model.

        Nudges are normally injected as ``{"role": "user", "content":
        "[system] ..."}`` messages — invisible to the user, even though
        the agent's behavior changes in response. Showing them inline
        closes the opacity gap: the user can correlate "agent suddenly
        switched approach" with the specific instruction it received.
        """
        if not content:
            return
        # Strip leading "[system] " marker since we render with our own.
        stripped = content
        if stripped.startswith("[system] "):
            stripped = stripped[len("[system] "):]
        # Cap each nudge so a long correction doesn't dominate the screen.
        # Most nudges are 1–4 short lines; 600 chars covers them and
        # truncates the rare wall-of-text.
        if len(stripped) > 600:
            stripped = stripped[:600].rstrip() + " …"
        # Drop any in-flight live region so the nudge isn't overwritten
        # when the spinner / streaming live refreshes.
        self.stop_thinking()
        # Single bordered line per nudge so the user can scan them quickly.
        self.console.print(
            Panel(
                rich_escape(stripped),
                title="◆ system nudge",
                title_align="left",
                border_style="dim yellow",
                padding=(0, 1),
            )
        )
 
    def info(self, s: str) -> None:
        self.console.print(f"[dim]{s}[/]")
 
    def warn(self, s: str) -> None:
        self.console.print(f"[yellow]! {s}[/]")
 
    def error(self, s: str) -> None:
        self.console.print(f"[red]✗ {s}[/]")

    def plan_panel(self, data: dict) -> None:
        """Render a structured plan in a Rich panel.

        All free-text fields here (plan/problem/solution/step
        descriptions, file paths) come from the LLM or user input and
        may contain ``[`` characters that Rich would otherwise treat as
        broken markup.  Escape every interpolation that isn't a literal
        style tag.
        """
        lines: list[str] = []

        if data.get("plan"):
            lines.append(f"[bold]{rich_escape(str(data['plan']))}[/]")
            lines.append("")

        lines.append(f"[bold red]Problem:[/]  {rich_escape(str(data.get('problem', '')))}")
        lines.append(f"[bold green]Solution:[/] {rich_escape(str(data.get('solution', '')))}")
        lines.append("")
        lines.append("[bold yellow]Steps:[/]")
        for i, step in enumerate(data.get("steps", []), 1):
            desc = step if isinstance(step, str) else step.get("description", "")
            status = "" if isinstance(step, str) else step.get("status", "pending")
            raw_icon = STATUS_ICONS.get(status, "○")
            color = {"done": "green", "in-progress": "cyan", "skipped": "dim", "blocked": "red"}.get(status, "dim")
            status_icon = f"[{color}]{raw_icon}[/{color}]"
            lines.append(f"  {status_icon} {i}. {rich_escape(str(desc))}")

        if data.get("files_to_create"):
            lines.append("")
            lines.append("[bold blue]Create:[/]")
            for f in data["files_to_create"]:
                lines.append(f"  [green]+[/] {rich_escape(str(f))}")

        if data.get("files_to_modify"):
            lines.append("")
            lines.append("[bold blue]Modify:[/]")
            for f in data["files_to_modify"]:
                lines.append(f"  [yellow]~[/] {rich_escape(str(f))}")

        self.console.print(Panel("\n".join(lines), title="📋 Plan", border_style="cyan"))

    def plan_progress(self, steps: list[dict]) -> None:
        """Show a compact progress line for the active plan.

        The count breakdown matches the bar so a user looking at
        ``▓▓░░…`` doesn't see ``0/7 resolved`` alongside it. We list
        whichever non-zero buckets exist (done / in-progress / blocked
        / skipped) plus the unresolved-pending tail.
        """
        total = len(steps)
        done = sum(1 for s in steps if s.get("status") == "done")
        skipped = sum(1 for s in steps if s.get("status") == "skipped")
        in_prog = sum(1 for s in steps if s.get("status") == "in-progress")
        blocked = sum(1 for s in steps if s.get("status") == "blocked")
        pending = max(0, total - done - skipped - in_prog - blocked)
        if total:
            bar_done = int(20 * done / total)
            bar_skip = int(20 * skipped / total)
            bar_active = int(20 * in_prog / total)
            bar_block = int(20 * blocked / total)
        else:
            bar_done = bar_skip = bar_active = bar_block = 0
        bar_empty = max(0, 20 - bar_done - bar_skip - bar_active - bar_block)
        bar = (
            "[green]█[/]" * bar_done
            + "[cyan]▓[/]" * bar_active
            + "[red]▓[/]" * bar_block
            + "[dim]▒[/]" * bar_skip
            + "[dim]░[/]" * bar_empty
        )
        bits: list[str] = []
        if done:
            bits.append(f"[green]{done} done[/]")
        if in_prog:
            bits.append(f"[cyan]{in_prog} in-progress[/]")
        if blocked:
            bits.append(f"[red]{blocked} blocked[/]")
        if skipped:
            bits.append(f"[dim]{skipped} skipped[/]")
        if pending:
            bits.append(f"[dim]{pending} pending[/]")
        breakdown = " · ".join(bits) if bits else f"{total} pending"
        self.console.print(f"  plan: {bar}  {breakdown}  ({total} total)")
 
    def summary(self, turns: int, elapsed_s: float) -> None:
        s = self.stats
        cw = s.context_window
        prompt_str = fmt_tokens(s.prompt_tokens, cw)
        comp_str = fmt_tokens(s.completion_tokens)
        lines = [
            f"turns: {turns}  |  elapsed: {elapsed_s:.1f}s  |  prompt: {prompt_str}  |  completion: {comp_str}",
        ]
        if s.files_created:
            lines.append(f"created: {', '.join(sorted(s.files_created))}")
        if s.files_edited:
            lines.append(f"edited:  {', '.join(sorted(s.files_edited))}")
        if s.commands_run:
            lines.append(f"commands: {s.commands_run}")
        self.console.print(Panel("\n".join(lines), title="✓ done", border_style="green"))

    def status(self, mode: str) -> None:
        """Display current configuration and tool availability."""
        allowed = get_allowed_tools(mode)
        
        self.console.rule(f"[bold]{mode.upper()} MODE[/]", style=MODE_COLORS.get(mode, "dim"))
        
        s = self.stats
        token_str = fmt_tokens(s.prompt_tokens + s.completion_tokens, s.context_window)
        lines = [
            f"mode:     {mode}",
            f"tokens:   {token_str}",
        ]
        
        if mode == "plan":
            lines.append("tools:    read-only + read-only shell allowlist")
            lines.append(f"  allowed: {', '.join(sorted(allowed))}")
            lines.append("  run_command: ls, cat, grep, find, git log/status/diff, pytest --collect-only, ...")
        elif mode == "edits":
            lines.append("tools:    read + write")
            lines.append(f"  allowed: {', '.join(sorted(allowed))}")
            lines.append("  denied:  run_command (requires prompt)")
        else:  # yolo
            lines.append("tools:    all (unrestricted)")
        
        self.console.print("\n".join(lines))
