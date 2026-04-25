"""Terminal output — turn headers, tool status lines, diff previews, summary box."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

MODE_COLORS = {"plan": "ansicyan", "edits": "ansigreen", "yolo": "ansimagenta"}


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


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
        # Streaming markdown state
        self._stream_buffer: str = ""
        self._live_render: Markdown | None = None
        self._live: Live | None = None
        self._use_live: bool = False

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

    def turn_header(self, turn: int, max_turns: int, tool_name: str, brief: str) -> None:
        icon = ICONS.get(tool_name, "•")
        self.console.print(f"[dim]\\[Turn {turn}/{max_turns}][/] {icon} {tool_name} [dim]{brief}[/]")

    def command_line(self, command: str) -> None:
        """Show the full shell command on its own line (markup-safe)."""
        t = Text(f"! {command}")
        t.stylize("yellow")
        self.console.print(t)
 
    def tool_result(self, success: bool, display: str, duration_ms: float) -> None:
        mark = "[green]✓[/]" if success else "[red]✗[/]"
        self.console.print(f"  {mark} {display} [dim]({duration_ms:.1f}ms)[/]")

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
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(f"  [green]{line}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(f"  [red]{line}[/]")
            elif line.startswith("@@"):
                self.console.print(f"  [dim]{line}[/]")
 
    def write_preview(self, path: str, content: str) -> None:
        lines = content.splitlines()
        snippet = lines if len(lines) <= 6 else lines[:3] + ["  ..."] + lines[-3:]
        for line in snippet:
            self.console.print(f"  [dim]│[/] {line}")
 
    def text(self, s: str) -> None:
        if s.strip():
            self.console.print(Text(s))
 
    # Streaming markdown state (initialized in __init__)

    def streaming_text_chunk(self, s: str) -> None:
        """Accumulate text chunks and render as streaming markdown.
        
        Uses Rich Live for smooth incremental rendering that updates
        in place rather than printing each chunk below previous output.
        """
        self._stream_buffer += s
        
        if not self._use_live:
            # First chunk: start Live rendering
            self._use_live = True
            self._live_render = Markdown(self._stream_buffer)
            self._live = Live(
                self._live_render,
                console=self.console,
                refresh_per_second=12,
            )
            self._live.start()
        else:
            # Update existing Live display
            if self._live_render is not None:
                self._live_render.update(Markdown(self._stream_buffer))
            if self._live is not None:
                self._live.refresh()

    def flush_streaming_text(self) -> None:
        """Finalize streaming text output. Call when a prose response completes.
        
        Stops the Live display and prints the final rendered markdown
        as a permanent output.
        """
        if self._live is not None:
            # Stop Live and render final state permanently
            if self._live_render is not None:
                self.console.print()  # blank line before
                self.console.print(self._live_render)
            self._live.stop()
            self._live = None
            self._live_render = None
        self._stream_buffer = ""
        self._use_live = False
 
    def info(self, s: str) -> None:
        self.console.print(f"[dim]{s}[/]")
 
    def warn(self, s: str) -> None:
        self.console.print(f"[yellow]! {s}[/]")
 
    def error(self, s: str) -> None:
        self.console.print(f"[red]✗ {s}[/]")

    def plan_panel(self, data: dict) -> None:
        """Render a structured plan in a Rich panel."""
        lines: list[str] = []

        if data.get("plan"):
            lines.append(f"[bold]{data['plan']}[/]")
            lines.append("")

        lines.append(f"[bold red]Problem:[/]  {data.get('problem', '')}")
        lines.append(f"[bold green]Solution:[/] {data.get('solution', '')}")
        lines.append("")
        lines.append("[bold yellow]Steps:[/]")
        from squishy.plan_state import STATUS_ICONS
        for i, step in enumerate(data.get("steps", []), 1):
            desc = step if isinstance(step, str) else step.get("description", "")
            status = "" if isinstance(step, str) else step.get("status", "pending")
            raw_icon = STATUS_ICONS.get(status, "○")
            color = {"done": "green", "in-progress": "cyan", "skipped": "dim", "blocked": "red"}.get(status, "dim")
            status_icon = f"[{color}]{raw_icon}[/{color}]"
            lines.append(f"  {status_icon} {i}. {desc}")

        if data.get("files_to_create"):
            lines.append("")
            lines.append("[bold blue]Create:[/]")
            for f in data["files_to_create"]:
                lines.append(f"  [green]+[/] {f}")

        if data.get("files_to_modify"):
            lines.append("")
            lines.append("[bold blue]Modify:[/]")
            for f in data["files_to_modify"]:
                lines.append(f"  [yellow]~[/] {f}")

        self.console.print(Panel("\n".join(lines), title="📋 Plan", border_style="cyan"))

    def plan_progress(self, steps: list[dict]) -> None:
        """Show a compact progress line for the active plan."""
        total = len(steps)
        done = sum(1 for s in steps if s.get("status") == "done")
        in_prog = sum(1 for s in steps if s.get("status") == "in-progress")
        bar_filled = int(20 * done / total) if total else 0
        bar_active = int(20 * in_prog / total) if total else 0
        bar_empty = 20 - bar_filled - bar_active
        bar = "[green]█[/]" * bar_filled + "[cyan]▓[/]" * bar_active + "[dim]░[/]" * bar_empty
        self.console.print(f"  plan: {bar} {done}/{total} steps done")
 
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
        from squishy.tool_restrictions import get_allowed_tools

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

    def progress(self, current: int, total: int, message: str = "") -> None:
        """Display progress indicator."""
        percent = 100 if total == 0 else (current * 100) // total
        
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        if message:
            self.console.print(f"[dim]{message}[/] {bar} {percent}% ({current}/{total})")
        else:
            self.console.print(f"{bar} {percent}% ({current}/{total})")

