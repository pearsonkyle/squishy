"""Terminal output — turn headers, tool status lines, diff previews, summary box."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from squishy.plan_tracker import PlanTracker


MODE_COLORS = {"plan": "ansicyan", "edits": "ansigreen", "yolo": "ansimagenta"}


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def format_tokens_k(n: int, window: int = 0) -> str:
    """Format a token count as "1.2 K (10%)" or "847 (1%)".

    When ``window`` is zero or negative, the percentage is omitted.
    Percentage is rounded to the nearest integer; values above 999 render
    with one decimal of K.
    """
    if n < 1000:
        head = f"{n}"
    else:
        head = f"{n / 1000:.1f} K"
    if window <= 0:
        return head
    pct = int(round((n / window) * 100))
    return f"{head} ({pct}%)"


ICONS = {
    "read_file": "[cyan]📖[/]",
    "write_file": "[green]✎[/]",
    "edit_file": "[yellow]✏️[/]",
    "list_directory": "[cyan]📁[/]",
    "search_files": "[cyan]🔍[/]",
    "run_command": "[magenta]🔧[/]",
}


@dataclass
class Stats:
    files_created: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    commands_run: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Display:
    def __init__(self, context_window: int = 0) -> None:
        self.console = Console()
        self.stats = Stats()
        self.model: str = ""
        self.context_window = context_window
        self.tracker = PlanTracker()

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
 
    def streaming_text_chunk(self, s: str) -> None:
        self.console.out(s, end="", highlight=False)
 
    def info(self, s: str) -> None:
        self.console.print(f"[dim]{s}[/]")
 
    def warn(self, s: str) -> None:
        self.console.print(f"[yellow]! {s}[/]")
 
    def error(self, s: str) -> None:
        self.console.print(f"[red]✗ {s}[/]")
 
    def summary(self, turns: int, elapsed_s: float) -> None:
        s = self.stats
        w = self.context_window
        lines = [
            f"turns: {turns}  |  elapsed: {elapsed_s:.1f}s  |  "
            f"tokens: {format_tokens_k(s.tokens, w)}  |  "
            f"prompt: {format_tokens_k(s.prompt_tokens, 0)}  |  "
            f"comp: {format_tokens_k(s.completion_tokens, 0)}",
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
        from squishy.tool_restrictions import get_allowed_tools, get_denied_tools

        allowed = get_allowed_tools(mode)
        
        self.console.rule(f"[bold]{mode.upper()} MODE[/]", style=MODE_COLORS.get(mode, "dim"))
        
        lines = [
            f"mode:     {mode}",
            f"tokens:   {format_tokens_k(self.stats.tokens, self.context_window)} total",
        ]
        
        if mode == "plan":
            lines.append("tools:    read-only only")
            lines.append(f"  allowed: {', '.join(sorted(allowed))}")
        elif mode == "edits":
            lines.append("tools:    read + write")
            lines.append(f"  allowed: {', '.join(sorted(allowed))}")
            lines.append("  denied:  run_command (requires prompt)")
        else:  # yolo
            lines.append("tools:    all (unrestricted)")
        
        self.console.print("\n".join(lines))

    def plan_status(self) -> None:
        """Render the active plan's step list with status glyphs."""
        t = self.tracker
        if not t.is_active():
            return
        glyph = {"done": "[green]✓[/]", "in_progress": "[yellow]►[/]", "pending": "[dim]·[/]", "skipped": "[dim]⊘[/]"}
        lines = ["[bold]Plan progress[/]"]
        for s in t.steps:
            lines.append(f"  {glyph.get(s.status, '·')} {s.index}. {s.description}")
        self.console.print(Panel("\n".join(lines), border_style="cyan"))

    def progress(self, current: int, total: int, message: str = "") -> None:
        """Display progress indicator."""
        if total == 0:
            percent = 100
        else:
            percent = (current * 100) // total
        
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        if message:
            self.console.print(f"[dim]{message}[/] {bar} {percent}% ({current}/{total})")
        else:
            self.console.print(f"{bar} {percent}% ({current}/{total})")

    def tool_categories(self, mode: str) -> None:
        """Display tools organized by category for given mode."""
        from squishy.tool_restrictions import get_allowed_tools, get_denied_tools
        
        allowed = get_allowed_tools(mode)
        denied = get_denied_tools(mode)
        
        self.console.rule("TOOL CATEGORIES", style="blue")
        
        if allowed:
            self.console.print(f"\n[bold green]ALLOWED IN {mode.upper()}[/]:")
            for tool in sorted(allowed):
                self.console.print(f"  ✓ {tool}")
        
        if denied:
            self.console.print(f"\n[bold red]DENIED IN {mode.upper()}[/]:")
            for tool in sorted(denied):
                self.console.print(f"  ✗ {tool}")