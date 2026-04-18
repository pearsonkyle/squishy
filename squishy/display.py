"""Terminal output — turn headers, tool status lines, diff previews, summary box."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


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
    def __init__(self) -> None:
        self.console = Console()
        self.stats = Stats()
 
    def banner(self, base_url: str, model: str) -> None:
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
        lines = [
            f"turns: {turns}  |  elapsed: {elapsed_s:.1f}s  |  prompt: {s.prompt_tokens:,}  |  completion: {s.completion_tokens:,}",
        ]
        if s.files_created:
            lines.append(f"created: {', '.join(sorted(s.files_created))}")
        if s.files_edited:
            lines.append(f"edited:  {', '.join(sorted(s.files_edited))}")
        if s.commands_run:
            lines.append(f"commands: {s.commands_run}")
        self.console.print(Panel("\n".join(lines), title="✓ done", border_style="green"))