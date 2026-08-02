"""Single source of truth for environment variables, constants, runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

PermissionMode = Literal["edits", "yolo", "bench"]
MODES: tuple[PermissionMode, ...] = ("edits", "yolo", "bench")
# Modes exposed to the interactive shift-tab cycle. "bench" is for the
# benchmark runner only — it drops the web tools and skips approval prompts —
# so it's excluded here.
INTERACTIVE_MODES: tuple[PermissionMode, ...] = ("edits", "yolo")


@dataclass
class Config:
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "SQUISHY_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        )
    )
    api_key: str = field(
        repr=False,
        default_factory=lambda: os.environ.get(
            "SQUISHY_API_KEY", os.environ.get("OPENAI_API_KEY", "local")
        ),
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SQUISHY_MODEL", "local-model")
    )
    temperature: float = 0.3
    max_tokens: int = 8192
    max_turns: int = 30
    permission_mode: PermissionMode = "edits"
    # "standard" = every tool the mode permits. "minimal" = a shell plus the
    # file primitives (the mini-swe-agent shape). "shell" = run_command alone.
    # Small models generalize better to a tool set they were trained on, and
    # the narrower schema is ~900 fewer tokens on every request.
    tool_profile: str = "standard"
    working_dir: str = field(default_factory=os.getcwd)
    sandbox_image: str = field(
        default_factory=lambda: os.environ.get("SQUISHY_SANDBOX_IMAGE", "python:3.11-slim")
    )
    use_sandbox: bool = False
    thinking: bool = False
    index_concurrency: int = 4
    max_tokens_per_index: int = 100_000
    auto_init: bool = False
    index_summaries: bool = True
    # Agent-loop safety thresholds. Tunable so bench runs can trade off
    # reliability vs. autonomy without code changes.
    max_consecutive_errors: int = 8
    max_history_messages: int = 10
    # Context window in tokens. 0 = auto-detect from the endpoint. Many local
    # servers (LM Studio, llama.cpp) do NOT advertise `context_length`; without
    # a value the compaction safety valve and dynamic history sizing are
    # silently disabled, so `assumed_context_window` is used as the fallback.
    context_window: int = 0
    assumed_context_window: int = 32_768
    max_tool_output_chars: int = 32_000
    compaction_threshold: float = 0.7
    max_system_nudges: int = 8  # cap total nudges to avoid flooding context
    # Session persistence.
    session_dir: str = field(
        default_factory=lambda: os.environ.get(
            "SQUISHY_SESSION_DIR",
            os.path.expanduser("~/.squishy/sessions"),
        )
    )
    save_sessions: bool = True

    def __post_init__(self) -> None:
        from squishy.tool_restrictions import TOOL_PROFILES
        if self.tool_profile not in TOOL_PROFILES:
            raise ValueError(
                f"tool_profile must be one of {sorted(TOOL_PROFILES)}, "
                f"got {self.tool_profile!r}"
            )

    def cycle_mode(self) -> PermissionMode:
        """Advance to the next interactive permission mode (skipping bench).

        If the current mode isn't in the interactive cycle (e.g. ``bench``
        set programmatically by the benchmark runner), land on the first
        interactive mode rather than rotating into another non-interactive
        slot.
        """
        if self.permission_mode in INTERACTIVE_MODES:
            i = (INTERACTIVE_MODES.index(self.permission_mode) + 1) % len(INTERACTIVE_MODES)
            self.permission_mode = INTERACTIVE_MODES[i]
        else:
            self.permission_mode = INTERACTIVE_MODES[0]
        return self.permission_mode
