"""Single source of truth for environment variables, constants, runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

PermissionMode = Literal["plan", "edits", "yolo", "bench"]
MODES: tuple[PermissionMode, ...] = ("plan", "edits", "yolo", "bench")
# Modes exposed to the interactive shift-tab cycle. "bench" is for the
# benchmark runner only — it strips planning tools and enables aggressive
# automation that's not useful at the REPL — so it's excluded here.
INTERACTIVE_MODES: tuple[PermissionMode, ...] = ("plan", "edits", "yolo")


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
    permission_mode: PermissionMode = "plan"
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
    max_plan_nudges: int = 4
    max_plan_investigation_turns: int = 4
    max_recall_skip_turns: int = 2
    max_history_messages: int = 10
    # Context window in tokens. 0 = auto-detect from the endpoint. Many local
    # servers (LM Studio, llama.cpp) do NOT advertise `context_length`; without
    # a value the compaction safety valve and dynamic history sizing are
    # silently disabled, so `assumed_context_window` is used as the fallback.
    context_window: int = 0
    assumed_context_window: int = 32_768
    max_tool_output_chars: int = 32_000
    max_quality_retries: int = 3
    compaction_threshold: float = 0.7
    max_system_nudges: int = 8  # cap total nudges to avoid flooding context
    # Phase-budget thresholds (bench/yolo modes only).
    max_explore_turns: int = 8
    # Turns allowed with no successful edit before `run_command` is removed
    # from the schema, leaving only read/edit tools. Small models otherwise
    # loop on "run the tests" forever and never attempt a fix — a patch that
    # fails tests still beats no patch at all. bench/yolo only; 0 disables.
    max_turns_without_edit: int = 12
    max_plan_turns: int = 3
    max_fix_verify_cycles: int = 6
    # v2 auto-pytest finish gate: cap on how many times the harness will
    # synthesize a pytest run when the agent tries to finish without
    # verifying the F2P tests. Bench mode only.
    max_auto_pytest_runs: int = 2
    # v5 pre-finish F2P partial-pass gate: how many times
    # ``check_finish_plan_gate`` may intercept ``finish_plan`` before
    # releasing.  Bounded so a structurally unrunnable test environment
    # cannot trap the agent.  Bench mode only.
    max_finish_gate_intercepts: int = 2
    # Session persistence.
    session_dir: str = field(
        default_factory=lambda: os.environ.get(
            "SQUISHY_SESSION_DIR",
            os.path.expanduser("~/.squishy/sessions"),
        )
    )
    save_sessions: bool = True

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
