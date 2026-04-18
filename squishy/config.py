"""Single source of truth for environment variables, constants, runtime settings."""
 
from __future__ import annotations
 
import os
from dataclasses import dataclass, field
 
PermissionMode = str  # "plan" | "edits" | "yolo"
MODES: tuple[PermissionMode, ...] = ("plan", "edits", "yolo")
 
 
@dataclass
class Config:
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "SQUISHY_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get(
            "SQUISHY_API_KEY", os.environ.get("OPENAI_API_KEY", "local")
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SQUISHY_MODEL", "local-model")
    )
    temperature: float = 0.3
    max_tokens: int = 8192
    max_turns: int = 30
    permission_mode: PermissionMode = "edits"
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
 
    def cycle_mode(self) -> PermissionMode:
        i = (MODES.index(self.permission_mode) + 1) % len(MODES)
        self.permission_mode = MODES[i]
        return self.permission_mode