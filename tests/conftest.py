from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from squishy.client import CompletionResult
from squishy.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)


@dataclass
class FakeClient:
    """Scripted fake LLM client for agent loop tests.

    Pops ``CompletionResult`` objects from ``script`` in order.  When the
    script is exhausted it returns a bare ``CompletionResult(text="done.",
    tool_calls=[])`` so tests don't have to pad the script.
    """

    script: list[CompletionResult]
    calls_seen: list[list[dict[str, Any]]] = field(default_factory=list)
    # Schemas offered per request — lets tests assert on what the model was
    # actually shown, not just what it was asked.
    tools_seen: list[list[dict[str, Any]]] = field(default_factory=list)
    _i: int = 0

    async def health(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: Any = None,
        on_retry: Any = None,
    ) -> CompletionResult:
        self.calls_seen.append(list(messages))
        self.tools_seen.append(list(tools or []))
        if self._i >= len(self.script):
            return CompletionResult(text="done.", tool_calls=[])
        result = self.script[self._i]
        self._i += 1
        return result


# A tiny importable package, used by every test that needs a real code graph.
APP_PY = '''\
"""Demo application entry point."""

from demo.services import UserService
from demo.utils import slugify


def main() -> None:
    """Run the demo app."""
    service = UserService()
    print(service.greet("Ada"))
    print(slugify("Hello World"))


if __name__ == "__main__":
    main()
'''

SERVICES_PY = '''\
"""Service layer."""

from demo.utils import slugify


class UserService:
    """Handles user-facing operations."""

    def greet(self, name: str) -> str:
        """Return a greeting for ``name``."""
        return f"Hello, {self.normalize(name)}!"

    def normalize(self, name: str) -> str:
        """Normalize a user name."""
        return slugify(name).title()


class AdminService(UserService):
    """Admin-only operations."""

    def ban(self, name: str) -> str:
        return f"banned:{self.normalize(name)}"
'''

UTILS_PY = '''\
"""Small utilities."""


def slugify(text: str) -> str:
    """Lowercase and dash-join ``text``."""
    return "-".join(text.lower().split())


def unused_helper() -> int:
    return 42
'''


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    """Write a tiny importable package to disk and return its root."""
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "app.py").write_text(APP_PY, encoding="utf-8")
    (pkg / "services.py").write_text(SERVICES_PY, encoding="utf-8")
    (pkg / "utils.py").write_text(UTILS_PY, encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path
