"""Shared fixtures: a small synthetic Python repo used across the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

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
