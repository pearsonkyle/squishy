"""Gitignore-aware filtering for the index walker.

Builds a single matcher from every ``.gitignore`` it finds in the tree —
not just the top-level one — so nested rules work the way ``git`` itself
treats them. Patterns from a ``foo/.gitignore`` only apply to paths under
``foo/``; root-level rules apply everywhere.

Uses ``pathspec`` (the same library black/dvc/etc. use) so glob support
(``**``, character classes, negation, directory-only patterns, anchored
vs. floating, ``!`` re-includes) matches git's own semantics.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pathspec import PathSpec

# pathspec 1.x emits a DeprecationWarning when you import the gitignore
# pattern class — the replacement (GitIgnoreSpecPattern) doesn't exist
# yet on the versions we pin against. Silence it at the import site
# rather than letting it leak into every caller's test output.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern


class GitignoreFilter:
    """Aggregate matcher for one repo's ``.gitignore`` files.

    Each ``.gitignore`` we see contributes a (anchor_dir, PathSpec) pair.
    When matching a path we test against every spec whose anchor dir is
    an ancestor of the path, prefixing the path's relative segment so a
    rule like ``build/`` in ``foo/.gitignore`` only matches ``foo/build``,
    not a top-level ``build``.
    """

    __slots__ = ("_specs",)

    def __init__(self) -> None:
        # Each entry is (anchor_posix, spec). anchor_posix is the
        # directory the .gitignore lived in, relative to repo root,
        # using forward slashes. "" for the repo-root .gitignore.
        self._specs: list[tuple[str, PathSpec]] = []

    def add_file(self, gitignore_path: Path, anchor: str) -> None:
        """Read ``gitignore_path`` and register its rules under ``anchor``."""
        try:
            with open(gitignore_path, encoding="utf-8", errors="replace") as f, \
                 warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                spec = PathSpec.from_lines(GitWildMatchPattern, f)
        except OSError:
            return
        if not spec.patterns:
            return
        self._specs.append((anchor.replace(os.sep, "/"), spec))

    def is_ignored(self, rel_posix: str, *, is_dir: bool = False) -> bool:
        """Return True if ``rel_posix`` (forward-slash, repo-root-relative)
        is ignored by any of the loaded ``.gitignore`` files.

        Directory paths should pass ``is_dir=True`` so directory-only
        rules (``build/``) match — git applies these only to dirs.
        """
        # pathspec wants a trailing slash for directory tests so its
        # "trailing-slash means directory-only" patterns match correctly.
        candidate = rel_posix + "/" if (is_dir and not rel_posix.endswith("/")) else rel_posix
        for anchor, spec in self._specs:
            if anchor:
                anchor_prefix = anchor + "/"
                if not candidate.startswith(anchor_prefix) and candidate != anchor:
                    continue
                sub = candidate[len(anchor_prefix):] if candidate.startswith(anchor_prefix) else ""
                if not sub:
                    continue
            else:
                sub = candidate
            if spec.match_file(sub):
                return True
        return False

    @property
    def has_rules(self) -> bool:
        return bool(self._specs)


def discover_gitignores(root: Path, *, skip_dirs: set[str] | None = None) -> GitignoreFilter:
    """Walk ``root`` and load every ``.gitignore`` we encounter.

    ``skip_dirs`` is honoured during the walk so we don't recurse into
    ``.git``, virtualenvs, etc. just to look for gitignore files there.
    """
    skip = skip_dirs or set()
    f = GitignoreFilter()
    for dirpath, dirs, files in os.walk(root):
        # Don't descend into skipped directories.
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        if ".gitignore" in files:
            gi = Path(dirpath) / ".gitignore"
            try:
                anchor = os.path.relpath(dirpath, root).replace(os.sep, "/")
            except ValueError:
                anchor = ""
            if anchor == ".":
                anchor = ""
            f.add_file(gi, anchor)
    return f


__all__ = ["GitignoreFilter", "discover_gitignores"]
