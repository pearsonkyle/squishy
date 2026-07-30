"""Recovery from near-miss paths.

Motivated by a live SWE-rebench run: the repo lived at `/vyper` and its package
was also `vyper/`, so the model asked for `vyper/vyper/ast/natspec.py`. A bare
"file not found" gave it nothing to correct, and it burned ~20 turns.
"""
from __future__ import annotations

import pytest

from squishy.tools import dispatch
from squishy.tools.fs import _path_candidates


@pytest.fixture
def repo(tmp_path):
    """A checkout whose directory name matches its package name."""
    root = tmp_path / "vyper"
    (root / "vyper" / "ast").mkdir(parents=True)
    (root / "vyper" / "ast" / "natspec.py").write_text("def parse():\n    return 1\n")
    (root / "pyproject.toml").write_text("[project]\nname='vyper'\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "natspec.py").write_text("decoy\n")
    return root


def test_strips_duplicated_repo_name(repo):
    assert _path_candidates("vyper/vyper/ast/natspec.py", str(repo)) == \
        ["vyper/ast/natspec.py"]


def test_reinterprets_rooted_path_as_repo_relative(repo):
    assert "vyper/ast/natspec.py" in _path_candidates(
        "/vyper/vyper/ast/natspec.py", str(repo))


def test_finds_file_by_basename(repo):
    assert _path_candidates("src/natspec.py", str(repo)) == ["vyper/ast/natspec.py"]


def test_basename_search_skips_vendored_dirs(repo):
    # node_modules/natspec.py must never be offered as the fix.
    assert all("node_modules" not in c
               for c in _path_candidates("src/natspec.py", str(repo)))


def test_no_candidates_for_a_genuinely_absent_file(repo):
    assert _path_candidates("nowhere/absent_thing.py", str(repo)) == []


async def test_read_file_auto_corrects_a_unique_match(ctx, repo):
    ctx.working_dir = str(repo)
    res = await dispatch("read_file", {"path": "vyper/vyper/ast/natspec.py"}, ctx)
    assert res.success
    assert "def parse()" in res.data["content"]
    # It must report the path it actually read, not the one it was given.
    assert res.data["path"] == "vyper/ast/natspec.py"
    assert "vyper/ast/natspec.py" in res.data["note"]


async def test_read_file_on_a_directory_says_so(ctx, repo):
    ctx.working_dir = str(repo)
    res = await dispatch("read_file", {"path": "vyper"}, ctx)
    assert not res.success
    assert "is a directory" in res.error
    assert "ast" in res.error  # lists what's inside


async def test_read_file_miss_names_the_working_dir(ctx, repo):
    ctx.working_dir = str(repo)
    res = await dispatch("read_file", {"path": "nowhere/absent_thing.py"}, ctx)
    assert not res.success
    assert str(repo) in res.error


async def test_edit_file_suggests_but_never_guesses(ctx, repo):
    """Editing the wrong file silently is worse than failing loudly."""
    ctx.working_dir = str(repo)
    res = await dispatch("edit_file", {
        "path": "vyper/vyper/ast/natspec.py",
        "old_str": "return 1", "new_str": "return 2",
    }, ctx)
    assert not res.success
    assert "vyper/ast/natspec.py" in res.error
    # The real file is untouched.
    assert (repo / "vyper" / "ast" / "natspec.py").read_text().endswith("return 1\n")


async def test_write_file_refuses_to_create_a_stray_duplicate(ctx, repo):
    ctx.working_dir = str(repo)
    res = await dispatch("write_file", {
        "path": "vyper/pyproject.toml", "content": "[project]\n",
    }, ctx)
    assert not res.success
    assert "pyproject.toml" in res.error
    assert not (repo / "vyper" / "pyproject.toml").exists()


async def test_write_file_still_creates_genuinely_new_files(ctx, repo):
    ctx.working_dir = str(repo)
    res = await dispatch("write_file", {
        "path": "vyper/ast/brand_new.py", "content": "x = 1\n",
    }, ctx)
    assert res.success, res.error
    assert (repo / "vyper" / "ast" / "brand_new.py").read_text() == "x = 1\n"
