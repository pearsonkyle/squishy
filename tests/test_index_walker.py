"""Walker: SKIP_DIRS, .gitignore, file cap."""
 
from __future__ import annotations
 
from pathlib import Path
 
from squishy.index.walker import FILE_CAP, walk_repo
 
 
def _touch(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p
 
 
def test_skip_dirs_and_dotfiles(tmp_path: Path) -> None:
    _touch(tmp_path, "a.py", "# a")
    _touch(tmp_path, "node_modules/bad.js", "x")
    _touch(tmp_path, "__pycache__/bad.pyc", "x")
    _touch(tmp_path, ".git/HEAD", "x")
    _touch(tmp_path, ".venv/x.py", "x")
    _touch(tmp_path, "src/b.py", "# b")
    records, hit_cap = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "a.py" in paths
    assert "src/b.py" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any(p.startswith(".git") for p in paths)
    assert not any(".venv" in p for p in paths)
    assert not hit_cap
 
 
def test_gitignore_excludes(tmp_path: Path) -> None:
    _touch(tmp_path, ".gitignore", "build/\nsecret.py\n")
    _touch(tmp_path, "a.py", "# a")
    _touch(tmp_path, "secret.py", "# no")
    _touch(tmp_path, "build/c.py", "# no")
    _touch(tmp_path, "src/keep.py", "# yes")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "a.py" in paths
    assert "src/keep.py" in paths
    assert "secret.py" not in paths
    assert "build/c.py" not in paths
 
 
def test_ext_filter_keeps_text_skips_binary(tmp_path: Path) -> None:
    _touch(tmp_path, "a.py", "# a")
    _touch(tmp_path, "readme.md", "# md")
    _touch(tmp_path, "image.png", "binary")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "a.py" in paths
    assert "readme.md" in paths
    assert "image.png" not in paths
 
 
def test_hash_stable(tmp_path: Path) -> None:
    _touch(tmp_path, "a.py", "# a")
    r1, _ = walk_repo(str(tmp_path))
    r2, _ = walk_repo(str(tmp_path))
    assert r1[0].hash == r2[0].hash
 
 
def test_cap_sanity(tmp_path: Path) -> None:
    assert FILE_CAP >= 1000


def test_gitignore_nested_only_applies_to_subtree(tmp_path: Path) -> None:
    """A .gitignore in subpkg/ should only ignore files under subpkg/.
    A top-level pattern with the same name should apply repo-wide."""
    _touch(tmp_path, "secret.py", "x")
    _touch(tmp_path, "subpkg/.gitignore", "secret.py\n")
    _touch(tmp_path, "subpkg/secret.py", "x")
    _touch(tmp_path, "subpkg/keep.py", "x")
    _touch(tmp_path, "other/secret.py", "x")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    # nested rule blocks subpkg/secret.py only
    assert "subpkg/secret.py" not in paths
    assert "subpkg/keep.py" in paths
    # top-level file with same name is untouched (rule was nested)
    assert "secret.py" in paths
    # sibling subtree is untouched
    assert "other/secret.py" in paths


def test_gitignore_double_star_globs(tmp_path: Path) -> None:
    """`**/genned/` should match dirs of that name at any depth.

    (We use 'genned' instead of 'build' because 'build' is always
    pruned by SKIP_DIRS regardless of gitignore.)"""
    _touch(tmp_path, ".gitignore", "**/genned/\n")
    _touch(tmp_path, "genned/a.py", "x")
    _touch(tmp_path, "src/genned/b.py", "x")
    _touch(tmp_path, "src/keep.py", "x")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert not any("genned/" in p for p in paths)
    assert "src/keep.py" in paths


def test_gitignore_negation_reincludes(tmp_path: Path) -> None:
    """A `!pattern` after a broader rule must re-include the path.

    The walker drops files whose extension isn't in TEXT_EXTS, so we
    use .txt (which is) for this test rather than .log (which isn't).
    """
    _touch(tmp_path, ".gitignore", "*.txt\n!important.txt\n")
    _touch(tmp_path, "a.txt", "x")
    _touch(tmp_path, "important.txt", "x")
    _touch(tmp_path, "keep.py", "x")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "a.txt" not in paths
    assert "important.txt" in paths
    assert "keep.py" in paths


def test_gitignore_directory_only_pattern(tmp_path: Path) -> None:
    """`out/` (trailing slash) ignores the dir but not similarly-named
    files. We only check the directory side because the walker drops
    extensionless files anyway."""
    _touch(tmp_path, ".gitignore", "out/\n")
    _touch(tmp_path, "out/x.py", "x")
    _touch(tmp_path, "outfile.txt", "x")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "out/x.py" not in paths
    assert "outfile.txt" in paths


def test_gitignore_anchored_pattern(tmp_path: Path) -> None:
    """A leading `/` anchors to the .gitignore's directory, so `/genned`
    should NOT match `src/genned`."""
    _touch(tmp_path, ".gitignore", "/genned\n")
    _touch(tmp_path, "genned/a.py", "x")
    _touch(tmp_path, "src/genned/b.py", "x")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "genned/a.py" not in paths
    assert "src/genned/b.py" in paths


def test_gitignore_does_not_descend_into_ignored_dir(tmp_path: Path) -> None:
    """If a directory is ignored, we shouldn't even read .gitignore
    files inside it. Cheap correctness + a perf guarantee."""
    _touch(tmp_path, ".gitignore", "genned/\n")
    _touch(tmp_path, "genned/.gitignore", "!keep.py\n")
    _touch(tmp_path, "genned/keep.py", "x")
    records, _ = walk_repo(str(tmp_path))
    paths = {r.path for r in records}
    assert "genned/keep.py" not in paths
