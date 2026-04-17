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
