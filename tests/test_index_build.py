"""Index build: structure, incremental reuse, hash-based invalidation."""
 
from __future__ import annotations
 
from pathlib import Path
 
from squishy.index import build_index, load_index, save_index
 
 
def _setup_repo(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text('"""A module."""\ndef alpha(): return 1\n')
    (root / "pkg" / "b.py").write_text('"""B module."""\nclass Widget: pass\n')
    (root / "README.md").write_text("# demo")
 
 
def test_tree_shape(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    idx = build_index(str(tmp_path))
    assert idx.root.kind == "repo"
    names = {c.name for c in idx.root.children}
    assert "README.md" in names
    assert "pkg" in names
    pkg = next(c for c in idx.root.children if c.name == "pkg")
    files = {f.name for f in pkg.children}
    assert {"a.py", "b.py"} <= files
 
 
def test_symbols_counted(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    idx = build_index(str(tmp_path))
    symbols = [n for n in idx.root.walk() if n.kind in ("class", "function", "method")]
    names = {s.name for s in symbols}
    assert "alpha" in names
    assert "Widget" in names
    assert idx.meta.stats["files"] == 3
    assert idx.meta.stats["symbols"] >= 2
 
 
def test_incremental_reuse_by_hash(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    idx1 = build_index(str(tmp_path))
    # Populate a summary on a.py's node so we can detect reuse.
    a = idx1.find_file("pkg/a.py")
    assert a is not None
    a.summary = "custom summary"
 
    # Rebuild: unchanged file should keep its summary.
    idx2 = build_index(str(tmp_path), prior=idx1)
    a2 = idx2.find_file("pkg/a.py")
    assert a2 is not None
    assert a2.summary == "custom summary"
 
    # Modify the file: summary should be discarded.
    (tmp_path / "pkg" / "a.py").write_text('"""A module, edited."""\ndef alpha(): return 2\n')
    idx3 = build_index(str(tmp_path), prior=idx2)
    a3 = idx3.find_file("pkg/a.py")
    assert a3 is not None
    assert a3.summary != "custom summary"
 
 
def test_roundtrip_save_load(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    idx = build_index(str(tmp_path))
    save_index(str(tmp_path), idx)
    loaded = load_index(str(tmp_path))
    assert loaded is not None
    assert loaded.meta.stats["files"] == idx.meta.stats["files"]
    assert {n.name for n in loaded.root.walk()} == {n.name for n in idx.root.walk()}
 
 
def test_deleted_file_dropped(tmp_path: Path) -> None:
    _setup_repo(tmp_path)
    idx1 = build_index(str(tmp_path))
    (tmp_path / "pkg" / "b.py").unlink()
    idx2 = build_index(str(tmp_path), prior=idx1)
    assert idx2.find_file("pkg/b.py") is None
    assert idx2.find_file("pkg/a.py") is not None
