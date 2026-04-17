"""Staleness detector: fresh index → empty; edit after → warns; new file → warns."""
 
from __future__ import annotations
 
import os
import time
from pathlib import Path
 
from squishy.index import build_index, save_index
from squishy.index.staleness import describe_staleness
 
 
def test_fresh_is_silent(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a(): pass\n")
    save_index(str(tmp_path), build_index(str(tmp_path)))
    assert describe_staleness(str(tmp_path)) == ""
 
 
def test_no_index_is_silent(tmp_path: Path) -> None:
    assert describe_staleness(str(tmp_path)) == ""
 
 
def test_edit_after_index_warns(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a(): pass\n")
    save_index(str(tmp_path), build_index(str(tmp_path)))
    # Push mtime forward a lot past grace window.
    future = time.time() + 60
    os.utime(tmp_path / "a.py", (future, future))
    msg = describe_staleness(str(tmp_path))
    assert "out of date" in msg
    assert "a.py" in msg
 
 
def test_new_file_warns(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a(): pass\n")
    save_index(str(tmp_path), build_index(str(tmp_path)))
    (tmp_path / "b.py").write_text("def b(): pass\n")
    msg = describe_staleness(str(tmp_path))
    assert "out of date" in msg
    assert "b.py" in msg
