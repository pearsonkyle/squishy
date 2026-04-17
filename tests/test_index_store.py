"""Store: save/load/meta round trip and graceful failure."""
 
from __future__ import annotations
 
import json
from pathlib import Path
 
from squishy.index.model import Index, IndexMeta, Node
from squishy.index.store import index_path, load_index, load_meta, meta_path, save_index
 
 
def _tiny_index() -> Index:
    root = Node(id="repo:.", kind="repo", name="r", path="")
    root.children.append(Node(id="file:a.py", kind="file", name="a.py", path="a.py", hash="deadbeef", summary="hello"))
    return Index(root=root, meta=IndexMeta(generated_at=123.0, file_hashes={"a.py": "deadbeef"}, model="m", stats={"files": 1}))
 
 
def test_round_trip(tmp_path: Path) -> None:
    save_index(str(tmp_path), _tiny_index())
    assert index_path(tmp_path).is_file()
    assert meta_path(tmp_path).is_file()
 
    loaded = load_index(str(tmp_path))
    assert loaded is not None
    assert loaded.root.children[0].hash == "deadbeef"
    assert loaded.meta.file_hashes == {"a.py": "deadbeef"}
    assert loaded.meta.model == "m"
    assert loaded.meta.stats == {"files": 1}
 
 
def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_index(str(tmp_path)) is None
    assert load_meta(str(tmp_path)) is None
 
 
def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    (tmp_path / ".squishy").mkdir()
    (tmp_path / ".squishy" / "index.json").write_text("{not json")
    (tmp_path / ".squishy" / "index.meta.json").write_text("{not json")
    assert load_index(str(tmp_path)) is None
    assert load_meta(str(tmp_path)) is None
 
 
def test_json_is_pretty(tmp_path: Path) -> None:
    save_index(str(tmp_path), _tiny_index())
    raw = index_path(tmp_path).read_text()
    # Pretty-printed = contains newlines and indentation.
    assert "\n" in raw
    data = json.loads(raw)
    assert "root" in data and "meta" in data
