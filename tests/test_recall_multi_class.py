"""F4: ``_recall_from_index`` must surface sibling classes in the same file.

The v27.1 scico-561 failure was that the recall section listed only
``XRayTransform2D`` from ``scico/linop/xray/_xray.py`` even though
``XRayTransform3D`` lived in the same file and needed the same fix.
The agent fixed the 2D class, ran the 2D test, saw it pass, and shipped
without ever opening the 3D class.

F4 changes ``_recall_from_index`` so that whenever a class node makes it
into the top results, the other class siblings in the same file are
appended too (up to the overall limit).
"""
from __future__ import annotations

import json
from pathlib import Path

from squishy.bench.swebench import _recall_from_index


def _write_index(workspace: Path, files: list[dict]) -> None:
    """Materialise a minimal ``.squishy/index.json`` with the given file
    nodes.  Each ``files`` entry is ``{"path": ..., "classes": [name, ...]}``.

    The class summaries are seeded from the file path so the recall
    scorer has something to match the query against.
    """
    sq = workspace / ".squishy"
    sq.mkdir(parents=True, exist_ok=True)
    children: list[dict] = []
    for f in files:
        path = f["path"]
        klasses = []
        for i, cls_name in enumerate(f["classes"]):
            klasses.append({
                "id": f"{path}:{cls_name}",
                "kind": "class",
                "name": cls_name,
                "path": path,
                "start_line": 10 + i * 50,
                "end_line": 50 + i * 50,
                "summary": f"{cls_name} — class for {f.get('summary', 'something')}",
            })
        children.append({
            "id": path,
            "kind": "file",
            "name": Path(path).name,
            "path": path,
            "summary": f.get("summary", ""),
            "children": klasses,
        })
    index = {
        "root": {
            "id": "",
            "kind": "repo",
            "name": "test-repo",
            "path": "",
            "children": children,
        },
        "meta": {
            "generated_at": 0.0,
            "file_hashes": {},
            "model": "",
            "squishy_version": "test",
            "stats": {},
            "summary_stats": {},
        },
    }
    (sq / "index.json").write_text(json.dumps(index))


def test_recall_surfaces_sibling_classes_in_same_file(tmp_path: Path) -> None:
    """When the index returns one class from a multi-class file, the
    other classes in that file should also appear in the recall output.
    This is the regression test for v27.1 scico-561."""
    _write_index(tmp_path, [
        {"path": "scico/linop/xray/_xray.py",
         "summary": "X-ray projection operators for adjoint matching",
         "classes": ["XRayTransform2D", "XRayTransform3D"]},
    ])
    # Query that scores XRayTransform2D high but not XRayTransform3D.
    out = _recall_from_index(
        tmp_path,
        "XRayTransform2D fails adjoint test for non-square detector",
        limit=8,
    )
    names = {p["name"] for p in out}
    assert "XRayTransform2D" in names, f"expected the targeted class, got {names}"
    assert "XRayTransform3D" in names, (
        "F4: sibling classes in the same file must be surfaced too "
        "(scico-561 regression)"
    )


def test_recall_does_not_surface_siblings_for_function_hits(tmp_path: Path) -> None:
    """Sibling-surfacing is class-only — function nodes don't trigger it
    (functions are usually independent; classes more often share a fix)."""
    sq = tmp_path / ".squishy"
    sq.mkdir(parents=True, exist_ok=True)
    file_node = {
        "id": "utils.py",
        "kind": "file",
        "name": "utils.py",
        "path": "utils.py",
        "children": [
            {"id": "utils.py:helper_a", "kind": "function", "name": "helper_a",
             "path": "utils.py", "summary": "helper_a does target_query things"},
            {"id": "utils.py:helper_b", "kind": "function", "name": "helper_b",
             "path": "utils.py", "summary": "unrelated helper"},
        ],
    }
    index = {
        "root": {"id": "", "kind": "repo", "name": "r", "path": "",
                 "children": [file_node]},
        "meta": {"generated_at": 0.0, "file_hashes": {}, "model": "",
                 "squishy_version": "t", "stats": {}, "summary_stats": {}},
    }
    (sq / "index.json").write_text(json.dumps(index))
    out = _recall_from_index(tmp_path, "target_query", limit=8)
    names = {p["name"] for p in out}
    assert "helper_a" in names
    # helper_b matches nothing in the query and should NOT be auto-surfaced
    # just because helper_a was — that's a function, not a class.
    assert "helper_b" not in names


def test_recall_sibling_surfacing_respects_limit(tmp_path: Path) -> None:
    """When sibling-surfacing would exceed ``limit``, stop at the cap."""
    _write_index(tmp_path, [
        {"path": "big.py",
         "summary": "module with many classes one matches target",
         "classes": [f"Klass{i}" for i in range(12)] + ["TargetClass"]},
    ])
    out = _recall_from_index(tmp_path, "TargetClass", limit=5)
    assert len(out) <= 5
    names = [p["name"] for p in out]
    assert "TargetClass" in names
