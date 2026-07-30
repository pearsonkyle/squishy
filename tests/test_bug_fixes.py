"""Regression tests for the bug-sweep findings."""

from __future__ import annotations

import pytest

from squishy.file_browser import parse_references_with_missing
from squishy.tools.base import ToolResult, _short_json


# --- #15: _short_json must never grow the payload -------------------------

def test_short_json_limit_zero_returns_empty():
    out = _short_json({"content": "x" * 100}, 0)
    assert out == ""


def test_short_json_negative_limit_returns_empty():
    assert _short_json({"a": 1}, -5) == ""


def test_to_message_zero_limit_does_not_grow():
    payload = {"content": "y" * 200}
    out = ToolResult(True, data=payload).to_message(0)
    assert len(out) == 0


# --- #13: update_plan must reject bool step_index -------------------------

async def test_update_plan_rejects_bool_step_index(tmp_path):
    from squishy.tools.base import ToolContext
    from squishy.tools.plan import _plan_task, _update_plan
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)
    await _plan_task({"problem": "p", "solution": "s", "steps": ["a", "b"]}, ctx)
    ctx.plan.mark_approved()
    r = await _update_plan({"step_index": True, "status": "done"}, ctx)
    assert not r.success
    assert "step_index" in r.error
    # Neither step was mutated.
    assert all(s.status == "pending" for s in ctx.plan.steps)


# --- #9: @-reference regex ignores emails, strips trailing punctuation ----

def test_at_reference_ignores_email(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    found, missing = parse_references_with_missing("ping bob@corp.com then @a.py", str(tmp_path))
    paths = [r.path for r in found]
    assert "a.py" in paths
    assert "corp.com" not in paths and "corp.com" not in missing


def test_at_reference_strips_trailing_punctuation(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    found, missing = parse_references_with_missing("see @a.py, and @a.py.", str(tmp_path))
    paths = {r.path for r in found}
    assert paths == {"a.py"}
    assert missing == []


def test_at_reference_missing_marker_uses_clean_path(tmp_path):
    from squishy.file_browser import inject_references_with_missing
    result, _refs, missing = inject_references_with_missing("open @nope.py.", str(tmp_path))
    assert "nope.py" in missing
    assert "[file not found: nope.py]" in result
    # The trailing sentence period survives outside the marker.
    assert result.rstrip().endswith(".")


# --- #5: model save_note cannot corrupt harness-reserved notes ------------

async def test_save_note_cannot_evict_reserved(tmp_path):
    from squishy.tools.base import ToolContext
    from squishy.tools.scratchpad import MAX_NOTES, _save_note
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="bench", use_sandbox=False)
    ctx.notes["fail_to_pass_tests"] = '["tests/a.py::t1"]'
    ctx.reserved_note_keys.add("fail_to_pass_tests")
    # Fill past capacity with model notes.
    for i in range(MAX_NOTES + 3):
        r = await _save_note({"key": f"n{i}", "content": f"v{i}"}, ctx)
        assert r.success
    # The reserved harness key survives eviction.
    assert ctx.notes.get("fail_to_pass_tests") == '["tests/a.py::t1"]'


async def test_save_note_cannot_overwrite_reserved(tmp_path):
    from squishy.tools.base import ToolContext
    from squishy.tools.scratchpad import _save_note
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="bench", use_sandbox=False)
    ctx.notes["install_status"] = '{"ok": false}'
    ctx.reserved_note_keys.add("install_status")
    r = await _save_note({"key": "install_status", "content": "garbage"}, ctx)
    assert not r.success
    assert "reserved" in r.error
    assert ctx.notes["install_status"] == '{"ok": false}'


async def test_save_note_caps_key_length(tmp_path):
    from squishy.tools.base import ToolContext
    from squishy.tools.scratchpad import MAX_NOTE_KEY_CHARS, _save_note
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)
    r = await _save_note({"key": "k" * 5000, "content": "v"}, ctx)
    assert r.success
    assert all(len(k) <= MAX_NOTE_KEY_CHARS for k in ctx.notes)


# --- read_file cache-hit loop (found in live testing) ---------------------

async def test_read_file_refuses_after_repeated_identical_reads(tmp_path):
    """Serving unlimited successful cache hits let weak models spin on the
    identical read until the generic loop detector killed the run."""
    from squishy.tools.base import ToolContext
    from squishy.tools.fs import read_file
    (tmp_path / "m.py").write_text("x = 1\n")
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)

    r1 = await read_file.run({"path": "m.py"}, ctx)
    assert r1.success and not r1.data.get("cache_hit")

    r2 = await read_file.run({"path": "m.py"}, ctx)
    assert r2.success and r2.data.get("cache_hit") is True  # first repeat tolerated

    r3 = await read_file.run({"path": "m.py"}, ctx)
    assert not r3.success
    assert "STOP reading" in r3.error


async def test_read_cache_counter_resets_after_edit(tmp_path):
    """A re-read after a real change must not be refused."""
    from squishy.tools.base import ToolContext
    from squishy.tools.fs import edit_file, read_file
    (tmp_path / "m.py").write_text("x = 1\n")
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)

    await read_file.run({"path": "m.py"}, ctx)
    await read_file.run({"path": "m.py"}, ctx)
    r = await read_file.run({"path": "m.py"}, ctx)
    assert not r.success  # looping

    await edit_file.run({"path": "m.py", "old_str": "x = 1", "new_str": "x = 2"}, ctx)
    r = await read_file.run({"path": "m.py"}, ctx)
    assert r.success, "re-read after an edit must be allowed"
    assert "x = 2" in r.data["content"]
