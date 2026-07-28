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
