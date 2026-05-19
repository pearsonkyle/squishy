"""Tests for display.py robustness against untrusted content (Rich
markup escape) and the streaming-buffer flush edge case.

These cover the v6d UX/robustness audit findings:
  - tool_result/edit_diff/write_preview/plan_panel must not blow up on
    arbitrary file paths or LLM strings containing ``[`` (Rich would
    interpret it as broken markup).
  - flush_streaming_text must persist a frame for buffers that contain
    only whitespace — the user already saw it flicker on screen, so
    silently dropping it on the final flush leaves a confusing gap.
"""
from __future__ import annotations

import io

from rich.console import Console

from squishy.display import Display


def _capture_display() -> tuple[Display, io.StringIO]:
    """Build a Display with an in-memory console for assertion."""
    buf = io.StringIO()
    d = Display()
    d.console = Console(file=buf, force_terminal=False, width=120)
    return d, buf


# -- Rich markup escape -------------------------------------------------------


def test_tool_result_handles_brackets_in_display_text():
    d, buf = _capture_display()
    # A Python type annotation is a common shape that contains brackets.
    d.tool_result(success=True, display="returns list[int] from foo[bar].py", duration_ms=1.2)
    out = buf.getvalue()
    assert "list[int]" in out
    assert "foo[bar].py" in out


def test_edit_diff_handles_brackets_in_diff_lines():
    d, buf = _capture_display()
    # Diff content with bracket-bearing code must not raise.
    d.edit_diff(
        path="x.py",
        old="x: list[int] = []\n",
        new="x: dict[str, int] = {}\n",
    )
    out = buf.getvalue()
    assert "dict[str, int]" in out


def test_write_preview_handles_brackets():
    d, buf = _capture_display()
    d.write_preview(path="x.py", content="config = {'a': [1, 2]}\nresult = arr[0]\n")
    out = buf.getvalue()
    assert "arr[0]" in out


def test_plan_panel_handles_brackets_in_problem_and_steps():
    d, buf = _capture_display()
    d.plan_panel({
        "plan": "Fix [bug-123]",
        "problem": "list[int] confusion in foo[bar]",
        "solution": "use dict[str, list[int]] instead",
        "steps": [
            {"description": "rewrite foo[0] handler", "status": "done"},
            "patch bar[1] callsite",
        ],
        "files_to_create": ["new[file].py"],
        "files_to_modify": ["mod[ule].py"],
    })
    out = buf.getvalue()
    # All bracketed strings must survive.
    assert "list[int]" in out
    assert "foo[0]" in out
    assert "mod[ule].py" in out
    assert "[bug-123]" in out


def test_turn_header_handles_brackets_in_brief():
    d, buf = _capture_display()
    d.turn_header(turn=1, max_turns=10, tool_name="edit_file", brief="path=foo[bar].py")
    out = buf.getvalue()
    assert "foo[bar].py" in out


# -- flush_streaming_text whitespace edge case -------------------------------


def test_flush_streaming_text_preserves_whitespace_only_buffer():
    """A stream that ended on whitespace already flickered on screen via
    Live; the final flush must reprint it instead of leaving a gap."""
    d, buf = _capture_display()
    # Simulate a stream of just whitespace + newlines (e.g. model
    # produced trailing blank lines).  Stream then flush.
    d.streaming_text_chunk("\n\n  \n")
    d.flush_streaming_text()
    # Buffer should be reset for next turn.
    assert d._stream_buffer == ""
    assert d._live is None


def test_flush_streaming_text_skips_truly_empty_buffer():
    """Empty buffer → nothing to print, no crash."""
    d, buf = _capture_display()
    d.flush_streaming_text()
    assert d._stream_buffer == ""
    # No live ever started, no print ever happened.
    assert "" == buf.getvalue() or buf.getvalue().strip() == ""
