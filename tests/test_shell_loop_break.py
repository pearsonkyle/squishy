"""The shell needs its own loop breaker.

Live shell-only run: 27 commands, most of them the same grep with a different
`| head -N`, the right file identified at turn 13, no edit ever made, and zero
tool failures. Re-running a grep always succeeds, so nothing pushed back — the
loop-breaking the file tools do (cache hits, refusal ladder) has no analogue
here.
"""
from __future__ import annotations

import pytest

from squishy.tools import dispatch
from squishy.tools.shell import _command_key


@pytest.mark.parametrize("a,b", [
    ("cd /vyper && grep -n x f.py | head -30", "grep -n x f.py | head -20"),
    ("cd /repo && cat a.py | head -100", "cat a.py"),
    ("cd /a && cd /b && ls", "ls"),
])
def test_cosmetic_variation_hashes_alike(a, b):
    assert _command_key(a) == _command_key(b)


@pytest.mark.parametrize("a,b", [
    ("grep -n x f.py", "grep -n y f.py"),
    ("cat a.py", "cat b.py"),
    ("pytest tests/a.py", "pytest tests/b.py"),
])
def test_genuinely_different_commands_stay_distinct(a, b):
    assert _command_key(a) != _command_key(b)


async def test_repeating_a_command_escalates_to_a_refusal(ctx, tmp_path):
    (tmp_path / "f.py").write_text("a = 1\nb = 2\n")
    ctx.working_dir = str(tmp_path)
    ctx.use_sandbox = False

    results = []
    for i in range(5):
        # Vary only the cosmetic tail, exactly as the model did live.
        results.append(await dispatch(
            "run_command", {"command": f"grep -n a f.py | head -{30 - i}"}, ctx))

    assert results[0].success and "note" not in results[0].data
    assert any("note" in r.data for r in results[1:3]), "should warn before refusing"
    assert not results[-1].success
    assert "same output" in results[-1].error
    # The refusal must point at the next action, not just say no.
    assert "git diff" in results[-1].error


async def test_a_command_whose_output_changes_is_never_refused(ctx, tmp_path):
    """Re-running tests after an edit is legitimate and must keep working."""
    ctx.working_dir = str(tmp_path)
    ctx.use_sandbox = False
    for i in range(6):
        (tmp_path / "f.py").write_text(f"value = {i}\n")
        res = await dispatch("run_command", {"command": "cat f.py"}, ctx)
        assert res.success, res.error


async def test_distinct_commands_are_unaffected(ctx, tmp_path):
    ctx.working_dir = str(tmp_path)
    ctx.use_sandbox = False
    for i in range(8):
        res = await dispatch("run_command", {"command": f"echo {i}"}, ctx)
        assert res.success, res.error
