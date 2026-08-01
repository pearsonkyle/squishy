"""The SDK driver's continuation loop.

``Runner.run`` returns the moment the model emits a message with no tool call
— including an empty one, which ornith does regularly. On cfn-lint-3965 both
SDK arms ended that way: a scratch file in /tmp, no edit, no answer, and an
``exit_status`` of "completed" over a run that had fixed nothing. A container
sweep costs hours to discover that; a stub costs milliseconds.
"""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "rebench_container" / "driver_graph.py"
)


@pytest.fixture(scope="module")
def dg():
    if not _SCRIPT.exists():
        pytest.skip("SDK driver not present")
    spec = importlib.util.spec_from_file_location("driver_graph", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class _Usage:
    requests = 1
    input_tokens = 10
    output_tokens = 2


class _Result:
    """Minimal stand-in for the SDK's RunResult."""

    def __init__(self, final_output: str, history: list[dict[str, Any]]) -> None:
        self.final_output = final_output
        self.context_wrapper = type("Ctx", (), {"usage": _Usage()})()
        self._history = history

    def to_input_list(self) -> list[dict[str, Any]]:
        return list(self._history)


def _task(retries: int) -> dict[str, Any]:
    return {
        "prompt": "fix the bug",
        "empty_patch_nudge": "no patch yet — edit the source",
        "empty_patch_retries": retries,
        "max_turns": 50,
        "task_timeout": None,
    }


def _drive_with(dg, monkeypatch, outputs: list[str], dirty_after: int | None,
                retries: int = 3, turns_per_call: int = 0):
    """Run ``_drive`` against a scripted model.

    Returns ``(state, inputs seen, final text, call count)``; each call's
    ``max_turns`` is appended to ``state["budgets"]``.
    """
    seen: list[Any] = []
    calls = {"n": 0}
    budgets: list[int] = []

    async def fake_run(agent, items, hooks=None, max_turns=0, **kwargs):
        seen.append(items)
        budgets.append(max_turns)
        calls["n"] += 1
        state["turns"] += turns_per_call
        text = outputs[min(calls["n"] - 1, len(outputs) - 1)]
        return _Result(text, [{"role": "assistant", "content": text}])

    monkeypatch.setattr(dg.Runner, "run", fake_run)
    monkeypatch.setattr(
        dg, "_dirty",
        lambda repo: dirty_after is not None and calls["n"] >= dirty_after,
    )
    monkeypatch.setattr(dg, "_flush", lambda state, trace: None)

    state = {
        "exit_status": "running", "turns": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "graph_build_s": 0.0, "graph_stats": {},
        "arm": "graph", "nudges": 0, "started_at": 0.0,
        "max_turns": 50, "task_timeout": None,
    }
    text = asyncio.run(dg._drive(None, _task(retries), None, state, "/repo"))
    state["budgets"] = budgets
    return state, seen, text, calls["n"]


def test_an_empty_answer_over_a_clean_tree_does_not_end_the_run(
    dg, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, seen, _, calls = _drive_with(
        dg, monkeypatch, [""], dirty_after=None, turns_per_call=10
    )
    assert calls == 5  # keeps going until the 50-turn budget is gone
    assert state["nudges"] == 4  # the fifth segment has no budget to answer one
    assert state["exit_status"] == "max_turns"
    assert "empty message" in seen[1][-1]["content"]


def test_retries_zero_leaves_the_sdk_behavior_untouched(
    dg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement of raw SDK behavior has to stay available."""
    state, _, _, calls = _drive_with(
        dg, monkeypatch, [""], dirty_after=None, retries=0, turns_per_call=10
    )
    assert calls == 1
    assert state["nudges"] == 0


def test_a_nonempty_answer_over_a_clean_tree_gets_the_patch_nudge(
    dg, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, seen, _, _ = _drive_with(
        dg, monkeypatch, ["I have analyzed the issue."], dirty_after=None,
        turns_per_call=10,
    )
    assert seen[1][-1]["content"] == "no patch yet — edit the source"


def test_continuation_resumes_the_transcript_instead_of_restarting(
    dg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh run would throw away the exploration that got it that far."""
    _, seen, _, _ = _drive_with(
        dg, monkeypatch, [""], dirty_after=None, turns_per_call=10
    )
    assert seen[0] == "fix the bug"
    assert seen[1][0] == {"role": "assistant", "content": ""}


def test_a_changed_tree_ends_the_run(dg, monkeypatch: pytest.MonkeyPatch) -> None:
    state, _, _, calls = _drive_with(dg, monkeypatch, [""], dirty_after=1)
    assert calls == 1
    assert state["nudges"] == 0
    assert state["exit_status"] == "completed"


def test_the_turn_budget_is_shared_across_continuations(
    dg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each Runner.run gets what is *left*, not a fresh 50 — otherwise three
    nudges quietly turn a 50-turn cap into a 200-turn one."""
    state, _, _, calls = _drive_with(
        dg, monkeypatch, [""], dirty_after=None, turns_per_call=20
    )
    assert state["budgets"] == [50, 30, 10]
    assert calls == 3  # the fourth attempt has no budget left
    assert state["exit_status"] == "max_turns"


def test_dirty_ignores_the_harness_own_artifacts(
    dg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".graphagent.json").write_text("{}", encoding="utf-8")
    assert dg._dirty(str(tmp_path)) is False
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    assert dg._dirty(str(tmp_path)) is True


def test_the_budget_reported_is_the_binding_one(dg, monkeypatch) -> None:
    """max_turns and task_timeout are set independently and disagree on slow
    images. qiskit runs at ~30s a turn, so 50 turns under a 1200s wall is
    really 40 — and the agent was being told it had nine left as the process
    was killed."""
    now = {"t": 1000.0}
    monkeypatch.setattr(dg.time, "time", lambda: now["t"])
    state = {"turns": 0, "started_at": 1000.0, "max_turns": 50,
             "task_timeout": 1200}

    # Nothing observed yet: the nominal budget stands.
    assert dg._effective_budget(state) == (0, 50)

    # 20 turns in 600s -> 30s a turn -> the wall allows 40, not 50.
    state["turns"], now["t"] = 20, 1600.0
    assert dg._effective_budget(state) == (20, 40)

    # A fast image is bounded by max_turns, not the wall.
    state["turns"], now["t"] = 20, 1020.0
    assert dg._effective_budget(state) == (20, 50)

    # Overshooting the projection must not promise a turn past the real cap:
    # the last turn of a 50-turn run reported "50 of 51".
    state["turns"], now["t"] = 50, 3000.0
    assert dg._effective_budget(state) == (50, 50)

    # No timeout configured: nothing to project against.
    state["turns"], state["task_timeout"] = 20, None
    assert dg._effective_budget(state) == (20, 50)
