"""Tests for agent_phases: cache_problem_text, maybe_reanchor_problem."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from squishy.agent_phases import (
    cache_problem_text,
    maybe_post_edit_pytest_nudge,
    maybe_reanchor_problem,
)
from squishy.agent_state import LoopState


def _make_agent(messages: list[dict[str, Any]], permission_mode: str = "bench") -> MagicMock:
    """Build a minimal mock Agent with .messages and .config."""
    agent = MagicMock()
    agent.messages = messages
    agent.config.permission_mode = permission_mode
    agent.config.max_explore_turns = 3
    agent.config.max_fix_verify_cycles = 6
    agent.config.max_system_nudges = 6
    agent.display = None
    return agent


def _make_state(**kwargs) -> LoopState:
    return LoopState(start=time.monotonic(), **kwargs)


# -- cache_problem_text --------------------------------------------------------

class TestCacheProblemText:
    def test_extracts_problem_text(self):
        messages = [
            {"role": "user", "content": "## Problem\nSomething is broken in foo.py\n\n## Steps"},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.problem_text == "Something is broken in foo.py\n\n## Steps"

    def test_truncates_long_problem_text(self):
        long_text = "x" * 2000
        messages = [
            {"role": "user", "content": f"## Problem\n{long_text}"},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert len(st.problem_text) == 1503  # 1500 + "..."
        assert st.problem_text.endswith("...")

    def test_extracts_fail_to_pass_tests(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "## Problem\nBug.\n\n"
                    "## Failing Tests\n"
                    "- `tests/test_foo.py::TestBar::test_baz`\n"
                    "- `tests/test_foo.py::test_qux`\n"
                    "\n## Other\nStuff"
                ),
            },
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.fail_to_pass_tests == [
            "tests/test_foo.py::TestBar::test_baz",
            "tests/test_foo.py::test_qux",
        ]

    def test_no_failing_tests_section(self):
        messages = [
            {"role": "user", "content": "## Problem\nBug."},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.fail_to_pass_tests == []

    def test_v6b_loads_full_f2p_from_notes_overrides_truncated_prompt(self):
        """v6b: when tool_ctx.notes['fail_to_pass_tests'] supplies the
        full F2P list, prefer it over the prompt regex (which caps at
        5 entries via build_prompt's '_(and N more)_' truncation)."""
        import json as _json
        full_f2p = [
            "tests/test_foo.py::test_one",
            "tests/test_foo.py::test_two",
            "tests/test_foo.py::test_three",
            "tests/test_foo.py::test_four",
            "tests/test_foo.py::test_five",
            "tests/test_foo.py::test_six",
            "tests/test_foo.py::test_seven",
        ]
        # Prompt only has the first 5 (truncated, like build_prompt does).
        truncated_section = "\n".join(f"- `{t}`" for t in full_f2p[:5])
        messages = [{
            "role": "user",
            "content": (
                "## Problem\nBug.\n\n"
                f"## Failing Tests\n{truncated_section}\n"
                "- _(and 2 more)_\n"
            ),
        }]
        agent = _make_agent(messages)
        agent.tool_ctx.notes = {"fail_to_pass_tests": _json.dumps(full_f2p)}
        st = _make_state()
        cache_problem_text(agent, st)
        # Should load the full 7-element list from notes, not just the
        # prompt's visible 5.
        assert st.fail_to_pass_tests == full_f2p

    def test_v6b_falls_back_to_prompt_regex_when_notes_absent(self):
        """v6b: interactive REPL or any caller that doesn't set
        tool_ctx.notes must still get F2P from the prompt regex."""
        messages = [{
            "role": "user",
            "content": (
                "## Problem\nBug.\n\n"
                "## Failing Tests\n"
                "- `tests/test_foo.py::test_one`\n"
                "- `tests/test_foo.py::test_two`\n"
            ),
        }]
        agent = _make_agent(messages)
        # Notes attribute either missing or an empty dict — both should
        # let the regex fallback fire.
        agent.tool_ctx.notes = {}
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.fail_to_pass_tests == [
            "tests/test_foo.py::test_one",
            "tests/test_foo.py::test_two",
        ]

    def test_v6b_handles_malformed_notes_gracefully(self):
        """v6b: malformed JSON in notes must not crash — fall back to
        prompt regex (or empty list if no prompt section)."""
        messages = [{
            "role": "user",
            "content": (
                "## Problem\nBug.\n\n"
                "## Failing Tests\n"
                "- `tests/test_foo.py::test_one`\n"
            ),
        }]
        agent = _make_agent(messages)
        agent.tool_ctx.notes = {"fail_to_pass_tests": "not-json-{["}
        st = _make_state()
        cache_problem_text(agent, st)
        # Falls back to the regex, which finds the one entry.
        assert st.fail_to_pass_tests == ["tests/test_foo.py::test_one"]

    def test_skips_system_messages(self):
        messages = [
            {"role": "user", "content": "[system] not a problem statement"},
            {"role": "user", "content": "## Problem\nReal bug."},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.problem_text == "Real bug."

    def test_no_problem_section(self):
        messages = [
            {"role": "user", "content": "Just fix it."},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.problem_text is None


# -- maybe_reanchor_problem ----------------------------------------------------

class TestMaybeReanchorProblem:
    def test_injects_problem_text(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.problem_text = "The bug is X."
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 1
        assert "The bug is X." in messages[0]["content"]
        assert st.last_reanchor_turn == 10

    def test_includes_fail_to_pass_tests(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.problem_text = "Bug."
        st.fail_to_pass_tests = ["tests/test_a.py::test_1", "tests/test_b.py::test_2"]
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        content = messages[0]["content"]
        assert "`tests/test_a.py::test_1`" in content
        assert "`tests/test_b.py::test_2`" in content

    def test_skips_when_too_recent(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.problem_text = "Bug."
        st.last_reanchor_turn = 8

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 0

    def test_skips_non_bench_mode(self):
        messages: list[dict] = []
        agent = _make_agent(messages, permission_mode="edits")
        st = _make_state()
        st.problem_text = "Bug."
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 0

    def test_skips_when_no_problem_text(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 0


# -- maybe_post_edit_pytest_nudge (v6c) ---------------------------------------

class TestMaybePostEditPytestNudge:
    def test_post_edit_nudge_fires_after_first_edit(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["a/test.py::test_x", "b/test.py::test_y"]

        maybe_post_edit_pytest_nudge(agent, st, turn=6)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "pytest a/test.py::test_x b/test.py::test_y" in content
        assert "`a/test.py::test_x`" in content
        assert "`b/test.py::test_y`" in content
        assert st.post_edit_pytest_nudge_sent is True

    def test_post_edit_nudge_fires_only_once(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["a/test.py::test_x"]

        maybe_post_edit_pytest_nudge(agent, st, turn=6)
        maybe_post_edit_pytest_nudge(agent, st, turn=10)
        assert len(messages) == 1, "second call must be a no-op"

    def test_post_edit_nudge_skips_without_edit(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 0  # no edit yet
        st.fail_to_pass_tests = ["a/test.py::test_x"]

        maybe_post_edit_pytest_nudge(agent, st, turn=10)
        assert len(messages) == 0
        assert st.post_edit_pytest_nudge_sent is False

    def test_post_edit_nudge_skips_non_bench_mode(self):
        messages: list[dict] = []
        agent = _make_agent(messages, permission_mode="edits")
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["a/test.py::test_x"]

        maybe_post_edit_pytest_nudge(agent, st, turn=10)
        assert len(messages) == 0
        assert st.post_edit_pytest_nudge_sent is False

    def test_post_edit_nudge_skips_when_no_f2p(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = []

        maybe_post_edit_pytest_nudge(agent, st, turn=10)
        assert len(messages) == 0
        assert st.post_edit_pytest_nudge_sent is False

    def test_v6d_post_edit_nudge_does_not_burn_one_shot_when_suppressed(self):
        """v6d: a min_gap or hard-cap rejection from inject_nudge must
        NOT set the post_edit_pytest_nudge_sent flag, so a later turn
        can re-attempt. Reproduces the rdt-670 v6c failure where the
        nudge was silently suppressed by an unrelated prior nudge.
        """
        messages: list[dict] = []
        agent = _make_agent(messages)
        # max_system_nudges = 3 (from _make_agent default), so hard
        # cap is 6. Saturate it to force inject_nudge → False.
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["a/test.py::test_x"]
        st.total_nudges = agent.config.max_system_nudges * 2  # at hard cap

        maybe_post_edit_pytest_nudge(agent, st, turn=6)
        assert len(messages) == 0, "nudge was suppressed (over hard cap)"
        assert st.post_edit_pytest_nudge_sent is False, \
            "flag must NOT be set when inject_nudge returns False"

        # Open the budget back up (simulating a window where nudges
        # are once again allowed) — the next call should fire and
        # set the flag.
        st.total_nudges = 0
        st.last_nudge_turn = 0
        maybe_post_edit_pytest_nudge(agent, st, turn=10)
        assert len(messages) == 1
        assert st.post_edit_pytest_nudge_sent is True

    def test_post_edit_nudge_truncates_long_f2p_list(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = [
            "tests/t.py::test_one",
            "tests/t.py::test_two",
            "tests/t.py::test_three",
            "tests/t.py::test_four",
            "tests/t.py::test_five",
            "tests/t.py::test_six",
            "tests/t.py::test_seven",
            "tests/t.py::test_eight",
        ]

        maybe_post_edit_pytest_nudge(agent, st, turn=10)
        assert len(messages) == 1
        content = messages[0]["content"]
        # Preview shows 5 + "and 3 more"
        assert "(and 3 more)" in content
        assert "`tests/t.py::test_one`" in content
        assert "`tests/t.py::test_five`" in content
        # Preview should NOT show test_six in the comma-joined backtick list,
        # but the executable command MUST include all 8 IDs.
        assert "pytest tests/t.py::test_one tests/t.py::test_two tests/t.py::test_three tests/t.py::test_four tests/t.py::test_five tests/t.py::test_six tests/t.py::test_seven tests/t.py::test_eight" in content


# -- v6e: test_cmd plumbing ---------------------------------------------------

class TestPostEditNudgeTestCmd:
    """v6e: maybe_post_edit_pytest_nudge should honor st.test_cmd when set
    (forwarded from V2's install_config.test_cmd via tool_ctx.notes)."""

    def test_uses_pytest_test_cmd_with_f2p_ids(self):
        """When test_cmd contains 'pytest', append F2P IDs as before but
        with the harness-supplied prefix (e.g. 'python -m pytest')."""
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["a/test.py::test_x", "b/test.py::test_y"]
        st.test_cmd = "python -m pytest"

        maybe_post_edit_pytest_nudge(agent, st, turn=6)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "python -m pytest a/test.py::test_x b/test.py::test_y" in content
        # Should NOT fall back to bare "pytest" prefix.
        assert "```\npytest a/" not in content

    def test_uses_bare_test_cmd_for_non_pytest_runner(self):
        """For foreign runners (npm/phpunit/cargo), suggest the bare cmd
        without appending pytest-style test IDs."""
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["foo.test.ts::should-bar"]
        st.test_cmd = "npm test"

        maybe_post_edit_pytest_nudge(agent, st, turn=6)
        assert len(messages) == 1
        content = messages[0]["content"]
        # Bare command in code-fence, no F2P IDs appended.
        assert "```\nnpm test\n```" in content
        assert "foo.test.ts::should-bar" not in content.split("```")[1]
        # Targets line still surfaces the IDs informationally.
        assert "`foo.test.ts::should-bar`" in content

    def test_falls_back_to_pytest_when_test_cmd_empty(self):
        """When the harness didn't supply test_cmd (V1 / interactive REPL),
        keep the v6c default of bare 'pytest <ids>'."""
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_edit_turn = 5
        st.fail_to_pass_tests = ["a/test.py::test_x"]
        # st.test_cmd left as default ""

        maybe_post_edit_pytest_nudge(agent, st, turn=6)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "pytest a/test.py::test_x" in content


class TestCacheProblemTextTestCmd:
    """v6e: cache_problem_text should populate st.test_cmd from tool_ctx.notes."""

    def test_reads_test_cmd_from_notes(self):
        agent = _make_agent([
            {"role": "user", "content": "## Problem\nbug\n"},
        ])
        agent.tool_ctx = MagicMock()
        agent.tool_ctx.notes = {"test_cmd": "./vendor/bin/phpunit"}
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.test_cmd == "./vendor/bin/phpunit"

    def test_test_cmd_defaults_empty_when_notes_absent(self):
        agent = _make_agent([
            {"role": "user", "content": "## Problem\nbug\n"},
        ])
        agent.tool_ctx = MagicMock()
        agent.tool_ctx.notes = None
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.test_cmd == ""

    def test_test_cmd_strips_whitespace(self):
        agent = _make_agent([
            {"role": "user", "content": "## Problem\nbug\n"},
        ])
        agent.tool_ctx = MagicMock()
        agent.tool_ctx.notes = {"test_cmd": "  cargo test  \n"}
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.test_cmd == "cargo test"
