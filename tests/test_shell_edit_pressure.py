"""A shell-only profile still needs a forcing function — just not a block.

Live evidence, same instance both ways:
  * blocking gate (minimal profile): refused run_command 25x, model re-called
    it every time, no patch;
  * no forcing function at all (shell profile): explored 80 turns, no patch.

So: push, never block, and detect edits from the commands themselves since
`files_edited` only records edit_file/write_file.

The push now rides on the tool result (`tools/pressure.py`) instead of an
injected `[system]` user turn, which is this loop's standing rule: feedback
paired with the call that earned it cannot desynchronize the transcript.
"""
from __future__ import annotations

import pytest
from conftest import FakeClient

from squishy.agent import Agent
from squishy.agent_state import looks_like_file_write
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display


@pytest.mark.parametrize("cmd", [
    "cat > b.py <<'EOF'\nx = 1\nEOF",
    "python3 - <<'EOF'\nopen('a.py','w').write('x')\nEOF",
    "sed -i 's/a/b/' f.py",
    "git apply /tmp/fix.patch",
    "echo hi >> notes.txt",
    "patch -p1 < d.diff",
])
def test_write_commands_are_recognized(cmd):
    assert looks_like_file_write(cmd)


@pytest.mark.parametrize("cmd", [
    "pytest -q 2>&1", "ls -la", "grep -rn foo src/", "cat a.py",
    "git diff", "go test ./...",
])
def test_read_only_commands_are_not_mistaken_for_writes(cmd):
    assert not looks_like_file_write(cmd)


def _pressure(result):
    """Every pressure notice the model was actually shown, in order."""
    return [
        str(m.get("content", ""))
        for m in result.messages
        if m.get("role") == "tool"
        and ("[budget]" in str(m.get("content", ""))
             or "[probes]" in str(m.get("content", "")))
    ]


async def _run(tmp_path, commands, max_turns=20):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "shell"
    cfg.max_turns = max_turns
    cfg.use_sandbox = False
    fake = FakeClient(script=[
        CompletionResult(tool_calls=[ToolCall(
            id=f"c{i}", name="run_command", args={"command": c})])
        for i, c in enumerate(commands)
    ])
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")
    return agent, result


async def test_pure_exploration_gets_pushed(tmp_path):
    agent, result = await _run(tmp_path, [f"ls -la {i}" for i in range(12)])
    notices = _pressure(result)
    assert notices, "a shell run that never writes should be prodded"
    assert any("[probes]" in n for n in notices), "a run of commands with no edit"
    assert any("[budget]" in n for n in notices), "and the clock running down"


async def test_the_push_arrives_in_the_tool_result_not_a_user_turn(tmp_path):
    """The rule the whole loop is built around, asserted directly."""
    _agent, result = await _run(tmp_path, [f"ls {i}" for i in range(12)])
    injected = [
        m for m in result.messages
        if m.get("role") == "user" and "[budget]" in str(m.get("content", ""))
    ]
    assert not injected, "pressure must never be an out-of-band user message"


async def test_a_run_that_writes_is_left_alone(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, result = await _run(
        tmp_path, ["ls -la", "cat > a.py <<'EOF'\nx = 2\nEOF"]
        + [f"git diff {i}" for i in range(10)])
    assert agent._active_st.shell_writes >= 1
    assert not _pressure(result), "the model already edited; nagging it is noise"


@pytest.mark.parametrize("cmd", [
    # Every shape the model actually produced across three instances.
    "cat > /tmp/repro.py <<'EOF'\nprint(1)\nEOF",
    "python - <<'EOF'\nopen('/tmp/probe.py','w').write('x')\nEOF",
    "cd /repo && cat > /tmp/t.yaml <<'EOF'\nx: 1\nEOF",
    # qiskit-terra-5662 wrote scratch files this way eleven times: the target
    # is relative, so only the `cd` says it is scratch.
    "cd /tmp && cat > test_bug.py <<'EOF'\nprint(1)\nEOF",
    "cd /tmp && cat > debug2.py <<'EOF'\nprint(1)\nEOF",
    # Not a file write at all, and the most ordinary command there is.
    "pytest -q > /dev/null",
    "python -c 'import x' >/dev/null 2>&1",
])
def test_a_scratch_write_is_not_a_source_edit(cmd):
    """The harness asks for /tmp repro scripts. Rewarding that by switching
    off edit pressure is how twelve container arms ran to a 100-turn cap
    without one of them ever calling edit_file.

    cfn-lint-3965 turn 10 was `cat > /tmp/repro.py <<EOF`. From that call on,
    `shell_writes` was non-zero, `source_edited` was True, and every [budget]
    and [probes] notice was suppressed for the remaining ninety turns.
    """
    assert not looks_like_file_write(cmd)


async def test_pressure_survives_a_scratch_repro_script(tmp_path):
    """End to end: the instructed action must not disarm the harness."""
    commands = ["cat > /tmp/sq_repro.py <<'EOF'\nprint(1)\nEOF"] + [
        f"grep -rn thing{i} src/" for i in range(12)
    ]
    _agent, result = await _run(tmp_path, commands)
    assert _pressure(result), "a /tmp script must not switch the brakes off"
