"""A shell-only profile still needs a forcing function — just not a block.

Live evidence, same instance both ways:
  * blocking gate (minimal profile): refused run_command 25x, model re-called
    it every time, no patch;
  * no forcing function at all (shell profile): explored 80 turns, no patch.

So: nudge, never block, and detect edits from the commands themselves since
`files_edited` only records edit_file/write_file.
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


async def _run(tmp_path, commands, max_turns=20, budget=2):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "shell"
    cfg.max_turns = max_turns
    cfg.max_turns_without_edit = budget
    cfg.use_sandbox = False
    fake = FakeClient(script=[
        CompletionResult(tool_calls=[ToolCall(
            id=f"c{i}", name="run_command", args={"command": c})])
        for i, c in enumerate(commands)
    ])
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")
    return agent, result


async def test_pure_exploration_gets_nudged(tmp_path):
    agent, result = await _run(tmp_path, ["ls -la"] * 12)
    nudges = [m for m in result.messages
              if m.get("role") == "user" and "not changed any file" in str(m.get("content", ""))]
    assert nudges, "a shell run that never writes should be prodded"
    assert "git diff" in nudges[0]["content"]




async def test_a_run_that_writes_is_left_alone(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, result = await _run(
        tmp_path, ["ls -la", "cat > a.py <<'EOF'\nx = 2\nEOF"] + ["git diff"] * 10)
    assert agent._active_st.shell_writes >= 1
    nudges = [m for m in result.messages
              if m.get("role") == "user" and "not changed any file" in str(m.get("content", ""))]
    assert not nudges, "the model already edited; nagging it is noise"
