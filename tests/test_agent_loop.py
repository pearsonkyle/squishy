"""Agent loop tests using a scripted fake Client."""

from __future__ import annotations

import os

from conftest import FakeClient

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)
 
 
async def test_agent_writes_then_finishes(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 5
 
    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[_tc("write_file", {"path": "hi.py", "content": "print('hi')\n"})]
            ),
            CompletionResult(text="Wrote hi.py.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("create hi.py that prints 'hi'")
 
    assert result.success
    assert result.final_text == "Wrote hi.py."
    assert result.turns_used == 2
    assert "hi.py" in result.files_created
    assert os.path.isfile(tmp_path / "hi.py")
    # Two LLM calls: one produced the tool_call, one produced the final text
    assert len(fake.calls_seen) == 2
 
 
 
 












 
 












async def test_agent_runs_headless_without_display(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 5

    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[_tc("write_file", {"path": "a.py", "content": "x"})]
            ),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    result = await agent.run("write a.py")

    assert result.success
    assert (tmp_path / "a.py").read_text() == "x"


async def test_agent_allows_many_consecutive_reads(tmp_path):
    """Verify that many consecutive reads are allowed (no artificial limit)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 20

    # Create multiple files so reads succeed
    for i in range(15):
        (tmp_path / f"file{i}.txt").write_text(f"content {i}")

    script = [
        CompletionResult(tool_calls=[_tc("read_file", {"path": f"file{i}.txt"}, call_id=f"c{i}")])
        for i in range(15)
    ] + [CompletionResult(text="done.", tool_calls=[])]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("read many files")

    # All 15 reads should succeed followed by final text - no artificial refusal limit
    assert result.success
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    refusal_msgs = [m for m in tool_msgs if "refused" in (m.get("content") or "").lower()]
    assert not refusal_msgs, f"Should allow many reads without refusal: {refusal_msgs}"


# -- Phase tracking and budget tests (bench/yolo) ----------------------------

async def test_agent_phase_transitions(tmp_path):
    """Phase should transition: explore -> fix (on edit) -> verify (on run_command)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10

    (tmp_path / "foo.py").write_text("old line\n")

    fake = FakeClient(
        script=[
            # Turn 1: read (explore phase)
            CompletionResult(tool_calls=[_tc("read_file", {"path": "foo.py"})]),
            # Turn 2: edit (transitions to fix)
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "old line", "new_str": "new line"})
            ]),
            # Turn 3: run test (transitions to verify)
            CompletionResult(tool_calls=[
                _tc("run_command", {"command": "echo ok"}, call_id="c3")
            ]),
            # Turn 4: finish
            CompletionResult(text="Fixed.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert result.success
    assert result.final_text == "Fixed."






# -- Goal drift and edit failure tracking tests ---------------------------







async def test_consecutive_identical_loop_detection(tmp_path):
    """Agent force-finishes after 8 consecutive identical tool calls (bench)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 20

    (tmp_path / "foo.py").write_text("x = 1\n")

    # 15 identical read_file calls — should trigger loop detection at call 8.
    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id=f"c{i}")]
        )
        for i in range(15)
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert not result.success
    assert "loop detected" in result.error
    assert result.turns_used <= 9  # should stop well before 20




async def test_turn_log_populated_in_bench(tmp_path):
    """TaskResult.turn_log is populated with per-turn events in bench mode."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 5

    (tmp_path / "foo.py").write_text("x = 1\n")

    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"})]
        ),
        CompletionResult(text="done."),
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert result.turn_log
    entry = result.turn_log[0]
    assert entry["turn"] == 1
    assert entry["tools"]
    assert "tools" in entry
    assert entry["tools"][0]["name"] == "read_file"




# ---------------------------------------------------------------------------
# Plan rejection with feedback
# ---------------------------------------------------------------------------














async def test_alias_tool_call_dispatches_canonically(tmp_path):
    """A model that emits `create`/`file_path` (another harness's vocabulary)
    should have it normalized and dispatched as write_file."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 4
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("create", {"file_path": "hi.py", "content": "print(1)\n"})]),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    result = await agent.run("make hi.py")
    assert result.success
    assert (tmp_path / "hi.py").read_text() == "print(1)\n"
    assert "hi.py" in result.files_created
    # The recorded assistant call was rewritten to the canonical tool name.
    tool_names = [
        tc["function"]["name"]
        for m in result.messages if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    ]
    assert "write_file" in tool_names
    assert "create" not in tool_names


async def test_user_message_is_persisted_to_session(tmp_path):
    """#1: the user turn must reach the session log even when the run ends via
    a nudge/continue path (previously silently dropped)."""
    from squishy.session import create_session, load_messages

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.session_dir = str(tmp_path / "sessions")
    cfg.max_turns = 4
    sess = create_session(model="fake", working_dir=str(tmp_path), mode="yolo",
                          tools=[], root=cfg.session_dir)
    fake = FakeClient(script=[CompletionResult(text="done", tool_calls=[])])
    agent = Agent(cfg, fake, display=None, session_id=sess.id)  # type: ignore[arg-type]
    await agent.run("REMEMBER-THIS-PROMPT")

    persisted = load_messages(sess.id, root=cfg.session_dir)
    user_msgs = [m for m in persisted if m.get("role") == "user"]
    assert any("REMEMBER-THIS-PROMPT" in (m.get("content") or "") for m in user_msgs)


async def test_result_messages_exclude_live_context_pair(tmp_path):
    """#11: the synthetic live-context pair must not leak into TaskResult."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 4
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("save_note", {"key": "k", "content": "v"})]),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    result = await agent.run("note something")
    assert not any(m.get("_squishy_live_ctx") for m in result.messages)
    assert not any(m.get("name") == "_squishy_context" for m in result.messages)
    assert not any(m.get("name") == "_squishy_context" for m in result.full_log)






async def test_must_edit_gate_not_applied_after_an_edit(tmp_path):
    """A successful edit keeps run_command available for verification."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 8
    (tmp_path / "a.py").write_text("x = 1\n")

    seen: list[list[str]] = []

    class _Recorder(FakeClient):
        async def complete(self, messages, tools, **kw):
            seen.append([t["function"]["name"] for t in tools])
            return await super().complete(messages, tools, **kw)

    script = [
        CompletionResult(tool_calls=[_tc("edit_file", {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})]),
        CompletionResult(tool_calls=[_tc("run_command", {"command": "ls"}, "c2")]),
        CompletionResult(tool_calls=[_tc("run_command", {"command": "pwd"}, "c3")]),
        CompletionResult(tool_calls=[_tc("run_command", {"command": "echo hi"}, "c4")]),
        CompletionResult(text="done", tool_calls=[]),
    ]
    agent = Agent(cfg, _Recorder(script=script), display=None)  # type: ignore[arg-type]
    await agent.run("fix it")
    assert "run_command" in seen[-1], "run_command must remain after an edit landed"




async def test_live_context_never_shows_the_model_a_phantom_tool_call(tmp_path):
    """The notes block must not reach the model as a call to a tool we don't have.

    It used to be a fabricated (assistant tool_calls, tool result) pair naming
    `_squishy_context`, which appears in no schema we send. That pair was
    filtered out of TaskResult and the SFT export — but not out of the
    transcript the model actually reads, and models imitate their own history.
    On msrest-for-python-43 the model called `_squishy_context` twice and got
    `unknown tool` back both times: the harness demonstrating a tool call and
    then refusing it.

    What the model is sent is `FakeClient`'s recorded input, so assert there.
    """
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 4
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("save_note", {"key": "k", "content": "v"})]),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    await agent.run("note something")

    sent = [m for call in fake.calls_seen for m in call]
    assert sent, "expected the fake client to have recorded outgoing messages"
    # The note itself still has to reach the model — this is not a deletion.
    assert any("Saved Notes" in (m.get("content") or "") for m in sent)
    # But never as a tool call, and never as a tool result.
    for m in sent:
        assert m.get("name") != "_squishy_context"
        for tc in m.get("tool_calls") or []:
            assert tc.get("function", {}).get("name") != "_squishy_context"
