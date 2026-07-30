from __future__ import annotations
 
import json
 
from squishy.context import (
    build_system_prompt,
    compact_messages,
    detect_project,
    normalize_messages,
    snip_old_tool_results,
    trim_history,
)


def _asst_tc(call_id, name="run_command"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": "{}"}},
        ],
    }


def test_normalize_drops_reverse_orphan_tool_message():
    """A tool message whose id was never declared by a preceding assistant
    (e.g. a synthetic result injected without its assistant call) is dropped."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "done"},  # no tool_calls
        {"role": "tool", "tool_call_id": "auto-pytest-0", "name": "run_command", "content": "F"},
        {"role": "user", "content": "[system] nudge"},
    ]
    out = normalize_messages(msgs)
    assert not any(m.get("role") == "tool" for m in out)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]


def test_normalize_keeps_well_formed_pairs():
    """A properly paired assistant tool_calls + tool result survives."""
    msgs = [
        {"role": "user", "content": "go"},
        _asst_tc("c1"),
        {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "ok"},
    ]
    out = normalize_messages(msgs)
    assert out == msgs


def test_normalize_keeps_result_after_intervening_user():
    """A user message between the assistant call and its result does not
    orphan the result — its id was still declared earlier."""
    msgs = [
        _asst_tc("c1"),
        {"role": "user", "content": "[system] nudge"},
        {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "ok"},
    ]
    out = normalize_messages(msgs)
    assert any(m.get("role") == "tool" and m["tool_call_id"] == "c1" for m in out)
 
 
def test_detect_node_nextjs(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"next": "14", "react": "18"}}))
    info = detect_project(str(tmp_path))
    assert info.language == "javascript"
    assert info.framework == "nextjs"
 
 
def test_detect_python_fastapi(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["fastapi"]\n')
    info = detect_project(str(tmp_path))
    assert info.language == "python"
    assert info.framework == "fastapi"
 
 
def test_detect_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"x\"\n")
    info = detect_project(str(tmp_path))
    assert info.language == "rust"
 
 
def test_detect_empty(tmp_path):
    info = detect_project(str(tmp_path))
    assert info.language == "unknown"
 
 
def test_build_system_prompt_includes_project_info(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["flask"]\n')
    info = detect_project(str(tmp_path))
    prompt = build_system_prompt(str(tmp_path), info)
    assert "python" in prompt.lower()
    assert "flask" in prompt.lower()
    assert "read files before editing" in prompt.lower()
 
 
def test_trim_history_preserves_system_and_first_user():
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "first user"})
    for i in range(20):
        msgs.append({"role": "assistant", "content": f"a{i}"})
 
    trimmed = trim_history(msgs, max_messages=10)
    assert trimmed[0]["content"] == "sys"
    assert trimmed[1]["content"] == "first user"
    assert len(trimmed) == 10
    # The tail should include the most recent assistant messages
    assert trimmed[-1]["content"] == "a19"
 
 
def test_trim_history_noop_when_short():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert trim_history(msgs) == msgs


def _asst_tc(tc_id: str, name: str = "read_file") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tc_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool(tc_id: str, body: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "name": "read_file", "content": body}


def test_trim_history_drops_orphan_tool_results_at_tail_start():
    """A tool result whose matching assistant tool_calls got trimmed must not
    survive as the first tail message — that confuses the LLM into re-reading.
    """
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user"},
    ]
    # 6 read cycles: assistant(tool_calls) → tool result. 12 non-system messages total.
    for i in range(6):
        msgs.append(_asst_tc(f"c{i}"))
        msgs.append(_tool(f"c{i}", f"file{i}-content"))

    trimmed = trim_history(msgs, max_messages=10)
    # Tail (after system + first_user) must not begin with an orphan tool result.
    tail = trimmed[2:]
    assert tail, "expected non-empty tail"
    assert tail[0].get("role") != "tool", (
        f"tail starts with orphan tool result: {tail[0]}"
    )

    # Every tool_call_id in the trimmed history must have a preceding assistant
    # message in the same trimmed list that declares that id.
    declared_ids: set[str] = set()
    for m in trimmed:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                declared_ids.add(tc["id"])
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in declared_ids, (
                f"orphan tool result for {m['tool_call_id']} in trimmed history"
            )


def test_trim_history_strips_orphan_assistant_tool_calls():
    """An assistant message whose tool_call ids have no following tool messages
    must have its tool_calls stripped (or be dropped) — Azure-strict endpoints
    reject the request otherwise.
    """
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user"},
    ]
    # Build a long history where the last 6 entries are matched pairs (3 pairs)
    # but turn-7 assistant has tool_calls whose tool result will be dropped.
    msgs.append({"role": "assistant", "content": "salvageable text",
                 "tool_calls": [{"id": "orphan_x", "type": "function",
                                 "function": {"name": "read_file", "arguments": "{}"}}]})
    # Note: NO tool message for orphan_x.
    for i in range(3):
        msgs.append(_asst_tc(f"c{i}"))
        msgs.append(_tool(f"c{i}"))

    trimmed = trim_history(msgs, max_messages=10)
    # Every assistant message with tool_calls in trimmed must have all of its
    # tool_call.ids present as tool messages later in the trimmed list.
    for i, m in enumerate(trimmed):
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            continue
        required = {tc["id"] for tc in tcs}
        following: set[str] = set()
        for j in range(i + 1, len(trimmed)):
            mj = trimmed[j]
            if mj.get("role") == "tool":
                following.add(mj.get("tool_call_id", ""))
            elif mj.get("role") == "assistant":
                break
        assert required.issubset(following), (
            f"orphan assistant tool_calls {required - following} survived trimming"
        )


def test_trim_history_orphan_assistant_preserves_text_content():
    """When orphan tool_calls are stripped, the assistant's prose content
    should be preserved as a normal assistant message.
    """
    from squishy.context import _strip_orphan_assistant_tool_calls
    msgs = [
        {"role": "assistant", "content": "I will read the file",
         "tool_calls": [{"id": "orphan", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "user", "content": "(no tool result followed)"},
    ]
    out = _strip_orphan_assistant_tool_calls(msgs)
    assert len(out) == 2
    assert out[0]["role"] == "assistant"
    assert "tool_calls" not in out[0]
    assert out[0]["content"] == "I will read the file"


def test_trim_history_orphan_assistant_drops_when_empty():
    """An orphan assistant message with neither content nor reasoning
    should be dropped entirely (no empty turn left behind).
    """
    from squishy.context import _strip_orphan_assistant_tool_calls
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "orphan", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
    ]
    out = _strip_orphan_assistant_tool_calls(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_trim_history_keeps_matched_pairs_intact():
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user"},
    ]
    for i in range(3):
        msgs.append(_asst_tc(f"c{i}"))
        msgs.append(_tool(f"c{i}"))
    # Total 8 non-system → system + first_user + 6 non-first = 8 messages, under cap.
    trimmed = trim_history(msgs, max_messages=10)
    # Every tool result must be immediately preceded (somewhere earlier) by its
    # assistant tool_calls message.
    ids_in_order = []
    for m in trimmed:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                ids_in_order.append(("asst", tc["id"]))
        elif m.get("role") == "tool":
            ids_in_order.append(("tool", m["tool_call_id"]))
    for i, (kind, tc_id) in enumerate(ids_in_order):
        if kind == "tool":
            assert ("asst", tc_id) in ids_in_order[:i]


def test_system_prompt_includes_recall_when_index_exists(tmp_path):
    """When an index is present the rules block should encourage `recall`
    use up front and not steer the model toward blind reads."""
    (tmp_path / ".squishy").mkdir()
    (tmp_path / ".squishy" / "index.json").write_text("{}")
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)))
    assert "`recall(query=...)`" in prompt
    assert ".squishy/index.json" in prompt
    # The "no index" hint must not appear when one exists.
    assert "No repo index yet" not in prompt


def test_system_prompt_softer_recall_rule_when_no_index(tmp_path):
    """When there's no index, suggest /init rather than push `recall`."""
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)))
    assert "No repo index yet" in prompt
    assert "/init" in prompt


def test_system_prompt_no_duplicated_planning_block(tmp_path):
    """`## Planning` used to repeat what mode blocks already cover —
    the planning rule now lives once inside `## Rules`."""
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)), mode="plan")
    assert "## Planning" not in prompt
    # The planning rule should appear exactly once (it's in `## Rules`).
    assert prompt.count("update_plan(step_index=N") <= 1


def test_system_prompt_drops_json_shape_example(tmp_path):
    """The plan_task tool schema documents the JSON shape — repeating it
    here just bloats the prompt."""
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)), mode="plan")
    assert '```json' not in prompt
    assert '"files_to_modify"' not in prompt
    assert '"files_to_create"' not in prompt


def test_system_prompt_drops_shell_allowlist_enumeration(tmp_path):
    """The runtime error already enumerates the allowlist when the model
    guesses wrong, so don't burn tokens spelling it all out in prose.
    A short hint is fine; a full enumeration is not."""
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)), mode="plan")
    # The block used to list every binary explicitly: ls, cat, head, tail,
    # wc, grep, rg, find, pwd, which, file, stat, tree, ruff check, mypy,
    # pyright, git status/log/diff/show/branch/blame/ls-files, …
    assert "stat" not in prompt or "tree" not in prompt or "blame" not in prompt


def test_system_prompt_top_files_dropped_when_index_present(tmp_path):
    """If we have an index, the index block already lists the structure
    — repeating top-level files on top of that is pure duplication."""
    (tmp_path / ".squishy").mkdir()
    (tmp_path / ".squishy" / "index.json").write_text("{}")
    (tmp_path / "README.md").write_text("hi")
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)))
    assert "## Top-level files" not in prompt


def test_system_prompt_project_block_is_one_line(tmp_path):
    """Language/framework/build/test used to be 4 separate lines."""
    from squishy.context import ProjectInfo
    project = ProjectInfo(
        language="python",
        framework="fastapi",
        build_command="python -m build",
        test_command="pytest -q",
    )
    prompt = build_system_prompt(str(tmp_path), project)
    assert "## Project" in prompt
    block = prompt.split("## Project", 1)[1].split("##", 1)[0].strip()
    # All four pieces collapsed into a single line, joined with " · ".
    assert block.count("\n") == 0
    assert "language=python" in block
    assert "framework=fastapi" in block


def test_trim_history_preserves_plan_status_system_message():
    """A <plan-status> system message must survive trimming alongside the
    primary system prompt, regardless of where it appears in the list."""
    msgs: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user"},
    ]
    for i in range(15):
        msgs.append({"role": "assistant", "content": f"a{i}"})
    msgs.append({"role": "system", "content": "<plan-status>\nplan: foo\n</plan-status>"})
    for i in range(15, 20):
        msgs.append({"role": "assistant", "content": f"a{i}"})

    trimmed = trim_history(msgs, max_messages=10)
    kinds = [(m["role"], m.get("content", "")[:13]) for m in trimmed]
    assert ("system", "sys") in kinds
    assert any(role == "system" and content.startswith("<plan-status>") for role, content in kinds)
    assert trimmed[-1]["content"] == "a19"


def test_trim_history_noop_preserves_system_order():
    """trim_history preserves the system + first-user prefix when the
    message list is already under the cap. Plan-status and notes are
    no longer carried in the system message — they're injected as a
    transient (assistant tool_calls, tool result) pair by
    ``Agent._refresh_live_context_pair`` after trim/compact runs."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    trimmed = trim_history(msgs)
    assert trimmed[0]["content"] == "sys"
    assert trimmed[1]["content"] == "u"


# --- snip_old_tool_results tests ---


def test_snip_old_tool_results_truncates_old_large_tool():
    # F1: old read_file results get a terse marker that does NOT invite
    # the model to re-read.  v27 regressed because the prior "call
    # read_file again if needed" wording trained the agent into re-read
    # storms.  The replacement must (a) say "content unchanged on disk",
    # (b) NOT mention a (bogus) line count.
    big_content = "x" * 5000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "tool", "tool_call_id": "c0", "name": "read_file", "content": big_content},
        *[{"role": "assistant", "content": f"a{i}"} for i in range(8)],
    ]
    snip_old_tool_results(msgs, max_chars=2000, preserve_last_n=6)
    snipped = msgs[2]["content"]
    assert len(snipped) < len(big_content)
    assert "content unchanged on disk" in snipped
    # Anti-regression: the harmful wording must not return.
    assert "call read_file again" not in snipped
    # Anti-regression: no bogus line counts (read_file results are
    # JSON-encoded so "\n" never matches and the count was always 1).
    assert "lines" not in snipped


def test_snip_uses_squishy_read_path_when_present():
    """F1b: when the dispatch layer stamps `_squishy_read_path` on the
    tool message, the snipper should prefer it over regex-scraping the
    JSON body — which may already be truncated by a prior pass.
    """
    # Opaque content with no recoverable JSON path.
    opaque = "y" * 5000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {
            "role": "tool",
            "tool_call_id": "c0",
            "name": "read_file",
            "content": opaque,
            "_squishy_read_path": "scico/_xray.py",
        },
        *[{"role": "assistant", "content": f"a{i}"} for i in range(8)],
    ]
    snip_old_tool_results(msgs, max_chars=2000, preserve_last_n=6)
    snipped = msgs[2]["content"]
    assert "path=scico/_xray.py" in snipped
    assert "content unchanged on disk" in snipped


def test_snip_old_tool_results_mid_snips_non_read_file():
    # Non-read_file tool results that aren't read_file still get the
    # legacy mid-content snip when older than preserve_last_n.
    big_content = "x" * 5000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "tool", "tool_call_id": "c0", "name": "search_files", "content": big_content},
        *[{"role": "assistant", "content": f"a{i}"} for i in range(8)],
    ]
    snip_old_tool_results(msgs, max_chars=2000, preserve_last_n=6)
    snipped = msgs[2]["content"]
    assert len(snipped) < len(big_content)
    assert "[..." in snipped
    assert "chars snipped" in snipped


def test_snip_old_tool_results_preserves_recent():
    big_content = "y" * 5000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a0"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "tool_call_id": "c0", "name": "read_file", "content": big_content},
    ]
    snip_old_tool_results(msgs, max_chars=2000, preserve_last_n=6)
    # Tool message is within the last 6 — should NOT be snipped
    assert msgs[5]["content"] == big_content


# ── compact_messages tests ─────────────────────────────────────────────


class _FakeCompactClient:
    """Minimal client stub for compact_messages tests."""

    async def complete(self, messages, tools, stream=False, on_text=None):
        class _R:
            text = "Summary of old messages."
        return _R()


async def test_compact_messages_pulls_tool_results_with_anchored_assistant():
    """Anchored assistant messages should bring their paired tool results along."""
    # Build a conversation long enough to trigger compaction.
    # Use large content so char/4 estimate exceeds threshold.
    big = "x" * 4000
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "fix the bug" + big},
        # Old assistant turn with anchored tool call
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "edit_file", "arguments": '{"path":"a.py"}'}},
            ],
            "_squishy_anchor": True,
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "ok"},
        # More messages to push the anchored one into the "old" region
        {"role": "user", "content": "keep going" + big},
        {"role": "assistant", "content": "working on it" + big},
        {"role": "user", "content": "status?" + big},
        {"role": "assistant", "content": "almost done" + big},
    ]
    result = await compact_messages(msgs, _FakeCompactClient(), context_limit=2000, threshold=0.1)

    # Find the anchored assistant message in the result
    anchored = [m for m in result if m.get("_squishy_anchor")]
    assert anchored, "anchored assistant message should survive compaction"

    # The matching tool result should also be present
    tool_results = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "c1"]
    assert tool_results, "tool result paired with anchored assistant should also survive compaction"


def test_normalize_merges_consecutive_user_nudges():
    """Stacked [system] nudges (sent as role=user) must be coalesced —
    strict-alternation templates (Mistral) reject consecutive user turns."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
        {"role": "user", "content": "[system] nudge A"},
        {"role": "user", "content": "[system] nudge B"},
    ]
    out = normalize_messages(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["system", "user"]
    body = out[-1]["content"]
    assert "do the thing" in body and "nudge A" in body and "nudge B" in body


def test_normalize_does_not_merge_across_assistant():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "b"},
    ]
    assert [m["role"] for m in normalize_messages(msgs)] == ["user", "assistant", "user"]


def test_normalize_never_merges_assistant_tool_calls():
    """Merging assistant messages that carry tool_calls would break pairing."""
    msgs = [
        _asst_tc("c1"),
        {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "ok"},
        _asst_tc("c2"),
        {"role": "tool", "tool_call_id": "c2", "name": "run_command", "content": "ok"},
    ]
    out = normalize_messages(msgs)
    assert len(out) == 4
    assert sum(1 for m in out if m.get("tool_calls")) == 2


def test_normalize_merges_consecutive_prose_assistants():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "part 1"},
        {"role": "assistant", "content": "part 2"},
    ]
    out = normalize_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert "part 1" in out[-1]["content"] and "part 2" in out[-1]["content"]
