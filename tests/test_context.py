from __future__ import annotations
 
import json
 
from squishy.context import build_system_prompt, detect_project, snip_old_tool_results, trim_history
 
 
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


def test_system_prompt_includes_recall_first_when_index_exists(tmp_path):
    (tmp_path / ".squishy").mkdir()
    (tmp_path / ".squishy" / "index.json").write_text("{}")
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)))
    assert "An index is present" in prompt
    assert "Do not read files blindly" in prompt


def test_system_prompt_softer_recall_rule_when_no_index(tmp_path):
    prompt = build_system_prompt(str(tmp_path), detect_project(str(tmp_path)))
    assert "prefer `recall" in prompt
    assert "Do not read files blindly" not in prompt


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


def test_trim_history_noop_orders_plan_status_after_primary_system():
    msgs = [
        {"role": "system", "content": "<plan-status>\nx\n</plan-status>"},
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    trimmed = trim_history(msgs)
    assert trimmed[0]["content"] == "sys"
    assert trimmed[1]["content"].startswith("<plan-status>")


# --- snip_old_tool_results tests ---


def test_snip_old_tool_results_truncates_old_large_tool():
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
