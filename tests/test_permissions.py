from __future__ import annotations

from squishy.config import MODES, Config
from squishy.tool_restrictions import is_readonly_shell
from squishy.tools import check_permission, openai_schemas
from squishy.tools.fs import edit_file, read_file, write_file
from squishy.tools.shell import run_command


def test_cycle_mode_rotates_through_all_modes():
    cfg = Config()
    cfg.permission_mode = "plan"
    seen = [cfg.permission_mode]
    for _ in range(len(MODES)):
        seen.append(cfg.cycle_mode())
    # After one full cycle plus the starting entry we expect each mode once
    assert set(seen) == set(MODES)


def test_plan_mode_allows_reads_blocks_writes():
    allowed, _ = check_permission(read_file, "plan")
    assert allowed
    allowed, reason = check_permission(write_file, "plan")
    assert not allowed
    assert "plan mode" in reason


def test_edits_mode_allows_writes_blocks_run():
    allowed, _ = check_permission(write_file, "edits")
    assert allowed
    allowed, _ = check_permission(edit_file, "edits")
    assert allowed
    allowed, reason = check_permission(run_command, "edits")
    assert not allowed
    assert reason == "prompt"


def test_yolo_allows_all():
    for tool in (read_file, write_file, edit_file, run_command):
        allowed, _ = check_permission(tool, "yolo")
        assert allowed


def test_is_readonly_shell_accepts_safe_commands():
    assert is_readonly_shell("ls")
    assert is_readonly_shell("ls -la src/")
    assert is_readonly_shell("cat README.md")
    assert is_readonly_shell("grep -r foo .")
    assert is_readonly_shell("rg --files")
    assert is_readonly_shell("git log --oneline -5")
    assert is_readonly_shell("git status")
    assert is_readonly_shell("git diff HEAD~1")
    assert is_readonly_shell("pytest --collect-only")
    assert is_readonly_shell("python -m pytest --collect-only tests/")
    assert is_readonly_shell("ruff check squishy")
    assert is_readonly_shell("mypy squishy")


def test_is_readonly_shell_rejects_mutating_and_chained():
    assert not is_readonly_shell("")
    assert not is_readonly_shell("   ")
    assert not is_readonly_shell("rm -rf /")
    assert not is_readonly_shell("mv a b")
    assert not is_readonly_shell("cp x y")
    assert not is_readonly_shell("python script.py")  # runs arbitrary code
    assert not is_readonly_shell("pytest")  # bare pytest runs tests (can mutate)
    # Metacharacter rejection — these contain safe heads but chain unsafe tails
    assert not is_readonly_shell("ls; rm x")
    assert not is_readonly_shell("ls && rm x")
    assert not is_readonly_shell("ls | xargs rm")
    assert not is_readonly_shell("cat foo > bar")
    assert not is_readonly_shell("cat < bar")
    assert not is_readonly_shell("echo `rm x`")
    assert not is_readonly_shell("echo $(rm x)")
    assert not is_readonly_shell("ls &")


def test_plan_mode_run_command_allowlist_via_check_permission():
    allowed, _ = check_permission(run_command, "plan", {"command": "ls -la"})
    assert allowed
    allowed, reason = check_permission(run_command, "plan", {"command": "rm x"})
    assert not allowed
    assert "read-only shell" in reason
    # Missing/blank command → denied
    allowed, _ = check_permission(run_command, "plan", {})
    assert not allowed


def test_schemas_are_mode_scoped():
    plan_names = {s["function"]["name"] for s in openai_schemas("plan")}
    assert "plan_task" in plan_names
    assert "update_plan" in plan_names
    assert "read_file" in plan_names
    assert "run_command" in plan_names  # gated per-call, but visible
    assert "write_file" not in plan_names
    assert "edit_file" not in plan_names

    edits_names = {s["function"]["name"] for s in openai_schemas("edits")}
    assert {"write_file", "edit_file", "run_command"}.issubset(edits_names)

    yolo_names = {s["function"]["name"] for s in openai_schemas("yolo")}
    assert "write_file" in yolo_names and "run_command" in yolo_names

    all_names = {s["function"]["name"] for s in openai_schemas()}
    assert "write_file" in all_names  # backwards-compatible (no mode → all)
