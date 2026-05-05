from __future__ import annotations

from squishy.config import INTERACTIVE_MODES, MODES, Config
from squishy.tool_restrictions import is_readonly_shell
from squishy.tools import check_permission, openai_schemas
from squishy.tools.fs import edit_file, read_file, write_file
from squishy.tools.shell import run_command


def test_cycle_mode_rotates_through_interactive_modes_only():
    """shift-tab cycle should never land on bench (benchmark-runner only)."""
    cfg = Config()
    cfg.permission_mode = "plan"
    seen = [cfg.permission_mode]
    for _ in range(len(INTERACTIVE_MODES) * 2):
        seen.append(cfg.cycle_mode())
    assert "bench" not in seen
    assert set(seen) == set(INTERACTIVE_MODES)


def test_cycle_mode_from_bench_lands_on_first_interactive():
    """If cfg was set to bench programmatically (e.g. by the bench runner),
    cycling escapes back into the interactive set rather than rotating to
    another non-interactive slot."""
    cfg = Config()
    cfg.permission_mode = "bench"
    assert cfg.cycle_mode() == INTERACTIVE_MODES[0]
    assert cfg.permission_mode in INTERACTIVE_MODES


def test_modes_constant_still_includes_bench():
    """MODES is the full universe (used by API validation); bench must
    still appear there even though shift-tab won't land on it."""
    assert "bench" in MODES


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
    # Each chain segment is checked independently — chains containing a
    # mutating segment are still rejected.
    assert not is_readonly_shell("ls; rm x")
    assert not is_readonly_shell("ls && rm x")
    assert not is_readonly_shell("ls | xargs rm")
    # File redirects always rejected — would write or read arbitrary paths.
    assert not is_readonly_shell("cat foo > bar")
    assert not is_readonly_shell("cat < bar")
    assert not is_readonly_shell("ls &> out.log")
    # Command substitution / variable expansion / background.
    assert not is_readonly_shell("echo `rm x`")
    assert not is_readonly_shell("echo $(rm x)")
    assert not is_readonly_shell("echo ${HOME}")
    assert not is_readonly_shell("ls &")


def test_is_readonly_shell_allows_pipes_between_readonly_segments():
    """Plan mode should allow pipes when *every* segment is read-only —
    e.g. piping linter output through `head` is a common ergonomics
    pattern with no extra capability beyond the segments themselves."""
    assert is_readonly_shell("ruff check . | head -100")
    assert is_readonly_shell("rg foo | wc -l")
    assert is_readonly_shell("ls -la | grep README")
    assert is_readonly_shell("git log --oneline | head")


def test_is_readonly_shell_allows_chains_of_readonly_segments():
    assert is_readonly_shell("ls; pwd")
    assert is_readonly_shell("git status && git log -1")
    assert is_readonly_shell("which ruff || which mypy")


def test_is_readonly_shell_allows_cd_prefix():
    """Agents commonly chain `cd /abs/path && readonly_cmd`. `cd` writes
    nothing, so the chain is safe as long as the trailing segment is."""
    assert is_readonly_shell("cd /tmp")
    assert is_readonly_shell("cd subdir && ruff check .")
    assert is_readonly_shell("cd /Users/me/proj && mypy 2>&1 | head -100")
    # cd chained with a mutating command is still rejected on the second
    # segment.
    assert not is_readonly_shell("cd /tmp && rm -rf foo")


def test_is_readonly_shell_allows_stderr_to_stdout_redirect():
    """`2>&1` and friends are pure fd swaps with no filesystem touch."""
    assert is_readonly_shell("ruff check . 2>&1")
    assert is_readonly_shell("mypy 2>&1")
    assert is_readonly_shell("ruff check . 2>&1 | head -100")
    assert is_readonly_shell("ls 1>&2")
    assert is_readonly_shell("git status 2>&1; git log -1 2>&1")


def test_plan_mode_run_command_allowlist_via_check_permission():
    allowed, _ = check_permission(run_command, "plan", {"command": "ls -la"})
    assert allowed
    allowed, reason = check_permission(run_command, "plan", {"command": "rm x"})
    assert not allowed
    assert "read-only" in reason
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


def test_schemas_hide_plan_task_when_plan_active():
    """Once a plan is approved the model should be executing, not
    replanning — `plan_task` is removed from the schema, but the
    progress-tracking tools stay."""
    for mode in ("edits", "yolo", "plan"):
        names = {s["function"]["name"] for s in openai_schemas(mode, plan_active=True)}
        assert "plan_task" not in names, f"{mode} mode still exposed plan_task"
        # update_plan and finish_plan must still be available so the
        # model can mark progress and end the work.
        assert "update_plan" in names, f"{mode} dropped update_plan"
        assert "finish_plan" in names, f"{mode} dropped finish_plan"


def test_schemas_keep_plan_task_when_no_plan_yet():
    """Without an approved plan, the agent still needs to be able to
    propose one — so `plan_task` stays visible."""
    plan_names = {s["function"]["name"] for s in openai_schemas("plan", plan_active=False)}
    assert "plan_task" in plan_names
    edits_names = {s["function"]["name"] for s in openai_schemas("edits", plan_active=False)}
    assert "plan_task" in edits_names
