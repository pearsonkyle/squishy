from __future__ import annotations
 
from squishy.config import MODES, Config
from squishy.tools import check_permission
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
