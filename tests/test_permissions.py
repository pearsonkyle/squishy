from __future__ import annotations

from squishy.config import INTERACTIVE_MODES, MODES, Config
from squishy.tools import check_permission
from squishy.tools.fs import edit_file, read_file, write_file
from squishy.tools.shell import run_command




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
























