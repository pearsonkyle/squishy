"""Tolerance layer: alternate tool/param vocabularies map to canonical tools."""

from __future__ import annotations

from squishy.tool_aliases import (
    canonical_tool_name,
    normalize_args,
    normalize_call,
)
from squishy.tools import REGISTRY


def test_tool_name_aliases_map_to_canonical():
    assert canonical_tool_name("bash") == "run_command"
    assert canonical_tool_name("grep") == "search_files"
    assert canonical_tool_name("str_replace_based_edit_tool") == "edit_file"
    assert canonical_tool_name("view") == "read_file"
    assert canonical_tool_name("create") == "write_file"
    assert canonical_tool_name("ls") == "list_directory"
    assert canonical_tool_name("glob") == "glob_files"


def test_canonical_names_are_not_remapped():
    for name in ("read_file", "write_file", "edit_file", "run_command", "recall"):
        assert canonical_tool_name(name) == name


def test_case_insensitive_name_alias():
    assert canonical_tool_name("Bash") == "run_command"
    assert canonical_tool_name("GREP") == "search_files"


def test_unknown_name_passes_through():
    assert canonical_tool_name("totally_made_up") == "totally_made_up"


def test_every_alias_target_is_a_real_tool():
    from squishy.tool_aliases import TOOL_NAME_ALIASES
    for target in TOOL_NAME_ALIASES.values():
        assert target in REGISTRY, f"alias target {target} is not a registered tool"


def test_no_alias_shadows_a_real_tool_name():
    """An alias key must not collide with an actual tool name (which would
    silently rewrite a legitimate call)."""
    from squishy.tool_aliases import TOOL_NAME_ALIASES
    for alias in TOOL_NAME_ALIASES:
        assert alias not in REGISTRY, f"alias {alias} shadows a real tool"


def test_param_aliases_remap_when_canonical_absent():
    _, args = normalize_call("bash", {"cmd": "ls -la"})
    assert args == {"command": "ls -la"}

    _, args = normalize_call("read_file", {"file_path": "a.py"})
    assert args == {"path": "a.py"}

    _, args = normalize_call("str_replace", {"file_path": "a.py", "old_string": "x", "new_string": "y"})
    assert args == {"path": "a.py", "old_str": "x", "new_str": "y"}

    _, args = normalize_call("grep", {"regex": "TODO", "directory": "src"})
    assert args == {"pattern": "TODO", "path": "src"}


def test_explicit_canonical_wins_over_alias():
    # Both canonical `path` and alias `file_path` present → keep canonical.
    args = normalize_args("read_file", {"path": "real.py", "file_path": "alias.py"})
    assert args["path"] == "real.py"


def test_non_dict_args_pass_through():
    # The client's parse-error sentinel is a dict, but a stray non-dict must
    # not crash normalization.
    assert normalize_args("read_file", None) is None
    assert normalize_args("read_file", "oops") == "oops"


def test_tool_arg_error_sentinel_survives():
    args = normalize_args("read_file", {"_tool_arg_error": "bad json", "_raw": "{"})
    assert args["_tool_arg_error"] == "bad json"
    assert args["_raw"] == "{"
