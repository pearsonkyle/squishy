"""Schema hygiene: the model only knows what the schema tells it.

Every failure these guard against costs at least one turn to discover by
trial and error, which is the most expensive way for a small model to learn
an interface.
"""
from __future__ import annotations

import pytest

from squishy.tool_aliases import PARAM_ALIASES
from squishy.tools import ALL_TOOLS

_TOOLS = {t.name: t for t in ALL_TOOLS}


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_parameter_is_documented(tool):
    props = (tool.parameters or {}).get("properties") or {}
    undocumented = [k for k, v in props.items() if not (v or {}).get("description")]
    assert not undocumented, (
        f"{tool.name} has undocumented parameters: {undocumented}")


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_descriptions_say_something(tool):
    assert len(tool.description) >= 40, f"{tool.name} description is too thin"
    props = (tool.parameters or {}).get("properties") or {}
    for name, spec in props.items():
        desc = (spec or {}).get("description", "")
        # A description that just restates the parameter name is noise.
        assert desc.lower().rstrip(".") != name.replace("_", " "), (
            f"{tool.name}.{name} description restates the name")


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_required_params_actually_exist(tool):
    params = tool.parameters or {}
    props = params.get("properties") or {}
    for req in params.get("required") or []:
        assert req in props, f"{tool.name} requires undeclared param {req!r}"


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_param_aliases_point_at_real_params(tool):
    """An alias mapping to a nonexistent param would silently drop the value."""
    props = (tool.parameters or {}).get("properties") or {}
    for alias, canonical in (PARAM_ALIASES.get(tool.name) or {}).items():
        assert canonical in props, (
            f"{tool.name}: alias {alias!r} -> {canonical!r}, which is not a parameter")
        assert alias not in props, (
            f"{tool.name}: {alias!r} is both a real parameter and an alias")


def test_no_tool_advertises_a_repl_only_command():
    """`/init` is a REPL slash command — the model has no way to call it.

    recall's description used to tell the model to "run /init", which it
    cannot do; the tool is now simply hidden when no index exists.
    """
    for tool in ALL_TOOLS:
        assert "/init" not in tool.description, (
            f"{tool.name} tells the model to run a command it cannot call")


def test_path_taking_tools_agree_on_the_parameter_name():
    for name in ("read_file", "write_file", "edit_file", "list_directory"):
        props = (_TOOLS[name].parameters or {}).get("properties") or {}
        assert "path" in props, f"{name} should take `path`, not a synonym"
