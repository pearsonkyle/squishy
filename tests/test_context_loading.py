"""Tests for agent-instruction loading and mode-aware system prompts."""

from __future__ import annotations

from pathlib import Path

from squishy.context import (
    ProjectInfo,
    build_system_prompt,
    load_agent_instructions,
)


def test_load_agent_instructions_empty_when_no_files(tmp_path: Path) -> None:
    assert load_agent_instructions(str(tmp_path)) == ""


def test_load_agent_instructions_reads_agents_md(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Project rules\nUse relative imports.\n")
    out = load_agent_instructions(str(tmp_path))
    assert "## Agent instructions (AGENTS.md)" in out
    assert "Use relative imports." in out


def test_load_agent_instructions_combines_multiple_sources(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("AGENTS content\n")
    (tmp_path / "CLAUDE.md").write_text("CLAUDE content\n")
    (tmp_path / "SQUISHY.md").write_text("SQUISHY content\n")
    squishy_dir = tmp_path / ".squishy"
    squishy_dir.mkdir()
    (squishy_dir / "AGENTS.md").write_text("generated content\n")

    out = load_agent_instructions(str(tmp_path))
    for marker in (
        "AGENTS content",
        "CLAUDE content",
        "SQUISHY content",
        "generated content",
    ):
        assert marker in out, marker
    # Order: root AGENTS.md before .squishy/AGENTS.md
    assert out.index("AGENTS content") < out.index("generated content")


def test_load_agent_instructions_truncates_huge_files(tmp_path: Path) -> None:
    big = "x" * 20_000
    (tmp_path / "AGENTS.md").write_text(big)
    out = load_agent_instructions(str(tmp_path))
    assert "…(truncated)" in out
    # Content cap is ~4 KB plus a bit of framing
    assert len(out) < 6_000


def test_build_system_prompt_includes_instructions(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("KEY_RULE: never use sudo\n")
    prompt = build_system_prompt(str(tmp_path), ProjectInfo(), mode="plan")
    assert "KEY_RULE: never use sudo" in prompt


def test_build_system_prompt_plan_mode_block(tmp_path: Path) -> None:
    prompt = build_system_prompt(str(tmp_path), ProjectInfo(), mode="plan")
    assert "Mode: plan" in prompt
    assert "plan_task" in prompt
    assert "write_file" in prompt  # mentioned as forbidden


def test_build_system_prompt_edits_mode_block(tmp_path: Path) -> None:
    prompt = build_system_prompt(str(tmp_path), ProjectInfo(), mode="edits")
    assert "Mode: edits" in prompt


def test_build_system_prompt_yolo_mode_block(tmp_path: Path) -> None:
    prompt = build_system_prompt(str(tmp_path), ProjectInfo(), mode="yolo")
    assert "Mode: yolo" in prompt
