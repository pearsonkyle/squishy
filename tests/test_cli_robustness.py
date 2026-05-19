"""Tests for CLI-layer UX hardening from the v6d display audit:

  - ``_run_direct_command`` enforces a timeout and surfaces interrupts
    instead of leaving an orphan child process running.
  - ``_create_session_for_agent`` surfaces session-creation failures
    via display.warn instead of swallowing them silently.
  - ``_slash_extra_args`` parses ``"/cmd args"`` into the components
    the no-arg-extra-warning logic expects.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import squishy.cli as cli


# -- _run_direct_command ------------------------------------------------------


async def test_run_direct_command_returns_exit_code():
    rc = await cli._run_direct_command("true")
    assert rc == 0
    rc = await cli._run_direct_command("false")
    assert rc == 1


async def test_run_direct_command_kills_on_timeout():
    """A long-running command is killed and 124 is returned (matches
    GNU timeout(1))."""
    rc = await cli._run_direct_command("sleep 10", timeout=0.2)
    assert rc == 124


async def test_run_direct_command_propagates_cancellation():
    """Cancelling the awaited future kills the child and re-raises
    CancelledError so the REPL can show ``[shell] interrupted``
    instead of leaving a zombie process.

    The helper re-raises CancelledError (not KeyboardInterrupt) so
    test runners' Ctrl-C handlers don't get confused; the REPL caller
    catches both shapes."""
    task = asyncio.create_task(cli._run_direct_command("sleep 10"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# -- _create_session_for_agent ------------------------------------------------


def test_create_session_disabled_returns_none_silently():
    cfg = SimpleNamespace(save_sessions=False)
    warns: list[str] = []
    display = SimpleNamespace(warn=warns.append)
    sid = cli._create_session_for_agent(cfg, "model-x", display)
    assert sid is None
    assert warns == []  # disabled is not a failure


def test_create_session_failure_surfaces_via_display(monkeypatch):
    """A broken session_dir should warn the user, not vanish."""
    cfg = SimpleNamespace(
        save_sessions=True,
        permission_mode="bench",
        working_dir="/tmp",
        session_dir="/dev/null/not-writable",
    )

    def boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "create_session", boom)
    warns: list[str] = []
    display = SimpleNamespace(warn=warns.append)
    sid = cli._create_session_for_agent(cfg, "model-x", display)
    assert sid is None
    assert any("session" in w and "disk full" in w for w in warns), warns


def test_create_session_failure_without_display_does_not_raise(monkeypatch):
    """Old call sites that don't pass a display still get None back."""
    cfg = SimpleNamespace(
        save_sessions=True,
        permission_mode="bench",
        working_dir="/tmp",
        session_dir="/x",
    )
    monkeypatch.setattr(cli, "create_session", lambda **_kw: (_ for _ in ()).throw(RuntimeError("x")))
    sid = cli._create_session_for_agent(cfg, "model-x", None)
    assert sid is None


# -- _slash_extra_args --------------------------------------------------------


def test_slash_extra_args_no_args():
    head, rest = cli._slash_extra_args("/clear")
    assert head == "/clear"
    assert rest == ""


def test_slash_extra_args_with_args():
    head, rest = cli._slash_extra_args("/mode plan")
    assert head == "/mode"
    assert rest == "plan"


def test_slash_extra_args_strips_trailing_whitespace():
    head, rest = cli._slash_extra_args("/clear   all   \n")
    assert head == "/clear"
    assert rest == "all"


def test_slash_extra_args_non_slash_returns_empty():
    head, rest = cli._slash_extra_args("hello world")
    assert head == ""
    assert rest == ""


def test_no_arg_slash_cmds_includes_expected_commands():
    """Sanity check — extra-arg warning fires for the no-arg commands."""
    expected = {"/clear", "/new", "/status", "/plan", "/help",
                "/quit", "/exit", "/q", "/session", "/sessions",
                "/exit-plan"}
    assert expected <= cli._NO_ARG_SLASH_CMDS
