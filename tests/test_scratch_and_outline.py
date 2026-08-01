"""Two things the file tools tell the model to do, and now let it do.

* Write reproduction scripts to /tmp. `write_file`'s own refusal message has
  said so for a while; until `_resolve_writable` existed, following that
  instruction returned "path outside working directory". The harness refusing
  the action it just demanded is this codebase's recurring failure mode.
* Read a file too long to return. The body gets snipped mid-JSON at the output
  cap, so the follow-up read was always going to happen — an outline is
  strictly better than a truncated prefix, and only there.
"""

from __future__ import annotations

import os
import pathlib

from squishy.graph import build_repo_graph
from squishy.tools import dispatch
from squishy.tools.base import ToolContext


def _ctx(tmp_path, mode="bench", **kw) -> ToolContext:
    return ToolContext(
        working_dir=str(tmp_path), permission_mode=mode, use_sandbox=False, **kw
    )


async def test_a_repro_script_can_be_written_to_the_scratch_dir(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    target = os.path.join("/tmp", f"squishy_repro_{os.getpid()}.py")
    try:
        res = await dispatch(
            "write_file", {"path": target, "content": "print('boom')\n"}, ctx
        )
        assert res.success, res.error
        assert res.data["scratch"] is True
        assert pathlib.Path(target).read_text() == "print('boom')\n"
    finally:
        if os.path.exists(target):
            os.unlink(target)


async def test_a_scratch_script_can_be_rewritten(tmp_path) -> None:
    """The new-files-only rule is about the graded diff; scratch has none.

    Blocking the second draft of a repro script would make the technique
    single-shot, which is not how reproducing a bug works.
    """
    ctx = _ctx(tmp_path)
    target = os.path.join("/tmp", f"squishy_repro2_{os.getpid()}.py")
    try:
        assert (await dispatch("write_file", {"path": target, "content": "a\n"}, ctx)).success
        res = await dispatch("write_file", {"path": target, "content": "b\n"}, ctx)
        assert res.success, res.error
        assert pathlib.Path(target).read_text() == "b\n"
    finally:
        if os.path.exists(target):
            os.unlink(target)


async def test_a_scratch_script_can_be_read_back(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    target = os.path.join("/tmp", f"squishy_repro3_{os.getpid()}.py")
    try:
        await dispatch("write_file", {"path": target, "content": "x = 1\n"}, ctx)
        res = await dispatch("read_file", {"path": target}, ctx)
        assert res.success, res.error
        assert "x = 1" in res.data["content"]
    finally:
        if os.path.exists(target):
            os.unlink(target)


async def test_paths_outside_the_repo_and_scratch_are_still_refused(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    res = await dispatch(
        "write_file", {"path": "/etc/squishy_should_not_exist", "content": "x"}, ctx
    )
    assert not res.success
    assert "outside working directory" in res.error


async def test_a_long_file_returns_an_outline_instead_of_a_snipped_body(
    sample_repo,
) -> None:
    body = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(400))
    (sample_repo / "demo" / "huge.py").write_text(body, encoding="utf-8")
    build_repo_graph(sample_repo)
    ctx = _ctx(sample_repo, max_tool_output_chars=2000)

    res = await dispatch("read_file", {"path": "demo/huge.py"}, ctx)
    assert res.success
    assert "outline" in res.data
    assert "def f399" in res.data["outline"], "the tail must be reachable"
    assert "content" not in res.data


async def test_a_short_file_is_never_outlined(sample_repo) -> None:
    """Measured: a turn costs 8.5-11k prompt tokens because the transcript is
    resent, while an outline saves 0.4-3.7k. An outline that forces a
    follow-up read is a net loss at every size below the cap."""
    build_repo_graph(sample_repo)
    ctx = _ctx(sample_repo)
    res = await dispatch("read_file", {"path": "demo/utils.py"}, ctx)
    assert res.success
    assert "content" in res.data
    assert "outline" not in res.data


async def test_an_explicit_range_is_never_outlined(sample_repo) -> None:
    """offset/limit is the model asking for exact lines. Answer the question."""
    body = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(400))
    (sample_repo / "demo" / "huge.py").write_text(body, encoding="utf-8")
    build_repo_graph(sample_repo)
    ctx = _ctx(sample_repo, max_tool_output_chars=2000)
    res = await dispatch(
        "read_file", {"path": "demo/huge.py", "offset": 10, "limit": 5}, ctx
    )
    assert res.success
    assert "content" in res.data and "outline" not in res.data


async def test_a_long_file_with_no_graph_still_returns_its_body(tmp_path) -> None:
    """No graph means no outline — the read must not silently return less."""
    (tmp_path / "huge.py").write_text(
        "\n".join(f"x{i} = {i}" for i in range(5000)), encoding="utf-8"
    )
    ctx = _ctx(tmp_path, max_tool_output_chars=2000)
    res = await dispatch("read_file", {"path": "huge.py"}, ctx)
    assert res.success
    assert "content" in res.data


def test_the_bench_prompt_does_not_ban_what_the_tools_enable(tmp_path) -> None:
    """The prompt, the tool's error message and the harness must agree.

    They did not: the bench block said "don't write reproduction scripts"
    while `write_file`'s own refusal told the model to write them under /tmp,
    and the SWE-rebench prompt asked for one in its second paragraph. Being
    told to do something and forbidden from doing it in the same request is
    the failure this codebase keeps rediscovering.
    """
    from squishy.context import build_system_prompt, detect_project

    for profile in ("standard", "minimal", "graph"):
        prompt = build_system_prompt(
            str(tmp_path), detect_project(str(tmp_path)), False, "bench", profile
        )
        assert "Don't write reproduction scripts" not in prompt
        if "## Task" in prompt:
            assert "Reproduce the failure" in prompt, profile
            assert "/tmp" in prompt, profile
