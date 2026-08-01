"""The bug-fixing arm: exploration tools plus the ability to change code.

:mod:`graphagent.agentkit.factory` builds read-only question-answering agents,
which is what the tool-call/token A/B measures. This module builds the agent
that has to land a patch, in the two variants the SWE-bench comparison needs:

* ``graph=None`` — filesystem discovery (``list_dir``/``grep``/``read_file``).
  This is the reference-agent shape that reaches a 100% patch rate.
* ``graph=<CodeGraph>`` — the same agent with ``explore``/``impact_of`` in
  place of the crawl.

Both variants get identical write tools and identical task framing, so the
toolset stays the only variable.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agents import Agent, function_tool
from agents.models.interface import Model

from graphagent.agentkit import edit as write_impl
from graphagent.agentkit import tools as impl
from graphagent.agentkit.llm import DEFAULT_MODEL
from graphagent.agentkit.metrics import ToolTrace
from graphagent.graph.store import CodeGraph

# Shared by both arms. Two of these paragraphs are paid for in failed runs:
# "change the source, not the tests" (a model that edits the test file scores
# zero), and "reproduce it first" (the technique the reference agent wins
# with — a throwaway script derived from the traceback is an oracle you can
# run in a second, and unlike the graded test it cannot already be green).
_TASK_FRAMING = """\
You are a senior software engineer fixing a bug in a repository you are
standing in. Work autonomously; nobody will answer questions.

Change the SOURCE CODE — the module that implements the behavior. Editing or
adding tests is never a fix on its own, and test files are not where the
graded change belongs.

Reproduce the failure before you fix it. Build the smallest snippet that
triggers what the report describes — `run_command` with `python -c "..."`, or
a scratch file under /tmp — and run it until you see the error yourself. That
reproduction is your oracle: it tells you the bug is real and, later, that
your fix landed. Keep scratch files in /tmp so they stay out of the diff.

Do not finish until you have actually edited a non-test source file. A run
that ends with no edit is scored as a failure. When the fix is in, re-run your
reproduction, then give a two-line summary."""

_FS_STRATEGY = """\
Find the relevant code with grep, list_dir, and read_file. Read only what you
need to make the change."""

_GRAPH_STRATEGY = """\
This repository has a pre-built code knowledge graph covering every symbol,
call, import, and inheritance edge. DO NOT crawl files to find things. Call
`explore` first: one call returns the matching symbols' verbatim source, their
callers and callees, and the impact radius. Use `impact_of` before you change
a shared function, and `repo_map` only when you need a survey. Trust what the
graph returns — do not re-read a file to confirm it. `read_file` returns text
as usual, except on a file too long to show at once, where it returns an
outline instead: take the symbol you want from that outline and call
`symbol_source`, or pass start_line/end_line for exact lines. `grep` is for
strings that are not Python symbols."""


# Consecutive commands with nothing edited before the harness says so. Eight
# is roughly where cfn-lint-3965's probe chain stopped producing new
# information and started restating the same query.
_PROBE_LIMIT = 8


def _clip(value: object, limit: int = 90) -> str:
    return " ".join(str(value).split())[:limit]


def _budget_notice(used: int, budget: int) -> str:
    """What to say to an agent that is running out of turns without editing.

    Nothing in the SDK tells the model the clock is running, so it spends the
    budget the way it would spend an unbounded one. qiskit-terra-5662's graph
    arm wrote sixteen reproduction scripts and hit the 50-turn cap having
    never touched a source file — a run scored zero for want of one sentence.
    Half the budget is a reminder; four fifths is an instruction.
    """
    if budget <= 0:
        return ""
    left = budget - used
    if used < budget // 2:
        return ""
    head = f"\n\n[budget] Turn {used} of {budget}, {left} left, and no source "
    if used >= budget * 4 // 5:
        return (
            head + "file has been edited. Make the edit on your next call, "
            "using the best explanation you have. An unedited run scores zero; "
            "a partial fix does not."
        )
    return (
        head + "file has been edited yet. Stop investigating and change the "
        "code you already have evidence for — you can keep testing afterwards."
    )


def build_swe_agent(
    root: Path,
    graph: CodeGraph | None = None,
    model: str | Model = DEFAULT_MODEL,
    trace: ToolTrace | None = None,
    progress: Callable[[], tuple[int, int]] | None = None,
) -> Agent[None]:
    """Agent that can explore *and* edit; graph tools when a graph is given.

    ``progress`` returns ``(turns_used, turn_budget)``. When given, tool
    results carry a budget notice until the first source edit lands.
    """
    root = root.resolve()
    trace = trace or ToolTrace()

    seen: dict[tuple[str, str], int] = {}
    edited = False
    probes = 0

    def _log(name: str, summary: str, result: str) -> str:
        nonlocal edited, probes
        failed = result.startswith("error:")
        trace.record(
            name,
            summary,
            ok=not failed,
            error=_clip(result[len("error:") :], 80) if failed else "",
        )
        # Only a repo file counts as an edit. A reproduction script under /tmp
        # is the right move and deliberately never reaches the diff, so
        # treating it as "the edit landed" would switch the pressure off at
        # exactly the moment it is needed.
        repo_edit = (
            not failed
            and name in ("edit_file", "write_file")
            and not Path(summary.split("  ")[0]).is_absolute()
        )
        edited = edited or repo_edit
        # Loop-breaking belongs in the tool result, not in an injected user
        # turn: it is causally paired with the call that caused it and cannot
        # desynchronize the transcript. On qiskit-terra-5662 the graph arm ran
        # the same dead-end `explore` three times and then gave up without
        # editing anything.
        key = (name, summary)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1 and name != "run_command":
            result += (
                f"\n\n[repeat] Identical call #{seen[key]}: the result above is "
                "the same as last time and will not change. Take a different "
                "step — a different query, or the edit itself."
            )
        # Investigation that never converges. cfn-lint-3965's graph arm spent
        # 37 of its 48 calls on `python -c` probes into one schema manager,
        # each a small variation of the last, and hit the turn cap without an
        # edit. The exact-repeat brake above cannot see this — no two probes
        # are identical — and it deliberately exempts `run_command`, because
        # re-running a reproduction *after* an edit is the correct move. Only
        # a run of them with nothing edited is the failure.
        if name == "run_command":
            probes += 1
        elif repo_edit:
            probes = 0
        if probes >= _PROBE_LIMIT and not edited:
            result += (
                f"\n\n[probes] {probes} commands run and no source file changed "
                "yet. Experiments are not converging on their own — make the "
                "edit your evidence already supports, then re-run this command "
                "to check it."
            )
        if progress is not None and not edited:
            result += _budget_notice(*progress())
        return result

    @function_tool
    def read_file(relative: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file slice with line numbers (max 400 lines per call).

        Args:
            relative: Repo-relative file path.
            start_line: First line to read (1-based).
            end_line: Last line to read; 0 means to end of file.
        """
        # Outline only for files that do not fit in one read. Measured, after
        # shipping the opposite: a turn costs 8.5-11k prompt tokens because
        # the whole transcript is resent, while an outline saves 0.4k on a
        # short file and 3.7k on a long one. An outline that forces a
        # follow-up `symbol_source` is therefore a net loss at every size —
        # which is what the numbers showed, cfn-lint's graph arm going from
        # 20/29/42 tool calls to 43/48/29 and losing both its resolves. Above
        # the read cap the file truncates anyway, so the extra call was always
        # going to happen and the outline is strictly better than a truncated
        # prefix.
        if (
            graph is not None
            and start_line <= 1
            and end_line == 0
            and impl.exceeds_read_cap(graph, relative)
        ):
            outline = impl.file_outline(graph, relative)
            if outline:
                return _log("read_file", f"{relative} [outline]", outline)
        return _log(
            "read_file",
            f"{relative} [{start_line}-{end_line or 'eof'}]",
            impl.read_file(root, relative, start_line, end_line or None),
        )

    @function_tool
    def grep(pattern: str, glob: str = "*.py") -> str:
        """Regex-search file contents; returns path:line: text matches.

        Args:
            pattern: Python regular expression.
            glob: Filename glob to restrict the search, default "*.py".
        """
        return _log("grep", _clip(pattern), impl.grep(root, pattern, glob))

    @function_tool
    def list_dir(relative: str = ".") -> str:
        """List entries of a directory inside the repo (dirs end with /).

        Args:
            relative: Repo-relative directory path, e.g. "src/pkg".
        """
        return _log("list_dir", relative, impl.list_dir(root, relative))

    @function_tool
    def edit_file(relative: str, old_str: str, new_str: str) -> str:
        """Replace one unique occurrence of old_str with new_str.

        Args:
            relative: Repo-relative file path.
            old_str: Exact text to replace, unique within the file.
            new_str: Replacement text.
        """
        return _log(
            "edit_file",
            f"{relative}  old_str={_clip(old_str.splitlines()[0] if old_str else '', 60)!r}",
            write_impl.edit_file(root, relative, old_str, new_str),
        )

    @function_tool
    def write_file(relative: str, content: str) -> str:
        """Create or overwrite a file. Use /tmp paths for scratch scripts.

        Args:
            relative: File path (repo-relative, or absolute under /tmp).
            content: Full file contents.
        """
        return _log(
            "write_file", relative, write_impl.write_file(root, relative, content)
        )

    @function_tool
    def run_command(command: str) -> str:
        """Run a shell command in the repository root and return its output.

        Args:
            command: Shell command, e.g. 'python -c "import x; x.f()"'.
        """
        return _log(
            "run_command", _clip(command, 140), write_impl.run_command(root, command)
        )

    tools = [read_file, edit_file, write_file, run_command]

    if graph is None:
        tools += [grep, list_dir]
        instructions = f"{_TASK_FRAMING}\n\n{_FS_STRATEGY}"
    else:

        @function_tool
        def explore(query: str) -> str:
            """Answer almost any code question in one call.

            Returns the best-matching symbols' verbatim source, their
            callers/callees, subclasses, and impact radius. Use this first.

            Args:
                query: A symbol name, dotted name (Class.method), or keyword.
            """
            return _log("explore", _clip(query), impl.explore(graph, root, query))

        @function_tool
        def impact_of(symbol: str, depth: int = 2) -> str:
            """Blast radius: everything that transitively depends on a symbol.

            Args:
                symbol: Symbol name to analyze.
                depth: Maximum reverse-dependency hops (default 2).
            """
            return _log(
                "impact_of", _clip(symbol), impl.impact_of(graph, symbol, depth)
            )

        @function_tool
        def repo_map() -> str:
            """Compact overview of every file with its classes and functions."""
            return _log("repo_map", "", impl.repo_map(graph))

        tools += [explore, impact_of, repo_map, grep, list_dir]
        instructions = f"{_TASK_FRAMING}\n\n{_GRAPH_STRATEGY}"

    return Agent(
        name="graph-swe" if graph is not None else "baseline-swe",
        instructions=instructions,
        model=model,
        tools=tools,
    )
