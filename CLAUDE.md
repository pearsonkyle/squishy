# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Squishy

Squishy is a minimal async Python coding agent that wraps any OpenAI-compatible LLM endpoint into a tool-calling assistant. It ships as both an interactive CLI (`squishy`) and a programmatic API (`squishy.api.Squishy`), plus SWE-bench and Terminal-bench evaluation harnesses (`squishy-bench`).

## Commands

```bash
# Install (Python 3.11+)
pip install -e '.[dev]'

# Run all unit tests (excludes smoke tests)
pytest -q

# Run a single test file or test
pytest tests/test_agent_loop.py -q
pytest tests/test_agent_loop.py::test_basic_loop -q

# Run smoke tests (requires live LLM endpoint)
SQUISHY_BASE_URL=http://host:port/v1 SQUISHY_MODEL=model-name pytest -m smoke

# Lint
ruff check squishy/ tests/
ruff format --check squishy/ tests/

# Type check (strict only on tools/base.py, context.py, config.py, errors.py)
mypy squishy/

# Run the CLI
squishy --base-url http://localhost:1234/v1 --model local-model
squishy -m "one-shot task"          # non-interactive
echo "task" | squishy               # pipe mode

# Run SWE-bench evaluation
squishy-bench swe --instances data.jsonl --model m --output predictions.jsonl
# Run Terminal-bench evaluation
squishy-bench term --tasks tasks.jsonl --model m --output results.jsonl
```

## Testing notes

- `asyncio_mode = "auto"` in pyproject.toml -- no `@pytest.mark.asyncio` decorator needed.
- `testpaths = ["tests"]` -- do NOT pass `squishy` as a path arg to pytest.
- `tests/conftest.py` provides `FakeClient` (scripted async mock that pops `CompletionResult` objects) and `ctx` fixture (`ToolContext` with a tmp working dir).
- Smoke tests (`test_live_api.py`) are skipped unless explicitly selected with `-m smoke`.

## Architecture

### Agent loop

`agent.py` is a slim orchestrator. The per-turn loop is deliberately thin --
complete, dispatch, append, repeat -- and delegates to two submodules:

- **`agent_state.py`** -- `TaskResult`, `LoopState`, message helpers, problem-file extraction.
- **`agent_dispatch.py`** -- Tool dispatch, display rendering, outcome tracking.
- **`agent_safety.py`** -- `inject_nudge()` only: soft cap (`max_system_nudges`), hard cap (2x soft), per-turn cap, minimum turn gap.

**Feedback belongs in the tool result, not in an injected user message.** This is
the load-bearing rule of the loop. A tool result is causally paired with the call
that produced it, is what the model is trained to read, and cannot desynchronize
the transcript. An out-of-band `[system]` user turn breaks assistant/tool pairing
and forces `normalize_messages` to repair it -- which it can only do by deleting
the assistant turn.

A quality gate, a five-phase state machine, a plan protocol, goal-drift and
edit-failure detectors, turn budgets and problem re-anchoring all used to live
between the model and its tools. They were removed after measurement: on
SWE-rebench, harness-generated refusals (`run_command: refused`, `read_file:
refused`) were the largest single failure bucket, and the quality gate's
"skip this turn" path erased the model's own tool call from its history before
scolding it for making one. `tests/test_transcript_integrity.py` guards the
regression. Loop-breaking now lives in the tools -- `read_file`'s span cache and
`run_command`'s output-hash echo counter both answer inline.

Only three things still inject, and each one is a case a tool result cannot
reach: the empty-response retry, the post-compaction "your history was
rewritten" notice, and the bench-mode refusal to end a run with an unchanged
tree. The last one qualifies precisely because there is no tool call to attach
it to -- *not* calling a tool is the thing being answered. It is bench-only:
`yolo` is unattended too, but a question there may legitimately end in prose,
and nagging the user to edit something would be the harness inventing a goal.
Bounded by `max_turns` and nothing else; a count cap was tried in the
reference harness and spent its three nudges by turn 21 of 50.

The periodic no-edit-yet reminder used to be a fourth; it now rides on the
tool result instead (see "Edit pressure lives in the tool result").

### Context management (two layers)

1. **`trim_history`** (in `context.py`): keeps `system + first_user + last N messages`. Semantically anchored messages (failed commands, successful edits, search hits) survive trimming.
2. **`compact_messages`** (in `context.py`): at >70% context window usage, summarizes old messages via LLM call. First user message is always protected.

### Tool system

Tools are `Tool` dataclasses (`tools/base.py`) with `name`, `description`, `parameters` (JSON schema), and an async `run`. The registry in `tools/__init__.py` builds `REGISTRY = {t.name: t for t in ALL_TOOLS}`. Permission checks happen at dispatch time via `tool_restrictions.py`.

MCP tools are dynamically registered at startup as `mcp__servername__toolname`.

### Permission modes

Three modes (`edits`, `yolo`, `bench`) control what tools are allowed. `edits` is
the default and prompts for `run_command` approval; `yolo` skips prompts; `bench`
is for evaluation harnesses (drops the web tools, never prompts). Enforced in
`tool_restrictions.py` at dispatch time.

Orthogonal to mode, `tool_profile` narrows what the model *sees*: `standard`
(everything the mode allows, ~1.5k tokens/request), `minimal` (shell + file
primitives, ~870), `graph` (minimal plus `explore`), `shell` (`run_command`
alone, ~420). A profile shapes the
schema only -- it adds no refusal path, and `tool_aliases.py` maps foreign tool
vocabularies (bash, str_replace, file_path) onto the canonical names, so a model
trained on another harness is never penalized.

### Config duality: `api.py` vs `config.py`

`Squishy` dataclass in `api.py` has its own defaults that must stay aligned with `Config` in `config.py`. `Squishy._make_config()` copies fields to `Config` -- if you change a default in one, update the other.

### Indexing

`/init` or `--init` builds `.squishy/index.json` -- a tree of files with symbols extracted via `ast` (Python) or regex fallback (other languages). Docstrings become summaries for free; files without docstrings get optional LLM-generated summaries. Rebuilds are incremental by file hash. The `recall` tool does lexical scored lookup against this index.

### The code graph

The same `/init` also writes `.squishy/graph.json` (`squishy/graph/`): every
Python file, class, function and method, plus `contains`/`imports`/`calls`/
`inherits` edges, stored in both directions under one `threading.Lock`. The
index answers *where does this live*; the graph answers *who calls this* and
*what breaks if I change it*, which otherwise cost a crawl.

- `graph/query.py` holds pure functions -- no tool plumbing, so every answer
  is testable without a model. `tools/graph.py` wraps them as `explore`,
  `impact_of`, `repo_map`.
- `explore` is deliberately one strong tool: source + callers + callees +
  subclasses + impact radius in a single call, which is the whole first phase
  of a bug fix. It filters to exact matches when the query hits one, because
  substring noise stays in the transcript for the rest of the run.
- The graph tools leave the schema entirely when `has_graph` is false. Same
  rule as `recall`: never advertise a tool whose only possible answer is "run
  /init first".
- `--tools graph` is a narrow profile: shell, file primitives, `explore`.
  `impact_of` and `repo_map` are excluded on purpose -- a narrow profile
  exists to be narrow. `minimal` is deliberately *not* widened with `explore`,
  or the two profiles stop being comparable in the A/B this repo runs.
- A miss returns the nearest names from the graph (`difflib`), not "try a
  shorter substring" -- advice the model cannot act on without another round
  trip, which on qiskit-terra-5662 produced the same failing query three times.

### Edit pressure lives in the tool result

`tools/pressure.py` is applied centrally in `dispatch()`, so every tool
carries it and no tool has to remember to:

- `[budget]` -- silent for the first half of the turn budget, a reminder at
  half, an instruction at four fifths. Nothing else tells the model the clock
  is running; qiskit-terra-5662 wrote sixteen repro scripts and hit the cap
  having never touched a source file.
- `[probes]` -- eight commands with nothing edited. cfn-lint-3965 spent 37 of
  48 calls on `python -c` variations, no two identical, so no repeat detector
  could see it. Resets on every source edit, because re-running a reproduction
  *after* an edit is the correct move.

- `[repeat]` -- the identical call, byte for byte, a second time. Advice, not
  a refusal, and it never forgets: `read_file`'s span cache *does* refuse
  repeats but deliberately forgets after trimming, so a long run loses the
  signal exactly when it needs it. A counter that never forgets is only safe
  because it does not block. `run_command` is exempt -- re-running a test
  after an edit is correct, and `shell.py` has its own output-hash counter.
  Found on qiskit-terra-5662: the same non-matching `edit_file` issued twice
  in consecutive turns, with nothing to say the second could not work.

**A scratch write is not a source edit, through any path.** `write_file` knows
this from its `scratch` flag; the shell needs `agent_state.writes_only_scratch`,
because the shell is how a model writes a heredoc. Getting this wrong is
expensive and silent: `cat > /tmp/repro.py <<EOF` on turn 10 set `shell_writes`,
which set `source_edited`, which suppressed every notice for the next ninety
turns. Twelve container arms hit the 100-turn cap and not one ever called
`edit_file`. Fixing it took cfn-lint-3965 from 0/2 to 2/2 patched. The harness
*asks* for that scratch script -- rewarding the instructed action by disarming
its own safety net is the instruct-then-block pattern inverted.

The tool event carries a `pressure` list of the tags attached, and the bench
driver totals them. Without it, "warned fifty times and ignored" and "never
fired" are the same observation from outside.

**Notices go outside the JSON payload**, as plain text after it, and their
length is reserved out of the output cap rather than competing with it. As a
key inside `data` they were measurably ignored -- 43-50 `[budget]` and 74-85
`[probes]` notices across a 100-turn run that still never edited -- because
they sat at the end of a 4,000-character blob that `_short_json` could snip.

**A failed `edit_file` never dead-ends.** When the fuzzy matcher finds
nothing, `_symbol_hint` names the exact line spans of the symbols the
`old_str` mentioned, from the graph. The old fallback was "read the file
first", which qiskit-terra-5662 had already done five times before its single
edit attempt died on it at turn 73.

This replaced a `[system]` user message injected every sixth turn (and the
`max_turns_without_edit` knob that drove it). Same content, but paired with
the call that earned it -- the rule the whole loop is built on.

A /tmp scratch write does not count as the edit. The repro script is the right
move and never reaches the diff, so counting it would switch the pressure off
at exactly the moment it is needed.

### Scratch files

`write_file`'s own refusal message tells the model to put reproduction scripts
under /tmp. Until `_resolve_writable` existed, doing so returned "path outside
working directory" -- the harness refusing the action it had just demanded.
`read_file`, `write_file` and `edit_file` now accept absolute paths under the
scratch dir; everything else still has to stay in the repo. Both
`tempfile.gettempdir()` and a literal `/tmp` count, because they are the same
directory in every bench container and different ones on macOS.

`read_file` returns an outline instead of a body only when the body exceeds
the run's output cap and the graph covers the file. Measured, after shipping
the opposite: a turn costs 8.5-11k prompt tokens because the whole transcript
is resent, while an outline saves 0.4-3.7k, so an outline that forces a
follow-up read is a net loss at every size. Above the cap the body is snipped
anyway, so the follow-up was always going to happen.

### Bench harnesses

- **SWE-bench** (`bench/swebench.py`): clones repo, runs install commands, builds prompt with problem statement + failing tests + optional index recall, runs agent, captures `git diff` as patch.
- **Terminal-bench** (`bench/terminalbench.py`): creates temp workspace, seeds files, runs agent, scores by verify-shell exit code.
- **Runner** (`bench/runner.py`): generic async batch runner with `asyncio.Semaphore` concurrency and append-only JSONL output.

### Measuring a change

- **`scripts/rebench_container/run_bench.py`** is the real evaluation: one
  clean container per instance, arms share it, `--tools gold` skips the agent
  and grades the dataset's own patch as a self-test of the pipeline. Run gold
  before trusting an instance. It falls back to a cached image when the pull
  fails, so an unreachable registry no longer fails every instance.
- **`scripts/parity/ab_local.py`** needs no containers: injected bugs in a
  synthetic package, graded by a real pytest. Fast enough to iterate on, and
  the only thing available when Docker Hub is down. `scripts/parity/README.md`
  records the measurement that retired the OpenAI-Agents-SDK reference agent
  (`graphagent/`) — squishy resolved 9/9 against its 9/9 and 8/9, with fewer
  tool calls than its filesystem baseline and 5-15% more tokens. It also
  records where the container run disagreed: on real instances `minimal`
  patches 6/7 and `graph` 3/7, because given `explore` this model explores
  rather than edits. The graph's value on real repositories is currently
  unproven and possibly negative.
- `--seeds N` repeats every arm. At one seed a difference between two arms is
  indistinguishable from the same arm run twice.
- **Patch rate is harness health; resolve rate is capability.** Drive the
  first to 100% and report the second separately. On the three gold-validated
  instances with the shipped defaults the patch rate is 6/6; resolve rate on
  the same runs was 0/6. A good patch rate must never stand in for
  correctness.
- `--empty-patch-retries 0` measures the loop alone; the shipped default is 1.
  Quote the default, but debug with 0 — the retry hides loop bugs by brute
  force, which is how a run with zero `edit_file` calls still looked healthy.
- Read the `commands` field of each result record: the arguments are the
  trajectory. Forty `read_file` entries say nothing; which file and which
  range say all of it.
- Sizing tool output against a round trip: a turn costs 8.5-11k prompt tokens
  because the whole transcript is resent, so a summary that saves less than
  that and forces a follow-up call is a net loss.
  not collide with `tests/conftest.py`).
