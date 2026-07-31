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

Only three things still inject: the periodic no-edit-yet reminder, the
empty-response retry, and the post-compaction "your history was rewritten"
notice. Nothing else may.

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
primitives, ~870), `shell` (`run_command` alone, ~420). A profile shapes the
schema only -- it adds no refusal path, and `tool_aliases.py` maps foreign tool
vocabularies (bash, str_replace, file_path) onto the canonical names, so a model
trained on another harness is never penalized.

### Config duality: `api.py` vs `config.py`

`Squishy` dataclass in `api.py` has its own defaults that must stay aligned with `Config` in `config.py`. `Squishy._make_config()` copies fields to `Config` -- if you change a default in one, update the other.

### Indexing

`/init` or `--init` builds `.squishy/index.json` -- a tree of files with symbols extracted via `ast` (Python) or regex fallback (other languages). Docstrings become summaries for free; files without docstrings get optional LLM-generated summaries. Rebuilds are incremental by file hash. The `recall` tool does lexical scored lookup against this index.

### Bench harnesses

- **SWE-bench** (`bench/swebench.py`): clones repo, runs install commands, builds prompt with problem statement + failing tests + optional index recall, runs agent, captures `git diff` as patch.
- **Terminal-bench** (`bench/terminalbench.py`): creates temp workspace, seeds files, runs agent, scores by verify-shell exit code.
- **Runner** (`bench/runner.py`): generic async batch runner with `asyncio.Semaphore` concurrency and append-only JSONL output.
