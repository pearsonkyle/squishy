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
pytest tests/test_quality.py -q
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

`agent.py` is a slim orchestrator (~400 lines). The per-turn loop delegates to four submodules:

- **`agent_state.py`** -- `TaskResult`, `LoopState`, message helpers, problem-file extraction.
- **`agent_safety.py`** -- Loop/stuck detection, quality gates, nudge injection. All nudges flow through `inject_nudge()` which enforces soft cap (`max_system_nudges`), hard cap (2x soft), and minimum turn gap.
- **`agent_dispatch.py`** -- Tool dispatch, plan approval flow, evidence recording.
- **`agent_phases.py`** -- Phase tracking (explore/fix/verify), turn budget injection, problem re-anchoring.

### Context management (two layers)

1. **`trim_history`** (in `context.py`): keeps `system + first_user + last N messages`. Semantically anchored messages (failed commands, successful edits, search hits) survive trimming.
2. **`compact_messages`** (in `context.py`): at >70% context window usage, summarizes old messages via LLM call. First user message is always protected.

### Tool system

Tools are `Tool` dataclasses (`tools/base.py`) with `name`, `description`, `parameters` (JSON schema), and an async `run`. The registry in `tools/__init__.py` builds `REGISTRY = {t.name: t for t in ALL_TOOLS}`. Permission checks happen at dispatch time via `tool_restrictions.py`.

MCP tools are dynamically registered at startup as `mcp__servername__toolname`.

### Permission modes

Four modes (`plan`, `edits`, `yolo`, `bench`) control what tools are allowed per turn. `plan` is the default. `bench` is for evaluation harnesses only (excludes plan tools). Mode restrictions enforced in `tool_restrictions.py`.

### Config duality: `api.py` vs `config.py`

`Squishy` dataclass in `api.py` has its own defaults that must stay aligned with `Config` in `config.py`. `Squishy._make_config()` copies fields to `Config` -- if you change a default in one, update the other.

### Indexing

`/init` or `--init` builds `.squishy/index.json` -- a tree of files with symbols extracted via `ast` (Python) or regex fallback (other languages). Docstrings become summaries for free; files without docstrings get optional LLM-generated summaries. Rebuilds are incremental by file hash. The `recall` tool does lexical scored lookup against this index.

### Bench harnesses

- **SWE-bench** (`bench/swebench.py`): clones repo, runs install commands, builds prompt with problem statement + failing tests + optional index recall, runs agent, captures `git diff` as patch.
- **Terminal-bench** (`bench/terminalbench.py`): creates temp workspace, seeds files, runs agent, scores by verify-shell exit code.
- **Runner** (`bench/runner.py`): generic async batch runner with `asyncio.Semaphore` concurrency and append-only JSONL output.
