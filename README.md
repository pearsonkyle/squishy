# squishy

Minimal local-LLM coding agent. OpenAI-compatible.

A small async Python CLI + library that turns any OpenAI-compatible endpoint (LM Studio, vLLM, llama.cpp with `--api`) into a tool-calling coding assistant. 13 tools, one agent loop, no Docker stack, no Go proxy. Ships with SWE-bench and Terminal-bench harnesses.

## Install

```bash
cd squishy
pip install -e .
```

Python 3.11+.

## Run the CLI

Start any OpenAI-compatible server first:

```bash
# LM Studio
lms server start

# vLLM
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 1234

# llama.cpp
./llama-server -m model.gguf --port 1234 --host 0.0.0.0
```

Then:

```bash
squishy --base-url http://localhost:1234/v1 --model local-model
```

Or via env vars:

```bash
export SQUISHY_BASE_URL=http://localhost:1234/v1
export SQUISHY_MODEL=local-model
squishy
```

Usage modes:

```bash
squishy                          # interactive REPL
squishy -m "create hello.py"     # one-shot (exit after)
echo "read pyproject.toml" | squishy   # pipe mode
```

Useful flags:

```
--timeout SECONDS          overall per-turn wall-clock timeout
--request-timeout SECONDS  per-HTTP-request timeout (default 120)
--max-retries N            transient-failure retry budget (default 4)
--sandbox                  wrap run_command in Docker
--init                     build .squishy/index.json before the REPL
--no-summaries             skip LLM summaries when indexing (docstrings only)
--index-concurrency N      parallel summary calls (default 4)
```

## Repo index (`/init`)

Type `/init` in the REPL (or pass `--init`) to build a hierarchical JSON
index of the repo under `.squishy/index.json`. Inspired by PageIndex: no
embeddings, no vector DB — just a tree that mirrors directories and files,
plus top-level symbols extracted with `ast` (Python) or a regex fallback
(JS/TS/Go/Rust/C/Java/Ruby/...).

- Module docstrings and header comments become summaries for free.
- Files without any docstring get one LLM-generated sentence each, capped
  at `--index-concurrency` parallel calls and `max_tokens_per_index`
  (default 100k total tokens).
- Rebuilds are incremental: unchanged files reuse their prior summary by
  file hash. Deleted files drop out; new files are summarized.
- Pass `--no-summaries` to skip the LLM entirely (offline-friendly).
Once an index exists, the new `recall(query=...)` tool surfaces ranked
matches (path + summary + line range) so the model can pick the right
module without walking the tree. A compact index header is injected into
the system prompt, and squishy nudges you when the index is stale.

## Permission modes

Four modes, cycle with **Shift+Tab** during a session:

| Mode | Reads | Writes / Edits | Shell |
|------|-------|----------------|-------|
| `plan` *(default)* | auto | refused | read-only allowlist |
| `edits` | auto | auto | prompt |
| `yolo` | auto | auto | auto |
| `bench` | auto | auto | auto |

`bench` mode is used by the SWE-bench harness. It excludes plan tools to
reduce schema size, enables quality monitoring and stuck detection, and
uses an action-biased prompt that pushes the agent to attempt fixes early.

Set the starting mode with `--plan`, `--edits`, or `--yolo`.

## Python API

`Squishy` is the programmatic surface — use it from scripts, tests, or benchmark harnesses:

```python
import asyncio
from squishy.api import Squishy

async def main():
    async with Squishy(
        model="local-model",
        base_url="http://localhost:1234/v1",
        permission_mode="yolo",
        request_timeout=120.0,
        max_retries=4,
    ) as sq:
        result = await sq.run(
            "Fix the TypeError in app.py",
            working_dir="/tmp/repo",
            timeout=300,                 # overall task timeout
            on_text=lambda chunk: print(chunk, end=""),
        )
        print(result.success, result.files_edited, result.turns_used)

asyncio.run(main())
```

`TaskResult` fields: `success`, `final_text`, `turns_used`, `tokens_used`, `files_created`, `files_edited`, `commands_run`, `elapsed_s`, `error`, `messages`, `plan_state`.

### Error model

Typed exceptions from `squishy.errors`:

- `LLMError` — LLM call failed after retries (connection / timeout / 5xx / rate-limit exhaustion).
- `AgentTimeout` — overall task timeout exceeded.
- `AgentCancelled` — caller cancelled the task.
- `BenchError` — benchmark harness setup/execution error.

## Benchmarks

`squishy-bench` runs SWE-bench instances or Terminal-bench tasks concurrently with a live LLM.

### SWE-bench

```bash
squishy-bench swe \
  --instances SWE-bench_Lite.jsonl \
  --model local-model \
  --model-name squishy-local \
  --output predictions.jsonl \
  --workspace-root ./_swe_workspaces \
  --concurrency 4 \
  --task-timeout 900
```

Emits a predictions JSONL compatible with the upstream SWE-bench harness:

```bash
python -m swebench.harness.run_evaluation \
  --predictions_path predictions.jsonl \
  --dataset_name princeton-nlp/SWE-bench_Lite
```

The bench prompt uses an **action-biased workflow**: the agent is instructed
to attempt an `edit_file` fix within its first 5 tool calls rather than
spending excessive turns on exploration. Stuck detection nudges the agent
after 3 turns with no file mutations.

### Terminal-bench

Tasks adapted from [Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench)
(file ops, data processing, regex, algorithmic puzzles, etc.).

Task schema (JSONL or JSON array):

```json
{
  "id": "task-001",
  "description": "Create fib.py that prints fib(10).",
  "setup": ["pip install -q pytest"],
  "verify": "python fib.py | grep -q '^55$'",
  "files": {"README.md": "..."},
  "timeout": 300
}
```

A curated mini set (`terminal_bench_mini.jsonl`, 8 tasks) is included for quick
validation:

```bash
squishy-bench term \
  --tasks terminal_bench_mini.jsonl \
  --model qwen-3-coder \
  --base-url http://vadc-dla17:10200/v1 \
  --api-key sk-no-key-required \
  --output results.jsonl \
  --concurrency 1
```

Each task runs in a fresh temp workspace, the verify shell is scored pass/fail
by exit code, and results include the transcript, plan state, setup/verify
command output, and a workspace snapshot.

### Custom harness

`squishy.bench.runner.run_batch` gives you an async batch runner with bounded concurrency, per-task timeout, and a streaming JSONL writer — use it to build your own harness over the `Squishy` facade.

## Tools

| Tool | Purpose |
|------|---------|
| `read_file` | Read with offset/limit |
| `write_file` | Full-file write (rejects existing files >100 lines — use `edit_file`) |
| `edit_file` | Exact-string replacement with unified-diff preview |
| `list_directory` | Gitignore-aware listing |
| `search_files` | Regex search via ripgrep if available, Python `re` otherwise |
| `glob_files` | Recursive file finding by glob pattern (e.g. `**/*.py`) |
| `recall` | Ranked lookup in the `/init` index — path, lines, summary |
| `run_command` | Shell execution, sandboxed in Docker when `--sandbox` and Docker are available |
| `plan_task` | Create a structured task plan (plan mode) |
| `update_plan` | Update plan step status (plan/edits mode) |
| `get_plan` | Retrieve the current plan |
| `save_note` | Persist key-value notes across conversation turns |
| `show_diff` | Display current workspace changes as a unified diff |

## Agent quality patterns

Safety and quality patterns ported from ATLAS and little-coder:

- **Conversation trim** to `system + first_user + last N` messages (configurable via `max_history_messages`).
- **LLM-based context compaction**: when token usage exceeds 70% of the context window, old messages are summarized via a single LLM call, preserving semantically anchored messages (failed tests, successful edits, search results).
- **Quality monitoring** (`quality.py`): heuristic checks run after every assistant turn in bench/yolo modes:
  - Hallucinated tool names (tool not in registry)
  - Repeated identical tool calls (same name + args as previous turn)
  - Malformed tool arguments (JSON parse failures)
  - Excessive same-file re-reads (3+ times in last 8 turns)
  - On detection, a targeted correction message is injected (up to 2 retries).
  - If the agent enters a verification loop after making edits, it is force-finished with success.
- **Stuck detection**: after 3 consecutive turns with no file mutations (bench mode), an escalating nudge pushes the agent to attempt an edit immediately.
- **Error loop breaker**: 8 consecutive tool failures → stop.
- **Semantic anchoring**: failed commands, successful edits, and search results are tagged to survive history trimming.
- **Write-file guard**: rejects `write_file` on existing files >100 lines, forces the model to use `edit_file`.
- **Tool output truncation**: head+tail strategy (60% head + 30% tail) preserves both the start and end of large outputs.
- **Docker sandbox** for `run_command` when Docker is on PATH (enable with `--sandbox`).

## Async + retries

- Every module is fully async (`asyncio`). The REPL uses `prompt_toolkit.PromptSession.prompt_async`; tools call `asyncio.create_subprocess_exec`; the client is `openai.AsyncOpenAI`.
- Transient LLM errors (timeout, connection reset, 429, 5xx) are retried with exponential backoff via `tenacity`. The SDK's built-in retries are disabled to avoid double-counting.
- Task-level `asyncio.timeout` wraps each user turn; `CancelledError` is re-raised as `AgentCancelled` for clean caller semantics.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

280 tests covering tools, agent loop, quality monitoring, context compaction, permissions, index, benchmarks, and the API facade. Smoke tests that hit a real endpoint are marked `@pytest.mark.smoke` and skipped by default.

## Layout

```
squishy/
  api.py          # Squishy facade (programmatic API)
  agent.py        # async agent loop + safety patterns
  client.py       # AsyncOpenAI wrapper + retries (only network module)
  cli.py          # REPL + Shift+Tab mode cycling
  config.py       # env vars, permission modes, tuning knobs
  context.py      # project detection, system prompt, history trim, LLM compaction
  display.py      # rich terminal output
  errors.py       # typed exception hierarchy
  quality.py      # response quality monitoring (degenerate loop detection)
  plan_state.py   # plan lifecycle management
  tool_restrictions.py  # per-mode tool allowlists + shell command validation
  bench/
    runner.py         # async batch runner + JSONL writer
    swebench.py       # SWE-bench harness (action-biased prompt)
    terminalbench.py  # Terminal-bench harness
    cli.py            # squishy-bench entry point
  index/
    model.py          # Node / Index / IndexMeta dataclasses
    walker.py         # repo walk + SKIP_DIRS + .gitignore + cap
    ast_py.py         # Python symbol extraction (stdlib ast)
    ast_generic.py    # regex fallback for JS/TS/Go/Rust/C/...
    build.py          # assembles tree; incremental hash-based reuse
    summarize.py      # optional LLM summaries with concurrency + budget
    staleness.py      # mtime-based freshness check
    store.py          # save/load .squishy/index.json
  tools/
    base.py       # Tool, ToolResult, ToolContext dataclasses
    fs.py         # 6 filesystem tools (read, write, edit, list, search, glob)
    recall.py     # lexical index lookup
    shell.py      # run_command (async, optional Docker sandbox)
    plan.py       # plan_task, update_plan, get_plan
    scratchpad.py # save_note, show_diff
    __init__.py   # registry + dispatch + permission gate
tests/
  test_tools_fs.py
  test_tools_shell.py
  test_parsing.py
  test_context.py
  test_agent_loop.py       # scripted FakeClient
  test_permissions.py
  test_api.py              # Squishy facade
  test_bench.py            # runner + terminal-bench
  test_retry.py            # client retry / error translation
  test_quality.py          # quality monitoring checks
  test_cli.py
  test_plan_tools.py
  test_plan_mode_e2e.py
  test_change_tracking.py
  test_edge_cases.py
  test_token_counting.py
  test_tool_context_state.py
  test_context_loading.py
  test_file_browser.py
  test_index_walker.py
  test_index_ast_py.py
  test_index_build.py
  test_index_store.py
  test_index_summarize.py
  test_index_staleness.py
  test_index_optimize.py
  test_recall_tool.py
  test_live_api.py         # @pytest.mark.smoke (skipped by default)
```

## License

MIT License. See [LICENSE](LICENSE).
