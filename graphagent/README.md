# graphagent

A coding agent built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) that queries a **pre-built code knowledge graph** instead of crawling files — plus an **A/B harness** that measures whether the graph actually reduces tool calls and tokens.

It lives inside squishy as a second, independent harness: same repo, same benchmark containers, same grading code path, so the two can be compared directly against squishy's own agent loop.

Inspired by [codegraph](https://github.com/colbymchenry/codegraph) (pre-indexed graph, one strong `explore` tool, fewer tool calls), [PageIndex](https://github.com/VectifyAI/PageIndex), and [graphify](https://github.com/Graphify-Labs/graphify).

## How it works

```
                 ┌──────────────────────────────┐
  repo ──ast──▶  │  CodeGraph (thread-safe)     │
                 │  nodes: file/class/func/meth │
                 │  edges: contains/imports/    │
                 │         calls/inherits       │
                 └──────────┬───────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
  baseline agent                           graph agent
  list_dir / read_file / grep              explore / repo_map /
  (discovers structure the slow way)       symbol_source / impact_of
                                           (+ read_file fallback)
        └───────────────┬────────────────────────┘
                        ▼
        Q&A A/B harness            SWE-bench A/B arms
        (tool calls · tokens)      (patch rate · resolve rate)
```

1. **Preprocessing** — `build_graph(repo)` parses every `.py` file with `ast` (extraction), then resolves imports → files and call/base names → symbols (resolution). One-time cost, always reported separately from run cost.
2. **Graph agent** — its flagship `explore(query)` returns, in **one call**: the matching symbols' line-numbered source, callers/callees, subclasses, and a depth-2 impact radius. Instructions steer it to trust the graph and not re-verify with reads.
3. **Baseline agent** — identical model, identical framing, only `list_dir` / `read_file` / `grep`. The toolset is the only variable.
4. **Metrics** — a thread-safe `RunHooks` recorder counts tool calls; tokens and request counts come from the SDK's per-run `Usage`. `ToolTrace` additionally records each call's *arguments*, which is what makes a failed run readable.

## Modules

| path | what it holds |
| --- | --- |
| `graph/models.py` | `Node`/`Edge`, `NodeKind`/`EdgeKind`, the stable id scheme (`pkg/mod.py::Class.method`) |
| `graph/store.py` | `CodeGraph`: forward+reverse adjacency, `search`, `impact`, `repo_map`, JSON persistence, one lock over all shared state |
| `graph/builder.py` | two-pass `ast` indexer; a `SyntaxError` in one file never aborts the index |
| `agentkit/tools.py` | pure read tools (no SDK imports, unit-testable without an LLM) |
| `agentkit/edit.py` | pure write tools: `edit_file`, `write_file`, `run_command` |
| `agentkit/llm.py` | `resolve_model` — hosted OpenAI, or any OpenAI-compatible local endpoint |
| `agentkit/factory.py` | the two read-only Q&A arms |
| `agentkit/swe.py` | `build_swe_agent` — the bug-fixing arm, with or without the graph |
| `agentkit/metrics.py` | `ToolCallRecorder` (SDK hook), `ToolTrace`, `RunMetrics` |
| `bench/harness.py` | `run_once` / `compare` / `run_ab` — median-of-N per arm |

## Usage

```bash
graphagent index /path/to/repo                 # build + cache the graph (.graphagent.json)
graphagent map /path/to/repo                   # compact file/symbol overview
graphagent explore /path/to/repo UserService   # one-shot: source + callers + impact

graphagent bench /path/to/repo "How does a request reach the database?" \
  --model ornith-1.0-35b \
  --base-url http://localhost:1234/v1 \
  --runs 3 --output report.json
```

`--model` is **required** whenever `--base-url` points at a local endpoint. LM Studio JIT-loads whatever id it is asked for, so a default here would silently pull a stray multi-gigabyte model off disk. With no base URL, the hosted OpenAI default applies and `OPENAI_API_KEY` is needed.

`bench` prints the per-arm medians and deltas; the full JSON report keeps every individual run with its per-tool breakdown, so you can see *how* each arm spent its budget.

## SWE-bench arms

`scripts/rebench_container/run_bench.py` gained two arms that run this agent instead of squishy's loop, in the same container, graded by the same code:

```bash
python scripts/rebench_container/run_bench.py \
  --instances scripts/rebench_container/instances_small.jsonl \
  --tools sdk,graph --model ornith-1.0-35b \
  --max-turns 50 --empty-patch-retries 3
```

* `sdk` — filesystem discovery plus edit tools (the reference-agent shape).
* `graph` — the same agent with the knowledge graph in front of the crawl.

`--seeds N` repeats every arm; at one seed a patch-rate difference between two profiles is indistinguishable from the same profile run twice.

`--empty-patch-retries` is load-bearing here, not an optimization. `Runner.run` returns as soon as the model emits a message with no tool call — **including an empty one** — and the SDK reports that as a completed run. The driver therefore refuses to accept an unchanged working tree as an ending: it resumes the same transcript with a nudge appended, sharing one turn budget across the continuations. `tests/kg/test_driver_graph.py` guards it.

### Brakes, and where they live

Feedback goes in the tool result, never in an injected user turn — a tool result is causally paired with the call that produced it and cannot desynchronize the transcript. Three brakes in `swe.py::_log`, each found by reading one trace:

| marker | fires when | the run it came from |
| --- | --- | --- |
| `[repeat]` | an identical non-command call | qiskit ran one dead-end `explore` three times, then quit |
| `[probes]` | 8 commands with no source edit | cfn-lint spent 37 of 48 calls on `python -c` variations |
| `[budget]` | half the turn budget gone, nothing edited | qiskit wrote 16 repro scripts and hit the cap having edited nothing |

The budget figure is the *binding* limit of `--max-turns` and `--task-timeout`, projected from the observed per-turn cost: they are set independently and disagree on slow images, and the agent was being told it had nine turns left as the process was killed.

### Sizing a tool result against a round trip

A turn costs 8.5–11k prompt tokens, because the whole transcript is resent every time. An outline of a Python file saves 0.4k tokens on a short file and 3.7k on a long one. So a summary that forces a follow-up call is a **net loss at every file size** — which is what shipping the opposite showed: cfn-lint's graph arm went from 20/29/42 tool calls to 43/48/29 and lost both its resolves. `read_file` therefore returns an outline only for files above the 400-line read cap, where the file truncates anyway and the extra call was already unavoidable.

## Development

```bash
pytest tests/kg -q          # no API key, no network
ruff check graphagent tests/kg
```

Design rules: the graph store guards all shared state with a `threading.Lock` (the SDK can run tools concurrently); tool logic is pure typed functions that the factories wrap with `@function_tool`; every filesystem path is validated against the repo root, with one deliberate exception — writes under the system temp dir, because the agent is told to put reproduction scripts there and refusing the path we asked for is the exact failure mode this harness exists to avoid.

## Known limitations

- Call resolution is name-based (same-file → imported-name → globally-unique). Dynamic dispatch, decorators that rebind, and duck typing produce no edge. Ambiguous names resolve to nothing: a wrong edge is worse than a missing one.
- Python only. `builder.py` isolates extraction per file, so a tree-sitter extractor emitting the same `Node`/`Edge` model would slot in.
- JSON persistence is fine to a few thousand nodes; past that, SQLite + FTS.
- Token deltas are noisy on small repos and small tasks. Benchmark on a few hundred files with `--runs 3+` and compare medians.
