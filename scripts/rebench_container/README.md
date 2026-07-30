# In-container SWE-rebench eval

Runs squishy **inside each instance's own Docker image**, the way it would
actually be used: in the repo checkout, with the project's real toolchain on
`PATH`. `run_command` is a plain local exec rather than a `docker run` per
call, so the agent can afford to run the test suite.

Grading runs the tests. Nothing here trusts a non-empty diff.

## Quick start

```bash
# one arm
python scripts/rebench_container/run_bench.py \
    --instances scripts/index_ablation/instances_sample.jsonl \
    --tools minimal --out results.jsonl

# compare tool profiles, and with/without the repo index
python scripts/rebench_container/run_bench.py \
    --instances scripts/index_ablation/instances_sample.jsonl \
    --tools minimal,standard --index both --resume --out results.jsonl

python scripts/rebench_container/summarize.py results.jsonl --failures
```

The LLM runs on the host; containers reach it at `host.docker.internal`
(override with `--base-url`).

## How grading works

Two real test runs per instance:

| run    | tree                                  | expectation |
|--------|---------------------------------------|-------------|
| `pre`  | base + gold test patch                | must FAIL   |
| `post` | base + gold test patch + model patch  | PASS ⇒ resolved |

The `pre` run is not ceremony. If the tests already pass without a fix, the
instance can't discriminate and a "resolved" from `post` would be meaningless —
those are reported as `eval_error` and excluded from rates, never counted as
successes. Patch-produced is recorded too, but it is not the headline: in
practice a full patch rate coexists with a much lower resolved rate.

One adaptation is applied automatically. Some images configure pytest with
xdist; if every worker crashes on an unrelated import error, pytest reports
"no tests ran" and exits 5 — indistinguishable from a genuinely empty
selection, and easy to misread as a failing baseline. When `pre` looks empty
the runner retries serially (`-n0`) and, if that collects tests, uses the
serial command for both runs so they stay comparable.

## How the agent is installed

Into an isolated venv (`/opt/squishy-venv`) with its own Python 3.11, fetched
by `uv`. Deliberately *not* into the project's interpreter: squishy's
dependency pins would otherwise land in the environment the tests run in and
could silently change grading results. The images ship Python 3.10 in some
cases, below squishy's floor, so a separate interpreter is needed regardless.

`uv` and the squishy source are cached on the host and copied into each
container, so only the first instance pays for the download.

## Metrics come from the event stream

`driver.py` accumulates turns, tokens, tool counts and failure causes from the
agent's `on_event` callbacks and rewrites its result file after every event.
Reading them off the returned `TaskResult` instead would lose everything on
the paths weak models hit most — the turn cap, the wall-clock timeout, and
upstream API errors. If the driver is killed mid-run the partial metrics on
disk are still correct.

`exit_status` (`completed` / `max_turns` / `wall_timeout` / `error:*`) is
recorded separately from `patched` and `resolved`, so "ran out of turns" is
never confused with "decided it was finished".

## Arms

Arms share a container per instance: the multi-GB image is pulled once and
every arm starts from byte-identical state, which is the only fair way to
compare. Between arms the repo is reset with `git checkout` + `git clean -fd`
(no `-x` — build outputs and vendored deps are baked into the image and
gitignored; removing them would break the test run).

- `--tools minimal` — shell + file primitives, no phase machine.
- `--tools standard` — every tool the mode allows, phase-gated.
- `--index on` — build `.squishy/index.json` first so `recall` is available.

## Caveats

- The dataset images are `linux/amd64`; on an arm64 host everything runs under
  emulation and is slow. Budget several minutes per arm per instance.
- Model nondeterminism is real. Treat a single 15-instance sweep as
  directional; re-run or widen the sample before drawing firm conclusions.
- `--resume` skips `(instance, tools, index)` combinations already in the
  output file, so an interrupted sweep can be continued.
