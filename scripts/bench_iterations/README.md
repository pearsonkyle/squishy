# Bench Iteration Tracking

Each iteration of the agent gets its own git branch (`bench/vN`), a tag at
the end (`bench/vN-final`), and a results file in this directory
(`vN_results.md`).

## Workflow per iteration

```bash
cd squishy/

# 1. Branch from previous iteration
git checkout bench/vN
git checkout -b bench/v(N+1)

# 2. Make changes, run tests
./docker/run.sh python -m pytest tests/ -q --ignore=tests/test_live_api.py

# 3. Run the bench (5-instance smoke first, then larger)
#    v1 baseline: seed=2026, py3.12 filter, sample=5. Reuse the v1 instance
#    set across subsequent iterations so v(N+1) is comparable to vN:
INSTANCES=scripts/rebench_eval/results/gemma-4-31b-v1/instances.jsonl
TAG=bench-v(N+1)-5

squishy-bench swe \
    --instances "$INSTANCES" \
    --model gemma-4-31b \
    --base-url http://10.132.212.37:10201/v1 \
    --output "scripts/rebench_eval/results/$TAG/predictions.jsonl" \
    --concurrency 5

# 4. Evaluate patches
./scripts/rebench_eval/run_all.sh \
    --predictions "scripts/rebench_eval/results/$TAG/predictions.jsonl" \
    --instances "$INSTANCES" \
    --tag "$TAG"

# 5. Commit results, document, tag
cd squishy/
git add scripts/bench_iterations/v(N+1)_results.md
git commit -m "v(N+1): <one-line summary>"
git tag bench/v(N+1)-final -m "<results summary>"

# 6. If results regressed, roll back:
git checkout bench/vN     # or git checkout pre-v0-archive for v27.x state
```

## Archive points

- `pre-v0-archive` — tag on main HEAD (commit 22d9ace) before v0 reset.
  All v27.x work-in-progress lives here. `git checkout pre-v0-archive`
  to inspect.

## Iteration index

| Iteration | Branch       | Tag (final)        | Headline change                                  | Resolved | Patched |
|-----------|--------------|--------------------|--------------------------------------------------|----------|---------|
| v0        | `bench/v0`   | _unmeasured_       | F2P-multi-file callout + sibling-class recall    | n/a      | n/a     |
| v1        | `bench/v0`   | `bench/v1-final`   | First measured baseline (carries v0 changes)     | 0/5      | 5/5     |
| v2        | `bench/v2`   | `bench/v2-final`   | Auto-pytest finish gate (synth pytest on finish-without-tests) | 0/5      | 5/5     |
| v3        | `bench/v3`   | `bench/v3-final`   | Diagnostic: cross-model (gemma vs deepseek-v4-flash) + cross-seed (555 vs 2026); no code change | 0/5 (both) | 5/5 (both) |
| v4        | `bench/v4`   | `bench/v4-final`   | Switch from rebench V1 → SWE-rebench-V2 (per-instance docker eval images, 5,675 strict-compat py3.10 pool); deepseek-v4-flash | **1/5** | 5/5 |
| v5        | `bench/v5`   | _untagged_ (v5b blocked by endpoint outage) | Pre-finish F2P partial-pass gate: bump max_intercepts 1→2, enumerate failing F2P test IDs in nudge | 1/5 (no change vs v4; gate fired 0×) | 5/5 |
| v6a       | `bench/v5`   | _no tag (no code change)_ | Cross-model A/B: same v5 code on **gemma-4-31b** instead of deepseek-v4-flash, same v4/v5 instance set | **2/5** ✓ vacanza, ✓ workalendar (gate fired 0×; capture-bound) | 5/5 |
| v6b       | `bench/v6b`  | `bench/v6b-final`  | F2P plumbing trio: notes-first F2P populator (uncaps from 5→full), `last_f2p_collection_error` flag for empty-failures pytest crashes, new collection-error gate branch | 2/5 (no regression; new branches plumbed+tested but didn't fire — agent skipped pytest entirely on rdt; generic v5 branch fired 1×) | 5/5 |
| v6c       | `bench/v6c`  | `bench/v6c-final`  | Post-edit pytest nudge (one-shot, end-of-turn, bench/yolo): surfaces exact `pytest <id1> <id2> ...` invocation when first edit lands. + sample 5→10. | 2/10 (20%); nudge fired 2× and **2/2 → pytest before finish** (cfn-lint-2927 nudge→pytest→success); rdt-670 missed nudge due to `min_gap=3` colliding with prior BLOCKED nudge — carried to v6d | 10/10 |
| v6d       | `bench/v6d`  | _no tag (regression)_ | Three coupled fixes: REINSTALL override in eval parser, `old_str-not-found` escape-hatch nudge (count≥2, action-specific), post-edit nudge flag-ordering + `min_gap` 3→1. + display/UX hardening (Rich-markup escape, async cancel, slash extra-arg). | 0/10 — **dataset-draw artifact**: 9/10 instances non-Python; `--compat-filter` does not imply `--language python`. The 1 Python instance (jsonpickle-469) was a real test-failure miss. v6d code changes correct but not exercised. → v6e fixes the filter | 8/10 (3 TEST_PATCH + 2 empty-patch unevaluable) |

> **v1 baseline note:** v1 instances are seed=2026, py3.12-filtered, sample=5
> against gemma-4-31b @ 10.132.212.37:10201. v0 was never measured (its
> reference instance set was deleted during the 2026-05-11 results reset);
> v1 is the new comparison anchor for v2+.

## Per-iteration result file format

Use `vN_results.md` template:

```markdown
# vN — <one-line summary>

**Branch:** `bench/vN` (parent: `bench/v(N-1)`)
**Date:** YYYY-MM-DD
**Model:** gemma-4-31b @ http://10.132.212.37:10201/v1
**Instances:** scripts/v27_5.jsonl (5 instances)

## Hypothesis

<what you're trying to fix and why>

## Changes

- file:line — what changed
- file:line — what changed

## Results

| Metric                 | Previous | This run |
|------------------------|----------|----------|
| Resolved (5-instance)  | x/5      | y/5      |
| F2P partial (10)       | x/10     | y/10     |
| Avg turns              | x        | y        |

## Per-instance notes

- `instance_id`: <observation from transcript>

## Decision

- [ ] Promote to next baseline (tag `bench/vN-final`)
- [ ] Roll back to `bench/v(N-1)`, try different approach
```
