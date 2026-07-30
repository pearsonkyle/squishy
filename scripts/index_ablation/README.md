# Index ablation: does `/init` + `recall` reduce the work?

Measures what the repo index actually buys on real issues from
[nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).

Each instance is run through the agent **twice** — once with no index (the
agent has to grep and read until it finds the right file) and once with
`.squishy/index.json` built (so `recall` can point it straight there). The two
numbers we care about:

1. **Patch rate** — did the agent produce a diff at all? A patch that fails the
   tests is still worth far more than no patch, so this is the primary metric.
2. **Navigation effort** — turns and tool calls used. If the index is working,
   the indexed run should need fewer.

Test-pass rate is deliberately *not* the headline: the local models this
harness targets often can't fully solve these issues, and that's expected.

## Running it

```bash
# 1. pick instances (3 per language, distinct repos)
python scripts/index_ablation/select_instances.py \
    --languages python,ts,go,rust,java --per-language 3 \
    --out scripts/index_ablation/instances.jsonl

# 2. run the ablation (pulls each instance's prebuilt image)
python scripts/index_ablation/run_ablation.py \
    --instances scripts/index_ablation/instances.jsonl \
    --model ornith-1.0-35b --base-url http://localhost:1234/v1 \
    --out ablation_results.jsonl

# 3. summarize
python scripts/index_ablation/summarize.py ablation_results.jsonl

# 4. clean up pulled images when you're done
python scripts/index_ablation/run_ablation.py --cleanup-images \
    --instances scripts/index_ablation/instances.jsonl
```

`instances_sample.jsonl` is a committed 15-instance sample (3 each of python,
ts, go, rust, java) so the harness can be run without re-querying the dataset.

## How the agent is run

The repo is copied out of the instance's own image into a host workspace, and
the agent runs against that copy. `run_command` is executed **inside that same
image** via squishy's Docker sandbox (`use_sandbox`), so the project's
toolchain and dependencies are real — otherwise every test run fails and the
agent burns its whole budget on commands that can't succeed.

Note the dataset images are `linux/amd64`; on an arm64 host they run under
emulation, which is slow. Use `--no-sandbox` for a much faster run when you
only care about patch rate and navigation effort (the agent then can't run the
project's tests).

## Caveats

- Model nondeterminism is real. Treat a single 15-instance run as directional,
  not conclusive; re-run or raise `--per-language` for a firmer signal.
- `run_ablation.py` reports the patch it generated; use
  `scripts/rebench_eval/run_eval.py` if you want to score those patches against
  the gold tests.
