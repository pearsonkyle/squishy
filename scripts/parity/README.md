# Parity with the OpenAI Agents SDK reference

`graphagent/` used to live in this repo: a second coding agent built on the
OpenAI Agents SDK, kept so squishy's loop could be measured against something
independent on the same instances, in the same containers, through the same
grader. It is gone. This is the measurement that justified removing it, and
`ab_local.py` is what produced it.

## The question

Squishy's loop carries things the SDK's `Runner` has no equivalent for — MCP
tools, three permission modes with approval prompts, session persistence,
context compaction, the rich display. All of that is only worth keeping if the
loop is at least as good at the actual job. So: same repository, same task,
same model, same knowledge graph — does squishy resolve as many bugs, and does
it spend more to do it?

## Setup

Three injected bugs in a small Python package with real call and inheritance
structure (`shapes/`), each with a pytest that fails before the fix and passes
after it. **That test is the only oracle.** "The model said it fixed it" is not
evidence and on these models it is frequently wrong: one arm below reported
success on a patch that left the test red.

Three tasks × three seeds per arm, `ornith-1.0-35b` on a local endpoint,
40-turn cap. Medians, because at one seed a difference between two arms is
indistinguishable from the same arm run twice.

## Result

| arm | resolved | tool calls | tokens | turns |
|---|---|---|---|---|
| squishy `graph` | **9/9** | 8 | 21,925 | 8 |
| squishy `minimal` | **9/9** | 7 | 20,087 | 7 |
| SDK + graph | 8/9 | 7 | 19,346 | 8 |
| SDK, filesystem only | 9/9 | 10 | 19,075 | 9 |

Squishy resolves everything, matching the better SDK arm and beating the
graph one. It uses fewer tool calls than the SDK's filesystem baseline and
ties its graph arm.

It still spends 5–15% more tokens per run. That gap is squishy's system
prompt: rules, task framing, project detection and the working-directory
header, ~400 tokens on every request that the SDK's bare `instructions` string
does not send. It buys the features the SDK arm does not have, and it is the
honest cost of them.

## What the measurement changed

The first pass showed squishy `graph` at 26.6k tokens — 39% worse than the SDK
baseline. The cause was not the loop. The bench system prompt said *"Don't
write reproduction scripts"* while the task prompt asked for one, `write_file`'s
own refusal message explained where to put it, and the SDK reference's framing
made reproducing the bug its second paragraph. Being told to do something and
forbidden from doing it in the same request is this codebase's recurring
failure mode. Removing the contradiction took the arm to 21.9k with no other
change. `tests/test_scratch_and_outline.py` holds it shut.

## Caveat, and what the container run said afterwards

These are injected bugs in a synthetic package. The container harness
(`scripts/rebench_container/run_bench.py`) is the real evaluation, and once
the registry came back it disagreed in one important way.

On cfn-lint-3965, qiskit-terra-5662 and cliquet-203 (all gold-validated 3/3),
100-turn cap, no empty-patch retry:

| | patched | brakes fired |
|---|---|---|
| before the scratch-write fix | 3/12 | never — 0 `edit_file` calls in 12 runs |
| after, `--empty-patch-retries 0` | 4/6 | every run |
| after, shipped default (`--empty-patch-retries 1`) | **6/6** | every run |

The 4/6 row deliberately disables the harness's own retry, to measure the loop
alone. The 6/6 row is the configuration squishy actually ships: one retry, the
default. Both matter — the first says the loop no longer suppresses its own
brakes, the second is the number to quote.

Resolve rate on that same 6/6 run was 0/6, and 0-2/12 across every sweep here.
Patch rate is harness health; resolve rate is capability, and on these three
instances with this model it is low and noisy. Do not let one stand in for
the other.

And the arm comparison inverts. Across every post-fix container run,
`minimal` patched **6/7** and `graph` patched **3/7** — and the graph arm's
own brake counts say why: it runs 74-85 probe notices and 43-50 budget notices
before editing, against `minimal`'s 24-36 and 3-6. Given `explore`, this model
explores. On a synthetic three-file package that costs nothing; on a real
repository it spends the budget.

So: the loop is fine, and the graph's value on real instances is currently
*negative*. Treat the table above as "squishy's loop is not leaving anything
on the table relative to the SDK" — not as evidence for the graph.
