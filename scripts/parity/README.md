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

## Caveat

These are injected bugs in a synthetic package, not SWE-rebench instances. The
container harness (`scripts/rebench_container/run_bench.py`) is still the real
evaluation; it was unavailable when this ran because Docker Hub was
unreachable and the local image cache was being reclaimed. Read this as
"squishy's loop is not leaving anything on the table relative to the SDK",
not as a capability score.
