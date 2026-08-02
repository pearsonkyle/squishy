#!/usr/bin/env python3
"""What did the model actually call, and what did each tool cost to advertise?

Tool sets tend to grow by intuition. This weighs each tool's share of real
calls against the schema tokens it costs on *every* request, so trimming is an
evidence-based decision rather than a preference.

    python scripts/rebench_container/tool_usage.py results.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from squishy.tools import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--arm", default=None,
                    help="Only count runs from this tool profile (e.g. standard)")
    args = ap.parse_args()

    calls: Counter[str] = Counter()
    fails: Counter[str] = Counter()
    runs = 0
    for path in args.results:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if args.arm and r.get("tools") != args.arm:
                continue
            runs += 1
            calls.update(r.get("tool_counts") or {})
            for k, v in (r.get("failure_reasons") or {}).items():
                fails[k.split(":", 1)[0]] += v

    total = sum(calls.values()) or 1
    print(f"{runs} runs, {total} tool calls\n")
    hdr = f"{'tool':16} {'calls':>7} {'share':>7} {'fails':>7} {'schema_tok':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, n in calls.most_common():
        tool = REGISTRY.get(name)
        cost = len(json.dumps(tool.openai_schema())) // 4 if tool else 0
        print(f"{name:16} {n:7} {n / total:6.1%} {fails.get(name, 0):7} {cost:11}")

    never = [t.name for t in REGISTRY.values() if t.name not in calls]
    if never:
        print(f"\nnever called: {', '.join(sorted(never))}")

    # The case for trimming: tokens paid on every request for tools that
    # barely earn a call.
    rare = [(n, c) for n, c in calls.items() if c / total < 0.01]
    if rare:
        cost = sum(len(json.dumps(REGISTRY[n].openai_schema())) // 4
                   for n, _ in rare if n in REGISTRY)
        got = sum(c for _, c in rare)
        print(f"\ntools under 1% of calls: {len(rare)} "
              f"({got} calls, {got / total:.1%}) costing ~{cost} schema tokens "
              f"on every request")


if __name__ == "__main__":
    main()
