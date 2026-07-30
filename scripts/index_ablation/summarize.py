#!/usr/bin/env python3
"""Summarize index-ablation results: patch rate and navigation effort."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.results).read_text().splitlines() if x.strip()]
    runs = [r for r in rows if "setup_error" not in r]
    skipped = [r for r in rows if "setup_error" in r]

    by_cond: dict[bool, list[dict]] = defaultdict(list)
    for r in runs:
        by_cond[r["index"]].append(r)

    print(f"{len(runs)} runs over {len({r['instance_id'] for r in runs})} instances"
          f"  ({len(skipped)} skipped at setup)\n")

    hdr = f"{'condition':10} {'patch rate':>12} {'turns':>8} {'tools':>8} {'recall':>7} {'search+read':>12}"
    print(hdr)
    print("-" * len(hdr))
    for cond in (False, True):
        rs = by_cond.get(cond) or []
        if not rs:
            continue
        n = len(rs)
        patched = sum(r["patched"] for r in rs)
        turns = sum(r["turns"] for r in rs) / n
        tools = sum(r["tool_calls"] for r in rs) / n
        recall = sum(r["tool_counts"].get("recall", 0) for r in rs) / n
        browse = sum(
            r["tool_counts"].get("search_files", 0)
            + r["tool_counts"].get("read_file", 0)
            + r["tool_counts"].get("glob_files", 0)
            + r["tool_counts"].get("list_directory", 0)
            for r in rs
        ) / n
        label = "index" if cond else "no index"
        print(f"{label:10} {patched}/{n} ({patched / n:>4.0%}) {turns:8.1f} {tools:8.1f} "
              f"{recall:7.1f} {browse:12.1f}")

    # Per-instance, so a single outlier doesn't hide behind the mean.
    paired = defaultdict(dict)
    for r in runs:
        paired[r["instance_id"]][r["index"]] = r
    both = {k: v for k, v in paired.items() if len(v) == 2}
    if both:
        print(f"\nper-instance (paired, n={len(both)}):")
        print(f"  {'instance':46} {'lang':7} {'patch':>11} {'tools':>11}")
        wins = losses = 0
        for iid, v in sorted(both.items()):
            a, b = v[False], v[True]
            wins += b["tool_calls"] < a["tool_calls"]
            losses += b["tool_calls"] > a["tool_calls"]
            print(f"  {iid[:46]:46} {a['language']:7} "
                  f"{str(a['patched'])[0]}->{str(b['patched'])[0]:>8} "
                  f"{a['tool_calls']:4}->{b['tool_calls']:<6}")
        print(f"\n  index used fewer tools on {wins}/{len(both)}, more on {losses}")


if __name__ == "__main__":
    main()
