#!/usr/bin/env python3
"""Summarize in-container bench results.

Headline is `resolved` (the gold tests actually passed). `patched` is shown
alongside deliberately: the gap between them is the point — a non-empty diff
says almost nothing about whether the bug got fixed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--failures", action="store_true",
                    help="Also show the aggregated tool-failure causes")
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.results).read_text().splitlines() if x.strip()]
    if not rows:
        print("no results")
        return

    # An instance that isn't gradable here can't count for or against an arm.
    gradable = [r for r in rows if not r.get("eval_error") and not r.get("setup_error")]
    ungradable = [r for r in rows if r.get("eval_error") or r.get("setup_error")]

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in gradable:
        by_arm[f"{r['tools']}{'+index' if r.get('index') else ''}"].append(r)

    print(f"{len(rows)} runs, {len(gradable)} gradable, {len(ungradable)} excluded\n")
    hdr = (f"{'arm':16} {'resolved':>10} {'patched':>10} {'turns':>7} {'tools':>7} "
           f"{'fails':>7} {'ktok':>7}")
    print(hdr)
    print("-" * len(hdr))
    for arm, rs in sorted(by_arm.items()):
        n = len(rs)
        res = sum(bool(r.get("resolved")) for r in rs)
        pat = sum(bool(r.get("patched")) for r in rs)
        ktok = _mean([(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)) / 1000
                      for r in rs])
        print(f"{arm:16} {res}/{n} ({res / n:>4.0%}) {pat}/{n} ({pat / n:>4.0%}) "
              f"{_mean([r.get('turns', 0) for r in rs]):7.1f} "
              f"{_mean([r.get('tool_calls', 0) for r in rs]):7.1f} "
              f"{_mean([r.get('tool_failures', 0) for r in rs]):7.1f} "
              f"{ktok:7.0f}")

    # Paired view: a single outlier shouldn't hide inside a mean.
    paired: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in gradable:
        paired[r["instance_id"]][f"{r['tools']}{'+index' if r.get('index') else ''}"] = r
    arms = sorted(by_arm)
    both = {k: v for k, v in paired.items() if len(v) == len(arms)} if len(arms) > 1 else {}
    if both:
        print(f"\nper-instance (paired, n={len(both)}):")
        print(f"  {'instance':42} {'lang':6} " + " ".join(f"{a:>18}" for a in arms))
        for iid, v in sorted(both.items()):
            cells = [f"{'R' if v[a].get('resolved') else ('p' if v[a].get('patched') else '-')}"
                     f" {v[a].get('tool_calls', 0):3}t" for a in arms]
            lang = v[arms[0]]["language"]
            print(f"  {iid[:42]:42} {lang:6} " + " ".join(f"{c:>18}" for c in cells))
        print("  legend: R=resolved  p=patched only  -=no patch  Nt=tool calls")

    if ungradable:
        print(f"\nexcluded ({len(ungradable)}):")
        for r in ungradable:
            why = r.get("eval_error") or r.get("setup_error")
            print(f"  {r['instance_id'][:44]:44} {str(why)[:80]}")

    if args.failures:
        tally: dict[str, int] = defaultdict(int)
        for r in rows:
            for k, v in (r.get("failure_reasons") or {}).items():
                tally[k] += v
        if tally:
            print("\ntool failures by cause:")
            for k, v in sorted(tally.items(), key=lambda x: -x[1])[:20]:
                print(f"  {v:4}  {k}")


if __name__ == "__main__":
    main()
