#!/usr/bin/env python3
"""Run the quant-tuner SWE agent against an existing OpenAI-compatible endpoint.

Its own CLI (`scripts/run_swebench_eval.py`) always spawns a llama-server from
a GGUF path. For a head-to-head against squishy the two agents must talk to the
*same* served model, so this calls `run_swebench_eval(base_url=...)` directly —
that entry point already supports it.

Nothing in quant-tuner is modified; this only supplies arguments.

    python scripts/rebench_container/run_quant_tuner.py \
        --holdout ../quant-tuner/out/external/swe-rebench/holdout.jsonl \
        --workspace /tmp/qt_compare --model ornith-1.0-35b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-tuner", default="../quant-tuner",
                    help="Path to the quant-tuner checkout")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--model", default="ornith-1.0-35b",
                    help="Model id the endpoint serves")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--agent", default="openai-agents",
                    choices=["openai-agents", "mini-swe"])
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--instance-timeout", type=int, default=1800)
    ap.add_argument("--step-timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cleanup-images", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    qt = Path(args.quant_tuner).resolve()
    if not (qt / "src" / "quant_tuner").is_dir():
        print(f"error: no quant_tuner package under {qt}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(qt / "src"))

    from quant_tuner.eval.swebench import run_swebench_eval
    from quant_tuner.eval.toolcall import Sampling

    holdout = Path(args.holdout).resolve()
    instances = [json.loads(x) for x in holdout.read_text().splitlines() if x.strip()]
    if args.limit:
        instances = instances[: args.limit]
        holdout = Path(args.workspace) / "holdout_subset.jsonl"
        holdout.parent.mkdir(parents=True, exist_ok=True)
        holdout.write_text("".join(json.dumps(i) + "\n" for i in instances))

    print(f"{len(instances)} instances, agent={args.agent}, model={args.model}")

    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    summary = run_swebench_eval(
        holdout,
        base_url=args.base_url,
        served_model=args.model,
        model_label=args.model,
        api_key=args.api_key,
        trajectory_dir=ws / "trajectories",
        agent=args.agent,
        sampling=Sampling(temperature=args.temperature),
        max_steps=args.max_steps,
        instance_timeout=args.instance_timeout,
        step_timeout=args.step_timeout,
        max_tokens=args.max_tokens,
        cleanup_images=args.cleanup_images,
        resume=args.resume,
        progress=True,
    )

    from dataclasses import asdict, is_dataclass
    payload = asdict(summary) if is_dataclass(summary) else summary
    (ws / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    # SweSummary names this `per_instance`; the other two keys never existed,
    # so this line printed "resolved 0/0 (0%)" over a run that had in fact
    # resolved everything it was given.
    records = payload.get("per_instance") or []
    n = len(records) or 1
    res = sum(1 for r in records if r.get("resolved"))
    pat = sum(1 for r in records if r.get("patch_produced"))
    print(f"\nquant-tuner/{args.agent}: resolved {res}/{len(records)} "
          f"({res / n:.0%})  patched {pat}/{len(records)} ({pat / n:.0%})")
    print(f"wrote {ws / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
