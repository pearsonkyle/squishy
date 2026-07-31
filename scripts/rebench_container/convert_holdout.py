#!/usr/bin/env python3
"""Convert a raw SWE-rebench holdout row into this harness's instance format.

Lets squishy run on exactly the same instances as another harness (e.g. the
quant-tuner SWE eval, whose holdout is raw dataset rows) so a head-to-head
comparison is on identical work rather than merely similar-looking samples.

    python scripts/rebench_container/convert_holdout.py \
        ../quant-tuner/out/external/swe-rebench/holdout.jsonl \
        --out scripts/rebench_container/holdout_shared.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(row: dict) -> dict:
    ic = row.get("install_config") or {}
    return {
        "instance_id": row["instance_id"],
        "repo": row.get("repo", ""),
        # Raw rows carry no language field; these holdouts are Python.
        "language": row.get("language", "python"),
        "image_name": row.get("image_name") or row.get("docker_image", ""),
        "base_commit": row.get("base_commit", ""),
        "problem_statement": row.get("problem_statement", ""),
        "test_cmd": ic.get("test_cmd", ""),
        "install": ic.get("install") or [],
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or [],
        "PASS_TO_PASS": row.get("PASS_TO_PASS") or [],
        "test_patch": row.get("test_patch") or "",
        # Named `patch` upstream; `gold_patch` here, to keep it clearly
        # distinct from the model's patch everywhere downstream.
        "gold_patch": row.get("patch") or "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("holdout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.holdout).read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out = [convert(r) for r in rows]

    missing = [r["instance_id"] for r in out if not r["image_name"] or not r["test_cmd"]]
    if missing:
        print(f"warning: {len(missing)} rows lack an image or test_cmd: {missing[:5]}")

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {len(out)} instances -> {args.out}")


if __name__ == "__main__":
    main()
