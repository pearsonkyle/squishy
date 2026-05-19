#!/usr/bin/env python3
"""Convert SWE-rebench parquet to JSONL for squishy-bench.

Usage:
    python prepare_instances.py --sample 5 --seed 42 --tag gemma4-31b-v1

Reads from the HF cache (HF_HUB_CACHE env var or default path) and writes
a JSONL file compatible with `squishy-bench swe --instances`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Default path to the cached V1 (rebench) dataset on this system.
_DEFAULT_CACHE = (
    "/data/gondor/projects/llm-training-quantization/hub/"
    "datasets--nebius--SWE-rebench/snapshots/"
    "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881/data"
)

# V2 cache snapshot path is resolved by _resolve_v2_cache_path() below
# (after the function definition).  The snapshot hash isn't pinned
# because the cache layout has only one snapshot dir today; if HF
# re-syncs, switch to a glob.

_SPLIT_FILES = {
    "filtered": ["filtered-00000-of-00001.parquet"],
    "test": ["test-00000-of-00002.parquet", "test-00001-of-00002.parquet"],
}

# V2 has a single training shard.
_V2_FILE = "train-00000-of-00001.parquet"


def _resolve_v2_cache_path() -> str | None:
    """Resolve the V2 dataset path by listing the single snapshot dir.

    Returns ``None`` if the cache root is missing — callers should error
    cleanly when the user explicitly asked for V2.
    """
    root = (
        "/data/gondor/projects/llm-training-quantization/hub/"
        "datasets--nebius--SWE-rebench-V2/snapshots"
    )
    if not os.path.isdir(root):
        return None
    snaps = sorted(os.listdir(root))
    if not snaps:
        return None
    return os.path.join(root, snaps[0], "data")


_DEFAULT_CACHE_V2 = _resolve_v2_cache_path()


def _detect_dataset_variant(data_dir: str) -> str:
    """Probe ``data_dir`` and return either ``"v1"`` or ``"v2"``.

    Raises ``FileNotFoundError`` listing what was found if neither
    expected parquet shape is present.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir does not exist: {data_dir}")
    if os.path.exists(os.path.join(data_dir, _V2_FILE)):
        return "v2"
    if os.path.exists(os.path.join(data_dir, "filtered-00000-of-00001.parquet")):
        return "v1"
    found = sorted(os.listdir(data_dir))
    raise FileNotFoundError(
        f"Could not detect V1/V2 dataset variant in {data_dir}. "
        f"Found files: {found}"
    )


# Compat filter: drop instances that won't install cleanly under our
# py3.12 docker user-mode container.  Heuristics match
# scripts/swe_rebench_v2_compat.jsonl from MEMORY.md.
_OLD_TORCH = re.compile(r"torch\s*[<=]+\s*(?:1\.|2\.0)")


def apply_compat_filter(instances: list[dict]) -> list[dict]:
    """Drop V2 instances unlikely to install cleanly under py3.12 user mode.

    - drop ``python_base_37`` (defensive: usually filtered earlier by
      ``--python-version 3.10``, but kept as a guard if the user skips
      that flag)
    - drop instances whose install commands contain ``apt-get`` (require
      root; our container runs as user kyle.pearson)
    - drop instances whose install pins an old torch version
    """
    out: list[dict] = []
    for inst in instances:
        ic = inst.get("install_config") or {}
        if ic.get("base_image_name") == "python_base_37":
            continue
        install_str = " ".join(ic.get("install") or [])
        if "apt-get" in install_str:
            continue
        if _OLD_TORCH.search(install_str):
            continue
        out.append(inst)
    return out


def _to_serializable(obj):
    """Recursively convert numpy/pandas types to plain Python."""
    if isinstance(obj, np.ndarray):
        return [_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj


def load_and_convert(data_dir: str, split: str) -> list[dict]:
    """Load V1 parquet files and convert each row to a plain dict."""
    files = _SPLIT_FILES.get(split)
    if not files:
        raise ValueError(f"Unknown split {split!r}, choose from {list(_SPLIT_FILES)}")

    frames = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        frames.append(pd.read_parquet(path))

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} instances from {split} split")

    instances = []
    for _, row in df.iterrows():
        d = _to_serializable(row.to_dict())
        instances.append(d)
    return instances


def load_v2(data_dir: str) -> list[dict]:
    """Load the V2 train parquet (single shard) and return a list of dicts."""
    path = os.path.join(data_dir, _V2_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing V2 parquet: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} instances from V2 train split")
    return [_to_serializable(row.to_dict()) for _, row in df.iterrows()]


def main():
    p = argparse.ArgumentParser(description="Prepare SWE-rebench instances for squishy-bench")
    p.add_argument(
        "--dataset",
        default=None,
        choices=["v1", "v2"],
        help="Explicit dataset choice. If unset, --data-dir is auto-detected. "
        "Selecting 'v2' switches the default --data-dir to the V2 cache.",
    )
    p.add_argument("--data-dir", default=None, help="Path to parquet directory")
    p.add_argument("--split", default="filtered", choices=list(_SPLIT_FILES),
                   help="V1 split (ignored on V2; V2 has a single train shard)")
    p.add_argument("--sample", type=int, default=5, help="Number of instances to sample (0=all)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--tag", default="run-01", help="Run tag for output directory")
    p.add_argument("--output", default=None, help="Override output path")
    p.add_argument(
        "--python-version",
        default=None,
        help="V1: filter to instances whose install_config.python starts with this string "
        "(e.g. '3.12' matches '3.12' and '3.12.2'). "
        "V2: filter to instances whose install_config.base_image_name == 'python_base_<digits>' "
        "(e.g. '3.10' matches 'python_base_310'). Default: no filter.",
    )
    p.add_argument(
        "--language",
        default=None,
        help="V2 only: filter by install_config language field (e.g. 'python', 'go', 'ts'). "
        "Ignored on V1.",
    )
    p.add_argument(
        "--compat-filter",
        action="store_true",
        help="V2 only: drop instances unlikely to install under py3.12 user-mode docker "
        "(python_base_37, apt-get installs, pinned old torch). Ignored on V1.",
    )
    args = p.parse_args()

    # Resolve data_dir: explicit --data-dir wins; else --dataset switches the
    # default; else fall back to V1 default for backward compatibility.
    if args.data_dir is None:
        if args.dataset == "v2":
            if _DEFAULT_CACHE_V2 is None:
                raise SystemExit(
                    "ERROR: --dataset v2 requested but the V2 cache was not found at "
                    "/data/gondor/projects/llm-training-quantization/hub/datasets--nebius--SWE-rebench-V2/snapshots/. "
                    "Pass --data-dir explicitly or download the dataset."
                )
            args.data_dir = _DEFAULT_CACHE_V2
        else:
            args.data_dir = _DEFAULT_CACHE

    variant = args.dataset or _detect_dataset_variant(args.data_dir)
    print(f"Dataset variant: {variant} (data_dir={args.data_dir})")

    if variant == "v2":
        instances = load_v2(args.data_dir)
        if args.language:
            before = len(instances)
            instances = [i for i in instances if i.get("language") == args.language]
            print(f"Filtered to language={args.language}: {len(instances)}/{before} instances")
        if args.python_version:
            digits = args.python_version.replace(".", "")
            target_image = f"python_base_{digits}"
            before = len(instances)
            instances = [
                i for i in instances
                if (i.get("install_config") or {}).get("base_image_name") == target_image
            ]
            print(f"Filtered to base_image_name={target_image}: {len(instances)}/{before} instances")
        if args.compat_filter:
            before = len(instances)
            instances = apply_compat_filter(instances)
            print(f"Compat filter applied: {len(instances)}/{before} instances")
    else:
        instances = load_and_convert(args.data_dir, args.split)
        if args.python_version:
            prefix = args.python_version
            before = len(instances)
            instances = [
                inst
                for inst in instances
                if str(((inst.get("install_config") or {}).get("python") or "")).startswith(prefix)
            ]
            print(f"Filtered to python=={prefix}*: {len(instances)}/{before} instances")
        if args.language:
            print("WARNING: --language is V2-only; ignoring on V1 dataset")
        if args.compat_filter:
            print("WARNING: --compat-filter is V2-only; ignoring on V1 dataset")

    if args.sample and args.sample < len(instances):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(instances), size=args.sample, replace=False)
        instances = [instances[i] for i in sorted(indices)]
        print(f"Sampled {len(instances)} instances (seed={args.seed})")

    # Determine output path.
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).parent / "results" / args.tag / "instances.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"Wrote {len(instances)} instances to {out_path}")

    # Print summary.
    repos = {inst["repo"] for inst in instances}
    print(f"Repos: {repos}")
    for inst in instances:
        ftp = inst.get("FAIL_TO_PASS", [])
        n_tests = len(ftp) if ftp else 0
        print(f"  {inst['instance_id']} — {n_tests} failing test(s)")


if __name__ == "__main__":
    main()
