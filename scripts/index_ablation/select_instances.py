#!/usr/bin/env python3
"""Pick SWE-rebench-V2 instances for the index ablation, N per language.

Reads instance metadata from the HuggingFace datasets-server (no full dataset
download), samples N distinct repos per language, then fetches the full row
(problem statement, test command, gold patch) for each pick.

Usage:
    python scripts/index_ablation/select_instances.py \
        --languages python,ts,go,rust,java --per-language 3 \
        --out scripts/index_ablation/instances.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

SERVER = "https://datasets-server.huggingface.co"
DATASET = "nebius/SWE-rebench-V2"


def _get(url: str, attempts: int = 4) -> dict | None:
    for _ in range(attempts):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception:  # noqa: BLE001
            time.sleep(4)
    return None


def _rows_for_language(lang: str, want: int, pool: int) -> list[dict]:
    """Fetch a pool of rows for *lang* and keep `want` distinct repos."""
    where = urllib.parse.quote(f"\"language\"='{lang}'")
    ds = urllib.parse.quote(DATASET, safe="")
    out: list[dict] = []
    seen_repos: set[str] = set()
    offset = 0
    while len(out) < want and offset < pool:
        url = (f"{SERVER}/filter?dataset={ds}&config=default&split=train"
               f"&where={where}&offset={offset}&limit=100")
        d = _get(url)
        if not d or not d.get("rows"):
            break
        for r in d["rows"]:
            row = r["row"]
            if row["repo"] in seen_repos or not row.get("problem_statement"):
                continue
            seen_repos.add(row["repo"])
            ic = row.get("install_config") or {}
            out.append({
                "instance_id": row["instance_id"], "repo": row["repo"],
                "language": row["language"], "image_name": row["image_name"],
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "test_cmd": ic.get("test_cmd", ""), "install": ic.get("install") or [],
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or [],
                "gold_patch": row.get("patch") or "",
            })
            if len(out) >= want:
                break
        offset += 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="python,ts,go,rust,java")
    ap.add_argument("--per-language", type=int, default=3)
    ap.add_argument("--pool", type=int, default=400,
                    help="rows to scan per language when looking for distinct repos")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="instances.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    picked: list[dict] = []
    for lang in args.languages.split(","):
        rows = _rows_for_language(lang, args.per_language, args.pool)
        picked.extend(rows)
        print(f"  {lang:8} {len(rows)} instance(s)")
        for r in rows:
            print(f"      {r['instance_id']:46} {r['image_name'][:52]}")

    Path(args.out).write_text("".join(json.dumps(p) + "\n" for p in picked))
    print(f"\nwrote {len(picked)} instances -> {args.out}")


if __name__ == "__main__":
    main()
