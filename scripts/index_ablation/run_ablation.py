#!/usr/bin/env python3
"""SWE-rebench-V2 eval for squishy: patch rate + steps, with vs without the index.

For each instance:
  1. pull the dataset's prebuilt image, copy the repo out to a host workspace
  2. run the agent on that workspace twice — index off, index on
  3. record whether a patch was produced (git diff), and how many tool calls /
     turns it took

The agent runs on the HOST against the extracted repo, not inside the
container: the images are language-specific (go/rust/java toolchains) and don't
all have a usable Python for squishy. That means `run_command` can't run the
project's real test suite, so this measures PATCH RATE and NAVIGATION EFFORT
(the two things we're asking about), not test-pass rate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from squishy.api import Squishy  # noqa: E402
from squishy.index import build_index, save_index  # noqa: E402

WORK = Path(os.environ.get("ABLATION_WORK", "/tmp/squishy_ablation/work"))
# Directive, not passive. A weak model told to "fix the issue" explores,
# concludes it understands the bug, and stops without editing. The forcing
# language ("do not stop until you have edited", "an empty diff is a failure")
# is what moves patch rate — borrowed from the quant-tuner SWE harness.
PROMPT = """Fix the bug described below in this repository.

{problem}

Your job is to CHANGE THE SOURCE CODE — the modules that implement the
behavior, not the test suite. Find the function or class responsible and edit
its implementation with `edit_file`. Writing or editing tests is never a valid
fix on its own.

Understanding the bug is not the goal; editing the code is. Do not stop until
you have actually edited a non-test source file — a run that ends with no edit
is scored as a failure. When the fix is in place, give a short summary.
"""

# Fires only when the agent stopped cleanly having changed nothing — the
# dominant empty-patch failure mode on small models.
EMPTY_PATCH_NUDGE = """STOP — you have not edited any file, so there is nothing
to grade. You said you understand the bug; now act on it. Use `edit_file` to
change the responsible non-test source file now. Do not stop again until a file
has actually been modified."""

def sh(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def extract_repo(inst: dict, dest: Path) -> str | None:
    """Copy the repo out of the instance image into *dest*. Returns error or None."""
    image = inst["image_name"]
    p = sh("docker", "pull", "-q", image, timeout=1800)
    if p.returncode != 0:
        return f"pull failed: {p.stderr.strip()[:200]}"
    cid = sh("docker", "create", image, "true").stdout.strip()
    if not cid:
        return "docker create failed"
    try:
        # The checkout path varies per image (often /<reponame>), so locate the
        # .git directory rather than guessing a fixed path.
        probe = sh("docker", "run", "--rm", "--entrypoint", "sh", image, "-c",
                   "find / -maxdepth 3 -name .git -type d 2>/dev/null | head -1",
                   timeout=600)
        gitdir = probe.stdout.strip().splitlines()
        if not gitdir or not gitdir[0]:
            return "no git repo found in image"
        src = str(Path(gitdir[0]).parent)
        dest.mkdir(parents=True, exist_ok=True)
        p = sh("docker", "cp", f"{cid}:{src}/.", str(dest), timeout=1800)
        if p.returncode != 0:
            return f"docker cp failed: {p.stderr.strip()[:200]}"
    finally:
        sh("docker", "rm", "-f", cid)
    return None


async def run_one(inst: dict, workspace: Path, use_index: bool, args) -> dict:
    if use_index:
        t0 = time.time()
        save_index(str(workspace), build_index(str(workspace)))
        index_s = time.time() - t0
    else:
        shutil.rmtree(workspace / ".squishy", ignore_errors=True)
        index_s = 0.0

    events: list[dict] = []
    nudges = 0
    t0 = time.time()
    err = ""
    async with Squishy(
        model=args.model, base_url=args.base_url, api_key="local",
        permission_mode=args.mode, max_turns=args.max_turns,
        request_timeout=180.0, max_retries=3,
        # run_command executes inside the instance's own image with the
        # workspace mounted, so the project's toolchain and dependencies are
        # available and the agent's test runs are real.
        use_sandbox=not args.no_sandbox, sandbox_image=inst["image_name"],
    ) as sq:
        try:
            res = await sq.run(
                PROMPT.format(problem=inst["problem_statement"][:6000]),
                working_dir=str(workspace), timeout=args.task_timeout,
                on_event=events.append,
                notes={"fail_to_pass_tests": json.dumps(inst.get("FAIL_TO_PASS") or []),
                       "test_cmd": inst.get("test_cmd", "")},
            )
            success, turns = res.success, res.turns_used
            # Empty-patch retry (bounded): if it stopped cleanly with an
            # unchanged tree, push once more before giving up.
            for _ in range(args.empty_patch_retries):
                if sh("git", "-C", str(workspace), "diff").stdout.strip():
                    break
                res = await sq.run(
                    EMPTY_PATCH_NUDGE, working_dir=str(workspace),
                    timeout=args.task_timeout, on_event=events.append,
                )
                turns += res.turns_used
                nudges += 1
        except Exception as e:  # noqa: BLE001
            success, turns, err = False, 0, f"{type(e).__name__}: {e}"

    diff = sh("git", "-C", str(workspace), "diff").stdout
    # Ordered (turn, tool) trace so we can see WHEN a tool was used, not just
    # how often — e.g. whether the must-edit gate actually withdrew run_command.
    trace: list[str] = []
    cur = 0
    for e in events:
        if e.get("type") == "turn":
            cur = e["turn"]
        elif e.get("type") == "tool":
            trace.append(f"{cur}:{e['name']}")
    tools = [e for e in events if e.get("type") == "tool"]
    counts: dict[str, int] = {}
    for e in tools:
        counts[e["name"]] = counts.get(e["name"], 0) + 1
    return {
        "instance_id": inst["instance_id"], "language": inst["language"],
        "index": use_index, "agent_success": success, "turns": turns,
        "tool_calls": len(tools), "tool_counts": counts,
        "patched": bool(diff.strip()), "patch_bytes": len(diff),
        "elapsed_s": round(time.time() - t0, 1), "index_build_s": round(index_s, 1),
        "error": err, "patch": diff, "trace": trace, "empty_patch_nudges": nudges,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True,
                    help="JSONL from select_instances.py")
    ap.add_argument("--out", default="ablation_results.jsonl")
    ap.add_argument("--model", default="ornith-1.0-35b")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--mode", default="bench")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--no-sandbox", action="store_true")
    ap.add_argument("--empty-patch-retries", type=int, default=2)
    ap.add_argument("--keep-images", action="store_true",
                    help="Keep pulled images (default: remove after each instance)")
    ap.add_argument("--task-timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--languages", default="")
    ap.add_argument("--cleanup-images", action="store_true",
                    help="Remove the images pulled for these instances and exit")
    args = ap.parse_args()

    insts = [json.loads(x) for x in Path(args.instances).read_text().splitlines() if x.strip()]
    if args.cleanup_images:
        for inst in insts:
            p = sh("docker", "rmi", "-f", inst["image_name"], timeout=300)
            print(f"  {'removed' if p.returncode == 0 else 'skip   '} {inst['image_name']}")
        shutil.rmtree(WORK, ignore_errors=True)
        return
    if args.languages:
        keep = set(args.languages.split(","))
        insts = [i for i in insts if i["language"] in keep]
    if args.limit:
        insts = insts[: args.limit]

    out = Path(args.out).open("a")
    for n, inst in enumerate(insts, 1):
        iid = inst["instance_id"]
        base = WORK / iid
        shutil.rmtree(base, ignore_errors=True)
        print(f"[{n}/{len(insts)}] {inst['language']:7} {iid}", flush=True)
        err = extract_repo(inst, base)
        if err:
            print(f"    SKIP: {err}", flush=True)
            out.write(json.dumps({"instance_id": iid, "language": inst["language"],
                                  "setup_error": err}) + "\n"); out.flush()
            continue
        for use_index in (False, True):
            sh("git", "-C", str(base), "checkout", "--", ".")
            sh("git", "-C", str(base), "clean", "-fd", "--", ".")
            r = await run_one(inst, base, use_index, args)
            tag = "index" if use_index else "plain"
            print(f"    {tag:5} patched={str(r['patched']):5} turns={r['turns']:3} "
                  f"tools={r['tool_calls']:3} {r['elapsed_s']:6.1f}s {r['error'][:40]}", flush=True)
            out.write(json.dumps(r) + "\n"); out.flush()
        shutil.rmtree(base, ignore_errors=True)
        if not args.keep_images:
            sh("docker", "rmi", "-f", inst["image_name"], timeout=300)
    out.close()


if __name__ == "__main__":
    asyncio.run(main())
