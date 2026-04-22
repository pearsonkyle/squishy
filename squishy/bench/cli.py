"""``squishy-bench`` CLI entry point.
 
Subcommands:
  swe   — run SWE-bench instances (from JSONL) -> predictions.jsonl
  term  — run Terminal-bench tasks (from JSONL) -> results.jsonl
"""
 
from __future__ import annotations
 
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
 
from squishy.api import Squishy
from squishy.bench.runner import BenchResult, PredictionWriter, run_batch
from squishy.bench.swebench import run_swebench_instance
from squishy.bench.terminalbench import TerminalTask, load_tasks, run_terminal_task
 
 
def _load_instances(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="squishy-bench", description="Run benchmarks with squishy.")
    sub = p.add_subparsers(dest="cmd", required=True)

    swe = sub.add_parser("swe", help="SWE-bench instances -> predictions.jsonl")
    swe.add_argument("--instances", required=True, help="Path to SWE-bench instances JSONL")
    swe.add_argument("--output", default="predictions.jsonl")
    swe.add_argument("--workspace-root", default="./_swe_workspaces")
    swe.add_argument("--model-name", required=True, help="Model identifier for predictions")
    swe.add_argument("--concurrency", type=int, default=2)
    swe.add_argument("--task-timeout", type=float, default=900.0)
    swe.add_argument("--limit", type=int, default=None)

    term = sub.add_parser("term", help="Terminal-bench tasks -> results.jsonl")
    term.add_argument("--tasks", required=True, help="Path to tasks JSONL or JSON")
    term.add_argument("--output", default="results.jsonl")
    term.add_argument("--workspace-root", default=None, help="If omitted, uses temp dirs")
    term.add_argument("--concurrency", type=int, default=4)
    term.add_argument("--limit", type=int, default=None)

    # SWE-bench tasks need more turns for planning + execution
    swe.add_argument("--max-turns", type=int, default=200,
                     help="Max turns per task (default 200 for SWE-bench)")
    term.add_argument("--max-turns", type=int, default=40,
                      help="Max turns per task (default 40 for Terminal-bench)")

    for sp in (swe, term):
        sp.add_argument("--base-url", default="http://localhost:1234/v1")
        sp.add_argument("--api-key", default="local")
        sp.add_argument("--model", required=True)
        sp.add_argument("--temperature", type=float, default=0.3)
        sp.add_argument("--request-timeout", type=float, default=120.0)
        sp.add_argument("--max-retries", type=int, default=4)
        sp.add_argument("--sandbox", action="store_true", help="Enable Docker sandbox for run_command")
        sp.add_argument("--max-history-messages", type=int, default=10,
                        help="Number of conversation messages to retain per turn (default 10)")
        sp.add_argument("--max-consecutive-errors", type=int, default=3,
                        help="Consecutive tool failures before aborting (default 3)")
        sp.add_argument("--max-plan-nudges", type=int, default=4,
                        help="Max plan-mode nudges before giving up (default 4)")
        sp.add_argument("--max-plan-investigation-turns", type=int, default=4,
                        help="Plan-mode read-only turns allowed before nudging toward plan_task")
        sp.add_argument("--max-recall-skip-turns", type=int, default=2,
                        help="Plan-mode consecutive reads allowed without calling recall")

    # SWE-bench specific: auto-init for repo indexing
    swe.add_argument("--auto-init", action="store_true",
                     help="Automatically build repo index before running tasks")

    return p.parse_args()


async def _amain(args: argparse.Namespace) -> int:
    async with Squishy(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_turns=args.max_turns,
        permission_mode="yolo",
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        use_sandbox=args.sandbox,
        max_consecutive_errors=args.max_consecutive_errors,
        max_plan_nudges=args.max_plan_nudges,
        max_plan_investigation_turns=args.max_plan_investigation_turns,
        max_recall_skip_turns=args.max_recall_skip_turns,
        max_history_messages=args.max_history_messages,
        auto_init=args.auto_init if args.cmd == "swe" else False,
    ) as squishy:
        if not await squishy.health():
            print(f"! cannot reach {args.base_url}", file=sys.stderr)
            return 1
 
        async with PredictionWriter(args.output) as writer:
            if args.cmd == "swe":
                instances = _load_instances(args.instances)
                if args.limit:
                    instances = instances[: args.limit]
                print(f"[squishy-bench] running {len(instances)} SWE-bench instances")
 
                async def _runner(inst: dict[str, Any]) -> BenchResult:
                    return await run_swebench_instance(
                        inst,
                        squishy=squishy,
                        workspace_root=args.workspace_root,
                        model_name=args.model_name,
                        task_timeout=args.task_timeout,
                        auto_init=args.auto_init,
                    )
 
                results = await run_batch(
                    instances,
                    _runner,
                    concurrency=args.concurrency,
                    writer=writer,
                    on_progress=_progress,
                )
 
            elif args.cmd == "term":
                tasks = load_tasks(args.tasks)
                if args.limit:
                    tasks = tasks[: args.limit]
                print(f"[squishy-bench] running {len(tasks)} terminal-bench tasks")
 
                async def _runner(t: TerminalTask) -> BenchResult:  # type: ignore[no-redef]
                    return await run_terminal_task(
                        t, squishy=squishy, workspace_root=args.workspace_root
                    )
 
                results = await run_batch(
                    tasks,
                    _runner,
                    concurrency=args.concurrency,
                    writer=writer,
                    on_progress=_progress,
                )
            else:
                print(f"unknown cmd: {args.cmd}", file=sys.stderr)
                return 2
 
        passed = sum(1 for r in results if r.success)
        print(f"\n[squishy-bench] {passed}/{len(results)} passed")
        return 0
 
 
def _progress(result: BenchResult, done: int, total: int) -> None:
    status = "✓" if result.success else "✗"
    suffix = f" ({result.error[:60]})" if result.error else ""
    print(f"  [{done:>3}/{total}] {status} {result.task_id} ({result.elapsed_s:.1f}s){suffix}")
 
 
def run() -> None:
    args = _parse()
    sys.exit(asyncio.run(_amain(args)))