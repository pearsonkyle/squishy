#!/usr/bin/env python3
"""Export SFT training data from squishy-bench predictions.

Reads a predictions JSONL file (output of `squishy-bench swe`) and extracts
conversation transcripts into a format suitable for LLM supervised fine-tuning.

Each output line is:
    {"messages": [...], "tools": [...], "instance_id": "...", "success": true/false}

Messages format follows the OpenAI chat convention:
    - System messages are stripped (SFT trainers typically don't include them)
    - Tool schemas are embedded in the first message under "tools" key
    - tool_calls[].function.arguments are dicts (not JSON strings)
    - Conversations are trimmed to end with an assistant message

Usage:
    python export_training.py \
        --predictions results/gemma4-31b-v1/predictions.jsonl \
        --output results/gemma4-31b-v1/training.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize_arguments(messages: list[dict]) -> list[dict]:
    """Ensure tool_calls arguments are dicts, not JSON strings."""
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    func["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
    return messages


def extract_training_record(record: dict) -> dict | None:
    """Extract a single SFT training record from a predictions JSONL line."""
    artifacts = record.get("artifacts", {})
    transcript = artifacts.get("transcript", [])
    if not transcript:
        return None

    tool_schemas = artifacts.get("tool_schemas", [])

    # Strip system messages.
    messages = [m for m in transcript if m.get("role") != "system"]
    if not messages:
        return None

    # Normalize tool_call arguments.
    messages = _normalize_arguments(messages)

    # Trim trailing non-assistant messages (SFT requires assistant as last turn).
    while messages and messages[-1].get("role") != "assistant":
        messages.pop()
    if not messages:
        return None

    # Embed tool schemas in the first message.
    if tool_schemas:
        messages[0] = {**messages[0], "tools": tool_schemas}

    instance_id = record.get("instance_id", record.get("task_id", "unknown"))
    has_error = "error" in record
    return {
        "messages": messages,
        "instance_id": instance_id,
        "success": not has_error,
    }


def main():
    p = argparse.ArgumentParser(description="Export SFT training data from predictions")
    p.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    p.add_argument("--output", required=True, help="Output training JSONL path")
    p.add_argument("--success-only", action="store_true",
                   help="Only export successful instances (non-empty patch)")
    p.add_argument("--eval-results", default=None,
                   help="Path to eval_results.jsonl — when given with --resolved-only, "
                        "only export instances that were verified as resolved")
    p.add_argument("--resolved-only", action="store_true",
                   help="Only export instances that passed evaluation (requires --eval-results)")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval results if provided (for --resolved-only filtering).
    resolved_ids = None
    if args.eval_results and args.resolved_only:
        eval_path = Path(args.eval_results)
        if eval_path.exists():
            resolved_ids = set()
            with open(eval_path) as ef:
                for eline in ef:
                    eline = eline.strip()
                    if not eline:
                        continue
                    er = json.loads(eline)
                    if er.get("resolved"):
                        resolved_ids.add(er["instance_id"])
            print("Loaded eval results: %d resolved instances" % len(resolved_ids))
        else:
            print("Warning: eval results file not found: %s" % eval_path)

    total = 0
    exported = 0
    skipped_no_transcript = 0
    skipped_failed = 0
    skipped_not_resolved = 0

    with open(pred_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)

            if args.success_only and "error" in record:
                skipped_failed += 1
                continue

            if resolved_ids is not None:
                iid = record.get("instance_id", record.get("task_id", ""))
                if iid not in resolved_ids:
                    skipped_not_resolved += 1
                    continue

            training_record = extract_training_record(record)
            if training_record is None:
                skipped_no_transcript += 1
                continue

            fout.write(json.dumps(training_record, ensure_ascii=False) + "\n")
            exported += 1

    print(f"Processed {total} predictions")
    print(f"Exported {exported} training records to {out_path}")
    if skipped_no_transcript:
        print(f"Skipped {skipped_no_transcript} (no transcript)")
    if skipped_failed:
        print(f"Skipped {skipped_failed} (failed, --success-only)")
    if skipped_not_resolved:
        print(f"Skipped {skipped_not_resolved} (not resolved, --resolved-only)")


if __name__ == "__main__":
    main()
