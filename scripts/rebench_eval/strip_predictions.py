#!/usr/bin/env python3
"""Strip artifacts from predictions JSONL for the swebench evaluator.

The swebench harness only needs: instance_id, model_name_or_path, model_patch.
Our predictions.jsonl includes large artifacts (transcripts, tool_schemas, etc.)
that bloat the file and can confuse some evaluator versions.

Usage:
    python strip_predictions.py predictions.jsonl predictions_eval.jsonl
"""
import json
import sys

KEEP_KEYS = {"instance_id", "model_name_or_path", "model_patch"}


def main():
    if len(sys.argv) < 3:
        print("Usage: strip_predictions.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    inpath, outpath = sys.argv[1], sys.argv[2]
    count = 0
    with open(inpath) as fin, open(outpath, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            clean = {k: record[k] for k in KEEP_KEYS if k in record}
            fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
            count += 1

    print("Wrote %d records to %s" % (count, outpath))


if __name__ == "__main__":
    main()
