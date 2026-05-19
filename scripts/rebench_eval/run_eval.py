#!/usr/bin/env python3
"""Evaluate SWE-rebench predictions using the official Docker eval images.

For each prediction, this script:
  1. Pulls the instance's swerebench Docker eval image
  2. Applies the test_patch (gold tests) and model_patch (LLM fix)
  3. Runs the test command from install_config
  4. Parses FAIL_TO_PASS and PASS_TO_PASS results
  5. Reports pass/fail per instance

Usage (from host, NOT inside llmtk container — needs Docker socket):
    python3 scripts/rebench_eval/run_eval.py \
        --predictions scripts/rebench_eval/results/gemma4-31b-v1/predictions.jsonl \
        --instances scripts/rebench_eval/results/gemma4-31b-v1/instances.jsonl \
        --output scripts/rebench_eval/results/gemma4-31b-v1/eval_results.jsonl

Requirements: Docker available on the host, Python 3.6+ on the host.
"""
from __future__ import print_function

import json
import os
import subprocess
import sys
import tempfile
import time


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records



def parse_pytest_output(output, fail_to_pass, pass_to_pass):
    """Parse pytest output to determine which tests passed/failed.

    Returns dict with fail_to_pass / pass_to_pass / eval_status fields.
    eval_status is one of: "ok", "eval_error" (env/install/test_patch broken).
    """
    # Detect environmental failure sentinels emitted by the eval script.
    eval_status = "ok"
    eval_error = ""
    if "EVAL_ERROR_TEST_PATCH" in output:
        eval_status = "eval_error"
        eval_error = "test_patch (gold tests) failed to apply"
    elif "EVAL_ERROR_REINSTALL" in output:
        eval_status = "eval_error"
        eval_error = "package re-install failed after patching"
    elif "EVAL_ERROR_PRE_INSTALL" in output:
        eval_status = "eval_error"
        eval_error = "pre_install commands failed"

    # Build lookup: test_id -> "PASSED" | "FAILED" | "ERROR"
    test_results = {}
    for line in output.split("\n"):
        line = line.strip()
        # pytest -rA format: "PASSED tests/foo.py::test_bar" or "FAILED tests/..."
        for status in ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS", "SKIPPED"):
            if line.startswith(status + " "):
                test_id = line[len(status) + 1:].strip()
                # Remove any trailing info after " -" (e.g. " - assert ...")
                if " - " in test_id:
                    test_id = test_id.split(" - ")[0].strip()
                test_results[test_id] = status
                break

    # Check FAIL_TO_PASS: these should now PASS
    f2p_passed = 0
    f2p_details = {}
    for test in fail_to_pass:
        # Try exact match first, then prefix match (parametrized tests)
        result = test_results.get(test)
        if result is None:
            # Try matching without parameters or with partial match
            for k, v in test_results.items():
                if test in k or k in test:
                    result = v
                    break
        status = result or "NOT_FOUND"
        f2p_details[test] = status
        if status == "PASSED":
            f2p_passed += 1

    # Check PASS_TO_PASS: these should still PASS
    p2p_passed = 0
    p2p_details = {}
    for test in (pass_to_pass or [])[:20]:  # cap at 20 for output sanity
        result = test_results.get(test)
        if result is None:
            for k, v in test_results.items():
                if test in k or k in test:
                    result = v
                    break
        status = result or "NOT_FOUND"
        p2p_details[test] = status
        if status in ("PASSED", "XFAIL"):
            p2p_passed += 1

    # Post-check: if all F2P tests are NOT_FOUND and the output contains
    # import/module errors, treat as eval_error rather than test_failure.
    if eval_status == "ok" and fail_to_pass:
        all_missing = all(v == "NOT_FOUND" for v in f2p_details.values())
        if all_missing:
            output_lower = output.lower()
            if (
                "modulenotfounderror" in output_lower
                or ("importerror" in output_lower and "no module named" in output_lower)
                or "warn: re-install failed" in output_lower
            ):
                eval_status = "eval_error"
                eval_error = "import/module error — likely environment/install failure"

    # v6d: REINSTALL is purely cosmetic when the test phase ran cleanly
    # and all F2P + checked P2P passed. The eval script's reinstall step
    # can fail due to environment quirks (missing conda env name,
    # /testbed bind quirks) without affecting the test results that
    # already ran. Don't penalize a clean pass for a post-test reinstall
    # hiccup. Scope: only REINSTALL — PRE_INSTALL means deps never
    # installed and TEST_PATCH means the gold tests didn't apply, both
    # of which genuinely break test measurement.
    if (
        eval_status == "eval_error"
        and "re-install" in eval_error.lower()
        and len(fail_to_pass) > 0 and f2p_passed == len(fail_to_pass)
        and len(p2p_details) > 0 and p2p_passed == len(p2p_details)
    ):
        eval_status = "ok"
        eval_error = ""

    return {
        "fail_to_pass": {"passed": f2p_passed, "total": len(fail_to_pass), "details": f2p_details},
        "pass_to_pass": {"passed": p2p_passed, "total": len(pass_to_pass or []), "checked": len(p2p_details), "details": p2p_details},
        "eval_status": eval_status,
        "eval_error": eval_error,
    }


def evaluate_instance(instance, prediction, timeout=600, scratch_dir=None):
    """Evaluate a single instance using its Docker eval image.

    scratch_dir: directory for temp files (must be on a filesystem visible to
    Docker — i.e. NOT container-local /tmp when running inside a sibling
    container).  Defaults to system tempdir.

    Returns a dict with evaluation results.
    """
    instance_id = instance["instance_id"]
    docker_image = instance.get("docker_image") or instance.get("image_name")
    if not docker_image:
        return {"instance_id": instance_id, "resolved": False, "error": "no docker_image"}

    model_patch = prediction.get("model_patch", "")
    test_patch = instance.get("test_patch", "")
    test_cmd = (instance.get("install_config") or {}).get("test_cmd", "")
    fail_to_pass = instance.get("FAIL_TO_PASS", [])
    pass_to_pass = instance.get("PASS_TO_PASS", [])

    if not model_patch.strip():
        return {"instance_id": instance_id, "resolved": False, "error": "empty patch"}

    if not test_cmd:
        return {"instance_id": instance_id, "resolved": False, "error": "no test_cmd"}

    install_cmd = (instance.get("install_config") or {}).get("install", "")
    pre_install = (instance.get("install_config") or {}).get("pre_install") or []

    # Pull the eval image
    print("  Pulling %s ..." % docker_image)
    rc = subprocess.call(
        ["docker", "pull", docker_image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if rc != 0:
        return {"instance_id": instance_id, "resolved": False, "error": "docker pull failed"}

    # Write patches to temp files.
    # When running inside a container, /tmp is container-local and invisible
    # to sibling Docker containers.  Use scratch_dir (on a shared volume) instead.
    tmpdir = tempfile.mkdtemp(prefix="swe_eval_", dir=scratch_dir)
    model_patch_path = os.path.join(tmpdir, "model.patch")
    test_patch_path = os.path.join(tmpdir, "test.patch")

    with open(model_patch_path, "w") as f:
        f.write(model_patch)
    with open(test_patch_path, "w") as f:
        f.write(test_patch)

    # Build the eval script that runs inside the container.
    # The eval images have the repo at /testbed/ checked out at base_commit
    # and a conda env called "testbed" with the correct Python + deps.
    #
    # Flow: activate env -> apply test_patch -> apply model_patch ->
    #       re-install package (so code changes are importable) -> run tests.
    lines = [
        "#!/bin/bash",
        "source /opt/conda/etc/profile.d/conda.sh",
        "conda activate testbed",
        "cd /testbed",
        "",
    ]
    # A2: emit pre_install commands from instance config.  Several rebench
    # instances (e.g. agentdojo) require build-tool installs or env tweaks
    # before the package itself is re-installable.
    if pre_install:
        lines.append("# pre_install commands from instance config")
        for cmd in pre_install:
            lines.append("(%s) || { echo 'EVAL_ERROR_PRE_INSTALL'; exit 3; }" % cmd)
        lines.append("")

    # A4: surface test_patch apply failures.  Without this, F2P test IDs
    # don't exist and every test scores NOT_FOUND, indistinguishable from
    # a wrong model patch.
    #
    # v6e: capture git-apply stderr to a temp file in the eval container and
    # echo it before exit.  Without this, output_tail just shows git-apply's
    # generic help text — we lose the actual rejection reason (which hunk
    # failed, what context didn't match).  Diagnoses v6d cases like
    # react-datepicker-4282 / koa-781 without manual re-runs.
    lines += [
        "# Apply test patch (gold tests) first",
        "if [ -s /tmp/patches/test.patch ]; then",
        "    if ! git apply --allow-empty /tmp/patches/test.patch 2>/tmp/test_patch.err && \\",
        "       ! git apply --allow-empty --3way /tmp/patches/test.patch 2>/tmp/test_patch.err; then",
        '        echo "EVAL_ERROR_TEST_PATCH"',
        '        echo "--- test_patch apply stderr ---"',
        "        cat /tmp/test_patch.err 2>/dev/null || true",
        '        echo "--- end test_patch apply stderr ---"',
        "        exit 2",
        "    fi",
        "fi",
        "",
        "# Apply model patch (LLM's fix)",
        "if [ -s /tmp/patches/model.patch ]; then",
        "    if ! git apply --allow-empty /tmp/patches/model.patch 2>/tmp/model_patch.err && \\",
        "       ! git apply --allow-empty --3way /tmp/patches/model.patch 2>/tmp/model_patch.err; then",
        '        echo "ERROR: model_patch apply failed"',
        '        echo "--- model_patch apply stderr ---"',
        "        cat /tmp/model_patch.err 2>/dev/null || true",
        '        echo "--- end model_patch apply stderr ---"',
        "        exit 1",
        "    fi",
        "fi",
        "",
    ]
    # A1: capture re-install exit code so install failures are scored as
    # eval_error (env break) rather than test_failure (wrong patch).
    # --no-deps since the eval image has deps pre-installed; --no-build-isolation
    # avoids fetching setuptools over the network.
    lines.append("# Re-install package after patching (no-deps, offline)")
    lines.append("pip install --no-deps --no-build-isolation -e . 2>&1")
    lines.append("REINSTALL_RC=$?")
    lines.append("if [ $REINSTALL_RC -ne 0 ]; then echo 'EVAL_ERROR_REINSTALL'; fi")
    lines.append("")
    # Run the FAIL_TO_PASS tests specifically (not the full suite).
    # This avoids unrelated collection errors and speeds up evaluation.
    if fail_to_pass:
        f2p_args = " ".join('"%s"' % t for t in fail_to_pass)
        lines.append("# Run FAIL_TO_PASS tests")
        lines.append("%s %s 2>&1 || true" % (test_cmd, f2p_args))
    else:
        lines.append("# Run the full test command (no specific F2P tests)")
        lines.append("%s 2>&1 || true" % test_cmd)
    eval_script = "\n".join(lines) + "\n"

    eval_script_path = os.path.join(tmpdir, "eval.sh")
    with open(eval_script_path, "w") as f:
        f.write(eval_script)
    os.chmod(eval_script_path, 0o755)

    # Run evaluation in Docker
    container_name = "swe_eval_%s_%d" % (instance_id.replace("/", "_"), int(time.time()))
    docker_cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,
        # Note: some test suites import modules that make network calls at
        # collection time, so we allow network access during eval.
        "-v", "%s:/tmp/patches:ro" % tmpdir,
        docker_image,
        "bash", "/tmp/patches/eval.sh",
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout,
        )
        output = result.stdout.decode("utf-8", "replace") + "\n" + result.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        # Kill the container
        subprocess.call(["docker", "rm", "-f", container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"instance_id": instance_id, "resolved": False, "error": "timeout (%ds)" % timeout}
    except Exception as e:
        return {"instance_id": instance_id, "resolved": False, "error": str(e)}
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Parse test results
    test_results = parse_pytest_output(output, fail_to_pass, pass_to_pass)

    f2p = test_results["fail_to_pass"]
    eval_status = test_results.get("eval_status", "ok")
    # Only consider "resolved" when the eval ran cleanly. Patches whose
    # evaluation hit env_error are not "failed", they're un-evaluated.
    resolved = (
        eval_status == "ok"
        and f2p["passed"] == f2p["total"]
        and f2p["total"] > 0
    )

    out = {
        "instance_id": instance_id,
        "resolved": resolved,
        "tests": test_results,
        "exit_code": result.returncode,
        "output_tail": output[-2000:] if len(output) > 2000 else output,
    }
    if eval_status != "ok":
        out["error"] = test_results.get("eval_error", "eval error")
        out["eval_status"] = eval_status
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate SWE-rebench predictions")
    p.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    p.add_argument("--instances", required=True, help="Path to instances JSONL")
    p.add_argument("--output", default=None, help="Output eval results JSONL")
    p.add_argument("--timeout", type=int, default=600, help="Per-instance timeout (seconds)")
    p.add_argument("--scratch-dir", default=None,
                   help="Directory for temp files (must be on a Docker-visible "
                        "filesystem, not container-local /tmp). Defaults to "
                        "a .scratch/ dir next to predictions file.")
    args = p.parse_args()

    # Scratch dir for temp files (patches, eval scripts) that must be visible
    # to sibling Docker containers.  When running inside a container, use a
    # path on a shared filesystem (e.g. /data/gondor/...) — NOT /tmp or
    # /project/... which are container-local bind mounts.
    scratch_dir = args.scratch_dir
    if scratch_dir is None:
        pred_dir = os.path.dirname(os.path.abspath(args.predictions))
        scratch_dir = os.path.join(pred_dir, ".scratch")
    scratch_dir = os.path.abspath(scratch_dir)

    # A3: container/host path guard.  The sibling docker daemon resolves bind
    # mounts against the HOST filesystem; container-local paths silently fail
    # ("bash: /tmp/patches/eval.sh: No such file or directory") and every
    # test scores NOT_FOUND.  Refuse to run rather than waste hours diagnosing.
    if os.path.exists("/.dockerenv"):
        bad_prefixes = ("/project/", "/tmp/", "/home/")
        if any(scratch_dir.startswith(p) for p in bad_prefixes):
            sys.stderr.write(
                "ERROR: running inside a container with scratch_dir=%s\n"
                "       Sibling Docker daemon resolves bind mounts on the HOST,\n"
                "       and the host has no such path. Pick a directory under\n"
                "       a shared filesystem (e.g. /data/gondor/...).\n"
                % scratch_dir
            )
            raise SystemExit(2)

    os.makedirs(scratch_dir, exist_ok=True)
    print("Using scratch dir: %s" % scratch_dir)

    predictions = load_jsonl(args.predictions)
    instances = load_jsonl(args.instances)

    # Build lookup by instance_id
    inst_by_id = {i["instance_id"]: i for i in instances}
    pred_by_id = {p.get("instance_id", p.get("task_id")): p for p in predictions}

    results = []
    resolved_count = 0

    for instance_id, pred in pred_by_id.items():
        inst = inst_by_id.get(instance_id)
        if inst is None:
            print("SKIP %s — no matching instance" % instance_id)
            continue

        print("[%d/%d] Evaluating %s ..." % (len(results) + 1, len(pred_by_id), instance_id))
        t0 = time.time()
        result = evaluate_instance(inst, pred, timeout=args.timeout, scratch_dir=scratch_dir)
        elapsed = time.time() - t0

        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)

        status = "RESOLVED" if result.get("resolved") else "FAILED"
        f2p = result.get("tests", {}).get("fail_to_pass", {})
        print("  %s  (%s)  FAIL_TO_PASS: %d/%d  (%.1fs)" % (
            status, instance_id,
            f2p.get("passed", 0), f2p.get("total", 0),
            elapsed,
        ))
        if result.get("resolved"):
            resolved_count += 1

    # Write results
    if args.output:
        outpath = args.output
    else:
        outpath = args.predictions.replace("predictions", "eval_results")

    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    eval_error_count = sum(
        1 for r in results
        if r.get("eval_status") == "eval_error"
        or r.get("tests", {}).get("eval_status") == "eval_error"
    )
    print("\n" + "=" * 60)
    print("RESULTS: %d/%d resolved (%d eval_error)" % (
        resolved_count, len(results), eval_error_count,
    ))
    print("=" * 60)
    for r in results:
        if r.get("resolved"):
            status = "PASS"
        elif r.get("eval_status") == "eval_error" or r.get("tests", {}).get("eval_status") == "eval_error":
            status = "EVAL_ERROR"
        else:
            status = "FAIL"
        f2p = r.get("tests", {}).get("fail_to_pass", {})
        err = r.get("error", "")
        suffix = "  (%s)" % err if err else ""
        print("  %-10s  %s  F2P: %d/%d%s" % (
            status, r["instance_id"],
            f2p.get("passed", 0), f2p.get("total", 0),
            suffix,
        ))
    print("\nResults saved to: %s" % outpath)


if __name__ == "__main__":
    main()
