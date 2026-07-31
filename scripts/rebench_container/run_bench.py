#!/usr/bin/env python3
"""SWE-rebench-V2 eval that runs squishy *inside* each instance's own image.

This is the realistic setup: one clean container per instance, squishy
installed into an isolated venv inside it, the agent working in the repo
checkout with the project's real toolchain on PATH. `run_command` is a plain
local exec, not a `docker run` per call, so running the test suite is
something the agent can actually afford to do.

Grading runs the tests. For each instance we do two real test runs:

    pre   = base + gold test patch                 -> expected to FAIL
    post  = base + gold test patch + model patch   -> PASS means resolved

The `pre` run is not ceremony: if the tests already pass without a fix, the
instance isn't gradable in this environment and a "resolved" from the `post`
run would be meaningless. Those are reported as `eval_error`, never as
successes. Patch-produced is recorded too, but it is not the headline —
a non-empty diff routinely coexists with a failing test suite.

Usage:
    python scripts/rebench_container/run_bench.py \
        --instances scripts/index_ablation/instances_sample.jsonl \
        --model ornith-1.0-35b --tools minimal --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UV_URL = ("https://github.com/astral-sh/uv/releases/latest/download/"
          "uv-x86_64-unknown-linux-gnu.tar.gz")
VENV = "/opt/squishy-venv"

PROMPT = """Fix the bug described below in this repository.

{problem}

Your job is to CHANGE THE SOURCE CODE — the modules that implement the
behavior, not the test suite. Find the function or class responsible and edit
its implementation. Writing or editing tests is never a valid fix on its own.

Understanding the bug is not the goal; editing the code is. Do not stop until
you have actually edited a non-test source file — a run that ends with no edit
is scored as a failure. When the fix is in place, give a short summary.
"""

EMPTY_PATCH_NUDGE = """STOP — you have not edited any file, so there is nothing
to grade. You said you understand the bug; now act on it. Edit the responsible
non-test source file now. Do not stop again until a file has actually been
modified."""


def sh(*args: str, timeout: int = 900, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, input=stdin)


def dexec(cid: str, script: str, *, workdir: str | None = None,
          timeout: int = 900, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec"]
    if workdir:
        cmd += ["-w", workdir]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [cid, "sh", "-c", script]
    return sh(*cmd, timeout=timeout)


# -- setup ------------------------------------------------------------------

def fetch_uv(cache: Path) -> Path:
    """Download the linux-amd64 uv binary once, reuse for every container."""
    binary = cache / "uv"
    if binary.is_file():
        return binary
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / "uv.tar.gz"
    urllib.request.urlretrieve(UV_URL, tgz)
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            if Path(m.name).name == "uv":
                m.name = "uv"
                tf.extract(m, cache)
                break
    binary.chmod(0o755)
    tgz.unlink(missing_ok=True)
    return binary


def source_tarball(cache: Path) -> Path:
    """Pack the working tree's squishy source for installation in-container."""
    out = cache / "squishy-src.tgz"
    with tarfile.open(out, "w:gz") as tf:
        for item in ("squishy", "pyproject.toml", "README.md"):
            p = REPO_ROOT / item
            if p.exists():
                tf.add(p, arcname=item, filter=_skip_junk)
    return out


def _skip_junk(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts or ".squishy" in parts:
        return None
    return info


def start_container(image: str) -> str:
    p = sh("docker", "pull", "-q", "--platform", "linux/amd64", image, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"pull failed: {p.stderr.strip()[:300]}")
    p = sh("docker", "run", "-d", "--platform", "linux/amd64",
           "--entrypoint", "sleep", image, "infinity", timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"run failed: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def find_repo(cid: str) -> str:
    """Locate the checkout. The path varies per image (/testbed, /vyper, ...)."""
    p = dexec(cid, "find / -maxdepth 3 -name .git -type d 2>/dev/null | head -1",
              timeout=600)
    line = p.stdout.strip().splitlines()
    if not line or not line[0]:
        raise RuntimeError("no git checkout found in image")
    return str(Path(line[0]).parent)


def install_agent(cid: str, uv: Path, src: Path, timeout: int = 1800) -> None:
    """Install squishy into an isolated venv with its own Python 3.11.

    Isolated on purpose: installing into the project's interpreter would drag
    squishy's dependency pins into the environment the tests run in, which can
    silently change grading results.
    """
    sh("docker", "cp", str(uv), f"{cid}:/usr/local/bin/uv", timeout=300)
    sh("docker", "cp", str(src), f"{cid}:/tmp/squishy-src.tgz", timeout=300)
    p = dexec(cid, (
        "set -e; mkdir -p /opt/squishy-src; "
        "tar -xzf /tmp/squishy-src.tgz -C /opt/squishy-src; "
        f"uv venv --python 3.11 {VENV} -q; "
        f"uv pip install --python {VENV}/bin/python -q /opt/squishy-src"
    ), timeout=timeout, env={"UV_PYTHON_INSTALL_DIR": "/opt/uvpy"})
    if p.returncode != 0:
        raise RuntimeError(f"agent install failed: {p.stderr.strip()[-500:]}")


# -- repo state -------------------------------------------------------------

def reset_repo(cid: str, repo: str, base_commit: str) -> None:
    # No -x on clean: build outputs and vendored deps (node_modules, target/)
    # are baked into the image and gitignored; removing them would break the
    # test run we are about to do.
    dexec(cid, (
        "rm -rf .squishy; git checkout -- . 2>/dev/null; git clean -fd -- . 2>/dev/null; "
        f"git checkout -f {base_commit} -- . 2>/dev/null; true"
    ), workdir=repo, timeout=600)


def collect_patch(cid: str, repo: str) -> str:
    """Diff of everything the agent changed, including files it created."""
    p = dexec(cid, (
        "rm -rf .squishy; git add -A -- . >/dev/null 2>&1; "
        "git diff --cached HEAD"
    ), workdir=repo, timeout=600)
    return p.stdout


def apply_patch(cid: str, repo: str, patch: str, label: str) -> tuple[bool, str]:
    if not patch.strip():
        return True, ""
    path = f"/tmp/{label}.patch"
    proc = subprocess.run(
        ["docker", "exec", "-i", cid, "sh", "-c", f"cat > {path}"],
        input=patch, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return False, f"could not stage {label}: {proc.stderr[:200]}"
    p = dexec(cid, (
        f"git apply --allow-empty -v {path} 2>&1 || "
        f"git apply --allow-empty -v --3way {path} 2>&1 || "
        f"patch -p1 --batch --fuzz=5 < {path} 2>&1"
    ), workdir=repo, timeout=600)
    return p.returncode == 0, ("" if p.returncode == 0 else p.stdout[-400:])


def run_tests(cid: str, repo: str, test_cmd: str, timeout: int) -> dict:
    t0 = time.time()
    try:
        p = dexec(cid, test_cmd, workdir=repo, timeout=timeout)
        return {"exit_code": p.returncode, "timed_out": False,
                "elapsed_s": round(time.time() - t0, 1),
                "tail": (p.stdout + p.stderr)[-3000:]}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "timed_out": True,
                "elapsed_s": round(time.time() - t0, 1), "tail": ""}


# -- the run ----------------------------------------------------------------

def run_agent(cid: str, repo: str, inst: dict, args, cache: Path) -> dict:
    task = {
        "repo": repo,
        "prompt": PROMPT.format(problem=inst["problem_statement"][:6000]),
        "empty_patch_nudge": EMPTY_PATCH_NUDGE,
        "empty_patch_retries": args.empty_patch_retries,
        "model": args.model,
        "base_url": args.base_url,
        "mode": args.mode,
        "tool_profile": args.tools,
        "max_turns": args.max_turns,
        "max_turns_without_edit": args.max_turns_without_edit,
        "task_timeout": args.task_timeout,
        "fail_to_pass": inst.get("FAIL_TO_PASS") or [],
        "test_cmd": inst.get("test_cmd", ""),
    }
    tf = cache / "task.json"
    tf.write_text(json.dumps(task))
    sh("docker", "cp", str(tf), f"{cid}:/opt/squishy-task.json", timeout=120)
    sh("docker", "cp", str(Path(__file__).parent / "driver.py"),
       f"{cid}:/opt/driver.py", timeout=120)

    # Hard wall on top of the agent's own timeout, so a wedged process can't
    # hold the whole sweep hostage.
    wall = int(args.task_timeout * (1 + args.empty_patch_retries) + 300)
    try:
        dexec(cid, f"{VENV}/bin/python /opt/driver.py", workdir=repo, timeout=wall)
    except subprocess.TimeoutExpired:
        pass  # partial metrics are already on disk — that's the point

    # Read back whatever the driver managed to write.
    p = dexec(cid, "cat /opt/squishy-result.json 2>/dev/null", timeout=120)
    try:
        return json.loads(p.stdout)
    except Exception:  # noqa: BLE001
        return {"exit_status": "error:no_result_file", "turns": 0,
                "tool_calls": 0, "tool_counts": {}, "trace": []}


# A pytest run whose xdist workers all crash reports "no tests ran" and exits
# 5 — indistinguishable from a genuinely empty selection, and easy to misread
# as a failing baseline. Observed on the vyper image, where every worker died
# on an unrelated typing_extensions ImportError while serial execution passed.
_NO_TESTS_MARKERS = ("no tests ran", "no tests were found", "crashed workers")


def _no_tests_ran(res: dict) -> bool:
    if res.get("timed_out"):
        return False
    if res.get("exit_code") == 5:
        return True
    tail = (res.get("tail") or "").lower()
    return any(m in tail for m in _NO_TESTS_MARKERS)


def _serial_variant(cmd: str) -> str | None:
    """Same command, parallelism off. None when that isn't a meaningful retry."""
    if "pytest" in cmd and " -n" not in cmd:
        return cmd + " -n0"
    return None


def grade(cid: str, repo: str, inst: dict, patch: str, args) -> dict:
    """Two real test runs. See module docstring for why `pre` matters."""
    test_cmd = inst.get("test_cmd") or ""
    if not test_cmd:
        return {"resolved": False, "eval_error": "instance has no test_cmd"}
    base, tp = inst["base_commit"], inst.get("test_patch") or ""
    adapted = ""

    reset_repo(cid, repo, base)
    ok, err = apply_patch(cid, repo, tp, "test")
    if not ok:
        return {"resolved": False, "eval_error": f"gold test patch failed to apply: {err}"}
    pre = run_tests(cid, repo, test_cmd, args.test_timeout)
    if _no_tests_ran(pre):
        alt = _serial_variant(test_cmd)
        if alt:
            retry = run_tests(cid, repo, alt, args.test_timeout)
            if not _no_tests_ran(retry):
                # Adopt it for both runs so pre and post stay comparable.
                test_cmd, pre, adapted = alt, retry, "serial"

    reset_repo(cid, repo, base)
    ok, err = apply_patch(cid, repo, tp, "test")
    if not ok:
        return {"resolved": False, "eval_error": f"gold test patch failed to apply: {err}"}
    applied, apply_err = apply_patch(cid, repo, patch, "model")
    post = run_tests(cid, repo, test_cmd, args.test_timeout) if applied else {
        "exit_code": None, "timed_out": False, "elapsed_s": 0.0, "tail": ""}

    out = {"pre": pre, "post": post, "model_patch_applied": applied,
           "apply_error": apply_err, "test_cmd_adapted": adapted}
    if _no_tests_ran(pre):
        out.update({"resolved": False,
                    "eval_error": "no tests ran at base — instance not gradable here"})
    elif pre["exit_code"] == 0 and not pre["timed_out"]:
        # Tests pass without any fix: this instance can't discriminate.
        out.update({"resolved": False,
                    "eval_error": "tests already pass at base — instance not gradable here"})
    elif pre["timed_out"]:
        out.update({"resolved": False, "eval_error": "baseline test run timed out"})
    elif not applied:
        out.update({"resolved": False, "eval_error": f"model patch did not apply: {apply_err}"})
    else:
        out["resolved"] = post["exit_code"] == 0 and not post["timed_out"]
    return out


def run_instance(inst: dict, args, cache: Path, uv: Path, src: Path,
                 arms: list[dict]) -> list[dict]:
    """Run every arm against one instance, reusing a single container.

    All arms share the container so the multi-GB image is pulled once and
    every arm sees byte-identical starting state — the only fair way to
    compare tool profiles.
    """
    iid = inst["instance_id"]
    cid = ""
    recs: list[dict] = []
    try:
        cid = start_container(inst["image_name"])
        repo = find_repo(cid)
        # A gold-only run never starts the agent, so skip the install.
        if any(a["tools"] != "gold" for a in arms):
            install_agent(cid, uv, src)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        if cid:
            sh("docker", "rm", "-f", cid, timeout=300)
        if not args.keep_images:
            sh("docker", "rmi", "-f", inst["image_name"], timeout=600)
        return [{"instance_id": iid, "language": inst["language"],
                 "setup_error": err, "resolved": False, **arm} for arm in arms]

    try:
        for arm in arms:
            t0 = time.time()
            rec = {"instance_id": iid, "language": inst["language"],
                   "model": args.model, **arm}
            arm_args = argparse.Namespace(**{**vars(args), **arm})
            try:
                reset_repo(cid, repo, inst["base_commit"])
                if arm["index"]:
                    # Built directly, not via `squishy --init`, which would
                    # need LLM round-trips for file summaries.
                    r = dexec(cid, (
                        f"{VENV}/bin/python -c \"from squishy.index import "
                        "build_index, save_index; save_index('.', build_index('.'))\""
                    ), workdir=repo, timeout=1800)
                    rec["index_built"] = r.returncode == 0
                    if r.returncode != 0:
                        rec["index_error"] = r.stderr[-300:]

                if arm["tools"] == "gold":
                    # Harness self-test: grade the dataset's own fix. If this
                    # doesn't come back resolved, the grading pipeline is
                    # broken and every other arm's 0% is meaningless.
                    patch = inst.get("gold_patch") or ""
                    rec["exit_status"] = "gold"
                else:
                    rec.update(run_agent(cid, repo, inst, arm_args, cache))
                    patch = collect_patch(cid, repo)
                rec["patched"] = bool(patch.strip())
                rec["patch_bytes"] = len(patch)
                rec["patch"] = patch
                rec.update(grade(cid, repo, inst, patch, arm_args))
            except Exception as e:  # noqa: BLE001
                rec["setup_error"] = f"{type(e).__name__}: {e}"
                rec.setdefault("resolved", False)
            rec["total_s"] = round(time.time() - t0, 1)
            recs.append(rec)
    finally:
        sh("docker", "rm", "-f", cid, timeout=300)
        if not args.keep_images:
            sh("docker", "rmi", "-f", inst["image_name"], timeout=600)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", default="container_results.jsonl")
    ap.add_argument("--model", default="ornith-1.0-35b")
    ap.add_argument("--base-url", default="http://host.docker.internal:1234/v1")
    ap.add_argument("--mode", default="bench")
    ap.add_argument("--tools", default="minimal",
                    help="Comma-separated arms to run per instance: minimal, "
                         "standard, or `gold` (skip the agent and grade the "
                         "dataset's own patch — a self-test of the grading "
                         "pipeline). Arms share one container.")
    ap.add_argument("--index", default="off",
                    help="Index arms to run: off, on, or both")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-turns-without-edit", type=int, default=12)
    ap.add_argument("--empty-patch-retries", type=int, default=1)
    ap.add_argument("--task-timeout", type=float, default=1200.0)
    ap.add_argument("--test-timeout", type=int, default=1800)
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--languages", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip instances already present in --out")
    args = ap.parse_args()

    insts = [json.loads(x) for x in Path(args.instances).read_text().splitlines() if x.strip()]
    if args.languages:
        keep = set(args.languages.split(","))
        insts = [i for i in insts if i["language"] in keep]
    if args.limit:
        insts = insts[: args.limit]

    # One arm per (tool profile x index setting), all sharing a container.
    index_arms = {"off": [False], "on": [True], "both": [False, True]}[args.index]
    arms = [{"tools": t, "index": i}
            for t in args.tools.split(",") for i in index_arms]
    print(f"{len(insts)} instances x {len(arms)} arms: "
          + ", ".join(f"{a['tools']}/index={a['index']}" for a in arms))

    outp = Path(args.out)
    done: set[tuple] = set()
    if args.resume and outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["instance_id"], r.get("tools"), r.get("index")))
        print(f"resuming: {len(done)} arm-runs already done")

    cache = Path(tempfile.gettempdir()) / "squishy_bench_cache"
    cache.mkdir(parents=True, exist_ok=True)
    uv = fetch_uv(cache)
    src = source_tarball(cache)

    with outp.open("a") as out:
        for n, inst in enumerate(insts, 1):
            iid = inst["instance_id"]
            todo = [a for a in arms if (iid, a["tools"], a["index"]) not in done]
            if not todo:
                print(f"[{n}/{len(insts)}] {iid} — skipped (resume)", flush=True)
                continue
            print(f"[{n}/{len(insts)}] {inst['language']:7} {iid}", flush=True)
            for rec in run_instance(inst, args, cache, uv, src, todo):
                out.write(json.dumps(rec) + "\n")
                out.flush()
                label = f"{rec['tools']}{'+index' if rec['index'] else ''}"
                print(
                    f"    {label:16} resolved={str(rec.get('resolved')):5} "
                    f"patched={str(rec.get('patched')):5} "
                    f"exit={str(rec.get('exit_status', '?')):14} "
                    f"turns={rec.get('turns', 0):3} tools={rec.get('tool_calls', 0):3} "
                    f"fails={rec.get('tool_failures', 0):3} "
                    f"{rec.get('total_s', 0):6.0f}s "
                    f"{rec.get('eval_error') or rec.get('setup_error') or ''}",
                    flush=True,
                )
    shutil.rmtree(cache / "task.json", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
