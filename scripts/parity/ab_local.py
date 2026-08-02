#!/usr/bin/env python3
"""Container-free A/B over squishy's tool profiles.

`scripts/rebench_container/run_bench.py` is the real evaluation, but it needs
a multi-gigabyte image per instance and therefore a reachable registry. This
answers a narrower question that does not: given the same repository, the same
task and the same model, what does a profile cost in tool calls and tokens,
and does it still land the fix?

It was written to settle a specific question — whether squishy's loop matched
the OpenAI-Agents-SDK reference it was being compared against — and it did
(see scripts/parity/README.md). The SDK arms are gone with the SDK; what
remains is a fast profile comparison that needs nothing but the endpoint.

Each task is a real bug injected into a scratch copy of a real Python package,
with a pytest that fails before the fix and passes after it. That test is the
only oracle — "the model said it fixed it" is not evidence, and on these
models it is frequently wrong.

    python scripts/parity/ab_local.py --model ornith-1.0-35b --seeds 3

`--model` is mandatory: a local endpoint will JIT-load whatever id it is
handed, so there is no default that is safe to guess.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TASK_PROMPT = """Fix the bug described below in this repository.

{problem}

Change the SOURCE CODE — the module that implements the behavior. Editing the
test suite is never a fix. Find the function responsible and edit it.

Reproduce the failure first: the smallest snippet that triggers it, run with
`run_command`. Keep scratch files in /tmp so they stay out of the diff.

Do not stop until you have actually edited a non-test source file.
"""


# --------------------------------------------------------------------- tasks

def _pkg(root: Path) -> Path:
    pkg = root / "shapes"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def _write_common(root: Path) -> None:
    """A small package with real call/inherit structure for the graph to see."""
    pkg = _pkg(root)
    (pkg / "geometry.py").write_text(
        '''"""Area and perimeter primitives."""


def area_rect(width, height):
    """Area of a rectangle."""
    return width * height


def area_circle(radius):
    """Area of a circle."""
    return 3.141592653589793 * radius * radius


def perimeter_rect(width, height):
    """Perimeter of a rectangle."""
    return 2 * (width + height)
''',
        encoding="utf-8",
    )
    (pkg / "shapes.py").write_text(
        '''"""Shape objects built on the geometry primitives."""

from shapes.geometry import area_circle, area_rect, perimeter_rect


class Shape:
    """Base shape."""

    def area(self):
        raise NotImplementedError

    def describe(self):
        """Human-readable summary."""
        return f"{type(self).__name__}(area={self.area()})"


class Rectangle(Shape):
    """An axis-aligned rectangle."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return area_rect(self.width, self.height)

    def perimeter(self):
        return perimeter_rect(self.width, self.height)


class Square(Rectangle):
    """A rectangle with equal sides."""

    def __init__(self, side):
        super().__init__(side, side)


class Circle(Shape):
    """A circle."""

    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return area_circle(self.radius)
''',
        encoding="utf-8",
    )
    (pkg / "report.py").write_text(
        '''"""Aggregate reporting over collections of shapes."""

from shapes.shapes import Shape


def total_area(shapes):
    """Sum the areas of every shape."""
    return sum(s.area() for s in shapes)


def largest(shapes):
    """Return the shape with the greatest area."""
    if not shapes:
        return None
    best = shapes[0]
    for shape in shapes[1:]:
        if shape.area() > best.area():
            best = shape
    return best


def summarize(shapes):
    """One line per shape, plus a total."""
    lines = [s.describe() for s in shapes if isinstance(s, Shape)]
    lines.append(f"total={total_area(shapes)}")
    return "\\n".join(lines)
''',
        encoding="utf-8",
    )


TASKS: list[dict] = [
    {
        "name": "perimeter",
        "problem": (
            "Rectangle(3, 4).perimeter() returns 12, but the perimeter of a "
            "3x4 rectangle is 14. Rectangle.area() is correct, so the bug is "
            "in the perimeter path, not the rectangle itself."
        ),
        "break": ("shapes/geometry.py", "return 2 * (width + height)", "return width + height"),
        "test": (
            "from shapes.shapes import Rectangle, Square\n"
            "def test_perimeter():\n"
            "    assert Rectangle(3, 4).perimeter() == 14\n"
            "    assert Square(5).perimeter() == 20\n"
        ),
    },
    {
        "name": "largest",
        "problem": (
            "report.largest() returns the SMALLEST shape instead of the "
            "largest. largest([Circle(1), Circle(5)]) gives the radius-1 "
            "circle. total_area() over the same list is correct."
        ),
        "break": ("shapes/report.py", "if shape.area() > best.area():", "if shape.area() < best.area():"),
        "test": (
            "from shapes.shapes import Circle\n"
            "from shapes.report import largest\n"
            "def test_largest():\n"
            "    assert largest([Circle(1), Circle(5)]).radius == 5\n"
            "    assert largest([]) is None\n"
        ),
    },
    {
        "name": "square",
        "problem": (
            "Square(4).area() returns 4 instead of 16. Rectangle(4, 4).area() "
            "returns 16 correctly, so only the Square construction path is "
            "affected."
        ),
        "break": ("shapes/shapes.py", "super().__init__(side, side)", "super().__init__(side, 1)"),
        "test": (
            "from shapes.shapes import Square\n"
            "def test_square():\n"
            "    assert Square(4).area() == 16\n"
            "    assert Square(3).perimeter() == 12\n"
        ),
    },
]


def make_workspace(task: dict, base: Path) -> Path:
    """A fresh checkout with the bug injected and its test in place."""
    root = base / task["name"]
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _write_common(root)

    rel, good, bad = task["break"]
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert good in text, f"{task['name']}: anchor not found in {rel}"
    path.write_text(text.replace(good, bad, 1), encoding="utf-8")

    tests = root / "tests"
    tests.mkdir()
    (tests / f"test_{task['name']}.py").write_text(task["test"], encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=ab", "commit", "-qm", "base"],
        cwd=root, check=True,
    )
    from squishy.graph import build_repo_graph

    build_repo_graph(root)
    return root


def grade(root: Path) -> tuple[bool, bool]:
    """``(tests_pass, source_changed)`` — the test is the only oracle."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=root, capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    diff = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    ).stdout
    changed = any(
        line.strip() and ".squishy" not in line and "/tests/" not in line
        for line in diff.splitlines()
    )
    return proc.returncode == 0, changed


# ---------------------------------------------------------------------- arms

async def run_squishy(root: Path, task: dict, args, profile: str) -> dict:
    from squishy.api import Squishy

    counts: dict[str, int] = {}
    usage = {"prompt": 0, "completion": 0}

    def on_event(ev: dict) -> None:
        if ev.get("type") == "tool":
            counts[ev["name"]] = counts.get(ev["name"], 0) + 1
        elif ev.get("type") == "usage":
            usage["prompt"] += ev["prompt_tokens"]
            usage["completion"] += ev["completion_tokens"]

    started = time.time()
    async with Squishy(
        model=args.model, base_url=args.base_url, api_key="local",
        permission_mode="bench", tool_profile=profile,
        max_turns=args.max_turns, use_sandbox=False, save_sessions=False,
    ) as sq:
        res = await sq.run(
            TASK_PROMPT.format(problem=task["problem"]),
            working_dir=str(root), timeout=args.task_timeout, on_event=on_event,
        )
    return {
        "turns": res.turns_used, "tool_calls": sum(counts.values()), "tool_counts": counts,
        "prompt_tokens": usage["prompt"], "completion_tokens": usage["completion"],
        "elapsed_s": round(time.time() - started, 1),
    }


ARMS = {
    "graph": lambda r, t, a: run_squishy(r, t, a, "graph"),
    "minimal": lambda r, t, a: run_squishy(r, t, a, "minimal"),
    "standard": lambda r, t, a: run_squishy(r, t, a, "standard"),
    "shell": lambda r, t, a: run_squishy(r, t, a, "shell"),
}


# ---------------------------------------------------------------------- main

async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="Model id. Required: a local endpoint loads whatever "
                         "id it is given, so there is no safe default.")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--arms", default="graph,minimal")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Repeats per arm. At one seed a difference between "
                         "two arms is indistinguishable from the same arm run "
                         "twice.")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--task-timeout", type=float, default=900.0)
    ap.add_argument("--out", default="/tmp/ab_local.jsonl")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        ap.error(f"unknown arm(s): {sorted(unknown)}; choose from {sorted(ARMS)}")

    out = Path(args.out)
    out.write_text("")
    base = Path(tempfile.mkdtemp(prefix="ab_local_"))
    records: list[dict] = []

    for task in TASKS:
        print(f"\n=== {task['name']} ===")
        for seed in range(args.seeds):
            for arm in arms:
                root = make_workspace(task, base / f"{arm}_{seed}")
                rec = {"task": task["name"], "arm": arm, "seed": seed}
                try:
                    rec.update(await ARMS[arm](root, task, args))
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                passed, changed = grade(root)
                rec["resolved"] = passed and changed
                rec["patched"] = changed
                records.append(rec)
                with out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(
                    f"  {arm:16} seed={seed} resolved={str(rec['resolved']):5} "
                    f"patched={str(rec['patched']):5} "
                    f"turns={rec.get('turns', 0):3} tools={rec.get('tool_calls', 0):3} "
                    f"tok={rec.get('prompt_tokens', 0) + rec.get('completion_tokens', 0):7} "
                    f"{rec.get('elapsed_s', 0):6}s {rec.get('error', '')}"
                )

    print(f"\n=== medians over {len(TASKS)} tasks x {args.seeds} seed(s) ===")
    print(f"{'arm':16} {'resolved':>9} {'patched':>8} {'tools':>6} {'tokens':>8} {'turns':>6}")
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        if not rows:
            continue

        def med(key: str, rows: list[dict] = rows) -> float:
            vals = [r.get(key, 0) for r in rows]
            return statistics.median(vals) if vals else 0

        tok = statistics.median(
            [r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rows]
        )
        print(
            f"{arm:16} {sum(r['resolved'] for r in rows):4}/{len(rows):<4} "
            f"{sum(r['patched'] for r in rows):3}/{len(rows):<4} "
            f"{med('tool_calls'):6.0f} {tok:8.0f} {med('turns'):6.0f}"
        )
    print(f"\nfull records: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
