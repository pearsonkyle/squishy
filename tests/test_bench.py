"""Tests for the bench runner and terminal-bench harness."""
 
from __future__ import annotations
 
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
 
from squishy.agent import TaskResult
from squishy.bench.runner import BenchResult, PredictionWriter, run_batch
from squishy.bench.terminalbench import TerminalTask, load_tasks, run_terminal_task
 
 
@dataclass
class _DummyTask:
    id: str
    delay: float = 0.0
    fail: bool = False
 
 
async def test_run_batch_collects_results_and_writes_jsonl(tmp_path):
    out = tmp_path / "out.jsonl"
    tasks = [_DummyTask(id=f"t{i}") for i in range(5)]
 
    async def runner(t: _DummyTask) -> BenchResult:
        return BenchResult(task_id=t.id, success=True, prediction={"n": int(t.id[1:])})
 
    async with PredictionWriter(out) as writer:
        results = await run_batch(tasks, runner, concurrency=2, writer=writer)
 
    assert len(results) == 5
    assert all(r.success for r in results)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]
    ids = sorted(p["task_id"] for p in parsed)
    assert ids == ["t0", "t1", "t2", "t3", "t4"]
 
 
async def test_run_batch_catches_exceptions():
    tasks = [_DummyTask(id="a"), _DummyTask(id="b", fail=True), _DummyTask(id="c")]
 
    async def runner(t: _DummyTask) -> BenchResult:
        if t.fail:
            raise RuntimeError("boom")
        return BenchResult(task_id=t.id, success=True)
 
    results = await run_batch(tasks, runner, concurrency=3)
    by_id = {r.task_id: r for r in results}
    assert by_id["a"].success
    assert not by_id["b"].success
    assert "boom" in by_id["b"].error
    assert by_id["c"].success
 
 
async def test_run_batch_per_task_timeout():
    async def runner(t: _DummyTask) -> BenchResult:
        await asyncio.sleep(5)
        return BenchResult(task_id=t.id, success=True)
 
    tasks = [_DummyTask(id="slow")]
    results = await run_batch(tasks, runner, concurrency=1, per_task_timeout=0.1)
    assert not results[0].success
    assert "timeout" in results[0].error
 
 
def test_terminal_task_from_dict():
    t = TerminalTask.from_dict(
        {
            "id": "ex",
            "description": "do a thing",
            "verify": "exit 0",
            "setup": ["echo set"],
            "files": {"a.txt": "x"},
            "timeout": 60,
        }
    )
    assert t.id == "ex"
    assert t.verify == "exit 0"
    assert t.setup == ["echo set"]
    assert t.files == {"a.txt": "x"}
    assert t.timeout == 60.0
 
 
def test_load_tasks_jsonl_and_json(tmp_path):
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(
        json.dumps({"id": "a", "description": "d"}) + "\n"
        + json.dumps({"id": "b", "description": "d"}) + "\n"
    )
    got = load_tasks(jsonl)
    assert [t.id for t in got] == ["a", "b"]
 
    jsn = tmp_path / "t.json"
    jsn.write_text(json.dumps([{"id": "c", "description": "d"}]))
    got = load_tasks(jsn)
    assert [t.id for t in got] == ["c"]
 
 
class _FakeSquishy:
    """A Squishy stand-in that reports success without touching the network."""
 
    def __init__(self, *, success: bool = True, final_text: str = "done") -> None:
        self._success = success
        self._final_text = final_text
        self.runs: list[tuple[str, str | None]] = []
 
    async def run(
        self, message: str, *, working_dir: str | None = None, timeout: float | None = None
    ) -> TaskResult:
        self.runs.append((message, working_dir))
        return TaskResult(
            success=self._success,
            final_text=self._final_text,
            turns_used=1,
            elapsed_s=0.01,
            messages=[{"role": "assistant", "content": self._final_text}],
            plan_state={"id": "plan-1", "progress": {"done": 1, "total": 1}},
        )
 
 
async def test_run_terminal_task_verifies_via_shell(tmp_path):
    task = TerminalTask(
        id="probe",
        description="doesn't actually need to run because Squishy is faked",
        verify="test -f marker",
        files={"marker": "ok"},
    )
    fake = _FakeSquishy(success=True)
    result = await run_terminal_task(task, squishy=fake, workspace_root=tmp_path)  # type: ignore[arg-type]
    assert result.success
    assert result.prediction["verify_exit_code"] == 0
    assert "marker" in result.prediction["task_id"] or result.task_id == "probe"
    assert result.prediction["plan_state"]["id"] == "plan-1"
    assert result.artifacts["transcript"][0]["content"] == "done"
    assert result.artifacts["verify_result"]["exit_code"] == 0
 
 
async def test_run_terminal_task_verify_failure(tmp_path):
    task = TerminalTask(
        id="v-fail",
        description="task",
        verify="exit 1",
    )
    fake = _FakeSquishy(success=True)
    result = await run_terminal_task(task, squishy=fake, workspace_root=tmp_path)  # type: ignore[arg-type]
    assert not result.success
    assert result.prediction["verify_exit_code"] == 1
    assert "verify failed" in result.error
    assert result.artifacts["workspace_snapshot"] == []
 
 
async def test_prediction_writer_records_errors(tmp_path):
    out = tmp_path / "p.jsonl"
    async with PredictionWriter(out) as w:
        await w.write({"task_id": "a", "ok": True})
        await w.write({"task_id": "b", "error": "nope"})
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["task_id"] == "a"
    assert lines[1]["error"] == "nope"
 
 
def test_path_types(tmp_path: Path) -> None:
    # Sanity: PredictionWriter accepts both str and Path
    p1 = PredictionWriter(str(tmp_path / "x.jsonl"))
    p1.close()
    p2 = PredictionWriter(tmp_path / "y.jsonl")
    p2.close()


def test_plan_adherence_no_plan():
    from squishy.bench.terminalbench import _plan_adherence

    assert _plan_adherence(None) == {"had_plan": False}
    assert _plan_adherence({}) == {"had_plan": False}


def test_plan_adherence_computes_completion_ratio():
    from squishy.bench.terminalbench import _plan_adherence

    plan_state = {
        "approved": True,
        "progress": {"done": 3, "total": 4, "blocked": 0, "skipped": 0, "pending": 1, "in_progress": 0},
    }
    result = _plan_adherence(plan_state)
    assert result["had_plan"] is True
    assert result["approved"] is True
    assert result["total_steps"] == 4
    assert result["done"] == 3
    assert result["completion_ratio"] == 0.75


def test_tool_counts_parses_transcript():
    from squishy.bench.terminalbench import _tool_counts

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "..."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "plan_task", "arguments": "{}"}},
            ],
        },
    ]
    assert _tool_counts(messages) == {"read_file": 2, "plan_task": 1}
    assert _tool_counts(None) == {}


def test_classify_error_maps_common_cases():
    from squishy.bench.terminalbench import _classify_error

    ok = TaskResult(success=True)
    assert _classify_error(ok, verified=True) == "ok"
    assert _classify_error(ok, verified=False) == "verify_failed"

    tr = TaskResult(success=False, error="max turns (30) reached")
    assert _classify_error(tr, verified=True) == "max_turns"

    tr = TaskResult(success=False, error="3 consecutive tool failures — stopping.")
    assert _classify_error(tr, verified=True) == "tool_error_loop"

    tr = TaskResult(success=False, error="plan-mode run finished without producing a plan_task")
    assert _classify_error(tr, verified=True) == "plan_mode_no_plan"
