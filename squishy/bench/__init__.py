"""Benchmark harnesses for squishy.
 
Two targets supported:
 
- SWE-bench (Python project bug-fix): see ``swebench.py``
- Terminal-bench (CLI task completion): see ``terminalbench.py``
 
Both share a generic async batch runner (``runner.py``) that caps concurrency,
times each task out, and streams predictions to disk. Evaluation itself
(running tests / verifying) is delegated to the upstream harness.
"""
 
from squishy.bench.runner import BenchTask, PredictionWriter, run_batch
from squishy.bench.swebench import run_swebench_instance
from squishy.bench.terminalbench import (
    LocalTerminalBackend,
    ShellResult,
    TerminalBackend,
    TerminalTask,
    run_terminal_task,
)
 
__all__ = [
    "BenchTask",
    "LocalTerminalBackend",
    "PredictionWriter",
    "ShellResult",
    "TerminalBackend",
    "TerminalTask",
    "run_batch",
    "run_swebench_instance",
    "run_terminal_task",
]
