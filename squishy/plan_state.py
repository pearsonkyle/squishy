"""Typed plan state plus persistence helpers."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PLAN_DIR = ".squishy"
PLAN_FILE = "active_plan.json"
STEP_STATUSES = frozenset({"pending", "in-progress", "done", "skipped", "blocked"})
RESOLVED_STEP_STATUSES = frozenset({"done", "skipped"})

STATUS_ICONS: dict[str, str] = {
    "done": "✓",
    "in-progress": "▶",
    "skipped": "—",
    "pending": "○",
    "blocked": "!",
}

PLAN_STATUS_OPEN_TAG = "<plan-status>"
PLAN_STATUS_CLOSE_TAG = "</plan-status>"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class PlanEvidence:
    kind: str
    timestamp: str = field(default_factory=utc_now)
    path: str = ""
    command: str = ""
    exit_code: int | None = None
    detail: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanEvidence:
        return cls(
            kind=str(data.get("kind", "")),
            timestamp=str(data.get("timestamp") or utc_now()),
            path=str(data.get("path", "")),
            command=str(data.get("command", "")),
            exit_code=int(data["exit_code"]) if isinstance(data.get("exit_code"), int) else None,
            detail=str(data.get("detail", "")),
        )


@dataclass
class PlanStep:
    id: str
    description: str
    status: str = "pending"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    note: str = ""
    evidence: list[PlanEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            id=str(data.get("id", "")),
            description=str(data.get("description", "")),
            status=str(data.get("status", "pending")),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            completed_at=str(data.get("completed_at")) if data.get("completed_at") else None,
            note=str(data.get("note", "")),
            evidence=[
                PlanEvidence.from_dict(item)
                for item in data.get("evidence", [])
                if isinstance(item, dict)
            ],
        )

    def apply(
        self,
        *,
        status: str,
        note: str = "",
        evidence: list[PlanEvidence] | None = None,
    ) -> None:
        self.status = status
        self.updated_at = utc_now()
        if note:
            self.note = note
        if evidence:
            self.evidence.extend(evidence)
        if status in RESOLVED_STEP_STATUSES:
            self.completed_at = self.updated_at
        else:
            self.completed_at = None


@dataclass
class PlanState:
    id: str
    problem: str
    solution: str
    steps: list[PlanStep]
    plan: str = ""
    files_to_create: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    approved: bool = False
    approved_at: str | None = None

    @classmethod
    def create(
        cls,
        *,
        problem: str,
        solution: str,
        steps: list[str],
        plan: str = "",
        files_to_create: list[str] | None = None,
        files_to_modify: list[str] | None = None,
    ) -> PlanState:
        now = utc_now()
        return cls(
            id=f"plan-{uuid.uuid4().hex[:12]}",
            plan=plan,
            problem=problem,
            solution=solution,
            steps=[
                PlanStep(id=f"step-{i + 1}", description=step, created_at=now, updated_at=now)
                for i, step in enumerate(steps)
            ],
            files_to_create=list(files_to_create or []),
            files_to_modify=list(files_to_modify or []),
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["progress"] = self.progress()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanState:
        return cls(
            id=str(data.get("id") or f"plan-{uuid.uuid4().hex[:12]}"),
            plan=str(data.get("plan", "")),
            problem=str(data.get("problem", "")),
            solution=str(data.get("solution", "")),
            steps=[
                PlanStep.from_dict(item)
                for item in data.get("steps", [])
                if isinstance(item, dict)
            ],
            files_to_create=[str(item) for item in data.get("files_to_create", [])],
            files_to_modify=[str(item) for item in data.get("files_to_modify", [])],
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            approved=bool(data.get("approved", False)),
            approved_at=str(data.get("approved_at")) if data.get("approved_at") else None,
        )

    def progress(self) -> dict[str, int]:
        total = len(self.steps)
        return {
            "done": sum(1 for step in self.steps if step.status == "done"),
            "in_progress": sum(1 for step in self.steps if step.status == "in-progress"),
            "blocked": sum(1 for step in self.steps if step.status == "blocked"),
            "pending": sum(1 for step in self.steps if step.status == "pending"),
            "skipped": sum(1 for step in self.steps if step.status == "skipped"),
            "total": total,
        }

    def unresolved_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.status not in RESOLVED_STEP_STATUSES]

    def mark_approved(self) -> None:
        self.approved = True
        self.approved_at = utc_now()
        self.updated_at = self.approved_at

    def update_step(
        self,
        *,
        step_index: int,
        status: str,
        note: str = "",
        evidence: list[PlanEvidence] | None = None,
    ) -> PlanStep:
        step = self.steps[step_index]
        step.apply(status=status, note=note, evidence=evidence)
        self.updated_at = utc_now()
        return step


def render_plan_status(plan: PlanState, *, step_desc_chars: int = 160) -> str:
    """Render the plan as a compact, re-injectable status block.

    The block is wrapped in ``<plan-status>``/``</plan-status>`` so the agent
    loop can strip any prior copy before injecting the fresh one each turn.
    Kept small (~200-400 tokens for typical plans) so re-injection is cheap.
    """
    lines: list[str] = [PLAN_STATUS_OPEN_TAG]
    title = plan.plan or plan.problem
    approved_marker = " [approved]" if plan.approved else " [proposed]"
    lines.append(f"plan: {title} ({plan.id}){approved_marker}")
    if plan.problem and plan.problem != title:
        lines.append(f"problem: {plan.problem}")
    if plan.solution:
        lines.append(f"solution: {plan.solution}")
    if plan.files_to_modify:
        lines.append(f"files_to_modify: {', '.join(plan.files_to_modify)}")
    if plan.files_to_create:
        lines.append(f"files_to_create: {', '.join(plan.files_to_create)}")
    lines.append("steps:")
    for i, step in enumerate(plan.steps, 1):
        icon = STATUS_ICONS.get(step.status, "?")
        desc = step.description or ""
        if len(desc) > step_desc_chars:
            desc = desc[: step_desc_chars - 1] + "…"
        suffix = ""
        if step.status == "blocked" and step.note:
            suffix = f" (blocked: {step.note[:80]})"
        elif step.note and step.status == "in-progress":
            suffix = f" (note: {step.note[:80]})"
        lines.append(f"  [{icon}] {i}. {desc}{suffix}")
    progress = plan.progress()
    lines.append(
        f"progress: {progress['done']}/{progress['total']} done, "
        f"{progress['in_progress']} in-progress, "
        f"{progress['blocked']} blocked, "
        f"{progress['pending']} pending, "
        f"{progress['skipped']} skipped"
    )
    lines.append(PLAN_STATUS_CLOSE_TAG)
    return "\n".join(lines)


def is_plan_status_message(message: dict[str, Any]) -> bool:
    """Return True if ``message`` is a previously-injected plan-status block."""
    if message.get("role") != "system":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    return content.startswith(PLAN_STATUS_OPEN_TAG)


def plan_dir(cwd: str | os.PathLike[str]) -> Path:
    return Path(cwd) / PLAN_DIR


def plan_path(cwd: str | os.PathLike[str]) -> Path:
    return plan_dir(cwd) / PLAN_FILE


def has_plan_file(cwd: str | os.PathLike[str]) -> bool:
    return plan_path(cwd).is_file()


def save_plan(cwd: str | os.PathLike[str], plan: PlanState) -> Path:
    path = plan_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_plan(cwd: str | os.PathLike[str]) -> PlanState | None:
    path = plan_path(cwd)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return PlanState.from_dict(raw)


def clear_plan(cwd: str | os.PathLike[str]) -> None:
    path = plan_path(cwd)
    try:
        path.unlink()
    except FileNotFoundError:
        return


__all__ = [
    "PLAN_DIR",
    "PLAN_FILE",
    "PLAN_STATUS_CLOSE_TAG",
    "PLAN_STATUS_OPEN_TAG",
    "PlanEvidence",
    "PlanState",
    "PlanStep",
    "STATUS_ICONS",
    "STEP_STATUSES",
    "clear_plan",
    "has_plan_file",
    "is_plan_status_message",
    "load_plan",
    "plan_path",
    "render_plan_status",
    "save_plan",
]
