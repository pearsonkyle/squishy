"""Typed plan state plus persistence helpers."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PLAN_DIR = ".squishy"
PLAN_FILE = "active_plan.json"
STEP_STATUSES = frozenset({"pending", "in-progress", "done", "skipped", "blocked"})
RESOLVED_STEP_STATUSES = frozenset({"done", "skipped"})


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "path": self.path,
            "command": self.command,
            "exit_code": self.exit_code,
            "detail": self.detail,
        }

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
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "note": self.note,
            "evidence": [item.to_dict() for item in self.evidence],
        }

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
        return {
            "id": self.id,
            "plan": self.plan,
            "problem": self.problem,
            "solution": self.solution,
            "steps": [step.to_dict() for step in self.steps],
            "files_to_create": self.files_to_create,
            "files_to_modify": self.files_to_modify,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "approved": self.approved,
            "approved_at": self.approved_at,
            "progress": self.progress(),
        }

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
    "PlanEvidence",
    "PlanState",
    "PlanStep",
    "STEP_STATUSES",
    "clear_plan",
    "has_plan_file",
    "load_plan",
    "plan_path",
    "save_plan",
]
