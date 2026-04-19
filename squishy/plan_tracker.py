"""Live tracking of implementation steps after a plan is approved.

A single PlanTracker instance is owned by Display and shared through ToolContext
so the exit_plan_mode tool can populate it on approval, and update_plan_step
can mark progress as the agent executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STEP_STATUSES = ("pending", "in_progress", "done", "skipped")


@dataclass
class PlanStep:
    index: int
    description: str
    status: str = "pending"


@dataclass
class PlanTracker:
    plan: str = ""
    problem: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    approved: bool = False

    def populate(self, plan: str, problem: str, descriptions: list[str]) -> None:
        self.plan = plan
        self.problem = problem
        self.steps = [
            PlanStep(index=i, description=d, status="pending")
            for i, d in enumerate(descriptions)
        ]
        self.approved = True

    def reset(self) -> None:
        self.plan = ""
        self.problem = ""
        self.steps = []
        self.approved = False

    def set_status(self, index: int, status: str) -> tuple[bool, str]:
        if status not in STEP_STATUSES:
            return False, f"invalid status: {status} (expected one of {STEP_STATUSES})"
        for s in self.steps:
            if s.index == index:
                s.status = status
                return True, ""
        return False, f"no step with index {index}"

    def is_active(self) -> bool:
        return self.approved and bool(self.steps)

    def current_step(self) -> PlanStep | None:
        for s in self.steps:
            if s.status == "in_progress":
                return s
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def render_block(self) -> str:
        """Compact plaintext for injection into the system prompt."""
        if not self.is_active():
            return ""
        symbol = {"done": "[x]", "in_progress": "[>]", "pending": "[ ]", "skipped": "[-]"}
        lines = ["## Active plan"]
        if self.plan:
            lines.append(self.plan)
        lines.append("")
        for s in self.steps:
            lines.append(f"  {symbol.get(s.status, '[?]')} {s.index}. {s.description}")
        lines.append("")
        lines.append(
            "Work on the step marked `[>]` (or the first `[ ]`). "
            "Call `update_plan_step(index=N, status=\"in_progress\")` when starting a step "
            "and `status=\"done\"` when it is fully complete."
        )
        return "\n".join(lines)
