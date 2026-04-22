"""Tests for ToolContext state management and persistence."""

from __future__ import annotations

from squishy.tools.base import ToolContext
from squishy.plan_state import PlanState, PlanStep


class TestToolContextInitialization:
    """Test ToolContext initialization and default values."""

    def test_default_values(self, tmp_path):
        """Verify default values are properly set."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )

        assert ctx.working_dir == str(tmp_path)
        assert ctx.permission_mode == "edits"
        assert ctx.use_sandbox is False
        assert ctx.sandbox_image == "python:3.11-slim"

    def test_empty_files_read_by_default(self, tmp_path):
        """Verify files_read is an empty dict by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.files_read == {}
        assert isinstance(ctx.files_read, dict)

    def test_empty_files_read_meta_by_default(self, tmp_path):
        """Verify files_read_meta is an empty dict by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.files_read_meta == {}
        assert isinstance(ctx.files_read_meta, dict)

    def test_empty_pending_plan_evidence_by_default(self, tmp_path):
        """Verify pending_plan_evidence is an empty list by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.pending_plan_evidence == []
        assert isinstance(ctx.pending_plan_evidence, list)

    def test_plan_is_none_by_default(self, tmp_path):
        """Verify plan is None by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.plan is None

    def test_plan_switch_prompted_false_by_default(self, tmp_path):
        """Verify plan_switch_prompted is False by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.plan_switch_prompted is False


class TestToolContextFilesRead:
    """Test files_read dictionary behavior."""

    def test_add_entries(self, tmp_path):
        """Verify entries can be added to files_read."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.files_read["test.py"] = "print('hello')\n"
        assert "test.py" in ctx.files_read
        assert ctx.files_read["test.py"] == "print('hello')\n"

    def test_overwrite_entries(self, tmp_path):
        """Verify entries can be overwritten."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.files_read["test.py"] = "version1\n"
        ctx.files_read["test.py"] = "version2\n"
        assert ctx.files_read["test.py"] == "version2\n"

    def test_remove_entries(self, tmp_path):
        """Verify entries can be removed."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.files_read["test.py"] = "content\n"
        del ctx.files_read["test.py"]
        assert "test.py" not in ctx.files_read


class TestToolContextFilesReadMeta:
    """Test files_read_meta dictionary behavior."""

    def test_cache_key_format(self, tmp_path):
        """Verify cache keys are tuples (path, offset, limit)."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        # Example cache key
        cache_key = ("test.py", 0, None)
        ctx.files_read_meta[cache_key] = {
            "content": "test",
            "total_lines": 10,
            "returned_lines": 5,
        }
        assert cache_key in ctx.files_read_meta

    def test_multiple_cache_entries(self, tmp_path):
        """Verify multiple cache entries for same file with different offsets."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        # Same file, different offsets
        ctx.files_read_meta[("test.py", 0, None)] = {"total_lines": 10}
        ctx.files_read_meta[("test.py", 5, None)] = {"total_lines": 5}
        ctx.files_read_meta[("test.py", 0, 3)] = {"total_lines": 3}

        assert len(ctx.files_read_meta) == 3

    def test_cache_invalidation_by_path(self, tmp_path):
        """Verify cache can be invalidated by path."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        # Add some cache entries
        ctx.files_read_meta[("test.py", 0, None)] = {"content": "a"}
        ctx.files_read_meta[("other.py", 0, None)] = {"content": "b"}

        # Invalidate test.py
        for key in [k for k in ctx.files_read_meta if k[0] == "test.py"]:
            del ctx.files_read_meta[key]

        assert ("test.py", 0, None) not in ctx.files_read_meta
        assert ("other.py", 0, None) in ctx.files_read_meta


class TestToolContextPendingPlanEvidence:
    """Test pending_plan_evidence list behavior."""

    def test_append_evidence(self, tmp_path):
        """Verify evidence can be appended to the list."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        evidence = {
            "kind": "edit_file",
            "path": "app.py",
            "detail": "patched bug",
        }
        ctx.pending_plan_evidence.append(evidence)
        assert len(ctx.pending_plan_evidence) == 1
        assert ctx.pending_plan_evidence[0]["path"] == "app.py"

    def test_clear_evidence(self, tmp_path):
        """Verify evidence list can be cleared."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.pending_plan_evidence.append({"kind": "read_file", "path": "a.py"})
        ctx.pending_plan_evidence.append({"kind": "read_file", "path": "b.py"})
        assert len(ctx.pending_plan_evidence) == 2

        ctx.pending_plan_evidence.clear()
        assert len(ctx.pending_plan_evidence) == 0


class TestToolContextPlan:
    """Test plan attribute behavior."""

    def test_set_plan(self, tmp_path):
        """Verify a PlanState can be assigned to ctx.plan."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        plan = PlanState.create(
            problem="Test problem",
            solution="Test solution",
            steps=["Step 1", "Step 2"],
        )
        ctx.plan = plan
        assert ctx.plan is not None
        assert ctx.plan.problem == "Test problem"
        assert len(ctx.plan.steps) == 2

    def test_plan_progress(self, tmp_path):
        """Verify plan progress tracking."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        plan = PlanState.create(
            problem="Test",
            solution="Solution",
            steps=["Step 1", "Step 2", "Step 3"],
        )
        ctx.plan = plan

        progress = plan.progress()
        assert progress["total"] == 3
        assert progress["pending"] == 3
        assert progress["done"] == 0

    def test_plan_step_update(self, tmp_path):
        """Verify plan steps can be updated."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        plan = PlanState.create(
            problem="Test",
            solution="Solution",
            steps=["Step 1", "Step 2"],
        )
        ctx.plan = plan

        # Update step 0 to done
        plan.update_step(step_index=0, status="done")
        assert plan.steps[0].status == "done"
        assert plan.steps[1].status == "pending"

    def test_plan_unresolved_steps(self, tmp_path):
        """Verify unresolved_steps returns correct steps."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        plan = PlanState.create(
            problem="Test",
            solution="Solution",
            steps=["Step 1", "Step 2", "Step 3"],
        )
        ctx.plan = plan

        # Initially all are unresolved
        assert len(plan.unresolved_steps()) == 3

        # Mark some as done
        plan.update_step(step_index=0, status="done")
        plan.update_step(step_index=1, status="skipped")

        # Only step 2 should be unresolved
        unresolved = plan.unresolved_steps()
        assert len(unresolved) == 1
        assert unresolved[0].description == "Step 3"


class TestToolContextPermissionMode:
    """Test permission_mode switching and behavior."""

    def test_mode_switch_edits_to_plan(self, tmp_path):
        """Verify mode can switch from edits to plan."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.permission_mode == "edits"

        ctx.permission_mode = "plan"
        assert ctx.permission_mode == "plan"

    def test_mode_switch_plan_to_yolo(self, tmp_path):
        """Verify mode can switch from plan to yolo."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="plan",
            use_sandbox=False,
        )
        assert ctx.permission_mode == "plan"

        ctx.permission_mode = "yolo"
        assert ctx.permission_mode == "yolo"


class TestToolContextSandbox:
    """Test sandbox configuration."""

    def test_sandbox_enabled(self, tmp_path):
        """Verify use_sandbox can be enabled."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=True,
        )
        assert ctx.use_sandbox is True
        assert ctx.sandbox_image == "python:3.11-slim"

    def test_sandbox_disabled(self, tmp_path):
        """Verify use_sandbox can be disabled."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.use_sandbox is False

    def test_custom_sandbox_image(self, tmp_path):
        """Verify custom sandbox image can be set."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=True,
            sandbox_image="custom:latest",
        )
        assert ctx.sandbox_image == "custom:latest"


class TestToolContextPersistence:
    """Test context state across operations."""

    def test_context_survives_multiple_operations(self, tmp_path):
        """Verify context maintains state across multiple tool calls."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )

        # Simulate multiple operations
        for i in range(5):
            ctx.files_read[f"file{i}.py"] = f"# file {i}\n"
            ctx.pending_plan_evidence.append(
                {"kind": "read_file", "path": f"file{i}.py"}
            )

        # Verify all state persisted
        assert len(ctx.files_read) == 5
        assert len(ctx.pending_plan_evidence) == 5

    def test_context_cleared_on_new_instance(self, tmp_path):
        """Verify new context instances start fresh."""
        ctx1 = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx1.files_read["test.py"] = "content\n"
        ctx1.pending_plan_evidence.append({"kind": "read_file"})

        # Create new instance
        ctx2 = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )

        # Should be empty
        assert ctx2.files_read == {}
        assert ctx2.pending_plan_evidence == []
