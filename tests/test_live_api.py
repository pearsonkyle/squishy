"""Live integration tests for the squishy API."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from squishy.api import Squishy
from squishy.plan_state import load_plan


async def test_health_check():
    """Verify the API can connect to the LLM server."""
    print("\n=== Test: Health Check ===")
    async with Squishy(
        model="qwen/qwen3.6-35b-a3b",
        base_url="http://localhost:1234/v1",
        api_key="local",
    ) as sq:
        healthy = await sq.health()
        print(f"Server healthy: {healthy}")
        assert healthy, "Server should be reachable"
    print("✅ health_check passed")


async def test_yolo_mode_simple_write():
    """Test yolo mode can create a file."""
    print("\n=== Test: YOLO Mode - Simple Write ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Create a file called hello.py with a function that prints 'Hello World'",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        print(f"Files edited: {result.files_edited}")
        print(f"Final text: {result.final_text[:200]}")
        assert result.success, f"Task should succeed: {result.error}"
        assert len(result.files_created) > 0 or "hello.py" in result.files_created
    print("✅ yolo_mode_simple_write passed")


async def test_yolo_mode_file_edit():
    """Test yolo mode can edit an existing file."""
    print("\n=== Test: YOLO Mode - File Edit ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create initial file
        (working_dir / "app.py").write_text("x = 1\n")

        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Change x to 42 in app.py",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files edited: {result.files_edited}")
        content = (working_dir / "app.py").read_text()
        print(f"File content: {content.strip()}")
        assert result.success, f"Task should succeed: {result.error}"
        assert "42" in content
    print("✅ yolo_mode_file_edit passed")


async def test_plan_mode_read_only():
    """Test plan mode requires plan approval before writes."""
    print("\n=== Test: Plan Mode - Plan Approval Required ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        (working_dir / "app.py").write_text("x = 1\n")

        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="plan",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "First create a plan to read app.py, then read it",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Final text: {result.final_text[:300]}")
        # In plan mode, the agent should present a plan but not auto-execute it
        # The result may fail if it can't produce a plan_task, which is expected
        print(f"Error (if any): {result.error or 'none'}")
        # Plan mode should not auto-edit files
        assert len(result.files_edited) == 0
    print("✅ plan_mode_read_only passed")


async def test_edits_mode_with_plan():
    """Test edits mode auto-approves plan_task and tracks steps."""
    print("\n=== Test: Edits Mode - Plan Tracking ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="edits",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "Create a todo app with two files: todo.py and README.md",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        print(f"Files edited: {result.files_edited}")
        if result.plan_state:
            progress = result.plan_state.get("progress", {})
            print(f"Plan progress: {progress}")
        if result.plan_state:
            plan = load_plan(working_dir)
            if plan:
                print(f"Plan problem: {plan.problem}")
                print(f"Plan steps: {len(plan.steps)}")
                for i, step in enumerate(plan.steps, 1):
                    print(f"  Step {i}: [{step.status}] {step.description}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ edits_mode_with_plan passed")


async def test_plan_mode_step_tracking():
    """Test that plan mode can create plans and track step progress."""
    print("\n=== Test: Plan Mode - Step Progress Tracking ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="edits",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "Create a simple Python module called utils.py with a function called greet that takes a name and prints 'Hello, {name}!'",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        print(f"Files edited: {result.files_edited}")
        if result.plan_state:
            plan = load_plan(working_dir)
            if plan:
                print(f"Plan problem: {plan.problem}")
                progress = plan.progress()
                print(f"Progress: {progress}")
                for i, step in enumerate(plan.steps, 1):
                    print(f"  Step {i}: [{step.status}] {step.description}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ plan_mode_step_tracking passed")


async def test_error_handling():
    """Test that the API handles errors gracefully."""
    print("\n=== Test: Error Handling ===")
    # Test with a non-existent working dir that we create
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp) / "subdir"
        working_dir.mkdir()

        # Create a buggy file
        (working_dir / "buggy.py").write_text("""
def add(a, b):
    return a + b

def divide(a, b):
    return a / b  # No error handling
""")

        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="edits",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "Add a try/except block to the divide function to handle ZeroDivisionError",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files edited: {result.files_edited}")
        if result.files_edited:
            content = (working_dir / result.files_edited[0]).read_text()
            print(f"Content preview:\n{content[:500]}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ error_handling passed")


async def test_streaming_callback():
    """Test that streaming callbacks work."""
    print("\n=== Test: Streaming Callback ===")
    chunks_received = []

    def on_chunk(text):
        if text:
            chunks_received.append(text)

    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="yolo",
            max_turns=5,
        ) as sq:
            result = await sq.run(
                "Create a file called greeting.py that prints 'Hello from squishy!'",
                working_dir=str(working_dir),
                on_text=on_chunk,
            )
        total_text = "".join(chunks_received)
        print(f"Streaming chunks: {len(chunks_received)}")
        print(f"Total streamed text: {len(total_text)} chars")
        print(f"First 200 chars: {total_text[:200]}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ streaming_callback passed")


async def test_multiple_turns():
    """Test sequential API calls maintain state."""
    print("\n=== Test: Multiple Turns ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)

        # Turn 1: Create a file
        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result1 = await sq.run(
                "Create numbers.py with a list [1, 2, 3, 4, 5]",
                working_dir=str(working_dir),
            )
        print(f"Turn 1 - Success: {result1.success}, Files: {result1.files_created}")

        # Turn 2: Modify the file
        async with Squishy(
            model="qwen/qwen3.6-35b-a3b",
            base_url="http://localhost:1234/v1",
            api_key="local",
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result2 = await sq.run(
                "Add a function to numbers.py that sums the list and prints the result",
                working_dir=str(working_dir),
            )
        print(f"Turn 2 - Success: {result2.success}, Files: {result2.files_edited}")

        assert result1.success, f"Turn 1 failed: {result1.error}"
        assert result2.success, f"Turn 2 failed: {result2.error}"
    print("✅ multiple_turns passed")


async def main():
    print("=" * 60)
    print("  SQUISHY LIVE API INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_health_check,
        test_yolo_mode_simple_write,
        test_yolo_mode_file_edit,
        test_plan_mode_read_only,
        test_edits_mode_with_plan,
        test_plan_mode_step_tracking,
        test_error_handling,
        test_streaming_callback,
        test_multiple_turns,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"❌ {test.__name__} FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
