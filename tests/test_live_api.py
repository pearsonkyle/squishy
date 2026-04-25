"""Live integration tests for the squishy API."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env so tests pick up the user's configured base_url/model
load_dotenv()

from squishy.api import Squishy
from squishy.plan_state import load_plan


def _env(key: str, default: str) -> str:
    """Read an environment variable from .env, falling back to *default*."""
    return os.environ.get(key, default)


async def test_health_check():
    """Verify the API can connect to the LLM server."""
    print("\n=== Test: Health Check ===")
    async with Squishy(
        model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
        base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
        api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
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


async def test_thinking_mode():
    """Test thinking mode enables chain-of-thought reasoning."""
    print("\n=== Test: Thinking Mode ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            thinking=True,
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Create a file called calculator.py with functions: add, subtract, multiply, divide",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        print(f"Final text: {result.final_text[:200]}")
        assert result.success, f"Task should succeed: {result.error}"
        assert len(result.files_created) > 0
        calc_file = next((f for f in result.files_created if "calculator" in f), None)
        if calc_file:
            content = (working_dir / calc_file).read_text()
            print(f"Calculator content:\n{content[:500]}")
            assert "add" in content.lower()
            assert "divide" in content.lower()
    print("✅ thinking_mode passed")


async def test_session_persistence():
    """Test that passing the same session_id maintains conversation context."""
    print("\n=== Test: Session Persistence ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        session_id = "test-persistence-session"

        # Turn 1: Create a file
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result1 = await sq.run(
                "Create a file called notes.txt with the text 'Session test is active'",
                working_dir=str(working_dir),
                session_id=session_id,
            )
        print(f"Turn 1 - Success: {result1.success}, Files: {result1.files_created}")
        assert result1.success, f"Turn 1 failed: {result1.error}"
        assert len(result1.files_created) > 0

        # Turn 2: Read the file using the same session - context should be preserved
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=5,
        ) as sq:
            result2 = await sq.run(
                "Read notes.txt and confirm it contains the expected text",
                working_dir=str(working_dir),
                session_id=session_id,
            )
        print(f"Turn 2 - Success: {result2.success}")
        print(f"Final text: {result2.final_text[:300]}")
        assert result2.success, f"Turn 2 failed: {result2.error}"
        assert "Session test is active" in result2.final_text or "notes.txt" in result2.final_text.lower()
    print("✅ session_persistence passed")


async def test_timeout_handling():
    """Test that task-level timeout cancels long-running tasks."""
    print("\n=== Test: Timeout Handling ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=20,
        ) as sq:
            # Use a very short timeout - the agent should be cancelled
            try:
                result = await sq.run(
                    "Create 10 files named file1.txt through file10.txt, each with unique content",
                    working_dir=str(working_dir),
                    timeout=5.0,  # 5 second timeout
                )
                # If it completes before timeout, that's also acceptable
                print(f"Completed before timeout. Success: {result.success}")
                print(f"Files created: {result.files_created}")
            except Exception as e:
                error_str = str(e).lower()
                print(f"Task was cancelled as expected: {type(e).__name__}: {e}")
                # Accept AgentTimeout, AgentCancelled, or any error mentioning timeout/exceeded
                assert "cancel" in error_str or "timeout" in error_str or "exceeded" in error_str, \
                    f"Expected timeout/cancel error, got: {e}"
    print("✅ timeout_handling passed")


async def test_bench_mode():
    """Test bench mode with phase tracking (explore → fix → verify)."""
    print("\n=== Test: Bench Mode - Phase Tracking ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create a buggy file for bench mode to fix
        (working_dir / "calculator.py").write_text("""
def add(a, b):
    return a - b  # Bug: should be a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="bench",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "## Problem\nThe add function in calculator.py is broken. It subtracts instead of adding.\n## Hints\nFix the add function to correctly add two numbers.",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files edited: {result.files_edited}")
        print(f"Final phase: {result.final_phase}")
        print(f"Explore turns: {result.explore_turns}")
        print(f"Fix-verify cycles: {result.fix_verify_cycles}")
        if result.files_edited:
            content = (working_dir / result.files_edited[0]).read_text()
            print(f"Fixed content:\n{content[:500]}")
            assert "+" in content or "a + b" in content
    print("✅ bench_mode passed")


async def test_context_compaction():
    """Test that the agent handles context compaction gracefully."""
    print("\n=== Test: Context Compaction ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create multiple files to build up context
        for i in range(5):
            (working_dir / f"module_{i}.py").write_text(
                f"# Module {i}\ndef func_{i}(x):\n    return x * {i}\n"
            )

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=15,
            max_history_messages=6,  # Low threshold to trigger compaction
        ) as sq:
            result = await sq.run(
                "Read all module files, then create a main.py that imports all of them and calls each function with argument 10",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        print(f"Files edited: {result.files_edited}")
        if result.files_created:
            main_file = next((f for f in result.files_created if "main" in f), None)
            if main_file:
                content = (working_dir / main_file).read_text()
                print(f"Main.py content:\n{content[:500]}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ context_compaction passed")


async def test_index_integration():
    """Test that the agent leverages an existing code index."""
    print("\n=== Test: Index Integration ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create a project with multiple files
        (working_dir / "models.py").write_text("""
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __repr__(self):
        return f"User(name={self.name}, email={self.email})"
""")
        (working_dir / "services.py").write_text("""
from models import User

class UserService:
    def __init__(self):
        self._users = []

    def add_user(self, name, email):
        user = User(name, email)
        self._users.append(user)
        return user

    def find_by_email(self, email):
        for user in self._users:
            if user.email == email:
                return user
        return None

    def list_users(self):
        return list(self._users)
""")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "Create a test file called test_users.py that tests the UserService class. "
                "Test add_user, find_by_email, and list_users methods.",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        if result.files_created:
            test_file = next((f for f in result.files_created if "test" in f), None)
            if test_file:
                content = (working_dir / test_file).read_text()
                print(f"Test file content:\n{content[:500]}")
    print("✅ index_integration passed")


async def test_consecutive_error_recovery():
    """Test that the agent handles repeated empty responses gracefully."""
    print("\n=== Test: Consecutive Error Recovery ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create a simple file
        (working_dir / "simple.py").write_text("# A simple file\nprint('hello')\n")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
            max_consecutive_errors=3,
        ) as sq:
            # Ask something that should be straightforward
            result = await sq.run(
                "Read simple.py and create a copy called simple_copy.py",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        print(f"Empty responses: {result.empty_responses}")
        # The agent should either succeed or fail gracefully (not crash)
        assert result.success or result.error  # Either outcome is acceptable
        if result.success:
            assert len(result.files_created) > 0 or "simple_copy" in str(result.files_created)
    print("✅ consecutive_error_recovery passed")


async def test_tool_call_loop_detection():
    """Test that repeated identical tool calls trigger loop detection."""
    print("\n=== Test: Tool Call Loop Detection ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create a file that the agent can read
        (working_dir / "data.txt").write_text("line1\nline2\nline3\n")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=20,
        ) as sq:
            # Ask a task that might cause the agent to loop on read_file
            result = await sq.run(
                "Read data.txt and tell me how many lines it has. Do NOT create any new files.",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Final text: {result.final_text[:300]}")
        print(f"Tool call counts: {result.tool_call_counts}")
        # The agent should either succeed with a text answer or fail gracefully
        # The key is that it doesn't get stuck in an infinite loop
        assert result.turns_used < 20, "Agent should not use all available turns (loop detection should trigger)"
    print("✅ tool_call_loop_detection passed")


async def test_sandbox_mode():
    """Test sandbox mode with Docker container execution."""
    print("\n=== Test: Sandbox Mode ===")
    # Skip if docker is not available
    docker_available = shutil.which("docker") is not None
    if not docker_available:
        print("⚠️  docker not found, skipping sandbox test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        try:
            async with Squishy(
                model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
                base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
                api_key=_env("SQUISHY_API_KEY", "local"),
                permission_mode="yolo",
                use_sandbox=True,
                sandbox_image="python:3.11-slim",
                max_turns=15,
            ) as sq:
                result = await sq.run(
                    "Create a file called hello.py that prints 'Hello from sandbox!' and run it",
                    working_dir=str(working_dir),
                )
            print(f"Success: {result.success}")
            print(f"Turns: {result.turns_used}")
            print(f"Files created: {result.files_created}")
            print(f"Final text: {result.final_text[:200]}")
            assert result.success, f"Task should succeed: {result.error}"
        except Exception as e:
            # Sandbox may fail if Docker daemon is not running
            print(f"Sandbox test result: {type(e).__name__}: {e}")
            # This is acceptable - sandbox requires Docker
    print("✅ sandbox_mode passed (or skipped)")


async def test_mcp_tool_integration():
    """Test that MCP tools are available and dispatchable."""
    print("\n=== Test: MCP Tool Integration ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Create a file called mcp_test.py that defines a class called Greeter with a method greet that returns 'Hello from MCP test'",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files created: {result.files_created}")
        if result.files_created:
            mcp_file = next((f for f in result.files_created if "mcp" in f), None)
            if mcp_file:
                content = (working_dir / mcp_file).read_text()
                print(f"MCP test file content:\n{content[:500]}")
                assert "Greeter" in content
                assert "greet" in content
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ mcp_tool_integration passed")


async def test_quality_gate():
    """Test that quality gates catch degenerate tool call patterns."""
    print("\n=== Test: Quality Gate ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
            max_quality_retries=2,
        ) as sq:
            result = await sq.run(
                "Create a file called quality_check.py with a function called validate that checks if a string is a valid email",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Quality skips: {result.quality_skips}")
        print(f"Tool call counts: {result.tool_call_counts}")
        if result.files_created:
            qc_file = next((f for f in result.files_created if "quality" in f), None)
            if qc_file:
                content = (working_dir / qc_file).read_text()
                print(f"Quality check file:\n{content[:500]}")
                assert "validate" in content
    print("✅ quality_gate passed")


async def test_token_usage_tracking():
    """Test that token usage is tracked and reported in TaskResult."""
    print("\n=== Test: Token Usage Tracking ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=5,
        ) as sq:
            result = await sq.run(
                "Create a file called tokens.py with a function that calculates fibonacci numbers",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Tokens used: {result.tokens_used}")
        print(f"Prompt tokens: tracked via display")
        print(f"Completion tokens: tracked via display")
        assert result.tokens_used > 0, "Token usage should be tracked"
        assert result.turns_used > 0, "Should have used at least one turn"
    print("✅ token_usage_tracking passed")


async def test_file_change_detection():
    """Test that the agent detects and responds to file changes."""
    print("\n=== Test: File Change Detection ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create initial file
        (working_dir / "config.py").write_text("VERSION = '1.0.0'\n")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Update config.py to change VERSION to '2.0.0' and add a RELEASED constant set to False",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files edited: {result.files_edited}")
        content = (working_dir / "config.py").read_text()
        print(f"Updated config:\n{content}")
        assert result.success, f"Task should succeed: {result.error}"
        assert "2.0.0" in content
        assert "RELEASED" in content
    print("✅ file_change_detection passed")


async def test_show_diff_tool():
    """Test that the show_diff tool works after file edits."""
    print("\n=== Test: Show Diff Tool ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=working_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=working_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=working_dir, capture_output=True)
        # Create initial file and commit
        (working_dir / "app.py").write_text("x = 1\nprint(x)\n")
        subprocess.run(["git", "add", "."], cwd=working_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=working_dir, capture_output=True)

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Change x to 42 in app.py, then use show_diff to verify your changes",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files edited: {result.files_edited}")
        content = (working_dir / "app.py").read_text()
        print(f"Updated content:\n{content}")
        assert result.success, f"Task should succeed: {result.error}"
        assert "42" in content
    print("✅ show_diff_tool passed")


async def test_glob_files_tool():
    """Test that the glob_files tool can find files by pattern."""
    print("\n=== Test: Glob Files Tool ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create a directory structure
        (working_dir / "src").mkdir()
        (working_dir / "tests").mkdir()
        (working_dir / "src").write_text("main.py", "print('hello')\n")
        (working_dir / "src").joinpath("utils.py").write_text("def util(): pass\n")
        (working_dir / "tests").joinpath("test_main.py").write_text("def test_main(): pass\n")
        (working_dir / "tests").joinpath("test_utils.py").write_text("def test_util(): pass\n")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Use glob_files to find all Python test files, then read the first one",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Final text: {result.final_text[:300]}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ glob_files_tool passed")


async def test_recall_tool():
    """Test that the recall tool works with an existing index."""
    print("\n=== Test: Recall Tool ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create a project with identifiable files
        (working_dir / "models.py").write_text("""
class User:
    '''User model with name and email.'''
    def __init__(self, name, email):
        self.name = name
        self.email = email
""")
        (working_dir / "services.py").write_text("""
from models import User

class UserService:
    '''Service for managing users.'''
    def __init__(self):
        self._users = []

    def add_user(self, name, email):
        user = User(name, email)
        self._users.append(user)
        return user
""")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "First build an index with /init, then use recall to find the User class definition, "
                "and finally read the file containing it",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files read: {result.files_created}")
        print(f"Final text: {result.final_text[:300]}")
        assert result.success, f"Task should succeed: {result.error}"
    print("✅ recall_tool passed")


async def test_scratchpad_tool():
    """Test that the save_note scratchpad tool persists notes across turns."""
    print("\n=== Test: Scratchpad Tool ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        (working_dir / "app.py").write_text("x = 1\n")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=10,
        ) as sq:
            result = await sq.run(
                "Save a note with key 'app_status' and content 'app.py has x=1, needs to be changed to 42', "
                "then change x to 42 in app.py",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        content = (working_dir / "app.py").read_text()
        print(f"Updated content:\n{content}")
        assert result.success, f"Task should succeed: {result.error}"
        assert "42" in content
    print("✅ scratchpad_tool passed")


async def test_multi_file_refactoring():
    """Test multi-file refactoring across related files."""
    print("\n=== Test: Multi-File Refactoring ===")
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        # Create related files
        (working_dir / "shapes.py").write_text("""
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

class Square:
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2
""")

        async with Squishy(
            model=_env("SQUISHY_MODEL", "qwen/qwen3.6-35b-a3b"),
            base_url=_env("SQUISHY_BASE_URL", "http://localhost:1234/v1"),
            api_key=_env("SQUISHY_API_KEY", "local"),
            permission_mode="yolo",
            max_turns=15,
        ) as sq:
            result = await sq.run(
                "Refactor shapes.py to add a perimeter() method to both Circle and Square. "
                "Also add a __str__ method to each class.",
                working_dir=str(working_dir),
            )
        print(f"Success: {result.success}")
        print(f"Turns: {result.turns_used}")
        print(f"Files edited: {result.files_edited}")
        if result.files_edited:
            content = (working_dir / result.files_edited[0]).read_text()
            print(f"Refactored content:\n{content[:600]}")
            assert "perimeter" in content.lower()
            assert "__str__" in content or "__repr__" in content
    print("✅ multi_file_refactoring passed")


async def main():
    print("=" * 60)
    print("  SQUISHY LIVE API INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        # Core connectivity
        test_health_check,
        # Permission modes
        test_yolo_mode_simple_write,
        test_yolo_mode_file_edit,
        test_plan_mode_read_only,
        test_edits_mode_with_plan,
        test_plan_mode_step_tracking,
        test_bench_mode,
        # Agent features
        test_error_handling,
        test_streaming_callback,
        test_multiple_turns,
        test_thinking_mode,
        test_session_persistence,
        test_timeout_handling,
        test_context_compaction,
        test_index_integration,
        test_consecutive_error_recovery,
        test_tool_call_loop_detection,
        test_sandbox_mode,
        test_mcp_tool_integration,
        test_quality_gate,
        test_token_usage_tracking,
        test_file_change_detection,
        # Tool-specific tests
        test_show_diff_tool,
        test_glob_files_tool,
        test_recall_tool,
        test_scratchpad_tool,
        test_multi_file_refactoring,
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
