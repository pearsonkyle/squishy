"""Token counting tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from squishy.display import estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_character(self) -> None:
        # 1 char / 4 = 0, but we return max(1, 0) = 1
        assert estimate_tokens("a") == 1

    def test_exact_multiple_of_4(self) -> None:
        # 4 chars = 1 token
        assert estimate_tokens("abcd") == 1

    def test_rounds_up(self) -> None:
        # 5 chars = 1.25, rounds up to 2
        assert estimate_tokens("abcde") == 2

    def test_longer_text(self) -> None:
        # 20 chars = 5 tokens
        assert estimate_tokens("abcdefghijklmnopqrst") == 5

    def test_unicode_handling(self) -> None:
        # Unicode chars count toward length
        assert estimate_tokens("こんにちは") == 2  # 5 chars / 4 = 1.25 -> 2


@pytest.mark.asyncio
class TestAgentTokenCounting:
    async def test_agent_starts_at_zero_before_first_turn(self, tmp_path) -> None:
        """Display reflects real usage from the API, not pre-request estimates."""
        from squishy.agent import Agent
        from squishy.config import Config
        from squishy.display import Display

        class FakeClient:
            async def health(self) -> bool:
                return True

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        _ = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        # No turn has run, so no real usage yet.
        assert display.stats.prompt_tokens == 0
        assert display.stats.completion_tokens == 0

    async def test_agent_uses_real_prompt_tokens_from_api(self, tmp_path) -> None:
        from squishy.agent import Agent
        from squishy.client import CompletionResult
        from squishy.config import Config
        from squishy.display import Display

        class FakeClient:
            async def health(self) -> bool:
                return True

            async def complete(
                self,
                messages: list[dict],
                tools: list[dict],
                *,
                stream: bool = True,
                on_text=None,
            ) -> CompletionResult:
                return CompletionResult(
                    text="done",
                    tool_calls=[],
                    usage={"prompt_tokens": 123, "completion_tokens": 45},
                )

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        agent = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        await agent.run("hello world")

        # prompt_tokens reflects the last turn's real API usage.
        assert display.stats.prompt_tokens == 123
        # completion_tokens accumulate across turns.
        assert display.stats.completion_tokens == 45

        await agent.run("second message")
        # Both reflect the latest run (completion accumulates within a run).
        assert display.stats.prompt_tokens == 123
        assert display.stats.completion_tokens == 45

    async def test_agent_falls_back_to_estimate_when_no_usage(self, tmp_path) -> None:
        """When the provider omits usage, fall back to char-based estimation."""
        from squishy.agent import Agent
        from squishy.client import CompletionResult
        from squishy.config import Config
        from squishy.display import Display

        class FakeClient:
            async def health(self) -> bool:
                return True

            async def complete(
                self,
                messages: list[dict],
                tools: list[dict],
                *,
                stream: bool = True,
                on_text=None,
            ) -> CompletionResult:
                return CompletionResult(text="done", tool_calls=[], usage={})

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        agent = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        await agent.run("hello world")
        # System prompt + user message estimated via char heuristic.
        assert display.stats.prompt_tokens > 0

    async def test_completion_result_has_token_properties(self) -> None:
        from squishy.client import CompletionResult

        result = CompletionResult(
            text="test",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    async def test_completion_result_empty_usage(self) -> None:
        from squishy.client import CompletionResult

        result = CompletionResult(text="test", usage={})

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    async def test_completion_result_none_usage(self) -> None:
        from squishy.client import CompletionResult

        result = CompletionResult(text="test", usage=None)  # type: ignore[arg-type]

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
