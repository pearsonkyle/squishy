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
    async def test_agent_counts_system_prompt_tokens(self, tmp_path) -> None:
        from squishy.agent import Agent
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
                    usage={"prompt_tokens": 50, "completion_tokens": 10},
                )

        from dataclasses import dataclass

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        agent = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        assert display.stats.prompt_tokens > 0  # System prompt was counted

    async def test_agent_counts_user_message_tokens(self, tmp_path) -> None:
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
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                )

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        agent = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        initial_tokens = display.stats.prompt_tokens
        await agent.run("hello world")

        # After run, prompt tokens should include system + user message
        assert display.stats.prompt_tokens > initial_tokens

        # Run again to verify user messages are counted each time
        initial_tokens = display.stats.prompt_tokens
        await agent.run("second message")
        assert display.stats.prompt_tokens > initial_tokens

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
