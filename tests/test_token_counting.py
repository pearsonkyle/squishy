"""Token counting tests."""

from __future__ import annotations

from squishy.display import estimate_tokens


class TestEstimateTokens:
    """Tests for estimate_tokens: ceil(len/3.5) + 4 overhead."""

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_character(self) -> None:
        # ceil(1/3.5) + 4 = 1 + 4 = 5
        assert estimate_tokens("a") == 5

    def test_short_text(self) -> None:
        # ceil(7/3.5) + 4 = 2 + 4 = 6
        assert estimate_tokens("abcdefg") == 6

    def test_longer_text(self) -> None:
        # ceil(20/3.5) + 4 = 6 + 4 = 10
        assert estimate_tokens("abcdefghijklmnopqrst") == 10

    def test_unicode_handling(self) -> None:
        # ceil(5/3.5) + 4 = 2 + 4 = 6
        assert estimate_tokens("こんにちは") == 6

    def test_overhead_is_per_message(self) -> None:
        # Two separate calls should each include 4-token overhead
        t1 = estimate_tokens("hello")
        t2 = estimate_tokens("world")
        combined = estimate_tokens("helloworld")
        # combined has only one 4-token overhead, two separate have 8 total
        assert t1 + t2 > combined


class TestAgentTokenCounting:
    async def test_agent_counts_system_prompt_tokens(self, tmp_path) -> None:
        from squishy.agent import Agent
        from squishy.config import Config
        from squishy.display import Display

        from squishy.client import CompletionResult

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
                on_retry=None,
            ) -> CompletionResult:
                return CompletionResult(
                    text="done",
                    tool_calls=[],
                    usage={"prompt_tokens": 50, "completion_tokens": 10},
                )

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        agent = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        assert display.stats.prompt_tokens > 0  # System prompt was counted

    async def test_agent_uses_api_prompt_tokens(self, tmp_path) -> None:
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
                on_retry=None,
            ) -> CompletionResult:
                return CompletionResult(
                    text="done",
                    tool_calls=[],
                    usage={"prompt_tokens": 100, "completion_tokens": 5},
                )

        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "yolo"
        cfg.max_turns = 5

        display = Display()
        agent = Agent(cfg, FakeClient(), display)  # type: ignore[arg-type]

        result = await agent.run("hello world")

        # After run, prompt tokens come from the API response (100 per call).
        assert display.stats.prompt_tokens == 100
        assert result.tokens_used == 105  # 100 prompt + 5 completion

        # Run again: tokens accumulate across runs via _LoopState per-run.
        result2 = await agent.run("second message")
        assert result2.tokens_used == 105  # fresh _LoopState per run

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
