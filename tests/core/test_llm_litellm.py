"""Tests for LiteLLM provider client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from q_ai.core.llm import (
    NormalizedResponse,
    ProviderError,
    ToolSpec,
    UnsupportedCapabilityError,
)
from q_ai.core.llm_litellm import (
    LiteLLMClient,
    _tool_spec_to_openai_format,
    complete_text,
    get_litellm_context_window,
    stream_text,
)


class TestToolSpecToOpenaiFormat:
    """Tests for _tool_spec_to_openai_format()."""

    def test_converts_correctly(self) -> None:
        spec = ToolSpec(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        result = _tool_spec_to_openai_format(spec)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather"
        assert result["function"]["parameters"]["properties"]["city"]["type"] == "string"


class TestLiteLLMClient:
    """Tests for LiteLLMClient.complete()."""

    async def test_complete_with_tool_calls(self) -> None:
        """Mock response with tool_calls returns correct NormalizedResponse."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call-weather"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"city": "London"})

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = ""

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {"id": "resp-1"}

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            client = LiteLLMClient()
            result = await client.complete(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "test"}],
                tools=[ToolSpec(name="get_weather", description="Get weather", parameters={})],
            )

        assert isinstance(result, NormalizedResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "London"}
        assert result.tool_calls[0].id == "call-weather"
        assert result.finish_reason == "tool_calls"
        assert result.content == ""

    async def test_complete_no_tool_calls(self) -> None:
        """Mock response with content only returns empty tool_calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "I cannot use that tool."

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {"id": "resp-1"}

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            client = LiteLLMClient()
            result = await client.complete(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

        assert result.tool_calls == []
        assert result.content == "I cannot use that tool."
        assert result.finish_reason == "stop"

    async def test_complete_passes_explicit_target_configuration(self) -> None:
        """Profile endpoint, key, and fixed parameters are passed directly."""
        mock_message = MagicMock(tool_calls=None, content="complete")
        mock_choice = MagicMock(message=mock_message, finish_reason="stop")
        mock_response = MagicMock(choices=[mock_choice])
        mock_response.model_dump.return_value = {"id": "resp-1"}

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            client = LiteLLMClient(
                api_base="https://models.example.test/v1",
                api_key="secret-key",
                generation_parameters={"temperature": 0.0, "seed": 7},
            )
            await client.complete(
                model="openai/model-a",
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                max_tokens=321,
            )

        kwargs = mock_acomp.await_args.kwargs
        assert kwargs["api_base"] == "https://models.example.test/v1"
        assert kwargs["api_key"] == "secret-key"
        assert kwargs["max_tokens"] == 321
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == 7

    async def test_complete_provider_error(self) -> None:
        """acompletion exception raises ProviderError."""
        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.side_effect = Exception("Rate limited")
            client = LiteLLMClient()
            with pytest.raises(ProviderError, match="Provider anthropic error"):
                await client.complete(
                    model="anthropic/claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "test"}],
                    tools=[],
                )

    async def test_complete_unsupported_capability(self) -> None:
        """Tool support error raises UnsupportedCapabilityError."""
        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.side_effect = Exception("This model does not support tools")
            client = LiteLLMClient()
            with pytest.raises(UnsupportedCapabilityError, match="does not support tool calling"):
                await client.complete(
                    model="ollama/llama3",
                    messages=[{"role": "user", "content": "test"}],
                    tools=[ToolSpec(name="t", description="t", parameters={})],
                )

    async def test_complete_string_arguments_parsed(self) -> None:
        """Tool call arguments arriving as strings are parsed through json.loads."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "get_data"
        mock_tool_call.function.arguments = '{"key": "value"}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {}

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            client = LiteLLMClient()
            result = await client.complete(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                tools=[ToolSpec(name="get_data", description="d", parameters={})],
            )

        assert result.tool_calls[0].arguments == {"key": "value"}
        assert result.content == ""

    async def test_complete_malformed_tool_arguments(self) -> None:
        """Malformed JSON in tool arguments raises ProviderError."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "get_data"
        mock_tool_call.function.arguments = "not valid json{{"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {}

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            client = LiteLLMClient()
            with pytest.raises(ProviderError, match="Malformed tool call arguments"):
                await client.complete(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                    tools=[ToolSpec(name="get_data", description="d", parameters={})],
                )


class TestBoundaryHelpers:
    """Tests for assistant-facing LiteLLM boundary helpers."""

    async def test_complete_text_forwards_optional_kwargs(self) -> None:
        """complete_text() should route completion through LiteLLM with auth overrides."""
        mock_message = MagicMock()
        mock_message.content = "assistant reply"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await complete_text(
                model="ollama/llama3.1",
                messages=[{"role": "user", "content": "test"}],
                api_base="http://localhost:11434",
                api_key="secret",
            )

        assert result == "assistant reply"
        mock_acomp.assert_awaited_once_with(
            model="ollama/llama3.1",
            messages=[{"role": "user", "content": "test"}],
            timeout=120.0,
            api_base="http://localhost:11434",
            api_key="secret",
        )

    async def test_complete_text_returns_empty_string_for_empty_choices(self) -> None:
        """complete_text() should return empty text for empty choice lists."""
        mock_response = MagicMock()
        mock_response.choices = []

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await complete_text("openai/gpt-4o", [])

        assert result == ""

    async def test_complete_text_returns_empty_string_when_message_missing(self) -> None:
        """complete_text() should return empty text when the message is missing."""
        mock_choice = MagicMock()
        mock_choice.message = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await complete_text("openai/gpt-4o", [])

        assert result == ""

    async def test_complete_text_returns_empty_string_when_content_missing(self) -> None:
        """complete_text() should return empty text when the content is missing."""
        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await complete_text("openai/gpt-4o", [])

        assert result == ""

    async def test_stream_text_yields_non_empty_chunks(self) -> None:
        """stream_text() should yield only chunks with content."""

        async def fake_stream() -> object:
            for content in ("hello", "", None, " world"):
                chunk = MagicMock()
                delta = MagicMock()
                delta.content = content
                choice = MagicMock()
                choice.delta = delta
                chunk.choices = [choice]
                yield chunk

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = fake_stream()
            result = [token async for token in stream_text("openai/gpt-4o", [])]

        assert result == ["hello", " world"]
        mock_acomp.assert_awaited_once_with(
            model="openai/gpt-4o",
            messages=[],
            timeout=120.0,
            stream=True,
        )

    async def test_stream_text_skips_malformed_chunks(self) -> None:
        """stream_text() should skip chunks with missing choices or delta."""

        async def fake_stream() -> object:
            empty_choice_chunk = MagicMock()
            empty_choice_chunk.choices = []
            yield empty_choice_chunk

            missing_delta_choice = MagicMock()
            missing_delta_choice.delta = None
            missing_delta_chunk = MagicMock()
            missing_delta_chunk.choices = [missing_delta_choice]
            yield missing_delta_chunk

            valid_delta = MagicMock()
            valid_delta.content = "token"
            valid_choice = MagicMock()
            valid_choice.delta = valid_delta
            valid_chunk = MagicMock()
            valid_chunk.choices = [valid_choice]
            yield valid_chunk

        with patch("q_ai.core.llm_litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = fake_stream()
            result = [token async for token in stream_text("openai/gpt-4o", [])]

        assert result == ["token"]

    def test_get_litellm_context_window_prefers_max_input_tokens(self) -> None:
        """get_litellm_context_window() should prefer max_input_tokens."""
        with patch(
            "q_ai.core.llm_litellm.get_model_info",
            return_value={"max_input_tokens": 8192, "max_tokens": 4096},
        ):
            result = get_litellm_context_window("anthropic/claude-sonnet-4-20250514")

        assert result == 8192

    def test_get_litellm_context_window_falls_back_to_max_tokens(self) -> None:
        """get_litellm_context_window() should use max_tokens when needed."""
        with patch("q_ai.core.llm_litellm.get_model_info", return_value={"max_tokens": 4096}):
            result = get_litellm_context_window("unknown/model")

        assert result == 4096
