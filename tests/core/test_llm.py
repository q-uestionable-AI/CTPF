"""Tests for the provider-agnostic LLM module."""

from __future__ import annotations

import json

import pytest

from q_ai.core.llm import (
    NormalizedResponse,
    ToolCall,
    get_provider_client,
    parse_model_string,
    serialize_evidence,
)


class TestParseModelString:
    """Tests for parse_model_string()."""

    def test_with_provider(self) -> None:
        provider, model_id = parse_model_string("anthropic/claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-20250514"

    def test_bare(self) -> None:
        provider, model_id = parse_model_string("claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-20250514"

    def test_ollama(self) -> None:
        provider, model_id = parse_model_string("ollama/llama3.1:8b")
        assert provider == "ollama"
        assert model_id == "llama3.1:8b"

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="both provider and model-id must be non-empty"):
            parse_model_string("openai/")

    def test_empty_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="both provider and model-id must be non-empty"):
            parse_model_string("/gpt-4o")

    def test_slash_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both provider and model-id must be non-empty"):
            parse_model_string("/")

    def test_openai(self) -> None:
        provider, model_id = parse_model_string("openai/gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"


class TestSerializeEvidence:
    """Tests for serialize_evidence()."""

    def test_dict_raw_response(self) -> None:
        response = NormalizedResponse(
            raw_response={"choices": [{"message": {"content": "hello"}}]},
            model="test",
        )
        result = json.loads(serialize_evidence(response))
        assert result["choices"][0]["message"]["content"] == "hello"

    def test_model_response_with_model_dump(self) -> None:
        class FakeResponse:
            def model_dump(self) -> dict:
                return {"id": "resp-1", "choices": []}

        response = NormalizedResponse(raw_response=FakeResponse(), model="test")
        result = json.loads(serialize_evidence(response))
        assert result["id"] == "resp-1"

    def test_none_raw_response(self) -> None:
        response = NormalizedResponse(
            content="hello",
            tool_calls=[ToolCall(name="test", arguments={"a": 1})],
            model="test",
        )
        result = json.loads(serialize_evidence(response))
        assert result["content"] == "hello"
        assert result["tool_calls"][0]["name"] == "test"


class TestGetProviderClient:
    """Tests for get_provider_client()."""

    def test_returns_litellm(self) -> None:
        from q_ai.core.llm_litellm import LiteLLMClient

        client = get_provider_client("anthropic/claude-sonnet-4-20250514")
        assert isinstance(client, LiteLLMClient)

    def test_returns_litellm_for_bare_model(self) -> None:
        from q_ai.core.llm_litellm import LiteLLMClient

        client = get_provider_client("claude-sonnet-4-20250514")
        assert isinstance(client, LiteLLMClient)
