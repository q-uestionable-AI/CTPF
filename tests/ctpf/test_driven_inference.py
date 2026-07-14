"""Tests for the OpenAI-compatible driven-inference boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from q_ai import driven_inference
from q_ai.core.db import create_target, get_connection
from q_ai.core.llm import NormalizedResponse, ToolCall, ToolSpec
from q_ai.driven_inference import (
    DrivenInferenceError,
    OpenAICompatibleDriver,
    OpenAICompatibleTargetProfile,
    load_openai_target_profile,
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} description"
        self.inputSchema = {"type": "object", "properties": {}}

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


class _FakeToolResult:
    def __init__(self, name: str) -> None:
        self.name = name

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": f"{self.name} result"}],
            "structuredContent": {"tool": self.name},
            "isError": False,
        }


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[_FakeTool("read_inbox"), _FakeTool("write_memo")])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeToolResult:
        self.calls.append((name, arguments))
        return _FakeToolResult(name)


class _FakeConnection:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeConnection:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeProviderClient:
    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
    ) -> NormalizedResponse:
        self.requests.append(
            {
                "model": model,
                "messages": json.loads(json.dumps(messages)),
                "tools": [tool.name for tool in tools],
                "max_tokens": max_tokens,
            }
        )
        return self.responses.pop(0)


def _profile() -> OpenAICompatibleTargetProfile:
    return OpenAICompatibleTargetProfile(
        target_id="1234567890abcdef",
        name="research endpoint",
        endpoint="https://inference.example.test/v1",
        model="research-model-1",
        credential_name="phase5b-test",
        max_tokens=321,
        temperature=0.0,
        seed=7,
        reasoning_effort="none",
    )


def _patch_connection(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    monkeypatch.setattr(
        driven_inference.MCPConnection,
        "streamable_http",
        lambda _url, **_kwargs: _FakeConnection(session),
    )


class TestTargetProfile:
    """Existing target rows provide the narrow Phase 5b profile store."""

    @pytest.mark.parametrize(
        "reasoning_effort",
        ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    def test_loads_and_coerces_flat_metadata(
        self,
        tmp_path: Path,
        reasoning_effort: str,
    ) -> None:
        db_path = tmp_path / "qai.db"
        with get_connection(db_path) as conn:
            target_id = create_target(
                conn,
                type="inference",
                name="remote model",
                uri="https://models.example.test/v1/",
                metadata={
                    "driver": "openai-compatible",
                    "model": "model-a",
                    "credential": "remote-a",
                    "max_tokens": "512",
                    "temperature": "0",
                    "seed": "42",
                    "reasoning_effort": reasoning_effort,
                },
            )

        profile = load_openai_target_profile(target_id[:8], db_path=db_path)

        assert profile.endpoint == "https://models.example.test/v1"
        assert profile.model == "model-a"
        assert profile.max_tokens == 512
        assert profile.generation_parameters() == {
            "temperature": 0.0,
            "seed": 42,
            "reasoning_effort": reasoning_effort,
        }

    def test_rejects_malformed_numeric_metadata(self, tmp_path: Path) -> None:
        db_path = tmp_path / "qai.db"
        with get_connection(db_path) as conn:
            target_id = create_target(
                conn,
                type="inference",
                name="bad remote",
                uri="https://models.example.test/v1",
                metadata={
                    "driver": "openai-compatible",
                    "model": "model-a",
                    "credential": "remote-a",
                    "max_tokens": "many",
                },
            )

        with pytest.raises(DrivenInferenceError, match=r"max_tokens.*integer"):
            load_openai_target_profile(target_id[:8], db_path=db_path)

    def test_rejects_unsupported_reasoning_effort(self, tmp_path: Path) -> None:
        db_path = tmp_path / "qai.db"
        with get_connection(db_path) as conn:
            target_id = create_target(
                conn,
                type="inference",
                name="bad remote",
                uri="https://models.example.test/v1",
                metadata={
                    "driver": "openai-compatible",
                    "model": "model-a",
                    "credential": "remote-a",
                    "reasoning_effort": "off",
                },
            )

        with pytest.raises(DrivenInferenceError, match=r"reasoning_effort.*none"):
            load_openai_target_profile(target_id[:8], db_path=db_path)


class TestOpenAICompatibleDriver:
    """The driver owns a bounded fresh conversation and proxied tool loop."""

    async def test_executes_tool_calls_and_preserves_transcript(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _FakeSession()
        _patch_connection(monkeypatch, session)
        client = _FakeProviderClient(
            [
                NormalizedResponse(
                    tool_calls=[ToolCall("read_inbox", {}, id="call-1")],
                    finish_reason="tool_calls",
                    raw_response={"id": "response-1"},
                    model="openai/research-model-1",
                ),
                NormalizedResponse(
                    content="Done.",
                    finish_reason="stop",
                    raw_response={"id": "response-2"},
                    model="openai/research-model-1",
                ),
            ]
        )
        transcript_path = tmp_path / "session-A.inference.json"

        result = await OpenAICompatibleDriver(_profile(), client=client).run(
            "Inspect the inbox.",
            "http://127.0.0.1:8765/mcp/",
            transcript_path,
        )

        assert result.final_content == "Done."
        assert result.round_count == 2
        assert session.calls == [("read_inbox", {})]
        assert client.requests[0]["model"] == "openai/research-model-1"
        second_messages = client.requests[1]["messages"]
        assert second_messages[-2]["tool_calls"][0]["id"] == "call-1"
        assert second_messages[-1]["tool_call_id"] == "call-1"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "complete"
        assert transcript["rounds"][0]["request"]["endpoint"] == _profile().endpoint
        assert transcript["rounds"][0]["request"]["reasoning_effort"] == "none"
        assert transcript["target_profile"]["generation_parameters"]["reasoning_effort"] == "none"
        assert transcript["rounds"][0]["tool_results"][0]["name"] == "read_inbox"
        assert "credential" not in json.dumps(transcript).lower().replace("credential_name", "")

    async def test_missing_tool_call_id_fails_and_preserves_diagnostics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_connection(monkeypatch, _FakeSession())
        client = _FakeProviderClient(
            [
                NormalizedResponse(
                    tool_calls=[ToolCall("read_inbox", {})],
                    finish_reason="tool_calls",
                    raw_response={"id": "response-1"},
                    model="openai/research-model-1",
                )
            ]
        )
        transcript_path = tmp_path / "failed.inference.json"

        with pytest.raises(DrivenInferenceError, match="correlation ID"):
            await OpenAICompatibleDriver(_profile(), client=client).run(
                "Inspect the inbox.",
                "http://127.0.0.1:8765/mcp/",
                transcript_path,
            )

        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "failed"
        assert transcript["error"]["type"] == "DrivenInferenceError"

    async def test_missing_keyring_credential_is_an_explicit_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(driven_inference, "get_keyring_credential", lambda _name: None)
        transcript_path = tmp_path / "missing-key.inference.json"

        with pytest.raises(DrivenInferenceError, match="OS keyring"):
            await OpenAICompatibleDriver(_profile()).run(
                "Inspect the inbox.",
                "http://127.0.0.1:8765/mcp/",
                transcript_path,
            )

        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "failed"
        assert "phase5b-test" in transcript["error"]["message"]
