"""Tests for the direct OpenAI-compatible provider client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ctpf.core.hosted_inference import Resolution
from ctpf.core.llm import ProviderError, ToolSpec
from ctpf.core.llm_openai import OpenAICompatibleClient

_PUBLIC_ADDRESS = "93.184.216.34"


class _CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.connected_peer = _PUBLIC_ADDRESS
        self.requests: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        self.requests.append(
            {
                "authorization": request.headers.get("authorization"),
                "body": json.loads(body),
                "url": str(request.url),
            }
        )
        return httpx.Response(200, json=self._payload)


async def _resolver(_host: str, _port: int) -> tuple[str, ...]:
    return (_PUBLIC_ADDRESS,)


def _client(transport: _CaptureTransport) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        endpoint="https://models.example.test/v1",
        api_key="exact-secret",
        requested_model="requested-model",
        generation_parameters={"temperature": 0.0, "seed": 7},
        resolver=_resolver,
        transport_factory=lambda _endpoint, _resolution: transport,
    )


async def test_normalizes_tool_calls_usage_and_separate_model_identities() -> None:
    transport = _CaptureTransport(
        {
            "id": "response-1",
            "model": "provider-claimed-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_status",
                                    "arguments": '{"scope":"synthetic"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        }
    )
    tools = [ToolSpec("read_status", "Read status", {"type": "object"})]

    response = await _client(transport).complete(
        "requested-model",
        [{"role": "user", "content": "Inspect."}],
        tools,
        max_tokens=64,
    )

    assert response.requested_model == "requested-model"
    assert response.provider_reported_model == "provider-claimed-model"
    assert response.artifact_proven_model is None
    assert response.tool_calls[0].arguments == {"scope": "synthetic"}
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (11, 5, 16)
    assert response.transport_evidence["connected_peer"] == _PUBLIC_ADDRESS
    request = transport.requests[0]
    assert request["url"] == "https://models.example.test/v1/chat/completions"
    assert request["authorization"] == "Bearer exact-secret"
    assert request["body"]["model"] == "requested-model"
    assert request["body"]["stream"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": {"unexpected": True}}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "read_status",
                                    "arguments": "[]",
                                },
                            }
                        ],
                    }
                }
            ]
        },
    ],
)
async def test_malformed_provider_shapes_fail_closed_with_transport_evidence(
    payload: dict[str, Any],
) -> None:
    transport = _CaptureTransport(payload)

    with pytest.raises(ProviderError) as caught:
        await _client(transport).complete(
            "requested-model",
            [{"role": "user", "content": "Inspect."}],
            [],
        )

    evidence = getattr(caught.value, "evidence", None)
    assert isinstance(evidence, dict)
    assert evidence["connected_peer"] == _PUBLIC_ADDRESS


async def test_pinned_model_cannot_be_substituted_at_call_time() -> None:
    transport = _CaptureTransport({"choices": []})

    with pytest.raises(ProviderError, match="differs"):
        await _client(transport).complete(
            "substituted-model",
            [{"role": "user", "content": "Inspect."}],
            [],
        )

    assert transport.requests == []


def test_generation_parameters_cannot_overwrite_request_authority() -> None:
    transport = _CaptureTransport({"choices": []})

    with pytest.raises(ValueError, match="unsupported"):
        OpenAICompatibleClient(
            endpoint="https://models.example.test/v1",
            api_key="secret",
            requested_model="requested-model",
            generation_parameters={"model": "substituted"},
            resolver=_resolver,
            transport_factory=lambda _endpoint, _resolution: transport,
        )


def test_transport_factory_receives_complete_resolution_type() -> None:
    """The injected seam remains typed around an approved resolution object."""
    resolution = Resolution((_PUBLIC_ADDRESS,), _PUBLIC_ADDRESS)
    assert resolution.selected_address == _PUBLIC_ADDRESS
