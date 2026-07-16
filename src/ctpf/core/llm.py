"""Provider-agnostic LLM interaction protocol and data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(Exception):
    """Error from a provider client."""


class UnsupportedCapabilityError(ProviderError):
    """The model does not support the requested capability (e.g. tool calling)."""


@dataclass
class ToolSpec:
    """Provider-agnostic tool definition.

    Converted to provider-specific format by the ProviderClient implementation.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    """A single tool invocation from the model response."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class NormalizedResponse:
    """Provider-agnostic LLM response.

    Attributes:
        tool_calls: List of tool invocations (empty if none).
        content: Full text content from the model (empty string if none).
        finish_reason: Provider finish reason (stop, tool_calls, length, etc.).
        raw_response: Sanitized bounded provider response evidence.
        requested_model: Exact model identity requested by CTPF.
        provider_reported_model: Untrusted identity claimed by the provider.
        artifact_proven_model: Independently proven deployment identity, when available.
        transport_evidence: Sanitized destination and transport observations.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    requested_model: str = ""
    provider_reported_model: str | None = None
    artifact_proven_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_microusd: int | None = None
    transport_evidence: dict[str, Any] = field(default_factory=dict)


class ProviderClient(Protocol):
    """Protocol for provider-agnostic LLM completion."""

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
    ) -> NormalizedResponse: ...
