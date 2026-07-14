"""Provider-agnostic LLM interaction protocol and data models.

Defines the ProviderClient protocol and normalized data structures that
campaign.py imports. No provider-specific imports belong here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Error from a provider client."""


class UnsupportedCapabilityError(ProviderError):
    """The model does not support the requested capability (e.g. tool calling)."""


@dataclass
class ToolSpec:
    """Provider-agnostic tool definition.

    Built from PayloadTemplate. Converted to provider-specific format
    by the ProviderClient implementation.
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
        raw_response: The original provider response object for evidence.
        model: The model string that was called.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""
    finish_reason: str | None = None
    raw_response: Any = None
    model: str = ""


class ProviderClient(Protocol):
    """Protocol for provider-agnostic LLM completion."""

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
    ) -> NormalizedResponse: ...


def parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model-id' into (provider, model_id).

    Bare strings (no slash) fall back to ('anthropic', model).

    Args:
        model: Model string, optionally prefixed with provider.

    Returns:
        Tuple of (provider, model_id).
    """
    if "/" in model:
        provider, _, model_id = model.partition("/")
        if not provider or not model_id:
            raise ValueError(
                f"Invalid model string {model!r}: both provider and model-id "
                "must be non-empty. Expected format: provider/model-id"
            )
        return provider, model_id
    return "anthropic", model


def tool_spec_from_template(template: Any) -> ToolSpec:
    """Convert a PayloadTemplate to a ToolSpec.

    Args:
        template: PayloadTemplate with tool metadata.

    Returns:
        ToolSpec suitable for any ProviderClient.
    """
    properties: dict[str, dict[str, str]] = {}
    for name, info in template.tool_params.items():
        prop: dict[str, str] = {"type": info.get("type", "string")}
        if desc := info.get("description", ""):
            prop["description"] = desc
        properties[name] = prop

    return ToolSpec(
        name=template.tool_name,
        description=template.tool_description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(template.tool_params.keys()),
        },
    )


def serialize_evidence(response: NormalizedResponse) -> str:
    """Serialize the raw provider response for evidence storage.

    Returns JSON string. Handles litellm ModelResponse objects,
    dicts, and arbitrary objects via best-effort serialization.

    Args:
        response: Normalized response containing raw_response.

    Returns:
        JSON string of the serialized evidence.
    """
    raw = response.raw_response
    if raw is None:
        return json.dumps(
            {
                "content": response.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            },
            indent=2,
            default=str,
        )

    if isinstance(raw, dict):
        return json.dumps(raw, indent=2, default=str)

    if hasattr(raw, "model_dump"):
        return json.dumps(raw.model_dump(), indent=2, default=str)

    try:
        return json.dumps(vars(raw), indent=2, default=str)
    except TypeError:
        return json.dumps({"raw": str(raw)}, indent=2, default=str)


def get_provider_client(model: str) -> ProviderClient:
    """Factory: return a ProviderClient for the given model string.

    Parses the provider prefix and returns the appropriate client.
    For MVP, always returns the litellm client (which handles all providers).

    Args:
        model: Model string (e.g. 'anthropic/claude-sonnet-4-20250514').

    Returns:
        A ProviderClient instance.
    """
    from q_ai.core.llm_litellm import LiteLLMClient

    return LiteLLMClient()
