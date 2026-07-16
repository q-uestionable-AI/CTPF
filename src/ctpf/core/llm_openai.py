"""Narrow direct OpenAI-compatible implementation of ``ProviderClient``."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ctpf.core.hosted_inference import (
    MAX_REQUEST_BYTES,
    CanonicalEndpoint,
    HostedInferenceBoundary,
    Resolver,
    TransportFactory,
    canonicalize_endpoint,
    system_resolver,
)
from ctpf.core.llm import NormalizedResponse, ProviderError, ToolCall, ToolSpec
from ctpf.core.redaction import sanitize_evidence

MAX_TOOL_CALLS_PER_RESPONSE = 12
_CHAT_COMPLETIONS_PATH = "/chat/completions"


class OpenAICompatibleClient:
    """Perform direct bounded chat completions against one exact endpoint."""

    def __init__(
        self,
        *,
        endpoint: str | CanonicalEndpoint,
        api_key: str,
        requested_model: str,
        generation_parameters: dict[str, int | float | str] | None = None,
        resolver: Resolver = system_resolver,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        """Pin the exact target and non-secret generation controls.

        Args:
            endpoint: Canonical endpoint or operator-declared base URL.
            api_key: Keyring-resolved credential used only for Authorization.
            requested_model: Exact model string sent in every request.
            generation_parameters: Fixed supported generation controls.
            resolver: Injectable deterministic A/AAAA resolver.
            transport_factory: Injectable controlled transport for tests.
        """
        if not requested_model or requested_model != requested_model.strip():
            raise ValueError("requested model must be normalized non-empty text")
        canonical = (
            endpoint if isinstance(endpoint, CanonicalEndpoint) else canonicalize_endpoint(endpoint)
        )
        boundary_kwargs: dict[str, Any] = {"resolver": resolver}
        if transport_factory is not None:
            boundary_kwargs["transport_factory"] = transport_factory
        self._boundary = HostedInferenceBoundary(canonical, api_key, **boundary_kwargs)
        self._requested_model = requested_model
        self._generation_parameters = _validated_generation_parameters(generation_parameters or {})
        self._request_lock = asyncio.Lock()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
    ) -> NormalizedResponse:
        """Call one non-streaming OpenAI-compatible chat completion."""
        if model != self._requested_model:
            raise ProviderError("requested model differs from the pinned target model")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ProviderError("max_tokens must be a positive integer")
        payload = self._request_payload(messages, tools, max_tokens)
        async with self._request_lock:
            response = await self._boundary.post_json(_CHAT_COMPLETIONS_PATH, payload)
        allowed_tools = {tool.name for tool in tools}
        return _normalize_response(response.payload, model, response.evidence, allowed_tools)

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            raise ProviderError("messages must be a non-empty array")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ProviderError("tool specifications must have unique names")
        payload: dict[str, Any] = {
            "max_tokens": max_tokens,
            "messages": messages,
            "model": self._requested_model,
            "stream": False,
        }
        if tools:
            payload["tools"] = [_tool_payload(tool) for tool in tools]
        payload.update(self._generation_parameters)
        try:
            encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderError("provider request was not JSON-compatible") from exc
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ProviderError("provider request exceeded the serialized byte ceiling")
        return payload


def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
    if not tool.name or not isinstance(tool.parameters, dict):
        raise ProviderError("tool specification is malformed")
    return {
        "type": "function",
        "function": {
            "description": tool.description,
            "name": tool.name,
            "parameters": dict(tool.parameters),
        },
    }


def _normalize_response(
    payload: dict[str, Any],
    requested_model: str,
    evidence: dict[str, Any],
    allowed_tools: set[str],
) -> NormalizedResponse:
    failure_evidence = dict(evidence)
    failure_evidence["provider_response"] = sanitize_evidence(payload)
    choice = _first_choice(payload, failure_evidence)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _response_error("provider response choice has no message", failure_evidence)
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise _response_error("provider response content must be text or null", failure_evidence)
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise _response_error("provider finish_reason must be text or null", failure_evidence)
    tool_calls = _parse_tool_calls(message.get("tool_calls"), failure_evidence, allowed_tools)
    usage = _usage(payload.get("usage"))
    provider_model = payload.get("model")
    if provider_model is not None and not isinstance(provider_model, str):
        raise _response_error("provider-reported model must be text or null", failure_evidence)
    return NormalizedResponse(
        tool_calls=tool_calls,
        content=content or "",
        finish_reason=finish_reason,
        raw_response=payload,
        requested_model=requested_model,
        provider_reported_model=provider_model,
        input_tokens=usage[0],
        output_tokens=usage[1],
        total_tokens=usage[2],
        transport_evidence=evidence,
    )


def _first_choice(payload: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _response_error("provider response requires a non-empty choices array", evidence)
    return choices[0]


def _parse_tool_calls(
    raw: Any,
    evidence: dict[str, Any],
    allowed_tools: set[str],
) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _response_error("provider tool_calls must be an array or null", evidence)
    if len(raw) > MAX_TOOL_CALLS_PER_RESPONSE:
        raise _response_error("provider response exceeded the tool-call ceiling", evidence)
    calls = [_parse_tool_call(item, evidence, allowed_tools) for item in raw]
    if len({call.id for call in calls}) != len(calls):
        raise _response_error("provider tool call IDs must be unique", evidence)
    return calls


def _parse_tool_call(
    raw: Any,
    evidence: dict[str, Any],
    allowed_tools: set[str],
) -> ToolCall:
    if not isinstance(raw, dict):
        raise _response_error("provider tool call must be an object", evidence)
    function = raw.get("function")
    call_id = raw.get("id")
    if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id:
        raise _response_error("provider tool call identity is malformed", evidence)
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, str) or not arguments:
        raise _response_error("provider tool call function is malformed", evidence)
    if name not in allowed_tools:
        raise _response_error("provider requested an unapproved tool", evidence)
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise _response_error("provider tool call arguments are malformed JSON", evidence) from exc
    if not isinstance(decoded, dict):
        raise _response_error("provider tool call arguments must be a JSON object", evidence)
    return ToolCall(name=name, arguments=decoded, id=call_id)


def _usage(raw: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(raw, dict):
        return None, None, None
    prompt = _optional_nonnegative_int(raw.get("prompt_tokens"))
    completion = _optional_nonnegative_int(raw.get("completion_tokens"))
    total = _optional_nonnegative_int(raw.get("total_tokens"))
    return prompt, completion, total


def _optional_nonnegative_int(raw: Any) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return int(raw)


def _validated_generation_parameters(
    raw: dict[str, int | float | str],
) -> dict[str, int | float | str]:
    allowed = {"reasoning_effort", "seed", "temperature"}
    if set(raw).difference(allowed):
        raise ValueError("unsupported OpenAI-compatible generation parameter")
    if any(isinstance(value, bool) for value in raw.values()):
        raise ValueError("generation parameters must not contain booleans")
    return dict(raw)


def _response_error(message: str, evidence: dict[str, Any]) -> ProviderError:
    error = ProviderError(message)
    error.evidence = sanitize_evidence(evidence)  # type: ignore[attr-defined]
    return error
