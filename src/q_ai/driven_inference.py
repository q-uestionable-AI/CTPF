"""OpenAI-compatible driven-inference loop for controlled experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from q_ai.core.config import get_keyring_credential
from q_ai.core.db import get_connection, get_target
from q_ai.core.llm import NormalizedResponse, ProviderClient, ToolCall, ToolSpec
from q_ai.core.models import Target
from q_ai.mcp.connection import MCPConnection
from q_ai.services.db_service import resolve_partial_id

_DRIVER_NAME = "openai-compatible"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_MAX_ROUNDS = 12
_MCP_CONNECT_TIMEOUT_SECONDS = 10.0
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


class DrivenInferenceError(RuntimeError):
    """Raised when driven inference cannot preserve experiment integrity."""


@dataclass(frozen=True)
class OpenAICompatibleTargetProfile:
    """Validated non-secret target settings for one inference endpoint.

    Args:
        target_id: Persisted target identifier.
        name: Human-readable target name.
        endpoint: OpenAI-compatible API base URL.
        model: Exact model identifier sent to the endpoint.
        credential_name: OS-keyring entry name; never the credential itself.
        max_tokens: Fixed maximum generated tokens per inference round.
        temperature: Optional fixed sampling temperature.
        seed: Optional fixed provider seed.
        reasoning_effort: Optional fixed provider reasoning effort.
    """

    target_id: str
    name: str
    endpoint: str
    model: str
    credential_name: str
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float | None = None
    seed: int | None = None
    reasoning_effort: str | None = None

    @property
    def litellm_model(self) -> str:
        """Return the LiteLLM routing form for an OpenAI-compatible model."""
        return f"openai/{self.model}"

    def generation_parameters(self) -> dict[str, int | float | str]:
        """Return supported fixed generation parameters for provider calls."""
        parameters: dict[str, int | float | str] = {}
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        if self.seed is not None:
            parameters["seed"] = self.seed
        if self.reasoning_effort is not None:
            parameters["reasoning_effort"] = self.reasoning_effort
        return parameters

    def evidence_payload(self) -> dict[str, Any]:
        """Return the complete profile pin without credential material."""
        return {
            "target_id": self.target_id,
            "name": self.name,
            "driver": _DRIVER_NAME,
            "endpoint": self.endpoint,
            "model": self.model,
            "credential_name": self.credential_name,
            "max_tokens": self.max_tokens,
            "generation_parameters": self.generation_parameters(),
        }


@dataclass(frozen=True)
class DrivenInferenceResult:
    """Summary of one completed fresh inference conversation."""

    final_content: str
    round_count: int
    tool_call_count: int
    transcript_path: Path


def load_openai_target_profile(
    target_ref: str,
    *,
    db_path: Path | None = None,
) -> OpenAICompatibleTargetProfile:
    """Load and validate an OpenAI-compatible profile from a target row.

    Args:
        target_ref: Full or partial target ID (minimum eight characters).
        db_path: Optional database path override for tests.

    Returns:
        Validated non-secret inference profile.

    Raises:
        DrivenInferenceError: If the target or its metadata is invalid.
    """
    reference = target_ref.strip()
    if len(reference) < 8:
        raise DrivenInferenceError("target ID prefix must be at least 8 characters")
    try:
        with get_connection(db_path) as conn:
            target_id = resolve_partial_id(conn, "targets", reference)
            target = get_target(conn, target_id)
    except ValueError as exc:
        raise DrivenInferenceError(str(exc)) from exc
    if target is None:
        raise DrivenInferenceError(f"target not found: {reference}")
    return _profile_from_target(target)


def _profile_from_target(target: Target) -> OpenAICompatibleTargetProfile:
    if target.type != "inference":
        raise DrivenInferenceError("driven inference requires a target with type 'inference'")
    endpoint = _validated_endpoint(target.uri)
    metadata = target.metadata
    if not isinstance(metadata, dict):
        raise DrivenInferenceError("inference target metadata must be a JSON object")
    driver = _required_string(metadata, "driver")
    if driver != _DRIVER_NAME:
        raise DrivenInferenceError(f"unsupported inference driver: {driver!r}")
    return OpenAICompatibleTargetProfile(
        target_id=target.id,
        name=target.name,
        endpoint=endpoint,
        model=_required_string(metadata, "model"),
        credential_name=_required_string(metadata, "credential"),
        max_tokens=_max_tokens(metadata),
        temperature=_optional_float(metadata, "temperature", minimum=0.0, maximum=2.0),
        seed=_optional_int(metadata, "seed", None),
        reasoning_effort=_optional_choice(metadata, "reasoning_effort", _REASONING_EFFORTS),
    )


def _validated_endpoint(raw: str | None) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise DrivenInferenceError("inference target URI must contain an API base URL")
    endpoint = raw.strip().rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DrivenInferenceError("inference target URI must be an absolute HTTP(S) URL")
    return endpoint


def _required_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DrivenInferenceError(f"inference target metadata requires non-empty {key!r}")
    return value.strip()


def _optional_int(
    metadata: dict[str, Any],
    key: str,
    default: int | None,
    *,
    minimum: int | None = None,
) -> int | None:
    value = metadata.get(key)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DrivenInferenceError(f"inference target {key!r} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise DrivenInferenceError(f"inference target {key!r} must be at least {minimum}")
    return parsed


def _max_tokens(metadata: dict[str, Any]) -> int:
    parsed = _optional_int(metadata, "max_tokens", _DEFAULT_MAX_TOKENS, minimum=1)
    if parsed is None:
        raise DrivenInferenceError("inference target 'max_tokens' could not be resolved")
    return parsed


def _optional_choice(
    metadata: dict[str, Any],
    key: str,
    choices: frozenset[str],
) -> str | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise DrivenInferenceError(f"inference target {key!r} must be a string")
    parsed = value.strip()
    if not parsed:
        return None
    if parsed not in choices:
        allowed = ", ".join(sorted(choices))
        raise DrivenInferenceError(f"inference target {key!r} must be one of: {allowed}")
    return parsed


def _optional_float(
    metadata: dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DrivenInferenceError(f"inference target {key!r} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise DrivenInferenceError(
            f"inference target {key!r} must be between {minimum} and {maximum}"
        )
    return parsed


class OpenAICompatibleDriver:
    """Own a fresh model conversation and execute requested MCP tools."""

    def __init__(
        self,
        profile: OpenAICompatibleTargetProfile,
        *,
        client: ProviderClient | None = None,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
    ) -> None:
        """Configure the driver with a profile and optional test client."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self._profile = profile
        self._client = client
        self._max_rounds = max_rounds

    async def run(
        self,
        prompt: str,
        mcp_endpoint: str,
        transcript_path: Path,
    ) -> DrivenInferenceResult:
        """Run one bounded inference/tool loop and preserve its transcript.

        Args:
            prompt: Fixed scenario prompt for this fresh conversation.
            mcp_endpoint: Loopback proxy endpoint used for every tool call.
            transcript_path: External artifact path for request/response evidence.

        Returns:
            Completion summary for the fresh conversation.
        """
        transcript = _new_transcript(self._profile, prompt, mcp_endpoint)
        _write_json(transcript_path, transcript)
        secret: str | None = None
        try:
            client, secret = self._configured_client()
            result = await self._run_connected(
                client,
                prompt,
                mcp_endpoint,
                transcript_path,
                transcript,
            )
        except BaseException as exc:
            transcript["status"] = "failed"
            transcript["error"] = _error_payload(exc, secret)
            _write_json(transcript_path, transcript)
            raise
        transcript["status"] = "complete"
        _write_json(transcript_path, transcript)
        return result

    def _configured_client(self) -> tuple[ProviderClient, str | None]:
        if self._client is not None:
            return self._client, None
        credential = get_keyring_credential(self._profile.credential_name)
        if credential is None or not credential:
            raise DrivenInferenceError(
                f"OS keyring has no credential named {self._profile.credential_name!r}"
            )
        from q_ai.core.llm_litellm import LiteLLMClient

        client = LiteLLMClient(
            api_base=self._profile.endpoint,
            api_key=credential,
            generation_parameters=self._profile.generation_parameters(),
        )
        return client, credential

    async def _run_connected(
        self,
        client: ProviderClient,
        prompt: str,
        mcp_endpoint: str,
        transcript_path: Path,
        transcript: dict[str, Any],
    ) -> DrivenInferenceResult:
        async with MCPConnection.streamable_http(
            mcp_endpoint,
            timeout=_MCP_CONNECT_TIMEOUT_SECONDS,
        ) as connection:
            listed = await connection.session.list_tools()
            tools, schemas = _tool_specs(listed)
            transcript["tool_schemas"] = schemas
            _write_json(transcript_path, transcript)
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            return await self._run_rounds(
                client,
                connection.session,
                messages,
                tools,
                transcript_path,
                transcript,
            )

    async def _run_rounds(
        self,
        client: ProviderClient,
        session: Any,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        transcript_path: Path,
        transcript: dict[str, Any],
    ) -> DrivenInferenceResult:
        tool_call_count = 0
        for index in range(self._max_rounds):
            record = {"index": index + 1, "request": self._request_payload(messages, tools)}
            transcript["rounds"].append(record)
            _write_json(transcript_path, transcript)
            response = await client.complete(
                self._profile.litellm_model,
                messages,
                tools,
                max_tokens=self._profile.max_tokens,
            )
            record["response"] = _response_payload(response)
            messages.append(_assistant_message(response))
            if not response.tool_calls:
                return DrivenInferenceResult(
                    response.content,
                    index + 1,
                    tool_call_count,
                    transcript_path,
                )
            results = await _execute_tool_calls(session, response.tool_calls, messages)
            tool_call_count += len(results)
            record["tool_results"] = results
            _write_json(transcript_path, transcript)
        raise DrivenInferenceError(f"model exceeded the {self._max_rounds}-round tool-loop limit")

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> dict[str, Any]:
        return {
            "endpoint": self._profile.endpoint,
            "model": self._profile.model,
            "messages": _json_value(messages),
            "tools": [_openai_tool(tool) for tool in tools],
            "max_tokens": self._profile.max_tokens,
            **self._profile.generation_parameters(),
        }


def _new_transcript(
    profile: OpenAICompatibleTargetProfile,
    prompt: str,
    mcp_endpoint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "target_profile": profile.evidence_payload(),
        "prompt": prompt,
        "mcp_endpoint": mcp_endpoint,
        "tool_schemas": [],
        "rounds": [],
    }


def _tool_specs(listed: Any) -> tuple[list[ToolSpec], list[dict[str, Any]]]:
    raw_tools = getattr(listed, "tools", None)
    if not isinstance(raw_tools, list) or not raw_tools:
        raise DrivenInferenceError("proxied MCP server returned no tool schemas")
    specs: list[ToolSpec] = []
    schemas: list[dict[str, Any]] = []
    for raw in raw_tools:
        name = getattr(raw, "name", None)
        schema = getattr(raw, "inputSchema", None)
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            raise DrivenInferenceError("proxied MCP server returned a malformed tool schema")
        description = getattr(raw, "description", None)
        specs.append(ToolSpec(name, description if isinstance(description, str) else "", schema))
        dumped = raw.model_dump(by_alias=True, exclude_none=True)
        if not isinstance(dumped, dict):
            raise DrivenInferenceError("proxied MCP tool schema did not serialize to an object")
        schemas.append(_json_value(dumped))
    return specs, schemas


async def _execute_tool_calls(
    session: Any,
    calls: list[ToolCall],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in calls:
        if not call.id:
            raise DrivenInferenceError("provider tool call is missing its correlation ID")
        if not isinstance(call.arguments, dict):
            raise DrivenInferenceError("provider tool call arguments must be a JSON object")
        result = await session.call_tool(call.name, call.arguments)
        dumped = result.model_dump(by_alias=True, exclude_none=True)
        if not isinstance(dumped, dict):
            raise DrivenInferenceError("MCP tool result did not serialize to an object")
        result_payload = _json_value(dumped)
        results.append(
            {
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": _json_value(call.arguments),
                "result": result_payload,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result_payload, sort_keys=True),
            }
        )
    return results


def _assistant_message(response: NormalizedResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content or None}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, sort_keys=True),
                },
            }
            for call in response.tool_calls
        ]
    return message


def _openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": _json_value(tool.parameters),
        },
    }


def _response_payload(response: NormalizedResponse) -> dict[str, Any]:
    return {
        "model": response.model,
        "finish_reason": response.finish_reason,
        "content": response.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": _json_value(call.arguments)}
            for call in response.tool_calls
        ],
        "raw": _json_value(response.raw_response),
    }


def _error_payload(exc: BaseException, secret: str | None) -> dict[str, str]:
    message = str(exc)
    if secret:
        message = message.replace(secret, "<redacted>")
    return {"type": type(exc).__name__, "message": message}


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
