"""OpenAI-compatible driven-inference loop for controlled experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from ctpf.automation.contracts import BillingClass, DataEgressClass
from ctpf.core.config import get_keyring_credential
from ctpf.core.db import get_connection, get_target
from ctpf.core.hosted_inference import MAX_INPUT_TOKENS, CanonicalEndpoint, canonicalize_endpoint
from ctpf.core.llm import NormalizedResponse, ProviderClient, ToolCall, ToolSpec
from ctpf.core.models import Target
from ctpf.core.redaction import redact_text, sanitize_evidence
from ctpf.mcp.connection import MCPConnection
from ctpf.services.db_service import resolve_partial_id

_DRIVER_NAME = "openai-compatible"
_DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_ROUNDS = 12
MAX_TOOL_CALLS_PER_SESSION = 12
_MCP_CONNECT_TIMEOUT_SECONDS = 10.0
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_AuthorityEnumT = TypeVar("_AuthorityEnumT", BillingClass, DataEgressClass)


class DrivenInferenceError(RuntimeError):
    """Raised when driven inference cannot preserve experiment integrity."""


class InferenceControl(Protocol):
    """Narrow governed inference boundary used without coupling the driver."""

    @property
    def expected_tool_names(self) -> frozenset[str]: ...

    def reserve(self, boundary: str, **reservation: int) -> dict[str, Any]: ...

    def record_provider_usage(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> dict[str, Any]: ...

    async def wait(self, awaitable: Any, boundary: str) -> Any: ...


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
        max_input_tokens: Conservative input-token reservation per provider request.
        billing_class: Human-declared billing basis.
        request_cost_ceiling_microusd: Maximum approved cost per provider request.
        data_egress_class: Approved class of data sent to the endpoint.
        retention_acknowledged: Operator acknowledgement of provider retention/privacy.
        residual_cost_acknowledged: Operator acknowledgement of provider-side residual cost.
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
    max_input_tokens: int = MAX_INPUT_TOKENS
    billing_class: BillingClass = BillingClass.UNMETERED
    request_cost_ceiling_microusd: int | None = None
    data_egress_class: DataEgressClass = DataEgressClass.LOCAL_ONLY
    retention_acknowledged: bool = False
    residual_cost_acknowledged: bool = False

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
            "max_input_tokens": self.max_input_tokens,
            "generation_parameters": self.generation_parameters(),
            "billing_class": self.billing_class.value,
            "request_cost_ceiling_microusd": self.request_cost_ceiling_microusd,
            "data_egress_class": self.data_egress_class.value,
            "retention_acknowledged": self.retention_acknowledged,
            "residual_cost_acknowledged": self.residual_cost_acknowledged,
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
    endpoint = _canonical_endpoint(target.uri)
    metadata = target.metadata
    if not isinstance(metadata, dict):
        raise DrivenInferenceError("inference target metadata must be a JSON object")
    driver = _required_string(metadata, "driver")
    if driver != _DRIVER_NAME:
        raise DrivenInferenceError(f"unsupported inference driver: {driver!r}")
    authority = _authority_metadata(metadata, endpoint)
    return OpenAICompatibleTargetProfile(
        target_id=target.id,
        name=target.name,
        endpoint=endpoint.normalized_url,
        model=_required_string(metadata, "model"),
        credential_name=_required_string(metadata, "credential"),
        max_tokens=_max_tokens(metadata),
        temperature=_optional_float(metadata, "temperature", minimum=0.0, maximum=2.0),
        seed=_optional_int(metadata, "seed", None),
        reasoning_effort=_optional_choice(metadata, "reasoning_effort", _REASONING_EFFORTS),
        max_input_tokens=_max_input_tokens(metadata),
        billing_class=authority[0],
        request_cost_ceiling_microusd=authority[1],
        data_egress_class=authority[2],
        retention_acknowledged=authority[3],
        residual_cost_acknowledged=authority[4],
    )


def _canonical_endpoint(raw: str | None) -> CanonicalEndpoint:
    if not isinstance(raw, str) or not raw.strip():
        raise DrivenInferenceError("inference target URI must contain an API base URL")
    try:
        return canonicalize_endpoint(raw.strip().rstrip("/"))
    except ValueError as exc:
        raise DrivenInferenceError(str(exc)) from exc


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


def _max_input_tokens(metadata: dict[str, Any]) -> int:
    parsed = _optional_int(metadata, "max_input_tokens", MAX_INPUT_TOKENS, minimum=1)
    if parsed is None:
        raise DrivenInferenceError("inference target 'max_input_tokens' could not be resolved")
    return parsed


def _authority_metadata(
    metadata: dict[str, Any],
    endpoint: CanonicalEndpoint,
) -> tuple[BillingClass, int | None, DataEgressClass, bool, bool]:
    if endpoint.network_class == "loopback":
        return _local_authority_metadata(metadata)
    billing = _enum_metadata(metadata, "billing_class", BillingClass)
    if billing == BillingClass.EXTERNAL_RUNTIME:
        raise DrivenInferenceError("inference target billing_class cannot be external_runtime")
    ceiling = _cost_ceiling(metadata, billing)
    egress = _enum_metadata(metadata, "data_egress_class", DataEgressClass)
    if egress != DataEgressClass.PACKAGED_SYNTHETIC_REMOTE:
        raise DrivenInferenceError(
            "public inference targets require packaged_synthetic_remote data egress"
        )
    retention = _required_bool(metadata, "retention_acknowledged")
    residual = _required_bool(metadata, "residual_cost_acknowledged")
    if not retention or not residual:
        raise DrivenInferenceError(
            "public inference targets require retention and residual-cost acknowledgements"
        )
    return billing, ceiling, egress, retention, residual


def _local_authority_metadata(
    metadata: dict[str, Any],
) -> tuple[BillingClass, int | None, DataEgressClass, bool, bool]:
    billing = _optional_enum_metadata(
        metadata,
        "billing_class",
        BillingClass,
        BillingClass.UNMETERED,
    )
    egress = _optional_enum_metadata(
        metadata,
        "data_egress_class",
        DataEgressClass,
        DataEgressClass.LOCAL_ONLY,
    )
    if billing != BillingClass.UNMETERED or egress != DataEgressClass.LOCAL_ONLY:
        raise DrivenInferenceError("loopback inference targets must be unmetered and local_only")
    if metadata.get("request_cost_ceiling_microusd") not in {None, ""}:
        raise DrivenInferenceError("loopback inference targets must not declare request cost")
    return billing, None, egress, False, False


def _cost_ceiling(metadata: dict[str, Any], billing: BillingClass) -> int | None:
    value = _optional_int(metadata, "request_cost_ceiling_microusd", None, minimum=0)
    if billing == BillingClass.METERED and value is None:
        raise DrivenInferenceError("metered inference targets require a request cost ceiling")
    if billing != BillingClass.METERED and value is not None:
        raise DrivenInferenceError("only metered inference targets may declare request cost")
    return value


def _enum_metadata(
    metadata: dict[str, Any],
    key: str,
    enum_type: type[_AuthorityEnumT],
) -> _AuthorityEnumT:
    value = _required_string(metadata, key)
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise DrivenInferenceError(f"inference target {key!r} must be one of: {choices}") from exc


def _optional_enum_metadata(
    metadata: dict[str, Any],
    key: str,
    enum_type: type[_AuthorityEnumT],
    default: _AuthorityEnumT,
) -> _AuthorityEnumT:
    if metadata.get(key) in {None, ""}:
        return default
    return _enum_metadata(metadata, key, enum_type)


def _required_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise DrivenInferenceError(f"inference target {key!r} must be a boolean")
    return value


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
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        control: InferenceControl | None = None,
    ) -> None:
        """Configure the driver with a profile and optional test client."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self._profile = profile
        self._client = client
        self._max_rounds = max_rounds
        self._control = control

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
        from ctpf.core.llm_openai import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            endpoint=self._profile.endpoint,
            api_key=credential,
            requested_model=self._profile.model,
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
            expected = self._control.expected_tool_names if self._control is not None else None
            tools, schemas = _tool_specs(listed, expected)
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
            response = await self._complete(client, messages, tools)
            record["response"] = _response_payload(response)
            messages.append(_assistant_message(response))
            if not response.tool_calls:
                return DrivenInferenceResult(
                    response.content,
                    index + 1,
                    tool_call_count,
                    transcript_path,
                )
            if tool_call_count + len(response.tool_calls) > MAX_TOOL_CALLS_PER_SESSION:
                raise DrivenInferenceError("model exceeded the per-session tool-call limit")
            results = await _execute_tool_calls(session, response.tool_calls, messages)
            tool_call_count += len(results)
            record["tool_results"] = results
            _write_json(transcript_path, transcript)
        raise DrivenInferenceError(f"model exceeded the {self._max_rounds}-round tool-loop limit")

    async def _complete(
        self,
        client: ProviderClient,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> NormalizedResponse:
        if self._control is None:
            return await client.complete(
                self._profile.model,
                messages,
                tools,
                max_tokens=self._profile.max_tokens,
            )
        self._control.reserve(
            "provider_request",
            cost_microusd=self._profile.request_cost_ceiling_microusd or 0,
            input_tokens_reserved=self._profile.max_input_tokens,
            output_tokens_reserved=self._profile.max_tokens,
            provider_requests=1,
        )
        completion = client.complete(
            self._profile.model,
            messages,
            tools,
            max_tokens=self._profile.max_tokens,
        )
        response = cast(
            NormalizedResponse,
            await self._control.wait(completion, "provider_request"),
        )
        _validate_reported_usage(response, self._profile)
        self._control.record_provider_usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )
        return response

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> dict[str, Any]:
        return {
            "endpoint": self._profile.endpoint,
            "model": self._profile.model,
            "messages": _evidence_value(messages),
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


def _tool_specs(
    listed: Any,
    expected_names: frozenset[str] | None = None,
) -> tuple[list[ToolSpec], list[dict[str, Any]]]:
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
        schemas.append(_evidence_value(dumped))
    if expected_names is not None and {spec.name for spec in specs} != expected_names:
        raise DrivenInferenceError("proxied MCP tool schemas differ from the scenario allowlist")
    return specs, schemas


def _validate_reported_usage(
    response: NormalizedResponse,
    profile: OpenAICompatibleTargetProfile,
) -> None:
    if response.input_tokens is not None and response.input_tokens > profile.max_input_tokens:
        raise DrivenInferenceError("provider reported input usage above the approved ceiling")
    if response.output_tokens is not None and response.output_tokens > profile.max_tokens:
        raise DrivenInferenceError("provider reported output usage above the approved ceiling")


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
        evidence_result = _evidence_value(result_payload)
        results.append(
            {
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": _evidence_value(call.arguments),
                "result": evidence_result,
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
            "parameters": _evidence_value(tool.parameters),
        },
    }


def _response_payload(response: NormalizedResponse) -> dict[str, Any]:
    return {
        "requested_model": response.requested_model,
        "provider_reported_model": response.provider_reported_model,
        "artifact_proven_model": response.artifact_proven_model,
        "finish_reason": response.finish_reason,
        "content": response.content,
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "cost_microusd": response.cost_microusd,
        },
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": _evidence_value(call.arguments)}
            for call in response.tool_calls
        ],
        "raw": _evidence_value(response.raw_response),
        "transport": _evidence_value(response.transport_evidence),
    }


def _error_payload(exc: BaseException, secret: str | None) -> dict[str, Any]:
    secrets = (secret,) if secret else ()
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": redact_text(str(exc), secrets),
    }
    evidence = getattr(exc, "evidence", None)
    if evidence is not None:
        payload["evidence"] = sanitize_evidence(evidence, secrets)
    return payload


def _json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise DrivenInferenceError("external value was not JSON-compatible") from exc


def _evidence_value(value: Any) -> Any:
    return sanitize_evidence(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
