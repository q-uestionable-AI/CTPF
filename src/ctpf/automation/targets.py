"""Scenario and target identity fingerprints for governed automation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ctpf import __version__, driven_inference, external_runtime
from ctpf.automation.canonical import sha256_digest
from ctpf.automation.contracts import (
    BillingClass,
    DataEgressClass,
    ExperimentMode,
    NetworkClass,
    TargetPolicy,
)
from ctpf.core import hosted_inference, llm_openai
from ctpf.core.hosted_inference import canonicalize_endpoint
from ctpf.driven_inference import OpenAICompatibleTargetProfile
from ctpf.external_runtime import (
    ClaudeCodeTargetProfile,
    ExternalRuntimeError,
    load_experiment_target_profile,
)

_FULL_ID = re.compile(r"^[0-9a-f]{32}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MIN_TEMPERATURE = Decimal("0")
_MAX_TEMPERATURE = Decimal("2")
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_INFERENCE_BEHAVIOR_KEYS = {
    "authority_contract_version",
    "billing",
    "credential_alias",
    "data_egress",
    "driver",
    "driver_source_hash",
    "endpoint",
    "generation_parameters",
    "limits",
    "model",
    "target_id",
    "target_type",
    "transport",
}
_RUNTIME_BEHAVIOR_KEYS = {
    "authority_contract_version",
    "billing",
    "data_egress",
    "driver",
    "driver_source_hash",
    "environment_policy",
    "executable",
    "executable_sha256",
    "identity_probe_processes",
    "identity_probe_timeout_seconds",
    "mcp_policy",
    "model",
    "runtime_version",
    "target_id",
    "target_type",
    "timeout_seconds",
}
_ENDPOINT_KEYS = {
    "base_path",
    "host",
    "network_class",
    "normalized_url",
    "origin",
    "port",
    "scheme",
}
_INFERENCE_LIMIT_KEYS = {
    "max_input_tokens",
    "max_output_tokens",
    "max_provider_rounds",
    "max_request_bytes",
    "max_response_bytes",
    "max_tool_calls_per_session",
}
_TRANSPORT_KEYS = {
    "concurrent_requests",
    "deadlines_seconds",
    "environment_proxy_policy",
    "http_protocol",
    "httpcore_version",
    "httpx_version",
    "max_attempts",
    "redirect_policy",
    "retry_count",
    "source_hashes",
    "tls_policy",
}


class TargetIdentityError(ValueError):
    """Raised when a scenario or target cannot be identified safely."""


@dataclass(frozen=True)
class ScenarioCapability:
    """Immutable installed capability exposed to a run proposal."""

    scenario: str
    contract_version: int
    modes: tuple[ExperimentMode, ...]
    conditions: tuple[str, ...]
    sessions_per_trial: int
    prompts: tuple[str, ...]
    tool_names: tuple[str, ...]
    effect_ids: tuple[str, ...]
    supported_target_types: tuple[str, ...]
    retry_policy: str
    package_version: str
    source_hashes: dict[str, str]
    fingerprint: str

    def to_payload(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return a canonical JSON-compatible capability object."""
        payload = {
            "conditions": list(self.conditions),
            "contract_version": self.contract_version,
            "effect_ids": list(self.effect_ids),
            "modes": [mode.value for mode in self.modes],
            "package_version": self.package_version,
            "prompts": list(self.prompts),
            "retry_policy": self.retry_policy,
            "scenario": self.scenario,
            "sessions_per_trial": self.sessions_per_trial,
            "source_hashes": dict(self.source_hashes),
            "supported_target_types": list(self.supported_target_types),
            "tool_names": list(self.tool_names),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class TargetIdentity:
    """Behaviorally relevant identity of one validated target profile."""

    target_id: str
    target_type: str
    network_class: NetworkClass
    behavior: dict[str, Any]
    fingerprint: str

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible target identity."""
        return {
            "behavior": dict(self.behavior),
            "fingerprint": self.fingerprint,
            "network_class": self.network_class.value,
            "target_id": self.target_id,
            "target_type": self.target_type,
        }


def installed_scenario_capabilities() -> tuple[ScenarioCapability, ...]:
    """Return the installed packaged experiment capability records."""
    from ctpf import experiment, kernel
    from ctpf.kernel import slice as kernel_slice
    from ctpf.kernel import trace as kernel_trace

    kernel_path = Path(kernel.__file__).parent
    shared_hash = _file_hash(Path(kernel_slice.__file__))
    experiment_hash = _file_hash(Path(experiment.__file__))
    cascade = _build_capability(
        scenario="cascade-memo",
        modes=(ExperimentMode.SINGLE, ExperimentMode.MATRIX),
        conditions=("baseline", "manipulated", "hardened"),
        sessions=6,
        prompts=(experiment.SESSION_A_PROMPT, experiment.SESSION_B_PROMPT),
        tools=(
            "apply_change",
            "read_inbox",
            "read_memo",
            "read_memo_meta",
            "read_sink",
            "write_memo",
        ),
        effects=("cascade-action-sink", "cascade-memo-persistence"),
        source_hashes={
            "experiment.py": experiment_hash,
            "cascade_memo_fixture.py": _file_hash(kernel_path / "cascade_memo_fixture.py"),
            "kernel/slice.py": shared_hash,
        },
    )
    pattern2 = _build_capability(
        scenario="pattern2",
        modes=(ExperimentMode.SINGLE,),
        conditions=("baseline", "manipulated", "hardened"),
        sessions=3,
        prompts=(experiment.PATTERN2_PROMPT,),
        tools=("apply_change", "read_sink", "read_status"),
        effects=("pattern2-action-sink",),
        source_hashes={
            "experiment.py": experiment_hash,
            "pattern2_fixture.py": _file_hash(kernel_path / "pattern2_fixture.py"),
            "kernel/slice.py": shared_hash,
        },
    )
    pattern3 = _build_capability(
        scenario="pattern3-scope",
        modes=(ExperimentMode.SINGLE,),
        conditions=("baseline", "opportunity", "hardened_opportunity"),
        sessions=3,
        prompts=(experiment.PATTERN3_PROMPT,),
        tools=("read_record", "read_sink", "write_record"),
        effects=("pattern3-write-sink",),
        source_hashes={
            "experiment.py": experiment_hash,
            "kernel/slice.py": shared_hash,
            "pattern3_scope.py": _file_hash(kernel_path / "pattern3_scope.py"),
            "pattern3_scope_fixture.py": _file_hash(kernel_path / "pattern3_scope_fixture.py"),
            "kernel/trace.py": _file_hash(Path(kernel_trace.__file__)),
        },
    )
    return (cascade, pattern2, pattern3)


def scenario_capability(scenario: str) -> ScenarioCapability:
    """Return one installed scenario capability by exact ID.

    Args:
        scenario: Exact packaged scenario ID.

    Returns:
        Installed scenario capability.

    Raises:
        TargetIdentityError: If the scenario is not installed.
    """
    for capability in installed_scenario_capabilities():
        if capability.scenario == scenario:
            return capability
    raise TargetIdentityError(f"unsupported packaged scenario: {scenario!r}")


def load_target_identity(target_id: str, *, db_path: Path | None = None) -> TargetIdentity:
    """Load and fingerprint one exact demonstrated target profile.

    Args:
        target_id: Full lowercase target UUID.
        db_path: Optional database path for tests.

    Returns:
        Validated behavior identity without credential values.

    Raises:
        TargetIdentityError: If the ID or target profile is invalid.
    """
    if not _FULL_ID.fullmatch(target_id):
        raise TargetIdentityError("target ID must be a full lowercase 32-character hex ID")
    try:
        profile = load_experiment_target_profile(target_id, db_path=db_path)
    except (ExternalRuntimeError, OSError, RuntimeError, ValueError) as exc:
        raise TargetIdentityError(str(exc)) from exc
    if profile.target_id != target_id:
        raise TargetIdentityError("resolved target ID does not match the requested full ID")
    return target_identity_from_profile(profile)


def target_identity_from_profile(
    profile: OpenAICompatibleTargetProfile | ClaudeCodeTargetProfile,
) -> TargetIdentity:
    """Fingerprint one already validated demonstrated target profile.

    Args:
        profile: Exact non-secret demonstrated target settings.

    Returns:
        Complete authority-bearing target identity.
    """
    if isinstance(profile, OpenAICompatibleTargetProfile):
        return _inference_identity(profile)
    if isinstance(profile, ClaudeCodeTargetProfile):
        return _runtime_identity(profile)
    raise TargetIdentityError(f"unsupported target profile: {type(profile).__name__}")


def target_identity_from_policy(target: TargetPolicy) -> TargetIdentity:
    """Validate one signed target snapshot without contacting its live target.

    Args:
        target: Policy target containing the complete non-secret behavior snapshot.

    Returns:
        Validated target identity suitable for deterministic policy evaluation.

    Raises:
        TargetIdentityError: If the snapshot is malformed, inconsistent, or unsafe.
    """
    behavior = dict(target.behavior)
    if sha256_digest(behavior) != target.target_fingerprint:
        raise TargetIdentityError("policy target behavior fingerprint is invalid")
    if behavior.get("target_id") != target.target_id:
        raise TargetIdentityError("policy target behavior has a mismatched target ID")
    if behavior.get("target_type") != target.target_type:
        raise TargetIdentityError("policy target behavior has a mismatched target type")
    _validate_policy_authority(target, behavior)
    if target.target_type == "inference":
        _validate_inference_snapshot(behavior, target.network_class)
    elif target.target_type == "agent-runtime":
        _validate_runtime_snapshot(behavior, target.network_class)
    else:
        raise TargetIdentityError(f"unsupported policy target type: {target.target_type!r}")
    return TargetIdentity(
        target.target_id,
        target.target_type,
        target.network_class,
        behavior,
        target.target_fingerprint,
    )


def execution_profile_from_policy(
    target: TargetPolicy,
) -> OpenAICompatibleTargetProfile | ClaudeCodeTargetProfile:
    """Reconstruct a non-secret execution profile from an authenticated snapshot."""
    identity = target_identity_from_policy(target)
    behavior = identity.behavior
    if identity.target_type == "inference":
        return _inference_profile_from_behavior(target, behavior)
    return _runtime_profile_from_behavior(target, behavior)


def _inference_profile_from_behavior(
    target: TargetPolicy,
    behavior: dict[str, Any],
) -> OpenAICompatibleTargetProfile:
    endpoint = behavior["endpoint"]
    generation = behavior["generation_parameters"]
    limits = behavior["limits"]
    temperature = generation["temperature"]
    return OpenAICompatibleTargetProfile(
        target_id=target.target_id,
        name=f"governed-{target.target_id[:8]}",
        endpoint=endpoint["normalized_url"],
        model=behavior["model"],
        credential_name=behavior["credential_alias"],
        max_tokens=limits["max_output_tokens"],
        temperature=float(temperature) if temperature is not None else None,
        seed=generation["seed"],
        reasoning_effort=generation["reasoning_effort"],
        max_input_tokens=limits["max_input_tokens"],
        billing_class=target.billing_class,
        request_cost_ceiling_microusd=target.request_cost_ceiling_microusd,
        data_egress_class=target.data_egress_class,
        retention_acknowledged=target.retention_acknowledged,
        residual_cost_acknowledged=target.residual_cost_acknowledged,
    )


def _runtime_profile_from_behavior(
    target: TargetPolicy,
    behavior: dict[str, Any],
) -> ClaudeCodeTargetProfile:
    return ClaudeCodeTargetProfile(
        target_id=target.target_id,
        name=f"governed-{target.target_id[:8]}",
        executable=behavior["executable"],
        model=behavior["model"],
        runtime_version=behavior["runtime_version"],
        timeout_seconds=behavior["timeout_seconds"],
        billing_class=target.billing_class,
        data_egress_class=target.data_egress_class,
        retention_acknowledged=target.retention_acknowledged,
        residual_cost_acknowledged=target.residual_cost_acknowledged,
    )


def classify_inference_endpoint(endpoint: str) -> NetworkClass:
    """Classify one normalized inference endpoint for the initial policy.

    Args:
        endpoint: Absolute OpenAI-compatible API base URL.

    Returns:
        Loopback or public-HTTPS network class.

    Raises:
        TargetIdentityError: If the endpoint requires an unsupported network
            authority or contains ambiguous URL components.
    """
    try:
        canonical = canonicalize_endpoint(endpoint)
    except ValueError as exc:
        raise TargetIdentityError(str(exc)) from exc
    return (
        NetworkClass.LOOPBACK
        if canonical.network_class == NetworkClass.LOOPBACK.value
        else NetworkClass.HTTPS_PUBLIC
    )


def _build_capability(  # noqa: PLR0913 - explicit immutable capability fields
    *,
    scenario: str,
    modes: tuple[ExperimentMode, ...],
    conditions: tuple[str, ...],
    sessions: int,
    prompts: tuple[str, ...],
    tools: tuple[str, ...],
    effects: tuple[str, ...],
    source_hashes: dict[str, str],
) -> ScenarioCapability:
    payload: dict[str, Any] = {
        "conditions": list(conditions),
        "contract_version": 2,
        "effect_ids": list(effects),
        "modes": [mode.value for mode in modes],
        "package_version": __version__,
        "prompts": list(prompts),
        "retry_policy": "none",
        "scenario": scenario,
        "sessions_per_trial": sessions,
        "source_hashes": dict(source_hashes),
        "supported_target_types": ["agent-runtime", "inference"],
        "tool_names": list(tools),
    }
    return ScenarioCapability(
        scenario=scenario,
        contract_version=2,
        modes=modes,
        conditions=conditions,
        sessions_per_trial=sessions,
        prompts=prompts,
        tool_names=tools,
        effect_ids=effects,
        supported_target_types=("agent-runtime", "inference"),
        retry_policy="none",
        package_version=__version__,
        source_hashes=source_hashes,
        fingerprint=sha256_digest(payload),
    )


def _inference_identity(profile: OpenAICompatibleTargetProfile) -> TargetIdentity:
    generation = profile.generation_parameters()
    endpoint = canonicalize_endpoint(profile.endpoint)
    network_class = classify_inference_endpoint(endpoint.normalized_url)
    _validate_inference_profile_authority(profile, network_class)
    behavior = {
        "authority_contract_version": 1,
        "billing": _billing_payload(
            profile.billing_class,
            profile.request_cost_ceiling_microusd,
            profile.residual_cost_acknowledged,
        ),
        "credential_alias": profile.credential_name,
        "data_egress": _egress_payload(
            profile.data_egress_class,
            profile.retention_acknowledged,
        ),
        "driver": "openai-compatible",
        "driver_source_hash": _file_hash(Path(driven_inference.__file__)),
        "endpoint": endpoint.to_payload(),
        "generation_parameters": {
            "reasoning_effort": generation.get("reasoning_effort"),
            "seed": generation.get("seed"),
            "temperature": _decimal_string(generation.get("temperature")),
        },
        "limits": {
            "max_input_tokens": profile.max_input_tokens,
            "max_output_tokens": profile.max_tokens,
            "max_provider_rounds": driven_inference.DEFAULT_MAX_ROUNDS,
            "max_request_bytes": hosted_inference.MAX_REQUEST_BYTES,
            "max_response_bytes": hosted_inference.MAX_RESPONSE_BYTES,
            "max_tool_calls_per_session": driven_inference.MAX_TOOL_CALLS_PER_SESSION,
        },
        "model": profile.model,
        "target_id": profile.target_id,
        "target_type": "inference",
        "transport": _transport_payload(endpoint),
    }
    return TargetIdentity(
        profile.target_id,
        "inference",
        network_class,
        behavior,
        sha256_digest(behavior),
    )


def _runtime_identity(profile: ClaudeCodeTargetProfile) -> TargetIdentity:
    if (
        profile.billing_class != BillingClass.EXTERNAL_RUNTIME
        or profile.data_egress_class != DataEgressClass.EXTERNAL_RUNTIME
        or not profile.retention_acknowledged
        or not profile.residual_cost_acknowledged
    ):
        raise TargetIdentityError("external runtime authority declarations are incomplete")
    executable = Path(profile.executable).resolve()
    executable_hash = _file_hash(executable)
    behavior = {
        "authority_contract_version": 1,
        "billing": _billing_payload(
            profile.billing_class,
            None,
            profile.residual_cost_acknowledged,
        ),
        "data_egress": _egress_payload(
            profile.data_egress_class,
            profile.retention_acknowledged,
        ),
        "driver": "claude-code-cli",
        "driver_source_hash": _file_hash(Path(external_runtime.__file__)),
        "environment_policy": "minimal non-secret allowlist",
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "identity_probe_processes": 1,
        "identity_probe_timeout_seconds": external_runtime.VERSION_PROBE_TIMEOUT_SECONDS,
        "mcp_policy": "strict loopback allowlisted-tools only",
        "model": profile.model,
        "runtime_version": profile.runtime_version,
        "target_id": profile.target_id,
        "target_type": "agent-runtime",
        "timeout_seconds": profile.timeout_seconds,
    }
    return TargetIdentity(
        profile.target_id,
        "agent-runtime",
        NetworkClass.EXTERNAL_RUNTIME,
        behavior,
        sha256_digest(behavior),
    )


def _validate_inference_profile_authority(
    profile: OpenAICompatibleTargetProfile,
    network_class: NetworkClass,
) -> None:
    if network_class == NetworkClass.LOOPBACK:
        if (
            profile.billing_class != BillingClass.UNMETERED
            or profile.request_cost_ceiling_microusd is not None
            or profile.data_egress_class != DataEgressClass.LOCAL_ONLY
            or profile.retention_acknowledged
            or profile.residual_cost_acknowledged
        ):
            raise TargetIdentityError("loopback inference authority declarations are invalid")
        return
    if profile.billing_class == BillingClass.EXTERNAL_RUNTIME:
        raise TargetIdentityError("hosted inference billing class is invalid")
    if profile.billing_class == BillingClass.METERED:
        ceiling = profile.request_cost_ceiling_microusd
        if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0:
            raise TargetIdentityError("metered hosted inference requires a request cost ceiling")
    elif profile.request_cost_ceiling_microusd is not None:
        raise TargetIdentityError("unmetered hosted inference must not declare request cost")
    if (
        profile.data_egress_class != DataEgressClass.PACKAGED_SYNTHETIC_REMOTE
        or not profile.retention_acknowledged
        or not profile.residual_cost_acknowledged
    ):
        raise TargetIdentityError("hosted inference authority declarations are incomplete")


def _validate_inference_snapshot(
    behavior: dict[str, Any],
    network_class: NetworkClass,
) -> None:
    _require_behavior_keys(behavior, _INFERENCE_BEHAVIOR_KEYS)
    if behavior["authority_contract_version"] != 1:
        raise TargetIdentityError("policy inference authority contract is unsupported")
    if behavior["driver"] != "openai-compatible":
        raise TargetIdentityError("policy inference driver is unsupported")
    _require_installed_driver_hash(behavior["driver_source_hash"], Path(driven_inference.__file__))
    endpoint = _validate_endpoint_snapshot(behavior["endpoint"])
    if classify_inference_endpoint(endpoint.normalized_url) != network_class:
        raise TargetIdentityError("policy inference network class does not match its endpoint")
    _require_text(behavior["model"], "model")
    _require_text(behavior["credential_alias"], "credential_alias")
    _validate_generation_snapshot(behavior["generation_parameters"])
    _validate_inference_limits(behavior["limits"])
    _validate_transport_snapshot(behavior["transport"], endpoint)
    _validate_billing_snapshot(behavior["billing"])
    _validate_egress_snapshot(behavior["data_egress"])


def _validate_runtime_snapshot(
    behavior: dict[str, Any],
    network_class: NetworkClass,
) -> None:
    _require_behavior_keys(behavior, _RUNTIME_BEHAVIOR_KEYS)
    if behavior["authority_contract_version"] != 1:
        raise TargetIdentityError("policy runtime authority contract is unsupported")
    if network_class != NetworkClass.EXTERNAL_RUNTIME:
        raise TargetIdentityError("policy runtime must use the external-runtime network class")
    if behavior["driver"] != "claude-code-cli":
        raise TargetIdentityError("policy external runtime driver is unsupported")
    if behavior["environment_policy"] != "minimal non-secret allowlist":
        raise TargetIdentityError("policy runtime environment policy is unsupported")
    if behavior["mcp_policy"] != "strict loopback allowlisted-tools only":
        raise TargetIdentityError("policy runtime MCP policy is unsupported")
    _require_installed_driver_hash(behavior["driver_source_hash"], Path(external_runtime.__file__))
    _require_digest(behavior["executable_sha256"], "executable_sha256")
    _require_absolute_path(behavior["executable"])
    _require_text(behavior["model"], "model")
    _require_text(behavior["runtime_version"], "runtime_version")
    _require_positive_int(behavior["timeout_seconds"], "timeout_seconds")
    if behavior["identity_probe_processes"] != 1:
        raise TargetIdentityError("policy runtime identity-probe process count is not installed")
    if behavior["identity_probe_timeout_seconds"] != external_runtime.VERSION_PROBE_TIMEOUT_SECONDS:
        raise TargetIdentityError("policy runtime identity-probe timeout is not installed")
    _validate_billing_snapshot(behavior["billing"])
    _validate_egress_snapshot(behavior["data_egress"])


def _billing_payload(
    billing_class: BillingClass,
    request_cost_ceiling_microusd: int | None,
    residual_cost_acknowledged: bool,
) -> dict[str, Any]:
    return {
        "billing_class": billing_class.value,
        "request_cost_ceiling_microusd": request_cost_ceiling_microusd,
        "residual_cost_acknowledged": residual_cost_acknowledged,
    }


def _egress_payload(
    data_egress_class: DataEgressClass,
    retention_acknowledged: bool,
) -> dict[str, Any]:
    return {
        "data_egress_class": data_egress_class.value,
        "retention_acknowledged": retention_acknowledged,
    }


def _transport_payload(endpoint: hosted_inference.CanonicalEndpoint) -> dict[str, Any]:
    return {
        "concurrent_requests": hosted_inference.MAX_CONCURRENT_REQUESTS,
        "deadlines_seconds": {
            "connect": hosted_inference.CONNECT_TIMEOUT_SECONDS,
            "overall": hosted_inference.OVERALL_TIMEOUT_SECONDS,
            "pool": hosted_inference.POOL_TIMEOUT_SECONDS,
            "read": hosted_inference.READ_TIMEOUT_SECONDS,
            "write": hosted_inference.WRITE_TIMEOUT_SECONDS,
        },
        "environment_proxy_policy": hosted_inference.ENVIRONMENT_PROXY_POLICY,
        "http_protocol": hosted_inference.HTTP_PROTOCOL,
        "httpcore_version": importlib.metadata.version("httpcore"),
        "httpx_version": importlib.metadata.version("httpx"),
        "max_attempts": hosted_inference.MAX_ATTEMPTS,
        "redirect_policy": hosted_inference.REDIRECT_POLICY,
        "retry_count": 0,
        "source_hashes": {
            "hosted_inference.py": _file_hash(Path(hosted_inference.__file__)),
            "llm_openai.py": _file_hash(Path(llm_openai.__file__)),
        },
        "tls_policy": (
            hosted_inference.TLS_POLICY if endpoint.scheme == "https" else "not_applicable"
        ),
    }


def _validate_policy_authority(target: TargetPolicy, behavior: dict[str, Any]) -> None:
    billing = behavior.get("billing")
    egress = behavior.get("data_egress")
    if not isinstance(billing, dict) or not isinstance(egress, dict):
        raise TargetIdentityError("policy target authority declarations are missing")
    expected_billing = _billing_payload(
        target.billing_class,
        target.request_cost_ceiling_microusd,
        target.residual_cost_acknowledged,
    )
    expected_egress = _egress_payload(
        target.data_egress_class,
        target.retention_acknowledged,
    )
    if billing != expected_billing or egress != expected_egress:
        raise TargetIdentityError("policy target authority differs from its fingerprint")


def _validate_endpoint_snapshot(raw: Any) -> hosted_inference.CanonicalEndpoint:
    if not isinstance(raw, dict):
        raise TargetIdentityError("policy inference endpoint must be an object")
    _require_behavior_keys(raw, _ENDPOINT_KEYS)
    normalized = _require_text(raw["normalized_url"], "endpoint.normalized_url")
    try:
        endpoint = canonicalize_endpoint(normalized)
    except ValueError as exc:
        raise TargetIdentityError(str(exc)) from exc
    if raw != endpoint.to_payload():
        raise TargetIdentityError("policy inference endpoint is not canonical")
    return endpoint


def _validate_inference_limits(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise TargetIdentityError("policy inference limits must be an object")
    _require_behavior_keys(raw, _INFERENCE_LIMIT_KEYS)
    _require_positive_int(raw["max_input_tokens"], "max_input_tokens")
    _require_positive_int(raw["max_output_tokens"], "max_output_tokens")
    installed = {
        "max_provider_rounds": driven_inference.DEFAULT_MAX_ROUNDS,
        "max_request_bytes": hosted_inference.MAX_REQUEST_BYTES,
        "max_response_bytes": hosted_inference.MAX_RESPONSE_BYTES,
        "max_tool_calls_per_session": driven_inference.MAX_TOOL_CALLS_PER_SESSION,
    }
    if any(raw[key] != value for key, value in installed.items()):
        raise TargetIdentityError("policy inference limits differ from installed controls")


def _validate_transport_snapshot(
    raw: Any,
    endpoint: hosted_inference.CanonicalEndpoint,
) -> None:
    if not isinstance(raw, dict):
        raise TargetIdentityError("policy inference transport must be an object")
    _require_behavior_keys(raw, _TRANSPORT_KEYS)
    if raw != _transport_payload(endpoint):
        raise TargetIdentityError("policy inference transport differs from installed controls")


def _validate_billing_snapshot(raw: Any) -> None:
    expected = {
        "billing_class",
        "request_cost_ceiling_microusd",
        "residual_cost_acknowledged",
    }
    if not isinstance(raw, dict):
        raise TargetIdentityError("policy target billing must be an object")
    _require_behavior_keys(raw, expected)
    _require_text(raw["billing_class"], "billing_class")
    ceiling = raw["request_cost_ceiling_microusd"]
    if ceiling is not None and (
        isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0
    ):
        raise TargetIdentityError("policy target request cost ceiling is invalid")
    if not isinstance(raw["residual_cost_acknowledged"], bool):
        raise TargetIdentityError("policy target residual-cost acknowledgement is invalid")


def _validate_egress_snapshot(raw: Any) -> None:
    expected = {"data_egress_class", "retention_acknowledged"}
    if not isinstance(raw, dict):
        raise TargetIdentityError("policy target data egress must be an object")
    _require_behavior_keys(raw, expected)
    _require_text(raw["data_egress_class"], "data_egress_class")
    if not isinstance(raw["retention_acknowledged"], bool):
        raise TargetIdentityError("policy target retention acknowledgement is invalid")


def _validate_generation_snapshot(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise TargetIdentityError("generation_parameters must be an object")
    expected = {"reasoning_effort", "seed", "temperature"}
    _require_behavior_keys(raw, expected)
    reasoning = raw["reasoning_effort"]
    if reasoning is not None:
        value = _require_text(reasoning, "reasoning_effort")
        if value not in _REASONING_EFFORTS:
            raise TargetIdentityError("policy target reasoning_effort is unsupported")
    seed = raw["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TargetIdentityError("seed must be an integer or null")
    temperature = raw["temperature"]
    if temperature is None:
        return
    value = _require_text(temperature, "temperature")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise TargetIdentityError("temperature must be a decimal string or null") from exc
    if not parsed.is_finite() or not _MIN_TEMPERATURE <= parsed <= _MAX_TEMPERATURE:
        raise TargetIdentityError("temperature must be finite and between 0 and 2")


def _require_behavior_keys(behavior: dict[str, Any], expected: set[str]) -> None:
    if set(behavior) != expected:
        raise TargetIdentityError("policy target behavior fields are incomplete or unknown")


def _require_text(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise TargetIdentityError(f"policy target {label} must be normalized non-empty text")
    return raw


def _require_digest(raw: Any, label: str) -> None:
    if not isinstance(raw, str) or not _DIGEST.fullmatch(raw):
        raise TargetIdentityError(f"policy target {label} must be a SHA-256 digest")


def _require_installed_driver_hash(raw: Any, path: Path) -> None:
    _require_digest(raw, "driver_source_hash")
    if raw != _file_hash(path):
        raise TargetIdentityError("policy target driver source is not the installed driver")


def _require_positive_int(raw: Any, label: str) -> None:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise TargetIdentityError(f"policy target {label} must be a positive integer")


def _require_absolute_path(raw: Any) -> None:
    value = _require_text(raw, "executable")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = posix.parts if posix.is_absolute() else windows.parts
    if not (posix.is_absolute() or windows.is_absolute()) or ".." in parts:
        raise TargetIdentityError("policy target executable must be an absolute local path")


def _decimal_string(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TargetIdentityError("target temperature must be numeric")
    value = format(raw, ".17g")
    return "0" if value in {"-0", "-0.0"} else value


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise TargetIdentityError(f"scenario source file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()
