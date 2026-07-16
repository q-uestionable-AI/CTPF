"""Scenario and target identity fingerprints for governed automation."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ctpf import __version__, driven_inference, external_runtime
from ctpf.automation.canonical import sha256_digest
from ctpf.automation.contracts import ExperimentMode, NetworkClass
from ctpf.driven_inference import OpenAICompatibleTargetProfile
from ctpf.external_runtime import (
    ClaudeCodeTargetProfile,
    ExternalRuntimeError,
    load_experiment_target_profile,
)

_FULL_ID = re.compile(r"^[0-9a-f]{32}$")


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
    """Return the two installed packaged experiment capability records."""
    from ctpf import experiment, kernel
    from ctpf.kernel import slice as kernel_slice

    kernel_path = Path(kernel.__file__).parent
    shared_hash = _file_hash(Path(kernel_slice.__file__))
    experiment_hash = _file_hash(Path(experiment.__file__))
    cascade = _build_capability(
        scenario="cascade-memo",
        modes=(ExperimentMode.SINGLE, ExperimentMode.MATRIX),
        sessions=2,
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
        sessions=1,
        prompts=(experiment.PATTERN2_PROMPT,),
        tools=("apply_change", "read_sink", "read_status"),
        effects=("pattern2-action-sink",),
        source_hashes={
            "experiment.py": experiment_hash,
            "pattern2_fixture.py": _file_hash(kernel_path / "pattern2_fixture.py"),
            "kernel/slice.py": shared_hash,
        },
    )
    return (cascade, pattern2)


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
    if isinstance(profile, OpenAICompatibleTargetProfile):
        return _inference_identity(profile)
    if isinstance(profile, ClaudeCodeTargetProfile):
        return _runtime_identity(profile)
    raise TargetIdentityError(f"unsupported target profile: {type(profile).__name__}")


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
        parsed = urlparse(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise TargetIdentityError(f"invalid inference endpoint: {exc}") from exc
    if parsed.username is not None or parsed.password is not None:
        raise TargetIdentityError("inference endpoint must not contain credentials")
    if parsed.query or parsed.fragment or not parsed.hostname:
        raise TargetIdentityError("inference endpoint must not contain query or fragment data")
    loopback = _is_loopback_host(parsed.hostname)
    if parsed.scheme == "http" and loopback:
        return NetworkClass.LOOPBACK
    if parsed.scheme != "https" or loopback:
        raise TargetIdentityError("non-loopback inference endpoints must use HTTPS")
    _validate_public_host(parsed.hostname)
    _ = port
    return NetworkClass.HTTPS_PUBLIC


def _build_capability(  # noqa: PLR0913 - explicit immutable capability fields
    *,
    scenario: str,
    modes: tuple[ExperimentMode, ...],
    sessions: int,
    prompts: tuple[str, ...],
    tools: tuple[str, ...],
    effects: tuple[str, ...],
    source_hashes: dict[str, str],
) -> ScenarioCapability:
    payload: dict[str, Any] = {
        "conditions": ["baseline", "manipulated", "hardened"],
        "contract_version": 1,
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
        contract_version=1,
        modes=modes,
        conditions=("baseline", "manipulated", "hardened"),
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
    behavior = {
        "credential_alias": profile.credential_name,
        "driver": "openai-compatible",
        "driver_source_hash": _file_hash(Path(driven_inference.__file__)),
        "endpoint": profile.endpoint,
        "generation_parameters": {
            "reasoning_effort": generation.get("reasoning_effort"),
            "seed": generation.get("seed"),
            "temperature": _decimal_string(generation.get("temperature")),
        },
        "max_tokens": profile.max_tokens,
        "max_provider_rounds": driven_inference.DEFAULT_MAX_ROUNDS,
        "model": profile.model,
        "target_id": profile.target_id,
        "target_type": "inference",
    }
    network_class = classify_inference_endpoint(profile.endpoint)
    return TargetIdentity(
        profile.target_id,
        "inference",
        network_class,
        behavior,
        sha256_digest(behavior),
    )


def _runtime_identity(profile: ClaudeCodeTargetProfile) -> TargetIdentity:
    executable = Path(profile.executable).resolve()
    executable_hash = _file_hash(executable)
    behavior = {
        "driver": "claude-code-cli",
        "driver_source_hash": _file_hash(Path(external_runtime.__file__)),
        "environment_policy": "minimal non-secret allowlist",
        "executable": str(executable),
        "executable_sha256": executable_hash,
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


def _decimal_string(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TargetIdentityError("target temperature must be numeric")
    value = format(raw, ".17g")
    return "0" if value in {"-0", "-0.0"} else value


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_public_host(host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            encoded = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise TargetIdentityError("inference endpoint hostname is invalid") from exc
        if not encoded or "." not in encoded:
            raise TargetIdentityError(
                "public HTTPS endpoint requires a fully qualified hostname"
            ) from None
        return
    if not address.is_global:
        raise TargetIdentityError("HTTPS endpoint IP must be globally routable")


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise TargetIdentityError(f"scenario source file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()
