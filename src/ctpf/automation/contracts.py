"""Typed versioned contracts for governed CTPF experiment automation."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Self, TypeVar

from ctpf.automation.canonical import CANONICALIZATION_ID

SCHEMA_VERSION = 1
SIGNING_ALGORITHM = "hmac-sha256"
_HEX_ID = re.compile(r"^[0-9a-f]{32}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_IDEMPOTENCY = re.compile(r"^[\x21-\x7e]{16,128}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class ContractError(ValueError):
    """Raised when an automation contract is malformed or unsupported."""


class AuthorizationTier(IntEnum):
    """Supported autonomous authorization tiers."""

    INSPECTION = 0
    LOCAL_SYNTHETIC = 1
    BOUNDED_REMOTE = 2
    ACTIVE_SIDE_EFFECT = 3


class ExperimentMode(StrEnum):
    """Supported packaged experiment execution modes."""

    SINGLE = "single"
    MATRIX = "matrix"


class BillingClass(StrEnum):
    """Human-declared target billing boundary."""

    UNMETERED = "unmetered"
    METERED = "metered"
    EXTERNAL_RUNTIME = "external_runtime"


class NetworkClass(StrEnum):
    """Initial autonomous target network classes."""

    LOOPBACK = "loopback"
    HTTPS_PUBLIC = "https_public"
    EXTERNAL_RUNTIME = "external_runtime"


class DecisionKind(StrEnum):
    """Deterministic policy decision kinds."""

    ALLOWED_STANDING_POLICY = "allowed_standing_policy"
    APPROVAL_REQUIRED = "approval_required"
    DENIED = "denied"


class GrantSource(StrEnum):
    """Source of an authenticated authorization grant."""

    STANDING_POLICY = "standing_policy"
    HUMAN_PER_RUN = "human_per_run"


class AutomationRunState(StrEnum):
    """Durable states for governed experiment controls."""

    READY = "READY"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class Requester:
    """Untrusted caller provenance attached to a run proposal."""

    kind: str
    name: str
    version: str

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {"kind": self.kind, "name": self.name, "version": self.version}


@dataclass(frozen=True)
class ResourceLimits:
    """Hard ceilings requested or authorized for one experiment control."""

    wall_clock_seconds: int
    provider_requests: int
    output_tokens_reserved: int
    tool_calls: int
    runtime_processes: int
    cost_limit_microusd: int

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            "cost_limit_microusd": self.cost_limit_microusd,
            "output_tokens_reserved": self.output_tokens_reserved,
            "provider_requests": self.provider_requests,
            "runtime_processes": self.runtime_processes,
            "tool_calls": self.tool_calls,
            "wall_clock_seconds": self.wall_clock_seconds,
        }

    def is_within(self, ceiling: ResourceLimits) -> bool:
        """Return whether every requested limit is within a ceiling."""
        return (
            self.wall_clock_seconds <= ceiling.wall_clock_seconds
            and self.provider_requests <= ceiling.provider_requests
            and self.output_tokens_reserved <= ceiling.output_tokens_reserved
            and self.tool_calls <= ceiling.tool_calls
            and self.runtime_processes <= ceiling.runtime_processes
            and self.cost_limit_microusd <= ceiling.cost_limit_microusd
        )


@dataclass(frozen=True)
class TargetReference:
    """Full target identity supplied by an untrusted run proposal."""

    target_id: str
    target_fingerprint: str

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            "target_fingerprint": self.target_fingerprint,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class ExperimentRequest:
    """Packaged experiment selection in a RunSpec."""

    scenario: str
    scenario_fingerprint: str
    mode: ExperimentMode
    trials_per_target: int
    targets: tuple[TargetReference, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            "mode": self.mode.value,
            "scenario": self.scenario,
            "scenario_fingerprint": self.scenario_fingerprint,
            "targets": [target.to_payload() for target in self.targets],
            "trials_per_target": self.trials_per_target,
        }


@dataclass(frozen=True)
class RunSpec:
    """Immutable autonomous experiment proposal."""

    idempotency_key: str
    requester: Requester
    purpose: str
    policy_id: str
    requested_tier: AuthorizationTier
    experiment: ExperimentRequest
    output_root_id: str
    limits: ResourceLimits
    schema_version: int = SCHEMA_VERSION
    canonicalization: str = CANONICALIZATION_ID

    def to_payload(self) -> dict[str, Any]:
        """Return the normalized canonical RunSpec object."""
        return {
            "canonicalization": self.canonicalization,
            "experiment": self.experiment.to_payload(),
            "idempotency_key": self.idempotency_key,
            "limits": self.limits.to_payload(),
            "output_root_id": self.output_root_id,
            "policy_id": self.policy_id,
            "purpose": self.purpose,
            "requested_tier": int(self.requested_tier),
            "requester": self.requester.to_payload(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        """Parse a strict RunSpec payload.

        Args:
            payload: Untrusted decoded JSON object.

        Returns:
            Validated RunSpec.
        """
        _require_shape(
            payload,
            required={
                "schema_version",
                "canonicalization",
                "idempotency_key",
                "requester",
                "purpose",
                "policy_id",
                "requested_tier",
                "experiment",
                "output_root_id",
                "limits",
            },
        )
        _validate_header(payload)
        return cls(
            idempotency_key=_idempotency(payload["idempotency_key"]),
            requester=_parse_requester(payload["requester"]),
            purpose=_bounded_text(payload["purpose"], "purpose", maximum=1_024),
            policy_id=_hex_id(payload["policy_id"], "policy_id"),
            requested_tier=_tier(payload["requested_tier"], allow_tier_three=False),
            experiment=_parse_experiment(payload["experiment"]),
            output_root_id=_safe_id(payload["output_root_id"], "output_root_id"),
            limits=_parse_limits(payload["limits"]),
        )


@dataclass(frozen=True)
class ScenarioPolicy:
    """Scenario fingerprints and modes approved by a human policy."""

    scenario: str
    fingerprints: tuple[str, ...]
    modes: tuple[ExperimentMode, ...]
    max_trials_per_target: int

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            "fingerprints": list(self.fingerprints),
            "max_trials_per_target": self.max_trials_per_target,
            "modes": [mode.value for mode in self.modes],
            "scenario": self.scenario,
        }


@dataclass(frozen=True)
class TargetPolicy:
    """Exact target authorization and cost/network classification."""

    target_id: str
    target_fingerprint: str
    network_class: NetworkClass
    billing_class: BillingClass
    request_cost_ceiling_microusd: int | None

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            "billing_class": self.billing_class.value,
            "network_class": self.network_class.value,
            "request_cost_ceiling_microusd": self.request_cost_ceiling_microusd,
            "target_fingerprint": self.target_fingerprint,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class OutputRootPolicy:
    """Human-approved logical evidence root mapping."""

    root_id: str
    resolved_path: str

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {"resolved_path": self.resolved_path, "root_id": self.root_id}


@dataclass(frozen=True)
class PolicyLimits:
    """Resource ceilings and local lifecycle controls in a signed policy."""

    resources: ResourceLimits
    concurrent_runs: int
    approval_lifetime_seconds: int
    loopback_port: int

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            **self.resources.to_payload(),
            "approval_lifetime_seconds": self.approval_lifetime_seconds,
            "concurrent_runs": self.concurrent_runs,
            "loopback_port": self.loopback_port,
        }


@dataclass(frozen=True)
class PolicyDocument:
    """Human-authored and authenticated autonomous execution policy."""

    policy_id: str
    name: str
    created_at: str
    expires_at: str
    standing_tiers: tuple[AuthorizationTier, ...]
    per_run_tiers: tuple[AuthorizationTier, ...]
    scenarios: tuple[ScenarioPolicy, ...]
    targets: tuple[TargetPolicy, ...]
    output_roots: tuple[OutputRootPolicy, ...]
    allowed_effects: tuple[str, ...]
    limits: PolicyLimits
    schema_version: int = SCHEMA_VERSION
    canonicalization: str = CANONICALIZATION_ID

    def to_payload(self) -> dict[str, Any]:
        """Return the normalized canonical policy object."""
        return {
            "allowed_effects": list(self.allowed_effects),
            "canonicalization": self.canonicalization,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "limits": self.limits.to_payload(),
            "name": self.name,
            "output_roots": [root.to_payload() for root in self.output_roots],
            "per_run_tiers": [int(tier) for tier in self.per_run_tiers],
            "policy_id": self.policy_id,
            "scenarios": [scenario.to_payload() for scenario in self.scenarios],
            "schema_version": self.schema_version,
            "standing_tiers": [int(tier) for tier in self.standing_tiers],
            "targets": [target.to_payload() for target in self.targets],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        """Parse a strict signed-policy body."""
        required = {
            "schema_version",
            "canonicalization",
            "policy_id",
            "name",
            "created_at",
            "expires_at",
            "standing_tiers",
            "per_run_tiers",
            "scenarios",
            "targets",
            "output_roots",
            "allowed_effects",
            "limits",
        }
        _require_shape(payload, required=required)
        _validate_header(payload)
        standing = _tier_list(payload["standing_tiers"], "standing_tiers")
        per_run = _tier_list(payload["per_run_tiers"], "per_run_tiers")
        if set(standing).intersection(per_run):
            raise ContractError("standing_tiers and per_run_tiers must not overlap")
        created_at = _timestamp(payload["created_at"], "created_at")
        expires_at = _timestamp(payload["expires_at"], "expires_at")
        if created_at >= expires_at:
            raise ContractError("policy expires_at must be later than created_at")
        return cls(
            policy_id=_hex_id(payload["policy_id"], "policy_id"),
            name=_bounded_text(payload["name"], "name", maximum=128),
            created_at=created_at,
            expires_at=expires_at,
            standing_tiers=standing,
            per_run_tiers=per_run,
            scenarios=_parse_scenarios(payload["scenarios"]),
            targets=_parse_target_policies(payload["targets"]),
            output_roots=_parse_output_roots(payload["output_roots"]),
            allowed_effects=_safe_id_list(payload["allowed_effects"], "allowed_effects"),
            limits=_parse_policy_limits(payload["limits"]),
        )


@dataclass(frozen=True)
class AuthorizationGrant:
    """Authenticated authorization bound to one exact RunSpec."""

    grant_id: str
    source: GrantSource
    spec_digest: str
    policy_id: str
    policy_digest: str
    scenario_fingerprint: str
    targets: tuple[TargetReference, ...]
    authorized_tier: AuthorizationTier
    limits: ResourceLimits
    issued_at: str
    expires_at: str
    nonce: str
    key_id: str
    signing_algorithm: str = SIGNING_ALGORITHM
    schema_version: int = SCHEMA_VERSION
    canonicalization: str = CANONICALIZATION_ID

    def to_payload(self) -> dict[str, Any]:
        """Return the normalized canonical grant body."""
        return {
            "authorized_tier": int(self.authorized_tier),
            "canonicalization": self.canonicalization,
            "expires_at": self.expires_at,
            "grant_id": self.grant_id,
            "issued_at": self.issued_at,
            "key_id": self.key_id,
            "limits": self.limits.to_payload(),
            "nonce": self.nonce,
            "policy_digest": self.policy_digest,
            "policy_id": self.policy_id,
            "scenario_fingerprint": self.scenario_fingerprint,
            "schema_version": self.schema_version,
            "signing_algorithm": self.signing_algorithm,
            "source": self.source.value,
            "spec_digest": self.spec_digest,
            "targets": [target.to_payload() for target in self.targets],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        """Parse a strict authorization-grant body."""
        required = {
            "schema_version",
            "canonicalization",
            "grant_id",
            "source",
            "spec_digest",
            "policy_id",
            "policy_digest",
            "scenario_fingerprint",
            "targets",
            "authorized_tier",
            "limits",
            "issued_at",
            "expires_at",
            "nonce",
            "key_id",
            "signing_algorithm",
        }
        _require_shape(payload, required=required)
        _validate_header(payload)
        if payload["signing_algorithm"] != SIGNING_ALGORITHM:
            raise ContractError("unsupported signing_algorithm")
        issued_at = _timestamp(payload["issued_at"], "issued_at")
        expires_at = _timestamp(payload["expires_at"], "expires_at")
        if issued_at >= expires_at:
            raise ContractError("grant expires_at must be later than issued_at")
        return cls(
            grant_id=_hex_id(payload["grant_id"], "grant_id"),
            source=_enum_value(GrantSource, payload["source"], "source"),
            spec_digest=_digest(payload["spec_digest"], "spec_digest"),
            policy_id=_hex_id(payload["policy_id"], "policy_id"),
            policy_digest=_digest(payload["policy_digest"], "policy_digest"),
            scenario_fingerprint=_digest(payload["scenario_fingerprint"], "scenario_fingerprint"),
            targets=_parse_target_references(payload["targets"]),
            authorized_tier=_tier(payload["authorized_tier"], allow_tier_three=False),
            limits=_parse_limits(payload["limits"]),
            issued_at=issued_at,
            expires_at=expires_at,
            nonce=_digest(payload["nonce"], "nonce"),
            key_id=_digest(payload["key_id"], "key_id"),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Deterministic result of evaluating one RunSpec against one policy."""

    kind: DecisionKind
    reason_code: str
    spec_digest: str
    policy_digest: str
    minimum_reservations: ResourceLimits
    warnings: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""
        return {
            "kind": self.kind.value,
            "minimum_reservations": self.minimum_reservations.to_payload(),
            "policy_digest": self.policy_digest,
            "reason_code": self.reason_code,
            "spec_digest": self.spec_digest,
            "warnings": list(self.warnings),
        }


def _validate_header(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("canonicalization") != CANONICALIZATION_ID:
        raise ContractError(f"canonicalization must be {CANONICALIZATION_ID!r}")


def _require_shape(payload: Any, *, required: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("contract value must be a JSON object")
    keys = set(payload)
    missing = required.difference(keys)
    unknown = keys.difference(required)
    if missing:
        raise ContractError(f"missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")
    return payload


def _parse_requester(raw: Any) -> Requester:
    payload = _require_shape(raw, required={"kind", "name", "version"})
    return Requester(
        _safe_id(payload["kind"], "requester.kind"),
        _bounded_text(payload["name"], "requester.name", maximum=128),
        _bounded_text(payload["version"], "requester.version", maximum=128),
    )


def _parse_experiment(raw: Any) -> ExperimentRequest:
    required = {"scenario", "scenario_fingerprint", "mode", "trials_per_target", "targets"}
    payload = _require_shape(raw, required=required)
    scenario = _safe_id(payload["scenario"], "experiment.scenario")
    mode = _enum_value(ExperimentMode, payload["mode"], "experiment.mode")
    trials = _bounded_int(payload["trials_per_target"], "trials_per_target", 1, 5)
    targets = _parse_target_references(payload["targets"])
    if mode == ExperimentMode.SINGLE and (trials != 1 or len(targets) != 1):
        raise ContractError("single mode requires one target and one trial")
    if mode == ExperimentMode.MATRIX and (scenario != "cascade-memo" or trials < 3):
        raise ContractError("matrix mode requires cascade-memo and 3-5 trials")
    if mode == ExperimentMode.MATRIX and len(targets) < 2:
        raise ContractError("matrix mode requires at least two targets")
    return ExperimentRequest(
        scenario,
        _digest(payload["scenario_fingerprint"], "scenario_fingerprint"),
        mode,
        trials,
        targets,
    )


def _parse_target_references(raw: Any) -> tuple[TargetReference, ...]:
    items = _object_list(raw, "targets")
    targets = tuple(_parse_target_reference(item) for item in items)
    ids = [target.target_id for target in targets]
    if len(set(ids)) != len(ids):
        raise ContractError("targets must contain distinct target IDs")
    return targets


def _parse_target_reference(payload: dict[str, Any]) -> TargetReference:
    _require_shape(payload, required={"target_id", "target_fingerprint"})
    return TargetReference(
        _hex_id(payload["target_id"], "target_id"),
        _digest(payload["target_fingerprint"], "target_fingerprint"),
    )


def _parse_limits(raw: Any) -> ResourceLimits:
    required = {
        "wall_clock_seconds",
        "provider_requests",
        "output_tokens_reserved",
        "tool_calls",
        "runtime_processes",
        "cost_limit_microusd",
    }
    payload = _require_shape(raw, required=required)
    return ResourceLimits(
        wall_clock_seconds=_positive_int(payload["wall_clock_seconds"], "wall_clock_seconds"),
        provider_requests=_positive_int(payload["provider_requests"], "provider_requests"),
        output_tokens_reserved=_positive_int(
            payload["output_tokens_reserved"], "output_tokens_reserved"
        ),
        tool_calls=_positive_int(payload["tool_calls"], "tool_calls"),
        runtime_processes=_positive_int(payload["runtime_processes"], "runtime_processes"),
        cost_limit_microusd=_bounded_int(
            payload["cost_limit_microusd"], "cost_limit_microusd", 0, (2**63) - 1
        ),
    )


def _parse_scenarios(raw: Any) -> tuple[ScenarioPolicy, ...]:
    scenarios = tuple(_parse_scenario_policy(item) for item in _object_list(raw, "scenarios"))
    _require_unique([item.scenario for item in scenarios], "scenario policy IDs")
    return scenarios


def _parse_scenario_policy(payload: dict[str, Any]) -> ScenarioPolicy:
    required = {"scenario", "fingerprints", "modes", "max_trials_per_target"}
    _require_shape(payload, required=required)
    fingerprints = _digest_list(payload["fingerprints"], "fingerprints")
    modes = _enum_list(ExperimentMode, payload["modes"], "modes")
    return ScenarioPolicy(
        _safe_id(payload["scenario"], "scenario"),
        fingerprints,
        modes,
        _bounded_int(payload["max_trials_per_target"], "max_trials_per_target", 1, 5),
    )


def _parse_target_policies(raw: Any) -> tuple[TargetPolicy, ...]:
    targets = tuple(_parse_target_policy(item) for item in _object_list(raw, "targets"))
    _require_unique([item.target_id for item in targets], "target policy IDs")
    return targets


def _parse_target_policy(payload: dict[str, Any]) -> TargetPolicy:
    required = {
        "target_id",
        "target_fingerprint",
        "network_class",
        "billing_class",
        "request_cost_ceiling_microusd",
    }
    _require_shape(payload, required=required)
    billing = _enum_value(BillingClass, payload["billing_class"], "billing_class")
    ceiling = _optional_nonnegative_int(
        payload["request_cost_ceiling_microusd"], "request_cost_ceiling_microusd"
    )
    if billing == BillingClass.METERED and ceiling is None:
        raise ContractError("metered targets require request_cost_ceiling_microusd")
    if billing != BillingClass.METERED and ceiling is not None:
        raise ContractError("only metered targets may set request_cost_ceiling_microusd")
    return TargetPolicy(
        _hex_id(payload["target_id"], "target_id"),
        _digest(payload["target_fingerprint"], "target_fingerprint"),
        _enum_value(NetworkClass, payload["network_class"], "network_class"),
        billing,
        ceiling,
    )


def _parse_output_roots(raw: Any) -> tuple[OutputRootPolicy, ...]:
    roots = tuple(_parse_output_root(item) for item in _object_list(raw, "output_roots"))
    _require_unique([root.root_id for root in roots], "output root IDs")
    return roots


def _parse_output_root(payload: dict[str, Any]) -> OutputRootPolicy:
    _require_shape(payload, required={"root_id", "resolved_path"})
    return OutputRootPolicy(
        _safe_id(payload["root_id"], "root_id"),
        _absolute_local_path(payload["resolved_path"]),
    )


def _absolute_local_path(raw: Any) -> str:
    value = _bounded_text(raw, "resolved_path", maximum=4_096)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    is_unc = value.startswith(("//", "\\\\"))
    if is_unc or not (posix.is_absolute() or windows.is_absolute()):
        raise ContractError("resolved_path must be an absolute local filesystem path")
    parts = posix.parts if posix.is_absolute() else windows.parts
    if ".." in parts:
        raise ContractError("resolved_path must not contain parent traversal")
    return value


def _parse_policy_limits(raw: Any) -> PolicyLimits:
    required = set(ResourceLimits.__dataclass_fields__) | {
        "concurrent_runs",
        "approval_lifetime_seconds",
        "loopback_port",
    }
    payload = _require_shape(raw, required=required)
    resource_payload = {key: payload[key] for key in ResourceLimits.__dataclass_fields__}
    concurrent = _bounded_int(payload["concurrent_runs"], "concurrent_runs", 1, 1)
    return PolicyLimits(
        resources=_parse_limits(resource_payload),
        concurrent_runs=concurrent,
        approval_lifetime_seconds=_bounded_int(
            payload["approval_lifetime_seconds"], "approval_lifetime_seconds", 60, 86_400
        ),
        loopback_port=_bounded_int(payload["loopback_port"], "loopback_port", 1, 65_535),
    )


def _tier_list(raw: Any, label: str) -> tuple[AuthorizationTier, ...]:
    values = _list(raw, label)
    tiers = tuple(_tier(item, allow_tier_three=False) for item in values)
    _require_unique([str(int(tier)) for tier in tiers], label)
    return tiers


def _tier(raw: Any, *, allow_tier_three: bool) -> AuthorizationTier:
    value = _bounded_int(raw, "authorization tier", 0, 3 if allow_tier_three else 2)
    return AuthorizationTier(value)


def _enum_value(enum_type: type[_StrEnumT], raw: Any, label: str) -> _StrEnumT:
    if not isinstance(raw, str):
        raise ContractError(f"{label} must be a string")
    try:
        return enum_type(raw)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ContractError(f"{label} must be one of: {choices}") from exc


def _enum_list(enum_type: type[_StrEnumT], raw: Any, label: str) -> tuple[_StrEnumT, ...]:
    values = tuple(_enum_value(enum_type, item, label) for item in _list(raw, label))
    _require_unique([item.value for item in values], label)
    return values


def _object_list(raw: Any, label: str) -> tuple[dict[str, Any], ...]:
    values = _list(raw, label)
    if not all(isinstance(item, dict) for item in values):
        raise ContractError(f"{label} must contain only objects")
    return tuple(values)


def _list(raw: Any, label: str) -> list[Any]:
    if not isinstance(raw, list) or not raw:
        raise ContractError(f"{label} must be a non-empty array")
    return raw


def _digest_list(raw: Any, label: str) -> tuple[str, ...]:
    values = tuple(_digest(item, label) for item in _list(raw, label))
    _require_unique(list(values), label)
    return values


def _safe_id_list(raw: Any, label: str) -> tuple[str, ...]:
    values = tuple(_safe_id(item, label) for item in _list(raw, label))
    _require_unique(list(values), label)
    return values


def _require_unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ContractError(f"{label} must be unique")


def _bounded_text(raw: Any, label: str, *, maximum: int) -> str:
    if not isinstance(raw, str) or not raw.strip() or len(raw) > maximum:
        raise ContractError(f"{label} must contain 1-{maximum} characters")
    return raw.strip()


def _safe_id(raw: Any, label: str) -> str:
    value = _bounded_text(raw, label, maximum=64)
    if not _SAFE_ID.fullmatch(value):
        raise ContractError(f"{label} contains unsupported characters")
    return value


def _hex_id(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not _HEX_ID.fullmatch(raw):
        raise ContractError(f"{label} must be a full lowercase 32-character hexadecimal ID")
    return raw


def _digest(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not _DIGEST.fullmatch(raw):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return raw


def _idempotency(raw: Any) -> str:
    if not isinstance(raw, str) or not _IDEMPOTENCY.fullmatch(raw):
        raise ContractError("idempotency_key must be 16-128 printable ASCII characters")
    return raw


def _timestamp(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not _TIMESTAMP.fullmatch(raw):
        raise ContractError(f"{label} must be a UTC RFC 3339 timestamp with second precision")
    try:
        datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
    except ValueError as exc:
        raise ContractError(f"{label} must contain a valid calendar timestamp") from exc
    return raw


def _positive_int(raw: Any, label: str) -> int:
    return _bounded_int(raw, label, 1, (2**63) - 1)


def _optional_nonnegative_int(raw: Any, label: str) -> int | None:
    if raw is None:
        return None
    return _bounded_int(raw, label, 0, (2**63) - 1)


def _bounded_int(raw: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ContractError(f"{label} must be an integer")
    if not minimum <= raw <= maximum:
        raise ContractError(f"{label} must be between {minimum} and {maximum}")
    return int(raw)
