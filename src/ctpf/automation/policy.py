"""Deterministic policy evaluation for governed experiment proposals."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass

from ctpf.automation.canonical import sha256_digest
from ctpf.automation.contracts import (
    AuthorizationTier,
    BillingClass,
    DecisionKind,
    ExperimentMode,
    NetworkClass,
    PolicyDecision,
    PolicyDocument,
    ResourceLimits,
    RunSpec,
    TargetPolicy,
)
from ctpf.automation.targets import ScenarioCapability, TargetIdentity

_MAX_INTEGER = (2**63) - 1
_ZERO_RESERVATIONS = ResourceLimits(1, 1, 1, 1, 1, 1, 0)
_MATRIX_TARGET_TYPE = "inference"


@dataclass
class _ReservationTotals:
    provider_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    runtime_processes: int = 0
    wall_clock: int = 0
    cost: int | None = 0


def evaluate_policy(  # noqa: PLR0911 - explicit fail-closed guard sequence
    spec: RunSpec,
    policy: PolicyDocument,
    capability: ScenarioCapability,
    target_identities: Iterable[TargetIdentity],
    *,
    now: datetime.datetime | None = None,
) -> PolicyDecision:
    """Evaluate one normalized RunSpec without performing external actions.

    Args:
        spec: Strict immutable experiment proposal.
        policy: Human-authored policy body whose signature is verified separately.
        capability: Installed scenario capability selected by the proposal.
        target_identities: Locally resolved target identities.
        now: Optional aware UTC evaluation time for deterministic tests.

    Returns:
        Allowed, approval-required, or denied policy decision.
    """
    digest = sha256_digest(policy.to_payload())
    spec_digest = sha256_digest(spec.to_payload())
    reason = _validate_policy_header(spec, policy, now or datetime.datetime.now(datetime.UTC))
    if reason:
        return _denied(reason, spec_digest, digest)
    reason = _validate_scenario(spec, policy, capability)
    if reason:
        return _denied(reason, spec_digest, digest)
    identities = tuple(target_identities)
    reason = _validate_targets(spec, policy, capability, identities)
    if reason:
        return _denied(reason, spec_digest, digest)
    reason = _validate_output_and_resources(spec, policy)
    if reason:
        return _denied(reason, spec_digest, digest)
    reservations, warnings, reason = _minimum_reservations(spec, policy, capability, identities)
    if reason:
        return _denied(reason, spec_digest, digest)
    if not reservations.is_within(spec.limits):
        return _denied("requested_limits_below_minimum", spec_digest, digest, reservations)
    if (
        spec.requested_tier == AuthorizationTier.BOUNDED_REMOTE
        and spec.requested_tier in policy.standing_tiers
        and spec_digest not in policy.standing_run_spec_digests
    ):
        return _denied("standing_run_spec_not_authorized", spec_digest, digest, reservations)
    kind = _decision_kind(spec.requested_tier, policy)
    if kind is None:
        return _denied("tier_not_authorized", spec_digest, digest, reservations)
    return PolicyDecision(kind, "policy_match", spec_digest, digest, reservations, warnings)


def _validate_policy_header(  # noqa: PLR0911 - reason-specific guard sequence
    spec: RunSpec,
    policy: PolicyDocument,
    now: datetime.datetime,
) -> str | None:
    if spec.policy_id != policy.policy_id:
        return "policy_id_mismatch"
    if now.tzinfo is None or now.utcoffset() is None:
        return "evaluation_time_not_aware"
    try:
        created = _parse_timestamp(policy.created_at)
        expires = _parse_timestamp(policy.expires_at)
    except ValueError:
        return "policy_timestamp_invalid"
    if created >= expires:
        return "policy_interval_invalid"
    if now.astimezone(datetime.UTC) < created:
        return "policy_not_yet_valid"
    if now.astimezone(datetime.UTC) >= expires:
        return "policy_expired"
    if spec.requested_tier == AuthorizationTier.INSPECTION:
        return "inspection_tier_cannot_execute"
    if spec.requested_tier == AuthorizationTier.ACTIVE_SIDE_EFFECT:
        return "active_side_effect_tier_unsupported"
    return None


def _validate_scenario(  # noqa: PLR0911 - reason-specific guard sequence
    spec: RunSpec,
    policy: PolicyDocument,
    capability: ScenarioCapability,
) -> str | None:
    experiment = spec.experiment
    if capability.scenario != experiment.scenario:
        return "scenario_capability_mismatch"
    if capability.fingerprint != experiment.scenario_fingerprint:
        return "scenario_fingerprint_mismatch"
    if experiment.mode not in capability.modes:
        return "scenario_mode_not_installed"
    selected = next(
        (item for item in policy.scenarios if item.scenario == experiment.scenario),
        None,
    )
    if selected is None:
        return "scenario_not_authorized"
    if experiment.scenario_fingerprint not in selected.fingerprints:
        return "scenario_fingerprint_not_authorized"
    if experiment.mode not in selected.modes:
        return "scenario_mode_not_authorized"
    if experiment.trials_per_target > selected.max_trials_per_target:
        return "trial_limit_exceeded"
    if not set(capability.effect_ids).issubset(policy.allowed_effects):
        return "scenario_effect_not_authorized"
    return None


def _validate_targets(  # noqa: PLR0911 - reason-specific guard sequence
    spec: RunSpec,
    policy: PolicyDocument,
    capability: ScenarioCapability,
    identities: tuple[TargetIdentity, ...],
) -> str | None:
    if len(identities) != len(spec.experiment.targets):
        return "target_identity_count_mismatch"
    by_id = {identity.target_id: identity for identity in identities}
    if len(by_id) != len(identities):
        return "target_identity_duplicate"
    policies = {item.target_id: item for item in policy.targets}
    for reference in spec.experiment.targets:
        identity = by_id.get(reference.target_id)
        if identity is None or identity.fingerprint != reference.target_fingerprint:
            return "target_identity_mismatch"
        if identity.target_type not in capability.supported_target_types:
            return "target_type_not_supported"
        if (
            spec.experiment.mode == ExperimentMode.MATRIX
            and identity.target_type != _MATRIX_TARGET_TYPE
        ):
            return "target_type_not_supported"
        target_policy = policies.get(reference.target_id)
        reason = _target_policy_reason(reference.target_fingerprint, identity, target_policy)
        if reason:
            return reason
        if _minimum_tier(identity, target_policy) > spec.requested_tier:
            return "target_requires_higher_tier"
    return None


def _target_policy_reason(  # noqa: PLR0911 - reason-specific fail-closed checks
    fingerprint: str,
    identity: TargetIdentity,
    target_policy: TargetPolicy | None,
) -> str | None:
    if target_policy is None:
        return "target_not_authorized"
    if target_policy.target_fingerprint != fingerprint:
        return "target_fingerprint_not_authorized"
    if target_policy.target_type != identity.target_type:
        return "target_type_not_authorized"
    if target_policy.behavior != identity.behavior:
        return "target_behavior_not_authorized"
    if target_policy.network_class != identity.network_class:
        return "target_network_class_mismatch"
    if identity.network_class == NetworkClass.EXTERNAL_RUNTIME:
        if target_policy.billing_class != BillingClass.EXTERNAL_RUNTIME:
            return "target_billing_class_mismatch"
    elif target_policy.billing_class == BillingClass.EXTERNAL_RUNTIME:
        return "target_billing_class_mismatch"
    return None


def _minimum_tier(
    identity: TargetIdentity,
    target_policy: TargetPolicy | None,
) -> AuthorizationTier:
    if target_policy is None:
        return AuthorizationTier.ACTIVE_SIDE_EFFECT
    if identity.network_class != NetworkClass.LOOPBACK:
        return AuthorizationTier.BOUNDED_REMOTE
    if target_policy.billing_class == BillingClass.METERED:
        return AuthorizationTier.BOUNDED_REMOTE
    return AuthorizationTier.LOCAL_SYNTHETIC


def _validate_output_and_resources(spec: RunSpec, policy: PolicyDocument) -> str | None:
    if not any(root.root_id == spec.output_root_id for root in policy.output_roots):
        return "output_root_not_authorized"
    if not spec.limits.is_within(policy.limits.resources):
        return "policy_resource_limit_exceeded"
    return None


def _minimum_reservations(
    spec: RunSpec,
    policy: PolicyDocument,
    capability: ScenarioCapability,
    identities: tuple[TargetIdentity, ...],
) -> tuple[ResourceLimits, tuple[str, ...], str | None]:
    sessions_per_target = capability.sessions_per_trial * spec.experiment.trials_per_target
    totals = _ReservationTotals()
    warnings: list[str] = []
    target_policies = {item.target_id: item for item in policy.targets}
    for identity in identities:
        reason = _reserve_target(
            totals,
            identity,
            target_policies[identity.target_id],
            sessions_per_target,
            warnings,
        )
        if reason:
            return _ZERO_RESERVATIONS, (), reason
    values = (
        totals.provider_requests,
        totals.input_tokens,
        totals.output_tokens,
        totals.tool_calls,
        totals.runtime_processes,
        totals.wall_clock,
    )
    if any(value > _MAX_INTEGER for value in values):
        return _ZERO_RESERVATIONS, (), "minimum_reservation_overflow"
    if totals.cost is not None and totals.cost > _MAX_INTEGER:
        return _ZERO_RESERVATIONS, (), "minimum_reservation_overflow"
    reservations = ResourceLimits(
        wall_clock_seconds=max(1, totals.wall_clock),
        provider_requests=max(1, totals.provider_requests),
        input_tokens_reserved=max(1, totals.input_tokens),
        output_tokens_reserved=max(1, totals.output_tokens),
        tool_calls=max(1, totals.tool_calls),
        runtime_processes=max(1, totals.runtime_processes),
        cost_limit_microusd=totals.cost,
    )
    return reservations, tuple(sorted(set(warnings))), None


def _reserve_target(
    totals: _ReservationTotals,
    identity: TargetIdentity,
    target_policy: TargetPolicy,
    sessions_per_target: int,
    warnings: list[str],
) -> str | None:
    if identity.target_type == "inference":
        values, reason = _inference_reservations(identity, sessions_per_target)
        if reason:
            return reason
        requests, input_tokens, output_tokens, tools, wall_clock = values
        totals.input_tokens += input_tokens
        totals.output_tokens += output_tokens
        totals.tool_calls += tools
    else:
        processes, wall_clock, reason = _runtime_reservations(identity, sessions_per_target)
        if reason:
            return reason
        requests = sessions_per_target
        totals.runtime_processes += processes
        totals.tool_calls += sessions_per_target
        totals.cost = None
        warnings.append("external runtime residual cost is acknowledged but not measurable")
    totals.provider_requests += requests
    totals.wall_clock += wall_clock
    ceiling = target_policy.request_cost_ceiling_microusd
    if ceiling is not None and totals.cost is not None:
        totals.cost += ceiling * requests
    return None


def _inference_reservations(
    identity: TargetIdentity,
    sessions_per_target: int,
) -> tuple[tuple[int, int, int, int, int], str | None]:
    limits = identity.behavior.get("limits")
    transport = identity.behavior.get("transport")
    if not isinstance(limits, dict) or not isinstance(transport, dict):
        return (0, 0, 0, 0, 0), "target_inference_authority_invalid"
    max_rounds = _positive_mapping_int(limits, "max_provider_rounds")
    max_input = _positive_mapping_int(limits, "max_input_tokens")
    max_output = _positive_mapping_int(limits, "max_output_tokens")
    max_tools = _positive_mapping_int(limits, "max_tool_calls_per_session")
    attempts = _positive_mapping_int(transport, "max_attempts")
    deadlines = transport.get("deadlines_seconds")
    overall = _positive_mapping_int(deadlines, "overall") if isinstance(deadlines, dict) else None
    if (
        max_rounds is None
        or max_input is None
        or max_output is None
        or max_tools is None
        or attempts is None
        or overall is None
    ):
        return (0, 0, 0, 0, 0), "target_inference_authority_invalid"
    requests = max_rounds * attempts * sessions_per_target
    return (
        requests,
        max_input * requests,
        max_output * requests,
        max_tools * sessions_per_target,
        overall * requests,
    ), None


def _positive_mapping_int(mapping: dict[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _runtime_reservations(
    identity: TargetIdentity,
    sessions_per_target: int,
) -> tuple[int, int, str | None]:
    timeout_seconds = _positive_behavior_int(identity, "timeout_seconds")
    if timeout_seconds is None:
        return 0, 0, "target_timeout_seconds_invalid"
    probe_processes = _positive_behavior_int(identity, "identity_probe_processes")
    if probe_processes is None:
        return 0, 0, "target_identity_probe_processes_invalid"
    probe_timeout = _positive_behavior_int(identity, "identity_probe_timeout_seconds")
    if probe_timeout is None:
        return 0, 0, "target_identity_probe_timeout_invalid"
    processes = sessions_per_target + probe_processes
    wall_clock = (timeout_seconds * sessions_per_target) + probe_timeout
    return processes, wall_clock, None


def _positive_behavior_int(identity: TargetIdentity, key: str) -> int | None:
    value = identity.behavior.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _decision_kind(
    tier: AuthorizationTier,
    policy: PolicyDocument,
) -> DecisionKind | None:
    if tier in policy.standing_tiers:
        return DecisionKind.ALLOWED_STANDING_POLICY
    if tier in policy.per_run_tiers:
        return DecisionKind.APPROVAL_REQUIRED
    return None


def _denied(
    reason: str,
    spec_digest: str,
    policy_digest: str,
    reservations: ResourceLimits = _ZERO_RESERVATIONS,
) -> PolicyDecision:
    return PolicyDecision(
        DecisionKind.DENIED,
        reason,
        spec_digest,
        policy_digest,
        reservations,
    )


def _parse_timestamp(value: str) -> datetime.datetime:
    return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
