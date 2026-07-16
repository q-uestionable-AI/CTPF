"""Deterministic policy evaluation for governed experiment proposals."""

from __future__ import annotations

import datetime
from collections.abc import Iterable

from ctpf.automation.canonical import sha256_digest
from ctpf.automation.contracts import (
    AuthorizationTier,
    BillingClass,
    DecisionKind,
    NetworkClass,
    PolicyDecision,
    PolicyDocument,
    ResourceLimits,
    RunSpec,
    TargetPolicy,
)
from ctpf.automation.targets import ScenarioCapability, TargetIdentity

_MAX_INTEGER = (2**63) - 1
_ZERO_RESERVATIONS = ResourceLimits(1, 1, 1, 1, 1, 0)


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
        target_policy = policies.get(reference.target_id)
        reason = _target_policy_reason(reference.target_fingerprint, identity, target_policy)
        if reason:
            return reason
        if _minimum_tier(identity, target_policy) > spec.requested_tier:
            return "target_requires_higher_tier"
    return None


def _target_policy_reason(
    fingerprint: str,
    identity: TargetIdentity,
    target_policy: TargetPolicy | None,
) -> str | None:
    if target_policy is None:
        return "target_not_authorized"
    if target_policy.target_fingerprint != fingerprint:
        return "target_fingerprint_not_authorized"
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
    sessions = sessions_per_target * len(identities)
    provider_requests = 0
    tokens = 0
    runtime_processes = 0
    cost = 0
    warnings: list[str] = []
    target_policies = {item.target_id: item for item in policy.targets}
    for identity in identities:
        if identity.target_type == "inference":
            max_tokens = identity.behavior.get("max_tokens")
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
                return _ZERO_RESERVATIONS, (), "target_max_tokens_invalid"
            max_rounds = identity.behavior.get("max_provider_rounds")
            if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
                return _ZERO_RESERVATIONS, (), "target_max_provider_rounds_invalid"
            target_requests = max_rounds * sessions_per_target
            provider_requests += target_requests
            tokens += max_tokens * target_requests
        else:
            target_requests = sessions_per_target
            provider_requests += target_requests
            runtime_processes += sessions_per_target
            warnings.append("external runtime cost is not measured by CTPF")
        target_policy = target_policies[identity.target_id]
        ceiling = target_policy.request_cost_ceiling_microusd
        if ceiling is not None:
            cost += ceiling * target_requests
    values = (provider_requests, tokens, runtime_processes, cost)
    if any(value > _MAX_INTEGER for value in values):
        return _ZERO_RESERVATIONS, (), "minimum_reservation_overflow"
    reservations = ResourceLimits(
        wall_clock_seconds=1,
        provider_requests=max(1, provider_requests),
        output_tokens_reserved=max(1, tokens),
        tool_calls=max(1, sessions),
        runtime_processes=max(1, runtime_processes),
        cost_limit_microusd=cost,
    )
    return reservations, tuple(sorted(set(warnings))), None


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
