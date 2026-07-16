"""Tests for deterministic fail-closed automation policy evaluation."""

from __future__ import annotations

import datetime
from dataclasses import replace

from ctpf.automation.contracts import (
    AuthorizationTier,
    BillingClass,
    DecisionKind,
    ExperimentMode,
    ExperimentRequest,
    NetworkClass,
    OutputRootPolicy,
    PolicyDocument,
    PolicyLimits,
    Requester,
    ResourceLimits,
    RunSpec,
    ScenarioPolicy,
    TargetPolicy,
    TargetReference,
)
from ctpf.automation.policy import evaluate_policy
from ctpf.automation.targets import ScenarioCapability, TargetIdentity

POLICY_ID = "a" * 32
TARGET_ID = "b" * 32
SCENARIO_FINGERPRINT = "c" * 64
TARGET_FINGERPRINT = "d" * 64
NOW = datetime.datetime(2026, 7, 16, 12, 0, tzinfo=datetime.UTC)


def _resources(**overrides: int) -> ResourceLimits:
    values = {
        "wall_clock_seconds": 120,
        "provider_requests": 12,
        "output_tokens_reserved": 3_072,
        "tool_calls": 12,
        "runtime_processes": 1,
        "cost_limit_microusd": 0,
    }
    values.update(overrides)
    return ResourceLimits(**values)


def _capability(**overrides: object) -> ScenarioCapability:
    values: dict[str, object] = {
        "scenario": "pattern2",
        "contract_version": 1,
        "modes": (ExperimentMode.SINGLE,),
        "conditions": ("baseline", "manipulated", "hardened"),
        "sessions_per_trial": 1,
        "prompts": ("Inspect the status.",),
        "tool_names": ("read_status", "apply_change", "read_sink"),
        "effect_ids": ("pattern2-action-sink",),
        "supported_target_types": ("agent-runtime", "inference"),
        "retry_policy": "none",
        "package_version": "0.13.1",
        "source_hashes": {"experiment.py": "e" * 64},
        "fingerprint": SCENARIO_FINGERPRINT,
    }
    values.update(overrides)
    return ScenarioCapability(**values)  # type: ignore[arg-type]


def _identity(
    *,
    network: NetworkClass = NetworkClass.LOOPBACK,
    target_type: str = "inference",
) -> TargetIdentity:
    behavior: dict[str, object] = {
        "driver": "openai-compatible",
        "max_tokens": 256,
        "max_provider_rounds": 12,
        "target_id": TARGET_ID,
        "target_type": target_type,
    }
    if target_type == "agent-runtime":
        behavior = {"driver": "claude-code-cli", "target_id": TARGET_ID}
    return TargetIdentity(TARGET_ID, target_type, network, behavior, TARGET_FINGERPRINT)


def _spec(**overrides: object) -> RunSpec:
    values: dict[str, object] = {
        "idempotency_key": "agent-request-0001",
        "requester": Requester("agent", "test-agent", "1"),
        "purpose": "Exercise the installed synthetic scenario.",
        "policy_id": POLICY_ID,
        "requested_tier": AuthorizationTier.LOCAL_SYNTHETIC,
        "experiment": ExperimentRequest(
            "pattern2",
            SCENARIO_FINGERPRINT,
            ExperimentMode.SINGLE,
            1,
            (TargetReference(TARGET_ID, TARGET_FINGERPRINT),),
        ),
        "output_root_id": "research-evidence",
        "limits": _resources(),
    }
    values.update(overrides)
    return RunSpec(**values)  # type: ignore[arg-type]


def _policy(
    *,
    target: TargetPolicy | None = None,
    resources: ResourceLimits | None = None,
    allowed_effects: tuple[str, ...] = ("pattern2-action-sink",),
) -> PolicyDocument:
    return PolicyDocument(
        policy_id=POLICY_ID,
        name="agent test policy",
        created_at="2026-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
        standing_tiers=(AuthorizationTier.LOCAL_SYNTHETIC,),
        per_run_tiers=(AuthorizationTier.BOUNDED_REMOTE,),
        scenarios=(
            ScenarioPolicy(
                "pattern2",
                (SCENARIO_FINGERPRINT,),
                (ExperimentMode.SINGLE,),
                1,
            ),
        ),
        targets=(
            target
            or TargetPolicy(
                TARGET_ID,
                TARGET_FINGERPRINT,
                NetworkClass.LOOPBACK,
                BillingClass.UNMETERED,
                None,
            ),
        ),
        output_roots=(OutputRootPolicy("research-evidence", "C:/research/evidence"),),
        allowed_effects=allowed_effects,
        limits=PolicyLimits(resources or _resources(), 1, 300, 8765),
    )


def test_loopback_unmetered_scenario_is_allowed_by_standing_policy() -> None:
    """Exact local target and scenario identities receive Tier 1 authority."""
    decision = evaluate_policy(_spec(), _policy(), _capability(), (_identity(),), now=NOW)

    assert decision.kind == DecisionKind.ALLOWED_STANDING_POLICY
    assert decision.reason_code == "policy_match"
    assert decision.minimum_reservations.provider_requests == 12
    assert decision.minimum_reservations.output_tokens_reserved == 3_072


def test_remote_metered_target_requires_tier_two_and_reserves_cost() -> None:
    """Public billed execution is per-run and includes declared worst-case cost."""
    target_policy = TargetPolicy(
        TARGET_ID,
        TARGET_FINGERPRINT,
        NetworkClass.HTTPS_PUBLIC,
        BillingClass.METERED,
        25_000,
    )
    policy = _policy(
        target=target_policy,
        resources=_resources(cost_limit_microusd=300_000),
    )
    limits = _resources(cost_limit_microusd=300_000)
    spec = _spec(requested_tier=AuthorizationTier.BOUNDED_REMOTE, limits=limits)
    identity = _identity(network=NetworkClass.HTTPS_PUBLIC)

    decision = evaluate_policy(spec, policy, _capability(), (identity,), now=NOW)

    assert decision.kind == DecisionKind.APPROVAL_REQUIRED
    assert decision.minimum_reservations.cost_limit_microusd == 300_000


def test_remote_target_is_denied_at_local_tier() -> None:
    """A policy listing does not let an agent understate required authority."""
    target_policy = TargetPolicy(
        TARGET_ID,
        TARGET_FINGERPRINT,
        NetworkClass.HTTPS_PUBLIC,
        BillingClass.UNMETERED,
        None,
    )

    decision = evaluate_policy(
        _spec(),
        _policy(target=target_policy),
        _capability(),
        (_identity(network=NetworkClass.HTTPS_PUBLIC),),
        now=NOW,
    )

    assert decision.kind == DecisionKind.DENIED
    assert decision.reason_code == "target_requires_higher_tier"


def test_fingerprint_effect_and_resource_drift_fail_closed() -> None:
    """Any changed executable identity, effect, or undersized budget is denied."""
    identity_drift = replace(_identity(), fingerprint="f" * 64)
    fingerprint = evaluate_policy(_spec(), _policy(), _capability(), (identity_drift,), now=NOW)
    effect = evaluate_policy(
        _spec(),
        _policy(allowed_effects=("different-effect",)),
        _capability(),
        (_identity(),),
        now=NOW,
    )
    low_limits = _resources(output_tokens_reserved=1)
    resource = evaluate_policy(
        _spec(limits=low_limits), _policy(), _capability(), (_identity(),), now=NOW
    )

    assert fingerprint.reason_code == "target_identity_mismatch"
    assert effect.reason_code == "scenario_effect_not_authorized"
    assert resource.reason_code == "requested_limits_below_minimum"


def test_expired_policy_and_unexecutable_inspection_tier_are_denied() -> None:
    """Policy time and Tier 0 are hard authorization boundaries."""
    expired = replace(_policy(), expires_at="2026-07-01T00:00:00Z")
    expiration = evaluate_policy(_spec(), expired, _capability(), (_identity(),), now=NOW)
    inspection = evaluate_policy(
        _spec(requested_tier=AuthorizationTier.INSPECTION),
        _policy(),
        _capability(),
        (_identity(),),
        now=NOW,
    )

    assert expiration.reason_code == "policy_expired"
    assert inspection.reason_code == "inspection_tier_cannot_execute"


def test_external_runtime_is_per_run_and_emits_cost_warning() -> None:
    """The demonstrated external runtime remains Tier 2 with explicit uncertainty."""
    target_policy = TargetPolicy(
        TARGET_ID,
        TARGET_FINGERPRINT,
        NetworkClass.EXTERNAL_RUNTIME,
        BillingClass.EXTERNAL_RUNTIME,
        None,
    )
    spec = _spec(requested_tier=AuthorizationTier.BOUNDED_REMOTE)
    identity = _identity(
        network=NetworkClass.EXTERNAL_RUNTIME,
        target_type="agent-runtime",
    )

    decision = evaluate_policy(
        spec, _policy(target=target_policy), _capability(), (identity,), now=NOW
    )

    assert decision.kind == DecisionKind.APPROVAL_REQUIRED
    assert decision.minimum_reservations.runtime_processes == 1
    assert decision.warnings == ("external runtime cost is not measured by CTPF",)
