"""Tests for deterministic fail-closed automation policy evaluation."""

from __future__ import annotations

import datetime
from dataclasses import replace

import pytest

from ctpf.automation.canonical import sha256_digest
from ctpf.automation.contracts import (
    AuthorizationTier,
    BillingClass,
    DataEgressClass,
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
TARGET_ID_TWO = "e" * 32
SCENARIO_FINGERPRINT = "c" * 64
TARGET_FINGERPRINT = "d" * 64
TARGET_FINGERPRINT_TWO = "f" * 64
NOW = datetime.datetime(2026, 7, 16, 12, 0, tzinfo=datetime.UTC)


def _resources(**overrides: int | None) -> ResourceLimits:
    values: dict[str, int | None] = {
        "wall_clock_seconds": 3_240,
        "provider_requests": 36,
        "input_tokens_reserved": 9_216,
        "output_tokens_reserved": 9_216,
        "tool_calls": 36,
        "runtime_processes": 4,
        "cost_limit_microusd": 0,
    }
    values.update(overrides)
    return ResourceLimits(**values)


def _capability(**overrides: object) -> ScenarioCapability:
    values: dict[str, object] = {
        "scenario": "pattern2",
        "contract_version": 2,
        "modes": (ExperimentMode.SINGLE,),
        "conditions": ("baseline", "manipulated", "hardened"),
        "sessions_per_trial": 3,
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
    target_id: str = TARGET_ID,
    fingerprint: str = TARGET_FINGERPRINT,
    billing: BillingClass = BillingClass.UNMETERED,
    request_cost: int | None = None,
) -> TargetIdentity:
    remote = network != NetworkClass.LOOPBACK
    egress = (
        DataEgressClass.EXTERNAL_RUNTIME
        if network == NetworkClass.EXTERNAL_RUNTIME
        else (DataEgressClass.PACKAGED_SYNTHETIC_REMOTE if remote else DataEgressClass.LOCAL_ONLY)
    )
    behavior: dict[str, object] = {
        "billing": {
            "billing_class": billing.value,
            "request_cost_ceiling_microusd": request_cost,
            "residual_cost_acknowledged": remote,
        },
        "data_egress": {
            "data_egress_class": egress.value,
            "retention_acknowledged": remote,
        },
        "driver": "openai-compatible",
        "limits": {
            "max_input_tokens": 256,
            "max_output_tokens": 256,
            "max_provider_rounds": 12,
            "max_tool_calls_per_session": 12,
        },
        "target_id": target_id,
        "target_type": target_type,
        "transport": {"max_attempts": 1, "deadlines_seconds": {"overall": 90}},
    }
    if target_type == "agent-runtime":
        behavior = {
            "billing": {
                "billing_class": BillingClass.EXTERNAL_RUNTIME.value,
                "request_cost_ceiling_microusd": None,
                "residual_cost_acknowledged": True,
            },
            "data_egress": {
                "data_egress_class": DataEgressClass.EXTERNAL_RUNTIME.value,
                "retention_acknowledged": True,
            },
            "driver": "claude-code-cli",
            "identity_probe_processes": 1,
            "identity_probe_timeout_seconds": 10,
            "target_id": target_id,
            "timeout_seconds": 90,
        }
    return TargetIdentity(target_id, target_type, network, behavior, fingerprint)


def _target_policy(
    *,
    network: NetworkClass = NetworkClass.LOOPBACK,
    billing: BillingClass = BillingClass.UNMETERED,
    request_cost: int | None = None,
    target_id: str = TARGET_ID,
    fingerprint: str = TARGET_FINGERPRINT,
) -> TargetPolicy:
    target_type = "agent-runtime" if network == NetworkClass.EXTERNAL_RUNTIME else "inference"
    identity = _identity(
        network=network,
        target_type=target_type,
        target_id=target_id,
        fingerprint=fingerprint,
        billing=billing,
        request_cost=request_cost,
    )
    return TargetPolicy(
        target_id,
        fingerprint,
        target_type,
        identity.behavior,
        network,
        billing,
        request_cost,
        (
            DataEgressClass.EXTERNAL_RUNTIME
            if network == NetworkClass.EXTERNAL_RUNTIME
            else (
                DataEgressClass.PACKAGED_SYNTHETIC_REMOTE
                if network != NetworkClass.LOOPBACK
                else DataEgressClass.LOCAL_ONLY
            )
        ),
        network != NetworkClass.LOOPBACK,
        network != NetworkClass.LOOPBACK,
    )


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
        targets=(target or _target_policy(),),
        output_roots=(OutputRootPolicy("research-evidence", "C:/research/evidence"),),
        allowed_effects=allowed_effects,
        limits=PolicyLimits(resources or _resources(), 1, 300, 8765),
    )


def _matrix_experiment() -> ExperimentRequest:
    return ExperimentRequest(
        "cascade-memo",
        SCENARIO_FINGERPRINT,
        ExperimentMode.MATRIX,
        3,
        (
            TargetReference(TARGET_ID, TARGET_FINGERPRINT),
            TargetReference(TARGET_ID_TWO, TARGET_FINGERPRINT_TWO),
        ),
    )


def _matrix_capability() -> ScenarioCapability:
    return _capability(
        scenario="cascade-memo",
        modes=(ExperimentMode.SINGLE, ExperimentMode.MATRIX),
        sessions_per_trial=6,
    )


def _matrix_policy(
    *,
    resources: ResourceLimits | None = None,
    first_network: NetworkClass = NetworkClass.LOOPBACK,
    first_billing: BillingClass = BillingClass.UNMETERED,
) -> PolicyDocument:
    return replace(
        _policy(resources=resources),
        scenarios=(
            ScenarioPolicy(
                "cascade-memo",
                (SCENARIO_FINGERPRINT,),
                (ExperimentMode.MATRIX,),
                3,
            ),
        ),
        targets=(
            _target_policy(network=first_network, billing=first_billing),
            _target_policy(target_id=TARGET_ID_TWO, fingerprint=TARGET_FINGERPRINT_TWO),
        ),
    )


def test_loopback_unmetered_scenario_is_allowed_by_standing_policy() -> None:
    """Exact local target and scenario identities receive Tier 1 authority."""
    decision = evaluate_policy(_spec(), _policy(), _capability(), (_identity(),), now=NOW)

    assert decision.kind == DecisionKind.ALLOWED_STANDING_POLICY
    assert decision.reason_code == "policy_match"
    assert decision.minimum_reservations.provider_requests == 36
    assert decision.minimum_reservations.output_tokens_reserved == 9_216
    assert decision.minimum_reservations.input_tokens_reserved == 9_216
    assert decision.minimum_reservations.tool_calls == 36


def test_remote_metered_target_requires_tier_two_and_reserves_cost() -> None:
    """Public billed execution is per-run and includes declared worst-case cost."""
    target_policy = _target_policy(
        network=NetworkClass.HTTPS_PUBLIC,
        billing=BillingClass.METERED,
        request_cost=25_000,
    )
    policy = _policy(
        target=target_policy,
        resources=_resources(cost_limit_microusd=900_000),
    )
    limits = _resources(cost_limit_microusd=900_000)
    spec = _spec(requested_tier=AuthorizationTier.BOUNDED_REMOTE, limits=limits)
    identity = _identity(
        network=NetworkClass.HTTPS_PUBLIC,
        billing=BillingClass.METERED,
        request_cost=25_000,
    )

    decision = evaluate_policy(spec, policy, _capability(), (identity,), now=NOW)

    assert decision.kind == DecisionKind.APPROVAL_REQUIRED
    assert decision.minimum_reservations.cost_limit_microusd == 900_000


def test_exact_digest_bound_remote_spec_uses_standing_campaign_authority() -> None:
    """A signed campaign may authorize one exact bounded-remote RunSpec without per-run approval."""
    target_policy = _target_policy(network=NetworkClass.HTTPS_PRIVATE)
    identity = _identity(network=NetworkClass.HTTPS_PRIVATE)
    spec = _spec(requested_tier=AuthorizationTier.BOUNDED_REMOTE)
    policy = replace(
        _policy(target=target_policy),
        standing_tiers=(AuthorizationTier.BOUNDED_REMOTE,),
        per_run_tiers=(),
        standing_run_spec_digests=(sha256_digest(spec.to_payload()),),
    )

    allowed = evaluate_policy(spec, policy, _capability(), (identity,), now=NOW)
    changed = replace(spec, idempotency_key="agent-request-0002")
    denied = evaluate_policy(changed, policy, _capability(), (identity,), now=NOW)

    assert allowed.kind == DecisionKind.ALLOWED_STANDING_POLICY
    assert denied.kind == DecisionKind.DENIED
    assert denied.reason_code == "standing_run_spec_not_authorized"


@pytest.mark.parametrize(
    "network",
    [NetworkClass.HTTPS_PUBLIC, NetworkClass.HTTPS_PRIVATE],
)
def test_remote_target_is_denied_at_local_tier(network: NetworkClass) -> None:
    """A policy listing does not let an agent understate required authority."""
    target_policy = _target_policy(network=network)

    decision = evaluate_policy(
        _spec(),
        _policy(target=target_policy),
        _capability(),
        (_identity(network=network),),
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
    target_policy = _target_policy(
        network=NetworkClass.EXTERNAL_RUNTIME,
        billing=BillingClass.EXTERNAL_RUNTIME,
    )
    runtime_resources = _resources(cost_limit_microusd=None)
    spec = _spec(
        requested_tier=AuthorizationTier.BOUNDED_REMOTE,
        limits=runtime_resources,
    )
    identity = _identity(
        network=NetworkClass.EXTERNAL_RUNTIME,
        target_type="agent-runtime",
    )

    decision = evaluate_policy(
        spec,
        _policy(target=target_policy, resources=runtime_resources),
        _capability(),
        (identity,),
        now=NOW,
    )

    assert decision.kind == DecisionKind.APPROVAL_REQUIRED
    assert decision.minimum_reservations.runtime_processes == 4
    assert decision.minimum_reservations.wall_clock_seconds == 280
    assert decision.minimum_reservations.cost_limit_microusd is None
    assert decision.warnings == (
        "external runtime residual cost is acknowledged but not measurable",
    )

    under_reserved = evaluate_policy(
        _spec(
            requested_tier=AuthorizationTier.BOUNDED_REMOTE,
            limits=_resources(wall_clock_seconds=279, cost_limit_microusd=None),
        ),
        _policy(target=target_policy, resources=runtime_resources),
        _capability(),
        (identity,),
        now=NOW,
    )
    assert under_reserved.kind == DecisionKind.DENIED
    assert under_reserved.reason_code == "requested_limits_below_minimum"


def test_external_runtime_identity_probe_reservations_fail_closed() -> None:
    """Missing or invalid identity-probe accounting cannot authorize a runtime."""
    target_policy = _target_policy(
        network=NetworkClass.EXTERNAL_RUNTIME,
        billing=BillingClass.EXTERNAL_RUNTIME,
    )
    identity = _identity(
        network=NetworkClass.EXTERNAL_RUNTIME,
        target_type="agent-runtime",
    )
    expected_reasons = {
        "identity_probe_processes": "target_identity_probe_processes_invalid",
        "identity_probe_timeout_seconds": "target_identity_probe_timeout_invalid",
    }

    for key, expected in expected_reasons.items():
        behavior = dict(identity.behavior)
        behavior[key] = 0
        invalid = replace(identity, behavior=behavior)
        decision = evaluate_policy(
            _spec(requested_tier=AuthorizationTier.BOUNDED_REMOTE),
            _policy(target=replace(target_policy, behavior=behavior)),
            _capability(),
            (invalid,),
            now=NOW,
        )
        assert decision.kind == DecisionKind.DENIED
        assert decision.reason_code == expected


def test_cascade_matrix_reserves_every_condition_session_for_every_target() -> None:
    """Three cascade trials over two targets reserve all 36 sessions."""
    limits = _resources(
        wall_clock_seconds=38_880,
        provider_requests=432,
        input_tokens_reserved=110_592,
        output_tokens_reserved=110_592,
        tool_calls=432,
    )
    spec = _spec(experiment=_matrix_experiment(), limits=limits)
    identities = (
        _identity(),
        _identity(target_id=TARGET_ID_TWO, fingerprint=TARGET_FINGERPRINT_TWO),
    )

    decision = evaluate_policy(
        spec,
        _matrix_policy(resources=limits),
        _matrix_capability(),
        identities,
        now=NOW,
    )

    assert decision.kind == DecisionKind.ALLOWED_STANDING_POLICY
    assert decision.minimum_reservations.provider_requests == 432
    assert decision.minimum_reservations.output_tokens_reserved == 110_592
    assert decision.minimum_reservations.tool_calls == 432


def test_matrix_rejects_external_runtime_before_reservation() -> None:
    """The demonstrated matrix workflow is inference-only."""
    spec = _spec(
        requested_tier=AuthorizationTier.BOUNDED_REMOTE,
        experiment=_matrix_experiment(),
    )
    identities = (
        _identity(
            network=NetworkClass.EXTERNAL_RUNTIME,
            target_type="agent-runtime",
        ),
        _identity(target_id=TARGET_ID_TWO, fingerprint=TARGET_FINGERPRINT_TWO),
    )

    policy = _matrix_policy(
        first_network=NetworkClass.EXTERNAL_RUNTIME,
        first_billing=BillingClass.EXTERNAL_RUNTIME,
    )
    decision = evaluate_policy(spec, policy, _matrix_capability(), identities, now=NOW)

    assert decision.kind == DecisionKind.DENIED
    assert decision.reason_code == "target_type_not_supported"
