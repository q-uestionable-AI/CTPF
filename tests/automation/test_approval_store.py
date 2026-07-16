"""Tests for authenticated grants, replay resistance, and lifecycle storage."""

from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path

import pytest

from ctpf.automation import approval
from ctpf.automation.approval import ApprovalError
from ctpf.automation.canonical import CanonicalizationError, sha256_digest
from ctpf.automation.contracts import (
    AuthorizationGrant,
    AuthorizationTier,
    AutomationRunState,
    BillingClass,
    DecisionKind,
    ExperimentMode,
    ExperimentRequest,
    GrantSource,
    NetworkClass,
    OutputRootPolicy,
    PolicyDecision,
    PolicyDocument,
    PolicyLimits,
    Requester,
    ResourceLimits,
    RunSpec,
    ScenarioPolicy,
    TargetPolicy,
    TargetReference,
)
from ctpf.automation.store import (
    AutomationStoreError,
    IdempotencyConflictError,
    append_event,
    bind_grant_and_create_ready_run,
    get_automation_run,
    get_grant,
    get_policy,
    list_events,
    revoke_grant,
    revoke_policy,
    save_grant,
    save_policy,
    transition_run_state,
)
from ctpf.core.db import get_connection

POLICY_ID = "a" * 32
TARGET_ID = "b" * 32
SCENARIO_FINGERPRINT = "c" * 64
TARGET_FINGERPRINT = "d" * 64
NOW = datetime.datetime(2026, 7, 16, 12, 0, tzinfo=datetime.UTC)


def _limits() -> ResourceLimits:
    return ResourceLimits(120, 12, 3_072, 12, 1, 0)


def _spec(*, purpose: str = "Exercise the packaged synthetic scenario.") -> RunSpec:
    return RunSpec(
        idempotency_key="agent-request-0001",
        requester=Requester("agent", "test-agent", "1"),
        purpose=purpose,
        policy_id=POLICY_ID,
        requested_tier=AuthorizationTier.LOCAL_SYNTHETIC,
        experiment=ExperimentRequest(
            "pattern2",
            SCENARIO_FINGERPRINT,
            ExperimentMode.SINGLE,
            1,
            (TargetReference(TARGET_ID, TARGET_FINGERPRINT),),
        ),
        output_root_id="research-evidence",
        limits=_limits(),
    )


def _policy() -> PolicyDocument:
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
            TargetPolicy(
                TARGET_ID,
                TARGET_FINGERPRINT,
                NetworkClass.LOOPBACK,
                BillingClass.UNMETERED,
                None,
            ),
        ),
        output_roots=(OutputRootPolicy("research-evidence", "C:/research/evidence"),),
        allowed_effects=("pattern2-action-sink",),
        limits=PolicyLimits(_limits(), 1, 300, 8765),
    )


def _decision(spec: RunSpec, policy: PolicyDocument) -> PolicyDecision:
    return PolicyDecision(
        DecisionKind.ALLOWED_STANDING_POLICY,
        "policy_match",
        sha256_digest(spec.to_payload()),
        sha256_digest(policy.to_payload()),
        ResourceLimits(1, 12, 3_072, 1, 1, 0),
    )


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Provide a process-local fake for the reserved keyring account."""
    secrets: dict[str, str] = {}
    monkeypatch.setattr(approval, "get_local_secret", secrets.get)
    monkeypatch.setattr(approval, "set_local_secret", secrets.__setitem__)
    monkeypatch.setattr(approval, "delete_local_secret", lambda name: secrets.pop(name, None))
    return secrets


def _signed_material() -> tuple[RunSpec, PolicyDocument, str, str, AuthorizationGrant, str]:
    spec = _spec()
    policy = _policy()
    policy_signature, key_id = approval.sign_policy(policy)
    grant, grant_signature = approval.issue_authorization_grant(
        spec,
        policy,
        _decision(spec, policy),
        GrantSource.STANDING_POLICY,
        policy_signature=policy_signature,
        policy_key_id=key_id,
        lifetime_seconds=120,
        issued_at=NOW,
    )
    return spec, policy, policy_signature, key_id, grant, grant_signature


def test_policy_and_grant_authentication_detects_tampering(
    fake_keyring: dict[str, str],
) -> None:
    """Canonical HMACs bind bodies, key identity, time, and the exact spec."""
    approval.initialize_approval_key()
    spec, policy, policy_signature, key_id, grant, grant_signature = _signed_material()

    assert approval.authenticate_policy(policy, policy_signature, key_id, now=NOW) == sha256_digest(
        policy.to_payload()
    )
    approval.authenticate_authorization_grant(
        grant, grant_signature, spec, policy, now=NOW + datetime.timedelta(seconds=30)
    )

    with pytest.raises(ApprovalError, match="signature"):
        approval.authenticate_policy(
            replace(policy, name="tampered"), policy_signature, key_id, now=NOW
        )
    with pytest.raises(ApprovalError, match="exact RunSpec"):
        approval.authenticate_authorization_grant(
            grant,
            grant_signature,
            _spec(purpose="Different authorized-looking purpose."),
            policy,
            now=NOW,
        )


def test_grant_issuance_rejects_denial_wrong_source_and_stale_decision(
    fake_keyring: dict[str, str],
) -> None:
    """Only the matching decision path can obtain a short-lived grant."""
    approval.initialize_approval_key()
    spec = _spec()
    policy = _policy()
    allowed = _decision(spec, policy)
    denied = replace(allowed, kind=DecisionKind.DENIED, reason_code="denied")
    policy_signature, key_id = approval.sign_policy(policy)

    with pytest.raises(ApprovalError, match="denied decision"):
        approval.issue_authorization_grant(
            spec,
            policy,
            denied,
            GrantSource.STANDING_POLICY,
            policy_signature=policy_signature,
            policy_key_id=key_id,
            lifetime_seconds=60,
            issued_at=NOW,
        )
    with pytest.raises(ApprovalError, match="source"):
        approval.issue_authorization_grant(
            spec,
            policy,
            allowed,
            GrantSource.HUMAN_PER_RUN,
            policy_signature=policy_signature,
            policy_key_id=key_id,
            lifetime_seconds=60,
            issued_at=NOW,
        )
    with pytest.raises(ApprovalError, match="does not match"):
        approval.issue_authorization_grant(
            replace(spec, purpose="changed"),
            policy,
            allowed,
            GrantSource.STANDING_POLICY,
            policy_signature=policy_signature,
            policy_key_id=key_id,
            lifetime_seconds=60,
            issued_at=NOW,
        )
    with pytest.raises(ApprovalError, match="policy signature"):
        approval.issue_authorization_grant(
            spec,
            policy,
            allowed,
            GrantSource.STANDING_POLICY,
            policy_signature="0" * 64,
            policy_key_id=key_id,
            lifetime_seconds=60,
            issued_at=NOW,
        )


def test_key_initialization_is_idempotent_and_deletion_fails_closed(
    fake_keyring: dict[str, str],
) -> None:
    """Key creation never rotates silently, and missing material cannot sign."""
    first = approval.initialize_approval_key()
    second = approval.initialize_approval_key()
    assert first == second

    approval.delete_approval_key()
    with pytest.raises(ApprovalError, match="not initialized"):
        approval.sign_policy(_policy())


def test_store_binds_grant_once_and_returns_exact_idempotent_run(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """One grant and idempotency key resolve to one durable READY control."""
    approval.initialize_approval_key()
    spec, policy, policy_signature, key_id, grant, grant_signature = _signed_material()
    db_path = tmp_path / "ctpf.db"

    with get_connection(db_path) as conn:
        save_policy(conn, policy, signature=policy_signature, key_id=key_id)
        save_grant(conn, grant, signature=grant_signature)
        first = bind_grant_and_create_ready_run(conn, spec, grant, now=NOW)
        second = bind_grant_and_create_ready_run(conn, spec, grant, now=NOW)

        assert first.created is True
        assert second.created is False
        assert first.run.run_id == second.run.run_id
        assert first.run.state == AutomationRunState.READY
        assert get_policy(conn, POLICY_ID) is not None
        stored_grant = get_grant(conn, grant.grant_id)
        assert stored_grant is not None and stored_grant.bound_run_id == first.run.run_id
        assert [event["event_type"] for event in list_events(conn, first.run.run_id)] == [
            "run_ready"
        ]


def test_store_rejects_idempotency_conflict_and_revoked_grant(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """Changed proposals and revoked grants fail before creating controls."""
    approval.initialize_approval_key()
    spec, policy, policy_signature, key_id, grant, grant_signature = _signed_material()
    changed = _spec(purpose="A materially different proposal.")
    changed_grant, changed_signature = approval.issue_authorization_grant(
        changed,
        policy,
        _decision(changed, policy),
        GrantSource.STANDING_POLICY,
        policy_signature=policy_signature,
        policy_key_id=key_id,
        lifetime_seconds=120,
        issued_at=NOW,
    )
    db_path = tmp_path / "ctpf.db"

    with get_connection(db_path) as conn:
        save_policy(conn, policy, signature=policy_signature, key_id=key_id)
        save_grant(conn, grant, signature=grant_signature)
        save_grant(conn, changed_grant, signature=changed_signature)
        bind_grant_and_create_ready_run(conn, spec, grant, now=NOW)
        with pytest.raises(IdempotencyConflictError):
            bind_grant_and_create_ready_run(conn, changed, changed_grant, now=NOW)

    other_spec = replace(spec, idempotency_key="agent-request-0002")
    other_grant, other_signature = approval.issue_authorization_grant(
        other_spec,
        policy,
        _decision(other_spec, policy),
        GrantSource.STANDING_POLICY,
        policy_signature=policy_signature,
        policy_key_id=key_id,
        lifetime_seconds=120,
        issued_at=NOW,
    )
    with get_connection(db_path) as conn:
        save_grant(conn, other_grant, signature=other_signature)
        assert revoke_grant(conn, other_grant.grant_id)
        with pytest.raises(AutomationStoreError, match="revoked"):
            bind_grant_and_create_ready_run(conn, other_spec, other_grant, now=NOW)


def test_store_requires_active_policy_and_grant_interval(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """Binding independently checks revocation and temporal authority."""
    approval.initialize_approval_key()
    spec, policy, policy_signature, key_id, grant, grant_signature = _signed_material()

    with get_connection(tmp_path / "expired.db") as conn:
        save_policy(conn, policy, signature=policy_signature, key_id=key_id)
        save_grant(conn, grant, signature=grant_signature)
        with pytest.raises(AutomationStoreError, match="validity interval"):
            bind_grant_and_create_ready_run(
                conn,
                spec,
                grant,
                now=NOW + datetime.timedelta(seconds=121),
            )

    with get_connection(tmp_path / "revoked.db") as conn:
        save_policy(conn, policy, signature=policy_signature, key_id=key_id)
        save_grant(conn, grant, signature=grant_signature)
        assert revoke_policy(conn, policy.policy_id, revoked_at="2026-07-16T12:00:00Z")
        with pytest.raises(AutomationStoreError, match="policy is revoked"):
            bind_grant_and_create_ready_run(conn, spec, grant, now=NOW)


def test_lifecycle_transitions_and_events_are_compare_and_set(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """Valid state changes append ordered evidence and stale transitions fail."""
    approval.initialize_approval_key()
    spec, policy, policy_signature, key_id, grant, grant_signature = _signed_material()
    with get_connection(tmp_path / "ctpf.db") as conn:
        save_policy(conn, policy, signature=policy_signature, key_id=key_id)
        save_grant(conn, grant, signature=grant_signature)
        run = bind_grant_and_create_ready_run(conn, spec, grant, now=NOW).run
        running = transition_run_state(
            conn, run.run_id, AutomationRunState.READY, AutomationRunState.RUNNING
        )
        append_event(conn, run.run_id, "budget_observed", {"provider_requests": 1})
        completed = transition_run_state(
            conn, run.run_id, AutomationRunState.RUNNING, AutomationRunState.COMPLETED
        )

        assert running.state == AutomationRunState.RUNNING
        assert running.started_at is not None
        assert completed.state == AutomationRunState.COMPLETED
        assert completed.finished_at is not None
        assert [event["sequence"] for event in list_events(conn, run.run_id)] == [1, 2, 3, 4]
        with pytest.raises(AutomationStoreError, match="state changed"):
            transition_run_state(
                conn, run.run_id, AutomationRunState.RUNNING, AutomationRunState.FAILED
            )


def test_failed_event_serialization_rolls_back_state_change(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """A lifecycle state can never commit without its matching event."""
    approval.initialize_approval_key()
    spec, policy, policy_signature, key_id, grant, grant_signature = _signed_material()
    with get_connection(tmp_path / "ctpf.db") as conn:
        save_policy(conn, policy, signature=policy_signature, key_id=key_id)
        save_grant(conn, grant, signature=grant_signature)
        run = bind_grant_and_create_ready_run(conn, spec, grant, now=NOW).run

        with pytest.raises(CanonicalizationError):
            transition_run_state(
                conn,
                run.run_id,
                AutomationRunState.READY,
                AutomationRunState.RUNNING,
                event_payload={"invalid_empty_string": ""},
            )

        stored = get_automation_run(conn, run.run_id)
        assert stored is not None and stored.state == AutomationRunState.READY
        assert [event["event_type"] for event in list_events(conn, run.run_id)] == ["run_ready"]
