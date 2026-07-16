"""Integration tests for the non-executing automation domain service."""

from __future__ import annotations

import datetime
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from ctpf import driven_inference
from ctpf.automation import approval
from ctpf.automation import service as automation_service
from ctpf.automation.contracts import (
    AuthorizationTier,
    AutomationRunState,
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
from ctpf.automation.envelope import ControlError
from ctpf.automation.service import AutomationService, ValidationResult
from ctpf.automation.targets import scenario_capability, target_identity_from_profile
from ctpf.core.db import get_readonly_connection

POLICY_ID = "a" * 32
TARGET_ID = "b" * 32
CONCURRENT_START_TIMEOUT_SECONDS = 10
NOW = datetime.datetime(2026, 7, 16, 12, 0, tzinfo=datetime.UTC)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Provide process-local signing-key storage."""
    secrets: dict[str, str] = {}
    monkeypatch.setattr(approval, "get_local_secret", secrets.get)
    monkeypatch.setattr(approval, "set_local_secret", secrets.__setitem__)
    monkeypatch.setattr(approval, "delete_local_secret", lambda name: secrets.pop(name, None))
    return secrets


def _limits(*, cost: int = 0) -> ResourceLimits:
    return ResourceLimits(3_240, 36, 9_216, 9_216, 36, 4, cost)


def _target(network: NetworkClass) -> TargetPolicy:
    endpoint = (
        "http://127.0.0.1:11434/v1"
        if network == NetworkClass.LOOPBACK
        else "https://models.example.test/v1"
    )
    remote = network == NetworkClass.HTTPS_PUBLIC
    identity = target_identity_from_profile(
        driven_inference.OpenAICompatibleTargetProfile(
            target_id=TARGET_ID,
            name="test target",
            endpoint=endpoint,
            model="test-model",
            credential_name="test-key",
            max_tokens=256,
            temperature=0.0,
            max_input_tokens=256,
            data_egress_class=(
                DataEgressClass.PACKAGED_SYNTHETIC_REMOTE if remote else DataEgressClass.LOCAL_ONLY
            ),
            retention_acknowledged=remote,
            residual_cost_acknowledged=remote,
        )
    )
    return TargetPolicy(
        TARGET_ID,
        identity.fingerprint,
        "inference",
        identity.behavior,
        network,
        BillingClass.UNMETERED,
        None,
        (DataEgressClass.PACKAGED_SYNTHETIC_REMOTE if remote else DataEgressClass.LOCAL_ONLY),
        remote,
        remote,
    )


def _material(tmp_path: Path, tier: AuthorizationTier) -> tuple[PolicyDocument, RunSpec, Path]:
    capability = scenario_capability("pattern2")
    network = (
        NetworkClass.LOOPBACK
        if tier == AuthorizationTier.LOCAL_SYNTHETIC
        else NetworkClass.HTTPS_PUBLIC
    )
    target = _target(network)
    output_root = tmp_path / "research-evidence"
    standing = (tier,) if tier == AuthorizationTier.LOCAL_SYNTHETIC else ()
    per_run = (tier,) if tier == AuthorizationTier.BOUNDED_REMOTE else ()
    policy = PolicyDocument(
        policy_id=POLICY_ID,
        name="service integration policy",
        created_at="2026-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
        standing_tiers=standing,
        per_run_tiers=per_run,
        scenarios=(
            ScenarioPolicy("pattern2", (capability.fingerprint,), (ExperimentMode.SINGLE,), 1),
        ),
        targets=(target,),
        output_roots=(OutputRootPolicy("research-evidence", str(output_root.resolve())),),
        allowed_effects=capability.effect_ids,
        limits=PolicyLimits(_limits(), 1, 300, 8765),
    )
    spec = RunSpec(
        idempotency_key="agent-request-0001",
        requester=Requester("agent", "test-agent", "1"),
        purpose="Exercise the packaged synthetic scenario.",
        policy_id=POLICY_ID,
        requested_tier=tier,
        experiment=ExperimentRequest(
            "pattern2",
            capability.fingerprint,
            ExperimentMode.SINGLE,
            1,
            (TargetReference(TARGET_ID, target.target_fingerprint),),
        ),
        output_root_id="research-evidence",
        limits=_limits(),
    )
    return policy, spec, output_root


def _assert_control_error(code: str, callback: Callable[[], object]) -> None:
    with pytest.raises(ControlError) as caught:
        callback()
    assert caught.value.code == code


def test_discovery_and_validation_do_not_create_a_database(tmp_path: Path) -> None:
    """Tier-0 discovery and failed validation are stateless and query-only."""
    db_path = tmp_path / "missing" / "ctpf.db"
    _, spec, output_root = _material(tmp_path, AuthorizationTier.LOCAL_SYNTHETIC)
    service = AutomationService(db_path=db_path)

    capabilities = service.capabilities()
    assert capabilities["execute_available"] is False
    assert capabilities["verify_available"] is False
    _assert_control_error("policy_not_found", lambda: service.validate(spec, now=NOW))
    assert not db_path.exists()
    assert not output_root.exists()


def test_standing_start_is_idempotent_and_cancel_has_no_research_effect(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """Tier-1 start creates one READY control and cancellation creates no output."""
    db_path = tmp_path / "ctpf.db"
    policy, spec, output_root = _material(tmp_path, AuthorizationTier.LOCAL_SYNTHETIC)
    service = AutomationService(db_path=db_path)
    approval.initialize_approval_key()
    service.create_policy(policy, now=NOW)

    validation = service.validate(spec, now=NOW)
    assert validation.decision.kind == DecisionKind.ALLOWED_STANDING_POLICY
    first = service.start(spec, now=NOW)
    second = service.start(spec, now=NOW)
    assert first["created"] is True
    assert second["created"] is False
    assert first["run_id"] == second["run_id"]
    assert first["state"] == AutomationRunState.READY.value
    assert not output_root.exists()

    status = service.status(first["run_id"])
    assert status["execute_available"] is False
    _assert_control_error("result_unavailable", lambda: service.result(first["run_id"]))
    cancelled = service.cancel(first["run_id"])
    assert cancelled["state"] == AutomationRunState.CANCELLED.value
    assert service.result(first["run_id"])["state"] == AutomationRunState.CANCELLED.value
    assert not output_root.exists()

    with get_readonly_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM automation_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM automation_grants").fetchone()[0] == 1


def test_concurrent_standing_starts_reuse_one_ready_run(
    tmp_path: Path,
    fake_keyring: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent standing starts issue one grant and return one READY run."""
    db_path = tmp_path / "ctpf.db"
    policy, spec, output_root = _material(tmp_path, AuthorizationTier.LOCAL_SYNTHETIC)
    service = AutomationService(db_path=db_path)
    approval.initialize_approval_key()
    service.create_policy(policy, now=NOW)
    validation_barrier = Barrier(2)
    original_validate = automation_service._validate_with_connection

    def synchronized_validate(
        conn: sqlite3.Connection,
        candidate: RunSpec,
        current: datetime.datetime,
    ) -> ValidationResult:
        result = original_validate(conn, candidate, current)
        validation_barrier.wait(timeout=CONCURRENT_START_TIMEOUT_SECONDS)
        return result

    monkeypatch.setattr(automation_service, "_validate_with_connection", synchronized_validate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(service.start, spec, now=NOW) for _ in range(2)]
        starts = [future.result(timeout=CONCURRENT_START_TIMEOUT_SECONDS) for future in futures]

    assert {start["run_id"] for start in starts} == {starts[0]["run_id"]}
    assert sorted(start["created"] for start in starts) == [False, True]
    assert all(start["state"] == AutomationRunState.READY.value for start in starts)
    assert not output_root.exists()
    with get_readonly_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM automation_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM automation_grants").fetchone()[0] == 1


def test_tier_two_requires_exact_human_approval_and_rejects_replay(
    tmp_path: Path,
    fake_keyring: dict[str, str],
) -> None:
    """Tier-2 start accepts only a signed grant bound to the immutable RunSpec."""
    db_path = tmp_path / "ctpf.db"
    policy, spec, output_root = _material(tmp_path, AuthorizationTier.BOUNDED_REMOTE)
    service = AutomationService(db_path=db_path)
    approval.initialize_approval_key()
    service.create_policy(policy, now=NOW)

    validation = service.validate(spec, now=NOW)
    assert validation.decision.kind == DecisionKind.APPROVAL_REQUIRED
    _assert_control_error("approval_required", lambda: service.start(spec, now=NOW))
    authorization = service.create_approval(spec, now=NOW)
    approval_id = authorization["approval"]["grant_id"]
    started = service.start(spec, approval_id=approval_id, now=NOW)
    assert started["state"] == AutomationRunState.READY.value
    assert service.start(spec, approval_id=approval_id, now=NOW)["created"] is False

    changed = replace(spec, idempotency_key="agent-request-0002")
    _assert_control_error(
        "approval_invalid",
        lambda: service.start(changed, approval_id=approval_id, now=NOW),
    )
    assert not output_root.exists()
