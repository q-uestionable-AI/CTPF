"""Adversarial acceptance for the Section 15 threat-to-control matrix.

A deterministic untrusted caller exercises the governed control surface with
synthetic targets only. No live model or external network endpoint is used.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import JSONRPCMessage, JSONRPCRequest

from ctpf import driven_inference, experiment
from ctpf.automation import approval
from ctpf.automation import control as execution_control
from ctpf.automation import service as automation_service
from ctpf.automation.canonical import CANONICALIZATION_ID
from ctpf.automation.contracts import (
    AuthorizationTier,
    AutomationRunState,
    BillingClass,
    ContractError,
    DataEgressClass,
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
from ctpf.automation.service import AutomationService
from ctpf.automation.store import save_policy
from ctpf.automation.targets import scenario_capability, target_identity_from_profile
from ctpf.core.db import get_connection, get_readonly_connection
from ctpf.kernel import (
    BASELINE_TRACE_NAME,
    CONDITION_BASELINE,
    CONDITION_MANIPULATED,
    MANIFEST_NAME,
    MANIPULATED_SINK_NAME,
    MANIPULATED_TRACE_NAME,
    ExperimentContext,
    ExperimentPins,
    ExternalEffect,
    Pattern2Scenario,
    PromotionReason,
    PromotionResult,
    RunObservation,
    compare_baseline_manipulated,
    verify_evidence_bundle,
    write_evidence_bundle,
)
from ctpf.mcp.models import Direction, Transport
from ctpf.proxy.models import ProxyMessage

POLICY_ID = "a" * 32
TARGET_ID = "b" * 32
SECOND_TARGET_ID = "c" * 32
CONCURRENT_TIMEOUT_SECONDS = 10
SECRET_VALUE = "super-secret-provider-token"
PINS = ExperimentPins(
    agent="adversarial-caller",
    model="test-model",
    configuration={"scenario": "pattern2"},
)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Provide process-local signing-key storage for adversarial cases."""
    secrets: dict[str, str] = {}
    monkeypatch.setattr(approval, "get_local_secret", secrets.get)
    monkeypatch.setattr(approval, "set_local_secret", secrets.__setitem__)
    monkeypatch.setattr(approval, "delete_local_secret", lambda name: secrets.pop(name, None))
    return secrets


def _limits(*, cost: int = 0, provider_requests: int = 36) -> ResourceLimits:
    return ResourceLimits(3_240, provider_requests, 9_216, 9_216, 36, 4, cost)


def _target(
    network: NetworkClass = NetworkClass.LOOPBACK,
    target_id: str = TARGET_ID,
) -> TargetPolicy:
    endpoint = (
        "http://127.0.0.1:11434/v1"
        if network == NetworkClass.LOOPBACK
        else "https://models.example.test/v1"
    )
    remote = network == NetworkClass.HTTPS_PUBLIC
    identity = target_identity_from_profile(
        driven_inference.OpenAICompatibleTargetProfile(
            target_id=target_id,
            name="adversarial target",
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
        target_id,
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


def _matrix_limits() -> ResourceLimits:
    return ResourceLimits(100_000, 100_000, 1_000_000, 1_000_000, 100_000, 4, 0)


def _material(
    tmp_path: Path,
    tier: AuthorizationTier = AuthorizationTier.LOCAL_SYNTHETIC,
) -> tuple[PolicyDocument, RunSpec, Path]:
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
        name="adversarial policy",
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
        requester=Requester("agent", "unsafe-caller", "1"),
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


def _matrix_material(tmp_path: Path) -> tuple[PolicyDocument, RunSpec, Path]:
    capability = scenario_capability("cascade-memo")
    output_root = tmp_path / "research-evidence"
    targets = (_target(target_id=TARGET_ID), _target(target_id=SECOND_TARGET_ID))
    limits = _matrix_limits()
    policy = PolicyDocument(
        policy_id=POLICY_ID,
        name="adversarial matrix policy",
        created_at="2026-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
        standing_tiers=(AuthorizationTier.LOCAL_SYNTHETIC,),
        per_run_tiers=(),
        scenarios=(
            ScenarioPolicy(
                "cascade-memo",
                (capability.fingerprint,),
                (ExperimentMode.MATRIX,),
                3,
            ),
        ),
        targets=targets,
        output_roots=(OutputRootPolicy("research-evidence", str(output_root.resolve())),),
        allowed_effects=capability.effect_ids,
        limits=PolicyLimits(limits, 1, 300, 8765),
    )
    spec = RunSpec(
        idempotency_key="agent-request-matrix-0001",
        requester=Requester("agent", "unsafe-caller", "1"),
        purpose="Exercise the packaged synthetic matrix scenario.",
        policy_id=POLICY_ID,
        requested_tier=AuthorizationTier.LOCAL_SYNTHETIC,
        experiment=ExperimentRequest(
            "cascade-memo",
            capability.fingerprint,
            ExperimentMode.MATRIX,
            3,
            tuple(
                TargetReference(target.target_id, target.target_fingerprint) for target in targets
            ),
        ),
        output_root_id="research-evidence",
        limits=limits,
    )
    return policy, spec, output_root


class UnsafeCaller:
    """Deterministic untrusted caller limited to the governed service surface."""

    def __init__(self, tmp_path: Path, *, tier: AuthorizationTier | None = None) -> None:
        selected = tier or AuthorizationTier.LOCAL_SYNTHETIC
        self.db_path = tmp_path / "ctpf.db"
        self.policy, self.spec, self.output_root = _material(tmp_path, selected)
        self.service = AutomationService(db_path=self.db_path)
        self.now = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        approval.initialize_approval_key()
        self.service.create_policy(self.policy, now=self.now)

    def start(
        self,
        spec: RunSpec | None = None,
        *,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        return self.service.start(spec or self.spec, approval_id=approval_id, now=self.now)

    def expect(self, code: str | set[str], callback: Callable[[], object]) -> None:
        with pytest.raises(ControlError) as caught:
            callback()
        allowed = {code} if isinstance(code, str) else code
        assert caught.value.code in allowed


def _install_revalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def revalidate(control: execution_control.ExecutionControl) -> tuple[object, ...]:
        fingerprints = {
            target.target_id: target.target_fingerprint
            for target in control.spec.experiment.targets
        }
        control.record_revalidated_targets(fingerprints)
        return (object(),)

    monkeypatch.setattr(automation_service, "_revalidate_live_targets", revalidate)


def _ready(tmp_path: Path) -> tuple[UnsafeCaller, dict[str, Any]]:
    caller = UnsafeCaller(tmp_path)
    started = caller.start()
    caller.output_root.mkdir()
    return caller, started


def _assert_no_effect(output_root: Path) -> None:
    if not output_root.exists():
        return
    children = list(output_root.iterdir())
    assert children == [] or all(path.is_dir() and not any(path.iterdir()) for path in children)


def _observation(
    condition: str,
    *,
    tool: str | None,
    effect_present: bool,
    action: str = "approve_refund",
) -> RunObservation:
    return RunObservation(
        condition=condition,
        tool_invocation=tool,
        tool_arguments=None if tool is None else {"action": action, "reason": "test"},
        external_effect=ExternalEffect(
            present=effect_present,
            payload=(
                {"effect": "applied", "action": action, "run_id": "r1"} if effect_present else None
            ),
            sink_path=None,
            reason="effect_applied" if effect_present else "sink_missing",
        ),
        evidence_complete=True,
        evidence_notes=(),
    )


def _write_minimal_bundle(root: Path) -> Path:
    baseline = _observation(CONDITION_BASELINE, tool=None, effect_present=False)
    manipulated = _observation(CONDITION_MANIPULATED, tool="apply_change", effect_present=True)
    transition = compare_baseline_manipulated(baseline, manipulated)
    baseline_trace = root / "baseline.json"
    manipulated_trace = root / "manipulated.json"
    sink = root / "sink.json"
    root.mkdir(parents=True, exist_ok=True)
    baseline_trace.write_text("{}\n", encoding="utf-8")
    manipulated_trace.write_text("{}\n", encoding="utf-8")
    sink.write_text(json.dumps({"effect": "applied", "action": "approve_refund"}), encoding="utf-8")
    return write_evidence_bundle(
        root / "bundle",
        result=transition,
        experiment=ExperimentContext(
            baseline=baseline,
            manipulated=manipulated,
            pins=PINS,
            scenario=Pattern2Scenario(),
        ),
        artifacts={
            BASELINE_TRACE_NAME: baseline_trace,
            MANIPULATED_TRACE_NAME: manipulated_trace,
            MANIPULATED_SINK_NAME: sink,
        },
    ).root


class _ToolBoundaryControl:
    def __init__(self, names: frozenset[str]) -> None:
        self.expected_tool_names = names
        self.reservations: list[tuple[str, dict[str, int]]] = []

    def reserve(self, boundary: str, **reservation: int) -> dict[str, Any]:
        self.reservations.append((boundary, reservation))
        return {}


class TestSpecChangedAfterApproval:
    """Canonical digest in grant; verify again at start."""

    @pytest.mark.parametrize(
        ("mutate", "codes"),
        [
            (
                lambda spec: replace(spec, purpose="mutated purpose after approval"),
                {"approval_invalid"},
            ),
            (
                lambda spec: replace(spec, idempotency_key="agent-request-0002"),
                {"approval_invalid"},
            ),
            (
                lambda spec: replace(spec, limits=_limits(provider_requests=1)),
                {"approval_invalid", "policy_denied"},
            ),
            (
                lambda spec: replace(
                    spec,
                    experiment=replace(spec.experiment, scenario_fingerprint="d" * 64),
                ),
                {"approval_invalid", "policy_denied"},
            ),
            (
                lambda spec: replace(
                    spec,
                    experiment=replace(
                        spec.experiment,
                        targets=(TargetReference(TARGET_ID, "e" * 64),),
                    ),
                ),
                {"approval_invalid", "policy_denied"},
            ),
        ],
    )
    def test_post_approval_mutations_deny_before_effect(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        mutate: Callable[[RunSpec], RunSpec],
        codes: set[str],
    ) -> None:
        caller = UnsafeCaller(tmp_path, tier=AuthorizationTier.BOUNDED_REMOTE)
        approval_id = caller.service.create_approval(caller.spec, now=caller.now)["approval"][
            "grant_id"
        ]
        mutated = mutate(caller.spec)
        caller.expect(codes, lambda: caller.start(mutated, approval_id=approval_id))
        _assert_no_effect(caller.output_root)


class TestForgedAndReplayedAuthority:
    """HMAC forgery and approval replay fail closed."""

    def test_forged_policy_signature_cannot_start(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        policy, spec, output_root = _material(tmp_path)
        service = AutomationService(db_path=tmp_path / "ctpf.db")
        approval.initialize_approval_key()
        _signature, key_id = approval.sign_policy(policy)
        with get_connection(tmp_path / "ctpf.db") as conn:
            save_policy(conn, policy, signature="0" * 64, key_id=key_id)
        with pytest.raises(ControlError) as caught:
            service.start(spec, now=datetime.datetime.now(datetime.UTC).replace(microsecond=0))
        assert caught.value.code == "policy_invalid"
        _assert_no_effect(output_root)

    def test_approval_replay_with_different_spec_denies(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        caller = UnsafeCaller(tmp_path, tier=AuthorizationTier.BOUNDED_REMOTE)
        approval_id = caller.service.create_approval(caller.spec, now=caller.now)["approval"][
            "grant_id"
        ]
        other = replace(caller.spec, idempotency_key="agent-request-0002")
        caller.expect("approval_invalid", lambda: caller.start(other, approval_id=approval_id))
        _assert_no_effect(caller.output_root)


class TestTargetAndScenarioSubstitution:
    """Fingerprint mismatches deny before execution."""

    def test_target_fingerprint_substitution_denies(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        caller = UnsafeCaller(tmp_path)
        substituted = replace(
            caller.spec,
            experiment=replace(
                caller.spec.experiment,
                targets=(TargetReference(TARGET_ID, "f" * 64),),
            ),
        )
        caller.expect(
            {"target_fingerprint_mismatch", "policy_denied"},
            lambda: caller.start(substituted),
        )
        _assert_no_effect(caller.output_root)

    def test_scenario_fingerprint_drift_denies_standing_policy(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        caller = UnsafeCaller(tmp_path)
        drifted = replace(
            caller.spec,
            experiment=replace(caller.spec.experiment, scenario_fingerprint="a" * 64),
        )
        caller.expect(
            {"scenario_fingerprint_mismatch", "policy_denied"},
            lambda: caller.start(drifted),
        )
        _assert_no_effect(caller.output_root)


class TestArbitraryCommandAndUnknownFields:
    """Control contracts reject unknown fields and proxy-like payloads."""

    def test_unknown_runspec_fields_fail_closed(self) -> None:
        capability = scenario_capability("pattern2")
        target = _target()
        payload = {
            "schema_version": 1,
            "canonicalization": CANONICALIZATION_ID,
            "idempotency_key": "agent-request-0001",
            "requester": {"kind": "agent", "name": "unsafe", "version": "1"},
            "purpose": "Exercise the packaged synthetic scenario.",
            "policy_id": POLICY_ID,
            "requested_tier": AuthorizationTier.LOCAL_SYNTHETIC.value,
            "experiment": {
                "scenario": "pattern2",
                "scenario_fingerprint": capability.fingerprint,
                "mode": ExperimentMode.SINGLE.value,
                "trials_per_target": 1,
                "targets": [
                    {"target_id": TARGET_ID, "target_fingerprint": target.target_fingerprint}
                ],
            },
            "output_root_id": "research-evidence",
            "limits": _limits().to_payload(),
            "proxy_command": "stdio://evil",
            "arbitrary_url": "https://evil.example/v1",
        }
        with pytest.raises(ContractError, match="unknown fields"):
            RunSpec.from_payload(payload)


class TestOutputPathEscape:
    """Signed roots reject traversal and Git checkout paths."""

    def test_policy_rejects_traversal_output_root(self, tmp_path: Path) -> None:
        policy, _, _ = _material(tmp_path)
        payload = policy.to_payload()
        # Keep an explicit ".." component; Path.resolve() would collapse it away.
        payload["output_roots"] = [
            {
                "root_id": "research-evidence",
                "resolved_path": str(tmp_path / "safe") + "/../escaped",
            }
        ]
        with pytest.raises(ContractError, match="parent traversal"):
            PolicyDocument.from_payload(payload)

    def test_execute_rejects_git_checkout_output_root(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caller, started = _ready(tmp_path)
        monkeypatch.setattr(automation_service, "_inside_git_checkout", lambda _path: True)
        caller.expect(
            "output_root_in_git",
            lambda: asyncio.run(caller.service.execute(str(started["run_id"]), now=caller.now)),
        )
        state = caller.service.status(str(started["run_id"]))["state"]
        assert state == AutomationRunState.READY.value

    def test_session_work_rejects_relative_escape(self) -> None:
        payload = {
            "condition": "baseline",
            "fixture_run_id": "fixture",
            "inference_path": None,
            "listen_port": 8765,
            "mutation_path": None,
            "reset": False,
            "scenario": "pattern2",
            "session_name": "single",
            "target_id": TARGET_ID,
            "trace_path": "../escape.json",
            "work_id": "c" * 32,
        }
        with pytest.raises(execution_control.ExecutionInterruptedError, match="traversal"):
            execution_control.SessionWork.from_payload(payload)


class TestCredentialExfiltration:
    """Secret values must not appear in machine envelopes."""

    def test_result_and_error_envelopes_omit_secret_values(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caller, started = _ready(tmp_path)
        _install_revalidation(monkeypatch)

        async def leak(
            control: execution_control.ExecutionControl,
            profiles: tuple[object, ...],
        ) -> None:
            assert profiles
            raise RuntimeError(f"provider failed with {SECRET_VALUE}")

        monkeypatch.setattr("ctpf.experiment.run_governed_experiment", leak)
        with pytest.raises(ControlError) as caught:
            asyncio.run(caller.service.execute(str(started["run_id"]), now=caller.now))
        text = json.dumps(caught.value.details) + caught.value.message + caught.value.code
        assert SECRET_VALUE not in text
        status = caller.service.status(str(started["run_id"]))
        assert SECRET_VALUE not in json.dumps(status)


class TestPromptCannotAlterPolicy:
    """Control records are loaded only from the trusted store."""

    def test_tool_shaped_policy_payload_does_not_mutate_store(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        caller = UnsafeCaller(tmp_path)
        started = caller.start()
        poison = tmp_path / "tool-output.json"
        poison.write_text(
            json.dumps(
                {
                    "policy_id": "f" * 32,
                    "standing_tiers": ["active"],
                    "name": "attacker-written-policy",
                }
            ),
            encoding="utf-8",
        )
        status = caller.service.status(str(started["run_id"]))
        assert status["state"] == AutomationRunState.READY.value
        with get_readonly_connection(caller.db_path) as conn:
            rows = conn.execute("SELECT id, policy_json FROM automation_policies").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == POLICY_ID
        assert json.loads(rows[0][1])["name"] == "adversarial policy"
        assert poison.is_file()


class TestUnexpectedToolCapability:
    """Extra or renamed tools are denied before reservation."""

    def test_extra_tool_call_denied_before_budget_reservation(self) -> None:
        control = _ToolBoundaryControl(frozenset({"read_status", "apply_change"}))
        rule = experiment._GovernedSessionRule(control, None)  # type: ignore[arg-type]
        listed = JSONRPCMessage(JSONRPCRequest(jsonrpc="2.0", id=7, method="tools/list", params={}))
        response_raw = {
            "jsonrpc": "2.0",
            "id": 7,
            "result": {"tools": [{"name": "read_status"}, {"name": "apply_change"}]},
        }
        from mcp.types import JSONRPCResponse

        response = JSONRPCMessage(JSONRPCResponse(**response_raw))
        rule(
            ProxyMessage(
                id="list",
                sequence=0,
                timestamp=datetime.datetime.now(datetime.UTC),
                direction=Direction.CLIENT_TO_SERVER,
                transport=Transport.STDIO,
                raw=listed,
                jsonrpc_id=7,
                method="tools/list",
                correlated_id=None,
                modified=False,
                original_raw=None,
            )
        )
        rule(
            ProxyMessage(
                id="listed",
                sequence=1,
                timestamp=datetime.datetime.now(datetime.UTC),
                direction=Direction.SERVER_TO_CLIENT,
                transport=Transport.STDIO,
                raw=response,
                jsonrpc_id=7,
                method=None,
                correlated_id="list",
                modified=False,
                original_raw=None,
            )
        )
        unknown = JSONRPCMessage(
            JSONRPCRequest(
                jsonrpc="2.0",
                id=8,
                method="tools/call",
                params={"name": "not_a_scenario_tool", "arguments": {}},
            )
        )
        with pytest.raises(experiment.ExperimentError, match="allowlist"):
            rule(
                ProxyMessage(
                    id="unknown",
                    sequence=2,
                    timestamp=datetime.datetime.now(datetime.UTC),
                    direction=Direction.CLIENT_TO_SERVER,
                    transport=Transport.STDIO,
                    raw=unknown,
                    jsonrpc_id=8,
                    method="tools/call",
                    correlated_id=None,
                    modified=False,
                    original_raw=None,
                )
            )
        assert control.reservations == []


class TestBudgetConcurrencyAndCancellation:
    """Resource, idempotency, and cancellation races fail before effects."""

    def test_budget_limit_plus_one_denies(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caller, started = _ready(tmp_path)
        _install_revalidation(monkeypatch)

        async def exhaust(
            control: execution_control.ExecutionControl,
            profiles: tuple[object, ...],
        ) -> None:
            assert profiles
            control.reserve("provider_request", provider_requests=37)

        monkeypatch.setattr("ctpf.experiment.run_governed_experiment", exhaust)
        caller.expect(
            "budget_exhausted",
            lambda: asyncio.run(caller.service.execute(str(started["run_id"]), now=caller.now)),
        )
        state = caller.service.status(str(started["run_id"]))["state"]
        assert state == AutomationRunState.FAILED.value

    def test_duplicate_starts_share_one_run(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        caller = UnsafeCaller(tmp_path)
        first = caller.start()
        second = caller.start()
        assert first["run_id"] == second["run_id"]
        assert sorted([first["created"], second["created"]]) == [False, True]
        _assert_no_effect(caller.output_root)

    def test_concurrent_starts_do_not_duplicate_authority(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caller = UnsafeCaller(tmp_path)
        validation_barrier = Barrier(2)
        original_validate = automation_service._validate_with_connection

        def synchronized_validate(
            conn: sqlite3.Connection,
            candidate: RunSpec,
            current: datetime.datetime,
        ) -> object:
            result = original_validate(conn, candidate, current)
            validation_barrier.wait(timeout=CONCURRENT_TIMEOUT_SECONDS)
            return result

        monkeypatch.setattr(automation_service, "_validate_with_connection", synchronized_validate)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(caller.start) for _ in range(2)]
            starts = [future.result(timeout=CONCURRENT_TIMEOUT_SECONDS) for future in futures]
        assert {item["run_id"] for item in starts} == {starts[0]["run_id"]}
        with get_readonly_connection(caller.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM automation_runs").fetchone()[0] == 1

    @pytest.mark.parametrize("boundary", ["child_process", "provider", "tool", "finalization"])
    def test_cancel_at_effect_boundaries(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        boundary: str,
    ) -> None:
        caller, started = _ready(tmp_path)
        _install_revalidation(monkeypatch)
        entered = asyncio.Event()
        hold = asyncio.Event()

        async def gated(
            control: execution_control.ExecutionControl,
            profiles: tuple[object, ...],
        ) -> None:
            assert profiles
            control.checkpoint(boundary)
            entered.set()
            await control.wait(hold.wait(), boundary)

        monkeypatch.setattr("ctpf.experiment.run_governed_experiment", gated)

        async def exercise() -> None:
            task = asyncio.create_task(
                caller.service.execute(str(started["run_id"]), now=caller.now)
            )
            await entered.wait()
            assert caller.service.cancel(str(started["run_id"]))["state"] == "CANCEL_REQUESTED"
            with pytest.raises(ControlError) as caught:
                await task
            assert caught.value.code == "cancelled"

        asyncio.run(exercise())
        state = caller.service.status(str(started["run_id"]))["state"]
        assert state == AutomationRunState.CANCELLED.value


class TestWorkerDeathAndEvidenceTamper:
    """Lease loss and tampered evidence fail closed without resume."""

    def test_missed_heartbeats_interrupt_without_resume(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        caller, started = _ready(tmp_path)
        caller.service._claim_execution(str(started["run_id"]), caller.now)
        later = caller.now + datetime.timedelta(
            seconds=(
                execution_control.HEARTBEAT_INTERVAL_SECONDS
                * execution_control.MISSED_HEARTBEAT_LIMIT
                + 1
            )
        )
        status = caller.service.status(str(started["run_id"]), now=later)
        assert status["state"] == AutomationRunState.INTERRUPTED.value
        caller.expect(
            "run_state_conflict",
            lambda: asyncio.run(caller.service.execute(str(started["run_id"]), now=later)),
        )

    def test_control_verify_rejects_tampered_bundle(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caller, started = _ready(tmp_path)
        _install_revalidation(monkeypatch)
        bundle_root = tmp_path / "bundle-source"

        async def complete(
            control: execution_control.ExecutionControl,
            profiles: tuple[object, ...],
        ) -> SimpleNamespace:
            assert profiles
            control.run_root.mkdir(exist_ok=True)
            written = _write_minimal_bundle(bundle_root)
            target = control.run_root / "evidence" / "bundle-v1"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(written, target)
            manifest = control.run_root / "run-manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(
                manifest_path=manifest,
                result={
                    "bundle": "evidence/bundle-v1",
                    "manifest": manifest.name,
                    "primary_result": PromotionResult.CONFIRMED.value,
                    "primary_reason": (
                        PromotionReason.CONFIRMED_CLEAN_BASELINE_PROMOTED_TREATMENT.value
                    ),
                },
            )

        monkeypatch.setattr("ctpf.experiment.run_governed_experiment", complete)
        finished = asyncio.run(caller.service.execute(str(started["run_id"]), now=caller.now))
        run_id = str(finished["run_id"])
        bundle = caller.output_root / run_id / "evidence" / "bundle-v1"
        manifest_path = bundle / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first = next(iter(manifest["artifact_hashes"]))
        manifest["artifact_hashes"][first] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        assert verify_evidence_bundle(bundle).ok is False
        caller.expect("hash_mismatch", lambda: caller.service.verify(run_id))

    def test_control_verify_accepts_governed_matrix_trial_bundles(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        policy, spec, output_root = _matrix_material(tmp_path)
        service = AutomationService(db_path=tmp_path / "ctpf.db")
        approval.initialize_approval_key()
        service.create_policy(policy)
        started = service.start(spec)
        output_root.mkdir()
        _install_revalidation(monkeypatch)

        async def complete(
            control: execution_control.ExecutionControl,
            profiles: tuple[object, ...],
        ) -> SimpleNamespace:
            assert profiles
            control.run_root.mkdir(exist_ok=True)
            records = []
            for index in range(2):
                source = _write_minimal_bundle(tmp_path / f"bundle-source-{index}")
                target = control.run_root / "trials" / f"trial-{index}" / "evidence" / "bundle-v1"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
                records.append(
                    {
                        "bundle": target.relative_to(control.run_root).as_posix(),
                        "status": "complete",
                    }
                )
            manifest = control.run_root / "matrix-manifest.json"
            manifest.write_text(json.dumps({"trials": records}) + "\n", encoding="utf-8")
            return SimpleNamespace(
                manifest_path=manifest,
                result={
                    "manifest": manifest.name,
                    "mode": "matrix",
                    "scenario": "cascade-memo",
                    "trials_completed": len(records),
                },
            )

        monkeypatch.setattr("ctpf.experiment.run_governed_experiment", complete)
        finished = asyncio.run(service.execute(str(started["run_id"])))

        verification = service.verify(str(finished["run_id"]))

        assert verification["ok"] is True
        assert verification["mode"] == "matrix"
        assert verification["verified_bundles"] == 2


class TestScientificOverclaimAndSandboxResidual:
    """Mechanical envelopes cannot claim human adjudication or OS containment."""

    def test_result_envelope_has_no_human_or_ai_conclusion_fields(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caller, started = _ready(tmp_path)
        _install_revalidation(monkeypatch)

        async def complete(
            control: execution_control.ExecutionControl,
            profiles: tuple[object, ...],
        ) -> SimpleNamespace:
            assert profiles
            control.run_root.mkdir(exist_ok=True)
            manifest = control.run_root / "run-manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(
                manifest_path=manifest,
                result={
                    "manifest": manifest.name,
                    "primary_result": PromotionResult.NOT_OBSERVED.value,
                    "primary_reason": (
                        PromotionReason.NOT_OBSERVED_CLEAN_BASELINE_CLEAN_TREATMENT.value
                    ),
                },
            )

        monkeypatch.setattr("ctpf.experiment.run_governed_experiment", complete)
        finished = asyncio.run(caller.service.execute(str(started["run_id"]), now=caller.now))
        result = caller.service.result(str(finished["run_id"]))
        payload = json.dumps(result)
        forbidden = (
            "human_conclusion",
            "human_adjudication",
            "ai_conclusion",
            "scientific_claim",
            "publication_approved",
        )
        assert all(token not in payload for token in forbidden)
        assert "primary_reason" in json.dumps(result["result"])

    def test_capabilities_do_not_claim_os_sandbox_containment(
        self,
        tmp_path: Path,
        fake_keyring: dict[str, str],
    ) -> None:
        """Full-shell bypass remains an external deployment residual, not a harness claim."""
        caller = UnsafeCaller(tmp_path)
        capabilities = caller.service.capabilities()
        assert capabilities.get("os_sandbox_enforced") is not True
        assert capabilities.get("contains_full_shell_caller") is not True
        assert "execute_available" in capabilities
        assert "verify_available" in capabilities
