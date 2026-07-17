"""Domain service for governed CTPF automation."""

from __future__ import annotations

import asyncio
import datetime
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ctpf.automation.approval import (
    ApprovalError,
    authenticate_authorization_grant,
    authenticate_policy,
    issue_authorization_grant,
    sign_policy,
)
from ctpf.automation.canonical import (
    CanonicalizationError,
    load_canonical_object,
    sha256_digest,
)
from ctpf.automation.contracts import (
    AuthorizationGrant,
    AutomationRunState,
    DecisionKind,
    GrantSource,
    PolicyDecision,
    PolicyDocument,
    RunSpec,
)
from ctpf.automation.control import (
    BudgetExhaustedError,
    ExecutionCancelledError,
    ExecutionControl,
    ExecutionDeadlineExceededError,
    ExecutionInterruptedError,
    deadline_at,
    format_timestamp,
    new_lease_id,
    stale_before,
)
from ctpf.automation.envelope import ControlError
from ctpf.automation.policy import evaluate_policy
from ctpf.automation.store import (
    AutomationRunRecord,
    AutomationStoreError,
    GrantReplayError,
    IdempotencyConflictError,
    StoredGrant,
    StoredPolicy,
    bind_grant_and_create_ready_run,
    claim_run_execution,
    get_automation_run,
    get_automation_run_by_idempotency,
    get_grant,
    get_policy,
    list_events,
    reconcile_stale_runs,
    save_grant,
    save_policy,
    transition_run_state,
)
from ctpf.automation.store import (
    revoke_grant as revoke_stored_grant,
)
from ctpf.automation.store import (
    revoke_policy as revoke_stored_policy,
)
from ctpf.automation.targets import (
    ScenarioCapability,
    TargetIdentity,
    TargetIdentityError,
    installed_scenario_capabilities,
    scenario_capability,
    target_identity_from_policy,
    target_identity_from_profile,
)
from ctpf.core.db import database_path, get_connection, get_readonly_connection
from ctpf.core.schema import CURRENT_VERSION

_TERMINAL_STATES = {
    AutomationRunState.COMPLETED,
    AutomationRunState.FAILED,
    AutomationRunState.CANCELLED,
    AutomationRunState.INTERRUPTED,
}


@dataclass(frozen=True)
class AuthenticatedPolicy:
    """Authenticated stored policy and its canonical metadata."""

    policy: PolicyDocument
    stored: StoredPolicy
    digest: str


@dataclass(frozen=True)
class ValidationResult:
    """Complete deterministic RunSpec validation result."""

    spec: RunSpec
    policy: AuthenticatedPolicy
    capability: ScenarioCapability
    targets: tuple[TargetIdentity, ...]
    decision: PolicyDecision

    def to_payload(self) -> dict[str, Any]:
        """Return the secret-free machine validation result."""
        source = {
            DecisionKind.ALLOWED_STANDING_POLICY: GrantSource.STANDING_POLICY.value,
            DecisionKind.APPROVAL_REQUIRED: GrantSource.HUMAN_PER_RUN.value,
        }.get(self.decision.kind)
        return {
            "authorization_source": source,
            "decision": self.decision.to_payload(),
            "normalized_spec": self.spec.to_payload(),
            "policy_id": self.policy.policy.policy_id,
            "scenario": {
                "fingerprint": self.capability.fingerprint,
                "scenario": self.capability.scenario,
            },
            "targets": [_target_summary(target) for target in self.targets],
        }


class AutomationService:
    """Operate the governed control plane and approved packaged experiments."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        """Configure the service with an optional test database path."""
        self._db_path = db_path

    def capabilities(
        self,
        policy_id: str | None = None,
        *,
        now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Return static capabilities and optional authenticated policy scope."""
        payload: dict[str, Any] = {
            "execute_available": True,
            "scenarios": [item.to_payload() for item in installed_scenario_capabilities()],
            "verify_available": True,
        }
        if policy_id is None:
            return payload
        current = _aware_utc(now)
        with self._read_connection("policy_not_found") as conn:
            authority = _load_active_policy(conn, _full_id(policy_id, "policy_id"), current)
        payload["policy"] = _authorized_policy_summary(authority)
        return payload

    def validate(
        self,
        spec: RunSpec,
        *,
        now: datetime.datetime | None = None,
    ) -> ValidationResult:
        """Validate and policy-evaluate one RunSpec without durable effects."""
        current = _aware_utc(now)
        with self._read_connection("policy_not_found") as conn:
            return _validate_with_connection(conn, spec, current)

    def start(
        self,
        spec: RunSpec,
        *,
        approval_id: str | None = None,
        now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Create or return one idempotent READY run without executing it."""
        current = _aware_utc(now)
        try:
            with get_connection(self._db_path) as conn:
                _require_current_schema(conn)
                existing = get_automation_run_by_idempotency(conn, spec.idempotency_key)
                if existing is not None:
                    return _existing_start(existing, spec, approval_id)
                validation = _validate_with_connection(conn, spec, current)
                _require_authorized(validation)
                conn.execute("BEGIN IMMEDIATE")
                existing = get_automation_run_by_idempotency(conn, spec.idempotency_key)
                if existing is not None:
                    return _existing_start(existing, spec, approval_id)
                grant = _grant_for_start(conn, validation, approval_id, current)
                binding = bind_grant_and_create_ready_run(conn, spec, grant, now=current)
        except ControlError:
            raise
        except IdempotencyConflictError as exc:
            raise ControlError("idempotency_conflict", "idempotency key conflicts") from exc
        except GrantReplayError as exc:
            raise ControlError("approval_replayed", "approval is already bound") from exc
        except AutomationStoreError as exc:
            raise ControlError("approval_invalid", "authorization binding failed") from exc
        return _start_payload(binding.run, created=binding.created)

    def status(
        self,
        run_id: str,
        *,
        now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Return one governed run's lifecycle state and bounded event history."""
        current = _aware_utc(now)
        with self._read_connection("run_not_found") as conn:
            run = _load_run(conn, run_id)
            events = list_events(conn, run.run_id)
        if run.state in {AutomationRunState.RUNNING, AutomationRunState.CANCEL_REQUESTED}:
            with get_connection(self._db_path) as conn:
                reconcile_stale_runs(
                    conn,
                    stale_before=format_timestamp(stale_before(current)),
                    now=format_timestamp(current),
                )
                run = _load_run(conn, run.run_id)
                events = list_events(conn, run.run_id)
        return _status_payload(run, events)

    async def execute(
        self,
        run_id: str,
        *,
        now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Claim and foreground-run one exact authorized READY control."""
        current = _aware_utc(now)
        control = self._claim_execution(_full_id(run_id, "run_id"), current)
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(control.heartbeat_loop(stop))
        try:
            return await _execute_claimed(control)
        except ExecutionCancelledError as exc:
            _finish_failure(control, AutomationRunState.CANCELLED, "cancelled")
            raise ControlError("cancelled", "governed execution was cancelled") from exc
        except ExecutionDeadlineExceededError as exc:
            _finish_failure(control, AutomationRunState.FAILED, "deadline_exceeded")
            raise ControlError("deadline_exceeded", "governed execution deadline expired") from exc
        except BudgetExhaustedError as exc:
            _finish_failure(control, AutomationRunState.FAILED, "budget_exhausted")
            raise ControlError(
                "budget_exhausted", "governed execution budget was exhausted"
            ) from exc
        except ExecutionInterruptedError as exc:
            _finish_failure(control, AutomationRunState.INTERRUPTED, "interrupted")
            raise ControlError("interrupted", "governed execution was interrupted") from exc
        except ControlError:
            raise
        except BaseException as exc:
            _finish_failure(control, AutomationRunState.FAILED, "execution_failed")
            raise ControlError("execution_failed", "governed experiment execution failed") from exc
        finally:
            stop.set()
            await asyncio.gather(heartbeat, return_exceptions=True)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Request cancellation or cancel a READY run without executing it."""
        try:
            with get_connection(self._db_path) as conn:
                _require_current_schema(conn)
                run = _load_run(conn, run_id)
                run = _cancel_run(conn, run)
                events = list_events(conn, run.run_id)
        except ControlError:
            raise
        except AutomationStoreError as exc:
            raise ControlError("run_state_conflict", "run state changed concurrently") from exc
        return _status_payload(run, events)

    def result(self, run_id: str) -> dict[str, Any]:
        """Return an available terminal mechanical record without reading evidence paths."""
        with self._read_connection("run_not_found") as conn:
            run = _load_run(conn, run_id)
            events = list_events(conn, run.run_id)
        if run.state not in _TERMINAL_STATES:
            raise ControlError("result_unavailable", "run has no terminal result")
        return {
            "error": _optional_stored_object(run.error_json, "error_json"),
            "events": events,
            "result": _optional_stored_object(run.result_json, "result_json"),
            "run_id": run.run_id,
            "state": run.state.value,
        }

    def verify(self, run_id: str) -> dict[str, Any]:
        """Verify the run's declared evidence bundle without accepting arbitrary paths."""
        with self._read_connection("run_not_found") as conn:
            run = _load_run(conn, run_id)
        if run.state not in _TERMINAL_STATES:
            raise ControlError("result_unavailable", "run has no terminal result")
        if run.run_root is None:
            raise ControlError("evidence_missing", "governed run has no output root")
        result = _optional_stored_object(run.result_json, "result_json")
        if result is None:
            raise ControlError("evidence_missing", "governed run has no mechanical result")
        bundle_rel = result.get("bundle")
        if not isinstance(bundle_rel, str) or not bundle_rel.strip():
            raise ControlError("evidence_missing", "mechanical result does not declare a bundle")
        bundle_dir = _resolve_run_relative(Path(run.run_root), bundle_rel, "bundle")
        from ctpf.kernel.verify import verify_evidence_bundle

        verification = verify_evidence_bundle(bundle_dir)
        payload = verification.to_payload()
        payload["bundle"] = bundle_rel
        payload["run_id"] = run.run_id
        if not verification.ok:
            issue = verification.failures[0] if verification.failures else None
            code = issue.code if issue is not None else "manifest_invalid"
            message = issue.message if issue is not None else "evidence verification failed"
            raise ControlError(code, message, details={"verification": payload})
        return payload

    def create_policy(
        self,
        policy: PolicyDocument,
        *,
        now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Validate, sign, and store one human-confirmed policy."""
        current = _aware_utc(now)
        _validate_policy_semantics(policy)
        try:
            signature, key_id = sign_policy(policy)
            authenticate_policy(policy, signature, key_id, now=current)
            with get_connection(self._db_path) as conn:
                _require_current_schema(conn)
                digest = save_policy(conn, policy, signature=signature, key_id=key_id)
        except ApprovalError as exc:
            raise ControlError("policy_invalid", "policy authentication failed") from exc
        except AutomationStoreError as exc:
            raise ControlError("policy_conflict", "policy could not be stored") from exc
        return {
            "key_id": key_id,
            "policy_digest": digest,
            "policy_id": policy.policy_id,
            "status": "active",
        }

    def create_approval(
        self,
        spec: RunSpec,
        *,
        now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Issue and store one human-confirmed per-run authorization grant."""
        current = _aware_utc(now)
        try:
            with get_connection(self._db_path) as conn:
                _require_current_schema(conn)
                validation = _validate_with_connection(conn, spec, current)
                _require_per_run_approval(validation)
                grant, signature = _issue_grant(validation, GrantSource.HUMAN_PER_RUN, current)
                save_grant(conn, grant, signature=signature)
        except ControlError:
            raise
        except (ApprovalError, AutomationStoreError) as exc:
            raise ControlError("approval_invalid", "approval could not be issued") from exc
        return {
            "approval": grant.to_payload(),
            "signature": signature,
        }

    def inspect_policy(self, policy_id: str) -> dict[str, Any]:
        """Return one stored policy body and non-secret authentication metadata."""
        with self._read_connection("policy_not_found") as conn:
            stored = get_policy(conn, _full_id(policy_id, "policy_id"))
        if stored is None:
            raise ControlError("policy_not_found", "policy was not found")
        return _stored_policy_payload(stored)

    def inspect_approval(self, approval_id: str) -> dict[str, Any]:
        """Return one stored approval body and non-secret binding metadata."""
        with self._read_connection("approval_invalid") as conn:
            stored = get_grant(conn, _full_id(approval_id, "approval_id"))
        if stored is None:
            raise ControlError("approval_invalid", "approval was not found")
        return _stored_grant_payload(stored)

    def revoke_policy(self, policy_id: str) -> dict[str, Any]:
        """Revoke one exact stored policy without deleting its audit record."""
        selected = _full_id(policy_id, "policy_id")
        with get_connection(self._db_path) as conn:
            _require_current_schema(conn)
            if not revoke_stored_policy(conn, selected):
                raise ControlError("policy_conflict", "policy is missing or already revoked")
        return {"policy_id": selected, "status": "revoked"}

    def revoke_approval(self, approval_id: str) -> dict[str, Any]:
        """Revoke one exact stored approval without deleting its audit record."""
        selected = _full_id(approval_id, "approval_id")
        with get_connection(self._db_path) as conn:
            _require_current_schema(conn)
            if not revoke_stored_grant(conn, selected):
                raise ControlError("approval_conflict", "approval is missing or already revoked")
        return {"approval_id": selected, "status": "revoked"}

    @contextmanager
    def _read_connection(self, missing_code: str) -> Iterator[sqlite3.Connection]:
        try:
            with get_readonly_connection(self._db_path) as conn:
                _require_current_schema(conn)
                yield conn
        except ControlError:
            raise
        except (FileNotFoundError, sqlite3.Error) as exc:
            raise ControlError(missing_code, "automation database is unavailable") from exc

    def _claim_execution(
        self,
        run_id: str,
        now: datetime.datetime,
    ) -> ExecutionControl:
        try:
            with get_connection(self._db_path) as conn:
                _require_current_schema(conn)
                run = _load_run(conn, run_id)
                _require_ready_run(run)
                spec = RunSpec.from_payload(_stored_object(run.spec_json, "spec_json"))
                validation = _validate_with_connection(conn, spec, now)
                _require_authorized(validation)
                grant = _load_approval(conn, run.grant_id, validation, now)
                _validate_execution_binding(run, grant, validation)
                root = _execution_output_root(validation.policy.policy, spec.output_root_id)
                planned_root = root / run.run_id
                _require_unused_run_root(planned_root)
                deadline = deadline_at(now, spec.limits.wall_clock_seconds)
                conn.execute("BEGIN IMMEDIATE")
                claimed = claim_run_execution(
                    conn,
                    run.run_id,
                    lease_id=new_lease_id(),
                    run_root=str(planned_root),
                    manifest_path=str(planned_root / _manifest_name(spec)),
                    deadline_at=format_timestamp(deadline),
                    concurrent_runs=validation.policy.policy.limits.concurrent_runs,
                    stale_before=format_timestamp(stale_before(now)),
                    now=format_timestamp(now),
                )
        except ControlError:
            raise
        except (AutomationStoreError, ValueError) as exc:
            code = "concurrency_exhausted" if "concurrency" in str(exc) else "run_state_conflict"
            raise ControlError(code, "governed run could not be claimed") from exc
        return ExecutionControl(
            db_path=database_path(self._db_path),
            run=claimed,
            spec=spec,
            policy=validation.policy.policy,
            capability=validation.capability,
            deadline_at=deadline,
        )


async def _execute_claimed(control: ExecutionControl) -> dict[str, Any]:
    profiles = await _revalidate_live_targets(control)
    from ctpf.experiment import run_governed_experiment

    outcome = await run_governed_experiment(control, profiles)
    control.checkpoint("finalization")
    provenance_path = control.run_root / "automation-provenance.json"
    _write_json(provenance_path, control.provenance_payload())
    result = dict(outcome.result)
    result["automation_provenance"] = provenance_path.relative_to(control.run_root).as_posix()
    try:
        finished = control.finish(
            AutomationRunState.COMPLETED,
            result=result,
            manifest_path=outcome.manifest_path,
        )
    except AutomationStoreError:
        control.checkpoint("completion")
        raise
    return _execution_payload(finished)


async def _revalidate_live_targets(control: ExecutionControl) -> tuple[Any, ...]:
    from ctpf.external_runtime import load_governed_target_profile

    profiles: list[Any] = []
    fingerprints: dict[str, str] = {}
    try:
        for reference in control.spec.experiment.targets:
            profile = await load_governed_target_profile(
                reference.target_id,
                control,
                db_path=control.db_path,
            )
            identity = target_identity_from_profile(profile)
            _require_live_target(reference.target_fingerprint, identity.fingerprint)
            profiles.append(profile)
            fingerprints[identity.target_id] = identity.fingerprint
    except ExecutionInterruptedError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExecutionInterruptedError("live target revalidation failed") from exc
    control.record_revalidated_targets(fingerprints)
    return tuple(profiles)


def _require_ready_run(run: AutomationRunRecord) -> None:
    if run.state != AutomationRunState.READY:
        raise ControlError("run_state_conflict", "run is not READY")


def _require_unused_run_root(path: Path) -> None:
    if path.exists():
        raise ControlError("output_root_changed", "governed run root already exists")


def _require_live_target(expected: str, actual: str) -> None:
    if actual != expected:
        raise ExecutionInterruptedError("live target fingerprint changed after approval")


def _resolve_run_relative(run_root: Path, relative: str, label: str) -> Path:
    """Resolve one declared run-relative path without accepting escapes."""
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or relative.startswith(("/", "\\"))
        or any(part in {"", ".", ".."} for part in posix.parts)
        or any(":" in part for part in posix.parts)
    ):
        raise ControlError("artifact_path_invalid", f"{label} path is unsafe")
    selected = run_root.joinpath(*posix.parts)
    try:
        resolved = selected.resolve(strict=False)
        boundary = run_root.resolve(strict=False)
    except OSError as exc:
        raise ControlError("evidence_missing", f"{label} path is unreadable") from exc
    if not resolved.is_relative_to(boundary):
        raise ControlError("artifact_path_invalid", f"{label} path escapes the run root")
    return selected


def _validate_execution_binding(
    run: AutomationRunRecord,
    grant: AuthorizationGrant,
    validation: ValidationResult,
) -> None:
    identity = (
        run.policy_id == validation.policy.policy.policy_id
        and run.policy_digest == validation.policy.digest
        and run.grant_id == grant.grant_id
        and run.spec_digest == validation.decision.spec_digest
        and grant.spec_digest == run.spec_digest
    )
    if not identity:
        raise ControlError("approval_spec_mismatch", "run authority no longer matches")


def _execution_output_root(policy: PolicyDocument, root_id: str) -> Path:
    selected = next((item for item in policy.output_roots if item.root_id == root_id), None)
    if selected is None:
        raise ControlError("output_root_denied", "output root is not authorized")
    declared = Path(selected.resolved_path)
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise ControlError("output_root_changed", "approved output root is unavailable") from exc
    if not resolved.is_dir() or str(resolved) != selected.resolved_path:
        raise ControlError("output_root_changed", "approved output root identity changed")
    if _inside_git_checkout(resolved):
        raise ControlError("output_root_in_git", "approved output root is inside Git")
    return resolved


def _manifest_name(spec: RunSpec) -> str:
    return "series-manifest.json" if spec.experiment.mode.value == "matrix" else "run-manifest.json"


def _finish_failure(
    control: ExecutionControl,
    state: AutomationRunState,
    code: str,
) -> None:
    try:
        control.finish(
            state, error={"code": code, "message": "governed execution did not complete"}
        )
    except AutomationStoreError:
        return


def _execution_payload(run: AutomationRunRecord) -> dict[str, Any]:
    return {
        "result": _optional_stored_object(run.result_json, "result_json"),
        "run_id": run.run_id,
        "state": run.state.value,
        "usage": _stored_object(run.usage_json, "usage_json"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_with_connection(
    conn: sqlite3.Connection,
    spec: RunSpec,
    now: datetime.datetime,
) -> ValidationResult:
    authority = _load_active_policy(conn, spec.policy_id, now)
    try:
        capability = scenario_capability(spec.experiment.scenario)
        identities = _policy_identities(authority.policy, spec)
    except TargetIdentityError as exc:
        raise ControlError("policy_invalid", "signed target snapshot is invalid") from exc
    decision = evaluate_policy(spec, authority.policy, capability, identities, now=now)
    return ValidationResult(spec, authority, capability, identities, decision)


def _load_active_policy(
    conn: sqlite3.Connection,
    policy_id: str,
    now: datetime.datetime,
) -> AuthenticatedPolicy:
    stored = get_policy(conn, policy_id)
    if stored is None:
        raise ControlError("policy_not_found", "policy was not found")
    if stored.status != "active":
        raise ControlError("policy_revoked", "policy is revoked")
    try:
        policy = PolicyDocument.from_payload(load_canonical_object(stored.policy_json))
        digest = authenticate_policy(policy, stored.signature, stored.key_id, now=now)
    except ApprovalError as exc:
        code = "policy_expired" if "validity interval" in str(exc) else "policy_invalid"
        raise ControlError(code, "policy authentication failed") from exc
    except (CanonicalizationError, ValueError) as exc:
        raise ControlError("policy_invalid", "stored policy is malformed") from exc
    identity = (
        policy.policy_id == stored.policy_id
        and digest == stored.policy_digest
        and policy.created_at == stored.created_at
        and policy.expires_at == stored.expires_at
    )
    if not identity:
        raise ControlError("policy_invalid", "stored policy metadata does not match")
    return AuthenticatedPolicy(policy, stored, digest)


def _policy_identities(policy: PolicyDocument, spec: RunSpec) -> tuple[TargetIdentity, ...]:
    targets = {target.target_id: target for target in policy.targets}
    identities: list[TargetIdentity] = []
    for reference in spec.experiment.targets:
        target = targets.get(reference.target_id)
        if target is None:
            continue
        identities.append(target_identity_from_policy(target))
    return tuple(identities)


def _require_authorized(validation: ValidationResult) -> None:
    if validation.decision.kind == DecisionKind.DENIED:
        raise ControlError(
            "policy_denied",
            "policy denied the RunSpec",
            details={"reason_code": validation.decision.reason_code},
        )


def _require_per_run_approval(validation: ValidationResult) -> None:
    if validation.decision.kind != DecisionKind.APPROVAL_REQUIRED:
        raise ControlError(
            "approval_invalid",
            "policy decision does not require a human per-run approval",
        )


def _grant_for_start(
    conn: sqlite3.Connection,
    validation: ValidationResult,
    approval_id: str | None,
    now: datetime.datetime,
) -> AuthorizationGrant:
    if validation.decision.kind == DecisionKind.ALLOWED_STANDING_POLICY:
        if approval_id is not None:
            raise ControlError("approval_invalid", "standing authorization takes no approval ID")
        grant, signature = _issue_grant(validation, GrantSource.STANDING_POLICY, now)
        save_grant(conn, grant, signature=signature)
        return grant
    if approval_id is None:
        raise ControlError(
            "approval_required",
            "a human per-run approval is required",
            details={"spec_digest": validation.decision.spec_digest},
        )
    return _load_approval(conn, approval_id, validation, now)


def _issue_grant(
    validation: ValidationResult,
    source: GrantSource,
    now: datetime.datetime,
) -> tuple[AuthorizationGrant, str]:
    policy = validation.policy.policy
    lifetime = _grant_lifetime_seconds(policy, now)
    return issue_authorization_grant(
        validation.spec,
        policy,
        validation.decision,
        source,
        policy_signature=validation.policy.stored.signature,
        policy_key_id=validation.policy.stored.key_id,
        lifetime_seconds=lifetime,
        issued_at=now,
    )


def _load_approval(
    conn: sqlite3.Connection,
    approval_id: str,
    validation: ValidationResult,
    now: datetime.datetime,
) -> AuthorizationGrant:
    stored = get_grant(conn, _full_id(approval_id, "approval_id"))
    if stored is None:
        raise ControlError("approval_invalid", "approval was not found")
    if stored.revoked_at is not None:
        raise ControlError("approval_revoked", "approval is revoked")
    try:
        grant = AuthorizationGrant.from_payload(load_canonical_object(stored.grant_json))
        authenticate_authorization_grant(
            grant,
            stored.signature,
            validation.spec,
            validation.policy.policy,
            now=now,
        )
    except ApprovalError as exc:
        code = "approval_expired" if "validity interval" in str(exc) else "approval_invalid"
        raise ControlError(code, "approval authentication failed") from exc
    except (CanonicalizationError, ValueError) as exc:
        raise ControlError("approval_invalid", "stored approval is malformed") from exc
    if grant.grant_id != stored.grant_id or grant.key_id != stored.key_id:
        raise ControlError("approval_invalid", "stored approval metadata does not match")
    return grant


def _existing_start(
    run: AutomationRunRecord,
    spec: RunSpec,
    approval_id: str | None,
) -> dict[str, Any]:
    if run.spec_digest != sha256_digest(spec.to_payload()):
        raise ControlError("idempotency_conflict", "idempotency key conflicts")
    if approval_id is not None and run.grant_id != _full_id(approval_id, "approval_id"):
        raise ControlError("idempotency_conflict", "idempotency key uses another approval")
    return _start_payload(run, created=False)


def _cancel_run(conn: sqlite3.Connection, run: AutomationRunRecord) -> AutomationRunRecord:
    if run.state == AutomationRunState.READY:
        return transition_run_state(
            conn,
            run.run_id,
            AutomationRunState.READY,
            AutomationRunState.CANCELLED,
            event_payload={"reason": "cancelled_before_execution"},
        )
    if run.state == AutomationRunState.RUNNING:
        return transition_run_state(
            conn,
            run.run_id,
            AutomationRunState.RUNNING,
            AutomationRunState.CANCEL_REQUESTED,
        )
    return run


def _load_run(conn: sqlite3.Connection, run_id: str) -> AutomationRunRecord:
    run = get_automation_run(conn, _full_id(run_id, "run_id"))
    if run is None:
        raise ControlError("run_not_found", "run was not found")
    return run


def _status_payload(run: AutomationRunRecord, events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "budget": _stored_object(run.budget_json, "budget_json"),
        "cancel_requested_at": run.cancel_requested_at,
        "created_at": run.created_at,
        "events": events,
        "execute_available": True,
        "finished_at": run.finished_at,
        "run_id": run.run_id,
        "scenario_fingerprint": run.scenario_fingerprint,
        "spec_digest": run.spec_digest,
        "started_at": run.started_at,
        "state": run.state.value,
        "updated_at": run.updated_at,
        "usage": _stored_object(run.usage_json, "usage_json"),
    }


def _start_payload(run: AutomationRunRecord, *, created: bool) -> dict[str, Any]:
    return {
        "created": created,
        "execute_available": True,
        "run_id": run.run_id,
        "spec_digest": run.spec_digest,
        "state": run.state.value,
    }


def _authorized_policy_summary(authority: AuthenticatedPolicy) -> dict[str, Any]:
    policy = authority.policy
    return {
        "expires_at": policy.expires_at,
        "output_root_ids": [root.root_id for root in policy.output_roots],
        "policy_digest": authority.digest,
        "policy_id": policy.policy_id,
        "scenarios": [item.to_payload() for item in policy.scenarios],
        "targets": [
            {
                "billing_class": item.billing_class.value,
                "network_class": item.network_class.value,
                "target_fingerprint": item.target_fingerprint,
                "target_id": item.target_id,
                "target_type": item.target_type,
            }
            for item in policy.targets
        ],
    }


def _target_summary(target: TargetIdentity) -> dict[str, Any]:
    return {
        "network_class": target.network_class.value,
        "target_fingerprint": target.fingerprint,
        "target_id": target.target_id,
        "target_type": target.target_type,
    }


def _stored_policy_payload(stored: StoredPolicy) -> dict[str, Any]:
    try:
        body = load_canonical_object(stored.policy_json)
    except CanonicalizationError as exc:
        raise ControlError("policy_invalid", "stored policy is malformed") from exc
    return {
        "key_id": stored.key_id,
        "policy": body,
        "policy_digest": stored.policy_digest,
        "policy_id": stored.policy_id,
        "revoked_at": stored.revoked_at,
        "status": stored.status,
    }


def _stored_grant_payload(stored: StoredGrant) -> dict[str, Any]:
    try:
        body = load_canonical_object(stored.grant_json)
    except CanonicalizationError as exc:
        raise ControlError("approval_invalid", "stored approval is malformed") from exc
    return {
        "approval": body,
        "approval_id": stored.grant_id,
        "bound_run_id": stored.bound_run_id,
        "key_id": stored.key_id,
        "revoked_at": stored.revoked_at,
    }


def _validate_policy_semantics(policy: PolicyDocument) -> None:
    capabilities = {item.scenario: item for item in installed_scenario_capabilities()}
    expected_effects: set[str] = set()
    for selected in policy.scenarios:
        capability = capabilities.get(selected.scenario)
        if capability is None or selected.fingerprints != (capability.fingerprint,):
            raise ControlError("policy_invalid", "policy scenario fingerprint is not installed")
        if not set(selected.modes).issubset(capability.modes):
            raise ControlError("policy_invalid", "policy scenario mode is not installed")
        expected_effects.update(capability.effect_ids)
    if set(policy.allowed_effects) != expected_effects:
        raise ControlError("policy_invalid", "policy effects must exactly match its scenarios")
    for target in policy.targets:
        try:
            target_identity_from_policy(target)
        except TargetIdentityError as exc:
            raise ControlError("policy_invalid", "policy target snapshot is invalid") from exc
    for root in policy.output_roots:
        if _inside_git_checkout(Path(root.resolved_path)):
            raise ControlError("policy_invalid", "policy output root must be outside Git")


def _inside_git_checkout(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    return any((candidate / ".git").exists() for candidate in (resolved, *resolved.parents))


def _grant_lifetime_seconds(policy: PolicyDocument, now: datetime.datetime) -> int:
    expires = datetime.datetime.strptime(policy.expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.UTC
    )
    remaining = int((expires - now).total_seconds())
    lifetime = min(policy.limits.approval_lifetime_seconds, remaining)
    if lifetime < 1:
        raise ControlError("policy_expired", "policy cannot authorize a new approval")
    return lifetime


def _stored_object(raw: str, label: str) -> dict[str, Any]:
    try:
        return load_canonical_object(raw)
    except CanonicalizationError as exc:
        raise ControlError("run_state_conflict", f"stored {label} is malformed") from exc


def _optional_stored_object(raw: str | None, label: str) -> dict[str, Any] | None:
    return None if raw is None else _stored_object(raw, label)


def _require_current_schema(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version != CURRENT_VERSION:
        raise ControlError(
            "schema_version_unsupported",
            "automation database schema is not current",
            details={"actual": version, "expected": CURRENT_VERSION},
        )


def _full_id(raw: str, label: str) -> str:
    value = raw.strip().lower()
    if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
        raise ControlError("invalid_field", f"{label} must be a full lowercase hexadecimal ID")
    return value


def _aware_utc(value: datetime.datetime | None) -> datetime.datetime:
    current = value or datetime.datetime.now(datetime.UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ControlError("invalid_field", "evaluation time must be timezone-aware")
    return current.astimezone(datetime.UTC).replace(microsecond=0)
