"""Parameterized SQLite storage for governed automation control records."""

from __future__ import annotations

import datetime
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from ctpf.automation.canonical import canonical_json, sha256_digest
from ctpf.automation.contracts import (
    AuthorizationGrant,
    AutomationRunState,
    PolicyDocument,
    RunSpec,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_STATES = {
    AutomationRunState.COMPLETED,
    AutomationRunState.FAILED,
    AutomationRunState.CANCELLED,
    AutomationRunState.INTERRUPTED,
}
_TRANSITIONS = {
    AutomationRunState.READY: {
        AutomationRunState.RUNNING,
        AutomationRunState.CANCELLED,
    },
    AutomationRunState.RUNNING: {
        AutomationRunState.CANCEL_REQUESTED,
        AutomationRunState.COMPLETED,
        AutomationRunState.FAILED,
        AutomationRunState.INTERRUPTED,
    },
    AutomationRunState.CANCEL_REQUESTED: {
        AutomationRunState.CANCELLED,
        AutomationRunState.FAILED,
        AutomationRunState.INTERRUPTED,
    },
}


class AutomationStoreError(RuntimeError):
    """Raised when an automation storage invariant is violated."""


class IdempotencyConflictError(AutomationStoreError):
    """Raised when an idempotency key is reused for a different spec."""


class GrantReplayError(AutomationStoreError):
    """Raised when one grant is bound to more than one automation run."""


@dataclass(frozen=True)
class StoredPolicy:
    """Authenticated policy row without parsing or trusting its body."""

    policy_id: str
    policy_json: str
    policy_digest: str
    signature: str
    key_id: str
    status: str
    created_at: str
    expires_at: str
    revoked_at: str | None


@dataclass(frozen=True)
class StoredGrant:
    """Authenticated authorization-grant row without trusting its body."""

    grant_id: str
    grant_json: str
    signature: str
    key_id: str
    spec_digest: str
    policy_id: str
    policy_digest: str
    issued_at: str
    expires_at: str
    revoked_at: str | None
    bound_run_id: str | None


@dataclass(frozen=True)
class AutomationRunRecord:
    """Durable governed experiment control row."""

    run_id: str
    idempotency_key: str
    spec_json: str
    spec_digest: str
    policy_id: str
    policy_digest: str
    grant_id: str
    state: AutomationRunState
    scenario_fingerprint: str
    target_fingerprints_json: str
    output_root_id: str
    budget_json: str
    usage_json: str
    created_at: str
    updated_at: str
    run_root: str | None = None
    manifest_path: str | None = None
    result_json: str | None = None
    error_json: str | None = None
    lease_id: str | None = None
    lease_heartbeat_at: str | None = None
    cancel_requested_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class RunBinding:
    """Result of idempotently binding one grant to one ready run."""

    run: AutomationRunRecord
    created: bool


def utc_now() -> str:
    """Return current UTC time in canonical second-precision form."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_policy(
    conn: sqlite3.Connection,
    policy: PolicyDocument,
    *,
    signature: str,
    key_id: str,
) -> str:
    """Insert one authenticated policy and return its canonical digest."""
    _validate_auth_fields(signature, key_id)
    payload = policy.to_payload()
    digest = sha256_digest(payload)
    try:
        conn.execute(
            """
            INSERT INTO automation_policies (
                id, policy_json, policy_digest, signature, key_id, status,
                created_at, expires_at, revoked_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, NULL)
            """,
            (
                policy.policy_id,
                canonical_json(payload),
                digest,
                signature,
                key_id,
                policy.created_at,
                policy.expires_at,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise AutomationStoreError(f"policy insert failed: {exc}") from exc
    return digest


def get_policy(conn: sqlite3.Connection, policy_id: str) -> StoredPolicy | None:
    """Return one stored policy row by full ID."""
    row = conn.execute(
        "SELECT * FROM automation_policies WHERE id = ?",
        (policy_id,),
    ).fetchone()
    if row is None:
        return None
    return StoredPolicy(
        policy_id=row["id"],
        policy_json=row["policy_json"],
        policy_digest=row["policy_digest"],
        signature=row["signature"],
        key_id=row["key_id"],
        status=row["status"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
    )


def revoke_policy(
    conn: sqlite3.Connection, policy_id: str, *, revoked_at: str | None = None
) -> bool:
    """Revoke an active policy without deleting its audit record."""
    result = conn.execute(
        """
        UPDATE automation_policies
        SET status = 'revoked', revoked_at = ?
        WHERE id = ? AND status = 'active'
        """,
        (revoked_at or utc_now(), policy_id),
    )
    return result.rowcount == 1


def save_grant(
    conn: sqlite3.Connection,
    grant: AuthorizationGrant,
    *,
    signature: str,
) -> str:
    """Insert one authenticated authorization grant and return its ID."""
    _validate_auth_fields(signature, grant.key_id)
    payload = grant.to_payload()
    target_json = canonical_json({"targets": [target.to_payload() for target in grant.targets]})
    try:
        conn.execute(
            """
            INSERT INTO automation_grants (
                id, grant_json, signature, key_id, spec_digest, policy_id,
                policy_digest, scenario_fingerprint, target_fingerprints,
                issued_at, expires_at, revoked_at, bound_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                grant.grant_id,
                canonical_json(payload),
                signature,
                grant.key_id,
                grant.spec_digest,
                grant.policy_id,
                grant.policy_digest,
                grant.scenario_fingerprint,
                target_json,
                grant.issued_at,
                grant.expires_at,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise AutomationStoreError(f"grant insert failed: {exc}") from exc
    return grant.grant_id


def get_grant(conn: sqlite3.Connection, grant_id: str) -> StoredGrant | None:
    """Return one stored authorization grant by full ID."""
    row = conn.execute(
        "SELECT * FROM automation_grants WHERE id = ?",
        (grant_id,),
    ).fetchone()
    if row is None:
        return None
    return StoredGrant(
        grant_id=row["id"],
        grant_json=row["grant_json"],
        signature=row["signature"],
        key_id=row["key_id"],
        spec_digest=row["spec_digest"],
        policy_id=row["policy_id"],
        policy_digest=row["policy_digest"],
        issued_at=row["issued_at"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
        bound_run_id=row["bound_run_id"],
    )


def revoke_grant(conn: sqlite3.Connection, grant_id: str, *, revoked_at: str | None = None) -> bool:
    """Revoke an unrevoked authorization grant without deleting it."""
    result = conn.execute(
        """
        UPDATE automation_grants
        SET revoked_at = ?
        WHERE id = ? AND revoked_at IS NULL
        """,
        (revoked_at or utc_now(), grant_id),
    )
    return result.rowcount == 1


def bind_grant_and_create_ready_run(
    conn: sqlite3.Connection,
    spec: RunSpec,
    grant: AuthorizationGrant,
    *,
    now: datetime.datetime | None = None,
) -> RunBinding:
    """Atomically bind a grant and create or return one idempotent ready run."""
    spec_payload = spec.to_payload()
    spec_digest = sha256_digest(spec_payload)
    evaluation_time = _evaluation_timestamp(now)
    conn.execute("SAVEPOINT automation_bind")
    try:
        binding = _bind_or_return_existing(
            conn,
            spec,
            spec_payload,
            spec_digest,
            grant,
            evaluation_time,
        )
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT automation_bind")
        conn.execute("RELEASE SAVEPOINT automation_bind")
        raise
    else:
        conn.execute("RELEASE SAVEPOINT automation_bind")
        return binding


def _bind_or_return_existing(
    conn: sqlite3.Connection,
    spec: RunSpec,
    spec_payload: dict[str, Any],
    spec_digest: str,
    grant: AuthorizationGrant,
    evaluation_time: str,
) -> RunBinding:
    existing = _get_run_by_idempotency(conn, spec.idempotency_key)
    if existing is not None:
        return _existing_binding(conn, existing, spec_digest, grant.grant_id)
    _validate_binding_contract(spec, grant, spec_digest)
    _validate_stored_grant(conn, grant, spec_digest, evaluation_time)
    run_id = uuid.uuid4().hex
    _claim_grant(conn, grant.grant_id, run_id)
    _insert_ready_run(conn, run_id, spec, spec_payload, spec_digest, grant)
    _append_event_unchecked(conn, run_id, "run_ready", {"spec_digest": spec_digest})
    run = get_automation_run(conn, run_id)
    if run is None:
        raise AutomationStoreError("new automation run could not be reloaded")
    return RunBinding(run, True)


def get_automation_run(
    conn: sqlite3.Connection,
    run_id: str,
) -> AutomationRunRecord | None:
    """Return one governed experiment control by full run ID."""
    row = conn.execute(
        "SELECT * FROM automation_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    return _run_from_row(row) if row is not None else None


def append_event(
    conn: sqlite3.Connection,
    run_id: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    created_at: str | None = None,
) -> int:
    """Append one canonical lifecycle event and return its sequence."""
    if not event_type or len(event_type) > 64:
        raise AutomationStoreError("event_type must contain 1-64 characters")
    return _append_event_unchecked(conn, run_id, event_type, payload, created_at=created_at)


def list_events(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Return lifecycle events ordered by their append-only sequence."""
    rows = conn.execute(
        """
        SELECT sequence, event_type, payload_json, created_at
        FROM automation_events WHERE run_id = ? ORDER BY sequence
        """,
        (run_id,),
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload: Any = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            payload = {"malformed": True}
        events.append(
            {
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "payload": payload if isinstance(payload, dict) else {"malformed": True},
                "created_at": row["created_at"],
            }
        )
    return events


def transition_run_state(
    conn: sqlite3.Connection,
    run_id: str,
    expected: AutomationRunState,
    target: AutomationRunState,
    *,
    event_payload: dict[str, Any] | None = None,
) -> AutomationRunRecord:
    """Compare-and-set one valid lifecycle transition and append its event."""
    if target not in _TRANSITIONS.get(expected, set()):
        raise AutomationStoreError(f"invalid automation transition: {expected} -> {target}")
    now = utc_now()
    conn.execute("SAVEPOINT automation_transition")
    try:
        run = _transition_and_record(conn, run_id, expected, target, event_payload, now)
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT automation_transition")
        conn.execute("RELEASE SAVEPOINT automation_transition")
        raise
    else:
        conn.execute("RELEASE SAVEPOINT automation_transition")
        return run


def _transition_and_record(
    conn: sqlite3.Connection,
    run_id: str,
    expected: AutomationRunState,
    target: AutomationRunState,
    event_payload: dict[str, Any] | None,
    now: str,
) -> AutomationRunRecord:
    started_at = now if target == AutomationRunState.RUNNING else None
    cancel_requested_at = now if target == AutomationRunState.CANCEL_REQUESTED else None
    finished_at = now if target in _TERMINAL_STATES else None
    result = conn.execute(
        """
        UPDATE automation_runs SET
            state = ?, updated_at = ?,
            started_at = COALESCE(started_at, ?),
            cancel_requested_at = COALESCE(cancel_requested_at, ?),
            finished_at = COALESCE(finished_at, ?)
        WHERE id = ? AND state = ?
        """,
        (
            target.value,
            now,
            started_at,
            cancel_requested_at,
            finished_at,
            run_id,
            expected.value,
        ),
    )
    if result.rowcount != 1:
        raise AutomationStoreError("automation run state changed or run was not found")
    _append_event_unchecked(
        conn,
        run_id,
        f"state_{target.value.lower()}",
        event_payload or {},
        created_at=now,
    )
    run = get_automation_run(conn, run_id)
    if run is None:
        raise AutomationStoreError("transitioned automation run could not be reloaded")
    return run


def _insert_ready_run(
    conn: sqlite3.Connection,
    run_id: str,
    spec: RunSpec,
    spec_payload: dict[str, Any],
    spec_digest: str,
    grant: AuthorizationGrant,
) -> None:
    now = utc_now()
    target_json = canonical_json(
        {"targets": [target.to_payload() for target in spec.experiment.targets]}
    )
    usage_json = canonical_json(
        {
            "cost_microusd": 0,
            "output_tokens_reserved": 0,
            "provider_requests": 0,
            "runtime_processes": 0,
            "tool_calls": 0,
        }
    )
    conn.execute(
        """
        INSERT INTO automation_runs (
            id, idempotency_key, spec_json, spec_digest, policy_id,
            policy_digest, grant_id, state, scenario_fingerprint,
            target_fingerprints, output_root_id, budget_json, usage_json,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            spec.idempotency_key,
            canonical_json(spec_payload),
            spec_digest,
            spec.policy_id,
            grant.policy_digest,
            grant.grant_id,
            AutomationRunState.READY.value,
            spec.experiment.scenario_fingerprint,
            target_json,
            spec.output_root_id,
            canonical_json(spec.limits.to_payload()),
            usage_json,
            now,
            now,
        ),
    )


def _validate_stored_grant(
    conn: sqlite3.Connection,
    grant: AuthorizationGrant,
    spec_digest: str,
    evaluation_time: str,
) -> None:
    row = conn.execute(
        """
        SELECT
            g.grant_json, g.spec_digest, g.policy_id, g.policy_digest,
            g.issued_at, g.expires_at, g.revoked_at, g.bound_run_id,
            p.policy_digest AS stored_policy_digest,
            p.status AS policy_status,
            p.created_at AS policy_created_at,
            p.expires_at AS policy_expires_at
        FROM automation_grants AS g
        JOIN automation_policies AS p ON p.id = g.policy_id
        WHERE g.id = ?
        """,
        (grant.grant_id,),
    ).fetchone()
    if row is None:
        raise AutomationStoreError("authorization grant is not stored")
    if row["revoked_at"] is not None:
        raise AutomationStoreError("authorization grant is revoked")
    if row["policy_status"] != "active":
        raise AutomationStoreError("authorizing policy is revoked")
    if not row["issued_at"] <= evaluation_time < row["expires_at"]:
        raise AutomationStoreError("authorization grant is outside its validity interval")
    if not row["policy_created_at"] <= evaluation_time < row["policy_expires_at"]:
        raise AutomationStoreError("authorizing policy is outside its validity interval")
    if row["bound_run_id"] is not None:
        raise GrantReplayError("authorization grant is already bound")
    expected = (spec_digest, grant.policy_id, grant.policy_digest)
    actual = (row["spec_digest"], row["policy_id"], row["policy_digest"])
    if (
        actual != expected
        or row["stored_policy_digest"] != grant.policy_digest
        or row["grant_json"] != canonical_json(grant.to_payload())
    ):
        raise AutomationStoreError("authorization grant does not match the RunSpec")


def _validate_binding_contract(
    spec: RunSpec,
    grant: AuthorizationGrant,
    spec_digest: str,
) -> None:
    if grant.spec_digest != spec_digest or grant.policy_id != spec.policy_id:
        raise AutomationStoreError("authorization grant does not identify this RunSpec")
    if grant.scenario_fingerprint != spec.experiment.scenario_fingerprint:
        raise AutomationStoreError("authorization grant scenario fingerprint does not match")
    if grant.targets != spec.experiment.targets:
        raise AutomationStoreError("authorization grant target identities do not match")
    if grant.authorized_tier != spec.requested_tier or grant.limits != spec.limits:
        raise AutomationStoreError("authorization grant authority does not match the RunSpec")


def _claim_grant(conn: sqlite3.Connection, grant_id: str, run_id: str) -> None:
    result = conn.execute(
        """
        UPDATE automation_grants SET bound_run_id = ?
        WHERE id = ? AND bound_run_id IS NULL AND revoked_at IS NULL
        """,
        (run_id, grant_id),
    )
    if result.rowcount != 1:
        raise GrantReplayError("authorization grant could not be bound")


def _get_run_by_idempotency(
    conn: sqlite3.Connection, idempotency_key: str
) -> AutomationRunRecord | None:
    row = conn.execute(
        "SELECT * FROM automation_runs WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    return _run_from_row(row) if row is not None else None


def _existing_binding(
    conn: sqlite3.Connection,
    run: AutomationRunRecord,
    spec_digest: str,
    grant_id: str,
) -> RunBinding:
    if run.spec_digest != spec_digest or run.grant_id != grant_id:
        raise IdempotencyConflictError(
            "idempotency key is already bound to a different spec or grant"
        )
    stored = conn.execute(
        "SELECT bound_run_id FROM automation_grants WHERE id = ?",
        (grant_id,),
    ).fetchone()
    if stored is None or stored["bound_run_id"] != run.run_id:
        raise GrantReplayError("stored grant binding does not match the idempotent run")
    return RunBinding(run, False)


def _append_event_unchecked(
    conn: sqlite3.Connection,
    run_id: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    created_at: str | None = None,
) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM automation_events WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    sequence = int(row["sequence"]) + 1
    try:
        conn.execute(
            """
            INSERT INTO automation_events (run_id, sequence, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, sequence, event_type, canonical_json(payload), created_at or utc_now()),
        )
    except sqlite3.IntegrityError as exc:
        raise AutomationStoreError(f"automation event append failed: {exc}") from exc
    return sequence


def _run_from_row(row: sqlite3.Row) -> AutomationRunRecord:
    return AutomationRunRecord(
        run_id=row["id"],
        idempotency_key=row["idempotency_key"],
        spec_json=row["spec_json"],
        spec_digest=row["spec_digest"],
        policy_id=row["policy_id"],
        policy_digest=row["policy_digest"],
        grant_id=row["grant_id"],
        state=AutomationRunState(row["state"]),
        scenario_fingerprint=row["scenario_fingerprint"],
        target_fingerprints_json=row["target_fingerprints"],
        output_root_id=row["output_root_id"],
        budget_json=row["budget_json"],
        usage_json=row["usage_json"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        run_root=row["run_root"],
        manifest_path=row["manifest_path"],
        result_json=row["result_json"],
        error_json=row["error_json"],
        lease_id=row["lease_id"],
        lease_heartbeat_at=row["lease_heartbeat_at"],
        cancel_requested_at=row["cancel_requested_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _validate_auth_fields(signature: str, key_id: str) -> None:
    if not _DIGEST.fullmatch(signature):
        raise AutomationStoreError("signature must be a lowercase HMAC-SHA-256 digest")
    if not _DIGEST.fullmatch(key_id):
        raise AutomationStoreError("key_id must be a lowercase SHA-256 digest")


def _evaluation_timestamp(value: datetime.datetime | None) -> str:
    current = value or datetime.datetime.now(datetime.UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise AutomationStoreError("grant evaluation time must be timezone-aware")
    return current.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
