"""Owned execution controls for governed packaged experiments."""

from __future__ import annotations

import asyncio
import datetime
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from ctpf.automation.approval import (
    ApprovalError,
    authenticate_authorization_grant,
    authenticate_policy,
)
from ctpf.automation.canonical import CanonicalizationError, load_canonical_object, sha256_digest
from ctpf.automation.contracts import (
    AuthorizationGrant,
    AutomationRunState,
    ExperimentMode,
    PolicyDocument,
    RunSpec,
    TargetPolicy,
)
from ctpf.automation.store import (
    AutomationRunRecord,
    AutomationStoreError,
    append_event,
    finish_run_execution,
    get_automation_run,
    get_grant,
    get_policy,
    heartbeat_run,
    list_events,
    record_provider_usage,
    reserve_run_budget,
)
from ctpf.automation.targets import ScenarioCapability, scenario_capability
from ctpf.core.db import database_path, get_connection, get_readonly_connection

HEARTBEAT_INTERVAL_SECONDS = 5
MISSED_HEARTBEAT_LIMIT = 3
CONTROL_POLL_SECONDS = 0.25
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
_T = TypeVar("_T")
_RUNNING_STATES = {AutomationRunState.RUNNING, AutomationRunState.CANCEL_REQUESTED}
_SINGLE_SESSION_SCENARIOS = {"pattern2", "pattern3-scope"}
_SESSION_FIELDS = {
    "condition",
    "fixture_run_id",
    "inference_path",
    "listen_port",
    "mutation_path",
    "reset",
    "scenario",
    "session_name",
    "target_id",
    "trace_path",
    "work_id",
}


class ExecutionControlError(RuntimeError):
    """Base class for governed execution failures."""


class ExecutionCancelledError(ExecutionControlError):
    """Raised after durable cancellation is observed."""


class ExecutionDeadlineExceededError(ExecutionControlError):
    """Raised when the signed wall-clock deadline expires."""


class ExecutionInterruptedError(ExecutionControlError):
    """Raised when the worker no longer owns a live lease."""


class BudgetExhaustedError(ExecutionControlError):
    """Raised before an effect whose reservation exceeds authority."""


@dataclass(frozen=True)
class SessionWork:
    """One internally prepared isolated session read from durable state."""

    work_id: str
    scenario: str
    condition: str
    session_name: str
    target_id: str
    fixture_run_id: str
    trace_path: str
    inference_path: str | None
    mutation_path: str | None
    reset: bool
    listen_port: int

    def to_payload(self) -> dict[str, Any]:
        """Return the strict durable work representation."""
        return {
            "condition": self.condition,
            "fixture_run_id": self.fixture_run_id,
            "inference_path": self.inference_path,
            "listen_port": self.listen_port,
            "mutation_path": self.mutation_path,
            "reset": self.reset,
            "scenario": self.scenario,
            "session_name": self.session_name,
            "target_id": self.target_id,
            "trace_path": self.trace_path,
            "work_id": self.work_id,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> SessionWork:
        """Parse one untrusted stored session-work object."""
        if not isinstance(raw, dict) or set(raw) != _SESSION_FIELDS:
            raise ExecutionInterruptedError("stored session work has an invalid shape")
        return cls(
            work_id=_hex_id(raw["work_id"], "work_id"),
            scenario=_text(raw["scenario"], "scenario"),
            condition=_text(raw["condition"], "condition"),
            session_name=_text(raw["session_name"], "session_name"),
            target_id=_hex_id(raw["target_id"], "target_id"),
            fixture_run_id=_text(raw["fixture_run_id"], "fixture_run_id"),
            trace_path=_relative_path(raw["trace_path"], "trace_path"),
            inference_path=_optional_relative_path(raw["inference_path"], "inference_path"),
            mutation_path=_optional_relative_path(raw["mutation_path"], "mutation_path"),
            reset=_boolean(raw["reset"], "reset"),
            listen_port=_port(raw["listen_port"]),
        )


class ExecutionControl:
    """Lease-bound cancellation, deadline, budget, and session authority."""

    def __init__(
        self,
        *,
        db_path: Path,
        run: AutomationRunRecord,
        spec: RunSpec,
        policy: PolicyDocument,
        capability: ScenarioCapability,
        deadline_at: datetime.datetime,
    ) -> None:
        """Bind one controller to an already-claimed run lease."""
        if run.lease_id is None or run.run_root is None:
            raise ExecutionInterruptedError("claimed run has no lease or run root")
        self.db_path = database_path(db_path)
        self.run_id = run.run_id
        self.lease_id = run.lease_id
        self.run_root = Path(run.run_root)
        self.spec = spec
        self.policy = policy
        self.capability = capability
        self.deadline_at = deadline_at
        self._heartbeat_failed = False

    @classmethod
    def attach(cls, run_id: str, db_path: Path) -> ExecutionControl:
        """Authenticate and attach an isolated child to one live run."""
        path = database_path(db_path)
        with get_readonly_connection(path) as conn:
            run = get_automation_run(conn, _hex_id(run_id, "run_id"))
            if run is None or run.state not in _RUNNING_STATES:
                raise ExecutionInterruptedError("automation run is not active")
            spec, policy = _authenticated_run_authority(conn, run)
            capability = scenario_capability(spec.experiment.scenario)
            events = list_events(conn, run.run_id)
        _require_revalidated_targets(spec, events)
        deadline = _deadline_from_events(events)
        return cls(
            db_path=path,
            run=run,
            spec=spec,
            policy=policy,
            capability=capability,
            deadline_at=deadline,
        )

    @property
    def expected_tool_names(self) -> frozenset[str]:
        """Return the exact installed tool allowlist for this scenario."""
        return frozenset(self.capability.tool_names)

    def target_policy(self, target_id: str) -> TargetPolicy:
        """Return the signed exact target snapshot selected by the RunSpec."""
        selected = next((item for item in self.policy.targets if item.target_id == target_id), None)
        if selected is None or not any(
            reference.target_id == target_id for reference in self.spec.experiment.targets
        ):
            raise ExecutionInterruptedError("session target is outside the RunSpec")
        return selected

    def checkpoint(self, boundary: str) -> None:
        """Fail before a boundary on cancellation, deadline, or lost ownership."""
        del boundary
        if self._heartbeat_failed:
            raise ExecutionInterruptedError("automation lease heartbeat failed")
        now = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        if now >= self.deadline_at:
            raise ExecutionDeadlineExceededError("governed execution deadline exceeded")
        try:
            with get_readonly_connection(self.db_path) as conn:
                run = get_automation_run(conn, self.run_id)
        except (FileNotFoundError, OSError) as exc:
            raise ExecutionInterruptedError("automation lifecycle store is unavailable") from exc
        self._validate_live_run(run, now)

    def reserve(self, boundary: str, **reservation: int) -> dict[str, Any]:
        """Consume an exact reservation before crossing an effect boundary."""
        self.checkpoint(boundary)
        try:
            with get_connection(self.db_path) as conn:
                return reserve_run_budget(
                    conn,
                    self.run_id,
                    self.lease_id,
                    reservation,
                    boundary=boundary,
                )
        except AutomationStoreError as exc:
            self._raise_store_failure(exc)
        raise ExecutionInterruptedError("budget reservation failed")

    def record_provider_usage(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> dict[str, Any]:
        """Record validated provider usage without treating it as authority."""
        reported = {
            key: value
            for key, value in {
                "input_tokens_reported": input_tokens,
                "output_tokens_reported": output_tokens,
                "total_tokens_reported": total_tokens,
            }.items()
            if value is not None
        }
        if not reported:
            usage = self.provenance_payload().get("usage")
            if not isinstance(usage, dict):
                raise ExecutionInterruptedError("stored provider usage is malformed")
            return usage
        self.checkpoint("provider_usage")
        try:
            with get_connection(self.db_path) as conn:
                return record_provider_usage(conn, self.run_id, self.lease_id, reported)
        except AutomationStoreError as exc:
            self._raise_store_failure(exc)
        raise ExecutionInterruptedError("provider usage recording failed")

    async def wait(self, awaitable: Awaitable[_T], boundary: str) -> _T:
        """Race one async effect against durable cancellation and deadline."""
        task = asyncio.ensure_future(awaitable)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=CONTROL_POLL_SECONDS)
                if task in done:
                    return task.result()
                self.checkpoint(boundary)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def heartbeat_loop(self, stop: asyncio.Event) -> None:
        """Refresh the parent lease until execution reaches a terminal path."""
        while not stop.is_set():
            try:
                with get_connection(self.db_path) as conn:
                    if not heartbeat_run(conn, self.run_id, self.lease_id):
                        self._heartbeat_failed = True
                        return
            except Exception:
                self._heartbeat_failed = True
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL_SECONDS)
            except TimeoutError:
                continue

    def record_revalidated_targets(self, fingerprints: dict[str, str]) -> None:
        """Record complete live target revalidation before output creation."""
        expected = {
            reference.target_id: reference.target_fingerprint
            for reference in self.spec.experiment.targets
        }
        if fingerprints != expected:
            raise ExecutionInterruptedError("live target identity differs from approved identity")
        self._append("targets_revalidated", {"fingerprints": fingerprints})

    def prepare_session(self, work: SessionWork) -> None:
        """Persist one exact internal work item before spawning its child."""
        self.checkpoint("session")
        _validate_session_work(self, work)
        with get_connection(self.db_path) as conn:
            events = list_events(conn, self.run_id)
            if _pending_work(events) is not None:
                raise ExecutionInterruptedError("another session work item is already pending")
            append_event(conn, self.run_id, "session_prepared", work.to_payload())

    def claim_session(self) -> SessionWork:
        """Atomically claim the sole prepared work item in an isolated child."""
        self.checkpoint("session_claim")
        with get_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            events = list_events(conn, self.run_id)
            payload = _pending_work(events)
            if payload is None:
                raise ExecutionInterruptedError("no prepared session work is available")
            work = SessionWork.from_payload(payload)
            _validate_session_work(self, work)
            append_event(conn, self.run_id, "session_claimed", {"work_id": work.work_id})
        return work

    def finish_session(self, work_id: str, status: str) -> None:
        """Append the isolated child outcome without erasing partial artifacts."""
        if status not in {"complete", "failed", "cancelled"}:
            raise ValueError("session status is unsupported")
        self._append(
            "session_finished",
            {"status": status, "work_id": _hex_id(work_id, "work_id")},
        )

    def finish(
        self,
        state: AutomationRunState,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        manifest_path: Path | None = None,
    ) -> AutomationRunRecord:
        """Persist one terminal execution outcome owned by this lease."""
        with get_connection(self.db_path) as conn:
            return finish_run_execution(
                conn,
                self.run_id,
                self.lease_id,
                state,
                result_payload=result,
                error_payload=error,
                manifest_path=str(manifest_path) if manifest_path is not None else None,
            )

    def provenance_payload(self) -> dict[str, Any]:
        """Return bounded lifecycle provenance for governed experiment export."""
        with get_readonly_connection(self.db_path) as conn:
            run = get_automation_run(conn, self.run_id)
            if run is None:
                raise ExecutionInterruptedError("automation run disappeared")
            events = list_events(conn, self.run_id)
            stored_grant = get_grant(conn, run.grant_id)
        if stored_grant is None:
            raise ExecutionInterruptedError("automation grant disappeared")
        try:
            grant_payload = load_canonical_object(stored_grant.grant_json)
        except CanonicalizationError as exc:
            raise ExecutionInterruptedError("automation grant payload is malformed") from exc
        if not isinstance(grant_payload, dict):
            raise ExecutionInterruptedError("automation grant payload is malformed")
        return {
            "budget": _stored_object(run.budget_json, "budget"),
            "events": events,
            "grant_digest": sha256_digest(grant_payload),
            "grant_id": run.grant_id,
            "policy_digest": run.policy_digest,
            "run_id": run.run_id,
            "scenario_fingerprint": run.scenario_fingerprint,
            "schema_version": 1,
            "spec_digest": run.spec_digest,
            "state": run.state.value,
            "timestamps": {
                "created_at": run.created_at,
                "started_at": run.started_at,
                "updated_at": run.updated_at,
            },
            "usage": _stored_object(run.usage_json, "usage"),
        }

    def path(self, relative: str) -> Path:
        """Resolve one validated run-relative artifact path."""
        selected = self.run_root.joinpath(*PurePosixPath(_relative_path(relative, "path")).parts)
        resolved = selected.resolve(strict=False)
        if not resolved.is_relative_to(self.run_root.resolve(strict=False)):
            raise ExecutionInterruptedError("session artifact path escapes the governed run root")
        return resolved

    def _append(self, event_type: str, payload: dict[str, Any]) -> None:
        with get_connection(self.db_path) as conn:
            append_event(conn, self.run_id, event_type, payload)

    def _validate_live_run(
        self,
        run: AutomationRunRecord | None,
        now: datetime.datetime,
    ) -> None:
        if run is None or run.lease_id != self.lease_id:
            raise ExecutionInterruptedError("automation lease ownership was lost")
        if run.state == AutomationRunState.CANCEL_REQUESTED:
            raise ExecutionCancelledError("governed execution cancellation was requested")
        if run.state != AutomationRunState.RUNNING or run.lease_heartbeat_at is None:
            raise ExecutionInterruptedError("automation run is no longer running")
        heartbeat = _timestamp(run.lease_heartbeat_at, "lease heartbeat")
        stale_after = HEARTBEAT_INTERVAL_SECONDS * MISSED_HEARTBEAT_LIMIT
        if (now - heartbeat).total_seconds() >= stale_after:
            raise ExecutionInterruptedError("automation lease heartbeat expired")

    def _raise_store_failure(self, exc: AutomationStoreError) -> None:
        message = str(exc)
        if "budget" in message and ("exhausted" in message or "measurable" in message):
            raise BudgetExhaustedError(message) from exc
        self.checkpoint("reservation_failure")
        raise ExecutionInterruptedError("automation reservation failed") from exc


def new_lease_id() -> str:
    """Return a random unguessable execution lease identifier."""
    return uuid.uuid4().hex


def deadline_at(now: datetime.datetime, seconds: int) -> datetime.datetime:
    """Return a second-precision UTC deadline for one signed wall-clock budget."""
    return now.astimezone(datetime.UTC).replace(microsecond=0) + datetime.timedelta(seconds=seconds)


def stale_before(now: datetime.datetime) -> datetime.datetime:
    """Return the heartbeat cutoff for three missed intervals."""
    seconds = HEARTBEAT_INTERVAL_SECONDS * MISSED_HEARTBEAT_LIMIT
    return now.astimezone(datetime.UTC).replace(microsecond=0) - datetime.timedelta(seconds=seconds)


def format_timestamp(value: datetime.datetime) -> str:
    """Format one aware datetime as canonical UTC text."""
    return value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _authenticated_run_authority(
    conn: Any,
    run: AutomationRunRecord,
) -> tuple[RunSpec, PolicyDocument]:
    stored_policy = get_policy(conn, run.policy_id)
    stored_grant = get_grant(conn, run.grant_id)
    if stored_policy is None or stored_grant is None or run.started_at is None:
        raise ExecutionInterruptedError("automation authority records are incomplete")
    try:
        spec = RunSpec.from_payload(load_canonical_object(run.spec_json))
        policy = PolicyDocument.from_payload(load_canonical_object(stored_policy.policy_json))
        grant = AuthorizationGrant.from_payload(load_canonical_object(stored_grant.grant_json))
        started = _timestamp(run.started_at, "run start")
        digest = authenticate_policy(
            policy, stored_policy.signature, stored_policy.key_id, now=started
        )
        authenticate_authorization_grant(
            grant,
            stored_grant.signature,
            spec,
            policy,
            now=started,
        )
    except (ApprovalError, CanonicalizationError, ValueError) as exc:
        raise ExecutionInterruptedError("automation authority authentication failed") from exc
    if digest != run.policy_digest or grant.grant_id != run.grant_id:
        raise ExecutionInterruptedError("automation authority metadata differs from the run")
    return spec, policy


def _require_revalidated_targets(spec: RunSpec, events: list[dict[str, Any]]) -> None:
    expected = {
        reference.target_id: reference.target_fingerprint for reference in spec.experiment.targets
    }
    payload = next(
        (
            event["payload"]
            for event in reversed(events)
            if event.get("event_type") == "targets_revalidated"
        ),
        None,
    )
    if not isinstance(payload, dict) or payload.get("fingerprints") != expected:
        raise ExecutionInterruptedError("live target revalidation was not recorded")


def _deadline_from_events(events: list[dict[str, Any]]) -> datetime.datetime:
    for event in events:
        if event.get("event_type") != "state_running":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            return _timestamp(payload.get("deadline_at"), "deadline")
    raise ExecutionInterruptedError("automation deadline is missing")


def _validate_session_work(control: ExecutionControl, work: SessionWork) -> None:
    if work.scenario != control.spec.experiment.scenario:
        raise ExecutionInterruptedError("session scenario differs from the RunSpec")
    control.target_policy(work.target_id)
    if work.condition not in control.capability.conditions:
        raise ExecutionInterruptedError("session condition is not installed")
    expected_names = {"single"} if work.scenario in _SINGLE_SESSION_SCENARIOS else {"A", "B"}
    if work.session_name not in expected_names:
        raise ExecutionInterruptedError("session name is not installed")
    if work.listen_port != control.policy.limits.loopback_port:
        raise ExecutionInterruptedError("session loopback port differs from signed policy")
    expected = _expected_session_fields(control, work)
    actual = (
        work.trace_path,
        work.inference_path,
        work.mutation_path,
        work.reset,
        work.fixture_run_id,
    )
    if actual != expected:
        raise ExecutionInterruptedError("session artifacts differ from installed work")
    for relative in (work.trace_path, work.inference_path, work.mutation_path):
        if relative is not None:
            control.path(relative)


def _expected_session_fields(
    control: ExecutionControl,
    work: SessionWork,
) -> tuple[str, str, str | None, bool, str]:
    prefix, series_id = _session_prefix_and_series(control, work)
    mutation = f"{prefix}/mutation.json" if work.condition != "baseline" else None
    if work.scenario in _SINGLE_SESSION_SCENARIOS:
        if work.scenario != "pattern2":
            mutation = None
        return (
            f"{prefix}/session.json",
            f"{prefix}/session.inference.json",
            mutation,
            True,
            f"{series_id}-{work.condition}",
        )
    label = work.session_name
    return (
        f"{prefix}/session-{label}.json",
        f"{prefix}/session-{label}.inference.json",
        mutation if label == "A" else None,
        label == "A",
        f"{series_id}-{work.condition}",
    )


def _session_prefix_and_series(control: ExecutionControl, work: SessionWork) -> tuple[str, str]:
    if work.scenario in _SINGLE_SESSION_SCENARIOS:
        return work.condition, control.run_id
    parts = PurePosixPath(work.trace_path).parts
    matrix = control.spec.experiment.mode == ExperimentMode.MATRIX
    if matrix:
        if len(parts) == 4 and parts[0] == "trials" and parts[2] == work.condition:
            series_id = _matrix_series_id(parts[1])
            prefix = PurePosixPath(parts[0], series_id, parts[2]).as_posix()
            return prefix, series_id
        raise ExecutionInterruptedError("session artifacts differ from installed matrix work")
    if len(parts) == 2 and parts[0] == work.condition:
        return work.condition, control.run_id
    raise ExecutionInterruptedError("session artifacts differ from installed work")


def _matrix_series_id(raw: str) -> str:
    value = _text(raw, "matrix trial series ID")
    if ":" in value:
        raise ExecutionInterruptedError("matrix trial series ID contains unsupported characters")
    return value


def _pending_work(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    claimed = {
        event.get("payload", {}).get("work_id")
        for event in events
        if event.get("event_type") == "session_claimed" and isinstance(event.get("payload"), dict)
    }
    pending = [
        event.get("payload")
        for event in events
        if event.get("event_type") == "session_prepared"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("work_id") not in claimed
    ]
    if len(pending) > 1:
        raise ExecutionInterruptedError("multiple unclaimed session work items exist")
    return pending[0] if pending else None


def _stored_object(raw: str, label: str) -> dict[str, Any]:
    try:
        return load_canonical_object(raw)
    except CanonicalizationError as exc:
        raise ExecutionInterruptedError(f"stored {label} is malformed") from exc


def _relative_path(raw: Any, label: str) -> str:
    value = _text(raw, label)
    if "\\" in value or value.startswith("/") or "//" in value:
        raise ExecutionInterruptedError(f"{label} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ExecutionInterruptedError(f"{label} must not contain traversal components")
    return path.as_posix()


def _optional_relative_path(raw: Any, label: str) -> str | None:
    return None if raw is None else _relative_path(raw, label)


def _timestamp(raw: Any, label: str) -> datetime.datetime:
    if not isinstance(raw, str):
        raise ExecutionInterruptedError(f"{label} is not timestamp text")
    try:
        return datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
    except ValueError as exc:
        raise ExecutionInterruptedError(f"{label} is malformed") from exc


def _text(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or len(raw) > 256:
        raise ExecutionInterruptedError(f"{label} must be normalized bounded text")
    return raw


def _hex_id(raw: Any, label: str) -> str:
    value = _text(raw, label)
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise ExecutionInterruptedError(f"{label} must be a full lowercase hexadecimal ID")
    return value


def _boolean(raw: Any, label: str) -> bool:
    if not isinstance(raw, bool):
        raise ExecutionInterruptedError(f"{label} must be a boolean")
    return raw


def _port(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 65_535:
        raise ExecutionInterruptedError("listen_port must be an integer between 1 and 65535")
    return int(raw)
