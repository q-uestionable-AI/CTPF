"""Bounded machine envelopes and stable exit codes for automation controls."""

from __future__ import annotations

from typing import Any

from ctpf.automation.canonical import canonical_json

ENVELOPE_SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_INPUT = 2
EXIT_AUTHORIZATION = 3
EXIT_STATE = 4
EXIT_EXECUTION = 5
EXIT_VERIFICATION = 6
EXIT_INTERNAL = 7

_INPUT_CODES = {
    "canonicalization_failed",
    "input_too_large",
    "invalid_field",
    "invalid_json",
    "schema_version_unsupported",
    "unknown_field",
}
_AUTHORIZATION_CODES = {
    "approval_expired",
    "approval_invalid",
    "approval_required",
    "approval_revoked",
    "approval_replayed",
    "approval_spec_mismatch",
    "confirmation_mismatch",
    "policy_denied",
    "policy_expired",
    "policy_invalid",
    "policy_revoked",
    "tty_required",
}
_STATE_CODES = {
    "approval_conflict",
    "database_unavailable",
    "execution_unavailable",
    "idempotency_conflict",
    "policy_conflict",
    "policy_not_found",
    "result_unavailable",
    "run_not_found",
    "run_state_conflict",
}
_EXECUTION_CODES = {"cancelled", "deadline_exceeded", "execution_failed", "interrupted"}
_VERIFICATION_CODES = {
    "artifact_missing",
    "artifact_path_invalid",
    "evidence_missing",
    "hash_mismatch",
    "manifest_invalid",
    "result_manifest_mismatch",
    "verification_unavailable",
}


class ControlError(RuntimeError):
    """Stable secret-safe failure returned by an automation operation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


def success_envelope(
    operation: str,
    data: dict[str, Any],
    *,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return one successful versioned machine envelope."""
    return {
        "data": data,
        "ok": True,
        "operation": operation,
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "warnings": list(warnings),
    }


def failure_envelope(
    operation: str,
    error: ControlError,
    *,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return one failed versioned machine envelope."""
    return {
        "error": {
            "code": error.code,
            "details": dict(error.details),
            "message": error.message,
        },
        "ok": False,
        "operation": operation,
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "warnings": list(warnings),
    }


def render_envelope(payload: dict[str, Any]) -> str:
    """Serialize one bounded envelope as canonical JSON."""
    return canonical_json(payload)


def exit_code_for(error: ControlError) -> int:
    """Return the stable process exit code for one control failure."""
    if error.code in _INPUT_CODES:
        return EXIT_INPUT
    if error.code in _AUTHORIZATION_CODES:
        return EXIT_AUTHORIZATION
    if error.code in _STATE_CODES:
        return EXIT_STATE
    if error.code in _EXECUTION_CODES:
        return EXIT_EXECUTION
    if error.code in _VERIFICATION_CODES:
        return EXIT_VERIFICATION
    return EXIT_INTERNAL


def internal_error() -> ControlError:
    """Return the generic secret-safe unexpected-failure representation."""
    return ControlError("internal_error", "unexpected internal automation failure")
