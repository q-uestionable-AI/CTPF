"""Machine control and TTY-only governance commands for CTPF automation."""

from __future__ import annotations

import asyncio
import hmac
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer

from ctpf.automation.approval import (
    ApprovalError,
    approval_key_id,
    initialize_approval_key,
    rotate_approval_key,
)
from ctpf.automation.canonical import (
    MAX_INPUT_BYTES,
    CanonicalizationError,
    canonical_json,
    load_canonical_object,
    sha256_digest,
)
from ctpf.automation.contracts import (
    ContractError,
    DecisionKind,
    PolicyDocument,
    RunSpec,
)
from ctpf.automation.envelope import (
    ControlError,
    exit_code_for,
    failure_envelope,
    internal_error,
    render_envelope,
    success_envelope,
)
from ctpf.automation.service import AutomationService

control_app = typer.Typer(no_args_is_help=True, help="Operate the governed machine lifecycle.")
govern_app = typer.Typer(no_args_is_help=True, help="Manage human-controlled automation authority.")
key_app = typer.Typer(no_args_is_help=True, help="Manage the local automation signing key.")
policy_app = typer.Typer(no_args_is_help=True, help="Create, inspect, and revoke signed policies.")
approval_app = typer.Typer(no_args_is_help=True, help="Create, inspect, and revoke run approvals.")

govern_app.add_typer(key_app, name="key")
govern_app.add_typer(policy_app, name="policy")
govern_app.add_typer(approval_app, name="approval")

_T = TypeVar("_T")
_MachineResult = tuple[dict[str, Any], tuple[str, ...]]


@control_app.command("capabilities")
def control_capabilities(
    policy: str | None = typer.Option(None, help="Optional full signed policy ID."),
) -> None:
    """Return installed and optionally policy-authorized capabilities."""
    _run_machine("capabilities", lambda: (_service().capabilities(policy), ()))


@control_app.command("validate")
def control_validate() -> None:
    """Validate and policy-evaluate one RunSpec JSON object from stdin."""
    _run_machine("validate", _validate_from_stdin)


@control_app.command("start")
def control_start(
    approval: str | None = typer.Option(None, help="Full human approval ID when required."),
) -> None:
    """Create or return one idempotent READY run without executing it."""

    def operation() -> _MachineResult:
        spec = _stdin_contract(RunSpec.from_payload)
        return _service().start(spec, approval_id=approval), ()

    _run_machine("start", operation)


@control_app.command("execute")
def control_execute(run_id: str) -> None:
    """Claim and foreground-run one exact authorized READY control."""

    def operation() -> _MachineResult:
        selected = _full_id(run_id, "run_id")
        return asyncio.run(_service().execute(selected)), ()

    _run_machine("execute", operation)


@control_app.command("status")
def control_status(run_id: str) -> None:
    """Return one governed run's lifecycle state."""
    _run_machine("status", lambda: (_service().status(run_id), ()))


@control_app.command("cancel")
def control_cancel(run_id: str) -> None:
    """Cancel a READY run or request cancellation of a running worker."""
    _run_machine("cancel", lambda: (_service().cancel(run_id), ()))


@control_app.command("result")
def control_result(run_id: str) -> None:
    """Return one available terminal mechanical record."""
    _run_machine("result", lambda: (_service().result(run_id), ()))


@control_app.command("verify")
def control_verify(run_id: str) -> None:
    """Verify the governed run's declared evidence bundle for internal consistency."""

    def operation() -> _MachineResult:
        selected = _full_id(run_id, "run_id")
        return _service().verify(selected), ()

    _run_machine("verify", operation)


@key_app.command("status")
def govern_key_status() -> None:
    """Show whether the local automation signing key is initialized."""

    def operation() -> _MachineResult:
        try:
            key_id = approval_key_id()
        except ApprovalError:
            return {"initialized": False, "key_id": None}, ()
        return {"initialized": True, "key_id": key_id}, ()

    _run_machine("govern.key.status", operation)


@key_app.command("initialize")
def govern_key_initialize() -> None:
    """Initialize the local signing key after typed TTY confirmation."""

    def operation() -> _MachineResult:
        payload = {"operation": "initialize-automation-key"}
        _confirm_mutation(payload)
        return {"initialized": True, "key_id": initialize_approval_key()}, ()

    _run_machine("govern.key.initialize", operation)


@key_app.command("rotate")
def govern_key_rotate() -> None:
    """Rotate the local signing key after explicit invalidation confirmation."""

    def operation() -> _MachineResult:
        try:
            current = approval_key_id()
        except ApprovalError as exc:
            raise ControlError("policy_invalid", "automation key is not initialized") from exc
        payload = {
            "current_key_id": current,
            "effect": "all existing policy and approval signatures become invalid",
            "operation": "rotate-automation-key",
        }
        _confirm_mutation(payload)
        previous, replacement = rotate_approval_key()
        return {"key_id": replacement, "previous_key_id": previous}, ()

    _run_machine("govern.key.rotate", operation)


@policy_app.command("create")
def govern_policy_create(
    input_path: Path = typer.Option(..., "--input", help="Human-authored policy JSON file."),
) -> None:
    """Validate, confirm, sign, and store one policy-v2 document."""

    def operation() -> _MachineResult:
        _require_tty()
        policy = _path_contract(input_path, PolicyDocument.from_payload)
        _confirm_mutation(policy.to_payload())
        return _service().create_policy(policy), ()

    _run_machine("govern.policy.create", operation)


@policy_app.command("inspect")
def govern_policy_inspect(policy_id: str) -> None:
    """Inspect one exact stored policy without mutation."""
    _run_machine("govern.policy.inspect", lambda: (_service().inspect_policy(policy_id), ()))


@policy_app.command("revoke")
def govern_policy_revoke(policy_id: str) -> None:
    """Revoke one exact policy after typed TTY confirmation."""

    def operation() -> _MachineResult:
        _require_tty()
        current = _service().inspect_policy(policy_id)
        _confirm_mutation(
            {
                "operation": "revoke-policy",
                "policy_digest": current["policy_digest"],
                "policy_id": current["policy_id"],
            }
        )
        return _service().revoke_policy(policy_id), ()

    _run_machine("govern.policy.revoke", operation)


@approval_app.command("create")
def govern_approval_create(
    input_path: Path = typer.Option(..., "--input", help="Exact RunSpec JSON file."),
) -> None:
    """Issue one per-run approval after exact RunSpec digest confirmation."""

    def operation() -> _MachineResult:
        _require_tty()
        spec = _path_contract(input_path, RunSpec.from_payload)
        _confirm_mutation(spec.to_payload())
        return _service().create_approval(spec), ()

    _run_machine("govern.approval.create", operation)


@approval_app.command("inspect")
def govern_approval_inspect(approval_id: str) -> None:
    """Inspect one exact stored approval without mutation."""
    _run_machine(
        "govern.approval.inspect",
        lambda: (_service().inspect_approval(approval_id), ()),
    )


@approval_app.command("revoke")
def govern_approval_revoke(approval_id: str) -> None:
    """Revoke one exact approval after typed TTY confirmation."""

    def operation() -> _MachineResult:
        _require_tty()
        current = _service().inspect_approval(approval_id)
        approval = current["approval"]
        _confirm_mutation(
            {
                "approval_id": current["approval_id"],
                "operation": "revoke-approval",
                "spec_digest": approval["spec_digest"],
            }
        )
        return _service().revoke_approval(approval_id), ()

    _run_machine("govern.approval.revoke", operation)


def _validate_from_stdin() -> _MachineResult:
    spec = _stdin_contract(RunSpec.from_payload)
    validation = _service().validate(spec)
    if validation.decision.kind == DecisionKind.DENIED:
        raise ControlError(
            "policy_denied",
            "policy denied the RunSpec",
            details={"decision": validation.decision.to_payload()},
        )
    return validation.to_payload(), validation.decision.warnings


def _run_machine(operation: str, callback: Callable[[], _MachineResult]) -> None:
    try:
        data, warnings = callback()
        rendered = render_envelope(success_envelope(operation, data, warnings=warnings))
    except ControlError as exc:
        rendered = render_envelope(failure_envelope(operation, exc))
        typer.echo(rendered)
        raise typer.Exit(exit_code_for(exc)) from None
    except Exception:
        error = internal_error()
        rendered = render_envelope(failure_envelope(operation, error))
        typer.echo(rendered)
        raise typer.Exit(exit_code_for(error)) from None
    typer.echo(rendered)


def _stdin_contract(parser: Callable[[dict[str, Any]], _T]) -> _T:
    raw = sys.stdin.read(MAX_INPUT_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ControlError("input_too_large", "JSON input exceeds the 65536-byte limit")
    return _parse_contract(raw, parser)


def _path_contract(path: Path, parser: Callable[[dict[str, Any]], _T]) -> _T:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ControlError("invalid_field", "input file could not be read") from exc
    if len(raw) > MAX_INPUT_BYTES:
        raise ControlError("input_too_large", "JSON input exceeds the 65536-byte limit")
    return _parse_contract(raw, parser)


def _parse_contract(raw: str | bytes, parser: Callable[[dict[str, Any]], _T]) -> _T:
    try:
        payload = load_canonical_object(raw)
        return parser(payload)
    except CanonicalizationError as exc:
        raise ControlError("invalid_json", "input is not valid canonical JSON") from exc
    except ContractError as exc:
        raise ControlError("invalid_field", "input contract is invalid") from exc


def _confirm_mutation(payload: dict[str, Any]) -> None:
    _require_tty()
    digest = sha256_digest(payload)
    typer.echo(canonical_json(payload))
    typer.echo(f"Confirmation digest: {digest}")
    confirmation = typer.prompt("Type the full digest to confirm", hide_input=False)
    if not hmac.compare_digest(confirmation.strip(), digest):
        raise ControlError("confirmation_mismatch", "typed digest did not match")


def _require_tty() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ControlError("tty_required", "governance mutation requires an interactive TTY")


def _full_id(raw: str, label: str) -> str:
    value = raw.strip().lower()
    if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
        raise ControlError("invalid_field", f"{label} must be a full lowercase hexadecimal ID")
    return value


def _service() -> AutomationService:
    return AutomationService()
