"""Tests for the agent-facing machine control CLI."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from ctpf.automation import cli as automation_cli
from ctpf.automation.envelope import ControlError
from ctpf.cli import app

runner = CliRunner()


class _FakeService:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self._result = result

    def capabilities(self, policy: str | None) -> dict[str, Any]:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    async def execute(self, run_id: str) -> dict[str, Any]:
        del run_id
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _envelope(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert isinstance(parsed, dict)
    return parsed


def test_experiment_help_registers_control_and_govern_without_hiding_run() -> None:
    """The sole package entry point exposes both new apps beside human runs."""
    result = runner.invoke(app, ["experiment", "--help"])

    assert result.exit_code == 0
    assert all(command in result.output for command in ("control", "govern", "run"))


def test_capabilities_emits_one_canonical_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful machine operations emit exactly one JSON object."""
    expected = {"execute_available": True, "verify_available": True}
    monkeypatch.setattr(automation_cli, "_service", lambda: _FakeService(expected))

    result = runner.invoke(app, ["experiment", "control", "capabilities"])
    payload = _envelope(result.output)

    assert result.exit_code == 0
    assert payload == {
        "data": expected,
        "ok": True,
        "operation": "capabilities",
        "schema_version": 1,
        "warnings": [],
    }


def test_invalid_input_and_missing_execution_use_stable_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input and lifecycle failures remain machine-branchable and traceback-free."""
    missing = _FakeService(ControlError("run_not_found", "run was not found"))
    monkeypatch.setattr(automation_cli, "_service", lambda: missing)
    malformed = runner.invoke(app, ["experiment", "control", "validate"], input="[]")
    absent = runner.invoke(
        app,
        ["experiment", "control", "execute", "a" * 32],
    )

    malformed_payload = _envelope(malformed.output)
    absent_payload = _envelope(absent.output)
    assert malformed.exit_code == 2
    assert malformed_payload["error"]["code"] == "invalid_json"
    assert absent.exit_code == 4
    assert absent_payload["error"]["code"] == "run_not_found"
    assert "Traceback" not in malformed.output + absent.output


def test_execute_emits_one_canonical_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreground execution remains a single machine-readable operation."""
    expected = {"run_id": "a" * 32, "state": "COMPLETED"}
    monkeypatch.setattr(automation_cli, "_service", lambda: _FakeService(expected))

    result = runner.invoke(app, ["experiment", "control", "execute", "a" * 32])
    payload = _envelope(result.output)

    assert result.exit_code == 0
    assert payload["data"] == expected
    assert payload["operation"] == "execute"


def test_not_found_is_state_exit_and_unexpected_errors_are_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known state failures use exit 4; unknown exceptions disclose no raw text."""
    missing = _FakeService(ControlError("policy_not_found", "policy was not found"))
    monkeypatch.setattr(automation_cli, "_service", lambda: missing)
    not_found = runner.invoke(
        app,
        ["experiment", "control", "capabilities", "--policy", "a" * 32],
    )
    assert not_found.exit_code == 4
    assert _envelope(not_found.output)["error"]["code"] == "policy_not_found"

    broken = _FakeService(RuntimeError("provider-secret-value"))
    monkeypatch.setattr(automation_cli, "_service", lambda: broken)
    internal = runner.invoke(app, ["experiment", "control", "capabilities"])
    payload = _envelope(internal.output)
    assert internal.exit_code == 7
    assert payload["error"] == {
        "code": "internal_error",
        "details": {},
        "message": "unexpected internal automation failure",
    }
    assert "provider-secret-value" not in internal.output
    assert "Traceback" not in internal.output
