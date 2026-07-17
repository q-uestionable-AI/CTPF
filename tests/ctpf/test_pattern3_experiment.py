"""Synthetic integration tests for the governed Pattern 3 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ctpf import experiment
from ctpf.automation.targets import scenario_capability
from ctpf.driven_inference import OpenAICompatibleTargetProfile
from ctpf.kernel import PromotionResult, verify_evidence_bundle
from ctpf.kernel.pattern3_scope import (
    APPROVED_VALUE,
    RECORD_ID,
    HardenedControlReason,
    Pattern3Condition,
    Pattern3Scenario,
    WorkflowAuthority,
    read_fixture_payload,
)

_TARGET_ID = "1" * 32
_TARGET_FINGERPRINT = "e" * 64


class _FakeControl:
    """Minimum governed control used without starting a target or model."""

    def __init__(self) -> None:
        self.run_id = "a" * 32
        self.capability = scenario_capability("pattern3-scope")
        reference = SimpleNamespace(
            target_id=_TARGET_ID,
            target_fingerprint=_TARGET_FINGERPRINT,
        )
        self.spec = SimpleNamespace(experiment=SimpleNamespace(targets=(reference,)))
        self.checkpoints: list[str] = []

    def checkpoint(self, boundary: str) -> None:
        self.checkpoints.append(boundary)

    def provenance_payload(self) -> dict[str, Any]:
        return {
            "grant_digest": "c" * 64,
            "grant_id": "d" * 32,
            "policy_digest": "b" * 64,
            "run_id": self.run_id,
            "scenario_fingerprint": self.capability.fingerprint,
            "spec_digest": "a" * 64,
        }

    def target_policy(self, target_id: str) -> SimpleNamespace:
        assert target_id == _TARGET_ID
        return SimpleNamespace(target_fingerprint=_TARGET_FINGERPRINT)


def _profile() -> OpenAICompatibleTargetProfile:
    return OpenAICompatibleTargetProfile(
        target_id=_TARGET_ID,
        name="synthetic-local-target",
        endpoint="http://127.0.0.1:8080/v1",
        model="synthetic-model",
        credential_name="unused-synthetic-key",
    )


def _trace_message(
    proxy_id: str,
    direction: str,
    payload: dict[str, Any],
    *,
    correlated_id: str | None = None,
) -> dict[str, Any]:
    return {
        "correlated_id": correlated_id,
        "direction": direction,
        "modified": False,
        "payload": payload,
        "proxy_id": proxy_id,
    }


def _read_exchange(
    condition: Pattern3Condition, scenario: Pattern3Scenario
) -> list[dict[str, Any]]:
    request = _trace_message(
        "read-request",
        "client_to_server",
        {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"arguments": {"record_id": RECORD_ID}, "name": "read_record"},
        },
    )
    response = _trace_message(
        "read-response",
        "server_to_client",
        {
            "id": 1,
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {
                        "text": json.dumps(
                            read_fixture_payload(condition, scenario), sort_keys=True
                        ),
                        "type": "text",
                    }
                ],
                "isError": False,
            },
        },
        correlated_id="read-request",
    )
    return [request, response]


def _write_exchange(*, error: str | None, response_text: str) -> list[dict[str, Any]]:
    request = _trace_message(
        "write-request",
        "client_to_server",
        {
            "id": 2,
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "arguments": {"record_id": RECORD_ID, "value": APPROVED_VALUE},
                "name": "write_record",
            },
        },
    )
    response = _trace_message(
        "write-response",
        "server_to_client",
        {
            "id": 2,
            "jsonrpc": "2.0",
            "result": {
                "content": [{"text": response_text, "type": "text"}],
                "isError": error is not None,
            },
        },
        correlated_id="write-request",
    )
    return [request, response]


def _write_synthetic_session(spec: experiment._SessionSpec) -> None:
    authority_payload = json.loads(
        (spec.trace_path.parent / "authority.json").read_text(encoding="utf-8")
    )
    authority = WorkflowAuthority.from_payload(authority_payload)
    scenario = Pattern3Scenario(series_id=authority.series_id)
    messages = _read_exchange(authority.condition, scenario)
    if authority.condition == Pattern3Condition.OPPORTUNITY:
        sink = experiment._pattern3_fixture_artifact_path(spec.run_id)
        effect = {
            "condition": authority.condition.value,
            "effect": "applied",
            "effect_id": authority.effect_id,
            "grant_source": "absent_vulnerable_arm",
            "record_id": authority.record_id,
            "run_id": spec.run_id,
            "sink_path": str(sink),
            "value": APPROVED_VALUE,
            "workflow_authority_digest": authority.digest(),
        }
        sink.parent.mkdir(parents=True, exist_ok=True)
        sink.write_text(json.dumps(effect), encoding="utf-8")
        messages.extend(_write_exchange(error=None, response_text=json.dumps(effect)))
    elif authority.condition == Pattern3Condition.HARDENED_OPPORTUNITY:
        messages.extend(
            _write_exchange(
                error="grant_missing",
                response_text="Error executing write_record: grant_missing",
            )
        )
    spec.trace_path.write_text(
        json.dumps({"ended_at": "synthetic-complete", "messages": messages}),
        encoding="utf-8",
    )
    if spec.inference_path is not None:
        spec.inference_path.write_text('{"synthetic":true}\n', encoding="utf-8")


def test_pattern3_worker_contract_is_single_reset_and_unmutated(tmp_path: Path) -> None:
    trace = tmp_path / "opportunity" / "session.json"
    spec = experiment._worker_session_spec(
        "pattern3-scope",
        Pattern3Condition.OPPORTUNITY,
        "single",
        trace,
        "series-opportunity",
        True,
        None,
        trace.with_suffix(".inference.json"),
    )

    assert spec.prompt == experiment.PATTERN3_PROMPT
    assert spec.fixture_command.endswith("-m ctpf.kernel.pattern3_scope_fixture")
    assert spec.expected_tool_count == 3
    assert spec.environment == {
        "CTPF_PATTERN3_AUTHORITY_PATH": str(trace.parent / "authority.json"),
        "CTPF_PATTERN3_CONDITION": "opportunity",
        "CTPF_PATTERN3_RESET_SINK": "1",
        "CTPF_PATTERN3_RUN_ID": "series-opportunity",
    }
    with pytest.raises(experiment.ExperimentError, match="reset single session"):
        experiment._worker_session_spec(
            "pattern3-scope",
            Pattern3Condition.OPPORTUNITY,
            "single",
            trace,
            "series-opportunity",
            False,
            None,
            None,
        )
    with pytest.raises(experiment.ExperimentError, match="does not permit proxy mutation"):
        experiment._worker_session_spec(
            "pattern3-scope",
            Pattern3Condition.OPPORTUNITY,
            "single",
            trace,
            "series-opportunity",
            True,
            tmp_path / "mutation.json",
            None,
        )


@pytest.mark.asyncio
async def test_governed_pattern3_synthetic_series_writes_verifiable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _FakeControl()
    profile = _profile()
    output_root = (tmp_path / "research-evidence").resolve()
    output_root.mkdir()
    monkeypatch.setenv("TEMP", str(tmp_path / "fixture-temp"))

    async def capture(
        spec: experiment._SessionSpec,
        _options: experiment.Pattern3ExperimentOptions,
        _operator: experiment._Operator,
        _control: _FakeControl,
    ) -> None:
        _write_synthetic_session(spec)

    monkeypatch.setattr(experiment, "_capture_session", capture)
    completed = await experiment.run_pattern3_scope(
        experiment.Pattern3ExperimentOptions(
            None,
            output_root,
            target=profile.target_id,
        ),
        profile=profile,
        control=control,  # type: ignore[arg-type]
    )

    assert completed.primary.promotion_result == PromotionResult.CONFIRMED
    assert completed.hardened.passed is True
    assert completed.hardened.reason == HardenedControlReason.BLOCKED_GRANT_MISSING
    verification = verify_evidence_bundle(completed.bundle.root)
    assert verification.ok is True
    manifest = json.loads((completed.root / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["scenario"] == "pattern3-scope"
    expected_order = [
        "baseline",
        "opportunity",
        "hardened_opportunity",
    ]
    assert manifest["condition_order"] == expected_order
    assert set(manifest["conditions"]) == set(expected_order)
    assert control.checkpoints.count("condition") == 3

    authority_artifact = completed.bundle.root / "artifacts" / "opportunity" / "authority.json"
    authority_artifact.write_text("{}\n", encoding="utf-8")
    assert verify_evidence_bundle(completed.bundle.root).ok is False
