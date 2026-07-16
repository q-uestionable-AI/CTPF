"""Tests for bounded canonical JSON and strict automation contracts."""

from __future__ import annotations

import json

import pytest

from ctpf.automation.canonical import (
    MAX_INPUT_BYTES,
    CanonicalizationError,
    canonical_json,
    load_canonical_object,
    sha256_digest,
)
from ctpf.automation.contracts import ContractError, PolicyDocument, RunSpec

POLICY_ID = "a" * 32
TARGET_ID = "b" * 32
SCENARIO_FINGERPRINT = "c" * 64
TARGET_FINGERPRINT = "d" * 64


def _limits() -> dict[str, int]:
    return {
        "cost_limit_microusd": 0,
        "output_tokens_reserved": 512,
        "provider_requests": 2,
        "runtime_processes": 1,
        "tool_calls": 8,
        "wall_clock_seconds": 120,
    }


def _run_payload() -> dict[str, object]:
    return {
        "canonicalization": "ctpf-canonical-json-v1",
        "experiment": {
            "mode": "single",
            "scenario": "pattern2",
            "scenario_fingerprint": SCENARIO_FINGERPRINT,
            "targets": [
                {
                    "target_fingerprint": TARGET_FINGERPRINT,
                    "target_id": TARGET_ID,
                }
            ],
            "trials_per_target": 1,
        },
        "idempotency_key": "agent-request-0001",
        "limits": _limits(),
        "output_root_id": "research-evidence",
        "policy_id": POLICY_ID,
        "purpose": "Test the installed synthetic scenario.",
        "requested_tier": 1,
        "requester": {"kind": "agent", "name": "test-agent", "version": "1"},
        "schema_version": 1,
    }


def _policy_payload() -> dict[str, object]:
    return {
        "allowed_effects": ["pattern2-action-sink"],
        "canonicalization": "ctpf-canonical-json-v1",
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "limits": {
            **_limits(),
            "approval_lifetime_seconds": 300,
            "concurrent_runs": 1,
            "loopback_port": 8765,
        },
        "name": "local synthetic policy",
        "output_roots": [{"resolved_path": "C:/research/evidence", "root_id": "research-evidence"}],
        "per_run_tiers": [2],
        "policy_id": POLICY_ID,
        "scenarios": [
            {
                "fingerprints": [SCENARIO_FINGERPRINT],
                "max_trials_per_target": 1,
                "modes": ["single"],
                "scenario": "pattern2",
            }
        ],
        "schema_version": 1,
        "standing_tiers": [1],
        "targets": [
            {
                "billing_class": "unmetered",
                "network_class": "loopback",
                "request_cost_ceiling_microusd": None,
                "target_fingerprint": TARGET_FINGERPRINT,
                "target_id": TARGET_ID,
            }
        ],
    }


def test_canonical_json_is_order_independent_and_stable() -> None:
    """Object insertion order does not affect bytes or digests."""
    left = {"b": [True, None, 3], "a": {"z": "value"}}
    right = {"a": {"z": "value"}, "b": [True, None, 3]}

    assert canonical_json(left) == '{"a":{"z":"value"},"b":[true,null,3]}'
    assert canonical_json(left) == canonical_json(right)
    assert sha256_digest(left) == sha256_digest(right)


@pytest.mark.parametrize(
    "raw",
    [
        '{"duplicate":1,"duplicate":2}',
        '{"float":1.0}',
        '{"nan":NaN}',
        "[]",
        "",
    ],
)
def test_canonical_parser_rejects_ambiguous_or_non_object_json(raw: str) -> None:
    """Authorization JSON rejects ambiguity before contract parsing."""
    with pytest.raises(CanonicalizationError):
        load_canonical_object(raw)


def test_canonical_parser_enforces_encoded_size_bound() -> None:
    """Oversized records fail before decoding or hashing."""
    raw = json.dumps({"value": "x" * MAX_INPUT_BYTES})

    with pytest.raises(CanonicalizationError, match="exceeds"):
        load_canonical_object(raw)


def test_runspec_round_trip_is_strict() -> None:
    """A valid body normalizes exactly and unknown fields fail closed."""
    payload = _run_payload()
    parsed = RunSpec.from_payload(payload)

    assert parsed.to_payload() == payload
    payload["unexpected"] = True
    with pytest.raises(ContractError, match="unknown fields"):
        RunSpec.from_payload(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("requested_tier", 3), "authorization tier"),
        (("policy_id", "short"), "full lowercase"),
        (("idempotency_key", "short"), "16-128"),
    ],
)
def test_runspec_rejects_unsupported_authority_and_partial_ids(
    mutation: tuple[str, object],
    message: str,
) -> None:
    """Untrusted proposals cannot widen authority or use ambiguous IDs."""
    payload = _run_payload()
    payload[mutation[0]] = mutation[1]

    with pytest.raises(ContractError, match=message):
        RunSpec.from_payload(payload)


def test_matrix_mode_requires_cascade_multiple_targets_and_three_trials() -> None:
    """The parser exposes only the demonstrated matrix workflow."""
    payload = _run_payload()
    experiment = payload["experiment"]
    assert isinstance(experiment, dict)
    experiment.update({"mode": "matrix", "trials_per_target": 2})

    with pytest.raises(ContractError, match="3-5 trials"):
        RunSpec.from_payload(payload)


def test_policy_round_trip_rejects_invalid_interval_and_cost_classification() -> None:
    """Policy validity and metered cost declarations are mandatory."""
    payload = _policy_payload()
    assert PolicyDocument.from_payload(payload).to_payload() == payload

    payload["expires_at"] = "2025-01-01T00:00:00Z"
    with pytest.raises(ContractError, match="later than"):
        PolicyDocument.from_payload(payload)

    payload = _policy_payload()
    targets = payload["targets"]
    assert isinstance(targets, list) and isinstance(targets[0], dict)
    targets[0]["billing_class"] = "metered"
    with pytest.raises(ContractError, match="require request_cost"):
        PolicyDocument.from_payload(payload)


def test_contract_timestamp_rejects_impossible_calendar_date() -> None:
    """Shape-valid but impossible timestamps fail during strict parsing."""
    payload = _policy_payload()
    payload["created_at"] = "2026-02-30T00:00:00Z"

    with pytest.raises(ContractError, match="valid calendar"):
        PolicyDocument.from_payload(payload)


def test_policy_output_root_must_be_absolute_local_and_traversal_free() -> None:
    """Signed evidence roots cannot depend on CWD or address a network share."""
    for path in ("relative/evidence", "../evidence", "//server/share/evidence"):
        payload = _policy_payload()
        roots = payload["output_roots"]
        assert isinstance(roots, list) and isinstance(roots[0], dict)
        roots[0]["resolved_path"] = path
        with pytest.raises(ContractError, match="resolved_path"):
            PolicyDocument.from_payload(payload)
