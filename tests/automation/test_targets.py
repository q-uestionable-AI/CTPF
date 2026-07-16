"""Tests for installed scenario and demonstrated target fingerprints."""

from __future__ import annotations

from pathlib import Path

import pytest

from ctpf import external_runtime
from ctpf.automation.contracts import ExperimentMode, NetworkClass
from ctpf.automation.targets import (
    TargetIdentityError,
    classify_inference_endpoint,
    installed_scenario_capabilities,
    load_target_identity,
)
from ctpf.core.db import create_target, get_connection


def test_installed_capabilities_are_stable_and_cover_demonstrated_scenarios() -> None:
    """Fingerprint inputs pin the exact packaged prompts, tools, effects, and code."""
    first = installed_scenario_capabilities()
    second = installed_scenario_capabilities()

    assert [item.scenario for item in first] == ["cascade-memo", "pattern2"]
    assert [item.fingerprint for item in first] == [item.fingerprint for item in second]
    cascade, pattern2 = first
    assert cascade.modes == (ExperimentMode.SINGLE, ExperimentMode.MATRIX)
    assert cascade.sessions_per_trial == 2
    assert set(cascade.effect_ids) == {
        "cascade-action-sink",
        "cascade-memo-persistence",
    }
    assert pattern2.tool_names == ("apply_change", "read_sink", "read_status")
    assert all(len(value) == 64 for item in first for value in item.source_hashes.values())


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://127.0.0.1:8000/v1", NetworkClass.LOOPBACK),
        ("http://localhost:8000/v1", NetworkClass.LOOPBACK),
        ("https://models.example.test/v1", NetworkClass.HTTPS_PUBLIC),
    ],
)
def test_inference_endpoint_classification(endpoint: str, expected: NetworkClass) -> None:
    """Only loopback HTTP and fully qualified public HTTPS are classifiable."""
    assert classify_inference_endpoint(endpoint) == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://models.example.test/v1",
        "https://127.0.0.1/v1",
        "https://192.168.1.20/v1",
        "https://user:secret@models.example.test/v1",
        "https://unqualified/v1",
        "https://models.example.test/v1?token=secret",
    ],
)
def test_inference_endpoint_rejects_ambiguous_or_unsafe_authorities(endpoint: str) -> None:
    """Endpoint identity fails closed for unsupported network authority."""
    with pytest.raises(TargetIdentityError):
        classify_inference_endpoint(endpoint)


def test_inference_target_identity_pins_behavior_without_secret_value(tmp_path: Path) -> None:
    """Persisted demonstrated inference settings produce a credential-free fingerprint."""
    db_path = tmp_path / "ctpf.db"
    with get_connection(db_path) as conn:
        target_id = create_target(
            conn,
            type="inference",
            name="remote research model",
            uri="https://models.example.test/v1/",
            metadata={
                "credential": "remote-a",
                "driver": "openai-compatible",
                "max_tokens": "512",
                "model": "model-a",
                "reasoning_effort": "low",
                "seed": "42",
                "temperature": "0",
            },
        )

    identity = load_target_identity(target_id, db_path=db_path)

    assert identity.target_id == target_id
    assert identity.network_class == NetworkClass.HTTPS_PUBLIC
    assert identity.behavior["credential_alias"] == "remote-a"
    assert identity.behavior["max_provider_rounds"] == 12
    assert len(identity.behavior["driver_source_hash"]) == 64
    assert identity.behavior["endpoint"] == "https://models.example.test/v1"
    assert identity.behavior["generation_parameters"] == {
        "reasoning_effort": "low",
        "seed": 42,
        "temperature": "0",
    }
    assert "remote research model" not in str(identity.to_payload())
    assert len(identity.fingerprint) == 64


def test_external_runtime_target_identity_pins_inspected_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The external seam binds the executable, runtime version, model, and controls."""
    db_path = tmp_path / "ctpf.db"
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"synthetic executable content")
    with get_connection(db_path) as conn:
        target_id = create_target(
            conn,
            type="agent-runtime",
            name="claude research runtime",
            uri="claude",
            metadata={
                "driver": "claude-code-cli",
                "model": "claude-opus-4-1-20250805",
                "timeout_seconds": "90",
            },
        )
    monkeypatch.setattr(
        external_runtime,
        "_inspect_claude_executable",
        lambda _raw: (str(executable), "2.1.121 (Claude Code)"),
    )

    identity = load_target_identity(target_id, db_path=db_path)

    assert identity.network_class == NetworkClass.EXTERNAL_RUNTIME
    assert identity.behavior["runtime_version"] == "2.1.121 (Claude Code)"
    assert len(identity.behavior["driver_source_hash"]) == 64
    assert len(identity.behavior["executable_sha256"]) == 64
    assert identity.behavior["mcp_policy"] == "strict loopback allowlisted-tools only"
    assert "credential" not in str(identity.to_payload()).lower()


def test_target_identity_requires_full_unambiguous_id(tmp_path: Path) -> None:
    """Automation never resolves the partial IDs accepted by interactive commands."""
    with pytest.raises(TargetIdentityError, match="full lowercase"):
        load_target_identity("abcdef12", db_path=tmp_path / "ctpf.db")
