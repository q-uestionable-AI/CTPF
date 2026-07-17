"""Tests for installed scenario and demonstrated target fingerprints."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from ctpf import driven_inference, external_runtime
from ctpf.automation import targets
from ctpf.automation.canonical import sha256_digest
from ctpf.automation.contracts import (
    BillingClass,
    DataEgressClass,
    ExperimentMode,
    NetworkClass,
    TargetPolicy,
)
from ctpf.automation.targets import (
    TargetIdentityError,
    classify_inference_endpoint,
    installed_scenario_capabilities,
    load_target_identity,
    target_identity_from_policy,
    target_identity_from_profile,
)
from ctpf.core.db import create_target, get_connection


def test_installed_capabilities_are_stable_and_cover_demonstrated_scenarios() -> None:
    """Fingerprint inputs pin the exact packaged prompts, tools, effects, and code."""
    first = installed_scenario_capabilities()
    second = installed_scenario_capabilities()

    assert [item.scenario for item in first] == [
        "cascade-memo",
        "pattern2",
        "pattern3-scope",
    ]
    assert [item.fingerprint for item in first] == [item.fingerprint for item in second]
    cascade, pattern2, pattern3 = first
    assert cascade.modes == (ExperimentMode.SINGLE, ExperimentMode.MATRIX)
    assert cascade.sessions_per_trial == 6
    assert set(cascade.effect_ids) == {
        "cascade-action-sink",
        "cascade-memo-persistence",
    }
    assert pattern2.tool_names == ("apply_change", "read_sink", "read_status")
    assert pattern2.sessions_per_trial == 3
    assert pattern3.conditions == ("baseline", "opportunity", "hardened_opportunity")
    assert pattern3.tool_names == ("read_record", "read_sink", "write_record")
    assert pattern3.effect_ids == ("pattern3-write-sink",)
    assert pattern3.sessions_per_trial == 3
    assert "kernel/slice.py" in pattern3.source_hashes
    assert all(len(value) == 64 for item in first for value in item.source_hashes.values())


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://127.0.0.1:8000/v1", NetworkClass.LOOPBACK),
        ("https://127.0.0.1:8000/v1", NetworkClass.LOOPBACK),
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
        "http://localhost:8000/v1",
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
                "billing_class": "unmetered",
                "data_egress_class": "packaged_synthetic_remote",
                "retention_acknowledged": True,
                "residual_cost_acknowledged": True,
            },
        )

    identity = load_target_identity(target_id, db_path=db_path)

    assert identity.target_id == target_id
    assert identity.network_class == NetworkClass.HTTPS_PUBLIC
    assert identity.behavior["credential_alias"] == "remote-a"
    assert identity.behavior["limits"]["max_provider_rounds"] == 12
    assert len(identity.behavior["driver_source_hash"]) == 64
    assert identity.behavior["endpoint"]["normalized_url"] == "https://models.example.test/v1"
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
                "retention_acknowledged": True,
                "residual_cost_acknowledged": True,
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
    assert identity.behavior["identity_probe_processes"] == 1
    assert (
        identity.behavior["identity_probe_timeout_seconds"]
        == external_runtime.VERSION_PROBE_TIMEOUT_SECONDS
    )
    assert identity.behavior["mcp_policy"] == "strict loopback allowlisted-tools only"
    assert "credential" not in str(identity.to_payload()).lower()


def test_authenticated_snapshots_reconstruct_fingerprint_equivalent_profiles(
    tmp_path: Path,
) -> None:
    """Isolated workers derive both adapter profiles without broadening signed identity."""
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"synthetic executable content")
    profiles = (
        driven_inference.OpenAICompatibleTargetProfile(
            target_id="a" * 32,
            name="local inference",
            endpoint="http://127.0.0.1:11434/v1",
            model="model-a",
            credential_name="local-a",
            max_tokens=256,
            max_input_tokens=512,
        ),
        external_runtime.ClaudeCodeTargetProfile(
            target_id="b" * 32,
            name="runtime",
            executable=str(executable),
            model="claude-test-model",
            runtime_version="2.1.121 (Claude Code)",
            timeout_seconds=90,
            retention_acknowledged=True,
            residual_cost_acknowledged=True,
        ),
    )

    for profile in profiles:
        identity = target_identity_from_profile(profile)
        policy_target = TargetPolicy(
            identity.target_id,
            identity.fingerprint,
            identity.target_type,
            identity.behavior,
            identity.network_class,
            profile.billing_class,
            getattr(profile, "request_cost_ceiling_microusd", None),
            profile.data_egress_class,
            profile.retention_acknowledged,
            profile.residual_cost_acknowledged,
        )
        reconstructed = targets.execution_profile_from_policy(policy_target)
        assert target_identity_from_profile(reconstructed).fingerprint == identity.fingerprint


def test_target_identity_requires_full_unambiguous_id(tmp_path: Path) -> None:
    """Automation never resolves the partial IDs accepted by interactive commands."""
    with pytest.raises(TargetIdentityError, match="full lowercase"):
        load_target_identity("abcdef12", db_path=tmp_path / "ctpf.db")


def test_signed_runtime_snapshot_validation_is_stateless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Machine validation never launches the Claude identity probe or reloads a profile."""
    target_id = "a" * 32
    behavior = {
        "authority_contract_version": 1,
        "billing": {
            "billing_class": "external_runtime",
            "request_cost_ceiling_microusd": None,
            "residual_cost_acknowledged": True,
        },
        "data_egress": {
            "data_egress_class": "external_runtime",
            "retention_acknowledged": True,
        },
        "driver": "claude-code-cli",
        "driver_source_hash": hashlib.sha256(
            Path(external_runtime.__file__).read_bytes()
        ).hexdigest(),
        "environment_policy": "minimal non-secret allowlist",
        "executable": str((tmp_path / "claude").resolve()),
        "executable_sha256": "c" * 64,
        "identity_probe_processes": 1,
        "identity_probe_timeout_seconds": 10,
        "mcp_policy": "strict loopback allowlisted-tools only",
        "model": "claude-test-model",
        "runtime_version": "2.1.121 (Claude Code)",
        "target_id": target_id,
        "target_type": "agent-runtime",
        "timeout_seconds": 90,
    }
    policy_target = TargetPolicy(
        target_id,
        sha256_digest(behavior),
        "agent-runtime",
        behavior,
        NetworkClass.EXTERNAL_RUNTIME,
        BillingClass.EXTERNAL_RUNTIME,
        None,
        DataEgressClass.EXTERNAL_RUNTIME,
        True,
        True,
    )

    def unexpected_profile_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stateless snapshot validation loaded a live target")

    monkeypatch.setattr(targets, "load_experiment_target_profile", unexpected_profile_load)
    identity = target_identity_from_policy(policy_target)
    assert identity.behavior["runtime_version"] == "2.1.121 (Claude Code)"

    tampered = dict(behavior)
    tampered["runtime_version"] = "changed"
    with pytest.raises(TargetIdentityError, match="fingerprint"):
        target_identity_from_policy(replace(policy_target, behavior=tampered))


def test_signed_inference_snapshot_cannot_understate_fixed_driver_rounds() -> None:
    """A self-consistent snapshot still fails if it changes installed driver constants."""
    target_id = "a" * 32
    identity = target_identity_from_profile(
        driven_inference.OpenAICompatibleTargetProfile(
            target_id=target_id,
            name="test target",
            endpoint="http://127.0.0.1:11434/v1",
            model="test-model",
            credential_name="test-key",
            max_tokens=256,
            max_input_tokens=256,
        )
    )
    behavior = dict(identity.behavior)
    limits = dict(behavior["limits"])
    limits["max_provider_rounds"] = 1
    behavior["limits"] = limits
    policy_target = TargetPolicy(
        target_id,
        sha256_digest(behavior),
        "inference",
        behavior,
        NetworkClass.LOOPBACK,
        BillingClass.UNMETERED,
        None,
        DataEgressClass.LOCAL_ONLY,
        False,
        False,
    )

    with pytest.raises(TargetIdentityError, match="installed controls"):
        target_identity_from_policy(policy_target)

    limits["max_provider_rounds"] = driven_inference.DEFAULT_MAX_ROUNDS
    behavior["limits"] = limits
    behavior["driver_source_hash"] = "d" * 64
    changed = replace(policy_target, behavior=behavior, target_fingerprint=sha256_digest(behavior))
    with pytest.raises(TargetIdentityError, match="installed driver"):
        target_identity_from_policy(changed)


def test_every_profile_authority_mutation_changes_inference_fingerprint() -> None:
    """Every configurable endpoint, model, credential, generation, and budget pin is hashed."""
    profile = driven_inference.OpenAICompatibleTargetProfile(
        target_id="a" * 32,
        name="remote test",
        endpoint="https://models.example.test/v1",
        model="model-a",
        credential_name="remote-a",
        max_tokens=256,
        temperature=0.0,
        seed=1,
        reasoning_effort="low",
        max_input_tokens=512,
        data_egress_class=DataEgressClass.PACKAGED_SYNTHETIC_REMOTE,
        retention_acknowledged=True,
        residual_cost_acknowledged=True,
    )
    baseline = target_identity_from_profile(profile).fingerprint
    mutations = (
        replace(profile, endpoint="https://other.example.test/v1"),
        replace(profile, model="model-b"),
        replace(profile, credential_name="remote-b"),
        replace(profile, max_tokens=257),
        replace(profile, max_input_tokens=513),
        replace(profile, temperature=0.1),
        replace(profile, seed=2),
        replace(profile, reasoning_effort="medium"),
        replace(
            profile,
            billing_class=BillingClass.METERED,
            request_cost_ceiling_microusd=10_000,
        ),
    )

    fingerprints = {target_identity_from_profile(item).fingerprint for item in mutations}

    assert baseline not in fingerprints
    assert len(fingerprints) == len(mutations)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("limits", "max_provider_rounds", 13),
        ("limits", "max_request_bytes", 1),
        ("limits", "max_response_bytes", 1),
        ("limits", "max_tool_calls_per_session", 13),
        ("transport", "max_attempts", 2),
        ("transport", "concurrent_requests", 2),
        ("transport", "retry_count", 1),
        ("transport", "redirect_policy", "follow"),
        ("transport", "environment_proxy_policy", "inherit"),
        ("transport", "http_protocol", "HTTP/2"),
        ("transport", "httpx_version", "changed"),
        ("transport", "tls_policy", "disabled"),
    ],
)
def test_signed_snapshot_cannot_mutate_installed_transport_authority(
    section: str,
    key: str,
    value: object,
) -> None:
    """A re-fingerprinted policy still cannot invent uninstalled transport controls."""
    profile = driven_inference.OpenAICompatibleTargetProfile(
        target_id="a" * 32,
        name="local test",
        endpoint="http://127.0.0.1:11434/v1",
        model="model-a",
        credential_name="local-a",
        max_tokens=256,
        max_input_tokens=512,
    )
    identity = target_identity_from_profile(profile)
    behavior = copy.deepcopy(identity.behavior)
    behavior[section][key] = value
    policy_target = TargetPolicy(
        profile.target_id,
        sha256_digest(behavior),
        "inference",
        behavior,
        NetworkClass.LOOPBACK,
        BillingClass.UNMETERED,
        None,
        DataEgressClass.LOCAL_ONLY,
        False,
        False,
    )

    with pytest.raises(TargetIdentityError, match="installed controls"):
        target_identity_from_policy(policy_target)
