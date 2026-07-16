"""Tests for provider-agnostic response models and evidence redaction."""

from __future__ import annotations

from ctpf.core.llm import NormalizedResponse, ToolCall
from ctpf.core.redaction import REDACTED, TRUNCATED, sanitize_evidence


def test_response_separates_requested_reported_and_proven_identity() -> None:
    """Provider claims do not overwrite requested or artifact-proven identity."""
    response = NormalizedResponse(
        requested_model="requested-model",
        provider_reported_model="claimed-model",
        artifact_proven_model=None,
        tool_calls=[ToolCall("read_status", {}, "call-1")],
    )

    assert response.requested_model == "requested-model"
    assert response.provider_reported_model == "claimed-model"
    assert response.artifact_proven_model is None


def test_recursive_sanitizer_redacts_keys_values_and_bounds_depth() -> None:
    """Nested secret echoes cannot reach durable evidence."""
    value: dict[str, object] = {
        "authorization": "Bearer exact-secret",
        "nested": {"message": "provider echoed exact-secret"},
    }
    cursor = value
    for _ in range(14):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child

    sanitized = sanitize_evidence(value, ("exact-secret",))

    assert sanitized["authorization"] == REDACTED
    assert sanitized["nested"]["message"] == f"provider echoed {REDACTED}"
    assert TRUNCATED in str(sanitized)
