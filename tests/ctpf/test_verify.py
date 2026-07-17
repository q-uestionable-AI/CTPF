"""Tests for the first-class evidence-bundle verifier."""

from __future__ import annotations

import json
from pathlib import Path

from ctpf.kernel import (
    BASELINE_TRACE_NAME,
    BUNDLE_SCHEMA_CURRENT,
    CONDITION_BASELINE,
    CONDITION_MANIPULATED,
    MANIFEST_NAME,
    MANIPULATED_SINK_NAME,
    MANIPULATED_TRACE_NAME,
    RESULT_NAME,
    ExperimentContext,
    ExperimentPins,
    ExternalEffect,
    Pattern2Scenario,
    PromotionReason,
    PromotionResult,
    RunObservation,
    compare_baseline_manipulated,
    verify_evidence_bundle,
    write_evidence_bundle,
)

PINS = ExperimentPins(
    agent="Cursor Agent",
    model="test-model",
    configuration={"scenario": "pattern2"},
)


def _observation(
    condition: str,
    *,
    tool: str | None,
    effect_present: bool,
    action: str = "approve_refund",
) -> RunObservation:
    return RunObservation(
        condition=condition,
        tool_invocation=tool,
        tool_arguments=None if tool is None else {"action": action, "reason": "test"},
        external_effect=ExternalEffect(
            present=effect_present,
            payload=(
                {"effect": "applied", "action": action, "run_id": "r1"} if effect_present else None
            ),
            sink_path=None,
            reason="effect_applied" if effect_present else "sink_missing",
        ),
        evidence_complete=True,
        evidence_notes=(),
    )


def _write_bundle(tmp_path: Path) -> Path:
    baseline = _observation(CONDITION_BASELINE, tool=None, effect_present=False)
    manipulated = _observation(CONDITION_MANIPULATED, tool="apply_change", effect_present=True)
    transition = compare_baseline_manipulated(baseline, manipulated)
    assert transition.promotion_result == PromotionResult.CONFIRMED
    assert (
        transition.promotion_reason == PromotionReason.CONFIRMED_CLEAN_BASELINE_PROMOTED_TREATMENT
    )
    baseline_trace = tmp_path / "baseline.json"
    manipulated_trace = tmp_path / "manipulated.json"
    sink = tmp_path / "sink.json"
    baseline_trace.write_text("{}\n", encoding="utf-8")
    manipulated_trace.write_text("{}\n", encoding="utf-8")
    sink.write_text(json.dumps({"effect": "applied", "action": "approve_refund"}), encoding="utf-8")
    return write_evidence_bundle(
        tmp_path / "bundle",
        result=transition,
        experiment=ExperimentContext(
            baseline=baseline,
            manipulated=manipulated,
            pins=PINS,
            scenario=Pattern2Scenario(),
        ),
        artifacts={
            BASELINE_TRACE_NAME: baseline_trace,
            MANIPULATED_TRACE_NAME: manipulated_trace,
            MANIPULATED_SINK_NAME: sink,
        },
    ).root


class TestVerifyEvidenceBundle:
    def test_current_bundle_passes(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        result = verify_evidence_bundle(bundle)
        assert result.ok
        assert result.schema_version == BUNDLE_SCHEMA_CURRENT
        assert result.legacy is False

    def test_tampered_hash_fails(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        manifest_path = bundle / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_key = next(iter(manifest["artifact_hashes"]))
        manifest["artifact_hashes"][first_key] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = verify_evidence_bundle(bundle)
        assert result.ok is False
        assert any(item.code == "hash_mismatch" for item in result.failures)

    def test_absolute_path_fails(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        manifest_path = bundle / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = next(iter(manifest["artifact_hashes"].values()))
        manifest["artifact_hashes"]["C:/escape.txt"] = digest
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = verify_evidence_bundle(bundle)
        assert result.ok is False
        assert any(item.code == "artifact_path_invalid" for item in result.failures)

    def test_legacy_bundle_without_reason_warns(self, tmp_path: Path) -> None:
        bundle = _write_bundle(tmp_path)
        manifest_path = bundle / MANIFEST_NAME
        result_path = bundle / RESULT_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        transition = json.loads(result_path.read_text(encoding="utf-8"))
        del manifest["schema_version"]
        del manifest["promotion_reason"]
        del transition["promotion_reason"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result_path.write_text(
            json.dumps(transition, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Re-hash trust_transition after mutation so hash checks pass.
        from ctpf.kernel import sha256_file

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_hashes"][RESULT_NAME] = sha256_file(result_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verified = verify_evidence_bundle(bundle)
        assert verified.ok
        assert verified.legacy is True
        assert any(item.code == "legacy_reason_absent" for item in verified.warnings)

    def test_missing_directory_fails_closed(self, tmp_path: Path) -> None:
        verified = verify_evidence_bundle(tmp_path / "missing")
        assert verified.ok is False
        assert verified.failures[0].code == "evidence_missing"
