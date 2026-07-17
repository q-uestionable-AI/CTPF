"""Tests for the driven multi-model cascade matrix."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctpf import experiment
from ctpf.automation.contracts import DataEgressClass
from ctpf.cli import app as root_app
from ctpf.driven_inference import OpenAICompatibleTargetProfile
from ctpf.kernel import EvidenceBundle, PromotionReason, PromotionResult, TrustTransition

_NO_COLOR_ENV = {"NO_COLOR": "1", "FORCE_COLOR": None, "TERM": "dumb"}
_cli_runner = CliRunner(env=_NO_COLOR_ENV)


def _profile(target_id: str, model: str) -> OpenAICompatibleTargetProfile:
    return OpenAICompatibleTargetProfile(
        target_id=target_id,
        name=f"Remote {model}",
        endpoint="https://models.example.test/v1",
        model=model,
        credential_name="remote-models",
        max_tokens=512,
        temperature=0.0,
        seed=0,
        max_input_tokens=1_024,
        data_egress_class=DataEgressClass.PACKAGED_SYNTHETIC_REMOTE,
        retention_acknowledged=True,
        residual_cost_acknowledged=True,
    )


def _transition(result: PromotionResult) -> TrustTransition:
    reason = {
        PromotionResult.CONFIRMED: PromotionReason.CONFIRMED_CLEAN_BASELINE_PROMOTED_TREATMENT,
        PromotionResult.NOT_OBSERVED: PromotionReason.NOT_OBSERVED_CLEAN_BASELINE_CLEAN_TREATMENT,
        PromotionResult.INCONCLUSIVE: PromotionReason.INCONCLUSIVE_INVOCATION_EFFECT_MISMATCH,
    }[result]
    return TrustTransition(
        source_event="controlled fixture response",
        source_trust_label="untrusted tool output",
        intended_audience="model reasoning only",
        destination_capability="apply_change",
        authority_required="privileged action",
        user_approved_scope="inspect only",
        observed_influence="test observation",
        policy_checkpoint="none",
        observed_tool_invocation=None,
        observed_tool_arguments=None,
        external_effect=None,
        promotion_result=result,
        promotion_reason=reason,
    )


def _completed_result(
    options: experiment.CascadeExperimentOptions,
    series_id: str,
) -> experiment.CascadeExperimentResult:
    root = options.output_root / series_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "run-manifest.json").write_text("{}\n", encoding="utf-8")
    bundle_root = root / "evidence" / "bundle-v1"
    bundle_root.mkdir(parents=True)
    manifest_path = bundle_root / "manifest.json"
    result_path = bundle_root / "trust_transition.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    result_path.write_text("{}\n", encoding="utf-8")
    bundle = EvidenceBundle(bundle_root, manifest_path, result_path)
    return experiment.CascadeExperimentResult(
        root,
        bundle,
        _transition(PromotionResult.INCONCLUSIVE),
        _transition(PromotionResult.CONFIRMED),
    )


def _patch_profiles(
    monkeypatch: pytest.MonkeyPatch,
    profiles: tuple[OpenAICompatibleTargetProfile, ...],
) -> None:
    by_id = {profile.target_id: profile for profile in profiles}

    def _load(reference: str, *, db_path: Path | None = None) -> OpenAICompatibleTargetProfile:
        del db_path
        return by_id[reference]

    monkeypatch.setattr(experiment, "load_openai_target_profile", _load)


@pytest.mark.asyncio
async def test_matrix_runs_balanced_trials_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profiles = (
        _profile("aaaaaaaa11111111", "model-a"),
        _profile("bbbbbbbb22222222", "model-b"),
    )
    _patch_profiles(monkeypatch, profiles)
    calls: list[tuple[str, tuple[str, ...]]] = []
    active = 0
    max_active = 0

    async def _run(
        options: experiment.CascadeExperimentOptions,
        *,
        operator: experiment._Operator | None = None,
        series_id: str | None = None,
        condition_order: tuple[experiment._Condition, ...] | None = None,
    ) -> experiment.CascadeExperimentResult:
        nonlocal active, max_active
        assert operator is None
        assert options.target is not None and series_id is not None and condition_order is not None
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        calls.append((options.target, tuple(condition.value for condition in condition_order)))
        result = _completed_result(options, series_id)
        active -= 1
        return result

    monkeypatch.setattr(experiment, "run_cascade_memo", _run)
    options = experiment.CascadeMatrixOptions(
        tuple(profile.target_id for profile in profiles),
        3,
        tmp_path / "output",
    )

    result = await experiment.run_cascade_matrix(options)

    expected_orders = [
        ("baseline", "manipulated", "hardened"),
        ("manipulated", "hardened", "baseline"),
        ("hardened", "baseline", "manipulated"),
    ]
    assert max_active == 1
    assert calls == [
        (profile.target_id, order) for profile in profiles for order in expected_orders
    ]
    assert len(result.trials) == 6
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["study_type"] == "exploratory_repeated_observations"
    assert manifest["execution_order"] == "sequential"
    assert manifest["retry_policy"] == "none"
    assert [target["model"] for target in manifest["targets"]] == ["model-a", "model-b"]
    assert [trial["status"] for trial in manifest["trials"]] == ["complete"] * 6
    assert {trial["primary_result"] for trial in manifest["trials"]} == {"INCONCLUSIVE"}
    assert {trial["hardened_result"] for trial in manifest["trials"]} == {"CONFIRMED"}
    assert all((result.root / trial["bundle"]).is_dir() for trial in manifest["trials"])


@pytest.mark.asyncio
async def test_matrix_preserves_failure_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profiles = (
        _profile("aaaaaaaa11111111", "model-a"),
        _profile("bbbbbbbb22222222", "model-b"),
    )
    _patch_profiles(monkeypatch, profiles)
    calls = 0

    async def _run(
        options: experiment.CascadeExperimentOptions,
        *,
        operator: experiment._Operator | None = None,
        series_id: str | None = None,
        condition_order: tuple[experiment._Condition, ...] | None = None,
    ) -> experiment.CascadeExperimentResult:
        nonlocal calls
        del operator, condition_order
        assert series_id is not None
        calls += 1
        if calls == 2:
            root = options.output_root / series_id
            root.mkdir(parents=True)
            (root / "run-manifest.json").write_text(
                '{"status":"failed"}\n',
                encoding="utf-8",
            )
            raise RuntimeError("provider unavailable")
        return _completed_result(options, series_id)

    monkeypatch.setattr(experiment, "run_cascade_memo", _run)
    output_root = tmp_path / "output"
    options = experiment.CascadeMatrixOptions(
        tuple(profile.target_id for profile in profiles),
        3,
        output_root,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await experiment.run_cascade_matrix(options)

    assert calls == 2
    matrix_roots = list(output_root.glob("cascade-memo-matrix-*"))
    assert len(matrix_roots) == 1
    manifest = json.loads(
        (matrix_roots[0] / experiment._MATRIX_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"] == "RuntimeError: provider unavailable"
    assert [trial["status"] for trial in manifest["trials"]] == [
        "complete",
        "failed",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert (matrix_roots[0] / manifest["trials"][1]["run_manifest"]).is_file()


@pytest.mark.parametrize(
    ("targets", "trials", "message"),
    [
        (("aaaaaaaa",), 3, "at least two"),
        (("aaaaaaaa", "bbbbbbbb"), 2, "3-5 trials"),
        (("aaaaaaaa", "bbbbbbbb"), 6, "3-5 trials"),
    ],
)
def test_matrix_rejects_out_of_scope_dimensions(
    targets: tuple[str, ...],
    trials: int,
    message: str,
    tmp_path: Path,
) -> None:
    options = experiment.CascadeMatrixOptions(targets, trials, tmp_path / "output")

    with pytest.raises(experiment.ExperimentError, match=message):
        experiment._prepare_matrix(options)


def test_matrix_rejects_duplicate_resolved_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile("aaaaaaaa11111111", "model-a")
    monkeypatch.setattr(
        experiment,
        "load_openai_target_profile",
        lambda _reference, *, db_path=None: profile,
    )
    options = experiment.CascadeMatrixOptions(
        ("aaaaaaaa", "bbbbbbbb"),
        3,
        tmp_path / "output",
    )

    with pytest.raises(experiment.ExperimentError, match="distinct target profiles"):
        experiment._prepare_matrix(options)


def test_cli_collects_repeatable_targets_for_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[experiment.CascadeMatrixOptions] = []

    async def _run(options: experiment.CascadeMatrixOptions) -> experiment.CascadeMatrixResult:
        captured.append(options)
        root = tmp_path / "matrix"
        return experiment.CascadeMatrixResult(root, root / "series-manifest.json", ())

    monkeypatch.setattr(experiment, "run_cascade_matrix", _run)
    result = _cli_runner.invoke(
        root_app,
        [
            "experiment",
            "run",
            "cascade-memo",
            "--target",
            "aaaaaaaa",
            "--target",
            "bbbbbbbb",
            "--trials",
            "3",
            "--output-root",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0
    assert captured[0].targets == ("aaaaaaaa", "bbbbbbbb")
    assert captured[0].trials_per_model == 3
    assert "Matrix complete" in result.output
