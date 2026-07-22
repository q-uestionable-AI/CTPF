"""First-class evidence-bundle verifier for internal consistency checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from ctpf.kernel.slice import (
    ARTIFACTS_DIRNAME,
    BUNDLE_SCHEMA_CURRENT,
    BUNDLE_SCHEMA_LEGACY,
    MANIFEST_NAME,
    MANIPULATED_SINK_NAME,
    REQUIRED_CASCADE_SPLIT_TRACE_NAMES,
    REQUIRED_CONFIRMED_CASCADE_ARTIFACTS,
    REQUIRED_TRACE_NAMES,
    RESULT_NAME,
    PromotionReason,
    PromotionResult,
    sha256_file,
)

_HASH_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_SCENARIO_CASCADE_MEMO = "cascade_memo"
_SCENARIO_PATTERN2 = "pattern2"
_PATTERN3_CONDITIONS = ("baseline", "opportunity", "hardened_opportunity")
_PATTERN3_REQUIRED_NAMES = ("authority.json", "observation.json", "session.json")
_PATTERN3_CONFIRMED_SINK = "opportunity/sink.json"
_ARTIFACT_REF_KEYS = frozenset({"authority_artifact", "memo_path", "sink_path"})


class VerificationStatus(StrEnum):
    """Top-level verifier outcome."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationIssue:
    """One structured verifier failure or warning."""

    code: str
    message: str


@dataclass(frozen=True)
class VerificationResult:
    """Typed bundle verification outcome that never raises for malformed input."""

    status: VerificationStatus
    schema_version: str | None
    legacy: bool
    failures: tuple[VerificationIssue, ...] = ()
    warnings: tuple[VerificationIssue, ...] = ()
    limitations: tuple[str, ...] = (
        "internal_consistency_only",
        "not_independent_authenticity",
        "not_scientific_validity",
        "not_generality",
    )

    @property
    def ok(self) -> bool:
        """Return whether verification passed."""
        return self.status == VerificationStatus.PASSED

    def to_payload(self) -> dict[str, Any]:
        """Serialize the verification result for machine consumers."""
        return {
            "failures": [{"code": item.code, "message": item.message} for item in self.failures],
            "legacy": self.legacy,
            "limitations": list(self.limitations),
            "ok": self.ok,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "warnings": [{"code": item.code, "message": item.message} for item in self.warnings],
        }


def verify_evidence_bundle(bundle_dir: Path | str) -> VerificationResult:
    """Verify one evidence bundle directory for internal consistency.

    Args:
        bundle_dir: Path to a written evidence bundle root.

    Returns:
        Typed verification result. Malformed external data becomes failures or
        warnings rather than raised exceptions.
    """
    root = Path(bundle_dir)
    failures: list[VerificationIssue] = []
    warnings: list[VerificationIssue] = []
    if not root.is_dir():
        return _failed(
            None,
            False,
            VerificationIssue("evidence_missing", "bundle directory is missing"),
        )
    manifest = _load_json_object(root / MANIFEST_NAME, "manifest", failures)
    if manifest is None:
        return _failed(None, False, *failures)
    transition = _load_json_object(root / RESULT_NAME, "trust_transition", failures)
    if transition is None:
        return _failed(None, False, *failures)

    schema_version, legacy, schema_failures, schema_warnings = _resolve_schema(manifest)
    failures.extend(schema_failures)
    warnings.extend(schema_warnings)

    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        failures.append(
            VerificationIssue("manifest_invalid", "artifact_hashes must be a non-empty object")
        )
        return _failed(schema_version, legacy, *failures, warnings=warnings)

    failures.extend(_validate_hash_map(root, hashes))
    if schema_version == BUNDLE_SCHEMA_CURRENT:
        failures.extend(_validate_required_artifacts(manifest, transition, hashes))
    failures.extend(_validate_result_agreement(manifest, transition, legacy))
    if legacy and "promotion_reason" not in transition:
        warnings.append(
            VerificationIssue(
                "legacy_reason_absent",
                "legacy trust transition has no promotion_reason; not invented",
            )
        )
    if failures:
        return _failed(schema_version, legacy, *failures, warnings=warnings)
    return VerificationResult(
        status=VerificationStatus.PASSED,
        schema_version=schema_version,
        legacy=legacy,
        warnings=tuple(warnings),
    )


def _resolve_schema(
    manifest: dict[str, Any],
) -> tuple[str, bool, list[VerificationIssue], list[VerificationIssue]]:
    """Normalize schema version and legacy classification."""
    if "schema_version" not in manifest:
        return (
            BUNDLE_SCHEMA_LEGACY,
            True,
            [],
            [
                VerificationIssue(
                    "legacy_bundle",
                    "bundle lacks explicit schema_version; verified as legacy",
                )
            ],
        )
    value = manifest.get("schema_version")
    if not isinstance(value, str) or not value.strip():
        return (
            BUNDLE_SCHEMA_LEGACY,
            True,
            [VerificationIssue("manifest_invalid", "manifest schema_version is invalid")],
            [],
        )
    version = value.strip()
    if version == BUNDLE_SCHEMA_CURRENT:
        return version, False, [], []
    if version == BUNDLE_SCHEMA_LEGACY:
        return version, True, [], []
    return (
        version,
        False,
        [
            VerificationIssue(
                "manifest_invalid",
                f"unsupported bundle schema_version: {version!r}",
            )
        ],
        [],
    )


def _validate_hash_map(root: Path, hashes: dict[str, Any]) -> list[VerificationIssue]:
    """Validate relative artifact paths and digests against the bundle tree."""
    failures: list[VerificationIssue] = []
    seen_casefold: set[str] = set()
    root_resolved = root.resolve(strict=False)
    for raw_name, raw_digest in hashes.items():
        issue = _validate_one_artifact(root, root_resolved, raw_name, raw_digest, seen_casefold)
        if issue is not None:
            failures.append(issue)
    return failures


def _validate_required_artifacts(
    manifest: dict[str, Any],
    transition: dict[str, Any],
    hashes: dict[str, Any],
) -> list[VerificationIssue]:
    """Validate current-schema required artifacts are declared in the hash map."""
    declared = {name for name in hashes if isinstance(name, str)}
    scenario_required = _scenario_required_artifacts(manifest)
    if isinstance(scenario_required, VerificationIssue):
        return [scenario_required]
    required = {RESULT_NAME}
    required.update(scenario_required)
    required.update(_artifact_refs(manifest))
    required.update(_artifact_refs(transition))
    return _missing_artifact_failures(required, declared)


def _scenario_required_artifacts(manifest: dict[str, Any]) -> set[str] | VerificationIssue:
    """Return required bundle-relative artifact paths for a current manifest."""
    scenario_id = _scenario_id(manifest)
    if scenario_id == _SCENARIO_CASCADE_MEMO:
        return _cascade_required_artifacts(manifest)
    if scenario_id == _SCENARIO_PATTERN2:
        return _pattern2_required_artifacts(manifest)
    if scenario_id is None and _is_pattern3_manifest(manifest):
        return _pattern3_required_artifacts(manifest)
    return VerificationIssue(
        "manifest_invalid",
        "current bundle has an unsupported or unidentifiable scenario",
    )


def _pattern2_required_artifacts(manifest: dict[str, Any]) -> set[str]:
    """Return Pattern 2 artifact declarations required by the bundle writer."""
    required = _artifact_paths(REQUIRED_TRACE_NAMES)
    if manifest.get("promotion_result") == PromotionResult.CONFIRMED.value:
        required.add(_artifact_path(MANIPULATED_SINK_NAME))
    return required


def _cascade_required_artifacts(manifest: dict[str, Any]) -> set[str]:
    """Return cascade artifact declarations required by the bundle writer."""
    required = _artifact_paths(REQUIRED_CASCADE_SPLIT_TRACE_NAMES)
    if manifest.get("promotion_result") == PromotionResult.CONFIRMED.value:
        required.update(_artifact_paths(REQUIRED_CONFIRMED_CASCADE_ARTIFACTS))
    return required


def _pattern3_required_artifacts(manifest: dict[str, Any]) -> set[str]:
    """Return Pattern 3 artifact declarations required by the bundle writer."""
    required = {
        _artifact_path(f"{condition}/{name}")
        for condition in _PATTERN3_CONDITIONS
        for name in _PATTERN3_REQUIRED_NAMES
    }
    if manifest.get("promotion_result") == PromotionResult.CONFIRMED.value:
        required.add(_artifact_path(_PATTERN3_CONFIRMED_SINK))
    return required


def _artifact_refs(value: Any) -> set[str]:
    """Return artifact refs from known reference-bearing fields."""
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _ARTIFACT_REF_KEYS and isinstance(item, str):
                refs.update(_artifact_ref(item))
            refs.update(_artifact_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_artifact_refs(item))
    return refs


def _artifact_ref(value: str) -> set[str]:
    """Return a singleton artifact ref when the string is bundle-relative."""
    if value.startswith(f"{ARTIFACTS_DIRNAME}/"):
        return {value}
    return set()


def _missing_artifact_failures(
    required: set[str],
    declared: set[str],
) -> list[VerificationIssue]:
    """Build missing-artifact failures for required undeclared paths."""
    return [
        VerificationIssue("artifact_missing", f"required artifact is not declared: {name}")
        for name in sorted(required.difference(declared))
    ]


def _declared_artifact_paths(manifest: dict[str, Any]) -> set[str]:
    """Return string artifact paths declared by the manifest hash map."""
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        return set()
    return {name for name in hashes if isinstance(name, str)}


def _artifact_paths(names: frozenset[str]) -> set[str]:
    """Prefix logical artifact names with the bundle artifact directory."""
    return {_artifact_path(name) for name in names}


def _artifact_path(name: str) -> str:
    """Return a bundle-relative artifact path for a logical artifact name."""
    return f"{ARTIFACTS_DIRNAME}/{name}"


def _is_pattern3_manifest(manifest: dict[str, Any]) -> bool:
    """Return whether a current manifest has the Pattern 3 condition shape."""
    conditions = manifest.get("conditions")
    if not isinstance(conditions, dict):
        return False
    return set(_PATTERN3_CONDITIONS).issubset(conditions)


def _scenario_id(manifest: dict[str, Any]) -> str | None:
    """Return a manifest scenario_id when present."""
    scenario = manifest.get("scenario")
    if not isinstance(scenario, dict):
        return None
    value = scenario.get("scenario_id")
    return value if isinstance(value, str) else None


def _validate_one_artifact(
    root: Path,
    root_resolved: Path,
    raw_name: Any,
    raw_digest: Any,
    seen_casefold: set[str],
) -> VerificationIssue | None:
    """Validate one declared artifact path and digest."""
    declared = _declared_artifact_name(raw_name, seen_casefold)
    if isinstance(declared, VerificationIssue):
        return declared
    if not isinstance(raw_digest, str) or not _is_sha256_hex(raw_digest):
        return VerificationIssue("hash_mismatch", f"malformed digest for {declared}")
    artifact = root.joinpath(*PurePosixPath(declared).parts)
    return _artifact_content_issue(artifact, root, root_resolved, declared, raw_digest)


def _declared_artifact_name(
    raw_name: Any,
    seen_casefold: set[str],
) -> str | VerificationIssue:
    """Normalize one artifact name or return a path validation issue."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        return VerificationIssue("artifact_path_invalid", "artifact path is empty")
    path_issue = _relative_path_issue(raw_name)
    if path_issue is not None:
        return path_issue
    folded = raw_name.casefold()
    if folded in seen_casefold:
        return VerificationIssue(
            "artifact_path_invalid",
            f"duplicate case-folded artifact path: {raw_name}",
        )
    seen_casefold.add(folded)
    return raw_name


def _artifact_content_issue(
    artifact: Path,
    root: Path,
    root_resolved: Path,
    declared: str,
    raw_digest: str,
) -> VerificationIssue | None:
    """Validate resolved artifact identity and digest."""
    try:
        resolved = artifact.resolve(strict=False)
    except OSError:
        return VerificationIssue("artifact_path_invalid", f"unresolvable artifact: {declared}")
    if not resolved.is_relative_to(root_resolved):
        return VerificationIssue("artifact_path_invalid", f"artifact escapes bundle: {declared}")
    if _escapes_via_symlink(artifact, root):
        return VerificationIssue("artifact_path_invalid", f"artifact link escape: {declared}")
    if not artifact.is_file():
        return VerificationIssue("artifact_missing", f"missing artifact: {declared}")
    if sha256_file(artifact) != raw_digest.lower():
        return VerificationIssue("hash_mismatch", f"digest mismatch: {declared}")
    return None


def _validate_result_agreement(
    manifest: dict[str, Any],
    transition: dict[str, Any],
    legacy: bool,
) -> list[VerificationIssue]:
    """Require manifest and trust-transition mechanical fields to agree."""
    failures: list[VerificationIssue] = []
    manifest_result = manifest.get("promotion_result")
    transition_result = transition.get("promotion_result")
    if not _is_promotion_result(manifest_result) or not _is_promotion_result(transition_result):
        failures.append(
            VerificationIssue("result_manifest_mismatch", "promotion_result is missing or invalid")
        )
    elif manifest_result != transition_result:
        failures.append(
            VerificationIssue(
                "result_manifest_mismatch",
                "manifest and trust_transition promotion_result disagree",
            )
        )
    manifest_reason = manifest.get("promotion_reason")
    transition_reason = transition.get("promotion_reason")
    if legacy and manifest_reason is None and transition_reason is None:
        return failures
    if not _is_promotion_reason(manifest_reason) or not _is_promotion_reason(transition_reason):
        failures.append(
            VerificationIssue(
                "result_manifest_mismatch",
                "promotion_reason is missing or invalid for current schema",
            )
        )
    elif manifest_reason != transition_reason:
        failures.append(
            VerificationIssue(
                "result_manifest_mismatch",
                "manifest and trust_transition promotion_reason disagree",
            )
        )
    return failures


def _load_json_object(
    path: Path,
    label: str,
    failures: list[VerificationIssue],
) -> dict[str, Any] | None:
    """Load one JSON object file, recording failures instead of raising."""
    if not path.is_file():
        failures.append(VerificationIssue("evidence_missing", f"{label} file is missing"))
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        failures.append(VerificationIssue("evidence_missing", f"{label} file is unreadable"))
        return None
    if not text.strip():
        failures.append(VerificationIssue("manifest_invalid", f"{label} file is empty"))
        return None
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        failures.append(VerificationIssue("manifest_invalid", f"{label} is not valid JSON"))
        return None
    if not isinstance(parsed, dict):
        failures.append(VerificationIssue("manifest_invalid", f"{label} must be a JSON object"))
        return None
    return parsed


def _relative_path_issue(raw_name: str) -> VerificationIssue | None:
    """Return an issue when a declared artifact path is unsafe."""
    posix = PurePosixPath(raw_name)
    if posix.is_absolute() or raw_name.startswith(("/", "\\")):
        return VerificationIssue("artifact_path_invalid", f"absolute artifact path: {raw_name}")
    parts = posix.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return VerificationIssue("artifact_path_invalid", f"unsafe artifact path: {raw_name}")
    if any(":" in part for part in parts):
        return VerificationIssue("artifact_path_invalid", f"drive-like artifact path: {raw_name}")
    return None


def _escapes_via_symlink(path: Path, root: Path) -> bool:
    """Return whether the artifact path uses a symlink under the bundle root."""
    try:
        cursor = path if path.is_symlink() else path.parent
        boundary = root.resolve(strict=False)
    except OSError:
        return True
    while True:
        if cursor.is_symlink():
            return True
        if cursor.resolve(strict=False) == boundary:
            return False
        if cursor.parent == cursor:
            return False
        cursor = cursor.parent


def _is_sha256_hex(value: str) -> bool:
    lowered = value.lower()
    return len(lowered) == _HASH_LENGTH and all(char in _HEX_DIGITS for char in lowered)


def _is_promotion_result(value: Any) -> bool:
    return isinstance(value, str) and value in {item.value for item in PromotionResult}


def _is_promotion_reason(value: Any) -> bool:
    return isinstance(value, str) and value in {item.value for item in PromotionReason}


def _failed(
    schema_version: str | None,
    legacy: bool,
    *failures: VerificationIssue,
    warnings: list[VerificationIssue] | None = None,
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.FAILED,
        schema_version=schema_version,
        legacy=legacy,
        failures=tuple(failures),
        warnings=tuple(warnings or ()),
    )
