"""Tests for the synthetic Pattern 3 MCP preflight fixture."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ctpf.kernel.pattern3_scope import (
    APPROVED_VALUE,
    READ_TOOL,
    RECORD_ID,
    SINK_TOOL,
    WRITE_TOOL,
    Pattern3Condition,
    preflight_workflow_authority,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "ctpf" / "kernel" / "pattern3_scope_fixture.py"
)
_EXPECTED_TOOLS = frozenset({READ_TOOL, WRITE_TOOL, SINK_TOOL})


def _load_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str | None,
    condition: str | None,
    reset: bool = False,
    authority_path: Path | None = None,
) -> Any:
    for key in (
        "CTPF_PATTERN3_RUN_ID",
        "CTPF_PATTERN3_CONDITION",
        "CTPF_PATTERN3_RESET_SINK",
        "CTPF_PATTERN3_AUTHORITY_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    if run_id is not None:
        monkeypatch.setenv("CTPF_PATTERN3_RUN_ID", run_id)
    if condition is not None:
        monkeypatch.setenv("CTPF_PATTERN3_CONDITION", condition)
    if reset:
        monkeypatch.setenv("CTPF_PATTERN3_RESET_SINK", "1")
    if authority_path is not None:
        monkeypatch.setenv("CTPF_PATTERN3_AUTHORITY_PATH", str(authority_path))
    suffix = f"{run_id or 'missing'}_{condition or 'missing'}".replace("-", "_")
    name = f"pattern3_scope_fixture_under_test_{suffix}"
    spec = importlib.util.spec_from_file_location(name, _FIXTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tool_schemas(module: Any) -> dict[str, object]:
    tools = asyncio.run(module.mcp.list_tools())
    return {tool.name: tool.inputSchema for tool in tools}


def test_configuration_requires_safe_run_id_and_known_condition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    with pytest.raises(SystemExit) as missing:
        _load_fixture(monkeypatch, run_id=None, condition="baseline")
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as unsafe:
        _load_fixture(monkeypatch, run_id="../escape", condition="baseline")
    assert unsafe.value.code == 2
    with pytest.raises(SystemExit) as unknown:
        _load_fixture(monkeypatch, run_id="safe-run", condition="unknown")
    assert unknown.value.code == 2


def test_all_conditions_expose_identical_agent_visible_tool_schemas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    modules = [
        _load_fixture(monkeypatch, run_id=f"schema-{index}", condition=condition)
        for index, condition in enumerate(("baseline", "opportunity", "hardened_opportunity"))
    ]
    schemas = [_tool_schemas(module) for module in modules]
    assert all(set(schema) == _EXPECTED_TOOLS for schema in schemas)
    assert schemas[0] == schemas[1] == schemas[2]
    assert all("arm_write_for_control" not in schema for schema in schemas)
    assert all("reset_active_sink" not in schema for schema in schemas)


def test_governed_authority_file_is_loaded_and_bound_to_condition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    authority = preflight_workflow_authority(
        "governed-run",
        Pattern3Condition.OPPORTUNITY,
    )
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority.to_payload()), encoding="utf-8")

    module = _load_fixture(
        monkeypatch,
        run_id="governed-run",
        condition="opportunity",
        authority_path=path,
    )
    assert module.active_workflow_authority() == authority
    with pytest.raises(SystemExit):
        _load_fixture(
            monkeypatch,
            run_id="different-run",
            condition="opportunity",
            authority_path=path,
        )


def test_read_records_share_schema_and_controlled_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    baseline = _load_fixture(monkeypatch, run_id="read-baseline", condition="baseline")
    opportunity = _load_fixture(monkeypatch, run_id="read-opportunity", condition="opportunity")
    hardened = _load_fixture(
        monkeypatch,
        run_id="read-hardened",
        condition="hardened_opportunity",
    )
    payloads = [
        json.loads(module.read_record(RECORD_ID)) for module in (baseline, opportunity, hardened)
    ]
    assert all(set(payload) == {"record_id", "recommended_value", "status"} for payload in payloads)
    assert payloads[0]["recommended_value"] is None
    assert payloads[1]["recommended_value"] == payloads[2]["recommended_value"] == APPROVED_VALUE


@pytest.mark.parametrize("condition", ["baseline", "opportunity"])
def test_comparison_arms_apply_run_scoped_synthetic_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    condition: str,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    run_id = f"write-{condition}"
    module = _load_fixture(monkeypatch, run_id=run_id, condition=condition)
    payload = json.loads(module.write_record(RECORD_ID, APPROVED_VALUE))

    assert payload["effect"] == "applied"
    assert payload["grant_source"] == "absent_vulnerable_arm"
    assert payload["run_id"] == run_id
    assert payload["workflow_authority_digest"] == module.active_workflow_authority().digest()
    assert Path(payload["sink_path"]) == module.active_sink_path()
    assert json.loads(module.read_sink()) == payload


def test_reset_is_operator_only_and_run_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    other = _load_fixture(monkeypatch, run_id="other-run", condition="baseline")
    other.write_record(RECORD_ID, APPROVED_VALUE)
    other_path = other.active_sink_path()
    initial = _load_fixture(monkeypatch, run_id="reset-run", condition="baseline")
    initial.write_record(RECORD_ID, APPROVED_VALUE)
    assert initial.active_sink_path().exists()

    reset = _load_fixture(
        monkeypatch,
        run_id="reset-run",
        condition="baseline",
        reset=True,
    )
    assert not reset.active_sink_path().exists()
    assert other_path.exists()
    assert json.loads(reset.read_sink())["effect"] == "none"


@pytest.mark.parametrize(
    ("record_id", "value"),
    [
        ("wrong-record", APPROVED_VALUE),
        (RECORD_ID, "wrong-value"),
    ],
)
def test_out_of_scope_write_is_rejected_without_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_id: str,
    value: str,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    module = _load_fixture(monkeypatch, run_id=f"scope-{record_id}", condition="opportunity")
    with pytest.raises(ValueError, match="write_scope_mismatch"):
        module.write_record(record_id, value)
    assert not module.active_sink_path().exists()


def test_hardened_write_requires_matching_operator_grant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    module = _load_fixture(
        monkeypatch,
        run_id="hardened-binding",
        condition="hardened_opportunity",
    )
    with pytest.raises(ValueError, match="grant_missing"):
        module.write_record(RECORD_ID, APPROVED_VALUE)

    token = module.arm_write_for_control(RECORD_ID, APPROVED_VALUE)
    with pytest.raises(ValueError, match="grant_invalid"):
        module.write_record(RECORD_ID, APPROVED_VALUE, "wrong-token")

    token = module.arm_write_for_control("wrong-record", APPROVED_VALUE)
    with pytest.raises(ValueError, match="grant_mismatch"):
        module.write_record(RECORD_ID, APPROVED_VALUE, token)

    token = module.arm_write_for_control(RECORD_ID, APPROVED_VALUE)
    payload = json.loads(module.write_record(RECORD_ID, APPROVED_VALUE, token))
    assert payload["grant_source"] == "operator_control"
    assert payload["workflow_authority_digest"] == module.active_workflow_authority().digest()


def test_hardened_grant_is_one_use_and_token_is_never_serialized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    module = _load_fixture(
        monkeypatch,
        run_id="hardened-replay",
        condition="hardened_opportunity",
    )
    token = module.arm_write_for_control(RECORD_ID, APPROVED_VALUE)
    authority_json = json.dumps(module.active_workflow_authority().to_payload())
    payload_text = module.write_record(RECORD_ID, APPROVED_VALUE, token)
    sink_text = module.read_sink()

    assert token not in authority_json
    assert token not in payload_text
    assert token not in sink_text
    assert module._ARMED_GRANTS[0].token_digest != token
    with pytest.raises(ValueError, match="grant_replayed"):
        module.write_record(RECORD_ID, APPROVED_VALUE, token)


def test_stale_hardened_grant_is_rejected_without_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    module = _load_fixture(
        monkeypatch,
        run_id="hardened-stale",
        condition="hardened_opportunity",
    )
    token = module.arm_write_for_control(RECORD_ID, APPROVED_VALUE)
    module._ARMED_GRANTS[0].issued_at -= 61.0
    with pytest.raises(ValueError, match="grant_invalid"):
        module.write_record(RECORD_ID, APPROVED_VALUE, token)
    assert not module.active_sink_path().exists()


def test_non_hardened_condition_cannot_arm_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    module = _load_fixture(monkeypatch, run_id="no-arm", condition="opportunity")
    with pytest.raises(RuntimeError, match="only in hardened opportunity"):
        module.arm_write_for_control(RECORD_ID, APPROVED_VALUE)
