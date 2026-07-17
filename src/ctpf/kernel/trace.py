"""Defensive conversion of proxy session traces into CTPF observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctpf.kernel.pattern3_scope import (
    HardenedControlReason,
    Pattern3Condition,
    Pattern3Observation,
    Pattern3Scenario,
    WorkflowAuthority,
    read_fixture_digest,
    read_fixture_payload,
)
from ctpf.kernel.slice import (
    CascadeArmObservation,
    CascadeMemoScenario,
    ExternalEffect,
    Pattern2Scenario,
    RunObservation,
)

_CALL_TOOL_METHOD = "tools/call"
_CLIENT_TO_SERVER = "client_to_server"
_CONCLUSIVE_NO_EFFECT_REASONS = frozenset({"effect_not_applied", "sink_missing"})
_SERVER_TO_CLIENT = "server_to_client"


@dataclass(frozen=True)
class _TraceRecord:
    proxy_id: str
    correlated_id: str | None
    direction: str
    modified: bool
    payload: dict[str, Any]


@dataclass(frozen=True)
class _TraceData:
    records: tuple[_TraceRecord, ...]
    modified: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _ToolCall:
    proxy_id: str
    jsonrpc_id: str | int
    name: str
    arguments: dict[str, Any]


def parse_pattern2_trace(
    condition: str,
    trace_path: Path,
    *,
    sink_effect: ExternalEffect,
    expect_modified: bool,
    scenario: Pattern2Scenario | None = None,
) -> RunObservation:
    """Build one Pattern 2 observation from a completed proxy trace.

    Args:
        condition: Experimental condition label.
        trace_path: Proxy session containing the status read and optional action.
        sink_effect: Independently observed external-effect oracle result.
        expect_modified: Whether the ``read_status`` response must be modified.
        scenario: Pattern 2 tool names and authority pins.

    Returns:
        A conservative Pattern 2 observation.
    """
    scenario = scenario or Pattern2Scenario()
    trace = _load_trace(trace_path, "session")
    notes = list(trace.notes)
    calls = _tool_calls(trace.records, "session", notes)
    status = _select_call(calls, scenario.read_tool, "session_read_status", True, notes)
    status_content = _read_result_content(
        trace.records,
        status,
        "session_read_status",
        notes,
    )
    _require_json_object(status_content, "session_read_status_response", notes)
    _validate_expected_mutation(trace.records, status, expect_modified, notes)
    action = _select_call(calls, scenario.action_tool, "session_apply_change", False, notes)
    return RunObservation(
        condition=condition,
        tool_invocation=action.name if action else None,
        tool_arguments=action.arguments if action else None,
        external_effect=sink_effect,
        evidence_complete=not notes,
        evidence_notes=tuple(notes),
    )


def parse_pattern3_trace(
    condition: Pattern3Condition,
    trace_path: Path,
    *,
    authority: WorkflowAuthority,
    sink_effect: ExternalEffect,
    scenario: Pattern3Scenario,
) -> Pattern3Observation:
    """Build one Pattern 3 observation from a completed proxy trace."""
    trace = _load_trace(trace_path, "session")
    notes = list(trace.notes)
    calls = _tool_calls(trace.records, "session", notes)
    read_call = _select_call(
        calls,
        scenario.read_tool,
        "session_read_record",
        True,
        notes,
    )
    read_content = _read_result_content(
        trace.records,
        read_call,
        "session_read_record",
        notes,
    )
    _require_expected_json(
        read_content,
        read_fixture_payload(condition, scenario),
        "session_read_record_response",
        notes,
    )
    write_call = _select_call(
        calls,
        scenario.write_tool,
        "session_write_record",
        False,
        notes,
    )
    effect = _pattern3_effect(trace.records, write_call, sink_effect, notes)
    return Pattern3Observation(
        condition=condition,
        authority=authority,
        read_fixture_digest=read_fixture_digest(condition, scenario),
        tool_invocation=write_call.name if write_call else None,
        tool_arguments=write_call.arguments if write_call else None,
        external_effect=effect,
        evidence_complete=not notes,
        evidence_notes=tuple(notes),
    )


def parse_cascade_arm_traces(
    condition: str,
    session_a_trace: Path,
    session_b_trace: Path,
    *,
    memo_effect: ExternalEffect,
    sink_effect: ExternalEffect,
    scenario: CascadeMemoScenario | None = None,
) -> CascadeArmObservation:
    """Build one cascade observation from completed Session A and B traces.

    Malformed or ambiguous external input is recorded in ``evidence_notes``
    and makes the observation incomplete. Parser failures never become clean
    negative evidence.

    Args:
        condition: Experimental condition label.
        session_a_trace: Proxy session containing the hop-1 interaction.
        session_b_trace: Proxy session containing the hop-2 interaction.
        memo_effect: Independently observed memo-persistence oracle result.
        sink_effect: Independently observed privileged-effect oracle result.
        scenario: Cascade tool names and authority pins.

    Returns:
        A conservative cascade arm observation.
    """
    scenario = scenario or CascadeMemoScenario()
    session_a = _load_trace(session_a_trace, "session_a")
    session_b = _load_trace(session_b_trace, "session_b")
    notes = [*session_a.notes, *session_b.notes]
    calls_a = _tool_calls(session_a.records, "session_a", notes)
    calls_b = _tool_calls(session_b.records, "session_b", notes)
    inbox = _select_call(calls_a, scenario.inbox_tool, "session_a_read_inbox", True, notes)
    _read_result_content(session_a.records, inbox, "session_a_read_inbox", notes)
    persist = _select_call(calls_a, scenario.persist_tool, "session_a_write_memo", False, notes)
    memo_read = _select_call(calls_b, scenario.memo_tool, "session_b_read_memo", True, notes)
    action = _select_call(calls_b, scenario.action_tool, "session_b_apply_change", False, notes)
    read_content = _read_result_content(
        session_b.records,
        memo_read,
        "session_b_read_memo",
        notes,
    )
    return CascadeArmObservation(
        condition=condition,
        hop1_persist_invocation=persist.name if persist else None,
        hop1_persist_arguments=persist.arguments if persist else None,
        hop1_memo_effect=memo_effect,
        hop2_action_invocation=action.name if action else None,
        hop2_action_arguments=action.arguments if action else None,
        hop2_sink_effect=sink_effect,
        session_b_mutated=session_b.modified,
        hop2_memo_read_invocation=memo_read.name if memo_read else None,
        hop2_memo_read_content=read_content,
        evidence_complete=not notes,
        evidence_notes=tuple(notes),
    )


def _load_trace(path: Path, label: str) -> _TraceData:
    notes: list[str] = []
    payload, error = _read_trace_payload(path, label)
    if error is not None or payload is None:
        return _TraceData((), False, (error or f"{label}_trace_invalid_shape",))
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return _TraceData((), False, (f"{label}_trace_messages_invalid",))
    if not isinstance(payload.get("ended_at"), str) or not payload["ended_at"].strip():
        notes.append(f"{label}_trace_unfinished")
    records, modified = _parse_records(raw_messages, label, notes)
    return _TraceData(tuple(records), modified, tuple(notes))


def _read_trace_payload(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"{label}_trace_missing"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, f"{label}_trace_unreadable"
    if not text.strip():
        return None, f"{label}_trace_empty"
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        return None, f"{label}_trace_invalid_json"
    if not isinstance(payload, dict):
        return None, f"{label}_trace_invalid_shape"
    return payload, None


def _parse_records(
    raw_messages: list[Any],
    label: str,
    notes: list[str],
) -> tuple[list[_TraceRecord], bool]:
    records: list[_TraceRecord] = []
    modified = False
    for index, raw in enumerate(raw_messages):
        if isinstance(raw, dict) and raw.get("modified") is True:
            modified = True
        try:
            record = _parse_record(raw)
        except (TypeError, ValueError, KeyError):
            _add_note(notes, f"{label}_message_{index}_malformed")
            continue
        records.append(record)
    return records, modified


def _parse_record(raw: Any) -> _TraceRecord:
    if not isinstance(raw, dict):
        raise TypeError("trace message must be an object")
    proxy_id = raw.get("proxy_id")
    direction = raw.get("direction")
    modified = raw.get("modified")
    payload = raw.get("payload")
    correlated_id = raw.get("correlated_id")
    if not isinstance(proxy_id, str) or not proxy_id.strip():
        raise ValueError("trace message requires proxy_id")
    if direction not in {_CLIENT_TO_SERVER, _SERVER_TO_CLIENT}:
        raise ValueError("trace message has invalid direction")
    if not isinstance(modified, bool) or not isinstance(payload, dict):
        raise TypeError("trace message has invalid payload metadata")
    if correlated_id is not None and not isinstance(correlated_id, str):
        raise TypeError("trace message has invalid correlation id")
    if modified and not isinstance(raw.get("original_payload"), dict):
        raise TypeError("modified trace message requires original_payload")
    return _TraceRecord(proxy_id, correlated_id, direction, modified, payload)


def _tool_calls(
    records: tuple[_TraceRecord, ...],
    label: str,
    notes: list[str],
) -> list[_ToolCall]:
    calls: list[_ToolCall] = []
    for record in records:
        if record.direction != _CLIENT_TO_SERVER:
            continue
        if record.payload.get("method") != _CALL_TOOL_METHOD:
            continue
        params = record.payload.get("params")
        if not isinstance(params, dict):
            _add_note(notes, f"{label}_tool_call_malformed")
            continue
        name = params.get("name")
        arguments = params.get("arguments", {})
        jsonrpc_id = record.payload.get("id")
        if (
            not isinstance(name, str)
            or not isinstance(arguments, dict)
            or isinstance(jsonrpc_id, bool)
            or not isinstance(jsonrpc_id, (str, int))
        ):
            _add_note(notes, f"{label}_tool_call_malformed")
            continue
        calls.append(_ToolCall(record.proxy_id, jsonrpc_id, name, dict(arguments)))
    return calls


def _select_call(
    calls: list[_ToolCall],
    tool_name: str,
    note_prefix: str,
    required: bool,
    notes: list[str],
) -> _ToolCall | None:
    matching = [call for call in calls if call.name == tool_name]
    if not matching:
        if required:
            _add_note(notes, f"{note_prefix}_missing")
        return None
    if len(matching) > 1:
        _add_note(notes, f"{note_prefix}_ambiguous")
        return None
    return matching[0]


def _read_result_content(
    records: tuple[_TraceRecord, ...],
    read_call: _ToolCall | None,
    note_prefix: str,
    notes: list[str],
) -> str | None:
    if read_call is None:
        return None
    responses = [
        record
        for record in records
        if record.direction == _SERVER_TO_CLIENT and record.correlated_id == read_call.proxy_id
    ]
    if not responses:
        _add_note(notes, f"{note_prefix}_response_missing")
        return None
    if len(responses) > 1:
        _add_note(notes, f"{note_prefix}_response_ambiguous")
        return None
    if responses[0].payload.get("id") != read_call.jsonrpc_id:
        _add_note(notes, f"{note_prefix}_response_correlation_mismatch")
        return None
    content, error = _extract_result_content(responses[0].payload)
    if error is not None:
        _add_note(notes, f"{note_prefix}_response_{error}")
    return content


def _require_json_object(
    raw: str | None,
    note_prefix: str,
    notes: list[str],
) -> None:
    """Require one readable JSON object without raising on external input."""
    if raw is None or not raw.strip():
        return
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        _add_note(notes, f"{note_prefix}_invalid_json")
        return
    if not isinstance(parsed, dict):
        _add_note(notes, f"{note_prefix}_invalid_shape")


def _require_expected_json(
    raw: str | None,
    expected: dict[str, object],
    note_prefix: str,
    notes: list[str],
) -> None:
    if raw is None or not raw.strip():
        return
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        _add_note(notes, f"{note_prefix}_invalid_json")
        return
    if not isinstance(parsed, dict):
        _add_note(notes, f"{note_prefix}_invalid_shape")
    elif parsed != expected:
        _add_note(notes, f"{note_prefix}_mismatch")


def _pattern3_effect(
    records: tuple[_TraceRecord, ...],
    write_call: _ToolCall | None,
    sink_effect: ExternalEffect,
    notes: list[str],
) -> ExternalEffect:
    if (
        write_call is None
        or sink_effect.present
        or sink_effect.reason not in _CONCLUSIVE_NO_EFFECT_REASONS
    ):
        return sink_effect
    reason = _pattern3_write_error_reason(records, write_call, notes)
    if reason is None:
        return sink_effect
    return ExternalEffect(False, sink_effect.payload, sink_effect.sink_path, reason)


def _pattern3_write_error_reason(
    records: tuple[_TraceRecord, ...],
    write_call: _ToolCall,
    notes: list[str],
) -> str | None:
    responses = [
        record
        for record in records
        if record.direction == _SERVER_TO_CLIENT and record.correlated_id == write_call.proxy_id
    ]
    if len(responses) != 1 or responses[0].payload.get("id") != write_call.jsonrpc_id:
        _add_note(notes, "session_write_record_response_missing_or_ambiguous")
        return None
    result = responses[0].payload.get("result")
    if not isinstance(result, dict) or result.get("isError") is not True:
        return None
    content = result.get("content")
    texts = (
        [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if isinstance(content, list)
        else []
    )
    combined = "\n".join(item for item in texts if isinstance(item, str))
    allowed = {
        HardenedControlReason.BLOCKED_GRANT_MISSING.value.removeprefix("blocked_"),
        HardenedControlReason.BLOCKED_GRANT_INVALID.value.removeprefix("blocked_"),
        HardenedControlReason.BLOCKED_GRANT_MISMATCH.value.removeprefix("blocked_"),
        HardenedControlReason.BLOCKED_GRANT_REPLAYED.value.removeprefix("blocked_"),
    }
    reason = next((item for item in allowed if item in combined), None)
    if reason is None:
        _add_note(notes, "session_write_record_error_unclassified")
    return reason


def _validate_expected_mutation(
    records: tuple[_TraceRecord, ...],
    read_call: _ToolCall | None,
    expected: bool,
    notes: list[str],
) -> None:
    """Require the correlated read response to match the condition treatment."""
    if read_call is None:
        return
    responses = [
        record
        for record in records
        if record.direction == _SERVER_TO_CLIENT and record.correlated_id == read_call.proxy_id
    ]
    if len(responses) != 1:
        return
    if expected and not responses[0].modified:
        _add_note(notes, "session_read_status_mutation_missing")
    if not expected and responses[0].modified:
        _add_note(notes, "session_read_status_mutation_unexpected")


def _extract_result_content(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    if "error" in payload:
        return None, "error"
    result = payload.get("result")
    if not isinstance(result, dict):
        return None, "unreadable"
    is_error = result.get("isError", False)
    if is_error is True:
        return None, "tool_error"
    if is_error is not False:
        return None, "unreadable"
    return _content_from_result(result)


def _content_from_result(result: dict[str, Any]) -> tuple[str | None, str | None]:
    content = result.get("content")
    if isinstance(content, list):
        text_items = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if len(text_items) == 1 and isinstance(text_items[0], str):
            return text_items[0], None
        if text_items:
            return None, "ambiguous_content"
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return json.dumps(structured, sort_keys=True, separators=(",", ":")), None
    return None, "unreadable"


def _add_note(notes: list[str], note: str) -> None:
    if note not in notes:
        notes.append(note)
