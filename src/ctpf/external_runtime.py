"""External agent-runtime adapter for controlled CTPF experiments."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import urlparse

from ctpf.core.db import get_connection, get_target
from ctpf.core.models import Target
from ctpf.driven_inference import (
    DrivenInferenceError,
    OpenAICompatibleTargetProfile,
    load_openai_target_profile,
)
from ctpf.services.db_service import resolve_partial_id

_TARGET_TYPE = "agent-runtime"
_DRIVER_NAME = "claude-code-cli"
_MCP_SERVER_NAME = "ctpf-cascade"
_DEFAULT_TIMEOUT_SECONDS = 300
_MIN_TIMEOUT_SECONDS = 30
_MAX_TIMEOUT_SECONDS = 1800
_VERSION_TIMEOUT_SECONDS = 10.0
_PROCESS_TERMINATE_GRACE_SECONDS = 5.0
_MIN_ALWAYS_LOAD_VERSION = (2, 1, 121)
_MODEL_ALIASES = frozenset({"default", "fable", "haiku", "opus", "sonnet"})
_SECRET_METADATA_KEYS = frozenset(
    {
        "anthropic_api_key",
        "api_key",
        "auth_token",
        "credential",
        "oauth_token",
        "password",
        "secret",
        "secret_key",
    }
)
_RUNTIME_ENV_NAMES = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NODE_EXTRA_CA_CERTS",
        "PATH",
        "PATHEXT",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
    }
)


class ExternalRuntimeError(RuntimeError):
    """Raised when an external runtime cannot preserve experiment integrity."""


@dataclass(frozen=True)
class ClaudeCodeTargetProfile:
    """Validated non-secret settings for one Claude Code CLI runtime.

    Args:
        target_id: Persisted target identifier.
        name: Human-readable target name.
        executable: Resolved Claude Code CLI executable.
        model: Exact model identifier passed to the runtime.
        runtime_version: Exact Claude Code CLI version output.
        timeout_seconds: Maximum wall-clock duration of one fresh session.
    """

    target_id: str
    name: str
    executable: str
    model: str
    runtime_version: str
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS

    def evidence_payload(self) -> dict[str, Any]:
        """Return the complete external-runtime pin without credentials."""
        return {
            "target_id": self.target_id,
            "name": self.name,
            "target_type": _TARGET_TYPE,
            "driver": _DRIVER_NAME,
            "executable": self.executable,
            "runtime_version": self.runtime_version,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "authentication": "runtime-managed secure login",
            "environment_policy": "minimal non-secret allowlist",
        }


ExperimentTargetProfile: TypeAlias = OpenAICompatibleTargetProfile | ClaudeCodeTargetProfile


@dataclass(frozen=True)
class ClaudeCodeResult:
    """Summary of one completed fresh Claude Code conversation."""

    event_count: int
    transcript_path: Path


def load_experiment_target_profile(
    target_ref: str,
    *,
    db_path: Path | None = None,
) -> ExperimentTargetProfile:
    """Load a supported driven-inference or external-runtime target profile.

    Args:
        target_ref: Full or partial target ID (minimum eight characters).
        db_path: Optional database path override for tests.

    Returns:
        A validated profile for the target's declared experiment driver.

    Raises:
        ExternalRuntimeError: If the target or its driver is unsupported.
    """
    target = _load_target(target_ref, db_path)
    if target.type == "inference":
        try:
            return load_openai_target_profile(target_ref, db_path=db_path)
        except DrivenInferenceError as exc:
            raise ExternalRuntimeError(str(exc)) from exc
    if target.type != _TARGET_TYPE:
        raise ExternalRuntimeError(f"unsupported experiment target type: {target.type!r}")
    return _claude_profile_from_target(target)


class ClaudeCodeDriver:
    """Run one fresh Claude Code CLI session against the loopback MCP proxy."""

    def __init__(self, profile: ClaudeCodeTargetProfile) -> None:
        """Configure the driver with an already validated non-secret profile."""
        self._profile = profile

    async def run(
        self,
        prompt: str,
        mcp_endpoint: str,
        transcript_path: Path,
        *,
        mcp_server_name: str = _MCP_SERVER_NAME,
    ) -> ClaudeCodeResult:
        """Run Claude Code non-interactively and preserve its complete output.

        Args:
            prompt: Fixed scenario prompt for this fresh conversation.
            mcp_endpoint: Loopback proxy endpoint used for every MCP call.
            transcript_path: External artifact path for runtime evidence.
            mcp_server_name: Scenario-supplied MCP connection label.

        Returns:
            Completion summary for the fresh conversation.
        """
        server_name = _validated_server_name(mcp_server_name)
        mcp_config = _mcp_config(mcp_endpoint, server_name)
        command = _claude_command(self._profile, prompt, mcp_config, server_name)
        transcript = _new_transcript(self._profile, prompt, mcp_endpoint, command, mcp_config)
        _write_json(transcript_path, transcript)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=transcript_path.parent,
                env=_runtime_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_raw, stderr_raw = await _communicate(
                process,
                self._profile.timeout_seconds,
            )
            stdout = stdout_raw.decode("utf-8", errors="replace")
            stderr = stderr_raw.decode("utf-8", errors="replace")
            events, warnings = _parse_json_lines(stdout)
            _record_output(transcript, process.returncode, stdout, stderr, events, warnings)
            _validate_completion(process.returncode, events)
        except BaseException as exc:
            await _stop_process(process)
            transcript["status"] = "failed"
            transcript["error"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(transcript_path, transcript)
            raise
        transcript["status"] = "complete"
        _write_json(transcript_path, transcript)
        return ClaudeCodeResult(len(events), transcript_path)


async def _communicate(
    process: asyncio.subprocess.Process,
    timeout_seconds: int,
) -> tuple[bytes, bytes]:
    try:
        return await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise ExternalRuntimeError(
            f"Claude Code exceeded the {timeout_seconds}-second runtime limit"
        ) from exc


def _load_target(target_ref: str, db_path: Path | None) -> Target:
    reference = target_ref.strip()
    if len(reference) < 8:
        raise ExternalRuntimeError("target ID prefix must be at least 8 characters")
    try:
        with get_connection(db_path) as conn:
            target_id = resolve_partial_id(conn, "targets", reference)
            target = get_target(conn, target_id)
    except ValueError as exc:
        raise ExternalRuntimeError(str(exc)) from exc
    if target is None:
        raise ExternalRuntimeError(f"target not found: {reference}")
    return target


def _claude_profile_from_target(target: Target) -> ClaudeCodeTargetProfile:
    metadata = target.metadata
    if not isinstance(metadata, dict):
        raise ExternalRuntimeError("agent-runtime target metadata must be a JSON object")
    secret_keys = sorted(str(key) for key in metadata if str(key).lower() in _SECRET_METADATA_KEYS)
    if secret_keys:
        joined = ", ".join(secret_keys)
        raise ExternalRuntimeError(
            f"Claude Code targets must not declare secret metadata: {joined}"
        )
    driver = _required_string(metadata, "driver")
    if driver != _DRIVER_NAME:
        raise ExternalRuntimeError(f"unsupported external runtime driver: {driver!r}")
    model = _exact_model(metadata)
    executable, version = _inspect_claude_executable(target.uri)
    _require_always_load_support(version)
    return ClaudeCodeTargetProfile(
        target_id=target.id,
        name=target.name,
        executable=executable,
        model=model,
        runtime_version=version,
        timeout_seconds=_timeout_seconds(metadata),
    )


def _required_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExternalRuntimeError(f"agent-runtime metadata requires non-empty {key!r}")
    return value.strip()


def _exact_model(metadata: dict[str, Any]) -> str:
    model = _required_string(metadata, "model")
    if model.lower() in _MODEL_ALIASES:
        raise ExternalRuntimeError(
            "Claude Code target model must be an exact model ID, not an alias"
        )
    return model


def _timeout_seconds(metadata: dict[str, Any]) -> int:
    value = metadata.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ExternalRuntimeError("agent-runtime 'timeout_seconds' must be an integer") from exc
    if not _MIN_TIMEOUT_SECONDS <= parsed <= _MAX_TIMEOUT_SECONDS:
        raise ExternalRuntimeError(
            "agent-runtime 'timeout_seconds' must be between "
            f"{_MIN_TIMEOUT_SECONDS} and {_MAX_TIMEOUT_SECONDS}"
        )
    return parsed


def _inspect_claude_executable(raw: str | None) -> tuple[str, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ExternalRuntimeError("Claude Code target URI must name the CLI executable")
    executable = shutil.which(raw.strip())
    if executable is None:
        raise ExternalRuntimeError(f"Claude Code executable not found: {raw.strip()!r}")
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExternalRuntimeError(f"Claude Code version check failed: {exc}") from exc
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not version:
        raise ExternalRuntimeError("Claude Code version check returned no usable version")
    return executable, version


def _require_always_load_support(version: str) -> None:
    parsed = _parse_runtime_version(version)
    if parsed >= _MIN_ALWAYS_LOAD_VERSION:
        return
    minimum = ".".join(str(part) for part in _MIN_ALWAYS_LOAD_VERSION)
    raise ExternalRuntimeError(
        f"Claude Code {minimum} or later is required for deterministic MCP readiness; "
        f"found {version!r}"
    )


def _parse_runtime_version(version: str) -> tuple[int, int, int]:
    prefix = version.split(maxsplit=1)[0]
    parts = prefix.split(".")
    if len(parts) != 3:
        raise ExternalRuntimeError(f"Claude Code returned an invalid version: {version!r}")
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError as exc:
        raise ExternalRuntimeError(f"Claude Code returned an invalid version: {version!r}") from exc
    return major, minor, patch


def _mcp_config(mcp_endpoint: str, server_name: str) -> dict[str, Any]:
    endpoint = _validated_mcp_endpoint(mcp_endpoint)
    return {
        "mcpServers": {
            server_name: {
                "alwaysLoad": True,
                "type": "http",
                "url": endpoint,
            }
        }
    }


def _validated_server_name(server_name: str) -> str:
    value = server_name.strip()
    if not value or any(not (character.isalnum() or character in "-_") for character in value):
        raise ExternalRuntimeError("Claude Code MCP server name contains unsupported characters")
    return value


def _validated_mcp_endpoint(mcp_endpoint: str) -> str:
    try:
        parsed = urlparse(mcp_endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ExternalRuntimeError(f"invalid MCP endpoint: {exc}") from exc
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None:
        raise ExternalRuntimeError(
            "Claude Code MCP endpoint must use HTTP on 127.0.0.1 with a port"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ExternalRuntimeError("Claude Code MCP endpoint must not contain credentials")
    return mcp_endpoint


def _claude_command(
    profile: ClaudeCodeTargetProfile,
    prompt: str,
    mcp_config: dict[str, Any],
    server_name: str,
) -> list[str]:
    return [
        profile.executable,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        profile.model,
        "--no-session-persistence",
        "--mcp-config",
        json.dumps(mcp_config, separators=(",", ":"), sort_keys=True),
        "--strict-mcp-config",
        "--tools",
        "",
        "--allowedTools",
        f"mcp__{server_name}__*",
        "--permission-mode",
        "dontAsk",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--no-chrome",
        prompt,
    ]


def _runtime_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.upper() in _RUNTIME_ENV_NAMES}


def _new_transcript(
    profile: ClaudeCodeTargetProfile,
    prompt: str,
    mcp_endpoint: str,
    command: list[str],
    mcp_config: dict[str, Any],
) -> dict[str, Any]:
    recorded_command = ["<prompt>" if value == prompt else value for value in command]
    return {
        "schema_version": 1,
        "status": "running",
        "target_profile": profile.evidence_payload(),
        "prompt": prompt,
        "mcp_endpoint": mcp_endpoint,
        "mcp_config": mcp_config,
        "command": recorded_command,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "events": [],
        "warnings": [],
    }


def _parse_json_lines(raw: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not raw.strip():
        return events, ["Claude Code produced no stdout events"]
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"stdout line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(parsed, dict):
            warnings.append(f"stdout line {line_number} is not a JSON object")
            continue
        events.append(parsed)
    return events, warnings


def _record_output(
    transcript: dict[str, Any],
    returncode: int | None,
    stdout: str,
    stderr: str,
    events: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    transcript.update(
        {
            "exit_code": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "events": events,
            "warnings": warnings,
        }
    )


def _completion_error(returncode: int | None, events: list[dict[str, Any]]) -> str | None:
    if returncode != 0:
        return f"Claude Code exited with status {returncode}"
    terminal = next((event for event in reversed(events) if event.get("type") == "result"), None)
    if terminal is None:
        return "Claude Code stream ended without a terminal result event"
    if terminal.get("subtype") != "success" or terminal.get("is_error") is not False:
        return "Claude Code terminal result was not successful"
    return None


def _validate_completion(returncode: int | None, events: list[dict[str, Any]]) -> None:
    error = _completion_error(returncode, events)
    if error is not None:
        raise ExternalRuntimeError(error)


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        process.kill()
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_TERMINATE_GRACE_SECONDS)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
