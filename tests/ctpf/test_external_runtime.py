"""Tests for the external agent-runtime boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ctpf import external_runtime
from ctpf.core.db import create_target, get_connection
from ctpf.driven_inference import OpenAICompatibleTargetProfile
from ctpf.external_runtime import (
    ClaudeCodeDriver,
    ClaudeCodeTargetProfile,
    ExternalRuntimeError,
    load_experiment_target_profile,
)


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int | None = 0) -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode: int | None = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def terminate(self) -> None:
        self.returncode = -1

    async def wait(self) -> int:
        return self.returncode or 0


class _SignalIgnoringProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._exited = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        return b"", b""

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode or 0


def _profile() -> ClaudeCodeTargetProfile:
    return ClaudeCodeTargetProfile(
        target_id="1234567890abcdef",
        name="claude research runtime",
        executable="C:/tools/claude.exe",
        model="claude-opus-4-1-20250805",
        runtime_version="2.1.121 (Claude Code)",
        timeout_seconds=90,
    )


def _create_runtime_target(
    db_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    with get_connection(db_path) as conn:
        return create_target(
            conn,
            type="agent-runtime",
            name="claude research runtime",
            uri="claude",
            metadata=metadata
            or {
                "driver": "claude-code-cli",
                "model": "claude-opus-4-1-20250805",
                "timeout_seconds": "90",
            },
        )


class TestClaudeCodeTargetProfile:
    """Persisted target rows pin one credential-free Claude Code runtime."""

    def test_loads_exact_runtime_profile(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "ctpf.db"
        target_id = _create_runtime_target(db_path)
        monkeypatch.setattr(
            external_runtime,
            "_inspect_claude_executable",
            lambda _raw: ("C:/tools/claude.exe", "2.1.121 (Claude Code)"),
        )

        profile = load_experiment_target_profile(target_id[:8], db_path=db_path)

        assert isinstance(profile, ClaudeCodeTargetProfile)
        assert profile.model == "claude-opus-4-1-20250805"
        assert profile.timeout_seconds == 90
        assert profile.runtime_version == "2.1.121 (Claude Code)"
        assert "credential" not in json.dumps(profile.evidence_payload()).lower()

    @pytest.mark.parametrize(
        ("version", "message"),
        [
            ("2.1.120 (Claude Code)", "2.1.121 or later"),
            ("unexpected", "invalid version"),
        ],
    )
    def test_rejects_runtime_without_always_load_support(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        version: str,
        message: str,
    ) -> None:
        db_path = tmp_path / "ctpf.db"
        target_id = _create_runtime_target(db_path)
        monkeypatch.setattr(
            external_runtime,
            "_inspect_claude_executable",
            lambda _raw: ("C:/tools/claude.exe", version),
        )

        with pytest.raises(ExternalRuntimeError, match=message):
            load_experiment_target_profile(target_id[:8], db_path=db_path)

    def test_preserves_existing_inference_target_dispatch(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        with get_connection(db_path) as conn:
            target_id = create_target(
                conn,
                type="inference",
                name="remote model",
                uri="https://models.example.test/v1",
                metadata={
                    "driver": "openai-compatible",
                    "model": "model-a",
                    "credential": "remote-a",
                },
            )

        profile = load_experiment_target_profile(target_id[:8], db_path=db_path)

        assert isinstance(profile, OpenAICompatibleTargetProfile)
        assert profile.model == "model-a"

    @pytest.mark.parametrize(
        ("metadata", "message"),
        [
            (
                {
                    "driver": "claude-code-cli",
                    "model": "sonnet",
                },
                "exact model ID",
            ),
            (
                {
                    "driver": "claude-code-cli",
                    "model": "claude-opus-4-1-20250805",
                    "credential": "forbidden",
                },
                "must not declare secret metadata",
            ),
        ],
    )
    def test_rejects_unpinned_or_secret_metadata(
        self,
        tmp_path: Path,
        metadata: dict[str, Any],
        message: str,
    ) -> None:
        db_path = tmp_path / "ctpf.db"
        target_id = _create_runtime_target(db_path, metadata=metadata)

        with pytest.raises(ExternalRuntimeError, match=message):
            load_experiment_target_profile(target_id[:8], db_path=db_path)


class TestClaudeCodeDriver:
    """The adapter launches one fresh restricted runtime and preserves its stream."""

    async def test_runs_restricted_session_without_credential_environment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        event = {"type": "result", "subtype": "success", "is_error": False}
        captured: dict[str, Any] = {}

        async def create_process(*command: str, **kwargs: Any) -> _FakeProcess:
            captured.update({"command": command, **kwargs})
            return _FakeProcess(json.dumps(event) + "\n")

        monkeypatch.setattr(external_runtime.asyncio, "create_subprocess_exec", create_process)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-boundary")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross-boundary")
        transcript_path = tmp_path / "session-A.inference.json"

        result = await ClaudeCodeDriver(_profile()).run(
            "Inspect the inbox.",
            "http://127.0.0.1:8765/mcp/",
            transcript_path,
        )

        command = captured["command"]
        assert result.event_count == 1
        assert "--strict-mcp-config" in command
        assert "--no-session-persistence" in command
        assert "--max-turns" not in command
        assert "--bare" not in command
        mcp_config = json.loads(command[command.index("--mcp-config") + 1])
        assert mcp_config == {
            "mcpServers": {
                "ctpf-cascade": {
                    "alwaysLoad": True,
                    "type": "http",
                    "url": "http://127.0.0.1:8765/mcp/",
                }
            }
        }
        assert captured["env"].get("ANTHROPIC_API_KEY") is None
        assert captured["env"].get("AWS_SECRET_ACCESS_KEY") is None
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "complete"
        assert transcript["command"][-1] == "<prompt>"
        assert transcript["events"] == [event]

    def test_custom_mcp_server_name_is_applied_without_scenario_logic(self) -> None:
        config = external_runtime._mcp_config(
            "http://127.0.0.1:8765/mcp/",
            "ctpf-pattern2",
        )
        command = external_runtime._claude_command(
            _profile(),
            "Inspect status.",
            config,
            "ctpf-pattern2",
        )
        assert "mcp__ctpf-pattern2__*" in command
        assert "ctpf-pattern2" in config["mcpServers"]

    @pytest.mark.parametrize("server_name", ["", "bad name", "bad/name"])
    def test_invalid_mcp_server_name_is_rejected(self, server_name: str) -> None:
        with pytest.raises(ExternalRuntimeError, match="unsupported characters"):
            external_runtime._validated_server_name(server_name)

    async def test_timeout_terminates_process_and_preserves_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _FakeProcess("", returncode=None)
        original_wait_for = asyncio.wait_for

        async def create_process(*_args: str, **_kwargs: Any) -> _FakeProcess:
            return process

        async def expire(awaitable: Any, *, timeout: float) -> Any:
            if timeout == 90:
                awaitable.close()
                raise TimeoutError
            return await original_wait_for(awaitable, timeout=timeout)

        monkeypatch.setattr(external_runtime.asyncio, "create_subprocess_exec", create_process)
        monkeypatch.setattr(external_runtime.asyncio, "wait_for", expire)
        transcript_path = tmp_path / "session-A.inference.json"

        with pytest.raises(ExternalRuntimeError, match="90-second runtime limit"):
            await ClaudeCodeDriver(_profile()).run(
                "Inspect the inbox.",
                "http://127.0.0.1:8765/mcp/",
                transcript_path,
            )

        assert process.returncode == -1
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "failed"

    async def test_timeout_kills_process_that_ignores_terminate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = _SignalIgnoringProcess()
        original_wait_for = asyncio.wait_for

        async def create_process(*_args: str, **_kwargs: Any) -> _SignalIgnoringProcess:
            return process

        async def expire_or_wait(awaitable: Any, *, timeout: float) -> Any:
            if timeout == 90:
                awaitable.close()
                raise TimeoutError
            if process.terminated and not process.killed:
                awaitable.close()
                raise TimeoutError
            return await original_wait_for(awaitable, timeout=timeout)

        monkeypatch.setattr(external_runtime.asyncio, "create_subprocess_exec", create_process)
        monkeypatch.setattr(external_runtime.asyncio, "wait_for", expire_or_wait)
        transcript_path = tmp_path / "session-A.inference.json"

        with pytest.raises(ExternalRuntimeError, match="90-second runtime limit"):
            await ClaudeCodeDriver(_profile()).run(
                "Inspect the inbox.",
                "http://127.0.0.1:8765/mcp/",
                transcript_path,
            )

        assert process.terminated is True
        assert process.killed is True
        assert process.returncode == -9
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "failed"

    async def test_failed_exit_preserves_diagnostics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def create_process(*_args: str, **_kwargs: Any) -> _FakeProcess:
            return _FakeProcess("", "authentication required", returncode=1)

        monkeypatch.setattr(external_runtime.asyncio, "create_subprocess_exec", create_process)
        transcript_path = tmp_path / "session-A.inference.json"

        with pytest.raises(ExternalRuntimeError, match="exited with status 1"):
            await ClaudeCodeDriver(_profile()).run(
                "Inspect the inbox.",
                "http://127.0.0.1:8765/mcp/",
                transcript_path,
            )

        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assert transcript["status"] == "failed"
        assert transcript["exit_code"] == 1
        assert transcript["stderr"] == "authentication required"
        assert transcript["warnings"] == ["Claude Code produced no stdout events"]

    async def test_rejects_non_loopback_mcp_endpoint(self, tmp_path: Path) -> None:
        with pytest.raises(ExternalRuntimeError, match=r"HTTP on 127\.0\.0\.1"):
            await ClaudeCodeDriver(_profile()).run(
                "Inspect the inbox.",
                "https://agents.example.test/mcp/",
                tmp_path / "session-A.inference.json",
            )
