"""Tests for proxy listen host and Pattern 2 stdio env forwarding."""

from __future__ import annotations

import pytest

from q_ai.proxy.adapters.sse import SseClientAdapter
from q_ai.proxy.adapters.streamable_http import StreamableHttpClientAdapter
from q_ai.proxy.constants import LISTEN_HOST, stdio_subprocess_env


def test_listen_host_is_loopback() -> None:
    """Client-facing proxy listeners default to 127.0.0.1."""
    assert LISTEN_HOST == "127.0.0.1"
    assert SseClientAdapter()._host == LISTEN_HOST
    assert StreamableHttpClientAdapter()._host == LISTEN_HOST


def test_stdio_subprocess_env_forwards_pattern2_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QAI_PATTERN2_* must be forwarded past MCP's default env whitelist."""
    monkeypatch.setenv("QAI_PATTERN2_RUN_ID", "m01")
    monkeypatch.setenv("QAI_PATTERN2_RESET_SINK", "1")
    env = stdio_subprocess_env()
    assert env["QAI_PATTERN2_RUN_ID"] == "m01"
    assert env["QAI_PATTERN2_RESET_SINK"] == "1"
    assert "PATH" in env


def test_stdio_subprocess_env_forwards_cascade_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QAI_CASCADE_* must be forwarded past MCP's default env whitelist."""
    monkeypatch.setenv("QAI_CASCADE_RUN_ID", "c-m01")
    monkeypatch.setenv("QAI_CASCADE_RESET", "1")
    env = stdio_subprocess_env()
    assert env["QAI_CASCADE_RUN_ID"] == "c-m01"
    assert env["QAI_CASCADE_RESET"] == "1"
