"""Tests for proxy listen host and research-fixture stdio env forwarding."""

from __future__ import annotations

import pytest

from ctpf.proxy.adapters.sse import SseClientAdapter
from ctpf.proxy.adapters.streamable_http import StreamableHttpClientAdapter
from ctpf.proxy.constants import LISTEN_HOST, stdio_subprocess_env


def test_listen_host_is_loopback() -> None:
    """Client-facing proxy listeners default to 127.0.0.1."""
    assert LISTEN_HOST == "127.0.0.1"
    assert SseClientAdapter()._host == LISTEN_HOST
    assert StreamableHttpClientAdapter()._host == LISTEN_HOST


def test_stdio_subprocess_env_forwards_pattern2_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CTPF_PATTERN2_* must be forwarded past MCP's default env whitelist."""
    monkeypatch.setenv("CTPF_PATTERN2_RUN_ID", "m01")
    monkeypatch.setenv("CTPF_PATTERN2_RESET_SINK", "1")
    env = stdio_subprocess_env()
    assert env["CTPF_PATTERN2_RUN_ID"] == "m01"
    assert env["CTPF_PATTERN2_RESET_SINK"] == "1"
    assert "PATH" in env


def test_stdio_subprocess_env_forwards_pattern3_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CTPF_PATTERN3_* must be forwarded past MCP's default env whitelist."""
    monkeypatch.setenv("CTPF_PATTERN3_AUTHORITY_PATH", "C:/evidence/authority.json")
    monkeypatch.setenv("CTPF_PATTERN3_CONDITION", "baseline")
    monkeypatch.setenv("CTPF_PATTERN3_RESET_SINK", "1")
    monkeypatch.setenv("CTPF_PATTERN3_RUN_ID", "p3-baseline")

    env = stdio_subprocess_env()

    assert env["CTPF_PATTERN3_AUTHORITY_PATH"] == "C:/evidence/authority.json"
    assert env["CTPF_PATTERN3_CONDITION"] == "baseline"
    assert env["CTPF_PATTERN3_RESET_SINK"] == "1"
    assert env["CTPF_PATTERN3_RUN_ID"] == "p3-baseline"


def test_stdio_subprocess_env_forwards_cascade_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CTPF_CASCADE_* must be forwarded past MCP's default env whitelist."""
    monkeypatch.setenv("CTPF_CASCADE_RUN_ID", "c-m01")
    monkeypatch.setenv("CTPF_CASCADE_RESET", "1")
    env = stdio_subprocess_env()
    assert env["CTPF_CASCADE_RUN_ID"] == "c-m01"
    assert env["CTPF_CASCADE_RESET"] == "1"
