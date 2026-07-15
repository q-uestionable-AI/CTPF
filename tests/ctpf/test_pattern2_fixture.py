"""Tests for the packaged Pattern 2 fixture."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "ctpf" / "kernel" / "pattern2_fixture.py"
)
_EXPECTED_MCP_TOOLS = frozenset({"read_status", "apply_change", "read_sink"})


def _load_fixture(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    monkeypatch.delenv("CTPF_PATTERN2_RUN_ID", raising=False)
    monkeypatch.delenv("CTPF_PATTERN2_RESET_SINK", raising=False)
    monkeypatch.delenv("CTPF_PATTERN2_REQUIRE_RUN_ID", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("pattern2_fixture_under_test", _FIXTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_run_id_aborts_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    with pytest.raises(SystemExit) as excinfo:
        _load_fixture(monkeypatch)
    assert excinfo.value.code == 2


def test_active_sink_path_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, CTPF_PATTERN2_REQUIRE_RUN_ID="0")
    assert mod.active_sink_path() == tmp_path / "ctpf-pattern2" / "sink.json"


def test_active_sink_path_run_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, CTPF_PATTERN2_RUN_ID="baseline-1")
    assert mod.active_sink_path() == tmp_path / "ctpf-pattern2" / "sink-baseline-1.json"


def test_reset_sink_removes_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, CTPF_PATTERN2_RUN_ID="r1")
    path = mod.active_sink_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"effect": "applied"}', encoding="utf-8")
    assert mod.reset_active_sink() is True
    assert not path.exists()
    assert mod.reset_active_sink() is False


def test_reset_on_start_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    sink_dir = tmp_path / "ctpf-pattern2"
    sink_dir.mkdir(parents=True, exist_ok=True)
    stale = sink_dir / "sink-clean.json"
    stale.write_text('{"effect": "applied"}', encoding="utf-8")
    mod = _load_fixture(
        monkeypatch,
        CTPF_PATTERN2_RUN_ID="clean",
        CTPF_PATTERN2_RESET_SINK="1",
    )
    assert not stale.exists()
    assert mod.active_sink_path() == stale


def test_apply_change_writes_run_scoped_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, CTPF_PATTERN2_RUN_ID="manip-1")
    payload = json.loads(mod.apply_change("approve_refund", "test"))
    assert payload["effect"] == "applied"
    assert payload["run_id"] == "manip-1"
    sink_path = Path(payload["sink_path"])
    assert sink_path.name == "sink-manip-1.json"
    assert sink_path.exists()
    on_disk = json.loads(sink_path.read_text(encoding="utf-8"))
    assert on_disk["action"] == "approve_refund"


def test_reset_sink_is_not_an_mcp_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, CTPF_PATTERN2_RUN_ID="tools-check")
    tool_names = {tool.name for tool in asyncio.run(mod.mcp.list_tools())}
    assert tool_names == _EXPECTED_MCP_TOOLS
    assert "reset_sink" not in tool_names
    assert callable(mod.reset_active_sink)
