"""Tests for Pattern 2 preflight fixture sink path hygiene."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_PATH = Path(__file__).resolve().parent / "pattern2_preflight.py"


def _load_fixture(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    """Load the fixture module with a clean env overlay."""
    monkeypatch.delenv("QAI_PATTERN2_RUN_ID", raising=False)
    monkeypatch.delenv("QAI_PATTERN2_RESET_SINK", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    spec = importlib.util.spec_from_file_location("pattern2_preflight_under_test", _FIXTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_active_sink_path_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default sink is sink.json under the temp preflight directory."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch)
    assert mod.active_sink_path() == tmp_path / "qai-phase2-preflight" / "sink.json"


def test_active_sink_path_run_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """QAI_PATTERN2_RUN_ID scopes the sink filename."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_PATTERN2_RUN_ID="baseline-1")
    assert mod.active_sink_path() == tmp_path / "qai-phase2-preflight" / "sink-baseline-1.json"


def test_reset_sink_removes_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """reset_active_sink deletes an existing sink file."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_PATTERN2_RUN_ID="r1")
    path = mod.active_sink_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"effect": "applied"}', encoding="utf-8")
    assert mod.reset_active_sink() is True
    assert not path.exists()
    assert mod.reset_active_sink() is False


def test_reset_on_start_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """QAI_PATTERN2_RESET_SINK=1 clears the active sink at import."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    sink_dir = tmp_path / "qai-phase2-preflight"
    sink_dir.mkdir(parents=True, exist_ok=True)
    stale = sink_dir / "sink-clean.json"
    stale.write_text('{"effect": "applied"}', encoding="utf-8")
    mod = _load_fixture(
        monkeypatch,
        QAI_PATTERN2_RUN_ID="clean",
        QAI_PATTERN2_RESET_SINK="1",
    )
    assert not stale.exists()
    assert mod.active_sink_path() == stale


def test_apply_change_writes_run_scoped_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """apply_change records effect under the run-scoped sink path."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_PATTERN2_RUN_ID="manip-1")
    raw = mod.apply_change("approve_refund", "test")
    payload = json.loads(raw)
    assert payload["effect"] == "applied"
    assert payload["run_id"] == "manip-1"
    sink_path = Path(payload["sink_path"])
    assert sink_path.name == "sink-manip-1.json"
    assert sink_path.exists()
    on_disk = json.loads(sink_path.read_text(encoding="utf-8"))
    assert on_disk["action"] == "approve_refund"
