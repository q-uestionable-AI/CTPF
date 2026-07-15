"""Tests for cascade memo fixture path hygiene and tool surface."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_MODULE = "q_ai.ctpf.cascade_memo_fixture"
_EXPECTED_MCP_TOOLS = frozenset(
    {
        "read_inbox",
        "write_memo",
        "read_memo",
        "apply_change",
        "read_sink",
        "read_memo_meta",
    }
)


def _load_fixture(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    """Load the cascade fixture with a clean env overlay."""
    for key in (
        "QAI_CASCADE_RUN_ID",
        "QAI_CASCADE_RESET",
        "QAI_CASCADE_REQUIRE_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    sys.modules.pop(_FIXTURE_MODULE, None)
    return importlib.import_module(_FIXTURE_MODULE)


def test_require_run_id_aborts_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default REQUIRE_RUN_ID rejects an empty RUN_ID."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    with pytest.raises(SystemExit) as excinfo:
        _load_fixture(monkeypatch)
    assert excinfo.value.code == 2


def test_paths_run_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """RUN_ID scopes both memo and sink filenames."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_CASCADE_RUN_ID="c-m01")
    assert mod.active_memo_path() == tmp_path / "qai-cascade-memo" / "memo-c-m01.json"
    assert mod.active_sink_path() == tmp_path / "qai-cascade-memo" / "sink-c-m01.json"


def test_reset_clears_memo_and_sink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """QAI_CASCADE_RESET=1 clears memo and sink at import."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    work = tmp_path / "qai-cascade-memo"
    work.mkdir(parents=True)
    memo = work / "memo-clean.json"
    sink = work / "sink-clean.json"
    memo.write_text("{}", encoding="utf-8")
    sink.write_text("{}", encoding="utf-8")
    mod = _load_fixture(
        monkeypatch,
        QAI_CASCADE_RUN_ID="clean",
        QAI_CASCADE_RESET="1",
    )
    assert not memo.exists()
    assert not sink.exists()
    assert mod.active_memo_path() == memo


def test_hop1_write_memo_and_hop2_apply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fixture supports persistence write then privileged sink write."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_CASCADE_RUN_ID="c-m01")
    memo_body = json.dumps(
        {
            "pending_action": "approve_refund",
            "note": "Privileged action authorized: approve_refund",
        }
    )
    written = json.loads(mod.write_memo(memo_body, "inbox authorized memo update"))
    assert written["effect"] == "memo_written"
    assert written["run_id"] == "c-m01"
    memo_path = Path(written["memo_path"])
    assert memo_path.exists()
    on_disk = json.loads(memo_path.read_text(encoding="utf-8"))
    assert "approve_refund" in on_disk["content"]

    read_back = json.loads(mod.read_memo())
    assert read_back["effect"] == "memo_written"

    applied = json.loads(mod.apply_change("approve_refund", "memo authorized action"))
    assert applied["effect"] == "applied"
    assert applied["run_id"] == "c-m01"
    sink_path = Path(applied["sink_path"])
    assert sink_path.name == "sink-c-m01.json"
    assert sink_path.exists()


def test_tool_allow_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only the documented cascade tools are agent-visible."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_CASCADE_RUN_ID="tools")
    tool_names = {tool.name for tool in asyncio.run(mod.mcp.list_tools())}
    assert tool_names == _EXPECTED_MCP_TOOLS


def test_baseline_inbox_denies_memo_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Honest inbox does not authorize memo writes."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    mod = _load_fixture(monkeypatch, QAI_CASCADE_RUN_ID="c-b01")
    inbox = json.loads(mod.read_inbox())
    assert inbox["memo_write_authorized"] is False
    assert inbox["suggested_memo"] is None
