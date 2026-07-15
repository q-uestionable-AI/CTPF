"""Tests for the CTPF Research Harness root CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from ctpf.cli import app

runner = CliRunner()

_DISPLAY_NAME = "CTPF Research Harness"
_SUBTITLE = "Evidence-first capability-trust experiments for agentic systems"


class TestCLIHelp:
    """Root help shows the CTPF identity and transitional commands."""

    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"], prog_name="ctpf")
        assert result.exit_code == 0
        assert _DISPLAY_NAME in result.output
        assert _SUBTITLE in " ".join(result.output.split())

    def test_help_shows_transitional_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("proxy", "targets", "runs", "findings", "config", "db"):
            assert cmd in result.output

    def test_help_hides_removed_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in (
            "ui",
            "audit",
            "assist",
            "rxp",
            "inject",
            "ipi",
            "chain",
            "cxp",
            "imports",
            "orchestrator",
        ):
            # Match as top-level command tokens, not substrings in longer words
            assert f"  {cmd} " not in result.output
            assert f"  {cmd}\n" not in result.output


class TestCLIVersion:
    """Version output reflects the canonical entry point."""

    def test_version_exits_zero(self) -> None:
        from ctpf import __version__

        result = runner.invoke(app, ["--version"], prog_name="ctpf")
        assert result.exit_code == 0
        assert f"ctpf {__version__}" in result.output

    def test_version_short_flag(self) -> None:
        from ctpf import __version__

        result = runner.invoke(app, ["-V"], prog_name="ctpf")
        assert result.exit_code == 0
        assert f"ctpf {__version__}" in result.output


class TestBareCLI:
    """A bare `ctpf` invocation prints the help screen."""

    def test_no_args_prints_help(self) -> None:
        result = runner.invoke(app, [], prog_name="ctpf")
        assert result.exit_code == 0
        normalized_output = " ".join(result.output.split())
        assert "Quick Start" in result.output
        assert _DISPLAY_NAME in result.output
        assert _SUBTITLE in normalized_output
        assert result.output.index("ctpf experiment") < result.output.index("ctpf proxy")
        assert "ctpf proxy" in result.output
        assert "ctpf targets" in result.output

    def test_no_args_has_no_ui_hint(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "ctpf ui" not in result.output

    def test_subcommands_still_work(self) -> None:
        result = runner.invoke(app, ["runs", "--help"])
        assert result.exit_code == 0
        assert "runs" in result.output.lower()


class TestRemovedUiCommand:
    """ctpf ui is no longer registered."""

    def test_ui_command_absent(self) -> None:
        result = runner.invoke(app, ["ui"])
        assert result.exit_code != 0


class TestGroupedHelp:
    """Root --help shows grouped command panels."""

    def test_help_shows_observe_group(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Observe" in result.output

    def test_help_shows_start_group(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Start" in result.output

    def test_help_shows_manage_group(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Manage" in result.output

    def test_help_has_no_modules_group(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Modules" not in result.output


def test_distribution_exposes_only_ctpf_entry_point() -> None:
    """Project metadata exposes only the canonical executable name."""
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]
    assert scripts == {"ctpf": "ctpf.cli:app"}
