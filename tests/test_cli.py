"""Tests for the q-ai root CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from q_ai.cli import app

runner = CliRunner()

_TAGLINE = "CTPF research harness"


class TestCLIHelp:
    """qai --help shows help text."""

    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert _TAGLINE in result.output

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
    """qai --version shows version string."""

    def test_version_exits_zero(self) -> None:
        from q_ai import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"qai {__version__}" in result.output

    def test_version_short_flag(self) -> None:
        from q_ai import __version__

        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert f"qai {__version__}" in result.output


class TestBareQai:
    """Bare `qai` prints help screen."""

    def test_no_args_prints_help(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Quick Start" in result.output
        assert _TAGLINE in result.output
        assert "qai proxy" in result.output
        assert "qai targets" in result.output

    def test_no_args_has_no_ui_hint(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "qai ui" not in result.output

    def test_subcommands_still_work(self) -> None:
        result = runner.invoke(app, ["runs", "--help"])
        assert result.exit_code == 0
        assert "runs" in result.output.lower()


class TestRemovedUiCommand:
    """qai ui is no longer registered."""

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
