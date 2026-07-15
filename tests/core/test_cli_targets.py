"""Tests for CTPF target CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from ctpf.cli import app
from ctpf.core.db import get_connection

runner = CliRunner()


def test_add_help_uses_canonical_ctpf_name() -> None:
    """Target help presents the canonical CLI name in its examples."""
    result = runner.invoke(app, ["targets", "add", "--help"], prog_name="ctpf")

    assert result.exit_code == 0
    assert "ctpf targets add" in result.output
    assert "driver=openai-compatible" in result.output
    assert "agent-runtime" in result.output


class TestTargetsList:
    def test_exits_zero_empty_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        with get_connection(db_path):
            pass
        result = runner.invoke(app, ["targets", "list", "--db-path", str(db_path)])
        assert result.exit_code == 0


class TestTargetsAddPositional:
    """targets add with positional NAME and URI."""

    def test_add_positional_args(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "My MCP Server",
                "http://localhost:3000/sse",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert "Created target" in result.output

    def test_add_with_spaces_in_name(self, tmp_path: Path) -> None:
        """Spaces in target name work as positional arg."""
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "Server With Spaces",
                "http://example.com",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert "Created target" in result.output

    def test_add_then_list_positional(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        runner.invoke(
            app,
            [
                "targets",
                "add",
                "visible-target",
                "http://localhost:8080",
                "--db-path",
                str(db_path),
            ],
        )
        result = runner.invoke(app, ["targets", "list", "--db-path", str(db_path)])
        assert result.exit_code == 0
        assert "visible-target" in result.output
        assert "http://localhost:8080" in result.output

    def test_default_type_is_server(self, tmp_path: Path) -> None:
        """Type defaults to 'server' when not specified."""
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "test-target",
                "http://localhost",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        # Verify by listing
        list_result = runner.invoke(app, ["targets", "list", "--db-path", str(db_path)])
        assert "server" in list_result.output

    def test_explicit_type_flag(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "test-target",
                "http://localhost",
                "--type",
                "endpoint",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0


class TestTargetsAddMeta:
    """targets add --meta key=value flag."""

    def test_meta_single(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "meta-target",
                "http://example.com",
                "--meta",
                "transport=sse",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0

    def test_meta_multiple(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "meta-target",
                "http://example.com",
                "--meta",
                "transport=sse",
                "--meta",
                "owner=alice",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0

    def test_meta_malformed_fails(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "meta-target",
                "http://example.com",
                "--meta",
                "nope",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code != 0


class TestTargetsAddInteractive:
    """Interactive prompting for targets add."""

    @patch("ctpf.core.cli.targets.is_tty", return_value=False)
    @patch("ctpf.core.cli.prompt.is_tty", return_value=False)
    def test_non_tty_no_args_fails(
        self, _mock_prompt: object, _mock_targets: object, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            ["targets", "add", "--db-path", str(db_path)],
        )
        assert result.exit_code != 0

    @patch("ctpf.core.cli.targets.is_tty", return_value=True)
    @patch("ctpf.core.cli.prompt.is_tty", return_value=True)
    def test_teaching_tip_shown_on_interactive(
        self, _mock_prompt: object, _mock_targets: object, tmp_path: Path
    ) -> None:
        """When args are prompted, a teaching tip is printed."""
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            ["targets", "add", "--db-path", str(db_path)],
            input="My Server\nhttp://example.com\n",
        )
        assert result.exit_code == 0
        assert "Tip:" in result.output

    def test_no_teaching_tip_when_args_provided(self, tmp_path: Path) -> None:
        """No teaching tip when all args are provided directly."""
        db_path = tmp_path / "ctpf.db"
        result = runner.invoke(
            app,
            [
                "targets",
                "add",
                "My Server",
                "http://example.com",
                "--db-path",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert "Tip:" not in result.output
