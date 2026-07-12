"""Tests for the audit CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from q_ai.audit.cli import app

runner = CliRunner()

# Rich ANSI codes can leak into CliRunner output on some platforms (macOS CI
# in particular), splitting "Connection failed" across escape sequences. The
# backward-compat tests assert on that exact substring, so they use this
# color-stripped runner locally to keep the assertion stable.
_NO_COLOR_ENV = {"NO_COLOR": "1", "FORCE_COLOR": None, "TERM": "dumb"}


class TestListChecks:
    """Tests for the list-checks command."""

    def test_lists_all_scanners(self) -> None:
        """Verify list-checks shows injection scanner."""
        result = runner.invoke(app, ["list-checks"])
        assert result.exit_code == 0
        assert "injection" in result.output
        assert "command_injection" in result.output

    def test_shows_10_scanners(self) -> None:
        """Verify all 10 scanner categories appear in list-checks output."""
        result = runner.invoke(app, ["list-checks"])
        assert result.exit_code == 0
        for cat in [
            "command_injection",
            "auth",
            "token_exposure",
            "permissions",
            "tool_poisoning",
            "prompt_injection",
            "audit_telemetry",
            "supply_chain",
            "shadow_servers",
            "context_sharing",
        ]:
            assert cat in result.output

    def test_framework_flag(self) -> None:
        """Verify --framework owasp_mcp_top10 shows OWASP IDs."""
        result = runner.invoke(app, ["list-checks", "--framework", "owasp_mcp_top10"])
        assert result.exit_code == 0
        assert "MCP05" in result.output


class TestAuditSubcommand:
    """Tests for the audit subcommand structure."""

    def test_audit_help(self) -> None:
        """Verify audit help shows all subcommands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "scan" in result.output
        assert "list-checks" in result.output
        assert "enumerate" in result.output
        assert "report" in result.output


class TestAuditScanPositionalTarget:
    """Tests for positional TARGET and transport inference in audit scan."""

    def test_scan_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["scan", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_scan_no_args_non_tty_fails(self, _mock: object) -> None:
        """Non-TTY with no target fails with clear error."""
        result = runner.invoke(app, ["scan"])
        assert result.exit_code != 0

    @patch("q_ai.audit.cli.MCPConnection")
    def test_backward_compat_explicit_flags(self, mock_mcp_cls: MagicMock) -> None:
        """--transport stdio --command still parses and reaches the connection attempt.

        The MCP connection is mocked to raise ConnectionError on entry, so no
        subprocess is spawned. This verifies flag parsing (backward compat) and
        the CLI's connection-error surface without exercising the real stdio
        client — which on Py3.13 can hit an anyio cross-task cancel-scope race
        when a misbehaving server writes non-JSONRPC output. Tracked separately.
        """
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(side_effect=ConnectionError("mocked failure"))
        mock_mcp_cls.stdio.return_value = mock_conn

        local_runner = CliRunner(env=_NO_COLOR_ENV)
        result = local_runner.invoke(
            app,
            ["scan", "--transport", "stdio", "--command", "not-a-real-server"],
        )
        assert result.exit_code != 0
        assert "Connection failed" in result.output


class TestAuditEnumeratePositionalTarget:
    """Tests for positional TARGET in audit enumerate."""

    def test_enumerate_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["enumerate", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_enumerate_no_args_non_tty_fails(self, _mock: object) -> None:
        result = runner.invoke(app, ["enumerate"])
        assert result.exit_code != 0

    @patch("q_ai.audit.cli.MCPConnection")
    def test_backward_compat_explicit_flags(self, mock_mcp_cls: MagicMock) -> None:
        """--transport stdio --command still parses and reaches the connection attempt.

        See the sibling scan test for the rationale behind mocking the MCP
        connection rather than launching a real subprocess.
        """
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(side_effect=ConnectionError("mocked failure"))
        mock_mcp_cls.stdio.return_value = mock_conn

        local_runner = CliRunner(env=_NO_COLOR_ENV)
        result = local_runner.invoke(
            app,
            ["enumerate", "--transport", "stdio", "--command", "not-a-real-server"],
        )
        assert result.exit_code != 0
        assert "Connection failed" in result.output
