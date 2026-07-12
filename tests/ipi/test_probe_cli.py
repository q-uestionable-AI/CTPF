"""Tests for the IPI probe CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from q_ai.ipi.cli import app

runner = CliRunner()


class TestProbeHelp:
    """Tests for probe command help and discoverability."""

    def test_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["probe", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output

    def test_ipi_help_lists_probe(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "probe" in result.output


class TestProbeDryRun:
    """Tests for probe --dry-run mode."""

    def test_dry_run_lists_probes(self) -> None:
        """--dry-run displays probe table without sending requests."""
        result = runner.invoke(app, ["probe", "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()
        assert "20" in result.output  # 20 probes

    def test_dry_run_with_custom_probe_set(self, tmp_path: Path) -> None:
        """--dry-run works with custom probe set."""
        probe_file = tmp_path / "custom.yaml"
        probe_file.write_text(
            "probes:\n"
            "  - id: c-001\n"
            "    category: custom\n"
            '    description: "Custom probe"\n'
            '    system_prompt: "Test"\n'
            '    user_prompt: "Say {canary}"\n'
            '    canary_match: "{canary}"\n'
        )
        result = runner.invoke(app, ["probe", "--dry-run", "--probe-set", str(probe_file)])
        assert result.exit_code == 0
        assert "1" in result.output  # 1 probe


class TestProbeArgParsing:
    """Tests for probe argument parsing and validation."""

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_no_endpoint_non_tty_fails(self, _mock: MagicMock) -> None:
        """Non-TTY with no endpoint fails with clear error."""
        result = runner.invoke(app, ["probe", "--model", "test"])
        assert result.exit_code != 0

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_no_model_non_tty_fails(self, _mock: MagicMock) -> None:
        """Non-TTY with no model fails with clear error."""
        result = runner.invoke(app, ["probe", "http://localhost/v1"])
        assert result.exit_code != 0

    def test_invalid_probe_set_fails(self) -> None:
        """Non-existent probe set file fails gracefully."""
        result = runner.invoke(
            app,
            [
                "probe",
                "http://localhost/v1",
                "--model",
                "test",
                "--probe-set",
                "/nonexistent/file.yaml",
            ],
        )
        assert result.exit_code != 0
        assert "Error loading probes" in result.output

    def test_concurrency_zero_fails(self) -> None:
        """--concurrency 0 fails with clear error."""
        result = runner.invoke(
            app,
            ["probe", "http://localhost/v1", "--model", "t", "--concurrency", "0"],
        )
        assert result.exit_code != 0
        assert "concurrency" in result.output.lower()


class TestProbeApiKey:
    """Tests for API key resolution."""

    @patch("q_ai.ipi.probe_service.run_probes")
    @patch.dict("os.environ", {"QAI_PROBE_API_KEY": "env-key"})
    def test_env_var_flows_to_run_probes(self, mock_run_probes: MagicMock) -> None:
        """QAI_PROBE_API_KEY env var is passed through to run_probes."""
        from q_ai.ipi.probe_service import ProbeRunResult

        # Make run_probes return a coroutine wrapping an empty result.
        async def _fake_run(**kwargs: object) -> ProbeRunResult:
            return ProbeRunResult()

        mock_run_probes.side_effect = _fake_run

        # persist_probe_run is imported lazily inside the command body, so the
        # live-import binding points at q_ai.ipi.probe_service — patch there.
        with patch("q_ai.ipi.probe_service.persist_probe_run", return_value="fake-run-id"):
            result = runner.invoke(
                app,
                ["probe", "http://localhost/v1", "--model", "test"],
            )

        assert result.exit_code == 0
        assert mock_run_probes.called
        # Verify the env var key was passed as api_key.
        call_kwargs = mock_run_probes.call_args.kwargs
        assert call_kwargs["api_key"] == "env-key"
