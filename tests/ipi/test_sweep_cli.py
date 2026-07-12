"""Tests for the IPI sweep CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from q_ai.ipi.cli import app

# Disable Rich's ANSI coloring so substring assertions on --help output are
# stable across Windows (no color) and Linux CI (color on by default). Rich
# respects NO_COLOR; we also unset FORCE_COLOR and override TERM so no other
# signal re-enables styling. Without this, option names like `--reps` render
# as ANSI-split spans (`\x1b[...]--\x1b[...]-reps\x1b[...]`) and break
# literal substring matches.
runner = CliRunner(env={"NO_COLOR": "1", "FORCE_COLOR": None, "TERM": "dumb"})


class TestSweepHelp:
    """Tests for sweep command help and discoverability."""

    def test_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["sweep", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output

    def test_help_lists_required_flags(self) -> None:
        result = runner.invoke(app, ["sweep", "--help"])
        assert result.exit_code == 0
        for flag in (
            "--model",
            "--target",
            "--templates",
            "--styles",
            "--payload-type",
            "--citation-frame",
            "--reps",
            "--dry-run",
            "--export",
        ):
            assert flag in result.output, f"--help missing {flag}"

    def test_help_lists_citation_frame(self) -> None:
        """PR #123-style drift guard: --citation-frame surfaces in --help.

        Uses the same literal substring check as the other flag assertions
        above so ANSI-color rendering can't silently hide the flag.
        """
        result = runner.invoke(app, ["sweep", "--help"])
        assert result.exit_code == 0
        assert "citation-frame" in result.output

    def test_ipi_help_lists_sweep(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "sweep" in result.output


class TestSweepDryRun:
    """Tests for sweep --dry-run mode."""

    def test_dry_run_lists_all_combinations_by_default(self) -> None:
        """--dry-run enumerates every non-GENERIC template x the obvious style."""
        result = runner.invoke(app, ["sweep", "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()
        # 11 templates in the registry minus GENERIC.
        assert "11" in result.output

    def test_dry_run_filters_templates(self) -> None:
        """--templates restricts the dry-run enumeration."""
        result = runner.invoke(
            app,
            ["sweep", "--dry-run", "--templates", "whois,report"],
        )
        assert result.exit_code == 0
        assert "whois" in result.output
        assert "report" in result.output
        assert "2" in result.output  # 2 combinations x 1 style

    def test_dry_run_cartesian_with_styles(self) -> None:
        """--styles multiplies the combination count."""
        result = runner.invoke(
            app,
            [
                "sweep",
                "--dry-run",
                "--templates",
                "whois",
                "--styles",
                "obvious,citation",
            ],
        )
        assert result.exit_code == 0
        # 1 template x 2 styles = 2 combinations
        assert "obvious" in result.output
        assert "citation" in result.output


class TestSweepFlagValidation:
    """Tests for CLI flag validation and error handling."""

    def test_invalid_template_fails(self) -> None:
        """Unknown template name produces a clear error."""
        result = runner.invoke(
            app,
            ["sweep", "--dry-run", "--templates", "nonexistent"],
        )
        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower()

    def test_invalid_style_fails(self) -> None:
        """Unknown style name produces a clear error."""
        result = runner.invoke(
            app,
            ["sweep", "--dry-run", "--styles", "bogus"],
        )
        assert result.exit_code != 0
        assert "bogus" in result.output.lower()

    def test_non_callback_payload_type_rejected(self) -> None:
        """Non-callback --payload-type is rejected with a clear message."""
        result = runner.invoke(
            app,
            ["sweep", "--dry-run", "--payload-type", "exfil_summary"],
        )
        assert result.exit_code != 0
        assert "callback" in result.output.lower()

    def test_callback_payload_type_accepted(self) -> None:
        """--payload-type callback is accepted (case-insensitive)."""
        result = runner.invoke(
            app,
            ["sweep", "--dry-run", "--payload-type", "CALLBACK"],
        )
        assert result.exit_code == 0

    def test_citation_frame_plain_accepted(self) -> None:
        """--citation-frame plain parses."""
        result = runner.invoke(app, ["sweep", "--dry-run", "--citation-frame", "plain"])
        assert result.exit_code == 0

    def test_citation_frame_template_aware_accepted(self) -> None:
        """--citation-frame template-aware parses."""
        result = runner.invoke(app, ["sweep", "--dry-run", "--citation-frame", "template-aware"])
        assert result.exit_code == 0

    def test_citation_frame_absent_defaults(self) -> None:
        """No --citation-frame flag is accepted (defaults to template-aware)."""
        result = runner.invoke(app, ["sweep", "--dry-run"])
        assert result.exit_code == 0

    def test_citation_frame_invalid_value_rejected(self) -> None:
        """Unknown --citation-frame value produces a non-zero exit + echoes the value."""
        result = runner.invoke(app, ["sweep", "--dry-run", "--citation-frame", "bogus"])
        assert result.exit_code != 0
        assert "bogus" in result.output.lower()

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_no_endpoint_non_tty_fails(self, _mock: MagicMock) -> None:
        """Non-TTY with no endpoint fails (dry-run off)."""
        result = runner.invoke(app, ["sweep", "--model", "test"])
        assert result.exit_code != 0

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_no_model_non_tty_fails(self, _mock: MagicMock) -> None:
        """Non-TTY with no model fails (dry-run off)."""
        result = runner.invoke(app, ["sweep", "http://localhost/v1"])
        assert result.exit_code != 0

    def test_concurrency_zero_fails(self) -> None:
        """--concurrency 0 fails with a clear error."""
        result = runner.invoke(
            app,
            [
                "sweep",
                "http://localhost/v1",
                "--model",
                "t",
                "--concurrency",
                "0",
            ],
        )
        assert result.exit_code != 0
        assert "concurrency" in result.output.lower()

    def test_reps_zero_fails(self) -> None:
        """--reps 0 fails with a clear error."""
        result = runner.invoke(
            app,
            ["sweep", "http://localhost/v1", "--model", "t", "--reps", "0"],
        )
        assert result.exit_code != 0
        assert "reps" in result.output.lower()


class TestSweepApiKey:
    """Tests for API key resolution — reuses the probe env var."""

    @patch("q_ai.ipi.sweep_service.run_sweep")
    @patch("q_ai.ipi.sweep_service.persist_sweep_run")
    @patch.dict("os.environ", {"QAI_PROBE_API_KEY": "env-key"})
    def test_env_var_flows_to_run_sweep(
        self, _mock_persist: MagicMock, mock_run_sweep: MagicMock
    ) -> None:
        """QAI_PROBE_API_KEY is read and passed through to run_sweep."""
        from q_ai.ipi.sweep_service import SweepRunResult

        async def _fake_run(**kwargs: object) -> SweepRunResult:
            return SweepRunResult()

        mock_run_sweep.side_effect = _fake_run

        with patch(
            "q_ai.ipi.sweep_service.persist_sweep_run",
            return_value="fake-run-id",
        ):
            result = runner.invoke(
                app,
                [
                    "sweep",
                    "http://localhost/v1",
                    "--model",
                    "test",
                    "--templates",
                    "whois",
                    "--reps",
                    "1",
                ],
            )

        assert result.exit_code == 0
        assert mock_run_sweep.called
        assert mock_run_sweep.call_args.kwargs["api_key"] == "env-key"


class TestSweepCitationFrameKwarg:
    """Tests that --citation-frame reaches run_sweep as a CitationFrame enum."""

    @patch("q_ai.ipi.sweep_service.run_sweep")
    @patch("q_ai.ipi.sweep_service.persist_sweep_run")
    def test_plain_flows_to_run_sweep(
        self, _mock_persist: MagicMock, mock_run_sweep: MagicMock
    ) -> None:
        """--citation-frame plain is forwarded as CitationFrame.PLAIN."""
        from q_ai.ipi.models import CitationFrame
        from q_ai.ipi.sweep_service import SweepRunResult

        async def _fake_run(**kwargs: object) -> SweepRunResult:
            return SweepRunResult()

        mock_run_sweep.side_effect = _fake_run

        with patch(
            "q_ai.ipi.sweep_service.persist_sweep_run",
            return_value="fake-run-id",
        ):
            result = runner.invoke(
                app,
                [
                    "sweep",
                    "http://localhost/v1",
                    "--model",
                    "test",
                    "--templates",
                    "whois",
                    "--reps",
                    "1",
                    "--citation-frame",
                    "plain",
                ],
            )

        assert result.exit_code == 0
        assert mock_run_sweep.call_args.kwargs["citation_frame"] is CitationFrame.PLAIN

    @patch("q_ai.ipi.sweep_service.run_sweep")
    @patch("q_ai.ipi.sweep_service.persist_sweep_run")
    def test_default_is_template_aware(
        self, _mock_persist: MagicMock, mock_run_sweep: MagicMock
    ) -> None:
        """No flag forwards CitationFrame.TEMPLATE_AWARE."""
        from q_ai.ipi.models import CitationFrame
        from q_ai.ipi.sweep_service import SweepRunResult

        async def _fake_run(**kwargs: object) -> SweepRunResult:
            return SweepRunResult()

        mock_run_sweep.side_effect = _fake_run

        with patch(
            "q_ai.ipi.sweep_service.persist_sweep_run",
            return_value="fake-run-id",
        ):
            result = runner.invoke(
                app,
                [
                    "sweep",
                    "http://localhost/v1",
                    "--model",
                    "test",
                    "--templates",
                    "whois",
                    "--reps",
                    "1",
                ],
            )

        assert result.exit_code == 0
        assert mock_run_sweep.call_args.kwargs["citation_frame"] is CitationFrame.TEMPLATE_AWARE
