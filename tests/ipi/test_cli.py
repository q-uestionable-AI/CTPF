"""Tests for the IPI CLI commands."""

from __future__ import annotations

import datetime
from unittest.mock import patch

from typer.testing import CliRunner

from q_ai.ipi.cli import app
from q_ai.ipi.generate_service import GenerateResult
from q_ai.ipi.models import CitationFrame, DocumentTemplate
from q_ai.ipi.sweep_selection import (
    NoFindings,
    SelectedTemplate,
    StaleRefusal,
    TieRefusal,
)

# Disable Rich's ANSI coloring so substring assertions on --help output are
# stable across Windows (no color) and Linux/macOS CI (color on by default).
# Rich respects NO_COLOR; we also unset FORCE_COLOR and override TERM so no
# other signal re-enables styling. Without this, option names like `--target`
# render as ANSI-split spans (`\x1b[...]--\x1b[...]-target\x1b[...]`) and
# break literal substring matches.
runner = CliRunner(env={"NO_COLOR": "1", "FORCE_COLOR": None, "TERM": "dumb"})


class TestIpiGenerateCallbackArg:
    """Tests for callback as positional arg in ipi generate."""

    def test_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_no_callback_non_tty_fails(self, _mock: object) -> None:
        """Non-TTY with no callback fails with clear error."""
        result = runner.invoke(app, ["generate"])
        assert result.exit_code != 0

    @patch("q_ai.ipi.commands.generate.generate_documents")
    def test_positional_callback(self, mock_gen: object) -> None:
        """Callback can be passed as first positional argument."""
        from q_ai.ipi.generate_service import GenerateResult

        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]
        result = runner.invoke(app, ["generate", "http://localhost:8080"])
        assert result.exit_code == 0

    @patch("q_ai.ipi.commands.generate.generate_documents")
    def test_callback_option_flag(self, mock_gen: object) -> None:
        """--callback flag still works for backward compatibility."""
        from q_ai.ipi.generate_service import GenerateResult

        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]
        result = runner.invoke(app, ["generate", "--callback", "http://localhost:8080"])
        assert result.exit_code == 0


class TestIpiHelpText:
    """Token-level assertions guarding against `--help` text drift.

    These tests pin the key corrections from the 2026-04-17 help-text
    sweep so future additions don't silently regress the help surface.
    Assert on tokens, not full strings, so Typer formatting changes
    (wrapping, whitespace, trailing punctuation) don't break them.
    """

    def test_techniques_help_enumerates_all_seven_formats(self) -> None:
        """`qai ipi techniques --help` must list every supported format."""
        result = runner.invoke(app, ["techniques", "--help"])
        assert result.exit_code == 0
        for fmt in ("pdf", "image", "markdown", "html", "docx", "ics", "eml"):
            assert fmt in result.output, f"format {fmt!r} missing from techniques --help"

    def test_generate_technique_help_mentions_none(self) -> None:
        """`qai ipi generate --help` must mention `none` as a technique option."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "none" in result.output, "`none` control condition not mentioned in generate --help"

    def test_probe_export_help_references_scored_prompts(self) -> None:
        """`qai ipi probe --help` must describe --export as scored-prompts JSON."""
        result = runner.invoke(app, ["probe", "--help"])
        assert result.exit_code == 0
        # Accept either hyphenated or space-separated form to tolerate Typer wrap.
        assert "scored-prompts" in result.output or "scored prompts" in result.output, (
            "probe --export help does not reference scored-prompts JSON"
        )

    def test_generate_help_mentions_target_auto_select(self) -> None:
        """`qai ipi generate --help` must mention the --target auto-select flag."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.output, "--target flag missing from generate --help"


class TestIpiGenerateTargetAutoSelect:
    """Tests for `qai ipi generate --target` auto-selecting a template."""

    _FIXED_NOW = datetime.datetime(2026, 4, 18, 12, 0, 0, tzinfo=datetime.UTC)

    def _selected(
        self,
        template: DocumentTemplate = DocumentTemplate.WHOIS,
        age_days: int = 2,
        stale_warn: bool = False,
        rate: float = 0.80,
    ) -> SelectedTemplate:
        return SelectedTemplate(
            template=template,
            run_id="run-id",
            completed_at=self._FIXED_NOW - datetime.timedelta(days=age_days),
            compliance_rate=rate,
            age_days=age_days,
            stale_warn=stale_warn,
        )

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_target_with_findings_emits_prefix_and_generates(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        mock_select.return_value = self._selected()  # type: ignore[attr-defined]
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]

        result = runner.invoke(
            app,
            ["generate", "http://localhost:8080", "--target", "target-123"],
        )

        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert "Auto-selected template: whois" in collapsed
        assert "80% compliance" in collapsed
        mock_select.assert_called_once_with("target-123")  # type: ignore[attr-defined]
        # Confirm the chosen template reached the generator call.
        call_kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["template"] == DocumentTemplate.WHOIS

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_explicit_template_bypasses_selection(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        """--template on the command line wins; no selection, no prefix."""
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]

        result = runner.invoke(
            app,
            [
                "generate",
                "http://localhost:8080",
                "--target",
                "target-123",
                "--template",
                "report",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Auto-selected template" not in result.output
        mock_select.assert_not_called()  # type: ignore[attr-defined]
        call_kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["template"] == DocumentTemplate.REPORT

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_no_findings_exits_non_zero_with_guidance(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        mock_select.return_value = NoFindings(target_id="target-123")  # type: ignore[attr-defined]

        result = runner.invoke(
            app,
            ["generate", "http://localhost:8080", "--target", "target-123"],
        )

        assert result.exit_code != 0
        collapsed = " ".join(result.output.split())
        assert "No sweep findings" in collapsed
        assert "qai ipi sweep" in collapsed
        assert "--template" in collapsed
        mock_gen.assert_not_called()  # type: ignore[attr-defined]

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_tie_exits_non_zero_listing_candidates(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        mock_select.return_value = TieRefusal(  # type: ignore[attr-defined]
            candidates=[
                (DocumentTemplate.WHOIS, 0.80),
                (DocumentTemplate.REPORT, 0.75),
            ],
            run_id="run-id",
        )

        result = runner.invoke(
            app,
            ["generate", "http://localhost:8080", "--target", "target-123"],
        )

        assert result.exit_code != 0
        collapsed = " ".join(result.output.split())
        assert "tied within 10pp" in collapsed
        assert "whois: 80% compliance" in collapsed
        assert "report: 75% compliance" in collapsed
        assert "--template" in collapsed
        mock_gen.assert_not_called()  # type: ignore[attr-defined]

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_stale_refuse_exits_non_zero_with_timestamp(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        completed_at = self._FIXED_NOW - datetime.timedelta(days=35)
        mock_select.return_value = StaleRefusal(  # type: ignore[attr-defined]
            run_id="run-id",
            completed_at=completed_at,
            age_days=35,
        )

        result = runner.invoke(
            app,
            ["generate", "http://localhost:8080", "--target", "target-123"],
        )

        assert result.exit_code != 0
        collapsed = " ".join(result.output.split())
        assert "35 days ago" in collapsed
        assert completed_at.isoformat() in collapsed
        assert "fresh sweep" in collapsed
        mock_gen.assert_not_called()  # type: ignore[attr-defined]

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_stale_warn_appends_rerun_suggestion(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        mock_select.return_value = self._selected(  # type: ignore[attr-defined]
            age_days=10,
            stale_warn=True,
        )
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]

        result = runner.invoke(
            app,
            ["generate", "http://localhost:8080", "--target", "target-123"],
        )

        assert result.exit_code == 0, result.output
        # Rich may line-wrap the prefix; collapse whitespace before checking.
        collapsed = " ".join(result.output.split())
        assert "Auto-selected template: whois" in collapsed
        assert "10 days ago" in collapsed
        assert "consider re-running sweep" in collapsed

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    @patch("q_ai.ipi.commands.generate.select_template_for_target")
    def test_no_target_preserves_existing_behavior(
        self,
        mock_select: object,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        """Without --target, selection is never invoked, even when no --template is passed."""
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]

        result = runner.invoke(app, ["generate", "http://localhost:8080"])

        assert result.exit_code == 0, result.output
        mock_select.assert_not_called()  # type: ignore[attr-defined]
        assert "Auto-selected template" not in result.output
        call_kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["template"] == DocumentTemplate.GENERIC


class TestIpiGenerateCitationFrame:
    """Tests for the `qai ipi generate --citation-frame` flag."""

    def test_help_lists_citation_frame(self) -> None:
        """PR #123-style drift guard: --citation-frame surfaces in --help."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "citation-frame" in result.output

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    def test_default_flag_forwards_template_aware(
        self,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        """No --citation-frame -> generate_documents receives TEMPLATE_AWARE."""
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]
        result = runner.invoke(app, ["generate", "http://localhost:8080"])

        assert result.exit_code == 0, result.output
        call_kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["citation_frame"] == CitationFrame.TEMPLATE_AWARE

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    def test_plain_value_forwards_plain_enum(
        self,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        """--citation-frame plain -> generate_documents receives PLAIN."""
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]
        result = runner.invoke(
            app,
            [
                "generate",
                "http://localhost:8080",
                "--citation-frame",
                "plain",
                "--payload-style",
                "citation",
                "--payload-type",
                "callback",
            ],
        )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["citation_frame"] == CitationFrame.PLAIN

    @patch("q_ai.ipi.mapper.persist_generate")
    @patch("q_ai.ipi.commands.generate.generate_documents")
    def test_template_aware_value_forwards_enum(
        self,
        mock_gen: object,
        _mock_persist: object,
    ) -> None:
        """--citation-frame template-aware -> TEMPLATE_AWARE forwarded explicitly."""
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]
        result = runner.invoke(
            app,
            [
                "generate",
                "http://localhost:8080",
                "--citation-frame",
                "template-aware",
            ],
        )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["citation_frame"] == CitationFrame.TEMPLATE_AWARE

    def test_invalid_value_exits_non_zero(self) -> None:
        """--citation-frame bogus exits non-zero with the parser's error."""
        result = runner.invoke(
            app,
            ["generate", "http://localhost:8080", "--citation-frame", "bogus"],
        )
        assert result.exit_code != 0
        assert "bogus" in result.output

    def test_sweep_citation_frame_still_parses(self) -> None:
        """Hoisting _parse_citation_frame to _shared did not break sweep-side parsing."""
        result = runner.invoke(
            app,
            [
                "sweep",
                "--dry-run",
                "--templates",
                "whois",
                "--styles",
                "citation",
                "--citation-frame",
                "plain",
            ],
        )
        assert result.exit_code == 0, result.output
