"""Tests for q_ai inject library CLI (fixtures-only surface)."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from q_ai.inject.cli import _load_config_names, app, console

runner = CliRunner()

# Set console width wide enough for Rich table columns to avoid truncation.
console.width = 200


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestInjectHelp:
    """inject --help shows fixture subcommands only."""

    def test_inject_help(self) -> None:
        """Serve and list-payloads appear; campaign/report do not."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "serve" in plain
        assert "list-payloads" in plain
        assert "campaign" not in plain
        assert "report" not in plain

    def test_module_entrypoint_exports_app(self) -> None:
        """``python -m q_ai.inject`` resolves via package ``__main__``."""
        import importlib

        main = importlib.import_module("q_ai.inject.__main__")
        assert main.app is app


class TestServe:
    """inject serve subcommand."""

    def test_serve_help(self) -> None:
        """Help text includes all serve options."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "--transport" in plain
        assert "--port" in plain
        assert "--payload-dir" in plain
        assert "--config" in plain

    def test_serve_invalid_transport(self) -> None:
        """Invalid transport value exits with error."""
        result = runner.invoke(app, ["serve", "--transport", "invalid"])
        assert result.exit_code != 0
        plain = _strip_ansi(result.output)
        assert "invalid" in plain.lower()

    def test_serve_stdio_calls_build_and_run(self) -> None:
        """Serve with stdio wires build_server and server.run correctly."""
        mock_server = MagicMock()
        with patch("q_ai.inject.cli.build_server", return_value=mock_server):
            result = runner.invoke(app, ["serve", "--transport", "stdio"])
        assert result.exit_code == 0
        mock_server.run.assert_called_once_with(transport="stdio")


class TestLoadConfigNames:
    """Tests for _load_config_names helper."""

    def test_valid_config(self, tmp_path: Path) -> None:
        """Loads a YAML list of strings."""
        config = tmp_path / "config.yaml"
        config.write_text("- exfil_via_important_tag\n- shadow_tool\n", encoding="utf-8")
        result = _load_config_names(config)
        assert result == ["exfil_via_important_tag", "shadow_tool"]

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Raises BadParameter for missing file."""
        import typer

        with pytest.raises(typer.BadParameter, match="Cannot read config file"):
            _load_config_names(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_structure(self, tmp_path: Path) -> None:
        """Raises BadParameter when YAML is not a list of strings."""
        config = tmp_path / "config.yaml"
        config.write_text("key: value\n", encoding="utf-8")
        import typer

        with pytest.raises(typer.BadParameter, match="YAML list of payload name strings"):
            _load_config_names(config)

    def test_list_of_non_strings(self, tmp_path: Path) -> None:
        """Raises BadParameter when YAML list contains non-strings."""
        config = tmp_path / "config.yaml"
        config.write_text("- 1\n- 2\n", encoding="utf-8")
        import typer

        with pytest.raises(typer.BadParameter, match="YAML list of payload name strings"):
            _load_config_names(config)


class TestListPayloads:
    """inject list-payloads subcommand."""

    def test_list_payloads_help(self) -> None:
        """Help text includes all list-payloads options."""
        result = runner.invoke(app, ["list-payloads", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "--technique" in plain
        assert "--target" in plain

    def test_list_payloads_exits_zero(self) -> None:
        """list-payloads exits successfully."""
        result = runner.invoke(app, ["list-payloads"])
        assert result.exit_code == 0

    def test_list_payloads_shows_all_payloads(self) -> None:
        """Known payload names appear in the table."""
        result = runner.invoke(app, ["list-payloads"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "exfil_via_important_tag" in plain
        assert "output_instruction_injection" in plain
        assert "shadow_tool" in plain

    def test_list_payloads_filter_description_poisoning(self) -> None:
        """Technique filter keeps description_poisoning rows only."""
        result = runner.invoke(app, ["list-payloads", "--technique", "description_poisoning"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "description_poisoning" in plain
        assert "output_injection" not in plain
        dp_names = [
            "exfil_via_important_tag",
            "preference_manipulation",
            "concealment_directive",
            "role_reassignment",
            "hidden_unicode_instruction",
            "long_description_buried",
        ]
        for name in dp_names:
            assert name in plain, f"Missing payload: {name}"

    def test_list_payloads_filter_output_injection(self) -> None:
        """Technique filter keeps output_injection rows."""
        result = runner.invoke(app, ["list-payloads", "--technique", "output_injection"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        oi_names = [
            "output_instruction_injection",
            "output_url_exfil",
            "output_tool_call_injection",
            "output_markdown_injection",
        ]
        for name in oi_names:
            assert name in plain, f"Missing payload: {name}"

    def test_list_payloads_filter_cross_tool(self) -> None:
        """Technique filter keeps cross_tool_escalation rows."""
        result = runner.invoke(app, ["list-payloads", "--technique", "cross_tool_escalation"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        ct_names = ["shadow_tool", "chain_via_description", "parameter_exfil"]
        for name in ct_names:
            assert name in plain, f"Missing payload: {name}"

    def test_list_payloads_invalid_technique(self) -> None:
        """Unknown technique exits with an error."""
        result = runner.invoke(app, ["list-payloads", "--technique", "nonexistent"])
        assert result.exit_code != 0
        plain = _strip_ansi(result.output)
        assert "Unknown technique" in plain
