"""Tests for the TTY-aware interactive prompt utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer

from q_ai.core.cli.prompt import (
    build_teaching_tip,
    infer_transport,
    is_tty,
    parse_meta_flags,
    prompt_or_fail,
    prompt_or_fail_multiple,
    prompt_transport,
)


class TestIsTty:
    """Tests for the is_tty helper."""

    @patch("q_ai.core.cli.prompt.sys.stdin")
    def test_returns_true_when_tty(self, mock_stdin: object) -> None:
        mock_stdin.isatty = lambda: True  # type: ignore[attr-defined]
        assert is_tty() is True

    @patch("q_ai.core.cli.prompt.sys.stdin")
    def test_returns_false_when_not_tty(self, mock_stdin: object) -> None:
        mock_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        assert is_tty() is False


class TestPromptOrFail:
    """Tests for prompt_or_fail."""

    def test_returns_value_when_provided(self) -> None:
        assert prompt_or_fail("NAME", "my-value", "Enter name") == "my-value"

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_exits_when_no_tty_and_missing(self, _mock: object) -> None:
        with pytest.raises(typer.Exit):
            prompt_or_fail("NAME", None, "Enter name")

    @patch("q_ai.core.cli.prompt.typer.prompt", return_value="prompted-val")
    @patch("q_ai.core.cli.prompt.is_tty", return_value=True)
    def test_prompts_when_tty_and_missing(self, _tty: object, _prompt: object) -> None:
        result = prompt_or_fail("NAME", None, "Enter name")
        assert result == "prompted-val"


class TestPromptOrFailMultiple:
    """Tests for prompt_or_fail_multiple."""

    def test_returns_all_when_provided(self) -> None:
        result = prompt_or_fail_multiple(
            [
                ("A", "val-a", "Enter A"),
                ("B", "val-b", "Enter B"),
            ]
        )
        assert result == ["val-a", "val-b"]

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_exits_listing_all_missing_non_tty(self, _mock: object) -> None:
        with pytest.raises(typer.Exit):
            prompt_or_fail_multiple(
                [
                    ("A", None, "Enter A"),
                    ("B", None, "Enter B"),
                ]
            )


class TestBuildTeachingTip:
    """Tests for build_teaching_tip."""

    def test_simple_args(self) -> None:
        tip = build_teaching_tip("qai targets add", ["myserver", "http://example.com"])
        assert tip == "Tip: next time, run: qai targets add myserver http://example.com"

    def test_quotes_args_with_spaces(self) -> None:
        tip = build_teaching_tip("qai targets add", ["My Server", "http://example.com"])
        assert '"My Server"' in tip


class TestInferTransport:
    """Tests for transport inference."""

    def test_sse_url(self) -> None:
        transport, confident = infer_transport("http://localhost:3000/sse")
        assert transport == "sse"
        assert confident is True

    def test_sse_url_trailing_slash(self) -> None:
        transport, confident = infer_transport("http://localhost:3000/sse/")
        assert transport == "sse"
        assert confident is True

    def test_https_sse_url(self) -> None:
        transport, confident = infer_transport("https://example.com/mcp/sse")
        assert transport == "sse"
        assert confident is True

    def test_generic_http_url_low_confidence(self) -> None:
        transport, confident = infer_transport("http://localhost:3000")
        assert transport == "streamable-http"
        assert confident is False

    def test_command_string(self) -> None:
        transport, confident = infer_transport("npx @modelcontextprotocol/server-everything")
        assert transport == "stdio"
        assert confident is True

    def test_file_path(self) -> None:
        transport, confident = infer_transport("python my_server.py")
        assert transport == "stdio"
        assert confident is True

    def test_url_with_query_string(self) -> None:
        transport, confident = infer_transport("http://localhost:3000/sse?token=abc")
        assert transport == "sse"
        assert confident is True


class TestPromptTransport:
    """Tests for prompt_transport."""

    @patch("q_ai.core.cli.prompt.is_tty", return_value=True)
    def test_high_confidence_no_prompt(self, _mock: object) -> None:
        result = prompt_transport("http://localhost:3000/sse")
        assert result == "sse"

    @patch("q_ai.core.cli.prompt.is_tty", return_value=False)
    def test_low_confidence_no_tty_exits(self, _mock: object) -> None:
        with pytest.raises(typer.Exit):
            prompt_transport("http://localhost:3000")

    @patch("q_ai.core.cli.prompt.typer.prompt", return_value="streamable-http")
    @patch("q_ai.core.cli.prompt.is_tty", return_value=True)
    def test_low_confidence_tty_prompts(self, _tty: object, _prompt: object) -> None:
        result = prompt_transport("http://localhost:3000")
        assert result == "streamable-http"


class TestParseMetaFlags:
    """Tests for --meta key=value parsing."""

    def test_none_returns_none(self) -> None:
        assert parse_meta_flags(None) is None

    def test_empty_list_returns_none(self) -> None:
        assert parse_meta_flags([]) is None

    def test_single_pair(self) -> None:
        result = parse_meta_flags(["transport=sse"])
        assert result == {"transport": "sse"}

    def test_multiple_pairs(self) -> None:
        result = parse_meta_flags(["transport=sse", "owner=alice"])
        assert result == {"transport": "sse", "owner": "alice"}

    def test_value_with_equals(self) -> None:
        result = parse_meta_flags(["query=x=1"])
        assert result == {"query": "x=1"}

    def test_malformed_no_equals_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid --meta format"):
            parse_meta_flags(["nope"])

    def test_empty_key_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Key cannot be empty"):
            parse_meta_flags(["=value"])
