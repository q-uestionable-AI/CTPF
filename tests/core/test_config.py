"""Tests for q-ai configuration management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from q_ai.core.config import (
    get_credential,
    get_lab_setting,
    load_config,
    resolve,
    set_credential,
)


class TestLoadConfig:
    """Tests for load_config()."""

    def test_returns_empty_dict_when_no_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing file returns empty dict."""
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config == {}

    def test_loads_existing_config(
        self,
        tmp_path: Path,
    ) -> None:
        """Existing YAML is parsed correctly."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("lab:\n  key: value\n")
        config = load_config(config_path)
        assert config["lab"]["key"] == "value"


class TestCredentials:
    """Tests for get_credential() and set_credential() with keyring."""

    def test_set_credential_writes_keyring(self) -> None:
        """set_credential calls keyring.set_password."""
        with patch("q_ai.core.config.keyring") as mock_kr:
            set_credential("anthropic", "sk-test-123")
        mock_kr.set_password.assert_called_once_with("q-ai", "anthropic", "sk-test-123")

    def test_get_credential_from_keyring(self) -> None:
        """get_credential reads from keyring."""
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = "sk-test-123"
            result = get_credential("anthropic")
        assert result == "sk-test-123"

    def test_get_credential_ignores_environment_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Environment variables never override the keyring."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = "sk-from-keyring"
            result = get_credential("anthropic")
        assert result == "sk-from-keyring"
        mock_kr.get_password.assert_called_once_with("q-ai", "anthropic")

    def test_get_credential_missing_provider(self) -> None:
        """Missing provider returns None."""
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = None
            assert get_credential("nonexistent") is None


class TestLabSettings:
    """Tests for get_lab_setting()."""

    def test_get_lab_setting(self, tmp_path: Path) -> None:
        """Reads a value from the lab section."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "lab:\n  endpoint: http://localhost:8080\n",
        )
        result = get_lab_setting("endpoint", config_path)
        assert result == "http://localhost:8080"

    def test_get_lab_setting_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing key in lab section returns None."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("lab:\n  key: val\n")
        assert get_lab_setting("nonexistent", config_path) is None


class TestResolve:
    """Tests for resolve() precedence chain."""

    def test_cli_value_wins(self, tmp_path: Path) -> None:
        """CLI value has highest precedence."""
        db_path = tmp_path / "qai.db"
        value, source = resolve(
            "key",
            cli_value="from-cli",
            db_path=db_path,
            config_path=tmp_path / "c.yaml",
        )
        assert value == "from-cli"
        assert source == "cli"

    def test_env_var_beats_db(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Environment variable beats DB setting."""
        db_path = tmp_path / "qai.db"
        monkeypatch.setenv("QAI_TEST_VAR", "from-env")
        value, source = resolve(
            "key",
            env_var="QAI_TEST_VAR",
            db_path=db_path,
            config_path=tmp_path / "c.yaml",
        )
        assert value == "from-env"
        assert source == "env"

    def test_db_beats_file(self, tmp_path: Path) -> None:
        """DB setting beats config file value."""
        db_path = tmp_path / "qai.db"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "audit:\n  transport: sse\n",
        )
        from q_ai.core.db import get_connection, set_setting

        with get_connection(db_path) as conn:
            set_setting(conn, "audit.transport", "stdio")
        value, source = resolve(
            "audit.transport",
            db_path=db_path,
            config_path=config_path,
        )
        assert value == "stdio"
        assert source == "db"

    def test_file_fallback(self, tmp_path: Path) -> None:
        """Config file is used when DB has no value."""
        db_path = tmp_path / "qai.db"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "audit:\n  transport: sse\n",
        )
        value, source = resolve(
            "audit.transport",
            db_path=db_path,
            config_path=config_path,
        )
        assert value == "sse"
        assert source == "file"

    def test_returns_none_default(
        self,
        tmp_path: Path,
    ) -> None:
        """No value anywhere returns (None, 'default')."""
        db_path = tmp_path / "qai.db"
        value, source = resolve(
            "nonexistent",
            db_path=db_path,
            config_path=tmp_path / "c.yaml",
        )
        assert value is None
        assert source == "default"
