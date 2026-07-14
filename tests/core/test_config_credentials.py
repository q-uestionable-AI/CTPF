"""Tests for keyring-based credential management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from q_ai.core.config import (
    _KEYRING_SERVICE,
    delete_credential,
    get_credential,
    get_keyring_credential,
    import_legacy_credentials,
    set_credential,
)


class TestGetCredential:
    """Tests for get_credential()."""

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env var takes precedence over keyring."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = "sk-from-keyring"
            result = get_credential("anthropic")
        assert result == "sk-from-env"

    def test_from_keyring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keyring is used when env var is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = "sk-from-keyring"
            result = get_credential("anthropic")
        assert result == "sk-from-keyring"
        mock_kr.get_password.assert_called_once_with(_KEYRING_SERVICE, "anthropic")

    def test_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when neither env var nor keyring has value."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = None
            result = get_credential("openai")
        assert result is None

    def test_normalises_provider_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed case and whitespace are normalised before lookup."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = "sk-key"
            result = get_credential("  Anthropic ")
        assert result == "sk-key"
        mock_kr.get_password.assert_called_once_with(_KEYRING_SERVICE, "anthropic")


class TestGetKeyringCredential:
    """The experiment credential boundary never reads environment variables."""

    def test_ignores_environment_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REMOTE_A_API_KEY", "sk-from-env")
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_password.return_value = "sk-from-keyring"
            result = get_keyring_credential(" Remote-A ")

        assert result == "sk-from-keyring"
        mock_kr.get_password.assert_called_once_with(_KEYRING_SERVICE, "remote-a")


class TestSetCredential:
    """Tests for set_credential()."""

    def test_writes_keyring(self) -> None:
        """Verifies keyring.set_password is called correctly."""
        with patch("q_ai.core.config.keyring") as mock_kr:
            set_credential("openai", "sk-test-key")
        mock_kr.set_password.assert_called_once_with(_KEYRING_SERVICE, "openai", "sk-test-key")

    def test_normalises_provider_name(self) -> None:
        """Mixed case and whitespace are normalised."""
        with patch("q_ai.core.config.keyring") as mock_kr:
            set_credential("  OpenAI ", "sk-key")
        mock_kr.set_password.assert_called_once_with(_KEYRING_SERVICE, "openai", "sk-key")


class TestDeleteCredential:
    """Tests for delete_credential()."""

    def test_deletes_from_keyring(self) -> None:
        """Verifies keyring.delete_password is called."""
        with patch("q_ai.core.config.keyring") as mock_kr:
            delete_credential("anthropic")
        mock_kr.delete_password.assert_called_once_with(_KEYRING_SERVICE, "anthropic")

    def test_missing_credential_no_error(self) -> None:
        """Deleting a non-existent credential does not raise."""
        import keyring.errors

        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.errors = keyring.errors
            mock_kr.delete_password.side_effect = keyring.errors.PasswordDeleteError()
            delete_credential("nonexistent")  # Should not raise


class TestInsecureBackendGuard:
    """Tests for _assert_secure_keyring() integration."""

    def test_set_credential_insecure_backend_raises(self) -> None:
        """set_credential raises RuntimeError on insecure backend."""
        fake_backend = type("PlaintextKeyring", (), {})()
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_keyring.return_value = fake_backend
            with pytest.raises(RuntimeError, match="Insecure keyring backend"):
                set_credential("openai", "sk-test")

    def test_get_credential_insecure_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_credential raises RuntimeError when env var absent and backend insecure."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fake_backend = type("PlaintextKeyring", (), {})()
        with patch("q_ai.core.config.keyring") as mock_kr:
            mock_kr.get_keyring.return_value = fake_backend
            with pytest.raises(RuntimeError, match="Insecure keyring backend"):
                get_credential("anthropic")

    def test_get_credential_env_var_bypasses_backend_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var path returns without touching keyring, even if backend is insecure."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        # Don't mock keyring at all — if it were called, the real backend
        # would be hit. The point is the env-var path returns early.
        result = get_credential("anthropic")
        assert result == "sk-from-env"


class TestImportLegacyCredentials:
    """Tests for import_legacy_credentials()."""

    def test_migrates_keys(self, tmp_path: Path) -> None:
        """Migrates plaintext keys from config.yaml to keyring."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "providers:\n"
            "  anthropic:\n"
            "    api_key: sk-ant-123\n"
            "  openai:\n"
            "    api_key: sk-oai-456\n"
            "lab:\n"
            "  endpoint: http://localhost\n",
            encoding="utf-8",
        )

        with patch("q_ai.core.config.keyring") as mock_kr:
            results = import_legacy_credentials(config_path)

        assert len(results) == 2
        assert all(s for _, s, _ in results)

        # Verify keyring was called
        calls = mock_kr.set_password.call_args_list
        providers_set = {c.args[1] for c in calls}
        assert "anthropic" in providers_set
        assert "openai" in providers_set

        # Verify config was cleaned
        import yaml

        cleaned = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "providers" not in cleaned
        assert cleaned["lab"]["endpoint"] == "http://localhost"

        # Verify backup
        backup = config_path.with_suffix(".yaml.bak")
        assert backup.exists()

    def test_no_legacy_credentials(self, tmp_path: Path) -> None:
        """Returns empty list when no providers section exists."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("lab:\n  key: val\n", encoding="utf-8")
        results = import_legacy_credentials(config_path)
        assert results == []

    def test_nonexistent_config(self, tmp_path: Path) -> None:
        """Returns empty list for missing config file."""
        results = import_legacy_credentials(tmp_path / "nonexistent.yaml")
        assert results == []
