"""Configuration management for q-ai.

Supports a hybrid approach:
- OS keyring for credentials (API keys)
- YAML file (~/.qai/config.yaml) for non-secret settings
- SQLite settings table for operational defaults
- Precedence resolver: CLI value > env var > DB setting > config file > default
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import keyring
import keyring.errors
import yaml

_DEFAULT_CONFIG_PATH = Path.home() / ".qai" / "config.yaml"

_KEYRING_SERVICE = "q-ai"


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to ~/.qai/config.yaml.

    Returns:
        Configuration dict. Empty dict if file doesn't exist.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f)
    return data or {}


def get_credential(provider: str) -> str | None:
    """Get an API key from the OS keyring.

    There is no environment-variable or config-file credential fallback.
    Legacy credentials must be migrated via
    ``qai config import-legacy-credentials``.

    Args:
        provider: Provider name (e.g. "anthropic", "openai").

    Returns:
        API key string or None if not found.
    """
    return get_keyring_credential(provider)


def get_keyring_credential(provider: str) -> str | None:
    """Read a provider credential only from the OS keyring.

    Experiment drivers use this explicit boundary so credentials cannot enter
    a run through process configuration.

    Args:
        provider: Provider or target-profile credential name.

    Returns:
        API key string or None if the keyring has no matching entry.
    """
    provider = provider.strip().lower()
    _assert_secure_keyring()
    result: str | None = keyring.get_password(_KEYRING_SERVICE, provider)
    return result


def set_credential(provider: str, api_key: str) -> None:
    """Store an API key in the OS keyring.

    Args:
        provider: Provider name.
        api_key: The API key to store.
    """
    provider = provider.strip().lower()
    _assert_secure_keyring()
    keyring.set_password(_KEYRING_SERVICE, provider, api_key)


def delete_credential(provider: str) -> None:
    """Remove an API key from the OS keyring.

    Args:
        provider: Provider name.
    """
    provider = provider.strip().lower()
    _assert_secure_keyring()
    with contextlib.suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(_KEYRING_SERVICE, provider)


def _assert_secure_keyring() -> None:
    """Raise if the active keyring backend is known to be insecure.

    On headless Linux without a keyring daemon, keyring may silently
    fall back to PlaintextKeyring or FailKeyring. Operators must configure
    an OS-backed keyring before using credentials in that environment.

    Raises:
        RuntimeError: If the active backend is insecure.
    """
    backend = keyring.get_keyring()
    insecure = ("PlaintextKeyring", "NullKeyring", "FailKeyring")
    if type(backend).__name__ in insecure:
        raise RuntimeError(
            f"Insecure keyring backend detected: {type(backend).__name__}. "
            "No secure keyring daemon is available. Configure an OS-backed "
            "keyring before storing or reading credentials."
        )


def get_lab_setting(
    key: str,
    config_path: Path | None = None,
) -> str | None:
    """Read a setting from the config file's lab section.

    Args:
        key: Setting key within the lab section.
        config_path: Optional override for config file path.

    Returns:
        Setting value or None if not found.
    """
    config = load_config(config_path)
    lab = config.get("lab", {})
    result: str | None = lab.get(key)
    return result


def resolve(
    key: str,
    cli_value: str | None = None,
    env_var: str | None = None,
    db_path: Path | None = None,
    config_path: Path | None = None,
) -> tuple[str | None, str]:
    """Resolve a configuration value using the precedence chain.

    Precedence:
        CLI value > environment variable > DB setting >
        config file > None.

    Args:
        key: The setting key to look up.
        cli_value: Optional value from CLI argument.
        env_var: Optional environment variable name to check.
        db_path: Optional override for database path.
        config_path: Optional override for config file path.

    Returns:
        Tuple of (value, source) where source is one of
        "cli", "env", "db", "file", or "default".
    """
    if cli_value is not None:
        return cli_value, "cli"

    if env_var is not None:
        env_value = os.environ.get(env_var)
        if env_value is not None:
            return env_value, "env"

    # Check DB settings
    from q_ai.core.db import get_connection, get_setting

    try:
        with get_connection(db_path) as conn:
            db_value = get_setting(conn, key)
            if db_value is not None:
                return db_value, "db"
    except Exception:  # noqa: S110 — DB may not exist yet on first run
        pass

    # Check config file
    config = load_config(config_path)
    # Walk nested keys
    # Walk nested keys, e.g. "audit.default_transport" resolves
    # through config -> audit -> default_transport
    parts = key.split(".")
    current: object = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            current = None
            break
    if current is not None and not isinstance(current, dict):
        return str(current), "file"

    return None, "default"


def _migrate_provider_key(
    provider_name: str,
    provider_config: dict,
    providers: dict,
) -> tuple[str, bool, str] | None:
    """Attempt to migrate a single provider's API key to the OS keyring.

    If the provider config is not a dict, has no ``api_key`` entry, or the
    entry is falsy, returns ``None`` (nothing to migrate).  On success the
    key is removed from *provider_config* (and *providers* is cleaned up if
    the config becomes empty).

    Args:
        provider_name: Name of the provider (e.g. "anthropic").
        provider_config: Mutable dict for this provider from config.yaml.
        providers: The top-level ``providers`` mapping (mutated in-place).

    Returns:
        A ``(provider, success, message)`` tuple, or ``None`` when the
        entry should be skipped.
    """
    if not isinstance(provider_config, dict):
        return None
    api_key = provider_config.get("api_key")
    if not api_key:
        return None

    try:
        set_credential(provider_name, api_key)
        del provider_config["api_key"]
        if not provider_config:
            del providers[provider_name]
    except Exception as exc:
        return (provider_name, False, str(exc))
    else:
        return (provider_name, True, "migrated to keyring")


def _write_cleaned_config(path: Path, config: dict) -> None:
    """Back up the original config and write the cleaned version.

    Args:
        path: Path to the config YAML file.
        config: The cleaned configuration dict to persist.
    """
    backup_path = path.with_suffix(".yaml.bak")
    if path.exists():
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    if sys.platform != "win32":
        path.chmod(0o600)


def import_legacy_credentials(config_path: Path | None = None) -> list[tuple[str, bool, str]]:
    """Migrate plaintext keys from config.yaml to OS keyring.

    Reads provider API keys from the YAML config, writes them to the
    OS keyring, and removes the keys from the YAML. Non-secret settings
    are preserved.

    Args:
        config_path: Optional override for config file path.

    Returns:
        List of (provider, success, message) tuples.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    config = load_config(path)
    providers = config.get("providers", {})

    if not providers:
        return []

    results: list[tuple[str, bool, str]] = []
    for provider_name, provider_config in list(providers.items()):
        result = _migrate_provider_key(provider_name, provider_config, providers)
        if result is not None:
            results.append(result)

    # Clean up empty providers section
    if not providers:
        config.pop("providers", None)

    # Write back the config without secrets
    if any(success for _, success, _ in results):
        _write_cleaned_config(path, config)

    return results
