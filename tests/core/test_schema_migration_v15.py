"""Tests for isolated governed-automation schema migration V15."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ctpf.core.schema import CURRENT_VERSION, _migrate_v15, migrate

AUTOMATION_TABLES = {
    "automation_policies",
    "automation_grants",
    "automation_runs",
    "automation_events",
}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }


def test_current_schema_version_is_15() -> None:
    """The isolated automation schema has a distinct version bump."""
    assert CURRENT_VERSION == 15


def test_fresh_database_has_all_automation_tables_and_foreign_keys(tmp_path: Path) -> None:
    """A fresh migration creates only the declared control-plane tables."""
    with sqlite3.connect(tmp_path / "ctpf.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        migrate(conn)

        assert AUTOMATION_TABLES.issubset(_tables(conn))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_VERSION
        run_fks = conn.execute("PRAGMA foreign_key_list(automation_runs)").fetchall()
        assert {row[2] for row in run_fks} == {
            "automation_grants",
            "automation_policies",
        }
        event_fks = conn.execute("PRAGMA foreign_key_list(automation_events)").fetchall()
        assert event_fks[0][2] == "automation_runs"
        assert event_fks[0][6].upper() == "CASCADE"


def test_v14_to_v15_preserves_existing_data(tmp_path: Path) -> None:
    """The additive migration leaves populated legacy tables untouched."""
    with sqlite3.connect(tmp_path / "ctpf.db") as conn:
        migrate(conn)
        for table in (
            "automation_events",
            "automation_runs",
            "automation_grants",
            "automation_policies",
        ):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("DROP INDEX IF EXISTS idx_automation_policies_status")
        conn.execute("DROP INDEX IF EXISTS idx_automation_grants_spec_digest")
        conn.execute("DROP INDEX IF EXISTS idx_automation_grants_expires_at")
        conn.execute("DROP INDEX IF EXISTS idx_automation_runs_state")
        conn.execute("DROP INDEX IF EXISTS idx_automation_runs_policy_id")
        conn.execute("DROP INDEX IF EXISTS idx_automation_runs_grant_id")
        conn.execute("DROP INDEX IF EXISTS idx_automation_events_run_id")
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("v14-preserved", "yes", "2026-07-16T00:00:00Z"),
        )
        conn.execute("PRAGMA user_version = 14")
        conn.commit()

        migrate(conn)

        assert AUTOMATION_TABLES.issubset(_tables(conn))
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", ("v14-preserved",)
        ).fetchone()
        assert row == ("yes",)


def test_v15_migration_is_idempotent_and_safe_without_legacy_tables(tmp_path: Path) -> None:
    """The isolated migration can be reapplied to a partial database."""
    with sqlite3.connect(tmp_path / "ctpf.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _migrate_v15(conn)
        _migrate_v15(conn)

        assert _tables(conn) == AUTOMATION_TABLES
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 15


def test_v15_constraints_reject_invalid_status_and_missing_policy(tmp_path: Path) -> None:
    """Database constraints fail closed even if a caller bypasses typed storage."""
    with sqlite3.connect(tmp_path / "ctpf.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _migrate_v15(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO automation_policies (
                    id, policy_json, policy_digest, signature, key_id,
                    status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("a" * 32, "{}", "b" * 64, "c" * 64, "d" * 64, "invalid", "x", "y"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO automation_grants (
                    id, grant_json, signature, key_id, spec_digest, policy_id,
                    policy_digest, scenario_fingerprint, target_fingerprints,
                    issued_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "e" * 32,
                    "{}",
                    "f" * 64,
                    "1" * 64,
                    "2" * 64,
                    "missing-policy",
                    "3" * 64,
                    "4" * 64,
                    "{}",
                    "x",
                    "y",
                ),
            )
