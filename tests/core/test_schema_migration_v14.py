"""Tests for V14 schema migration (ipi_hits.via_tunnel column)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ctpf.core.schema import (
    CURRENT_VERSION,
    V1_INDEXES,
    V1_TABLES,
    V6_INDEXES,
    V6_TABLES,
    _migrate_v14,
    migrate,
)


class TestSchemaV14:
    """Schema V14 adds a non-null boolean via_tunnel column to ipi_hits."""

    def test_current_version_includes_v14(self) -> None:
        """The latest schema remains newer than the preserved V14 migration."""
        assert CURRENT_VERSION >= 14

    def test_fresh_db_has_via_tunnel_column(self, tmp_path: Path) -> None:
        """A freshly migrated database exposes via_tunnel on ipi_hits with
        the expected type, NOT NULL constraint, and default of 0."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            migrate(conn)

            col_info = {
                row[1]: row for row in conn.execute("PRAGMA table_info(ipi_hits)").fetchall()
            }
            assert "via_tunnel" in col_info
            # PRAGMA table_info row: (cid, name, type, notnull, dflt_value, pk)
            assert col_info["via_tunnel"][2].upper() == "INTEGER"
            assert col_info["via_tunnel"][3] == 1, "via_tunnel must be NOT NULL"
            assert str(col_info["via_tunnel"][4]) == "0", "via_tunnel default must be 0"

    def test_user_version_after_migrate(self, tmp_path: Path) -> None:
        """migrate() advances user_version to the current schema."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            migrate(conn)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_VERSION

    def test_v14_on_existing_v14_db_is_noop(self, tmp_path: Path) -> None:
        """Re-running _migrate_v14 on an already-migrated DB is idempotent."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            migrate(conn)

            # Re-apply v14; should not raise (column already exists)
            _migrate_v14(conn)

            col_info = {
                row[1]: row for row in conn.execute("PRAGMA table_info(ipi_hits)").fetchall()
            }
            assert "via_tunnel" in col_info
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == 14

    def test_v13_to_v14_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Migrating a populated v13 DB to v14 keeps rows with via_tunnel=0."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # Build a V13 state by running the full migrate chain then resetting
            # user_version back to 13 so the v14 dispatch branch fires below.
            migrate(conn)
            conn.execute("PRAGMA user_version = 13")
            conn.commit()

            # Drop the via_tunnel column we just added so we can test the
            # migration path that adds it. SQLite < 3.35 can't DROP COLUMN
            # directly; recreate the table without it by rebuilding.
            conn.execute("ALTER TABLE ipi_hits RENAME TO ipi_hits_pre_v14")
            conn.executescript(V6_TABLES)  # re-adds ipi_payloads + ipi_hits
            conn.execute(
                "INSERT INTO ipi_hits (id, uuid, source_ip, user_agent,"
                " headers, body, token_valid, confidence, timestamp)"
                " SELECT id, uuid, source_ip, user_agent,"
                " headers, body, token_valid, confidence, timestamp"
                " FROM ipi_hits_pre_v14"
            )
            conn.execute("DROP TABLE ipi_hits_pre_v14")
            conn.commit()

            # Confirm via_tunnel does not yet exist.
            pre_cols = {row[1] for row in conn.execute("PRAGMA table_info(ipi_hits)").fetchall()}
            assert "via_tunnel" not in pre_cols

            # Seed a legacy hit (no via_tunnel column yet).
            conn.execute(
                """
                INSERT INTO ipi_hits (
                    id, uuid, source_ip, user_agent, headers, body,
                    token_valid, confidence, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-hit-1",
                    "campaign-uuid",
                    "10.0.0.1",
                    "Mozilla/5.0",
                    "{}",
                    None,
                    1,
                    "high",
                    "2025-01-01T00:00:00+00:00",
                ),
            )
            conn.commit()

            # Migrate to v14.
            migrate(conn)

            post_cols = {row[1] for row in conn.execute("PRAGMA table_info(ipi_hits)").fetchall()}
            assert "via_tunnel" in post_cols

            row = conn.execute(
                "SELECT id, uuid, via_tunnel FROM ipi_hits WHERE id = ?",
                ("legacy-hit-1",),
            ).fetchone()
            assert row is not None
            assert row[0] == "legacy-hit-1"
            assert row[1] == "campaign-uuid"
            # Pre-existing row has via_tunnel set to the column default 0.
            assert row[2] == 0

    def test_v14_with_missing_ipi_hits_is_safe(self, tmp_path: Path) -> None:
        """_migrate_v14 is a no-op (for the ALTER) when ipi_hits does not exist."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            # Build a partially-migrated DB with no ipi_* tables.
            conn.executescript(V1_TABLES)
            conn.executescript(V1_INDEXES)
            conn.execute("PRAGMA user_version = 13")
            conn.commit()

            _migrate_v14(conn)

            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == 14
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "ipi_hits" not in tables

    def test_v6_to_v14_end_to_end(self, tmp_path: Path) -> None:
        """A v6 DB (has ipi_hits but no v14 column) migrates cleanly to v14."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(V1_TABLES)
            conn.executescript(V1_INDEXES)
            conn.executescript(V6_TABLES)
            conn.executescript(V6_INDEXES)
            conn.execute("PRAGMA user_version = 6")
            conn.commit()

            migrate(conn)

            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_VERSION
            cols = {row[1] for row in conn.execute("PRAGMA table_info(ipi_hits)").fetchall()}
            assert "via_tunnel" in cols
