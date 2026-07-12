"""Database management service — backup, reset, and delete operations."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_BACKUPS_DIR = Path.home() / ".qai" / "backups"

# Tables cleared by reset_database (order matters for FK constraints).
# Child tables must be deleted before parent tables.
_RESET_TABLES = (
    # IPI module
    "ipi_hits",
    "ipi_payloads",
    # Historical module tables (schema retained for existing DBs)
    "chain_step_outputs",
    "chain_executions",
    "cxp_test_results",
    "rxp_validations",
    "audit_scans",
    "inject_results",
    "proxy_sessions",
    # Core tables
    "evidence",
    "findings",
    "runs",
    "targets",
)


def backup_database(
    db_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Copy the database file to a backup location.

    Args:
        db_path: Path to the source database file.
        output_path: Destination path. If None, a timestamped file is
            created under ``~/.qai/backups/``.

    Returns:
        Path to the created backup file.

    Raises:
        FileNotFoundError: If db_path does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if output_path is None:
        _BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
        output_path = _BACKUPS_DIR / f"qai-{stamp}.db"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use SQLite backup API for WAL-safe backup (shutil.copy2 would miss -wal/-shm)
    # Explicit close() required on Windows to release file handles
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(output_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return output_path


def reset_database(
    conn: sqlite3.Connection,
    db_path: Path,
    *,
    auto_backup: bool = True,
) -> Path | None:
    """Delete all operational data while preserving settings.

    Clears: ipi_hits, ipi_payloads, evidence, findings, runs, targets.
    Preserves: settings, schema_version.

    Args:
        conn: Active database connection.
        db_path: Path to the database file (used for backup).
        auto_backup: If True, create a backup before resetting.

    Returns:
        Path to the backup file if created, None otherwise.
    """
    backup_path = None
    if auto_backup:
        backup_path = backup_database(db_path)

    for table in _RESET_TABLES:
        conn.execute(f"DELETE FROM {table}")  # noqa: S608

    conn.commit()
    conn.execute("VACUUM")
    return backup_path


def delete_target(
    conn: sqlite3.Connection,
    target_id: str,
) -> int:
    """Delete a target and orphan its associated runs.

    Sets ``target_id = NULL`` on all runs referencing this target,
    then deletes the target record.

    Args:
        conn: Active database connection.
        target_id: Full UUID of the target to delete.

    Returns:
        Count of runs that were orphaned (had target_id nullified).

    Raises:
        ValueError: If the target does not exist.
    """
    row = conn.execute("SELECT id FROM targets WHERE id = ?", (target_id,)).fetchone()
    if row is None:
        raise ValueError(f"Target {target_id!r} not found")

    cursor = conn.execute(
        "UPDATE runs SET target_id = NULL WHERE target_id = ?",
        (target_id,),
    )
    orphaned = cursor.rowcount

    conn.execute("DELETE FROM targets WHERE id = ?", (target_id,))
    return orphaned


def resolve_partial_id(
    conn: sqlite3.Connection,
    table: str,
    partial_id: str,
) -> str:
    """Resolve a partial ID prefix to a full UUID.

    Args:
        conn: Active database connection.
        table: Table name to query (must be ``targets`` or ``runs``).
        partial_id: ID prefix (typically first 8 chars).

    Returns:
        The full UUID string.

    Raises:
        ValueError: If zero or multiple matches found.
    """
    if table not in ("targets", "runs"):
        raise ValueError(f"Invalid table: {table!r}")

    rows = conn.execute(
        f"SELECT id FROM {table} WHERE id LIKE ?",  # noqa: S608
        (f"{partial_id}%",),
    ).fetchall()

    if len(rows) == 0:
        raise ValueError(f"No {table[:-1]} found matching '{partial_id}'")
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous ID '{partial_id}' matches {len(rows)} {table}. Provide more characters."
        )
    result: str = rows[0]["id"]
    return result
