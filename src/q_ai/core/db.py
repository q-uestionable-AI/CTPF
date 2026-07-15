"""Database connection manager and CRUD helpers for q-ai."""

from __future__ import annotations

import datetime
import logging
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from q_ai.core.models import (
    Evidence,
    Finding,
    Run,
    RunStatus,
    Severity,
    Target,
    _dump_json,
)
from q_ai.core.schema import migrate

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".qai" / "qai.db"


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _new_id() -> str:
    """Generate a new UUID hex string."""
    return uuid.uuid4().hex


@contextmanager
def get_connection(
    db_path: Path | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a sqlite3 connection with WAL and FK enforcement.

    Creates the database file and parent directories if they don't exist.
    Runs schema migration on first connect.

    Args:
        db_path: Path to database file. Defaults to ~/.qai/qai.db.

    Yields:
        A sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    path = db_path or _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        migrate(conn)
        conn.commit()
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    module: str,
    name: str | None = None,
    target_id: str | None = None,
    parent_run_id: str | None = None,
    config: dict | None = None,
    run_id: str | None = None,
    source: str | None = None,
    started_at: str | None = None,
) -> str:
    """Insert a new run and return its ID.

    Args:
        conn: Active database connection.
        module: Name of the module that owns this run.
        name: Optional human-readable run name.
        target_id: Optional target reference.
        parent_run_id: Optional parent run for chained runs.
        config: Optional configuration dict.
        run_id: Optional pre-generated run ID. If None, a new UUID is generated.
        source: Optional provenance tag (e.g. "web", "cli").
        started_at: Optional ISO-8601 timestamp for the run's wall-clock start.
            When ``None`` (the default), the current UTC time is stamped at
            INSERT. Callers that do async work before persisting (e.g. network
            sweep/probe) should capture the timestamp before the work begins
            and pass it here so stored duration reflects wall-clock duration.

    Returns:
        The hex UUID of the newly created run.
    """
    run_id = run_id or _new_id()
    conn.execute(
        """
        INSERT INTO runs
            (id, module, name, target_id, parent_run_id,
             config, status, started_at, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            module,
            name,
            target_id,
            parent_run_id,
            _dump_json(config),
            int(RunStatus.PENDING),
            started_at or now_iso(),
            source,
        ),
    )
    return run_id


def update_run_status(
    conn: sqlite3.Connection,
    run_id: str,
    status: RunStatus,
    finished_at: str | None = None,
) -> None:
    """Update the status of a run.

    Automatically sets finished_at for terminal statuses
    (COMPLETED, FAILED, CANCELLED) if not provided.

    Args:
        conn: Active database connection.
        run_id: ID of the run to update.
        status: New status value.
        finished_at: Optional ISO timestamp override.
    """
    terminal = {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.PARTIAL,
    }
    if finished_at is None and status in terminal:
        finished_at = now_iso()
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
        (int(status), finished_at, run_id),
    )


def get_run(
    conn: sqlite3.Connection,
    run_id: str,
) -> Run | None:
    """Retrieve a run by ID.

    Args:
        conn: Active database connection.
        run_id: ID of the run to retrieve.

    Returns:
        A Run instance or None if not found.
    """
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return Run.from_row(dict(row))


def list_runs(
    conn: sqlite3.Connection,
    module: str | None = None,
    status: RunStatus | None = None,
    target_id: str | None = None,
    parent_run_id: str | None = None,
    name: str | None = None,
) -> list[Run]:
    """List runs with optional filters.

    Args:
        conn: Active database connection.
        module: Filter by module name.
        status: Filter by run status.
        target_id: Filter by target ID.
        parent_run_id: Filter by parent run ID.
        name: Filter by run name (workflow ID for parent runs).

    Returns:
        List of Run objects ordered by started_at descending.
    """
    query = "SELECT * FROM runs"
    conditions: list[str] = []
    params: list[object] = []

    if module is not None:
        conditions.append("module = ?")
        params.append(module)
    if status is not None:
        conditions.append("status = ?")
        params.append(int(status))
    if target_id is not None:
        conditions.append("target_id = ?")
        params.append(target_id)
    if parent_run_id is not None:
        conditions.append("parent_run_id = ?")
        params.append(parent_run_id)
    if name is not None:
        conditions.append("name = ?")
        params.append(name)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY started_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [Run.from_row(dict(row)) for row in rows]


# Module-specific tables that have a direct run_id FK
_MODULE_TABLES_WITH_RUN_ID = (
    "audit_scans",
    "inject_results",
    "proxy_sessions",
    "ipi_payloads",
    "cxp_test_results",
    "rxp_validations",
)


def delete_run_cascade(
    conn: sqlite3.Connection,
    run_id: str,
) -> list[str]:
    """Delete a parent run and all related data in cascade order.

    Deletes child runs, findings, evidence, and module-specific data within
    the current transaction. Returns a list of file paths that should be
    deleted after the transaction commits.

    Args:
        conn: Active database connection (caller manages transaction).
        run_id: ID of the parent run to delete.

    Returns:
        List of file paths to clean up after commit.

    Raises:
        ValueError: If run_id does not exist.
    """
    parent = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    if parent is None:
        raise ValueError(f"Run {run_id!r} not found")

    child_rows = conn.execute("SELECT id FROM runs WHERE parent_run_id = ?", (run_id,)).fetchall()
    child_ids = [r["id"] for r in child_rows]
    all_run_ids = [run_id, *child_ids]

    files_to_delete = _collect_files_for_cleanup(conn, all_run_ids)
    _delete_ipi_hits_for_runs(conn, all_run_ids)
    _delete_chain_data_for_runs(conn, all_run_ids)
    _delete_module_tables_for_runs(conn, all_run_ids)
    _delete_evidence_for_runs(conn, all_run_ids)
    _delete_findings_for_runs(conn, all_run_ids)
    _delete_run_rows(conn, child_ids, run_id)

    return files_to_delete


_QAI_DATA_DIR = Path.home() / ".qai"


def _collect_files_for_cleanup(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> list[str]:
    """Collect file paths from evidence and proxy sessions before deletion.

    Only returns paths that resolve to locations inside ~/.qai to prevent
    deletion of files outside the application data directory.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to collect files for.

    Returns:
        List of validated file paths to delete after commit.
    """
    if not run_ids:
        return []

    candidates: list[str] = []
    placeholders = ", ".join("?" for _ in run_ids)

    # Evidence file paths (directly linked to runs)
    rows = conn.execute(
        f"SELECT path FROM evidence WHERE run_id IN ({placeholders}) "  # noqa: S608
        "AND path IS NOT NULL",
        run_ids,
    ).fetchall()
    candidates.extend(r["path"] for r in rows)

    # Evidence linked via findings for these runs
    rows = conn.execute(
        f"SELECT e.path FROM evidence e JOIN findings f ON e.finding_id = f.id WHERE f.run_id IN ({placeholders}) AND e.path IS NOT NULL",  # noqa: S608, E501
        run_ids,
    ).fetchall()
    candidates.extend(r["path"] for r in rows)

    # Proxy session files
    rows = conn.execute(
        f"SELECT session_file FROM proxy_sessions WHERE run_id IN ({placeholders}) AND session_file IS NOT NULL",  # noqa: S608, E501
        run_ids,
    ).fetchall()
    candidates.extend(r["session_file"] for r in rows)

    return _validate_file_paths(candidates)


def _validate_file_paths(candidates: list[str]) -> list[str]:
    """Filter file paths to only those inside the application data directory.

    Args:
        candidates: Raw file paths from the database.

    Returns:
        Paths that resolve inside ~/.qai.
    """
    allowed_base = _QAI_DATA_DIR.resolve()
    safe: list[str] = []
    for raw in candidates:
        try:
            resolved = Path(raw).resolve()
        except (OSError, ValueError):
            logger.warning("Skipping unresolvable cleanup path: %s", raw)
            continue
        if resolved.is_relative_to(allowed_base):
            safe.append(str(resolved))
        else:
            logger.warning("Skipping cleanup path outside data dir: %s", raw)
    return safe


def _delete_ipi_hits_for_runs(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> None:
    """Delete IPI hits linked via payload UUIDs for the given runs.

    Args:
        conn: Active database connection.
        run_ids: Run IDs whose IPI payload UUIDs to match.
    """
    if not run_ids:
        return
    placeholders = ", ".join("?" for _ in run_ids)
    uuid_rows = conn.execute(
        f"SELECT uuid FROM ipi_payloads WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    ).fetchall()
    uuids = [r["uuid"] for r in uuid_rows]
    if not uuids:
        return
    uuid_ph = ", ".join("?" for _ in uuids)
    conn.execute(
        f"DELETE FROM ipi_hits WHERE uuid IN ({uuid_ph})",  # noqa: S608
        uuids,
    )


def _delete_chain_data_for_runs(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> None:
    """Delete chain step outputs and executions for the given runs.

    Step outputs must be deleted before executions due to FK constraint.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to delete chain data for.
    """
    if not run_ids:
        return
    placeholders = ", ".join("?" for _ in run_ids)
    exec_rows = conn.execute(
        f"SELECT id FROM chain_executions WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    ).fetchall()
    exec_ids = [r["id"] for r in exec_rows]
    if exec_ids:
        exec_ph = ", ".join("?" for _ in exec_ids)
        conn.execute(
            f"DELETE FROM chain_step_outputs WHERE execution_id IN ({exec_ph})",  # noqa: S608
            exec_ids,
        )
    conn.execute(
        f"DELETE FROM chain_executions WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    )


def _delete_module_tables_for_runs(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> None:
    """Delete module-specific data from all direct-FK tables.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to delete data for.
    """
    if not run_ids:
        return
    placeholders = ", ".join("?" for _ in run_ids)
    for table in _MODULE_TABLES_WITH_RUN_ID:
        conn.execute(
            f"DELETE FROM {table} WHERE run_id IN ({placeholders})",  # noqa: S608
            run_ids,
        )


def _delete_evidence_for_runs(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> None:
    """Delete evidence linked to runs directly or via findings.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to delete evidence for.
    """
    if not run_ids:
        return
    placeholders = ", ".join("?" for _ in run_ids)
    # Evidence linked via findings
    conn.execute(
        f"DELETE FROM evidence WHERE finding_id IN (SELECT id FROM findings WHERE run_id IN ({placeholders}))",  # noqa: S608, E501
        run_ids,
    )
    # Evidence linked directly to runs
    conn.execute(
        f"DELETE FROM evidence WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    )


def _delete_findings_for_runs(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> None:
    """Delete all findings for the given runs.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to delete findings for.
    """
    if not run_ids:
        return
    placeholders = ", ".join("?" for _ in run_ids)
    conn.execute(
        f"DELETE FROM findings WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    )


def _delete_run_rows(
    conn: sqlite3.Connection,
    child_ids: list[str],
    parent_id: str,
) -> None:
    """Delete child run rows then the parent run row.

    Args:
        conn: Active database connection.
        child_ids: IDs of child runs to delete first.
        parent_id: ID of the parent run to delete after children.
    """
    if child_ids:
        placeholders = ", ".join("?" for _ in child_ids)
        conn.execute(
            f"DELETE FROM runs WHERE id IN ({placeholders})",  # noqa: S608
            child_ids,
        )
    conn.execute("DELETE FROM runs WHERE id = ?", (parent_id,))


def export_run_bundle(
    conn: sqlite3.Connection,
    run_id: str,
) -> dict:
    """Export a complete run as a schema-versioned dict.

    Includes parent run metadata, child runs, findings, evidence references
    (without inline content), module-specific data, and target record.

    Args:
        conn: Active database connection.
        run_id: ID of the parent run to export.

    Returns:
        Dict ready for JSON serialization with schema_version key.

    Raises:
        ValueError: If run_id does not exist.
    """
    parent = get_run(conn, run_id)
    if parent is None:
        raise ValueError(f"Run {run_id!r} not found")

    children = list_runs(conn, parent_run_id=run_id)
    all_ids = [run_id] + [c.id for c in children]
    findings = list_findings(conn, run_ids=all_ids) if all_ids else []
    evidence_rows = _export_evidence_refs(conn, all_ids)
    module_data = _export_module_data(conn, all_ids)

    target = None
    target_id = parent.target_id or (parent.config or {}).get("target_id")
    if target_id:
        t = get_target(conn, target_id)
        if t is not None:
            target = t.to_dict()

    bundle: dict = {
        "schema_version": "run-bundle-v1",
        "run": parent.to_dict(),
        "child_runs": [c.to_dict() for c in children],
        "findings": [f.to_dict() for f in findings],
        "evidence": evidence_rows,
        "target": target,
    }
    bundle.update(module_data)
    return bundle


def _export_evidence_refs(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> list[dict]:
    """Export evidence references without inline content.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to collect evidence for.

    Returns:
        List of evidence dicts with metadata but no content blobs.
    """
    if not run_ids:
        return []
    placeholders = ", ".join("?" for _ in run_ids)
    ev_cols = "id, type, mime_type, storage, path, finding_id, run_id, hash, created_at"
    rows = conn.execute(
        f"SELECT {ev_cols} FROM evidence WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    ).fetchall()
    # Also evidence linked via findings (where run_id is NULL on evidence)
    finding_rows = conn.execute(
        f"SELECT e.id, e.type, e.mime_type, e.storage, e.path, e.finding_id, e.run_id, e.hash, e.created_at FROM evidence e JOIN findings f ON e.finding_id = f.id WHERE f.run_id IN ({placeholders}) AND e.run_id IS NULL",  # noqa: S608, E501
        run_ids,
    ).fetchall()
    all_rows = list(rows) + list(finding_rows)
    return [
        {
            "id": r["id"],
            "type": r["type"],
            "mime_type": r["mime_type"],
            "storage": r["storage"],
            "path": r["path"],
            "finding_id": r["finding_id"],
            "run_id": r["run_id"],
            "hash": r["hash"],
            "created_at": r["created_at"],
        }
        for r in all_rows
    ]


def _export_module_data(
    conn: sqlite3.Connection,
    run_ids: list[str],
) -> dict:
    """Export module-specific data for all run IDs.

    Args:
        conn: Active database connection.
        run_ids: Run IDs to export module data for.

    Returns:
        Dict with keys for each module table containing list of row dicts.
    """
    empty: dict[str, list[dict]] = {
        "audit_scans": [],
        "inject_results": [],
        "proxy_sessions": [],
        "chain_executions": [],
        "chain_step_outputs": [],
        "ipi_payloads": [],
        "cxp_test_results": [],
        "rxp_validations": [],
    }
    if not run_ids:
        return empty
    placeholders = ", ".join("?" for _ in run_ids)
    result: dict[str, list[dict]] = {}

    for table in (
        "audit_scans",
        "inject_results",
        "cxp_test_results",
        "rxp_validations",
        "ipi_payloads",
    ):
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE run_id IN ({placeholders})",  # noqa: S608
            run_ids,
        ).fetchall()
        result[table] = [dict(r) for r in rows]

    # Proxy sessions metadata
    proxy_cols = (
        "id, run_id, transport, server_name, message_count, "
        "duration_seconds, session_file, created_at"
    )
    rows = conn.execute(
        f"SELECT {proxy_cols} FROM proxy_sessions WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    ).fetchall()
    result["proxy_sessions"] = [dict(r) for r in rows]

    # Chain executions + step outputs
    exec_rows = conn.execute(
        f"SELECT * FROM chain_executions WHERE run_id IN ({placeholders})",  # noqa: S608
        run_ids,
    ).fetchall()
    result["chain_executions"] = [dict(r) for r in exec_rows]
    exec_ids = [r["id"] for r in exec_rows]
    if exec_ids:
        exec_ph = ", ".join("?" for _ in exec_ids)
        step_rows = conn.execute(
            f"SELECT * FROM chain_step_outputs "  # noqa: S608
            f"WHERE execution_id IN ({exec_ph})",
            exec_ids,
        ).fetchall()
        result["chain_step_outputs"] = [dict(r) for r in step_rows]
    else:
        result["chain_step_outputs"] = []

    return result


# ---------------------------------------------------------------------------
# Target CRUD
# ---------------------------------------------------------------------------


def create_target(
    conn: sqlite3.Connection,
    type: str,
    name: str,
    uri: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Insert a new target and return its ID.

    Args:
        conn: Active database connection.
        type: Target type (e.g. "server", "endpoint").
        name: Human-readable target name.
        uri: Optional URI or address.
        metadata: Optional metadata dict.

    Returns:
        The hex UUID of the newly created target.
    """
    target_id = _new_id()
    conn.execute(
        """
        INSERT INTO targets (id, type, name, uri, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            target_id,
            type,
            name,
            uri,
            _dump_json(metadata),
            now_iso(),
        ),
    )
    return target_id


def get_target(
    conn: sqlite3.Connection,
    target_id: str,
) -> Target | None:
    """Retrieve a target by ID.

    Args:
        conn: Active database connection.
        target_id: ID of the target to retrieve.

    Returns:
        A Target instance or None if not found.
    """
    row = conn.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
    if row is None:
        return None
    return Target.from_row(dict(row))


def list_targets(conn: sqlite3.Connection) -> list[Target]:
    """List all targets ordered by name alphabetically.

    Args:
        conn: Active database connection.

    Returns:
        List of Target objects.
    """
    rows = conn.execute("SELECT * FROM targets ORDER BY name COLLATE NOCASE").fetchall()
    return [Target.from_row(dict(row)) for row in rows]


# ---------------------------------------------------------------------------
# Finding CRUD
# ---------------------------------------------------------------------------


def create_finding(
    conn: sqlite3.Connection,
    run_id: str,
    module: str,
    category: str,
    severity: Severity,
    title: str,
    description: str | None = None,
    framework_ids: dict | None = None,
    mitigation: dict | None = None,
    source_ref: str | None = None,
) -> str:
    """Insert a new finding and return its ID.

    Args:
        conn: Active database connection.
        run_id: ID of the run that produced this finding.
        module: Name of the producing module.
        category: Finding category (e.g. "command_injection").
        severity: Severity level.
        title: Short human-readable title.
        description: Optional detailed description.
        framework_ids: Optional framework identifier mapping.
        mitigation: Optional MitigationGuidance dict for JSON serialization.
        source_ref: Optional source reference.

    Returns:
        The hex UUID of the newly created finding.
    """
    finding_id = _new_id()
    conn.execute(
        """
        INSERT INTO findings
            (id, run_id, module, category, severity,
             title, description, framework_ids,
             mitigation, source_ref, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            finding_id,
            run_id,
            module,
            category,
            int(severity),
            title,
            description,
            _dump_json(framework_ids),
            _dump_json(mitigation),
            source_ref,
            now_iso(),
        ),
    )
    return finding_id


def list_findings(
    conn: sqlite3.Connection,
    module: str | None = None,
    category: str | None = None,
    min_severity: Severity | None = None,
    run_id: str | None = None,
    run_ids: list[str] | None = None,
    target_id: str | None = None,
) -> list[Finding]:
    """List findings with optional filters.

    Args:
        conn: Active database connection.
        module: Filter by module name.
        category: Filter by finding category.
        min_severity: Filter findings at or above this severity.
        run_id: Filter by a single run ID.
        run_ids: Filter by multiple run IDs (WHERE run_id IN (...)).
        target_id: Filter by target ID (joins on runs table).

    Returns:
        List of Finding objects ordered by severity DESC,
        created_at DESC.
    """
    needs_join = target_id is not None
    if needs_join:
        query = "SELECT f.* FROM findings f JOIN runs r ON f.run_id = r.id"
    else:
        query = "SELECT * FROM findings"

    conditions: list[str] = []
    params: list[object] = []
    prefix = "f." if needs_join else ""

    if module is not None:
        conditions.append(f"{prefix}module = ?")
        params.append(module)
    if category is not None:
        conditions.append(f"{prefix}category = ?")
        params.append(category)
    if min_severity is not None:
        conditions.append(f"{prefix}severity >= ?")
        params.append(int(min_severity))
    if run_id is not None:
        conditions.append(f"{prefix}run_id = ?")
        params.append(run_id)
    if run_ids is not None:
        placeholders = ", ".join("?" for _ in run_ids)
        conditions.append(f"{prefix}run_id IN ({placeholders})")
        params.extend(run_ids)
    if target_id is not None:
        conditions.append("r.target_id = ?")
        params.append(target_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += f" ORDER BY {prefix}severity DESC, {prefix}created_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [Finding.from_row(dict(row)) for row in rows]


def get_previously_seen_finding_keys(
    conn: sqlite3.Connection,
    target_id: str,
    before_run_started_at: str,
    current_run_id: str,
) -> set[tuple[str, str]]:
    """Return (category, title) pairs from prior completed runs on the same target.

    Only considers top-level runs (parent_run_id IS NULL) with COMPLETED
    or PARTIAL status that started before the given timestamp.

    Args:
        conn: Active database connection.
        target_id: Target to scope the query.
        before_run_started_at: ISO timestamp upper bound (exclusive).
        current_run_id: Current run ID to exclude.

    Returns:
        Set of (category, title) tuples.
    """
    rows = conn.execute(
        """SELECT DISTINCT f.category, f.title
           FROM findings f
           JOIN runs r ON f.run_id = r.id
           WHERE r.target_id = ?
             AND r.parent_run_id IS NULL
             AND r.status IN (?, ?)
             AND r.started_at < ?
             AND r.id != ?""",
        (
            target_id,
            int(RunStatus.COMPLETED),
            int(RunStatus.PARTIAL),
            before_run_started_at,
            current_run_id,
        ),
    ).fetchall()
    return {(row["category"], row["title"]) for row in rows}


def get_prior_run_counts_by_target(
    conn: sqlite3.Connection,
    target_ids: list[str],
) -> dict[str, int]:
    """Return completed/partial top-level run counts per target.

    Args:
        conn: Active database connection.
        target_ids: Target IDs to query.

    Returns:
        Dict mapping target_id to run count. Targets with zero runs are omitted.
    """
    if not target_ids:
        return {}
    ph = ", ".join("?" for _ in target_ids)
    rows = conn.execute(
        f"""SELECT target_id, COUNT(*) as cnt
            FROM runs
            WHERE target_id IN ({ph})
              AND parent_run_id IS NULL
              AND status IN (?, ?)
            GROUP BY target_id""",  # noqa: S608
        [*target_ids, int(RunStatus.COMPLETED), int(RunStatus.PARTIAL)],
    ).fetchall()
    return {row["target_id"]: row["cnt"] for row in rows}


# ---------------------------------------------------------------------------
# Run Guidance
# ---------------------------------------------------------------------------


def save_run_guidance(
    conn: sqlite3.Connection,
    run_id: str,
    guidance_json: str,
) -> None:
    """Persist guidance JSON on an existing run.

    Args:
        conn: Active database connection.
        run_id: ID of the run to update.
        guidance_json: JSON-serialized RunGuidance string.

    Raises:
        ValueError: If run_id does not exist.
    """
    cursor = conn.execute(
        "UPDATE runs SET guidance = ? WHERE id = ?",
        (guidance_json, run_id),
    )
    if cursor.rowcount == 0:
        raise ValueError(f"Run {run_id!r} not found")


def get_run_guidance(
    conn: sqlite3.Connection,
    run_id: str,
) -> str | None:
    """Retrieve guidance JSON for a run.

    Args:
        conn: Active database connection.
        run_id: ID of the run to query.

    Returns:
        The raw JSON string or None if the run exists but has no guidance set.

    Raises:
        ValueError: If run_id does not exist.
    """
    row = conn.execute("SELECT guidance FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"Run {run_id!r} not found")
    result: str | None = row[0]
    return result


# ---------------------------------------------------------------------------
# Evidence CRUD
# ---------------------------------------------------------------------------


def create_evidence(
    conn: sqlite3.Connection,
    type: str,
    finding_id: str | None = None,
    run_id: str | None = None,
    mime_type: str | None = None,
    hash: str | None = None,
    storage: str = "inline",
    content: str | None = None,
    path: str | None = None,
) -> str:
    """Insert a new evidence record and return its ID.

    Args:
        conn: Active database connection.
        type: Evidence type (e.g. "request", "response").
        finding_id: Optional associated finding ID.
        run_id: Optional associated run ID.
        mime_type: Optional MIME type of the content.
        hash: Optional content hash for integrity.
        storage: Storage mode - "inline" or "file".
        content: Optional inline content.
        path: Optional file path for file-based storage.

    Returns:
        The hex UUID of the newly created evidence.
    """
    evidence_id = _new_id()
    conn.execute(
        """
        INSERT INTO evidence
            (id, type, finding_id, run_id, mime_type,
             hash, storage, content, path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence_id,
            type,
            finding_id,
            run_id,
            mime_type,
            hash,
            storage,
            content,
            path,
            now_iso(),
        ),
    )
    return evidence_id


def list_evidence(
    conn: sqlite3.Connection,
    finding_id: str | None = None,
    run_id: str | None = None,
) -> list[Evidence]:
    """List evidence records with optional filters.

    Args:
        conn: Active database connection.
        finding_id: Filter by associated finding ID.
        run_id: Filter by associated run ID.

    Returns:
        List of Evidence objects ordered by created_at descending.
    """
    query = "SELECT * FROM evidence"
    conditions: list[str] = []
    params: list[object] = []

    if finding_id is not None:
        conditions.append("finding_id = ?")
        params.append(finding_id)
    if run_id is not None:
        conditions.append("run_id = ?")
        params.append(run_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [Evidence.from_row(dict(row)) for row in rows]


# ---------------------------------------------------------------------------
# Settings CRUD
# ---------------------------------------------------------------------------


def get_setting(
    conn: sqlite3.Connection,
    key: str,
    default: str | None = None,
) -> str | None:
    """Retrieve a setting value by key.

    Args:
        conn: Active database connection.
        key: Setting key to look up.
        default: Value to return if the key is not found.

    Returns:
        The setting value, or default if not found.
    """
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    result: str = row[0]
    return result


def set_setting(
    conn: sqlite3.Connection,
    key: str,
    value: str,
) -> None:
    """Create or update a setting.

    Uses INSERT ... ON CONFLICT DO UPDATE to upsert.

    Args:
        conn: Active database connection.
        key: Setting key.
        value: Setting value.
    """
    conn.execute(
        """
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE
            SET value = excluded.value,
                updated_at = excluded.updated_at
        """,
        (key, value, now_iso()),
    )


def _delete_setting(conn: sqlite3.Connection, key: str) -> None:
    """Delete a setting by key. No-op if the key does not exist.

    Args:
        conn: Active database connection.
        key: Setting key to remove.
    """
    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
