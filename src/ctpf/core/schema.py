"""Database schema definitions and migration for CTPF."""

from __future__ import annotations

import sqlite3

CURRENT_VERSION = 15

V1_TABLES = """
CREATE TABLE IF NOT EXISTS targets (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    uri TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    parent_run_id TEXT REFERENCES runs(id),
    module TEXT NOT NULL,
    name TEXT,
    target_id TEXT REFERENCES targets(id),
    config TEXT,
    status INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    module TEXT NOT NULL,
    category TEXT NOT NULL,
    severity INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    framework_ids TEXT,
    source_ref TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    finding_id TEXT REFERENCES findings(id),
    run_id TEXT REFERENCES runs(id),
    type TEXT NOT NULL,
    mime_type TEXT,
    hash TEXT,
    storage TEXT NOT NULL DEFAULT 'inline',
    content TEXT,
    path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL
);
"""

V1_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_runs_parent_run_id ON runs(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_runs_module ON runs(module);
CREATE INDEX IF NOT EXISTS idx_runs_target_id ON runs(target_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_findings_run_id ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_module ON findings(module);
CREATE INDEX IF NOT EXISTS idx_findings_category ON findings(category);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_evidence_finding_id ON evidence(finding_id);
CREATE INDEX IF NOT EXISTS idx_evidence_run_id ON evidence(run_id);
"""


V2_TABLES = """
CREATE TABLE IF NOT EXISTS audit_scans (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    run_id TEXT NOT NULL REFERENCES runs(id),
    transport TEXT NOT NULL,
    server_name TEXT,
    server_version TEXT,
    scanners_run TEXT,
    finding_count INTEGER DEFAULT 0,
    scan_duration_seconds REAL,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

V2_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_audit_scans_run_id ON audit_scans(run_id);
"""

V3_TABLES = """
CREATE TABLE IF NOT EXISTS inject_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    payload_name TEXT NOT NULL,
    technique TEXT NOT NULL,
    outcome TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    evidence TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

V3_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_inject_results_run_id ON inject_results(run_id);
"""

V4_TABLES = """
CREATE TABLE IF NOT EXISTS proxy_sessions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    run_id TEXT NOT NULL REFERENCES runs(id),
    transport TEXT NOT NULL,
    server_name TEXT,
    message_count INTEGER DEFAULT 0,
    duration_seconds REAL,
    session_file TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

V4_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_proxy_sessions_run_id ON proxy_sessions(run_id);
"""

V5_TABLES = """
CREATE TABLE IF NOT EXISTS chain_executions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    run_id TEXT NOT NULL REFERENCES runs(id),
    chain_id TEXT NOT NULL,
    chain_name TEXT,
    dry_run INTEGER NOT NULL DEFAULT 1,
    template_path TEXT,
    target_config TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    trust_boundaries TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS chain_step_outputs (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    execution_id TEXT NOT NULL REFERENCES chain_executions(id),
    step_id TEXT NOT NULL,
    module TEXT NOT NULL,
    technique TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    status TEXT,
    artifacts TEXT,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

V5_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chain_executions_run_id ON chain_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_chain_step_outputs_execution_id ON chain_step_outputs(execution_id);
"""

V6_TABLES = """
CREATE TABLE IF NOT EXISTS ipi_payloads (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    uuid TEXT NOT NULL,
    token TEXT NOT NULL,
    filename TEXT,
    output_path TEXT,
    format TEXT NOT NULL,
    technique TEXT NOT NULL,
    payload_style TEXT,
    payload_type TEXT,
    callback_url TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ipi_hits (
    id TEXT PRIMARY KEY,
    uuid TEXT NOT NULL,
    source_ip TEXT,
    user_agent TEXT,
    headers TEXT,
    body TEXT,
    token_valid INTEGER NOT NULL DEFAULT 0,
    confidence TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""

V6_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_ipi_payloads_uuid ON ipi_payloads(uuid);
CREATE INDEX IF NOT EXISTS idx_ipi_payloads_run_id ON ipi_payloads(run_id);
CREATE INDEX IF NOT EXISTS idx_ipi_hits_uuid ON ipi_hits(uuid);
"""


V7_TABLES = """
CREATE TABLE IF NOT EXISTS cxp_test_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    campaign_id TEXT NOT NULL,
    technique_id TEXT NOT NULL,
    assistant TEXT NOT NULL,
    model TEXT,
    trigger_prompt TEXT NOT NULL,
    capture_mode TEXT NOT NULL,
    captured_files TEXT,
    raw_output TEXT NOT NULL,
    validation_result TEXT NOT NULL DEFAULT 'pending',
    validation_details TEXT,
    notes TEXT,
    rules_inserted TEXT,
    format_id TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

V7_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_cxp_test_results_run_id ON cxp_test_results(run_id);
CREATE INDEX IF NOT EXISTS idx_cxp_test_results_campaign_id ON cxp_test_results(campaign_id);
"""


V8_TABLES = """
CREATE TABLE IF NOT EXISTS rxp_validations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    model_id TEXT NOT NULL,
    profile_id TEXT,
    total_queries INTEGER NOT NULL,
    poison_retrievals INTEGER NOT NULL,
    retrieval_rate REAL NOT NULL,
    mean_poison_rank REAL,
    top_k INTEGER NOT NULL,
    results_json TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

V8_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_rxp_validations_run_id ON rxp_validations(run_id);
CREATE INDEX IF NOT EXISTS idx_rxp_validations_model_id ON rxp_validations(model_id);
"""


V15_TABLES = """
CREATE TABLE IF NOT EXISTS automation_policies (
    id TEXT PRIMARY KEY,
    policy_json TEXT NOT NULL,
    policy_digest TEXT NOT NULL UNIQUE,
    signature TEXT NOT NULL,
    key_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS automation_grants (
    id TEXT PRIMARY KEY,
    grant_json TEXT NOT NULL,
    signature TEXT NOT NULL,
    key_id TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    policy_id TEXT NOT NULL REFERENCES automation_policies(id),
    policy_digest TEXT NOT NULL,
    scenario_fingerprint TEXT NOT NULL,
    target_fingerprints TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    bound_run_id TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS automation_runs (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    spec_json TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    policy_id TEXT NOT NULL REFERENCES automation_policies(id),
    policy_digest TEXT NOT NULL,
    grant_id TEXT NOT NULL REFERENCES automation_grants(id),
    state TEXT NOT NULL CHECK (
        state IN (
            'READY', 'RUNNING', 'CANCEL_REQUESTED', 'COMPLETED',
            'FAILED', 'CANCELLED', 'INTERRUPTED'
        )
    ),
    scenario_fingerprint TEXT NOT NULL,
    target_fingerprints TEXT NOT NULL,
    output_root_id TEXT NOT NULL,
    run_root TEXT,
    manifest_path TEXT,
    budget_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    result_json TEXT,
    error_json TEXT,
    lease_id TEXT,
    lease_heartbeat_at TEXT,
    cancel_requested_at TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS automation_events (
    run_id TEXT NOT NULL REFERENCES automation_runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);
"""

V15_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_automation_policies_status
    ON automation_policies(status, expires_at);
CREATE INDEX IF NOT EXISTS idx_automation_grants_spec_digest
    ON automation_grants(spec_digest);
CREATE INDEX IF NOT EXISTS idx_automation_grants_expires_at
    ON automation_grants(expires_at);
CREATE INDEX IF NOT EXISTS idx_automation_runs_state
    ON automation_runs(state, updated_at);
CREATE INDEX IF NOT EXISTS idx_automation_runs_policy_id
    ON automation_runs(policy_id);
CREATE INDEX IF NOT EXISTS idx_automation_runs_grant_id
    ON automation_runs(grant_id);
CREATE INDEX IF NOT EXISTS idx_automation_events_run_id
    ON automation_events(run_id, sequence);
"""


def _migrate_v9(conn: sqlite3.Connection) -> None:
    """V9: Add mitigation column to findings table.

    Guards against: (1) findings table not existing in partial-migration
    test scenarios, and (2) column already existing for idempotency.
    Uses conn.execute() instead of executescript() to avoid implicit COMMIT.
    """
    has_findings = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='findings'"
    ).fetchone()
    if not has_findings:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(findings)").fetchall()}
    if "mitigation" not in columns:
        conn.execute("ALTER TABLE findings ADD COLUMN mitigation TEXT")
    conn.execute("PRAGMA user_version = 9")


def _migrate_v10(conn: sqlite3.Connection) -> None:
    """V10: Add guidance column to runs table.

    Guards against: (1) runs table not existing in partial-migration
    test scenarios, and (2) column already existing for idempotency.
    Uses conn.execute() instead of executescript() to avoid implicit COMMIT.
    """
    has_runs = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
    ).fetchone()
    if not has_runs:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "guidance" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN guidance TEXT")
    conn.execute("PRAGMA user_version = 10")


def _migrate_v11(conn: sqlite3.Connection) -> None:
    """V11: Add source column to runs table for run provenance.

    Guards against: (1) runs table not existing in partial-migration
    test scenarios, and (2) column already existing for idempotency.
    Uses conn.execute() instead of executescript() to avoid implicit COMMIT.
    """
    has_runs = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
    ).fetchone()
    if not has_runs:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "source" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN source TEXT")
    conn.execute("PRAGMA user_version = 11")


def _migrate_v12(conn: sqlite3.Connection) -> None:
    """V12: Add chain correlation columns to proxy_sessions table.

    Adds chain_run_id and chain_step_id so proxy traffic can be queried
    by chain execution context after the fact.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='proxy_sessions'"
    ).fetchone()
    if has_table:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(proxy_sessions)").fetchall()}
        if "chain_run_id" not in columns:
            conn.execute("ALTER TABLE proxy_sessions ADD COLUMN chain_run_id TEXT")
        if "chain_step_id" not in columns:
            conn.execute("ALTER TABLE proxy_sessions ADD COLUMN chain_step_id TEXT")
    conn.execute("PRAGMA user_version = 12")


def _migrate_v13(conn: sqlite3.Connection) -> None:
    """V13: Add template_id column to ipi_payloads for per-run provenance.

    Records which ``DocumentTemplate`` framed each generated payload so the
    Historical exports can display the template
    alongside format/technique. Nullable by design — legacy rows predate
    the template system and remain NULL rather than being backfilled.

    Guards against: (1) ipi_payloads table not existing in partial-migration
    test scenarios, and (2) column already existing for idempotency.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ipi_payloads'"
    ).fetchone()
    if has_table:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ipi_payloads)").fetchall()}
        if "template_id" not in columns:
            conn.execute("ALTER TABLE ipi_payloads ADD COLUMN template_id TEXT")
    conn.execute("PRAGMA user_version = 13")


def _migrate_v14(conn: sqlite3.Connection) -> None:
    """V14: Add via_tunnel column to ipi_hits for forwarded-header provenance.

    Records whether each hit's ``source_ip`` was resolved from a trusted
    forwarded header (tunnel mode — currently ``CF-Connecting-IP``) or
    from the TCP peer (direct). Lets the hit feed render a qualifier
    badge next to the IP so researchers running mixed campaigns can
    distinguish tunnel-routed hits from direct ones during evidence
    interpretation and tunnel-health debugging.

    INTEGER-as-boolean matches the existing ``token_valid`` column on the
    same table. ``DEFAULT 0`` means legacy rows (which predate the tunnel
    feature) read as ``direct`` — accurate rather than backfilled.

    Guards against: (1) ipi_hits table not existing in partial-migration
    test scenarios, and (2) column already existing for idempotency.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ipi_hits'"
    ).fetchone()
    if has_table:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ipi_hits)").fetchall()}
        if "via_tunnel" not in columns:
            conn.execute("ALTER TABLE ipi_hits ADD COLUMN via_tunnel INTEGER NOT NULL DEFAULT 0")
    conn.execute("PRAGMA user_version = 14")


def _migrate_v15(conn: sqlite3.Connection) -> None:
    """V15: Add isolated policy and lifecycle tables for governed automation."""
    conn.executescript(V15_TABLES)
    conn.executescript(V15_INDEXES)
    conn.execute("PRAGMA user_version = 15")


def migrate(conn: sqlite3.Connection) -> None:  # noqa: C901, PLR0912
    """Run schema migrations up to CURRENT_VERSION.

    Checks PRAGMA user_version and runs any pending migrations sequentially.
    Version 1 creates all shared tables and indexes.
    Version 2 adds the audit_scans table.
    Version 3 adds the inject_results table.
    Version 4 adds the proxy_sessions table.
    Version 5 adds chain_executions and chain_step_outputs tables.
    Version 6 adds ipi_payloads and ipi_hits tables.
    Version 7 adds cxp_test_results table.
    Version 8 adds rxp_validations table.
    Version 9 adds mitigation column to findings table.
    Version 10 adds guidance column to runs table.
    Version 11 adds source column to runs table for provenance.
    Version 12 adds chain correlation columns to proxy_sessions table.
    Version 13 adds template_id column to ipi_payloads for per-run provenance.
    Version 14 adds via_tunnel column to ipi_hits for forwarded-header provenance.
    Version 15 adds isolated governed-automation policy and lifecycle tables.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        conn.executescript(V1_TABLES)
        conn.executescript(V1_INDEXES)
        conn.execute("PRAGMA user_version = 1")
        version = 1
    if version < 2:
        conn.executescript(V2_TABLES)
        conn.executescript(V2_INDEXES)
        conn.execute("PRAGMA user_version = 2")
    if version < 3:
        conn.executescript(V3_TABLES)
        conn.executescript(V3_INDEXES)
        conn.execute("PRAGMA user_version = 3")
    if version < 4:
        conn.executescript(V4_TABLES)
        conn.executescript(V4_INDEXES)
        conn.execute("PRAGMA user_version = 4")
    if version < 5:
        conn.executescript(V5_TABLES)
        conn.executescript(V5_INDEXES)
        conn.execute("PRAGMA user_version = 5")
    if version < 6:
        conn.executescript(V6_TABLES)
        conn.executescript(V6_INDEXES)
        conn.execute("PRAGMA user_version = 6")
    if version < 7:
        conn.executescript(V7_TABLES)
        conn.executescript(V7_INDEXES)
        conn.execute("PRAGMA user_version = 7")
    if version < 8:
        conn.executescript(V8_TABLES)
        conn.executescript(V8_INDEXES)
        conn.execute("PRAGMA user_version = 8")
    if version < 9:
        _migrate_v9(conn)
    if version < 10:
        _migrate_v10(conn)
    if version < 11:
        _migrate_v11(conn)
    if version < 12:
        _migrate_v12(conn)
    if version < 13:
        _migrate_v13(conn)
    if version < 14:
        _migrate_v14(conn)
    if version < 15:
        _migrate_v15(conn)
