"""CLI commands for managing runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ctpf.core.db import delete_run_cascade, get_connection, list_runs
from ctpf.core.models import RunStatus
from ctpf.services.db_service import resolve_partial_id

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="runs",
    help="Manage stored research and operational runs.",
    no_args_is_help=True,
)
console = Console()


@app.command("list")
def list_cmd(
    module: str | None = typer.Option(None, "--module", "-m", help="Filter by module."),
    status: int | None = typer.Option(None, "--status", "-s", help="Filter by status (0-4)."),
    target: str | None = typer.Option(None, "--target", "-t", help="Filter by target ID."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
    db_path: Path | None = typer.Option(None, hidden=True),
) -> None:
    """List stored runs."""
    run_status = RunStatus(status) if status is not None else None
    with get_connection(db_path) as conn:
        runs = list_runs(conn, module=module, status=run_status, target_id=target)
    table = Table(title="Runs")
    table.add_column("ID", style="dim")
    table.add_column("Module")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Target")
    table.add_column("Started")
    for run in runs[:limit]:
        table.add_row(
            run.id[:8],
            run.module,
            run.name or "",
            run.status.name,
            run.target_id[:8] if run.target_id else "",
            str(run.started_at or ""),
        )
    console.print(table)


@app.command("delete")
def delete_cmd(
    run_id: Annotated[str, typer.Argument(help="Run ID (full or partial).")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    db_path: Path | None = typer.Option(None, hidden=True),
) -> None:
    """Delete a run and its associated findings and evidence.

    Accepts a partial ID prefix (minimum 8 characters).

    Args:
        run_id: Full or partial run ID (minimum 8 characters).
        yes: Skip the confirmation prompt.
        db_path: Database path override (hidden, for testing).

    Raises:
        typer.Exit: If the run ID is too short, not found, or ambiguous.
    """
    if len(run_id) < 8:
        console.print("[red]Error: ID prefix must be at least 8 characters.[/red]")
        raise typer.Exit(code=1)

    with get_connection(db_path) as conn:
        try:
            full_id = resolve_partial_id(conn, "runs", run_id)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(code=1) from None

        # Preview counts for the confirmation prompt — include child runs,
        # which delete_run_cascade also removes.
        findings_count: int = conn.execute(
            "SELECT COUNT(*) FROM findings "
            "WHERE run_id = ? OR run_id IN (SELECT id FROM runs WHERE parent_run_id = ?)",
            (full_id, full_id),
        ).fetchone()[0]
        evidence_count: int = conn.execute(
            "SELECT COUNT(*) FROM evidence "
            "WHERE run_id = ? "
            "   OR run_id IN (SELECT id FROM runs WHERE parent_run_id = ?) "
            "   OR finding_id IN ("
            "      SELECT id FROM findings "
            "      WHERE run_id = ? OR run_id IN (SELECT id FROM runs WHERE parent_run_id = ?)"
            "   )",
            (full_id, full_id, full_id, full_id),
        ).fetchone()[0]

        if not yes:
            if not sys.stdin.isatty():
                console.print("[red]Error: --yes required in non-interactive mode.[/red]")
                raise typer.Exit(code=1)
            typer.confirm(
                f"Delete run '{full_id[:8]}'? This will delete "
                f"{findings_count} findings and {evidence_count} evidence records.",
                default=False,
                abort=True,
            )

        try:
            files_to_delete = delete_run_cascade(conn, full_id)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(code=1) from None

    _cleanup_files(files_to_delete)

    console.print(
        f"Deleted run '{full_id[:8]}' with "
        f"{findings_count} findings and "
        f"{evidence_count} evidence records."
    )


def _cleanup_files(files: list[str]) -> None:
    """Delete files returned by delete_run_cascade, logging failures."""
    for file_path in files:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete file: %s", file_path)
