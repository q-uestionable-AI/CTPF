"""CLI commands for managing findings."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ctpf.core.db import get_connection, list_findings
from ctpf.core.models import Severity

app = typer.Typer(name="findings", help="Manage stored findings.", no_args_is_help=True)
console = Console()


@app.command("list")
def list_cmd(
    module: str | None = typer.Option(None, "--module", "-m", help="Filter by module."),
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category."),
    severity: str | None = typer.Option(
        None,
        "--severity",
        "-s",
        help="Minimum severity (INFO/LOW/MEDIUM/HIGH/CRITICAL).",
    ),
    target: str | None = typer.Option(None, "--target", "-t", help="Filter by target ID."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
    db_path: Path | None = typer.Option(None, hidden=True),
) -> None:
    """List stored findings."""
    min_sev = Severity[severity.upper()] if severity else None
    with get_connection(db_path) as conn:
        findings = list_findings(
            conn,
            module=module,
            category=category,
            min_severity=min_sev,
            target_id=target,
        )
    table = Table(title="Findings")
    table.add_column("ID", style="dim")
    table.add_column("Module")
    table.add_column("Category", no_wrap=True)
    table.add_column("Severity")
    table.add_column("Title")
    table.add_column("Created")
    for f in findings[:limit]:
        table.add_row(
            f.id[:8],
            f.module,
            f.category,
            f.severity.name,
            f.title,
            str(f.created_at or ""),
        )
    console.print(table)
