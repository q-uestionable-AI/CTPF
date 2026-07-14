"""CLI commands for managing targets."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from q_ai.core.cli.prompt import (
    build_teaching_tip,
    is_tty,
    parse_meta_flags,
    prompt_or_fail_multiple,
)
from q_ai.core.db import create_target, get_connection, get_target, list_targets
from q_ai.services.db_service import delete_target, resolve_partial_id

app = typer.Typer(name="targets", help="Manage MCP and inference targets.", no_args_is_help=True)
console = Console()


@app.command("list")
def list_cmd(
    db_path: Path | None = typer.Option(None, hidden=True),
) -> None:
    """List all targets."""
    with get_connection(db_path) as conn:
        targets = list_targets(conn)
    table = Table(title="Targets")
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Name")
    table.add_column("URI")
    for t in targets:
        table.add_row(
            t.id[:8],
            t.type,
            t.name,
            t.uri or "",
        )
    console.print(table)


@app.command(
    "add",
    epilog=(
        "Examples:\n"
        '  ctpf targets add "My Server" http://localhost:3000/sse\n'
        '  ctpf targets add "Local MCP" http://localhost:8080 --meta transport=sse\n'
        "  ctpf targets add  (interactive — prompts for name and URI)\n"
        "\n"
        "Common --meta keys: transport, environment, owner, notes"
    ),
)
def add_cmd(
    name: str | None = typer.Argument(
        None,
        help="Target name (prompted interactively if omitted).",
    ),
    uri: str | None = typer.Argument(
        None,
        help="Target URI (prompted interactively if omitted).",
    ),
    type: str = typer.Option(  # noqa: A002
        "server",
        "--type",
        "-t",
        help="Target type (e.g. server, endpoint).",
    ),
    meta: list[str] | None = typer.Option(
        None,
        "--meta",
        help="Metadata as key=value (repeatable). Example: --meta transport=sse",
    ),
    db_path: Path | None = typer.Option(None, hidden=True),
) -> None:
    """Add a new target.

    NAME and URI can be provided as positional arguments or entered
    interactively when running in a terminal. Type defaults to "server".

    Args:
        name: Target name. Prompted interactively if omitted in a TTY.
        uri: Target URI. Prompted interactively if omitted in a TTY.
        type: Target type (e.g. server, endpoint). Defaults to "server".
        meta: Metadata as repeatable key=value strings.
        db_path: Database path override (hidden, for testing).

    Raises:
        typer.Exit: If required args are missing in a non-TTY context.
    """
    all_provided = name is not None and uri is not None
    name, uri = prompt_or_fail_multiple(
        [
            ("NAME", name, "Target name"),
            ("URI", uri, "Target URI"),
        ]
    )

    meta_dict = parse_meta_flags(meta)

    with get_connection(db_path) as conn:
        tid = create_target(
            conn,
            type=type,
            name=name,
            uri=uri,
            metadata=meta_dict,
        )
    console.print(f"Created target {tid[:8]}")

    if not all_provided and is_tty():
        tip = build_teaching_tip("ctpf targets add", [name, uri])
        console.print(f"[dim]{tip}[/dim]")


@app.command("delete")
def delete_cmd(
    target_id: Annotated[str, typer.Argument(help="Target ID (full or partial).")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    db_path: Path | None = typer.Option(None, hidden=True),
) -> None:
    """Delete a target and unlink its associated runs.

    Accepts a partial ID prefix (minimum 8 characters). Associated
    runs keep their data but lose the target reference.

    Args:
        target_id: Full or partial target ID (minimum 8 characters).
        yes: Skip the confirmation prompt.
        db_path: Database path override (hidden, for testing).

    Raises:
        typer.Exit: If the target ID is too short, not found, or ambiguous.
    """
    if len(target_id) < 8:
        console.print("[red]Error: ID prefix must be at least 8 characters.[/red]")
        raise typer.Exit(code=1)

    with get_connection(db_path) as conn:
        try:
            full_id = resolve_partial_id(conn, "targets", target_id)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(code=1) from None

        target = get_target(conn, full_id)
        if target is None:
            console.print("[red]Error: target not found.[/red]")
            raise typer.Exit(code=1)

        if not yes:
            if not sys.stdin.isatty():
                console.print("[red]Error: --yes required in non-interactive mode.[/red]")
                raise typer.Exit(code=1)
            typer.confirm(
                f"Delete target '{target.name}'? Associated runs will be unlinked.",
                default=False,
                abort=True,
            )

        orphaned = delete_target(conn, full_id)

    console.print(f"Deleted target '{target.name}'. {orphaned} runs unlinked.")
