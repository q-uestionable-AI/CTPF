"""Root CLI for the CTPF research harness."""

from __future__ import annotations

import typer
from rich.console import Console

from q_ai import __version__
from q_ai.core.cli.config import app as config_app
from q_ai.core.cli.db import app as db_app
from q_ai.core.cli.findings import app as findings_app
from q_ai.core.cli.runs import app as runs_app
from q_ai.core.cli.targets import app as targets_app
from q_ai.proxy.cli import app as proxy_app

app = typer.Typer(
    name="qai",
    help="CTPF research harness — MCP observation, controlled fixtures, and evidence.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)
console = Console()

# ---------------------------------------------------------------------------
# Help screen content
# ---------------------------------------------------------------------------

_QUICK_START = """\
[bold]Quick Start[/bold]
  qai proxy start ...                      Intercept MCP traffic
  qai targets add "My Server" http://...   Register a target
"""

_HINT_NO_TARGETS = (
    "[yellow]No targets found yet[/yellow] — run [bold]qai targets add[/bold] to get started."
)


def _print_help_screen() -> None:
    """Print the grouped help screen with quick-start examples."""
    console.print(
        f"\n[bold]qai[/bold] v{__version__} — CTPF research harness — "
        "MCP observation, controlled fixtures, and evidence.\n"
    )
    console.print(_QUICK_START)

    # Contextual hint: check if any targets exist
    try:
        from pathlib import Path

        from q_ai.core.db import _DEFAULT_DB_PATH, get_connection, list_targets

        if not Path(_DEFAULT_DB_PATH).exists():
            console.print(f"  {_HINT_NO_TARGETS}\n")
        else:
            with get_connection() as conn:
                targets = list_targets(conn)
            if not targets:
                console.print(f"  {_HINT_NO_TARGETS}\n")
    except Exception:  # noqa: S110
        pass  # DB may not exist yet on first run

    console.print(
        "[dim]Run qai --help for full command list, or qai <command> --help for details.[/dim]"
    )


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"qai {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Root callback: bare `qai` prints help, subcommands pass through
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Root callback — prints help screen when invoked without a subcommand.

    Args:
        ctx: Typer context for subcommand detection.
        version: If True, print version and exit (eager option).
    """
    if ctx.invoked_subcommand is not None:
        return

    _print_help_screen()


# ---------------------------------------------------------------------------
# Register transitional public subcommands
# ---------------------------------------------------------------------------

app.add_typer(
    proxy_app,
    name="proxy",
    help="MCP traffic interception and replay.",
    rich_help_panel="Observe",
)
app.add_typer(targets_app, rich_help_panel="Start")
app.add_typer(runs_app, rich_help_panel="Manage")
app.add_typer(findings_app, rich_help_panel="Manage")
app.add_typer(config_app, rich_help_panel="Manage")
app.add_typer(db_app, rich_help_panel="Manage")
