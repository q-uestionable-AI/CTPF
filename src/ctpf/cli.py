"""Root CLI for the CTPF research harness."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ctpf import __version__
from ctpf.core.cli.config import app as config_app
from ctpf.core.cli.db import app as db_app
from ctpf.core.cli.findings import app as findings_app
from ctpf.core.cli.runs import app as runs_app
from ctpf.core.cli.targets import app as targets_app
from ctpf.experiment import app as experiment_app
from ctpf.proxy.cli import app as proxy_app

_DEFAULT_COMMAND_NAME = "ctpf"
_DISPLAY_NAME = "CTPF Research Harness"
_SUBTITLE = "Trust-boundary testing for agentic systems"

app = typer.Typer(
    name=_DEFAULT_COMMAND_NAME,
    help=f"{_DISPLAY_NAME} — {_SUBTITLE}.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)
console = Console()

# ---------------------------------------------------------------------------
# Help screen content
# ---------------------------------------------------------------------------


def _invocation_name(ctx: typer.Context) -> str:
    """Return the normalized executable name for the current invocation."""
    if not ctx.info_name:
        return _DEFAULT_COMMAND_NAME
    return Path(ctx.info_name).stem


def _quick_start(command_name: str) -> str:
    """Build quick-start examples for the invoked console entry point."""
    return f"""\
[bold]Quick Start[/bold]
  {command_name} proxy start ...                      Intercept MCP traffic
  {command_name} experiment run --help                List demonstrated experiments
  {command_name} targets add "My Server" http://...   Register a target
"""


def _hint_no_targets(command_name: str) -> str:
    """Build the no-targets hint for the invoked console entry point."""
    return (
        "[yellow]No targets found yet[/yellow] — run "
        f"[bold]{command_name} targets add[/bold] to get started."
    )


def _print_help_screen(command_name: str) -> None:
    """Print the grouped help screen with invocation-aware examples."""
    console.print(
        f"\n[bold]{command_name}[/bold] v{__version__} — {_DISPLAY_NAME} — {_SUBTITLE}.\n"
    )
    console.print(_quick_start(command_name))
    hint_no_targets = _hint_no_targets(command_name)

    # Contextual hint: check if any targets exist
    try:
        from ctpf.core.db import _DEFAULT_DB_PATH, get_connection, list_targets

        if not Path(_DEFAULT_DB_PATH).exists():
            console.print(f"  {hint_no_targets}\n")
        else:
            with get_connection() as conn:
                targets = list_targets(conn)
            if not targets:
                console.print(f"  {hint_no_targets}\n")
    except Exception:  # noqa: S110
        pass  # DB may not exist yet on first run

    console.print(
        f"[dim]Run {command_name} --help for the full command list, or "
        f"{command_name} <command> --help for details.[/dim]"
    )


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def version_callback(ctx: typer.Context, value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"{_invocation_name(ctx)} {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Root callback: a bare `ctpf` prints help; subcommands pass through
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

    _print_help_screen(_invocation_name(ctx))


# ---------------------------------------------------------------------------
# Register public subcommands
# ---------------------------------------------------------------------------

app.add_typer(
    proxy_app,
    name="proxy",
    help="MCP traffic interception and replay.",
    rich_help_panel="Observe",
)
app.add_typer(
    experiment_app,
    name="experiment",
    help="Run controlled CTPF experiments.",
    rich_help_panel="Research",
)
app.add_typer(targets_app, rich_help_panel="Start")
app.add_typer(runs_app, rich_help_panel="Manage")
app.add_typer(findings_app, rich_help_panel="Manage")
app.add_typer(config_app, rich_help_panel="Manage")
app.add_typer(db_app, rich_help_panel="Manage")
