"""Library CLI for inject MCP fixtures.

Provides subcommands for serving a malicious MCP server with configurable
payloads and listing available payload templates. Not registered on the
root ``qai`` CLI — invoke via ``python -m q_ai.inject`` or import
:data:`app` directly.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from q_ai.inject.models import InjectionTechnique
from q_ai.inject.payloads.loader import filter_templates, load_all_templates
from q_ai.inject.server import build_server

app = typer.Typer(no_args_is_help=True)
console = Console()

_VALID_TRANSPORTS = ("stdio", "streamable-http")


def _parse_technique(technique: str | None) -> InjectionTechnique | None:
    """Parse a technique string into InjectionTechnique enum.

    Args:
        technique: Technique name string, or None.

    Returns:
        Parsed InjectionTechnique, or None if input is None.

    Raises:
        typer.BadParameter: If the technique string is not a valid enum value.
    """
    if technique is None:
        return None
    try:
        return InjectionTechnique(technique)
    except ValueError:
        valid = ", ".join(t.value for t in InjectionTechnique)
        raise typer.BadParameter(
            f"Unknown technique: {technique}. Valid techniques: {valid}"
        ) from None


def _load_config_names(config_path: Path) -> list[str]:
    """Load payload name whitelist from a YAML config file.

    Args:
        config_path: Path to a YAML file containing a list of payload names.

    Returns:
        List of payload name strings.

    Raises:
        typer.BadParameter: If the file cannot be read or parsed.
    """
    import yaml

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"Cannot read config file: {exc}") from exc

    if not isinstance(raw, list) or not all(isinstance(n, str) for n in raw):
        raise typer.BadParameter("Config file must contain a YAML list of payload name strings")

    return raw


@app.command()
def serve(
    transport: str = typer.Option(..., help="Transport: 'stdio' or 'streamable-http'"),
    port: int = typer.Option(8888, help="Port for streamable-http listener"),
    payload_dir: str | None = typer.Option(None, help="Directory of payload templates to serve"),
    config: str | None = typer.Option(None, help="Payload configuration YAML file"),
) -> None:
    """Start a malicious MCP server serving configurable payloads.

    The server presents MCP tools with poisoned descriptions and/or
    returns injection payloads in tool responses. Connect any MCP
    client to test how it handles adversarial tool content.

    Args:
        transport: Transport mode (``stdio`` or ``streamable-http``).
        port: Port for streamable-http listener.
        payload_dir: Optional directory of payload templates.
        config: Optional YAML whitelist of payload names.

    Raises:
        typer.BadParameter: If transport or config is invalid.
    """
    if transport not in _VALID_TRANSPORTS:
        raise typer.BadParameter(
            f"Invalid transport: {transport}. Must be one of: {', '.join(_VALID_TRANSPORTS)}"
        )

    dir_path = Path(payload_dir) if payload_dir else None
    templates = load_all_templates(template_dir=dir_path)

    if config is not None:
        names = _load_config_names(Path(config))
        templates = [t for t in templates if t.name in names]

    if not templates:
        console.print(
            "[yellow]Warning:[/yellow] No payload templates loaded. Server will have no tools."
        )

    console.print(f"[green]Building server with {len(templates)} payload(s)...[/green]")
    server = build_server(templates)

    console.print(f"[green]Starting inject server via {transport}...[/green]")
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.settings.host = "0.0.0.0"  # noqa: S104  # nosec B104
        server.settings.port = port
        server.run(transport="streamable-http")


@app.command(name="list-payloads")
def list_payloads(
    technique: str | None = typer.Option(
        None,
        help="Filter by technique: 'description_poisoning', 'output_injection', "
        "'cross_tool_escalation'",
    ),
    target: str | None = typer.Option(
        None,
        help="Filter by target agent: 'claude', 'gpt', 'copilot', etc.",
    ),
) -> None:
    """List available injection payload templates.

    Args:
        technique: Optional technique filter string.
        target: Optional target-agent filter string.

    Raises:
        typer.BadParameter: If ``technique`` is not a valid enum value.
    """
    templates = load_all_templates()
    templates = filter_templates(
        templates, technique=_parse_technique(technique), target_agent=target
    )

    table = Table(title="Injection Payload Templates")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Technique", style="green", no_wrap=True)
    table.add_column("Tool Name", style="magenta", no_wrap=True)
    table.add_column("OWASP IDs", style="yellow", no_wrap=True)
    table.add_column("Description", max_width=50)

    for t in templates:
        desc = t.description[:50] + "..." if len(t.description) > 50 else t.description
        table.add_row(
            t.name,
            t.technique.value,
            t.tool_name,
            ", ".join(t.owasp_ids),
            desc,
        )

    console.print(table)
