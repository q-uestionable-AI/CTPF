"""``qai ipi listen`` / ``python -m q_ai.ipi listen`` — callback listener."""

from __future__ import annotations

from typing import Annotated

import typer

from q_ai.ipi.callback_state import build_state, delete_state, write_state
from q_ai.ipi.commands._shared import SUPPORTED_TUNNEL_PROVIDERS, app, console
from q_ai.ipi.server import start_server
from q_ai.ipi.tunnel import TunnelError, get_tunnel_adapter


def _run_listen_with_tunnel(
    *,
    host: str,
    port: int,
    tunnel_provider: str,
) -> None:
    """Start the listener behind a named tunnel provider.

    Args:
        host: Listener bind interface.
        port: Listener bind port.
        tunnel_provider: Tunnel provider name (e.g. ``"cloudflare"``).

    Raises:
        typer.Exit: On unknown provider, missing binary, or startup
            failure.
    """
    if tunnel_provider not in SUPPORTED_TUNNEL_PROVIDERS:
        supported = ", ".join(SUPPORTED_TUNNEL_PROVIDERS)
        console.print(f"[red]X Unknown tunnel provider: {tunnel_provider}[/red]")
        console.print(f"  Supported: {supported}")
        raise typer.Exit(1)

    adapter = get_tunnel_adapter(tunnel_provider)

    if not adapter.is_available():
        console.print(f"[red]X Tunnel provider '{tunnel_provider}' is not available[/red]")
        console.print()
        console.print(adapter.install_instructions())
        raise typer.Exit(1)

    console.print(
        f"[bold]Starting {tunnel_provider} tunnel to localhost:{port}... "
        "(this may take a few seconds)[/bold]"
    )
    try:
        public_url = adapter.start(local_port=port)
    except TunnelError as err:
        console.print(f"[red]X Failed to start {tunnel_provider} tunnel: {err}[/red]")
        adapter.stop()
        raise typer.Exit(1) from err

    console.print(f"[bold green]Tunnel active:[/bold green] [blue]{public_url}[/blue]")
    console.print(f"   Callback URL: [blue]{public_url}/c/<uuid>/<token>[/blue]")

    # State-file write is inside the try so an exception here still
    # triggers the finally block that stops the tunnel subprocess.
    # Otherwise a write_state() failure would leak a live cloudflared
    # process with a public tunnel URL.
    try:
        state = build_state(
            public_url=public_url,
            provider=tunnel_provider,
            local_host=host,
            local_port=port,
            manager="cli",
        )
        write_state(state)

        start_server(
            host=host,
            port=port,
            tunnel_provider=tunnel_provider,
        )
    finally:
        delete_state()
        adapter.stop()


@app.command()
def listen(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8080,
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "127.0.0.1",
    tunnel: Annotated[
        str | None,
        typer.Option(
            "--tunnel",
            help=(
                "Expose the listener via a public tunnel. Supported providers: "
                + ", ".join(SUPPORTED_TUNNEL_PROVIDERS)
                + ". Requires the provider's CLI binary on PATH."
            ),
        ),
    ] = None,
) -> None:
    """Start the callback listener server.

    Launches the FastAPI server that receives and logs callback
    requests from AI agents that execute the hidden payloads.

    With ``--tunnel cloudflare``, a Cloudflare Quick Tunnel is started
    alongside the listener, the public HTTPS URL is printed, and the
    listener records forwarded client IPs via the ``CF-Connecting-IP``
    header.
    """
    if tunnel is None:
        start_server(host=host, port=port)
        return

    _run_listen_with_tunnel(
        host=host,
        port=port,
        tunnel_provider=tunnel,
    )
