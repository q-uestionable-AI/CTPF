"""Programmable lifecycle for the MCP proxy pipeline."""

from __future__ import annotations

import asyncio
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import cast

from q_ai.mcp.models import Transport
from q_ai.mcp.transport import TransportAdapter
from q_ai.proxy.adapters.sse import SseClientAdapter, SseServerAdapter
from q_ai.proxy.adapters.stdio import StdioClientAdapter, StdioServerAdapter
from q_ai.proxy.adapters.streamable_http import (
    StreamableHttpClientAdapter,
    StreamableHttpServerAdapter,
)
from q_ai.proxy.constants import LISTEN_HOST, stdio_subprocess_env
from q_ai.proxy.pipeline import PipelineSession, run_pipeline

DEFAULT_LISTEN_PORT = 8080
_NETWORK_TRANSPORTS = frozenset({Transport.SSE, Transport.STREAMABLE_HTTP})


@dataclass(frozen=True)
class ProxyRuntimeConfig:
    """Validated target and listener configuration for a proxy run.

    Args:
        transport: How the proxy connects to the target MCP server.
        server_command: Target command for stdio transport.
        server_url: Target endpoint for SSE or Streamable HTTP.
        listen_transport: Optional network transport exposed to the MCP client.
            When omitted, the client-facing transport is stdio.
        listen_port: Loopback port for a network client-facing transport.
    """

    transport: Transport
    server_command: str | None = None
    server_url: str | None = None
    listen_transport: Transport | None = None
    listen_port: int = DEFAULT_LISTEN_PORT

    def __post_init__(self) -> None:
        """Reject incomplete or unsafe runtime configurations."""
        if not 1 <= self.listen_port <= 65535:
            raise ValueError("listen_port must be between 1 and 65535")
        if self.listen_transport is not None and self.listen_transport not in _NETWORK_TRANSPORTS:
            raise ValueError("listen_transport must be sse or streamable-http")
        if self.transport == Transport.STDIO:
            _split_command(self.server_command)
        elif not self.server_url or not self.server_url.strip():
            raise ValueError("server_url is required for network target transports")


class ProxyRuntime:
    """Own adapter lifecycle and execute one proxy pipeline session.

    Args:
        session: Pipeline dependencies, capture store, and callbacks.
    """

    def __init__(self, session: PipelineSession) -> None:
        self._session = session
        self._ready = asyncio.Event()
        self._running = False
        self._client_adapter: TransportAdapter | None = None
        self._server_adapter: TransportAdapter | None = None

    @property
    def ready(self) -> bool:
        """Whether both adapters are active and the pipeline has started."""
        return self._ready.is_set()

    async def wait_until_ready(self) -> None:
        """Wait until both adapters are active."""
        await self._ready.wait()

    async def run(self, config: ProxyRuntimeConfig) -> None:
        """Build configured adapters and run until either side disconnects.

        Args:
            config: Validated target and listener configuration.
        """
        self._begin()
        try:
            async with AsyncExitStack() as stack:
                server = cast(
                    TransportAdapter,
                    await stack.enter_async_context(_build_server_adapter(config)),
                )
                client = cast(
                    TransportAdapter,
                    await stack.enter_async_context(_build_client_adapter(config)),
                )
                await self._execute(client, server)
        finally:
            self._finish()

    async def run_with_adapters(
        self,
        client_adapter: TransportAdapter,
        server_adapter: TransportAdapter,
    ) -> None:
        """Run with already-entered adapters without assuming context ownership.

        Args:
            client_adapter: Client-facing adapter.
            server_adapter: Server-facing adapter.
        """
        self._begin()
        try:
            await self._execute(client_adapter, server_adapter)
        finally:
            self._finish()

    async def stop(self) -> None:
        """Drop held intercepts and close adapters so the pipeline exits.

        Closing adapters alone is not enough when a forward loop is awaiting
        an interactive intercept decision; held messages must be released too.
        """
        self._session.intercept_engine.drop_held()
        adapters = [
            adapter
            for adapter in (self._client_adapter, self._server_adapter)
            if adapter is not None
        ]
        if adapters:
            await asyncio.gather(*(adapter.close() for adapter in adapters))

    def _begin(self) -> None:
        if self._running:
            raise RuntimeError("ProxyRuntime is already running")
        self._running = True

    async def _execute(
        self,
        client_adapter: TransportAdapter,
        server_adapter: TransportAdapter,
    ) -> None:
        self._client_adapter = client_adapter
        self._server_adapter = server_adapter
        self._ready.set()
        await run_pipeline(client_adapter, server_adapter, self._session)

    def _finish(self) -> None:
        self._session.session_store.finish()
        self._ready.clear()
        self._client_adapter = None
        self._server_adapter = None
        self._running = False


def _split_command(server_command: str | None) -> tuple[str, list[str]]:
    if not server_command or not server_command.strip():
        raise ValueError("server_command is required for stdio target transport")
    parts = shlex.split(server_command)
    if not parts or not parts[0]:
        raise ValueError("server_command must contain an executable")
    return parts[0], parts[1:]


def _build_server_adapter(
    config: ProxyRuntimeConfig,
) -> StdioServerAdapter | SseServerAdapter | StreamableHttpServerAdapter:
    if config.transport == Transport.STDIO:
        command, args = _split_command(config.server_command)
        return StdioServerAdapter(command=command, args=args, env=stdio_subprocess_env())
    if config.transport == Transport.SSE:
        return SseServerAdapter(url=config.server_url or "")
    return StreamableHttpServerAdapter(url=config.server_url or "")


def _build_client_adapter(
    config: ProxyRuntimeConfig,
) -> StdioClientAdapter | SseClientAdapter | StreamableHttpClientAdapter:
    if config.listen_transport is None:
        return StdioClientAdapter()
    if config.listen_transport == Transport.SSE:
        return SseClientAdapter(host=LISTEN_HOST, port=config.listen_port)
    return StreamableHttpClientAdapter(host=LISTEN_HOST, port=config.listen_port)
