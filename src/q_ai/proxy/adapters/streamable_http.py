"""Streamable HTTP transport adapters for mcp-proxy.

Provides server-facing and client-facing adapters that bridge MCP SDK
anyio streams to asyncio queues. The pipeline never sees anyio — only
these adapters touch SDK transport internals.

StreamableHttpServerAdapter wraps ``streamable_http_client()`` — connects to
a remote Streamable HTTP MCP server.
StreamableHttpClientAdapter wraps ``StreamableHTTPServerTransport`` — exposes
a Streamable HTTP endpoint that remote MCP clients connect to.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from types import TracebackType

import uvicorn
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import streamable_http_client
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.shared.message import SessionMessage
from starlette.applications import Starlette
from starlette.routing import Mount

from q_ai.mcp.transport import TransportClosedError
from q_ai.proxy.constants import LISTEN_HOST

logger = logging.getLogger(__name__)

# Sentinel pushed into the read queue when the SDK stream ends
_STREAM_CLOSED = object()
_UVICORN_CONNECTION_SHUTDOWN_TIMEOUT = 1
_UVICORN_TASK_SHUTDOWN_TIMEOUT = 2.0


class StreamableHttpServerAdapter:
    """Server-facing adapter — proxy connects to a remote MCP server via Streamable HTTP.

    Wraps the MCP SDK ``streamable_http_client()`` context manager. Connects
    to the target Streamable HTTP server and bridges its anyio streams to
    asyncio queues for consumption by the pipeline.

    Args:
        url: The Streamable HTTP endpoint URL of the remote MCP server.
        terminate_on_close: Whether to send a terminate request when closing.

    Example:
        async with StreamableHttpServerAdapter(url="http://localhost:3000/mcp") as adapter:
            msg = await adapter.read()
            await adapter.write(response)
    """

    def __init__(
        self,
        url: str,
        terminate_on_close: bool = True,
    ) -> None:
        self._url = url
        self._terminate_on_close = terminate_on_close
        self._read_queue: asyncio.Queue[SessionMessage | object] = asyncio.Queue()
        self._write_queue: asyncio.Queue[SessionMessage] = asyncio.Queue()
        self._closed = False
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._get_session_id: Callable[[], str | None] | None = None

    async def __aenter__(self) -> StreamableHttpServerAdapter:
        """Enter the adapter context — start SDK transport and bridge tasks."""
        self._sdk_cm = streamable_http_client(
            self._url,
            terminate_on_close=self._terminate_on_close,
        )
        read_stream, write_stream, get_session_id = await self._sdk_cm.__aenter__()
        self._get_session_id = get_session_id
        self._reader_task = asyncio.create_task(
            self._reader_loop(read_stream),
            name="streamable-http-server-reader",
        )
        self._writer_task = asyncio.create_task(
            self._writer_loop(write_stream),
            name="streamable-http-server-writer",
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the adapter context — clean up tasks and SDK transport."""
        await self.close()
        if hasattr(self, "_sdk_cm"):
            try:
                await self._sdk_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                logger.debug("SDK context exit error (suppressed)", exc_info=True)

    async def read(self) -> SessionMessage:
        """Read the next message from the server.

        Returns:
            The next SessionMessage from the server.

        Raises:
            RuntimeError: If the adapter has been closed.
        """
        if self._closed:
            raise TransportClosedError("StreamableHttpServerAdapter is closed")
        item = await self._read_queue.get()
        if item is _STREAM_CLOSED:
            raise TransportClosedError("StreamableHttpServerAdapter is closed")
        return item  # type: ignore[return-value]

    async def write(self, message: SessionMessage) -> None:
        """Write a message to the server.

        Args:
            message: The SessionMessage to send to the server.

        Raises:
            RuntimeError: If the adapter has been closed.
        """
        if self._closed:
            raise TransportClosedError("StreamableHttpServerAdapter is closed")
        await self._write_queue.put(message)

    async def close(self) -> None:
        """Shut down the adapter. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        # Signal the read queue so any waiting read() unblocks
        await self._read_queue.put(_STREAM_CLOSED)
        # Cancel bridge tasks
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task

    async def _reader_loop(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
    ) -> None:
        """Bridge: pull from SDK anyio read stream, push to asyncio queue.

        Args:
            read_stream: The anyio receive stream from the SDK.
        """
        try:
            async for item in read_stream:
                if self._closed:
                    break
                if isinstance(item, Exception):
                    logger.warning("Exception from server stream: %s", item)
                    continue
                await self._read_queue.put(item)
        except Exception:
            if not self._closed:
                logger.debug("Reader loop ended", exc_info=True)
        finally:
            if not self._closed:
                await self._read_queue.put(_STREAM_CLOSED)

    async def _writer_loop(
        self,
        write_stream: MemoryObjectSendStream[SessionMessage],
    ) -> None:
        """Bridge: pull from asyncio queue, send to SDK anyio write stream.

        Args:
            write_stream: The anyio send stream from the SDK.
        """
        try:
            while not self._closed:
                message = await self._write_queue.get()
                if self._closed:
                    break
                await write_stream.send(message)
        except Exception:
            if not self._closed:
                logger.debug("Writer loop ended", exc_info=True)
                await self.close()


class StreamableHttpClientAdapter:
    """Client-facing adapter — remote MCP client connects to proxy via Streamable HTTP.

    Exposes a Streamable HTTP endpoint backed by
    ``StreamableHTTPServerTransport``. Starts an embedded Starlette + uvicorn
    server on the given host/port. When an MCP client connects, the SDK's
    anyio streams are bridged to asyncio queues for consumption by the pipeline.

    Args:
        host: Bind address for the HTTP server.
        port: Port number for the HTTP server.

    Example:
        async with StreamableHttpClientAdapter(host=LISTEN_HOST, port=8091) as adapter:
            msg = await adapter.read()   # message from remote client
            await adapter.write(response) # response to remote client
    """

    def __init__(self, host: str = LISTEN_HOST, port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._read_queue: asyncio.Queue[SessionMessage | object] = asyncio.Queue()
        self._write_queue: asyncio.Queue[SessionMessage] = asyncio.Queue()
        self._closed = False
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._connect_task: asyncio.Task[None] | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._client_connected: asyncio.Event = asyncio.Event()

    _STARTUP_TIMEOUT: float = 5.0

    async def __aenter__(self) -> StreamableHttpClientAdapter:
        """Enter the adapter context — start HTTP server and wait for it to listen."""
        session_id = uuid.uuid4().hex
        self._http_transport = StreamableHTTPServerTransport(
            mcp_session_id=session_id,
        )

        self._connect_task = asyncio.create_task(
            self._run_transport_connect(),
            name="streamable-http-client-connect",
        )

        starlette_app = Starlette(
            routes=[
                Mount("/mcp", app=self._http_transport.handle_request),
            ],
        )

        config = uvicorn.Config(
            app=starlette_app,
            host=self._host,
            port=self._port,
            log_level="warning",
            loop="none",
            timeout_graceful_shutdown=_UVICORN_CONNECTION_SHUTDOWN_TIMEOUT,
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            self._uvicorn_server.serve(),
            name="streamable-http-client-uvicorn",
        )
        await self._wait_for_server_ready()
        return self

    async def _wait_for_server_ready(self) -> None:
        """Wait for uvicorn to bind and start listening.

        Raises:
            RuntimeError: If the server fails to start within the timeout.
        """
        deadline = asyncio.get_event_loop().time() + self._STARTUP_TIMEOUT
        while not self._uvicorn_server or not self._uvicorn_server.started:
            if self._uvicorn_server and self._uvicorn_server.should_exit:
                raise RuntimeError(f"Uvicorn failed to start on {self._host}:{self._port}")
            if asyncio.get_event_loop().time() > deadline:
                await self.close()
                raise RuntimeError(
                    f"Uvicorn did not start within {self._STARTUP_TIMEOUT}s "
                    f"on {self._host}:{self._port}"
                )
            await asyncio.sleep(0.05)

    async def _run_transport_connect(self) -> None:
        """Enter the SDK transport connect context and bridge streams.

        Runs for the lifetime of the adapter — the connect context stays
        open until the adapter is closed.
        """
        try:
            async with self._http_transport.connect() as (read_stream, write_stream):
                self._reader_task = asyncio.create_task(
                    self._reader_loop(read_stream),
                    name="streamable-http-client-reader",
                )
                self._writer_task = asyncio.create_task(
                    self._writer_loop(write_stream),
                    name="streamable-http-client-writer",
                )
                self._client_connected.set()
                # Keep connection alive until adapter closes
                while not self._closed:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        except Exception:
            if not self._closed:
                logger.debug("Transport connect ended", exc_info=True)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the adapter context — shut down server and clean up."""
        await self.close()

    async def read(self) -> SessionMessage:
        """Read the next message from the remote client.

        Returns:
            The next SessionMessage from the client.

        Raises:
            RuntimeError: If the adapter has been closed.
        """
        if self._closed:
            raise TransportClosedError("StreamableHttpClientAdapter is closed")
        item = await self._read_queue.get()
        if item is _STREAM_CLOSED:
            raise TransportClosedError("StreamableHttpClientAdapter is closed")
        return item  # type: ignore[return-value]

    async def write(self, message: SessionMessage) -> None:
        """Write a message to the remote client.

        Args:
            message: The SessionMessage to send to the client.

        Raises:
            RuntimeError: If the adapter has been closed.
        """
        if self._closed:
            raise TransportClosedError("StreamableHttpClientAdapter is closed")
        await self._write_queue.put(message)

    async def close(self) -> None:
        """Shut down the adapter. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        await self._read_queue.put(_STREAM_CLOSED)
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connect_task
        await self._wait_for_http_server_exit()

    async def _wait_for_http_server_exit(self) -> None:
        """Let Uvicorn release its listener before falling back to cancellation."""
        if self._server_task is None or self._server_task.done():
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(self._server_task),
                timeout=_UVICORN_TASK_SHUTDOWN_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("Uvicorn shutdown timed out; cancelling the server task")
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task

    async def _reader_loop(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
    ) -> None:
        """Bridge: pull from SDK anyio read stream, push to asyncio queue.

        Args:
            read_stream: The anyio receive stream from the SDK.
        """
        try:
            async for item in read_stream:
                if self._closed:
                    break
                if isinstance(item, Exception):
                    logger.warning("Exception from client stream: %s", item)
                    continue
                await self._read_queue.put(item)
        except Exception:
            if not self._closed:
                logger.debug("Reader loop ended", exc_info=True)
        finally:
            if not self._closed:
                await self._read_queue.put(_STREAM_CLOSED)

    async def _writer_loop(
        self,
        write_stream: MemoryObjectSendStream[SessionMessage],
    ) -> None:
        """Bridge: pull from asyncio queue, send to SDK anyio write stream.

        Args:
            write_stream: The anyio send stream from the SDK.
        """
        try:
            while not self._closed:
                message = await self._write_queue.get()
                if self._closed:
                    break
                await write_stream.send(message)
        except Exception:
            if not self._closed:
                logger.debug("Writer loop ended", exc_info=True)
                await self.close()
