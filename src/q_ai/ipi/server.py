"""FastAPI callback server to capture agent hits.

This module implements the HTTP callback listener that receives
out-of-band requests from AI agents that have processed and executed
hidden payloads in canary documents. Incoming hits are recorded to
a SQLite database and logged to the console in real time.

Supports both authenticated (/c/{uuid}/{token}) and unauthenticated
(/c/{uuid}) callback URLs. Authenticated callbacks validate the
per-campaign token and receive high confidence scores. Unauthenticated
callbacks are still recorded but scored based on User-Agent analysis.

The server returns a fake 404 response on callback endpoints to avoid
alerting the target system that the payload was successfully executed.

Usage:
    From the IPI listen command (preferred)::

        python -m q_ai.ipi listen --port 8080

    Programmatic::

        from q_ai.ipi.server import start_server
        start_server(host="127.0.0.1", port=8080)
"""

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse, Response
from rich.console import Console
from rich.markup import escape
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from q_ai.ipi import db
from q_ai.ipi.listener import record_hit, score_confidence
from q_ai.ipi.models import Hit, HitConfidence

console = Console()

# Module-level tunnel-mode state, set by start_server() when --tunnel is active.
# Drives forwarded-header trust in _resolve_source_info() and governs whether
# public-exposure hardening middleware is installed.
_tunnel_mode_provider: str | None = None

_CF_CONNECTING_IP_HEADER = "cf-connecting-ip"
"""Cloudflare-originated client-IP header. Only trusted in tunnel mode."""

_UNKNOWN_IP = "unknown"
"""Fallback value when no client IP can be resolved."""


def _set_tunnel_mode(provider: str | None) -> None:
    """Set the module-level tunnel-mode flag.

    Exposed for test setup/teardown; production code sets this via
    :func:`start_server`.

    Args:
        provider: Tunnel provider name (e.g. ``"cloudflare"``) or
            ``None`` for local-only mode.
    """
    global _tunnel_mode_provider  # noqa: PLW0603
    _tunnel_mode_provider = provider


def _resolve_source_info(request: Request) -> tuple[str, bool]:
    """Resolve the client IP and tunnel-source flag for a request.

    When tunnel mode is active and the provider is Cloudflare, the
    trusted ``CF-Connecting-IP`` header is preferred so hits record the
    real remote caller rather than the local cloudflared daemon, and
    the returned ``via_tunnel`` flag is True. Forwarded headers are
    never trusted when the listener is running locally (prevents IP
    spoofing via forged headers on an exposed local listener), and the
    flag is False in that case.

    Args:
        request: The incoming FastAPI request.

    Returns:
        ``(source_ip, via_tunnel)``. ``source_ip`` is the resolved
        client IP or ``"unknown"`` if nothing is available.
        ``via_tunnel`` is True iff the IP came from a trusted
        forwarded header.
    """
    if _tunnel_mode_provider == "cloudflare":
        forwarded = request.headers.get(_CF_CONNECTING_IP_HEADER)
        if forwarded:
            return str(forwarded), True
    if request.client is not None:
        return str(request.client.host), False
    return _UNKNOWN_IP, False


# ---------------------------------------------------------------------------
# Public-exposure hardening (installed only when ``tunnel_provider`` is set).
# ---------------------------------------------------------------------------

_MAX_BODY_BYTES = 1 * 1024 * 1024
"""Maximum accepted request body in bytes (1 MiB) when tunnel mode is active."""

_RATE_LIMIT_WINDOW_SECS = 60.0
"""Sliding-window duration in seconds for the per-IP rate limiter."""

_RATE_LIMIT_MAX_REQUESTS = 120
"""Max requests per IP within ``_RATE_LIMIT_WINDOW_SECS`` before 429."""

_UVICORN_KEEPALIVE_SECS = 5
"""Conservative keep-alive timeout for tunneled listeners."""

_UVICORN_GRACEFUL_SHUTDOWN_SECS = 10
"""Conservative graceful-shutdown timeout for tunneled listeners."""

_Dispatch = Callable[[Request], Awaitable[Response]]


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose ``Content-Length`` exceeds a configured cap.

    The listener's purpose is proof-of-execution callbacks, not bulk
    transfers. When exposed publicly via a tunnel we cap body size at
    1 MiB. Requests without ``Content-Length`` pass through; chunked
    bodies exceeding the cap are rejected downstream by uvicorn's
    default request limits.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: _Dispatch) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self._max_bytes:
                return PlainTextResponse(
                    "Request body too large",
                    status_code=413,
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding-window rate limiter.

    Uses the resolved client IP (forwarded header when tunneled, TCP
    peer otherwise) so abuse from a single real caller is throttled
    even though every request arrives from the local tunnel daemon.

    In-memory and process-local: no external state (Redis etc.) is
    needed for the callback listener's traffic profile.
    """

    def __init__(
        self,
        app: ASGIApp,
        window_secs: float,
        max_requests: int,
    ) -> None:
        super().__init__(app)
        self._window = window_secs
        self._max = max_requests
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: _Dispatch) -> Response:
        # Rate limiting only needs the IP bucket key; the via_tunnel flag is
        # discarded here and consumed by the Hit construction sites below.
        client_ip, _ = _resolve_source_info(request)
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            bucket = self._buckets[client_ip]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._max:
                return PlainTextResponse(
                    "Rate limit exceeded",
                    status_code=429,
                )
            bucket.append(now)
        return await call_next(request)


def install_hardening_middleware(app: FastAPI) -> None:
    """Register body-size and rate-limit middleware on ``app``, idempotently.

    Factored out so tests can exercise the middleware against an
    isolated :class:`FastAPI` instance without relying on the global
    listener ``app`` singleton (which would leak state between tests).

    Calling this multiple times on the same ``app`` is a no-op after
    the first call: each middleware class is only added if it is not
    already present in ``app.user_middleware``. This matters because
    the production listener registers middleware on the module-level
    ``app`` singleton; repeated ``start_server(tunnel_provider=...)``
    calls in the same process would otherwise stack duplicate middleware.

    Args:
        app: The FastAPI application to decorate.
    """
    installed_classes = {mw.cls for mw in app.user_middleware}
    if BodySizeLimitMiddleware not in installed_classes:
        app.add_middleware(BodySizeLimitMiddleware, max_bytes=_MAX_BODY_BYTES)
    if RateLimitMiddleware not in installed_classes:
        app.add_middleware(
            RateLimitMiddleware,
            window_secs=_RATE_LIMIT_WINDOW_SECS,
            max_requests=_RATE_LIMIT_MAX_REQUESTS,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Yields control to the application. Cleanup tasks run after yield
    on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None: Control is passed to the running application.
    """
    console.print("[green][OK][/green] Database initialized")
    yield


app = FastAPI(
    title="q-ai IPI Listener",
    description="Callback server for Indirect Prompt Injection detection",
    lifespan=lifespan,
)


_CONFIDENCE_STYLES = {
    HitConfidence.HIGH: "bold green",
    HitConfidence.MEDIUM: "bold yellow",
    HitConfidence.LOW: "bold red",
}
"""Rich markup styles for confidence level display."""


def log_hit_to_console(hit: Hit) -> None:
    """Print hit details to console with Rich formatting.

    Displays a prominent visual banner with the hit UUID, source IP,
    user-agent string, confidence level, token validation status,
    and any captured exfil data.

    Args:
        hit: Hit object containing callback metadata.
    """
    conf_style = _CONFIDENCE_STYLES.get(hit.confidence, "dim")
    token_indicator = "[green]+ valid[/green]" if hit.token_valid else "[red]x missing[/red]"

    console.print()
    console.print("=" * 60, style="bold yellow")
    console.print(f"[bold red]>>> HIT RECEIVED[/bold red] at {hit.timestamp.strftime('%H:%M:%S')}")
    console.print(f"   [bold]UUID:[/bold]       {escape(hit.uuid)}")
    console.print(f"   [bold]Token:[/bold]      {token_indicator}")
    console.print(
        f"   [bold]Confidence:[/bold] [{conf_style}]{hit.confidence.value}[/{conf_style}]"
    )
    console.print(f"   [bold]IP:[/bold]         {escape(hit.source_ip)}")
    console.print(f"   [bold]UA:[/bold]         {escape(hit.user_agent[:60])}...")
    if hit.body:
        console.print(f"   [bold yellow]DATA:[/bold yellow]       {escape(hit.body[:200])}")
        if len(hit.body) > 200:
            console.print(f"               [dim]({len(hit.body)} bytes total)[/dim]")
    console.print("=" * 60, style="bold yellow")
    console.print()


def _record_and_log_hit(hit: Hit) -> None:
    """Persist a hit to the database and log it to the console.

    Called as a background task from the callback endpoint so the
    HTTP response is returned immediately without blocking on I/O.

    Args:
        hit: Hit object to save and display.
    """
    record_hit(hit)
    log_hit_to_console(hit)


# =========================================================================
# Authenticated callback routes (/c/{uuid}/{token})
# These must be defined BEFORE the unauthenticated /c/{uuid} routes
# so FastAPI matches the more specific path first.
# =========================================================================


@app.get("/c/{canary_uuid}/{token}")
async def callback_authenticated(
    canary_uuid: str,
    token: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PlainTextResponse:
    """Receive and record an authenticated canary callback (GET).

    Validates the per-campaign token against the database. If the token
    matches, the hit is recorded with high confidence. If the UUID exists
    but the token is wrong, the hit is still recorded but with reduced
    confidence.

    Args:
        canary_uuid: UUID path parameter identifying the canary campaign.
        token: Per-campaign authentication token.
        request: Incoming FastAPI request object.
        background_tasks: FastAPI background task queue.

    Returns:
        PlainTextResponse with a spoofed 404 status code and body.
    """
    query_string = str(request.query_params) if request.query_params else None
    user_agent = request.headers.get("user-agent", "unknown")

    # Validate token against database
    campaign = db.get_campaign_by_token(canary_uuid, token)
    token_valid = campaign is not None
    confidence = score_confidence(token_valid, user_agent)
    source_ip, via_tunnel = _resolve_source_info(request)

    hit = Hit(
        id=uuid.uuid4().hex,
        uuid=canary_uuid,
        source_ip=source_ip,
        user_agent=user_agent,
        headers=json.dumps(dict(request.headers)),
        body=query_string,
        token_valid=token_valid,
        via_tunnel=via_tunnel,
        confidence=confidence,
        timestamp=datetime.now(UTC),
    )

    background_tasks.add_task(_record_and_log_hit, hit)

    return PlainTextResponse(
        "404 Not Found: The requested resource could not be located.",
        status_code=404,
    )


@app.post("/c/{canary_uuid}/{token}")
async def callback_authenticated_post(
    canary_uuid: str,
    token: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PlainTextResponse:
    """Receive and record an authenticated canary callback (POST).

    Validates the per-campaign token and captures POST body data
    for exfil payload types.

    Args:
        canary_uuid: UUID path parameter identifying the canary campaign.
        token: Per-campaign authentication token.
        request: Incoming FastAPI request object.
        background_tasks: FastAPI background task queue.

    Returns:
        PlainTextResponse with a spoofed 404 status code and body.
    """
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else None
    user_agent = request.headers.get("user-agent", "unknown")

    campaign = db.get_campaign_by_token(canary_uuid, token)
    token_valid = campaign is not None
    confidence = score_confidence(token_valid, user_agent)
    source_ip, via_tunnel = _resolve_source_info(request)

    hit = Hit(
        id=uuid.uuid4().hex,
        uuid=canary_uuid,
        source_ip=source_ip,
        user_agent=user_agent,
        headers=json.dumps(dict(request.headers)),
        body=body_text,
        token_valid=token_valid,
        via_tunnel=via_tunnel,
        confidence=confidence,
        timestamp=datetime.now(UTC),
    )

    background_tasks.add_task(_record_and_log_hit, hit)

    return PlainTextResponse(
        "404 Not Found: The requested resource could not be located.",
        status_code=404,
    )


# =========================================================================
# Unauthenticated callback routes (/c/{uuid})
# These still accept callbacks but mark them as token_valid=False.
# Confidence is scored based on User-Agent analysis only.
# =========================================================================


@app.get("/c/{canary_uuid}")
async def callback(
    canary_uuid: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PlainTextResponse:
    """Receive and record an unauthenticated canary callback (GET).

    Records the callback with token_valid=False. Confidence is scored
    based on User-Agent analysis (medium for programmatic clients,
    low for browsers/scanners).

    Args:
        canary_uuid: UUID path parameter identifying the canary campaign.
        request: Incoming FastAPI request object.
        background_tasks: FastAPI background task queue.

    Returns:
        PlainTextResponse with a spoofed 404 status code and body.
    """
    query_string = str(request.query_params) if request.query_params else None
    user_agent = request.headers.get("user-agent", "unknown")
    confidence = score_confidence(False, user_agent)
    source_ip, via_tunnel = _resolve_source_info(request)

    hit = Hit(
        id=uuid.uuid4().hex,
        uuid=canary_uuid,
        source_ip=source_ip,
        user_agent=user_agent,
        headers=json.dumps(dict(request.headers)),
        body=query_string,
        token_valid=False,
        via_tunnel=via_tunnel,
        confidence=confidence,
        timestamp=datetime.now(UTC),
    )

    background_tasks.add_task(_record_and_log_hit, hit)

    return PlainTextResponse(
        "404 Not Found: The requested resource could not be located.",
        status_code=404,
    )


@app.post("/c/{canary_uuid}")
async def callback_post(
    canary_uuid: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PlainTextResponse:
    """Receive and record an unauthenticated canary callback (POST).

    Records the callback with token_valid=False. Confidence is scored
    based on User-Agent analysis only.

    Args:
        canary_uuid: UUID path parameter identifying the canary campaign.
        request: Incoming FastAPI request object.
        background_tasks: FastAPI background task queue.

    Returns:
        PlainTextResponse with a spoofed 404 status code and body.
    """
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else None
    user_agent = request.headers.get("user-agent", "unknown")
    confidence = score_confidence(False, user_agent)
    source_ip, via_tunnel = _resolve_source_info(request)

    hit = Hit(
        id=uuid.uuid4().hex,
        uuid=canary_uuid,
        source_ip=source_ip,
        user_agent=user_agent,
        headers=json.dumps(dict(request.headers)),
        body=body_text,
        token_valid=False,
        via_tunnel=via_tunnel,
        confidence=confidence,
        timestamp=datetime.now(UTC),
    )

    background_tasks.add_task(_record_and_log_hit, hit)

    return PlainTextResponse(
        "404 Not Found: The requested resource could not be located.",
        status_code=404,
    )


@app.get("/health")
async def health() -> dict:
    """Return server health status.

    Provides a simple liveness check for monitoring and automated
    testing. Does not verify database connectivity.

    Returns:
        Dictionary with ``{"status": "ok"}``.
    """
    return {"status": "ok"}


def start_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    tunnel_provider: str | None = None,
) -> None:
    """Start the callback listener server.

    Launches the uvicorn ASGI server bound to the specified host
    and port. The server runs in the foreground until interrupted
    with Ctrl+C.

    When ``tunnel_provider`` is set, the listener records forwarded
    client IPs from the provider's trusted header (e.g.
    ``CF-Connecting-IP`` for Cloudflare) rather than the local TCP peer.
    Forwarded headers are ignored in local-only mode to prevent IP
    spoofing.

    Args:
        host: Network interface to bind (default ``"127.0.0.1"``).
        port: TCP port to listen on (default ``8080``).
        tunnel_provider: If not ``None``, indicates the listener is
            running behind a reverse tunnel of the named provider
            (currently ``"cloudflare"``). Enables forwarded-header
            trust. ``None`` means local-only.
    """
    _set_tunnel_mode(tunnel_provider)

    console.print(f"[bold green]Starting q-ai IPI listener on {host}:{port}[/bold green]")
    console.print(f"   Callback URL: [blue]http://<your-ip>:{port}/c/<uuid>/<token>[/blue]")
    console.print("   Press [bold]Ctrl+C[/bold] to stop\n")

    uvicorn_kwargs: dict[str, object] = {
        "host": host,
        "port": port,
        "log_level": "warning",
    }
    if tunnel_provider is not None:
        install_hardening_middleware(app)
        console.print(
            "[bold yellow]"
            "WARNING: Listener is publicly reachable. "
            "Do not share tunnel URLs beyond your test scope."
            "[/bold yellow]\n"
        )
        uvicorn_kwargs["timeout_keep_alive"] = _UVICORN_KEEPALIVE_SECS
        uvicorn_kwargs["timeout_graceful_shutdown"] = _UVICORN_GRACEFUL_SHUTDOWN_SECS

    try:
        uvicorn.run(app, **uvicorn_kwargs)  # type: ignore[arg-type]
    finally:
        _set_tunnel_mode(None)
