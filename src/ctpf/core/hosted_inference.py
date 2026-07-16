"""Controlled HTTP boundary for hosted OpenAI-compatible inference."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import ssl
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpcore
import httpx

from ctpf.core.llm import ProviderError
from ctpf.core.redaction import redact_text, sanitize_evidence

MAX_REQUEST_BYTES = 131_072
MAX_RESPONSE_BYTES = 1_048_576
MAX_INPUT_TOKENS = 131_072
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
WRITE_TIMEOUT_SECONDS = 30
POOL_TIMEOUT_SECONDS = 10
OVERALL_TIMEOUT_SECONDS = 90
MAX_ATTEMPTS = 1
MAX_CONCURRENT_REQUESTS = 1
HTTP_PROTOCOL = "HTTP/1.1"
REDIRECT_POLICY = "deny"
ENVIRONMENT_PROXY_POLICY = "ignore"
TLS_POLICY = "system trust and requested-host verification"
_ALLOWED_PATH = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$&'()*+,;=:@/"
)
SocketOption = (
    tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]
)


class Resolver(Protocol):
    """Injectable A/AAAA resolver used before any connection attempt."""

    async def __call__(self, host: str, port: int) -> Sequence[str]: ...


class TransportFactory(Protocol):
    """Construct one no-reuse transport for an already approved address."""

    def __call__(
        self,
        endpoint: CanonicalEndpoint,
        resolution: Resolution,
    ) -> httpx.AsyncBaseTransport: ...


class HostedInferenceError(ProviderError):
    """Sanitized hosted-inference failure with bounded transport evidence."""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


class HostedInferenceCancelled(asyncio.CancelledError):
    """Observable client cancellation with unknown provider-side outcome."""

    def __init__(self, evidence: dict[str, Any]) -> None:
        super().__init__("hosted-inference request was cancelled")
        self.evidence = evidence


class PeerAddressError(OSError):
    """Raised when the connected peer differs from the approved address."""


@dataclass(frozen=True)
class CanonicalEndpoint:
    """Unambiguous endpoint authority shared by policy and transport."""

    scheme: str
    host: str
    port: int
    origin: str
    base_path: str
    normalized_url: str
    network_class: str

    def to_payload(self) -> dict[str, Any]:
        """Return the authority-bearing endpoint representation."""
        return {
            "base_path": self.base_path,
            "host": self.host,
            "network_class": self.network_class,
            "normalized_url": self.normalized_url,
            "origin": self.origin,
            "port": self.port,
            "scheme": self.scheme,
        }

    def child_url(self, path: str) -> str:
        """Return a fixed child endpoint beneath the approved base path."""
        if (
            not path.startswith("/")
            or ".." in path.split("/")
            or "?" in path
            or "#" in path
            or "\\" in path
        ):
            raise ValueError("hosted-inference child path must be absolute and traversal-free")
        return f"{self.origin}{self.base_path}{path}"


@dataclass(frozen=True)
class Resolution:
    """Validated complete resolution set and deterministic selected peer."""

    addresses: tuple[str, ...]
    selected_address: str

    def to_payload(self) -> dict[str, Any]:
        """Return resolution evidence."""
        return {
            "addresses": list(self.addresses),
            "selected_address": self.selected_address,
        }


@dataclass(frozen=True)
class HostedResponse:
    """One bounded JSON response and its transport evidence."""

    payload: dict[str, Any]
    evidence: dict[str, Any]


def canonicalize_endpoint(raw: str) -> CanonicalEndpoint:
    """Normalize and validate an OpenAI-compatible API base URL.

    Args:
        raw: Operator-declared endpoint.

    Returns:
        Canonical endpoint authority.

    Raises:
        ValueError: If the URL is ambiguous or outside the supported boundary.
    """
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ValueError("inference endpoint must be normalized non-empty text")
    if any(ord(character) < 33 for character in raw) or "\\" in raw:
        raise ValueError("inference endpoint contains unsupported whitespace or separators")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid inference endpoint: {exc}") from exc
    _validate_url_parts(parsed)
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("inference endpoint port must be between 1 and 65535")
    scheme = parsed.scheme.lower()
    host = _canonical_host(parsed.hostname or "")
    effective_port = port or (443 if scheme == "https" else 80)
    network_class = _network_class(scheme, host)
    base_path = _canonical_path(parsed.path)
    authority = _authority(host, effective_port, scheme)
    origin = f"{scheme}://{authority}"
    return CanonicalEndpoint(
        scheme,
        host,
        effective_port,
        origin,
        base_path,
        urlunsplit((scheme, authority, base_path, "", "")),
        network_class,
    )


async def system_resolver(host: str, port: int) -> Sequence[str]:
    """Resolve all TCP A/AAAA candidates using the platform resolver."""
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    addresses: list[str] = []
    for record in records:
        raw_address = record[4][0]
        if not isinstance(raw_address, str):
            raise TypeError("platform resolver returned an invalid address")
        addresses.append(raw_address)
    return tuple(addresses)


async def resolve_endpoint(
    endpoint: CanonicalEndpoint,
    resolver: Resolver = system_resolver,
) -> Resolution:
    """Resolve and fail closed unless the full address set is approved."""
    literal = _ip_address(endpoint.host)
    raw_addresses: Sequence[str]
    if literal is not None:
        raw_addresses = (str(literal),)
    else:
        raw_addresses = await resolver(endpoint.host, endpoint.port)
    addresses = _validated_addresses(raw_addresses, endpoint.network_class)
    selected = sorted(addresses, key=_address_sort_key)[0]
    return Resolution(addresses, selected)


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connect to one resolved address while retaining the requested TLS host."""

    def __init__(
        self,
        endpoint: CanonicalEndpoint,
        resolution: Resolution,
        delegate: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._resolution = resolution
        self._delegate = delegate or httpcore.AnyIOBackend()
        self.connected_peer: str | None = None

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Connect only to the selected address and verify the actual peer."""
        if _canonical_host(host) != self._endpoint.host or port != self._endpoint.port:
            raise PeerAddressError("transport attempted an unapproved destination")
        stream = await self._delegate.connect_tcp(
            self._resolution.selected_address,
            port,
            timeout,
            local_address,
            socket_options,
        )
        peer = _stream_peer(stream)
        if peer != self._resolution.selected_address:
            await stream.aclose()
            raise PeerAddressError("connected peer did not match the approved address")
        self.connected_peer = peer
        return stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Reject Unix sockets because they are outside endpoint authority."""
        raise PeerAddressError("Unix sockets are not permitted for hosted inference")

    async def sleep(self, seconds: float) -> None:
        """Delegate scheduler sleeps required by HTTP Core."""
        await self._delegate.sleep(seconds)


class _CoreResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterable[bytes]) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for part in self._stream:
            yield part

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()


class AddressPinnedTransport(httpx.AsyncBaseTransport):
    """HTTPX adapter backed by a one-address HTTP Core pool."""

    def __init__(self, endpoint: CanonicalEndpoint, resolution: Resolution) -> None:
        self._backend = PinnedNetworkBackend(endpoint, resolution)
        context = ssl.create_default_context() if endpoint.scheme == "https" else None
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=context,
            max_connections=MAX_CONCURRENT_REQUESTS,
            max_keepalive_connections=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=self._backend,
        )

    @property
    def connected_peer(self) -> str | None:
        """Return the peer verified before request bytes were sent."""
        return self._backend.connected_peer

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Translate one HTTPX request into HTTP Core without route changes."""
        if not isinstance(request.stream, httpx.AsyncByteStream):
            raise TypeError("hosted-inference request stream must be asynchronous")
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        response = await self._pool.handle_async_request(core_request)
        if not isinstance(response.stream, AsyncIterable):
            raise TypeError("hosted-inference response stream must be asynchronous")
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_CoreResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        """Close the one-request connection pool."""
        await self._pool.aclose()


class HostedInferenceBoundary:
    """Perform one bounded, pinned, no-retry hosted-inference request."""

    def __init__(
        self,
        endpoint: CanonicalEndpoint,
        api_key: str,
        *,
        resolver: Resolver = system_resolver,
        transport_factory: TransportFactory = AddressPinnedTransport,
    ) -> None:
        if not api_key:
            raise ValueError("hosted-inference API key must be non-empty")
        self._endpoint = endpoint
        self._api_key = api_key
        self._resolver = resolver
        self._transport_factory = transport_factory

    async def post_json(self, path: str, payload: dict[str, Any]) -> HostedResponse:
        """POST one bounded JSON object through the controlled transport."""
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        evidence = self._initial_evidence(len(body))
        if len(body) > MAX_REQUEST_BYTES:
            raise HostedInferenceError("inference request exceeded byte ceiling", evidence)
        try:
            resolution = await resolve_endpoint(self._endpoint, self._resolver)
            evidence["resolution"] = resolution.to_payload()
            return await self._post(path, body, resolution, evidence)
        except HostedInferenceError:
            raise
        except asyncio.CancelledError as exc:
            evidence["cancellation_outcome"] = "client_cancelled_provider_outcome_unknown"
            raise HostedInferenceCancelled(evidence) from exc
        except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException) as exc:
            evidence["terminal_failure"] = f"client_{type(exc).__name__}_provider_outcome_unknown"
            raise HostedInferenceError("hosted-inference request timed out", evidence) from exc
        except Exception as exc:
            evidence["terminal_failure"] = type(exc).__name__
            message = redact_text(str(exc), (self._api_key,))
            raise HostedInferenceError(
                f"hosted-inference transport failed: {message}", evidence
            ) from exc

    async def _post(
        self,
        path: str,
        body: bytes,
        resolution: Resolution,
        evidence: dict[str, Any],
    ) -> HostedResponse:
        transport = self._transport_factory(self._endpoint, resolution)
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
            write=WRITE_TIMEOUT_SECONDS,
            pool=POOL_TIMEOUT_SECONDS,
        )
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {self._api_key}",
            "Connection": "close",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with asyncio.timeout(OVERALL_TIMEOUT_SECONDS):
                evidence["attempts"] = 1
                async with client.stream(
                    "POST",
                    self._endpoint.child_url(path),
                    headers=headers,
                    content=body,
                ) as response:
                    evidence["connected_peer"] = getattr(transport, "connected_peer", None)
                    if evidence["connected_peer"] != resolution.selected_address:
                        evidence["terminal_failure"] = "connected_peer_unverified"
                        raise HostedInferenceError(
                            "hosted-inference connected peer was not verified", evidence
                        )
                    raw = await self._read_response(response, evidence)
        return self._decode_response(response.status_code, raw, evidence)

    async def _read_response(
        self,
        response: httpx.Response,
        evidence: dict[str, Any],
    ) -> bytes:
        if 300 <= response.status_code < 400:
            evidence["terminal_failure"] = "redirect_denied"
            raise HostedInferenceError("hosted-inference redirect was denied", evidence)
        encoding = response.headers.get("content-encoding", "identity").lower()
        if encoding not in {"", "identity"}:
            evidence["terminal_failure"] = "unexpected_content_encoding"
            raise HostedInferenceError("hosted-inference response encoding was denied", evidence)
        if response.is_stream_consumed:
            if len(response.content) > MAX_RESPONSE_BYTES:
                evidence["response_bytes"] = len(response.content)
                evidence["terminal_failure"] = "response_byte_ceiling_exceeded"
                raise HostedInferenceError("inference response exceeded byte ceiling", evidence)
            evidence["response_bytes"] = len(response.content)
            return bytes(response.content)
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_raw():
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                evidence["response_bytes"] = total
                evidence["terminal_failure"] = "response_byte_ceiling_exceeded"
                raise HostedInferenceError("inference response exceeded byte ceiling", evidence)
            chunks.append(chunk)
        evidence["response_bytes"] = total
        return b"".join(chunks)

    def _decode_response(
        self,
        status_code: int,
        raw: bytes,
        evidence: dict[str, Any],
    ) -> HostedResponse:
        try:
            decoded = json.loads(raw) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            evidence["terminal_failure"] = "malformed_json_response"
            raise HostedInferenceError(
                "hosted-inference response was not valid JSON", evidence
            ) from exc
        sanitized = sanitize_evidence(decoded, (self._api_key,))
        if status_code < 200 or status_code >= 300:
            evidence["provider_status"] = status_code
            evidence["provider_error"] = sanitized
            raise HostedInferenceError(
                f"hosted-inference provider returned HTTP {status_code}",
                evidence,
            )
        if not isinstance(sanitized, dict):
            evidence["terminal_failure"] = "non_object_json_response"
            raise HostedInferenceError("hosted-inference response must be a JSON object", evidence)
        evidence["provider_status"] = status_code
        evidence["cancellation_outcome"] = "request_completed"
        return HostedResponse(sanitized, evidence)

    def _initial_evidence(self, request_bytes: int) -> dict[str, Any]:
        return {
            "attempts": 0,
            "cancellation_outcome": "not_started",
            "concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "deadlines_seconds": {
                "connect": CONNECT_TIMEOUT_SECONDS,
                "overall": OVERALL_TIMEOUT_SECONDS,
                "pool": POOL_TIMEOUT_SECONDS,
                "read": READ_TIMEOUT_SECONDS,
                "write": WRITE_TIMEOUT_SECONDS,
            },
            "destination": self._endpoint.to_payload(),
            "environment_proxy_policy": ENVIRONMENT_PROXY_POLICY,
            "http_protocol": HTTP_PROTOCOL,
            "max_attempts": MAX_ATTEMPTS,
            "redirect_policy": REDIRECT_POLICY,
            "request_bytes": request_bytes,
            "response_byte_ceiling": MAX_RESPONSE_BYTES,
            "retry_count": 0,
            "tls_policy": TLS_POLICY if self._endpoint.scheme == "https" else "not_applicable",
        }


def _validate_url_parts(parsed: SplitResult) -> None:
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("inference endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("inference endpoint must not contain credentials")
    if parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("inference endpoint must not contain query or fragment data")


def _canonical_host(raw: str) -> str:
    if not raw or "%" in raw or raw.endswith("."):
        raise ValueError("inference endpoint host is ambiguous")
    literal = _ip_address(raw)
    if literal is not None:
        if isinstance(literal, ipaddress.IPv6Address) and literal.ipv4_mapped is not None:
            raise ValueError("IPv4-mapped IPv6 endpoint hosts are unsupported")
        return str(literal)
    try:
        encoded = raw.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("inference endpoint hostname is invalid") from exc
    labels = encoded.split(".")
    if len(labels) < 2 or any(not label or len(label) > 63 for label in labels):
        raise ValueError("public HTTPS endpoint requires a valid fully qualified hostname")
    return encoded


def _canonical_path(raw: str) -> str:
    if not raw or raw == "/":
        return ""
    if "%" in raw or "//" in raw or any(character not in _ALLOWED_PATH for character in raw):
        raise ValueError("inference endpoint base path is ambiguous")
    if any(segment in {".", ".."} for segment in raw.split("/")):
        raise ValueError("inference endpoint base path must not contain dot segments")
    return raw.rstrip("/")


def _network_class(scheme: str, host: str) -> str:
    address = _ip_address(host)
    if address is not None and _is_exact_loopback(address):
        return "loopback"
    if scheme != "https":
        raise ValueError("non-loopback inference endpoints must use HTTPS")
    if address is not None and not _is_approved_global(address):
        raise ValueError("HTTPS endpoint IP must be globally routable")
    return "https_public"


def _authority(host: str, port: int, scheme: str) -> str:
    literal = _ip_address(host)
    rendered = f"[{host}]" if isinstance(literal, ipaddress.IPv6Address) else host
    default = 443 if scheme == "https" else 80
    return rendered if port == default else f"{rendered}:{port}"


def _validated_addresses(raw: Sequence[str], network_class: str) -> tuple[str, ...]:
    if not raw:
        raise ValueError("inference endpoint resolution returned no addresses")
    parsed: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for item in raw:
        address = _ip_address(item)
        if address is None or (
            isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None
        ):
            raise ValueError("inference endpoint resolution returned an invalid address")
        parsed.add(address)
    if network_class == "loopback":
        if not all(_is_exact_loopback(address) for address in parsed):
            raise ValueError("loopback endpoint resolved outside its exact authority")
    elif not all(_is_approved_global(address) for address in parsed):
        raise ValueError("public endpoint resolution included a non-global address")
    return tuple(sorted((str(address) for address in parsed), key=_address_sort_key))


def _stream_peer(stream: httpcore.AsyncNetworkStream) -> str:
    raw = stream.get_extra_info("server_addr")
    if not isinstance(raw, tuple) or not raw or not isinstance(raw[0], str):
        raise PeerAddressError("connected peer address was unavailable")
    address = _ip_address(raw[0])
    if address is None:
        raise PeerAddressError("connected peer address was invalid")
    return str(address)


def _ip_address(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(raw)
    except ValueError:
        return None


def _is_exact_loopback(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address == ipaddress.ip_address("127.0.0.1") or address == ipaddress.ip_address("::1")


def _is_approved_global(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global and not (
        isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None
    )


def _address_sort_key(raw: str) -> tuple[int, bytes]:
    address = ipaddress.ip_address(raw)
    return address.version, address.packed
