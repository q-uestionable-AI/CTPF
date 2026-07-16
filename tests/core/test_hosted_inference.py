"""Adversarial tests for the hosted-inference network and byte boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

import httpcore
import httpx
import pytest

from ctpf.core.hosted_inference import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    HostedInferenceBoundary,
    HostedInferenceCancelled,
    HostedInferenceError,
    PeerAddressError,
    PinnedNetworkBackend,
    Resolution,
    SocketOption,
    canonicalize_endpoint,
    resolve_endpoint,
)

_PUBLIC_ADDRESS = "93.184.216.34"


async def _public_resolver(_host: str, _port: int) -> Sequence[str]:
    return (_PUBLIC_ADDRESS,)


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, outcome: httpx.Response | BaseException, peer: str) -> None:
        self._outcome = outcome
        self.connected_peer = peer
        self.calls = 0

    async def handle_async_request(self, _request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _boundary(
    outcome: httpx.Response | BaseException,
    *,
    secret: str = "exact-secret",
) -> tuple[HostedInferenceBoundary, _FakeTransport]:
    endpoint = canonicalize_endpoint("https://models.example.test/v1")
    transport = _FakeTransport(outcome, _PUBLIC_ADDRESS)
    boundary = HostedInferenceBoundary(
        endpoint,
        secret,
        resolver=_public_resolver,
        transport_factory=lambda _endpoint, _resolution: transport,
    )
    return boundary, transport


@pytest.mark.parametrize(
    ("raw", "normalized", "network"),
    [
        (
            "https://ABC-1234.proxy.runpod.net:443/v1/",
            "https://abc-1234.proxy.runpod.net/v1",
            "https_public",
        ),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1", "loopback"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1", "loopback"),
    ],
)
def test_canonical_endpoint_normalizes_complete_authority(
    raw: str,
    normalized: str,
    network: str,
) -> None:
    endpoint = canonicalize_endpoint(raw)

    assert endpoint.normalized_url == normalized
    assert endpoint.network_class == network
    assert endpoint.origin in normalized


@pytest.mark.parametrize(
    "raw",
    [
        "http://models.example.test/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.2:8000/v1",
        "https://10.0.0.1/v1",
        "https://[::ffff:8.8.8.8]/v1",
        "https://user:secret@models.example.test/v1",
        "https://models.example.test/v1?key=secret",
        "https://models.example.test/v1#fragment",
        "https://models.example.test/v1/../admin",
    ],
)
def test_canonical_endpoint_rejects_unsafe_or_ambiguous_authority(raw: str) -> None:
    with pytest.raises(ValueError):
        canonicalize_endpoint(raw)


async def test_resolution_rejects_non_global_and_mixed_address_sets() -> None:
    endpoint = canonicalize_endpoint("https://models.example.test/v1")

    async def mixed(_host: str, _port: int) -> Sequence[str]:
        return (_PUBLIC_ADDRESS, "10.0.0.1")

    with pytest.raises(ValueError, match="non-global"):
        await resolve_endpoint(endpoint, mixed)


class _FakeStream(httpcore.AsyncNetworkStream):
    def __init__(self, peer: str) -> None:
        self.peer = peer
        self.closed = False

    async def read(self, _max_bytes: int, _timeout: float | None = None) -> bytes:
        return b""

    async def write(self, _buffer: bytes, _timeout: float | None = None) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        _ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return self

    def get_extra_info(self, info: str) -> Any:
        return (self.peer, 443) if info == "server_addr" else None


class _FakeBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.hosts: list[str] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.hosts.append(host)
        return self.stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise AssertionError("unexpected Unix socket")

    async def sleep(self, _seconds: float) -> None:
        return None


async def test_pinned_backend_closes_peer_mismatch_before_returning_stream() -> None:
    endpoint = canonicalize_endpoint("https://models.example.test/v1")
    resolution = Resolution((_PUBLIC_ADDRESS,), _PUBLIC_ADDRESS)
    stream = _FakeStream("93.184.216.35")
    delegate = _FakeBackend(stream)
    backend = PinnedNetworkBackend(endpoint, resolution, delegate)

    with pytest.raises(PeerAddressError, match="did not match"):
        await backend.connect_tcp(endpoint.host, endpoint.port)

    assert delegate.hosts == [_PUBLIC_ADDRESS]
    assert stream.closed is True


async def test_redirect_is_denied_without_a_second_attempt() -> None:
    boundary, transport = _boundary(
        httpx.Response(302, headers={"location": "https://other.example.test/v1"})
    )

    with pytest.raises(HostedInferenceError, match="redirect") as caught:
        await boundary.post_json("/chat/completions", {"model": "test"})

    assert transport.calls == 1
    assert caught.value.evidence["attempts"] == 1
    assert caught.value.evidence["redirect_policy"] == "deny"
    assert caught.value.evidence["environment_proxy_policy"] == "ignore"


async def test_provider_secret_echo_is_recursively_redacted() -> None:
    boundary, _transport = _boundary(
        httpx.Response(
            400,
            json={
                "error": {
                    "authorization": "Bearer exact-secret",
                    "message": "echo exact-secret",
                }
            },
        )
    )

    with pytest.raises(HostedInferenceError) as caught:
        await boundary.post_json("/chat/completions", {"model": "test"})

    assert "exact-secret" not in str(caught.value)
    assert "exact-secret" not in str(caught.value.evidence)
    assert "<redacted>" in str(caught.value.evidence)


async def test_malformed_and_oversized_responses_fail_closed() -> None:
    malformed, _transport = _boundary(httpx.Response(200, content=b"not-json"))
    oversized, _transport = _boundary(httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1)))

    with pytest.raises(HostedInferenceError, match="valid JSON"):
        await malformed.post_json("/chat/completions", {"model": "test"})
    with pytest.raises(HostedInferenceError, match="byte ceiling"):
        await oversized.post_json("/chat/completions", {"model": "test"})


async def test_oversized_request_fails_before_resolution_or_transport() -> None:
    calls = 0

    async def resolver(_host: str, _port: int) -> Sequence[str]:
        nonlocal calls
        calls += 1
        return (_PUBLIC_ADDRESS,)

    endpoint = canonicalize_endpoint("https://models.example.test/v1")
    boundary = HostedInferenceBoundary(endpoint, "secret", resolver=resolver)

    with pytest.raises(HostedInferenceError, match="request exceeded"):
        await boundary.post_json("/chat/completions", {"value": "x" * MAX_REQUEST_BYTES})

    assert calls == 0


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectError("certificate verification failed"),
    ],
)
async def test_transport_failure_is_single_attempt_and_sanitized(
    failure: Exception,
) -> None:
    boundary, transport = _boundary(failure)

    with pytest.raises(HostedInferenceError) as caught:
        await boundary.post_json("/chat/completions", {"model": "test"})

    assert transport.calls == 1
    assert caught.value.evidence["attempts"] == 1
    assert caught.value.evidence["retry_count"] == 0


async def test_cancellation_remains_cancellation_and_records_unknown_provider_outcome() -> None:
    boundary, transport = _boundary(asyncio.CancelledError())

    with pytest.raises(HostedInferenceCancelled) as caught:
        await boundary.post_json("/chat/completions", {"model": "test"})

    assert transport.calls == 1
    assert caught.value.evidence["cancellation_outcome"].endswith("provider_outcome_unknown")
