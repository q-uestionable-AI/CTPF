"""Tests for the IPI managed-listener HTTP routes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from q_ai.ipi.callback_state import build_state, write_state
from q_ai.server.app import create_app
from q_ai.services.managed_listener import (
    ForeignListenerRecord,
    ListenerState,
    ManagedListenerConflictError,
    ManagedListenerHandle,
    ManagedListenerStartupError,
    ManagedListenerStuckStopError,
)


@pytest.fixture
def qai_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".qai"
    d.mkdir()
    return d


def _client(tmp_db: Path, qai_dir: Path) -> TestClient:
    app = create_app(db_path=tmp_db, qai_dir=qai_dir)
    return TestClient(app)


def _make_handle(
    listener_id: str = "abc123",
    state: ListenerState = ListenerState.RUNNING,
) -> ManagedListenerHandle:
    return ManagedListenerHandle(
        listener_id=listener_id,
        pid=os.getpid(),
        public_url="https://test.trycloudflare.com",
        provider="cloudflare",
        local_host="127.0.0.1",
        local_port=8080,
        instance_id="inst-1",
        created_at="2026-04-16T12:00:00+00:00",
        state=state,
    )


# ---------------------------------------------------------------------------
# POST /api/ipi/managed-listener/start
# ---------------------------------------------------------------------------


class TestStartEndpoint:
    """Coverage for ``POST /api/ipi/managed-listener/start``: success
    partial rendering, 409 conflict wording, and 502 on startup failure."""

    def test_success_returns_partial_with_public_url(
        self,
        tmp_db: Path,
        qai_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle = _make_handle()

        def _fake_start(
            registry: dict[str, ManagedListenerHandle],
            **_kwargs: object,
        ) -> ManagedListenerHandle:
            registry[handle.listener_id] = handle
            return handle

        import q_ai.server.routes.modules.ipi as routes

        monkeypatch.setattr(routes, "start_managed_listener", _fake_start)

        with _client(tmp_db, qai_dir) as client:
            resp = client.post("/api/ipi/managed-listener/start")

        assert resp.status_code == 200
        assert handle.public_url in resp.text
        assert handle.listener_id in resp.text

    def test_conflict_returns_409_with_rfc_wording(
        self,
        tmp_db: Path,
        qai_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake_start(
            _registry: dict[str, ManagedListenerHandle],
            **_kwargs: object,
        ) -> ManagedListenerHandle:
            raise ManagedListenerConflictError(
                "A tunneled listener is already active (PID 42, started via cli). "
                "Stop it first or use the CLI for a parallel tunnel."
            )

        import q_ai.server.routes.modules.ipi as routes

        monkeypatch.setattr(routes, "start_managed_listener", _fake_start)

        with _client(tmp_db, qai_dir) as client:
            resp = client.post("/api/ipi/managed-listener/start")

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "already active" in detail
        assert "PID 42" in detail
        assert "started via cli" in detail

    def test_startup_failure_returns_502(
        self,
        tmp_db: Path,
        qai_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake_start(
            _registry: dict[str, ManagedListenerHandle],
            **_kwargs: object,
        ) -> ManagedListenerHandle:
            raise ManagedListenerStartupError(
                "Listener did not publish its callback state within 45s; subprocess terminated."
            )

        import q_ai.server.routes.modules.ipi as routes

        monkeypatch.setattr(routes, "start_managed_listener", _fake_start)

        with _client(tmp_db, qai_dir) as client:
            resp = client.post("/api/ipi/managed-listener/start")

        assert resp.status_code == 502
        assert "did not publish" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/ipi/managed-listener/stop
# ---------------------------------------------------------------------------


class TestStopEndpoint:
    """Coverage for ``POST /api/ipi/managed-listener/stop``: JSON/form-body
    parsing, 204 idempotency on unknown ids, 422 on bad body, and 500 on
    stuck-stop."""

    def test_success_returns_204(
        self,
        tmp_db: Path,
        qai_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def _fake_stop(
            _registry: dict[str, ManagedListenerHandle],
            listener_id: str,
            **_kwargs: object,
        ) -> None:
            calls.append(listener_id)

        import q_ai.server.routes.modules.ipi as routes

        monkeypatch.setattr(routes, "stop_managed_listener", _fake_stop)

        with _client(tmp_db, qai_dir) as client:
            resp = client.post(
                "/api/ipi/managed-listener/stop",
                json={"listener_id": "abc123"},
            )

        assert resp.status_code == 204
        assert resp.content == b""
        assert calls == ["abc123"]

    def test_success_via_form_encoded_body(
        self,
        tmp_db: Path,
        qai_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTMX native hx-vals sends application/x-www-form-urlencoded; the
        route must accept that too."""
        calls: list[str] = []

        def _fake_stop(
            _registry: dict[str, ManagedListenerHandle],
            listener_id: str,
            **_kwargs: object,
        ) -> None:
            calls.append(listener_id)

        import q_ai.server.routes.modules.ipi as routes

        monkeypatch.setattr(routes, "stop_managed_listener", _fake_stop)

        with _client(tmp_db, qai_dir) as client:
            resp = client.post(
                "/api/ipi/managed-listener/stop",
                data={"listener_id": "via-form"},
            )

        assert resp.status_code == 204
        assert calls == ["via-form"]

    def test_missing_listener_id_returns_422(
        self,
        tmp_db: Path,
        qai_dir: Path,
    ) -> None:
        with _client(tmp_db, qai_dir) as client:
            resp = client.post("/api/ipi/managed-listener/stop", json={})

        assert resp.status_code == 422
        assert "listener_id" in resp.json()["detail"]

    def test_unknown_listener_id_returns_204_idempotent(
        self,
        tmp_db: Path,
        qai_dir: Path,
    ) -> None:
        """The service layer treats unknown ids as no-op; the route surfaces
        204 rather than 404 so repeated clicks from a stale UI don't error."""
        with _client(tmp_db, qai_dir) as client:
            resp = client.post(
                "/api/ipi/managed-listener/stop",
                json={"listener_id": "does-not-exist"},
            )

        assert resp.status_code == 204

    def test_stuck_stop_returns_500(
        self,
        tmp_db: Path,
        qai_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake_stop(
            _registry: dict[str, ManagedListenerHandle],
            _listener_id: str,
            **_kwargs: object,
        ) -> None:
            raise ManagedListenerStuckStopError(
                "Failed to stop listener (PID 7777); manual termination may be required"
            )

        import q_ai.server.routes.modules.ipi as routes

        monkeypatch.setattr(routes, "stop_managed_listener", _fake_stop)

        with _client(tmp_db, qai_dir) as client:
            resp = client.post(
                "/api/ipi/managed-listener/stop",
                json={"listener_id": "stuck"},
            )

        assert resp.status_code == 500
        assert "manual termination" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/ipi/managed-listener
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    """Coverage for ``GET /api/ipi/managed-listener``: empty-state prompt
    plus running / adopted / crashed / foreign-card rendering variants."""

    def test_empty_renders_prompt(self, tmp_db: Path, qai_dir: Path) -> None:
        with _client(tmp_db, qai_dir) as client:
            resp = client.get("/api/ipi/managed-listener")

        assert resp.status_code == 200
        assert "No tunneled listener active" in resp.text

    def test_running_handle_renders_public_url_and_stop_button(
        self,
        tmp_db: Path,
        qai_dir: Path,
    ) -> None:
        app = create_app(db_path=tmp_db, qai_dir=qai_dir)
        with TestClient(app) as client:
            handle = _make_handle("abc", state=ListenerState.RUNNING)
            app.state.managed_listeners[handle.listener_id] = handle
            resp = client.get("/api/ipi/managed-listener")

        assert resp.status_code == 200
        assert handle.public_url in resp.text
        assert "running" in resp.text
        assert 'data-testid="ipi-managed-listener-stop"' in resp.text

    def test_adopted_handle_shows_stderr_unavailable_note(
        self,
        tmp_db: Path,
        qai_dir: Path,
    ) -> None:
        app = create_app(db_path=tmp_db, qai_dir=qai_dir)
        with TestClient(app) as client:
            app.state.managed_listeners["abc"] = _make_handle("abc", state=ListenerState.ADOPTED)
            resp = client.get("/api/ipi/managed-listener")

        assert resp.status_code == 200
        assert "stderr unavailable" in resp.text
        assert "adopted from a prior server instance" in resp.text

    def test_crashed_handle_shows_exit_code_and_clear_button(
        self,
        tmp_db: Path,
        qai_dir: Path,
    ) -> None:
        app = create_app(db_path=tmp_db, qai_dir=qai_dir)
        with TestClient(app) as client:
            crashed = _make_handle("crashed-id", state=ListenerState.CRASHED)
            crashed.exit_code = 7
            app.state.managed_listeners["crashed-id"] = crashed
            resp = client.get("/api/ipi/managed-listener")

        assert resp.status_code == 200
        assert "exit code 7" in resp.text
        assert 'data-testid="ipi-managed-listener-clear"' in resp.text

    def test_foreign_listener_renders_read_only_card(
        self,
        tmp_db: Path,
        qai_dir: Path,
    ) -> None:
        # Seeding a live state file during startup populates the foreign record.
        foreign_url = "https://foreign.trycloudflare.com"
        write_state(
            build_state(
                public_url=foreign_url,
                provider="cloudflare",
                local_host="127.0.0.1",
                local_port=8080,
                listener_pid=os.getpid(),
                manager="cli",
            ),
            qai_dir=qai_dir,
        )
        app = create_app(db_path=tmp_db, qai_dir=qai_dir)
        with TestClient(app) as client:
            assert isinstance(app.state.foreign_listener, ForeignListenerRecord)
            resp = client.get("/api/ipi/managed-listener")

        assert resp.status_code == 200
        assert "External listener detected" in resp.text
        assert foreign_url in resp.text
        # No Stop/Clear control on the foreign card.
        assert 'data-testid="ipi-managed-listener-stop"' not in resp.text
