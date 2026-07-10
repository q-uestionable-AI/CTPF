"""Tests for stranded WAITING_FOR_USER run detection and conclusion."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from q_ai.core.db import create_run, get_connection, get_run, update_run_status
from q_ai.core.models import RunStatus
from q_ai.server.app import create_app


def _make_waiting_run(db_path: Path, name: str = "test_workflow") -> str:
    """Seed a run that is in WAITING_FOR_USER status and return its id."""
    with get_connection(db_path) as conn:
        run_id = create_run(conn, module="workflow", name=name)
        update_run_status(conn, run_id, RunStatus.RUNNING)
        update_run_status(conn, run_id, RunStatus.WAITING_FOR_USER)
    return run_id


def _make_running_run(db_path: Path, name: str = "test_workflow") -> str:
    """Seed a run that is RUNNING (not waiting) and return its id."""
    with get_connection(db_path) as conn:
        run_id = create_run(conn, module="workflow", name=name)
        update_run_status(conn, run_id, RunStatus.RUNNING)
    return run_id


@pytest.fixture
def client_with_stranded(tmp_db: Path) -> Generator[tuple[TestClient, str], None, None]:
    """Seed a stranded run, start a fresh TestClient so lifespan picks it up."""
    run_id = _make_waiting_run(tmp_db)
    app = create_app(db_path=tmp_db)
    with TestClient(app) as c:
        yield c, run_id


class TestStartupDetection:
    """_lifespan scans for WAITING_FOR_USER runs at startup."""

    def test_finds_waiting_runs(self, tmp_db: Path) -> None:
        run_id = _make_waiting_run(tmp_db)
        app = create_app(db_path=tmp_db)
        with TestClient(app):
            assert run_id in app.state.stranded_runs
            name, started_at = app.state.stranded_runs[run_id]
            assert name == "test_workflow"
            assert started_at is not None

    def test_ignores_non_waiting_runs(self, tmp_db: Path) -> None:
        running_id = _make_running_run(tmp_db)
        app = create_app(db_path=tmp_db)
        with TestClient(app):
            assert running_id not in app.state.stranded_runs

    def test_empty_when_no_waiting_runs(self, tmp_db: Path) -> None:
        app = create_app(db_path=tmp_db)
        with TestClient(app):
            assert app.state.stranded_runs == {}

    def test_logs_warning_on_detection(
        self, tmp_db: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_id = _make_waiting_run(tmp_db)
        caplog.set_level(logging.WARNING, logger="q_ai.server.app")
        app = create_app(db_path=tmp_db)
        with TestClient(app):
            pass
        messages = [r.getMessage() for r in caplog.records]
        assert any("stranded" in m.lower() and run_id in m for m in messages)

    def test_no_warning_when_clean(self, tmp_db: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="q_ai.server.app")
        app = create_app(db_path=tmp_db)
        with TestClient(app):
            pass
        assert not any("stranded" in r.getMessage().lower() for r in caplog.records)


class TestConcludeStrandedEndpoint:
    """POST /api/runs/{run_id}/conclude-stranded."""

    def test_transitions_waiting_to_cancelled(
        self, tmp_db: Path, client_with_stranded: tuple[TestClient, str]
    ) -> None:
        client, run_id = client_with_stranded
        resp = client.post(f"/api/runs/{run_id}/conclude-stranded")
        assert resp.status_code == 200
        with get_connection(tmp_db) as conn:
            run = get_run(conn, run_id)
            assert run is not None
            assert run.status == RunStatus.CANCELLED
            assert run.finished_at is not None

    def test_removes_from_stranded_state(
        self, client_with_stranded: tuple[TestClient, str]
    ) -> None:
        client, run_id = client_with_stranded
        assert run_id in client.app.state.stranded_runs
        resp = client.post(f"/api/runs/{run_id}/conclude-stranded")
        assert resp.status_code == 200
        assert run_id not in client.app.state.stranded_runs

    def test_returns_empty_body_for_htmx(
        self, client_with_stranded: tuple[TestClient, str]
    ) -> None:
        client, run_id = client_with_stranded
        resp = client.post(f"/api/runs/{run_id}/conclude-stranded")
        assert resp.status_code == 200
        assert resp.text == ""

    def test_rejects_running_run(self, tmp_db: Path) -> None:
        running_id = _make_running_run(tmp_db)
        app = create_app(db_path=tmp_db)
        with TestClient(app) as client:
            resp = client.post(f"/api/runs/{running_id}/conclude-stranded")
        assert resp.status_code == 409
        assert "WAITING_FOR_USER" in resp.json()["detail"]

    def test_rejects_completed_run(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            run_id = create_run(conn, module="workflow", name="done")
            update_run_status(conn, run_id, RunStatus.COMPLETED)
        app = create_app(db_path=tmp_db)
        with TestClient(app) as client:
            resp = client.post(f"/api/runs/{run_id}/conclude-stranded")
        assert resp.status_code == 409

    def test_missing_run_returns_404(self, tmp_db: Path) -> None:
        app = create_app(db_path=tmp_db)
        with TestClient(app) as client:
            resp = client.post("/api/runs/does-not-exist/conclude-stranded")
        assert resp.status_code == 404

    def test_rejects_when_active_runner_exists(
        self, client_with_stranded: tuple[TestClient, str]
    ) -> None:
        client, run_id = client_with_stranded
        client.app.state.active_workflows[run_id] = object()
        try:
            resp = client.post(f"/api/runs/{run_id}/conclude-stranded")
            assert resp.status_code == 409
            assert "active runner" in resp.json()["detail"].lower()
            with get_connection(client.app.state.db_path) as conn:
                run = get_run(conn, run_id)
                assert run is not None
                assert run.status == RunStatus.WAITING_FOR_USER
        finally:
            client.app.state.active_workflows.pop(run_id, None)

    def test_normal_run_status_unaffected_by_endpoint(self, tmp_db: Path) -> None:
        """Active RUNNING runs are untouched — only WAITING_FOR_USER rows transition."""
        running_id = _make_running_run(tmp_db)
        app = create_app(db_path=tmp_db)
        with TestClient(app) as client:
            client.post(f"/api/runs/{running_id}/conclude-stranded")
        with get_connection(tmp_db) as conn:
            run = get_run(conn, running_id)
            assert run is not None
            assert run.status == RunStatus.RUNNING


class TestRunHistoryBanner:
    """The /runs page exposes a banner listing stranded runs."""

    def test_banner_rendered_when_stranded(
        self, client_with_stranded: tuple[TestClient, str]
    ) -> None:
        client, run_id = client_with_stranded
        resp = client.get("/runs")
        assert resp.status_code == 200
        html = resp.text
        assert "stranded-banner" in html
        assert run_id[:8] in html
        assert "cannot be resumed" in html

    def test_banner_absent_when_clean(self, tmp_db: Path) -> None:
        app = create_app(db_path=tmp_db)
        with TestClient(app) as client:
            resp = client.get("/runs")
        assert resp.status_code == 200
        assert "stranded-banner" not in resp.text

    def test_banner_excludes_run_with_active_runner(
        self, client_with_stranded: tuple[TestClient, str]
    ) -> None:
        client, run_id = client_with_stranded
        client.app.state.active_workflows[run_id] = object()
        try:
            resp = client.get("/runs")
            assert "stranded-banner" not in resp.text
        finally:
            client.app.state.active_workflows.pop(run_id, None)
