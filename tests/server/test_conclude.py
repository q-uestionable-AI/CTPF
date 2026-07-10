"""Tests for the Conclude Campaign endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from q_ai.core.db import create_run, get_connection, get_run, update_run_status
from q_ai.core.models import RunStatus
from q_ai.server.app import create_app


@pytest.fixture
def client(tmp_db: Path) -> TestClient:
    """Create a test client with a temporary database."""
    app = create_app(db_path=tmp_db)
    return TestClient(app)


class TestConcludeCampaign:
    """Tests for POST /api/workflows/{run_id}/conclude."""

    def test_conclude_waiting_run(self, tmp_db: Path, client: TestClient) -> None:
        """Concluding a WAITING_FOR_USER run transitions to COMPLETED."""
        with get_connection(tmp_db) as conn:
            run_id = create_run(conn, module="workflow", name="test_docs")
            update_run_status(conn, run_id, RunStatus.RUNNING)
            update_run_status(conn, run_id, RunStatus.WAITING_FOR_USER)

        resp = client.post(f"/api/workflows/{run_id}/conclude")
        assert resp.status_code == 200
        assert resp.json()["status"] == "concluded"

        with get_connection(tmp_db) as conn:
            run = get_run(conn, run_id)
            assert run is not None
            assert run.status == RunStatus.COMPLETED
            assert run.finished_at is not None

    def test_conclude_running_run(self, tmp_db: Path, client: TestClient) -> None:
        """Concluding a RUNNING run transitions to COMPLETED."""
        with get_connection(tmp_db) as conn:
            run_id = create_run(conn, module="workflow", name="test_assistant")
            update_run_status(conn, run_id, RunStatus.RUNNING)

        resp = client.post(f"/api/workflows/{run_id}/conclude")
        assert resp.status_code == 200

        with get_connection(tmp_db) as conn:
            run = get_run(conn, run_id)
            assert run is not None
            assert run.status == RunStatus.COMPLETED

    def test_conclude_nonexistent_run(self, client: TestClient) -> None:
        """Concluding a nonexistent run returns 404."""
        resp = client.post("/api/workflows/nonexistent123/conclude")
        assert resp.status_code == 404

    def test_conclude_already_completed_is_idempotent(
        self, tmp_db: Path, client: TestClient
    ) -> None:
        """Concluding an already-COMPLETED run returns success (idempotent)."""
        with get_connection(tmp_db) as conn:
            run_id = create_run(conn, module="workflow", name="test_docs")
            update_run_status(conn, run_id, RunStatus.COMPLETED)

        resp = client.post(f"/api/workflows/{run_id}/conclude")
        assert resp.status_code == 200
        assert resp.json()["status"] == "concluded"

    def test_conclude_already_failed_is_idempotent(self, tmp_db: Path, client: TestClient) -> None:
        """Concluding an already-FAILED run returns success (idempotent)."""
        with get_connection(tmp_db) as conn:
            run_id = create_run(conn, module="workflow", name="test_docs")
            update_run_status(conn, run_id, RunStatus.FAILED)

        resp = client.post(f"/api/workflows/{run_id}/conclude")
        assert resp.status_code == 200
        assert resp.json()["status"] == "concluded"

    def test_conclude_transitions_child_runs(self, tmp_db: Path, client: TestClient) -> None:
        """Concluding a parent also transitions WAITING_FOR_USER children."""
        with get_connection(tmp_db) as conn:
            parent_id = create_run(conn, module="workflow", name="test_docs")
            update_run_status(conn, parent_id, RunStatus.WAITING_FOR_USER)

            # Child in WAITING_FOR_USER — should be transitioned
            child1_id = create_run(conn, module="ipi", parent_run_id=parent_id)
            update_run_status(conn, child1_id, RunStatus.WAITING_FOR_USER)

            # Child already COMPLETED — should be left alone
            child2_id = create_run(conn, module="ipi", parent_run_id=parent_id)
            update_run_status(conn, child2_id, RunStatus.COMPLETED)

        resp = client.post(f"/api/workflows/{parent_id}/conclude")
        assert resp.status_code == 200

        with get_connection(tmp_db) as conn:
            parent = get_run(conn, parent_id)
            child1 = get_run(conn, child1_id)
            child2 = get_run(conn, child2_id)

            assert parent is not None
            assert parent.status == RunStatus.COMPLETED
            assert parent.finished_at is not None

            assert child1 is not None
            assert child1.status == RunStatus.COMPLETED
            assert child1.finished_at is not None

            assert child2 is not None
            assert child2.status == RunStatus.COMPLETED  # unchanged
