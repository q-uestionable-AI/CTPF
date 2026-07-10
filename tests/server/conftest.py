"""Shared fixtures for server tests."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from q_ai.core.schema import migrate
from q_ai.server.app import create_app


@pytest.fixture(scope="session")
def migrated_db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create one pristine migrated database for the server test session."""
    template_path = tmp_path_factory.mktemp("server-db-template") / "template.db"
    with sqlite3.connect(template_path) as conn:
        migrate(conn)
        conn.commit()
    return template_path


def _copy_migrated_database(template_path: Path, tmp_path: Path) -> Path:
    """Copy the pristine database into a test's isolated directory."""
    db_path = tmp_path / "test.db"
    shutil.copyfile(template_path, db_path)
    return db_path


@pytest.fixture
def tmp_db(tmp_path: Path, migrated_db_template: Path) -> Path:
    """Provide an isolated migrated database for a server test."""
    return _copy_migrated_database(migrated_db_template, tmp_path)


@pytest.fixture
def db_path(tmp_path: Path, migrated_db_template: Path) -> Path:
    """Provide the alternate database fixture name used by runner tests."""
    return _copy_migrated_database(migrated_db_template, tmp_path)


@pytest.fixture
def client(tmp_db: Path) -> Generator[TestClient, None, None]:
    """Create a test client with a temporary database."""
    app = create_app(db_path=tmp_db)
    with TestClient(app) as c:
        yield c
