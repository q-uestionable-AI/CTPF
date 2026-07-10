"""Tests for managed-listener reattach in the web-server lifespan handler.

At startup the server reads ``~/.qai/active-callback`` and classifies
any live listener as either an ``adopted`` managed listener (if
``manager == "web-ui"``) or a foreign listener (otherwise). These
tests drive that classification through :func:`create_app` with a
temporary ``qai_dir`` so the real ``~/.qai`` is never touched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from q_ai.ipi.callback_state import build_state, write_state
from q_ai.server.app import create_app
from q_ai.services.managed_listener import ListenerState


@pytest.fixture
def qai_dir(tmp_path: Path) -> Path:
    """Isolated ``~/.qai`` replacement."""
    d = tmp_path / ".qai"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# web-ui manager → adopted
# ---------------------------------------------------------------------------


def test_live_web_ui_state_file_produces_adopted_entry(
    tmp_db: Path,
    qai_dir: Path,
) -> None:
    write_state(
        build_state(
            public_url="https://adopted.trycloudflare.com",
            provider="cloudflare",
            local_host="127.0.0.1",
            local_port=8080,
            listener_pid=os.getpid(),
            manager="web-ui",
        ),
        qai_dir=qai_dir,
    )

    app = create_app(db_path=tmp_db, qai_dir=qai_dir)
    with TestClient(app):
        assert app.state.foreign_listener is None
        assert len(app.state.managed_listeners) == 1
        handle = next(iter(app.state.managed_listeners.values()))
        assert handle.state == ListenerState.ADOPTED
        assert handle.pid == os.getpid()
        assert handle.public_url == "https://adopted.trycloudflare.com"
        assert handle.provider == "cloudflare"
        # Adopted handles carry no stderr ring — server did not spawn.
        assert handle.stderr_tail is None


# ---------------------------------------------------------------------------
# cli / legacy / unknown manager → foreign
# ---------------------------------------------------------------------------


def test_live_cli_state_file_populates_foreign_listener(
    tmp_db: Path,
    qai_dir: Path,
) -> None:
    write_state(
        build_state(
            public_url="https://cli.trycloudflare.com",
            provider="cloudflare",
            local_host="127.0.0.1",
            local_port=8080,
            listener_pid=os.getpid(),
            manager="cli",
        ),
        qai_dir=qai_dir,
    )

    app = create_app(db_path=tmp_db, qai_dir=qai_dir)
    with TestClient(app):
        assert app.state.managed_listeners == {}
        foreign = app.state.foreign_listener
        assert foreign is not None
        assert foreign.pid == os.getpid()
        assert foreign.public_url == "https://cli.trycloudflare.com"
        assert foreign.manager == "cli"


def test_legacy_state_file_without_manager_is_treated_as_foreign(
    tmp_db: Path,
    qai_dir: Path,
) -> None:
    """A pre-``manager`` CLI listener has no ``manager`` field; reattach
    classifies it as foreign with ``manager=None``."""
    write_state(
        build_state(
            public_url="https://legacy.trycloudflare.com",
            provider="cloudflare",
            local_host="127.0.0.1",
            local_port=8080,
            listener_pid=os.getpid(),
            # manager omitted
        ),
        qai_dir=qai_dir,
    )

    app = create_app(db_path=tmp_db, qai_dir=qai_dir)
    with TestClient(app):
        assert app.state.managed_listeners == {}
        foreign = app.state.foreign_listener
        assert foreign is not None
        assert foreign.manager is None


# ---------------------------------------------------------------------------
# Dead PID and missing state → clean empty registries
# ---------------------------------------------------------------------------


def test_dead_pid_in_state_file_leaves_both_registries_empty(
    tmp_db: Path,
    qai_dir: Path,
) -> None:
    # Spawn a trivial subprocess and wait for it to exit — its PID is
    # guaranteed-dead afterward regardless of kernel pid_max settings.
    # A hardcoded PID constant (e.g. 999_999) can transiently be a real
    # process on Linux hosts with large pid_max.
    dead_proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dead_proc.wait(timeout=5)
    time.sleep(0.05)  # OS settle for POSIX zombie reap

    write_state(
        build_state(
            public_url="https://dead.trycloudflare.com",
            provider="cloudflare",
            local_host="127.0.0.1",
            local_port=8080,
            listener_pid=dead_proc.pid,
            manager="web-ui",
        ),
        qai_dir=qai_dir,
    )

    app = create_app(db_path=tmp_db, qai_dir=qai_dir)
    with TestClient(app):
        assert app.state.managed_listeners == {}
        assert app.state.foreign_listener is None


def test_no_state_file_leaves_both_registries_empty(
    tmp_db: Path,
    qai_dir: Path,
) -> None:
    app = create_app(db_path=tmp_db, qai_dir=qai_dir)
    with TestClient(app):
        assert app.state.managed_listeners == {}
        assert app.state.foreign_listener is None
