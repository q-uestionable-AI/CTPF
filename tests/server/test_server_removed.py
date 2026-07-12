"""CI placeholder for the removed Web UI server package.

CTPF reconnect Phase 1 deleted ``src/q_ai/server`` and its former tests.
The Windows CI matrix still runs a ``tests/server`` shard; keep one
assertion here so that path remains collectable until the workflow drops
the shard.
"""

from __future__ import annotations

import importlib.util


def test_legacy_server_package_removed() -> None:
    """Assert the legacy ``q_ai.server`` package is no longer importable."""
    assert importlib.util.find_spec("q_ai.server") is None
