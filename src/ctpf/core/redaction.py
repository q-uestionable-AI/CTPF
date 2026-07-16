"""Bounded recursive sanitization for untrusted evidence and errors."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "<redacted>"
TRUNCATED = "<truncated>"
UNSUPPORTED = "<unsupported>"
_MAX_DEPTH = 12
_MAX_ITEMS = 256
_MAX_STRING_CHARS = 32_768
_SENSITIVE_KEY = re.compile(
    r"(?:authorization|api[-_]?key|auth[-_]?token|credential|oauth|password|secret)",
    re.IGNORECASE,
)


def redact_text(value: str, secrets: Sequence[str] = ()) -> str:
    """Redact exact secret values and bound one untrusted string.

    Args:
        value: Untrusted text.
        secrets: Exact non-empty secret strings to replace.

    Returns:
        Bounded text with every supplied secret replaced.
    """
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, REDACTED)
    if len(redacted) <= _MAX_STRING_CHARS:
        return redacted
    return redacted[:_MAX_STRING_CHARS] + TRUNCATED


def sanitize_evidence(value: Any, secrets: Sequence[str] = ()) -> Any:
    """Return a recursively redacted, bounded JSON-compatible value.

    Args:
        value: Untrusted external value.
        secrets: Exact non-empty secret strings to replace.

    Returns:
        A JSON-compatible value without arbitrary object stringification.
    """
    return _sanitize(value, tuple(secret for secret in secrets if secret), depth=0)


def _sanitize(  # noqa: PLR0911 - explicit JSON-type dispatch
    value: Any,
    secrets: tuple[str, ...],
    *,
    depth: int,
) -> Any:
    if depth > _MAX_DEPTH:
        return TRUNCATED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else UNSUPPORTED
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, Mapping):
        return _sanitize_mapping(value, secrets, depth=depth)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        items = [_sanitize(item, secrets, depth=depth + 1) for item in value[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            items.append(TRUNCATED)
        return items
    return UNSUPPORTED


def _sanitize_mapping(
    value: Mapping[Any, Any],
    secrets: tuple[str, ...],
    *,
    depth: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (raw_key, raw_value) in enumerate(value.items()):
        if index >= _MAX_ITEMS:
            result[TRUNCATED] = TRUNCATED
            break
        key = raw_key if isinstance(raw_key, str) else UNSUPPORTED
        key = redact_text(key, secrets)
        result[key] = (
            REDACTED
            if _SENSITIVE_KEY.search(key)
            else _sanitize(raw_value, secrets, depth=depth + 1)
        )
    return result
