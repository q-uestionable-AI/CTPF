"""Bounded canonical JSON for CTPF authorization records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

CANONICALIZATION_ID = "ctpf-canonical-json-v1"
MAX_INPUT_BYTES = 65_536
MAX_DEPTH = 16
MAX_COLLECTION_ITEMS = 1_024
MAX_STRING_LENGTH = 16_384
MAX_INTEGER = (2**63) - 1


class CanonicalizationError(ValueError):
    """Raised when an authorization record cannot be canonicalized safely."""


def load_canonical_object(raw: str | bytes) -> dict[str, Any]:
    """Parse one bounded JSON object and reject ambiguous JSON features.

    Args:
        raw: UTF-8 JSON text or bytes.

    Returns:
        Parsed JSON object after recursive validation.

    Raises:
        CanonicalizationError: If the input is empty, oversized, malformed,
            duplicated, non-object, or contains unsupported values.
    """
    text = _decode_input(raw)
    try:
        parsed: Any = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CanonicalizationError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CanonicalizationError("canonical JSON root must be an object")
    _validate_value(parsed, depth=0)
    return parsed


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 bytes for one validated object.

    Args:
        payload: Object containing only canonical-JSON values.

    Returns:
        Deterministic UTF-8 encoding.

    Raises:
        CanonicalizationError: If the object contains unsupported values or
            exceeds the canonical record bounds.
    """
    normalized = dict(payload)
    _validate_value(normalized, depth=0)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(f"canonical JSON encoding failed: {exc}") from exc
    if len(encoded) > MAX_INPUT_BYTES:
        raise CanonicalizationError(f"canonical JSON exceeds the {MAX_INPUT_BYTES}-byte limit")
    return encoded


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON text for one validated object.

    Args:
        payload: Object containing only canonical-JSON values.

    Returns:
        Deterministic JSON text.
    """
    return canonical_bytes(payload).decode("utf-8")


def sha256_digest(payload: Mapping[str, Any]) -> str:
    """Return the lowercase SHA-256 digest of canonical object bytes.

    Args:
        payload: Object containing only canonical-JSON values.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _decode_input(raw: str | bytes) -> str:
    if isinstance(raw, bytes):
        if len(raw) > MAX_INPUT_BYTES:
            raise CanonicalizationError(f"JSON input exceeds the {MAX_INPUT_BYTES}-byte limit")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CanonicalizationError(f"JSON input is not valid UTF-8: {exc}") from exc
    elif isinstance(raw, str):
        text = raw
        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            raise CanonicalizationError(f"JSON input exceeds the {MAX_INPUT_BYTES}-byte limit")
    else:
        raise CanonicalizationError("JSON input must be text or bytes")
    if not text.strip():
        raise CanonicalizationError("JSON input must not be empty")
    return text


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) > MAX_COLLECTION_ITEMS:
        raise CanonicalizationError(f"JSON object exceeds the {MAX_COLLECTION_ITEMS}-item limit")
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalizationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_float(_value: str) -> NoReturn:
    raise CanonicalizationError("floating-point JSON values are not permitted")


def _reject_constant(value: str) -> NoReturn:
    raise CanonicalizationError(f"non-finite JSON value is not permitted: {value}")


def _validate_value(value: Any, *, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise CanonicalizationError(f"JSON nesting exceeds the depth limit of {MAX_DEPTH}")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_INTEGER:
            raise CanonicalizationError("JSON integer exceeds the signed 64-bit limit")
        return
    if isinstance(value, str):
        if not value or len(value) > MAX_STRING_LENGTH:
            raise CanonicalizationError(
                f"JSON strings must contain 1-{MAX_STRING_LENGTH} characters"
            )
        return
    if isinstance(value, Mapping):
        _validate_mapping(value, depth=depth)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _validate_sequence(value, depth=depth)
        return
    raise CanonicalizationError(f"unsupported canonical JSON value: {type(value).__name__}")


def _validate_mapping(value: Mapping[Any, Any], *, depth: int) -> None:
    if len(value) > MAX_COLLECTION_ITEMS:
        raise CanonicalizationError(f"JSON object exceeds the {MAX_COLLECTION_ITEMS}-item limit")
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > MAX_STRING_LENGTH:
            raise CanonicalizationError("JSON object keys must be bounded non-empty strings")
        _validate_value(item, depth=depth + 1)


def _validate_sequence(value: Sequence[Any], *, depth: int) -> None:
    if len(value) > MAX_COLLECTION_ITEMS:
        raise CanonicalizationError(f"JSON array exceeds the {MAX_COLLECTION_ITEMS}-item limit")
    for item in value:
        _validate_value(item, depth=depth + 1)
