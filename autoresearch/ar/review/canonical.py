# Copyright (c) Kaden Schutt
"""Small, dependency-free canonical JSON helpers for review records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
import hashlib
import json
import math
import re
from typing import Any


DEFAULT_MAX_BYTES = 1 << 20
MAX_SAFE_INTEGER = (2**53) - 1
_NUMBER_RE = re.compile(r"^(?P<sign>-?)(?P<mantissa>\d+(?:\.\d+)?)(?:[eE](?P<exp>[+-]?\d+))?$")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _key_order(key: str) -> bytes:
    # JCS sorts object member names by their UTF-16 code units.
    return key.encode("utf-16-be", "surrogatepass")


def _string(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("unsupported JSON value: object keys must be strings")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("unsupported JSON value: lone surrogate")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("numbers must be finite")
    if value == 0:
        return "0"
    text = repr(value).lower()
    match = _NUMBER_RE.fullmatch(text)
    if match is None:
        raise ValueError("unsupported number")
    sign = match.group("sign")
    mantissa = match.group("mantissa")
    exponent = int(match.group("exp") or 0)
    if "." in mantissa:
        whole, fraction = mantissa.split(".")
        digits = whole + fraction
        exponent -= len(fraction)
    else:
        digits = mantissa
    digits = digits.lstrip("0") or "0"
    exponent += len(digits) - 1
    # JSON.stringify uses ordinary notation for [1e-6, 1e21).
    if -6 <= exponent < 21:
        decimal_index = exponent + 1
        if decimal_index <= 0:
            result = "0." + "0" * (-decimal_index) + digits
        elif decimal_index >= len(digits):
            result = digits + "0" * (decimal_index - len(digits))
        else:
            result = digits[:decimal_index] + "." + digits[decimal_index:]
        result = result.rstrip("0").rstrip(".") if "." in result else result
        return sign + result
    exponent_text = ("+" if exponent >= 0 else "") + str(exponent)
    coefficient = digits if len(digits) == 1 else digits[0] + "." + digits[1:]
    return sign + coefficient + "e" + exponent_text


def _encode(value: Any) -> bytes:
    value = _plain(value)
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int) and not isinstance(value, bool):
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise ValueError("integer is outside the IEEE-754 safe range")
        return str(value).encode("ascii")
    if isinstance(value, float):
        return _float(value).encode("ascii")
    if isinstance(value, str):
        return _string(value).encode("utf-8")
    if isinstance(value, (list, tuple)):
        return b"[" + b",".join(_encode(item) for item in value) + b"]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("unsupported JSON value: object keys must be strings")
        members = []
        for key in sorted(value, key=_key_order):
            members.append(_string(key).encode("utf-8") + b":" + _encode(value[key]))
        return b"{" + b",".join(members) + b"}"
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any, *, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    """Encode supported values using deterministic RFC 8785-compatible JSON."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    encoded = _encode(value)
    if len(encoded) > max_bytes:
        raise ValueError("canonical JSON exceeds configured byte limit")
    return encoded


def canonical_loads(payload: str | bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Any:
    """Parse JSON while rejecting duplicate keys and non-standard constants."""
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise ValueError("JSON input must be text or bytes")
    if len(raw) > max_bytes:
        raise ValueError("JSON exceeds configured byte limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"non-finite number {value} is not supported")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed JSON") from exc
    canonical_json(value, max_bytes=max_bytes)
    return value


def metadata_digest(metadata: Mapping[str, Any], *, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Hash metadata without its self-referential digest field."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    if "report_body_sha256" not in metadata:
        raise ValueError("metadata must include report_body_sha256")
    value = {key: item for key, item in metadata.items() if key != "metadata_digest"}
    return hashlib.sha256(canonical_json(value, max_bytes=max_bytes)).hexdigest()


def canonical_digest(value: Any, *, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    return hashlib.sha256(canonical_json(value, max_bytes=max_bytes)).hexdigest()
