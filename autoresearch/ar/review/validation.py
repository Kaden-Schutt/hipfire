# Copyright (c) Kaden Schutt
"""Pure, deterministic rendering for the protected validation section."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
from typing import Any

from .canonical import canonical_json


VALIDATION_HEADING = "### Hardware/model smoke validation"
VALIDATION_HEADER = (
    "| ID | Capability | Model architecture | Representative | Covered hardware | "
    "Status | Validator | Result |"
)
VALIDATION_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- |"
MAX_VALIDATION_ROWS = 64
MAX_VALIDATION_FIELD_BYTES = 128
MAX_VALIDATION_RATIONALE_BYTES = 1024
MAX_VALIDATION_RESULT_BYTES = 128
MAX_VALIDATION_LEDGER_BYTES = 64 * 1024
VALIDATION_ROW_FIELDS = frozenset({
    "request_id", "profile_snapshot", "profile_digest", "capability_id", "contract_digest",
    "model_architecture", "fixture_id", "fixture_digest", "representative_hardware",
    "covered_hardware", "coverage_kind", "status", "validator_snapshot", "result_snapshot", "rationales",
})


def _bounded_text(value: Any, name: str, limit: int = MAX_VALIDATION_FIELD_BYTES) -> None:
    if not isinstance(value, str) or len(value.encode("utf-8")) > limit:
        raise ValueError(f"{name} exceeds its maximum UTF-8 length")


def validate_ledger_row_mapping(value: Any) -> Mapping[str, Any]:
    """Validate dependency-free row shape and all bounded row fields."""
    if not isinstance(value, Mapping) or frozenset(value) != VALIDATION_ROW_FIELDS:
        raise ValueError("validation ledger row has unexpected or missing keys")
    if value["status"] != "pending" or value["validator_snapshot"] != {} or value["result_snapshot"] != {}:
        raise ValueError("validation ledger row snapshots must be empty and pending")
    profile = value["profile_snapshot"]
    if not isinstance(profile, Mapping):
        raise ValueError("validation profile snapshot must be an object")
    for name in (
        "request_id", "profile_digest", "capability_id", "contract_digest", "model_architecture",
        "fixture_id", "fixture_digest", "representative_hardware", "coverage_kind", "status",
    ):
        _bounded_text(value[name], name)
    for name in ("id", "capability_id", "model_architecture", "fixture_id", "fixture_digest", "representative_hardware"):
        _bounded_text(profile.get(name), f"profile_snapshot.{name}")
    covered = value["covered_hardware"]
    if not isinstance(covered, (list, tuple)) or not covered:
        raise ValueError("covered_hardware must be a non-empty list")
    for item in covered:
        _bounded_text(item, "covered_hardware")
    rationales = value["rationales"]
    if not isinstance(rationales, (list, tuple)) or any(not isinstance(item, str) for item in rationales):
        raise ValueError("ledger rationales must be a list of strings")
    for item in rationales:
        _bounded_text(item, "rationale", MAX_VALIDATION_RATIONALE_BYTES)
    if len(canonical_json(value["result_snapshot"])) > MAX_VALIDATION_RESULT_BYTES:
        raise ValueError("validation result snapshot exceeds 128 bytes")
    return value


def validate_ledger_payload_shape(ledger: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate row count, canonical serialized size, shape, and ordering."""
    if not isinstance(ledger, (list, tuple)) or len(ledger) > MAX_VALIDATION_ROWS:
        raise ValueError("validation ledger must contain at most 64 rows")
    if len(canonical_json(ledger)) > MAX_VALIDATION_LEDGER_BYTES:
        raise ValueError("validation ledger exceeds 64 KiB")
    rows = tuple(validate_ledger_row_mapping(item) for item in ledger)
    request_ids = tuple(row["request_id"] for row in rows)
    if len(request_ids) != len(set(request_ids)) or request_ids != tuple(sorted(request_ids)):
        raise ValueError("validation ledger request IDs must be sorted and unique")
    return rows


def _cell(value: Any) -> str:
    normalized = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(normalized, quote=True).replace("|", "&#124;").replace("\n", "<br>")


def _snapshot(value: Any) -> str:
    return "—" if not value else canonical_json(value).decode("utf-8")


def _row_value(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def render_validation_section(rows: Sequence[Any], *, exempt: bool = False, scope: Any = None) -> str:
    """Render the exact visible section represented by a typed or raw ledger."""
    lines = [VALIDATION_HEADING, ""]
    if scope is not None:
        model_architectures = _row_value(scope, "model_architectures")
        hardware_architectures = _row_value(scope, "hardware_architectures")
        lines.append(
            "Scope: model_architectures=" + ",".join(_cell(item) for item in model_architectures)
            + "; hardware_architectures=" + ",".join(_cell(item) for item in hardware_architectures)
        )
        lines.append("")
    if rows:
        lines.extend((VALIDATION_HEADER, VALIDATION_SEPARATOR))
        for row in sorted(rows, key=lambda item: _row_value(item, "request_id")):
            lines.append("| " + " | ".join((
                _cell(_row_value(row, "request_id")),
                _cell(_row_value(row, "capability_id")),
                _cell(_row_value(row, "model_architecture")),
                _cell(_row_value(row, "representative_hardware")),
                _cell(", ".join(_row_value(row, "covered_hardware"))),
                _cell(_row_value(row, "status")),
                _cell(_snapshot(_row_value(row, "validator_snapshot"))),
                _cell(_snapshot(_row_value(row, "result_snapshot"))),
            )) + " |")
    elif exempt:
        lines.append("No validation required (protected exemption).")
    else:
        raise ValueError("empty validation ledger is not exempt")
    return "\n".join(lines)


def validate_rendered_validation_section(
    body: str, rows: Sequence[Any], *, exempt: bool = False, scope: Any = None,
) -> None:
    """Require one exact validation section at the end of a report body."""
    expected = render_validation_section(rows, exempt=exempt, scope=scope)
    expected_suffix = "\n\n" + expected
    if not body.endswith(expected_suffix):
        raise ValueError("report validation section does not match validation ledger")


__all__ = [
    "MAX_VALIDATION_FIELD_BYTES", "MAX_VALIDATION_LEDGER_BYTES", "MAX_VALIDATION_RATIONALE_BYTES",
    "MAX_VALIDATION_RESULT_BYTES", "MAX_VALIDATION_ROWS", "VALIDATION_HEADING", "VALIDATION_ROW_FIELDS",
    "VALIDATION_HEADER", "VALIDATION_SEPARATOR",
    "render_validation_section", "validate_ledger_payload_shape", "validate_ledger_row_mapping",
    "validate_rendered_validation_section",
]
