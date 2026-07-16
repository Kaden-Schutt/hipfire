# Copyright (c) Kaden Schutt
"""Validation rules for the append-only repository review protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
import hashlib
from typing import Any

from .canonical import canonical_digest, canonical_json, metadata_digest
from .models import ReviewTarget


_RECORD_TYPES = {"intent", "report", "completion", "review-metadata", "revocation"}
_TARGET_KEYS = {
    "repository",
    "number",
    "head_repository",
    "head_sha",
    "base_ref",
    "base_sha",
    "merge_base_sha",
}


def _plain(value: Any) -> Any:
    if isinstance(value, ReviewTarget):
        return {
            "repository": value.repository,
            "number": value.number,
            "head_repository": value.head_repository,
            "head_sha": value.head_sha,
            "base_ref": value.base_ref,
            "base_sha": value.base_sha,
            "merge_base_sha": value.merge_base_sha,
        }
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _target(value: Any) -> ReviewTarget:
    if isinstance(value, ReviewTarget):
        return value
    if not isinstance(value, Mapping) or set(value) != _TARGET_KEYS:
        raise ValueError("record must contain the full ReviewTarget")
    try:
        return ReviewTarget(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ReviewTarget") from exc


def _text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _time(record: Mapping[str, Any]) -> str:
    value = _text(record, "created_at")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    return value


def _trusted(record: Mapping[str, Any], trusted_authors: Iterable[str] | None) -> None:
    author = _text(record, "author")
    if trusted_authors is not None and author not in set(trusted_authors):
        raise ValueError("author is not trusted")


def _same_binding(record: Mapping[str, Any], intent: Mapping[str, Any]) -> ReviewTarget:
    target = _target(record.get("target"))
    intent_target = _target(intent.get("target"))
    if target != intent_target:
        raise ValueError("record target does not match intent")
    for field in ("target_key", "attempt_id", "intent_node_id"):
        if record.get(field) != intent.get(field):
            raise ValueError(f"record {field} does not match intent")
    if record.get("head_sha") not in (None, target.head_sha):
        raise ValueError("record head SHA does not match target")
    return target


def _digest_without(record: Mapping[str, Any], *excluded: str) -> str:
    value = {key: _plain(item) for key, item in record.items() if key not in set(excluded)}
    return canonical_digest(value)


def _intent_digest(intent: Mapping[str, Any]) -> str:
    return _digest_without(intent, "canonical_digest")


def validate_intent(intent: Mapping[str, Any], *, trusted_authors: Iterable[str] | None = None) -> str:
    if not isinstance(intent, Mapping) or intent.get("record_type") != "intent":
        raise ValueError("invalid intent record")
    required = {
        "record_type", "record_id", "intent_node_id", "target", "target_key", "attempt_id",
        "author", "created_at", "canonical_digest",
    }
    if set(intent) != required:
        raise ValueError("intent has unexpected or missing keys")
    target = _target(intent["target"])
    if intent["target_key"] != target.target_key():
        raise ValueError("intent target_key does not match target")
    _text(intent, "record_id")
    _text(intent, "intent_node_id")
    _text(intent, "attempt_id")
    _time(intent)
    _trusted(intent, trusted_authors)
    digest = _intent_digest(intent)
    if intent["canonical_digest"] not in {"pending", digest}:
        raise ValueError("intent canonical digest does not match record")
    return digest


def validate_report(
    report: Mapping[str, Any],
    intent: Mapping[str, Any],
    *,
    trusted_authors: Iterable[str] | None = None,
) -> str:
    required = {
        "record_type", "record_id", "target", "target_key", "attempt_id", "intent_node_id", "head_sha",
        "author", "created_at", "report_body", "report_body_sha256",
    }
    if not isinstance(report, Mapping) or set(report) != required or report.get("record_type") != "report":
        raise ValueError("invalid report record")
    target = _same_binding(report, intent)
    if report["head_sha"] != target.head_sha:
        raise ValueError("report head SHA does not match target")
    _text(report, "record_id")
    _time(report)
    _trusted(report, trusted_authors)
    body = report["report_body"]
    if not isinstance(body, str):
        raise ValueError("report body must be text")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if report["report_body_sha256"] not in {digest, "sha256:" + digest}:
        raise ValueError("report body digest does not match body")
    return digest


def validate_completion(
    completion: Mapping[str, Any],
    intent: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    canonical_intent: Mapping[str, Any] | None = None,
    trusted_authors: Iterable[str] | None = None,
) -> None:
    required = {
        "record_type", "record_id", "target", "target_key", "attempt_id", "intent_node_id", "head_sha",
        "author", "created_at", "canonical_intent_digest", "report_id",
    }
    if not isinstance(completion, Mapping) or set(completion) != required or completion.get("record_type") != "completion":
        raise ValueError("invalid completion record")
    target = _same_binding(completion, intent)
    if completion["head_sha"] != target.head_sha:
        raise ValueError("completion head SHA does not match target")
    if canonical_intent is not None and completion["intent_node_id"] != canonical_intent.get("intent_node_id"):
        raise ValueError("completion is not for the canonical intent")
    if completion["canonical_intent_digest"] != _intent_digest(intent):
        raise ValueError("completion canonical intent digest does not match")
    if report is None:
        raise ValueError("completion references a deleted or missing report")
    validate_report(report, intent)
    if completion["report_id"] != report.get("record_id"):
        raise ValueError("completion report reference does not match report")
    _text(completion, "record_id")
    _time(completion)
    _trusted(completion, trusted_authors)


def validate_review_metadata(
    metadata: Mapping[str, Any],
    intent: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    trusted_authors: Iterable[str] | None = None,
) -> str:
    required = {
        "record_type", "record_id", "target", "target_key", "attempt_id", "intent_node_id", "head_sha", "author",
        "created_at", "report_id", "report_body_sha256", "metadata_digest",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required or metadata.get("record_type") != "review-metadata":
        raise ValueError("invalid review metadata record")
    target = _same_binding(metadata, intent)
    if metadata["head_sha"] != target.head_sha:
        raise ValueError("review metadata head SHA does not match target")
    validate_report(report, intent, trusted_authors=trusted_authors)
    if metadata["report_id"] != report.get("record_id"):
        raise ValueError("review metadata references a missing or altered report")
    if metadata["report_body_sha256"] != report.get("report_body_sha256"):
        raise ValueError("review metadata report body digest does not match report")
    _text(metadata, "record_id")
    _time(metadata)
    _trusted(metadata, trusted_authors)
    digest = metadata_digest(metadata)
    if metadata["metadata_digest"] != digest:
        raise ValueError("review metadata digest does not match metadata")
    return digest


def validate_revocation(
    revocation: Mapping[str, Any],
    intent: Mapping[str, Any],
    *,
    trusted_authors: Iterable[str] | None = None,
) -> None:
    required = {
        "record_type", "record_id", "target_key", "attempt_id", "canonical_intent_digest", "author",
        "reason", "authenticated", "created_at",
    }
    if not isinstance(revocation, Mapping) or set(revocation) != required or revocation.get("record_type") != "revocation":
        raise ValueError("invalid revocation record")
    target = _target(intent["target"])
    if revocation["target_key"] != target.target_key() or revocation["target_key"] != intent.get("target_key"):
        raise ValueError("revocation target key does not match")
    if revocation["attempt_id"] != intent.get("attempt_id"):
        raise ValueError("revocation attempt ID does not match")
    if revocation["canonical_intent_digest"] != _intent_digest(intent):
        raise ValueError("revocation canonical intent digest does not match")
    if revocation["authenticated"] is not True:
        raise ValueError("revocation is not authenticated")
    _text(revocation, "record_id")
    _text(revocation, "reason")
    _time(revocation)
    _trusted(revocation, trusted_authors)


def validate_append_only(records: Sequence[Mapping[str, Any]], previous: Sequence[Mapping[str, Any]] = ()) -> None:
    """Ensure a log only appends records and never changes or deletes one."""
    old = {record.get("record_id"): canonical_json(_plain(record)) for record in previous}
    seen: set[str] = set()
    for record in records:
        record_id = _text(record, "record_id")
        if record_id in seen:
            raise ValueError("append-only log contains duplicate record ID")
        seen.add(record_id)
        if record_id in old and old[record_id] != canonical_json(_plain(record)):
            raise ValueError("append-only log altered an existing record")
    if not set(old).issubset(seen):
        raise ValueError("append-only log deleted a record")


def elect_canonical_attempt(
    intents: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
    *,
    revocations: Sequence[Mapping[str, Any]] = (),
    reports: Sequence[Mapping[str, Any]] = (),
    trusted_authors: Iterable[str] | None = None,
) -> Mapping[str, Any]:
    valid = []
    revoked: set[tuple[str, str]] = set()
    consumed_revocations: set[int] = set()
    for intent in intents:
        digest = validate_intent(intent, trusted_authors=trusted_authors)
        for index, revocation in enumerate(revocations):
            if (
                revocation.get("target_key") == intent.get("target_key")
                and revocation.get("attempt_id") == intent.get("attempt_id")
            ):
                validate_revocation(revocation, intent, trusted_authors=trusted_authors)
                consumed_revocations.add(index)
                revoked.add((intent["target_key"], intent["attempt_id"]))
        if (intent["target_key"], intent["attempt_id"]) not in revoked:
            valid.append((intent["created_at"], intent["intent_node_id"], intent, digest))
    if len(consumed_revocations) != len(revocations):
        raise ValueError("revocation references an unknown intent")
    if not valid:
        raise ValueError("no valid non-revoked intent")
    valid.sort(key=lambda item: (item[0], item[1]))
    selected = valid[0][2]
    report_by_id = {report.get("record_id"): report for report in reports}
    for completion in completions:
        matching = next((item for item in valid if item[2]["attempt_id"] == completion.get("attempt_id")), None)
        if matching is None:
            raise ValueError("completion is for a noncanonical or revoked intent")
        # A completion may only be accepted after election has selected its intent.
        validate_completion(
            completion,
            matching[2],
            report=report_by_id.get(completion.get("report_id")),
            canonical_intent=selected,
            trusted_authors=trusted_authors,
        )
    return selected


def validate_protocol(
    records: Sequence[Mapping[str, Any]], *, trusted_authors: Iterable[str] | None = None
) -> Mapping[str, Any]:
    """Validate a complete append-only log and return its canonical intent."""
    validate_append_only(records)
    grouped = {record_type: [] for record_type in _RECORD_TYPES}
    for record in records:
        record_type = record.get("record_type")
        if record_type not in grouped:
            raise ValueError("unknown review record type")
        grouped[record_type].append(record)
    intents = grouped["intent"]
    for intent in intents:
        validate_intent(intent, trusted_authors=trusted_authors)
    intent_by_attempt = {intent.get("attempt_id"): intent for intent in intents}
    reports = grouped["report"]
    report_by_id = {report.get("record_id"): report for report in reports}
    for report in reports:
        intent = intent_by_attempt.get(report.get("attempt_id"))
        if intent is None:
            raise ValueError("report references an unknown intent")
        validate_report(report, intent, trusted_authors=trusted_authors)
    selected = elect_canonical_attempt(
        intents,
        grouped["completion"],
        revocations=grouped["revocation"],
        reports=reports,
        trusted_authors=trusted_authors,
    )
    for metadata in grouped["review-metadata"]:
        intent = intent_by_attempt.get(metadata.get("attempt_id"))
        report = report_by_id.get(metadata.get("report_id"))
        if intent is None or report is None:
            raise ValueError("review metadata references a deleted report or unknown intent")
        validate_review_metadata(metadata, intent, report, trusted_authors=trusted_authors)
    return selected
