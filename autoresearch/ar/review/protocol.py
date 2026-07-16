# Copyright (c) Kaden Schutt
"""Validation rules for the append-only repository review protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
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


def _time(record: Mapping[str, Any]) -> datetime:
    value = _text(record, "created_at")
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    return timestamp.astimezone(timezone.utc)


def _trusted(record: Mapping[str, Any], trusted_authors: Iterable[str]) -> None:
    author = _text(record, "author")
    if author not in trusted_authors:
        raise ValueError("author is not trusted")


def _trust_policy(trusted_authors: Iterable[str] | None) -> frozenset[str]:
    if trusted_authors is None:
        raise ValueError("trusted_authors policy is required")
    policy = frozenset(trusted_authors)
    if not policy or any(not isinstance(author, str) or not author.strip() for author in policy):
        raise ValueError("trusted_authors policy must not be empty")
    return policy


def _expected_target(value: ReviewTarget) -> ReviewTarget:
    if not isinstance(value, ReviewTarget):
        raise ValueError("expected_target must be a ReviewTarget")
    return value


def _require_expected_target(record: Mapping[str, Any], expected_target: ReviewTarget) -> ReviewTarget:
    target = _target(record.get("target"))
    if target != expected_target or record.get("target_key") != expected_target.target_key():
        raise ValueError("record target does not match expected target")
    return target


def _require_canonical_binding(
    record: Mapping[str, Any], canonical_intent: Mapping[str, Any], digest_field: str
) -> None:
    target = _target(record.get("target"))
    canonical_target = _target(canonical_intent.get("target"))
    if (
        target != canonical_target
        or record.get("target_key") != canonical_intent.get("target_key")
        or record.get("attempt_id") != canonical_intent.get("attempt_id")
        or record.get("intent_node_id") != canonical_intent.get("intent_node_id")
        or record.get("head_sha") != canonical_target.head_sha
        or record.get(digest_field) != canonical_intent.get("canonical_digest")
    ):
        raise ValueError("record is not bound to the canonical intent")


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
    trusted = _trust_policy(trusted_authors)
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
    _trusted(intent, trusted)
    digest = _intent_digest(intent)
    if intent["canonical_digest"] != digest:
        raise ValueError("intent canonical digest does not match record")
    return digest


def validate_report(
    report: Mapping[str, Any],
    intent: Mapping[str, Any],
    *,
    trusted_authors: Iterable[str] | None = None,
) -> str:
    trusted = _trust_policy(trusted_authors)
    required = {
        "record_type", "record_id", "target", "target_key", "attempt_id", "intent_node_id", "head_sha",
        "author", "created_at", "report_body", "report_body_sha256",
    }
    if not isinstance(report, Mapping) or set(report) != required or report.get("record_type") != "report":
        raise ValueError("invalid report record")
    validate_intent(intent, trusted_authors=trusted)
    target = _same_binding(report, intent)
    if report["head_sha"] != target.head_sha:
        raise ValueError("report head SHA does not match target")
    _text(report, "record_id")
    _time(report)
    _trusted(report, trusted)
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
    trusted = _trust_policy(trusted_authors)
    required = {
        "record_type", "record_id", "target", "target_key", "attempt_id", "intent_node_id", "head_sha",
        "author", "created_at", "canonical_intent_digest", "report_id",
    }
    if not isinstance(completion, Mapping) or set(completion) != required or completion.get("record_type") != "completion":
        raise ValueError("invalid completion record")
    validate_intent(intent, trusted_authors=trusted)
    target = _same_binding(completion, intent)
    if completion["head_sha"] != target.head_sha:
        raise ValueError("completion head SHA does not match target")
    if canonical_intent is not None:
        validate_intent(canonical_intent, trusted_authors=trusted)
        canonical_target = _target(canonical_intent["target"])
        if (
            target != canonical_target
            or completion["target_key"] != canonical_intent.get("target_key")
            or completion["attempt_id"] != canonical_intent.get("attempt_id")
            or completion["intent_node_id"] != canonical_intent.get("intent_node_id")
            or completion["head_sha"] != canonical_target.head_sha
            or completion["canonical_intent_digest"] != canonical_intent.get("canonical_digest")
        ):
            raise ValueError("completion is not bound to the canonical intent")
    if completion["canonical_intent_digest"] != _intent_digest(intent):
        raise ValueError("completion canonical intent digest does not match")
    if report is None:
        raise ValueError("completion references a deleted or missing report")
    validate_report(report, intent, trusted_authors=trusted)
    if completion["report_id"] != report.get("record_id"):
        raise ValueError("completion report reference does not match report")
    _text(completion, "record_id")
    _time(completion)
    _trusted(completion, trusted)


def validate_review_metadata(
    metadata: Mapping[str, Any],
    intent: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    canonical_intent: Mapping[str, Any],
    trusted_authors: Iterable[str] | None = None,
) -> str:
    trusted = _trust_policy(trusted_authors)
    required = {
        "record_type", "record_id", "target", "target_key", "attempt_id", "intent_node_id", "head_sha", "author",
        "created_at", "report_id", "report_body_sha256", "canonical_intent_digest", "metadata_digest",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required or metadata.get("record_type") != "review-metadata":
        raise ValueError("invalid review metadata record")
    validate_intent(intent, trusted_authors=trusted)
    validate_intent(canonical_intent, trusted_authors=trusted)
    target = _same_binding(metadata, intent)
    if metadata["head_sha"] != target.head_sha:
        raise ValueError("review metadata head SHA does not match target")
    _require_canonical_binding(metadata, canonical_intent, "canonical_intent_digest")
    validate_report(report, intent, trusted_authors=trusted)
    if metadata["report_id"] != report.get("record_id"):
        raise ValueError("review metadata references a missing or altered report")
    if metadata["report_body_sha256"] != report.get("report_body_sha256"):
        raise ValueError("review metadata report body digest does not match report")
    _text(metadata, "record_id")
    _time(metadata)
    _trusted(metadata, trusted)
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
    trusted = _trust_policy(trusted_authors)
    required = {
        "record_type", "record_id", "target_key", "attempt_id", "canonical_intent_digest", "author",
        "reason", "authenticated", "created_at",
    }
    if not isinstance(revocation, Mapping) or set(revocation) != required or revocation.get("record_type") != "revocation":
        raise ValueError("invalid revocation record")
    canonical_digest = validate_intent(intent, trusted_authors=trusted)
    target = _target(intent["target"])
    if revocation["target_key"] != target.target_key() or revocation["target_key"] != intent.get("target_key"):
        raise ValueError("revocation target key does not match")
    if revocation["attempt_id"] != intent.get("attempt_id"):
        raise ValueError("revocation attempt ID does not match")
    if revocation["canonical_intent_digest"] != canonical_digest:
        raise ValueError("revocation canonical intent digest does not match")
    if revocation["authenticated"] is not True:
        raise ValueError("revocation is not authenticated")
    _text(revocation, "record_id")
    _text(revocation, "reason")
    _time(revocation)
    _trusted(revocation, trusted)


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
    expected_target: ReviewTarget,
    revocations: Sequence[Mapping[str, Any]] = (),
    reports: Sequence[Mapping[str, Any]] = (),
    trusted_authors: Iterable[str] | None = None,
) -> Mapping[str, Any]:
    trusted = _trust_policy(trusted_authors)
    expected = _expected_target(expected_target)
    validated_intents: dict[str, Mapping[str, Any]] = {}
    events = []
    for intent in intents:
        validate_intent(intent, trusted_authors=trusted)
        _require_expected_target(intent, expected)
        attempt_id = intent["attempt_id"]
        if attempt_id in validated_intents:
            raise ValueError("intent attempt IDs must be unique")
        validated_intents[attempt_id] = intent
        events.append((_time(intent), intent["intent_node_id"], 0, "intent", intent))

    for revocation in revocations:
        events.append((_time(revocation), _text(revocation, "record_id"), 2, "revocation", revocation))

    report_by_id = {report.get("record_id"): report for report in reports}
    for completion in completions:
        events.append((_time(completion), _text(completion, "intent_node_id"), 1, "completion", completion))

    events.sort(key=lambda event: (event[0], event[1], event[2]))
    active: list[tuple[datetime, str, Mapping[str, Any]]] = []
    for _, _, _, event_type, record in events:
        if event_type == "intent":
            active.append((_time(record), record["intent_node_id"], record))
            continue
        if event_type == "revocation":
            if not active:
                raise ValueError("revocation has no current canonical intent")
            current = min(active, key=lambda item: (item[0], item[1]))[2]
            if (
                record.get("target_key") != current.get("target_key")
                or record.get("attempt_id") != current.get("attempt_id")
                or record.get("canonical_intent_digest") != current.get("canonical_digest")
            ):
                raise ValueError("revocation does not target the current canonical intent")
            validate_revocation(record, current, trusted_authors=trusted)
            active = [item for item in active if item[2] is not current]
            continue

        intent = validated_intents.get(record.get("attempt_id"))
        if intent is None or not active:
            raise ValueError("completion is for an unknown or noncanonical intent")
        current = min(active, key=lambda item: (item[0], item[1]))[2]
        validate_completion(
            record,
            intent,
            report=report_by_id.get(record.get("report_id")),
            canonical_intent=current,
            trusted_authors=trusted,
        )

    if not active:
        raise ValueError("no valid non-revoked intent")
    return min(active, key=lambda item: (item[0], item[1]))[2]


def validate_protocol(
    records: Sequence[Mapping[str, Any]], *, expected_target: ReviewTarget,
    trusted_authors: Iterable[str] | None = None
) -> Mapping[str, Any]:
    """Validate a complete append-only log and return its canonical intent."""
    trusted = _trust_policy(trusted_authors)
    expected = _expected_target(expected_target)
    validate_append_only(records)
    grouped = {record_type: [] for record_type in _RECORD_TYPES}
    for record in records:
        record_type = record.get("record_type")
        if record_type not in grouped:
            raise ValueError("unknown review record type")
        grouped[record_type].append(record)
    intents = grouped["intent"]
    for intent in intents:
        validate_intent(intent, trusted_authors=trusted)
    intent_by_attempt = {intent.get("attempt_id"): intent for intent in intents}
    reports = grouped["report"]
    report_by_id = {report.get("record_id"): report for report in reports}
    for report in reports:
        intent = intent_by_attempt.get(report.get("attempt_id"))
        if intent is None:
            raise ValueError("report references an unknown intent")
        _require_expected_target(report, expected)
        validate_report(report, intent, trusted_authors=trusted)
    selected = elect_canonical_attempt(
        intents,
        grouped["completion"],
        expected_target=expected,
        revocations=grouped["revocation"],
        reports=reports,
        trusted_authors=trusted,
    )
    for metadata in grouped["review-metadata"]:
        intent = intent_by_attempt.get(metadata.get("attempt_id"))
        report = report_by_id.get(metadata.get("report_id"))
        if intent is None or report is None:
            raise ValueError("review metadata references a deleted report or unknown intent")
        validate_review_metadata(
            metadata,
            intent,
            report,
            canonical_intent=selected,
            trusted_authors=trusted,
        )
    return selected
