# Copyright (c) Kaden Schutt
"""Validation rules for immutable payloads plus caller-authenticated GitHub facts.

The protocol validates a :class:`GitHubEnvelope` supplied by an authenticated
source; it never authenticates arbitrary mappings or treats an unkeyed digest
as provenance.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, TYPE_CHECKING

from .canonical import canonical_digest, canonical_json, metadata_digest
from .capsule import ReviewCapsule, capsule_coverage
from .models import (
    GitHubEnvelope,
    ReviewTarget,
    ReviewScope,
    ValidationLedgerRow,
    capability_contract_digest,
    derive_protected_review_scope,
    profile_digest,
    protected_exemption_evidence,
    validate_capability_policy,
)
from .validation import validate_ledger_payload_shape, validate_rendered_validation_section

if TYPE_CHECKING:
    from .config import ReviewConfiguration


_RECORD_TYPES = {"intent", "report", "completion", "review-metadata", "revocation"}
_SCHEMA = "agentic-review/v1"
_SCHEMAS = {_SCHEMA}
_COVERAGE_FIELDS = {
    "retrieved_file_count", "expected_file_count", "retrieved_blob_count", "expected_blob_count",
    "retrieved_content_count", "expected_content_count", "coverage_complete",
}
_APP_FIELDS = {"app_id", "installation_id", "repository_id", "credential_attestation_digest"}
_SERVER_FIELDS = {"node_id", "author", "created_at", "payload_digest", "intent_node_id"}
_TARGET_KEYS = {
    "repository", "number", "head_repository", "head_sha", "base_ref", "base_sha", "merge_base_sha"
}
_VALIDATION_FIELDS = {"validation_ledger", "configuration_source_digest"}
_EXEMPTION_FIELDS = {"exemption_ids", "exemption_paths"}
_SCOPE_FIELDS = {"scope"}
_CAPSULE_FIELDS = {"capsule_digest", "capsule_paths", "capsule_target_key"}
_MAX_RENDERED_REPORT_BYTES = 256 * 1024
_MAX_NODE_ID_BYTES = 128


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


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _trust_policy(trusted_authors: Iterable[str] | None) -> frozenset[str]:
    if trusted_authors is None:
        raise ValueError("trusted_authors policy is required")
    if isinstance(trusted_authors, (str, bytes, bytearray)) or not isinstance(trusted_authors, Collection):
        raise ValueError("trusted_authors must be a collection of complete identities")
    policy = frozenset(trusted_authors)
    if not policy or any(not isinstance(author, str) or not author.strip() for author in policy):
        raise ValueError("trusted_authors policy must not be empty")
    return policy


def _target(value: Any) -> ReviewTarget:
    if isinstance(value, ReviewTarget):
        return value
    if not isinstance(value, Mapping) or set(value) != _TARGET_KEYS:
        raise ValueError("record must contain the full ReviewTarget")
    try:
        return ReviewTarget(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ReviewTarget") from exc


def _parse_time(value: Any, name: str) -> datetime:
    value = _text(value, name)
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return timestamp.astimezone(timezone.utc)


def _time(envelope: Mapping[str, Any]) -> datetime:
    return _parse_time(envelope.get("created_at"), "created_at")


def _event_key(envelope: Mapping[str, Any]) -> tuple[datetime, str]:
    return _time(envelope), _text(envelope.get("node_id"), "node_id")


def _require_author(envelope: Mapping[str, Any], trusted: frozenset[str]) -> None:
    if _text(envelope.get("author"), "author") not in trusted:
        raise ValueError("author is not trusted")


def _payload(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = envelope.payload if isinstance(envelope, GitHubEnvelope) else envelope
    if not isinstance(payload, Mapping):
        raise ValueError("GitHub envelope payload must be an object")
    if set(payload) & _SERVER_FIELDS:
        raise ValueError("payload must not assert authenticated server facts")
    _text(payload.get("record_id"), "logical record ID")
    if payload.get("record_type") not in _RECORD_TYPES:
        raise ValueError("unknown review record type")
    if payload.get("schema") not in _SCHEMAS:
        raise ValueError("record schema must be agentic-review/v1")
    return payload


def _coverage(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    present = _COVERAGE_FIELDS & set(payload)
    if not present:
        return None
    if present != _COVERAGE_FIELDS:
        raise ValueError("review record is missing complete coverage evidence")
    values = {field: payload[field] for field in _COVERAGE_FIELDS}
    for prefix in ("file", "blob", "content"):
        retrieved = values[f"retrieved_{prefix}_count"]
        expected = values[f"expected_{prefix}_count"]
        if (
            isinstance(retrieved, bool) or not isinstance(retrieved, int) or retrieved < 0
            or isinstance(expected, bool) or not isinstance(expected, int) or expected < 0
            or retrieved > expected
        ):
            raise ValueError("coverage counts are malformed")
    if not isinstance(values["coverage_complete"], bool):
        raise ValueError("coverage_complete must be a boolean")
    if values["coverage_complete"] and any(
        values[f"retrieved_{prefix}_count"] != values[f"expected_{prefix}_count"]
        for prefix in ("file", "blob", "content")
    ):
        raise ValueError("coverage_complete is inconsistent with coverage counts")
    return values


def _app_provenance(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    present = _APP_FIELDS & set(payload)
    if not present:
        return None
    if present != _APP_FIELDS:
        raise ValueError("App provenance is incomplete")
    for field in ("app_id", "installation_id", "repository_id"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("App provenance identifiers are malformed")
    digest = payload["credential_attestation_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("App provenance attestation is malformed")
    return {field: payload[field] for field in _APP_FIELDS}


def _require_matching_coverage(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    first_coverage = _coverage(first)
    second_coverage = _coverage(second)
    if first_coverage != second_coverage:
        raise ValueError("review records do not carry matching coverage evidence")


def _require_matching_app_provenance(*payloads: Mapping[str, Any]) -> None:
    values = [_app_provenance(payload) for payload in payloads]
    if any(value != values[0] for value in values[1:]):
        raise ValueError("review records do not carry matching App provenance")


def _required(payload: Mapping[str, Any], fields: set[str]) -> set[str]:
    return fields | (_COVERAGE_FIELDS if _COVERAGE_FIELDS & set(payload) else set()) | (
        _APP_FIELDS if _APP_FIELDS & set(payload) else set()
    )


def _validation_fields(payload: Mapping[str, Any]) -> set[str]:
    present = _VALIDATION_FIELDS & set(payload)
    if present and present != _VALIDATION_FIELDS:
        raise ValueError("validation ledger binding is incomplete")
    return _VALIDATION_FIELDS if present else set()


def _scope(payload: Mapping[str, Any]) -> ReviewScope | None:
    if "scope" not in payload:
        return None
    try:
        return ReviewScope.from_mapping(payload["scope"])
    except (TypeError, ValueError) as exc:
        raise ValueError("review scope is malformed") from exc


def _validate_scope_coverage(scope: ReviewScope | None, rows: Sequence[ValidationLedgerRow]) -> None:
    if scope is None:
        raise ValueError("validation report is missing review scope")
    models = {row.model_architecture for row in rows}
    hardware = {item for row in rows for item in row.covered_hardware}
    if not set(scope.model_architectures).issubset(models):
        raise ValueError("declared model architecture is not covered by a selected profile")
    if not set(scope.hardware_architectures).issubset(hardware):
        raise ValueError("declared hardware architecture is not covered by a selected profile")


def _validate_capsule_scope(
    payload: Mapping[str, Any], *, configuration: "ReviewConfiguration", capsule: Any = None,
) -> None:
    if not configuration.is_protected or configuration.source is None or not configuration.source.authenticated:
        raise ValueError("protected configuration is required for capsule scope validation")
    if not isinstance(capsule, ReviewCapsule) or not capsule.complete:
        raise ValueError("complete authenticated review capsule is required for ledger history")
    target = _target(payload.get("target")) if "target" in payload else capsule.target
    capsule_digest = payload.get("capsule_digest")
    capsule_target_key = payload.get("capsule_target_key")
    if (
        not isinstance(capsule_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", capsule_digest)
        or capsule_target_key != target.target_key()
    ):
        raise ValueError("report capsule binding is malformed")
    if capsule.target != target or capsule.digest != capsule_digest:
        raise ValueError("report capsule binding does not match authenticated capsule")
    if _coverage(payload) != capsule_coverage(capsule):
        raise ValueError("report coverage does not match authenticated capsule")
    paths = payload.get("capsule_paths")
    manifest_paths = tuple(entry.path for entry in capsule.manifest)
    if not isinstance(paths, (list, tuple)) or tuple(paths) != manifest_paths:
        raise ValueError("report capsule paths do not match authenticated capsule")
    expected = derive_protected_review_scope(capsule, configuration.capabilities)
    if _scope(payload) != expected:
        raise ValueError("report scope does not match protected capsule scope")


def _exemption_fields(payload: Mapping[str, Any]) -> set[str]:
    present = _EXEMPTION_FIELDS & set(payload)
    if present and present != _EXEMPTION_FIELDS:
        raise ValueError("exemption evidence is incomplete")
    return _EXEMPTION_FIELDS if present else set()


def validate_validation_ledger(
    payload: Mapping[str, Any], *, configuration: "ReviewConfiguration | None" = None,
    capsule: Any = None,
) -> None:
    """Validate a report ledger against the live authenticated policy binding.

    Configuration changes intentionally invalidate historical ledger-bearing
    completions; discovery must then requeue the pull request for review.
    """
    fields = _validation_fields(payload)
    if not fields:
        return
    ledger = payload.get("validation_ledger")
    source_digest = payload.get("configuration_source_digest")
    if not isinstance(source_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_digest):
        raise ValueError("configuration source digest is malformed")
    rows_raw = validate_ledger_payload_shape(ledger)
    rows: list[ValidationLedgerRow] = []
    for item in rows_raw:
        try:
            row = ValidationLedgerRow.from_mapping(item)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("validation ledger row is malformed") from exc
        rows.append(row)
    if configuration is None:
        raise ValueError("protected configuration is required for a validation ledger")
    if not configuration.is_protected or configuration.source is None or not configuration.source.authenticated:
        raise ValueError("validation ledger requires an authenticated protected configuration")
    if source_digest != configuration.source.config_digest:
        raise ValueError("configuration source digest does not match authenticated configuration")
    if "target" in payload and configuration.source.repository != _target(payload.get("target")).repository:
        raise ValueError("configuration source repository does not match report target")
    try:
        validate_capability_policy(configuration.capabilities)
        profiles = {item["id"]: item for item in configuration.capabilities["profiles"]}
        capabilities = {item["id"]: item for item in configuration.capabilities["capabilities"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("protected validation policy is malformed") from exc
    for row in rows:
        profile_mapping = profiles.get(row.profile_snapshot.get("id"))
        if profile_mapping is None or canonical_json(row.profile_snapshot) != canonical_json(profile_mapping):
            raise ValueError("validation ledger profile is not from protected policy")
        capability = capabilities.get(row.capability_id)
        if capability is None or row.profile_digest != profile_digest(profile_mapping):
            raise ValueError("validation ledger profile digest does not match protected policy")
        if row.contract_digest != capability_contract_digest(capability):
            raise ValueError("validation ledger capability digest does not match protected policy")
        if row.coverage_kind != "representative":
            raise ValueError("validation ledger coverage kind is not protected")
    _validate_scope_coverage(_scope(payload), rows)
    _validate_capsule_scope(payload, configuration=configuration, capsule=capsule)


def _validate_exemption_binding(
    payload: Mapping[str, Any], *, configuration: "ReviewConfiguration | None", capsule: Any = None,
) -> bool:
    fields = _exemption_fields(payload)
    ledger = payload.get("validation_ledger")
    if not fields:
        if isinstance(ledger, (list, tuple)) and not ledger:
            raise ValueError("empty validation ledger lacks protected exemption evidence")
        return False
    if not isinstance(ledger, (list, tuple)) or ledger:
        raise ValueError("exemption evidence cannot accompany validation rows")
    if configuration is None or not configuration.is_protected or configuration.source is None:
        raise ValueError("exemption evidence requires protected configuration")
    source_digest = payload.get("configuration_source_digest")
    if source_digest != configuration.source.config_digest:
        raise ValueError("configuration source digest does not match authenticated configuration")
    target = _target(payload.get("target"))
    if configuration.source.repository != target.repository:
        raise ValueError("exemption source repository does not match report target")
    paths = payload.get("exemption_paths")
    if not isinstance(payload.get("exemption_ids"), (list, tuple)) or not isinstance(paths, (list, tuple)):
        raise ValueError("exemption evidence is malformed")
    try:
        expected = protected_exemption_evidence(configuration.capabilities["exemptions"], paths)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("protected exemption policy is malformed") from exc
    if expected != (tuple(payload["exemption_ids"]), tuple(paths)):
        raise ValueError("exemption evidence does not match protected policy")
    _validate_capsule_scope(payload, configuration=configuration, capsule=capsule)
    return True


def _matching_schema(*payloads: Mapping[str, Any]) -> str:
    schemas = {payload.get("schema") for payload in payloads}
    schema = next(iter(schemas), None)
    if len(schemas) != 1 or not isinstance(schema, str):
        raise ValueError("review records use incompatible schema versions")
    return schema


def _validate_envelope(envelope: Mapping[str, Any], trusted: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(envelope, GitHubEnvelope):
        raise ValueError("protocol requires a typed GitHubEnvelope from an authenticated source")
    payload = _payload(envelope.payload)
    _text(envelope.node_id, "node_id")
    if len(envelope.node_id.encode("utf-8")) > _MAX_NODE_ID_BYTES:
        raise ValueError("node_id exceeds the maximum UTF-8 length")
    _require_author(envelope, trusted)
    if _parse_time(envelope.updated_at, "updated_at") != _time(envelope):
        raise ValueError("edited protocol records are not allowed: updated_at differs from created_at")
    return payload


def _payload_digest(envelope: Mapping[str, Any]) -> str:
    """Return an integrity digest; this does not authenticate the envelope."""
    return canonical_digest(_plain(envelope.get("payload")))


def _expected_target(value: ReviewTarget) -> ReviewTarget:
    if not isinstance(value, ReviewTarget):
        raise ValueError("expected_target must be a ReviewTarget")
    return value


def _require_target(payload: Mapping[str, Any], expected: ReviewTarget) -> ReviewTarget:
    target = _target(payload.get("target"))
    if target != expected or payload.get("target_key") != expected.target_key():
        raise ValueError("record target does not match expected target")
    return target


def _same_binding(payload: Mapping[str, Any], intent: Mapping[str, Any]) -> ReviewTarget:
    target = _target(payload.get("target"))
    if target != _target(intent.get("target")):
        raise ValueError("record target does not match intent")
    for field in ("target_key", "attempt_id", "intent_record_id"):
        if payload.get(field) != intent.get(field if field != "intent_record_id" else "record_id"):
            raise ValueError(f"record {field} does not match intent")
    if payload.get("head_sha") not in (None, target.head_sha):
        raise ValueError("record head SHA does not match target")
    return target


def _canonical_binding(
    payload: Mapping[str, Any], canonical_intent: Mapping[str, Any], digest_field: str
) -> None:
    target = _target(payload.get("target"))
    canonical_target = _target(canonical_intent.get("target"))
    if (
        target != canonical_target
        or payload.get("target_key") != canonical_intent.get("target_key")
        or payload.get("attempt_id") != canonical_intent.get("attempt_id")
        or payload.get("intent_record_id") != canonical_intent.get("record_id")
        or payload.get("canonical_intent_node_id") != canonical_intent.get("_node_id")
        or payload.get("head_sha") != canonical_target.head_sha
        or payload.get(digest_field) != canonical_intent.get("canonical_digest")
    ):
        raise ValueError("record is not bound to the canonical intent")


def _intent_digest(payload: Mapping[str, Any]) -> str:
    return canonical_digest({key: _plain(value) for key, value in payload.items() if key != "canonical_digest"})


def _before(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return _event_key(first) < _event_key(second)


def validate_intent(
    envelope: Mapping[str, Any], *, trusted_authors: Iterable[str] | None = None
) -> str:
    trusted = _trust_policy(trusted_authors)
    payload = _validate_envelope(envelope, trusted)
    required = {"schema", "record_type", "record_id", "target", "target_key", "attempt_id", "canonical_digest"}
    if set(payload) != _required(payload, required) or payload["record_type"] != "intent":
        raise ValueError("invalid intent payload")
    _app_provenance(payload)
    target = _target(payload["target"])
    if payload["target_key"] != target.target_key():
        raise ValueError("intent target_key does not match target")
    _text(payload.get("attempt_id"), "attempt_id")
    if payload["canonical_digest"] != _intent_digest(payload):
        raise ValueError("intent canonical digest does not match payload")
    return payload["canonical_digest"]


def validate_report(
    envelope: Mapping[str, Any],
    intent_envelope: Mapping[str, Any],
    *,
    canonical_intent: Mapping[str, Any],
    trusted_authors: Iterable[str] | None = None,
    configuration: "ReviewConfiguration | None" = None,
    capsule: Any = None,
) -> str:
    trusted = _trust_policy(trusted_authors)
    payload = _validate_envelope(envelope, trusted)
    intent = _payload(intent_envelope)
    _matching_schema(payload, intent)
    _app_provenance(payload)
    validate_intent(intent_envelope, trusted_authors=trusted)
    required = {
        "schema", "record_type", "record_id", "target", "target_key", "attempt_id", "intent_record_id", "head_sha",
        "canonical_intent_node_id", "canonical_intent_digest", "report_body", "report_body_sha256",
    }
    if set(payload) != (
        _required(payload, required) | _validation_fields(payload) | _exemption_fields(payload)
        | ({"scope"} if "scope" in payload else set())
        | (_CAPSULE_FIELDS if _validation_fields(payload) else set())
    ) or payload["record_type"] != "report":
        raise ValueError("invalid report payload")
    if "scope" in payload and not _validation_fields(payload):
        raise ValueError("scope-bearing reports require an authenticated capsule")
    _coverage(payload)
    validate_validation_ledger(payload, configuration=configuration, capsule=capsule)
    exempt = _validate_exemption_binding(payload, configuration=configuration, capsule=capsule)
    target = _same_binding(payload, intent)
    if payload["head_sha"] != target.head_sha:
        raise ValueError("report head SHA does not match target")
    validate_intent(canonical_intent, trusted_authors=trusted)
    canonical_payload = dict(_payload(canonical_intent), _node_id=canonical_intent["node_id"])
    _canonical_binding(payload, canonical_payload, "canonical_intent_digest")
    if not _before(intent_envelope, envelope):
        raise ValueError("report was published before its intent")
    body = payload["report_body"]
    if not isinstance(body, str):
        raise ValueError("report body must be text")
    if _validation_fields(payload) and body != body.strip():
        raise ValueError("ledger-bearing report body must not have leading or trailing whitespace")
    if len(body.encode("utf-8")) > _MAX_RENDERED_REPORT_BYTES:
        raise ValueError("rendered report exceeds 256 KiB")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if payload["report_body_sha256"] not in {digest, "sha256:" + digest}:
        raise ValueError("report body digest does not match body")
    if _validation_fields(payload):
        if "scope" not in payload:
            raise ValueError("validation report is missing review scope")
        validate_rendered_validation_section(
            body, payload["validation_ledger"], exempt=exempt, scope=_scope(payload),
        )
    return _payload_digest(envelope)


def validate_review_metadata(
    envelope: Mapping[str, Any],
    intent_envelope: Mapping[str, Any],
    report_envelope: Mapping[str, Any],
    *,
    canonical_intent: Mapping[str, Any],
    trusted_authors: Iterable[str] | None = None,
    configuration: "ReviewConfiguration | None" = None,
    capsule: Any = None,
) -> str:
    trusted = _trust_policy(trusted_authors)
    payload = _validate_envelope(envelope, trusted)
    intent = _payload(intent_envelope)
    report = _payload(report_envelope)
    _matching_schema(payload, intent, report)
    _app_provenance(payload)
    validate_intent(intent_envelope, trusted_authors=trusted)
    required = {
        "schema", "record_type", "record_id", "target", "target_key", "attempt_id", "intent_record_id", "head_sha",
        "report_record_id", "report_node_id", "report_digest", "report_body_sha256",
        "canonical_intent_digest", "canonical_intent_node_id", "metadata_digest",
    }
    if set(payload) != _required(payload, required) or payload["record_type"] != "review-metadata":
        raise ValueError("invalid review metadata payload")
    _coverage(payload)
    target = _same_binding(payload, intent)
    if payload["head_sha"] != target.head_sha:
        raise ValueError("review metadata head SHA does not match target")
    validate_report(
        report_envelope,
        intent_envelope,
        canonical_intent=canonical_intent,
        trusted_authors=trusted,
        configuration=configuration,
        capsule=capsule,
    )
    if not _before(intent_envelope, envelope) or not _before(report_envelope, envelope):
        raise ValueError("review metadata was published before its dependency")
    if payload["report_record_id"] != report.get("record_id"):
        raise ValueError("review metadata references the wrong report")
    if payload["report_node_id"] != report_envelope.get("node_id"):
        raise ValueError("review metadata report node binding does not match")
    if payload["report_digest"] != _payload_digest(report_envelope):
        raise ValueError("review metadata report digest does not match")
    if payload["report_body_sha256"] != report.get("report_body_sha256"):
        raise ValueError("review metadata report body digest does not match")
    _require_matching_coverage(report, payload)
    _require_matching_app_provenance(intent, report, payload)
    canonical_payload = _payload(canonical_intent)
    validate_intent(canonical_intent, trusted_authors=trusted)
    canonical_payload = dict(canonical_payload, _node_id=canonical_intent["node_id"])
    _canonical_binding(payload, canonical_payload, "canonical_intent_digest")
    digest = metadata_digest(payload)
    if payload["metadata_digest"] != digest:
        raise ValueError("metadata digest does not match payload")
    return payload["metadata_digest"]


def validate_completion(
    envelope: Mapping[str, Any],
    intent_envelope: Mapping[str, Any],
    report_envelope: Mapping[str, Any] | None,
    metadata_envelope: Mapping[str, Any] | None,
    *,
    canonical_intent: Mapping[str, Any],
    trusted_authors: Iterable[str] | None = None,
    configuration: "ReviewConfiguration | None" = None,
    capsule: Any = None,
) -> None:
    trusted = _trust_policy(trusted_authors)
    payload = _validate_envelope(envelope, trusted)
    intent = _payload(intent_envelope)
    required = {
        "schema", "record_type", "record_id", "target", "target_key", "attempt_id", "intent_record_id", "head_sha",
        "canonical_intent_digest", "canonical_intent_node_id", "report_record_id", "report_node_id",
        "report_digest", "metadata_record_id", "metadata_digest",
    }
    if set(payload) != _required(payload, required) or payload["record_type"] != "completion":
        raise ValueError("invalid completion payload")
    _coverage(payload)
    if report_envelope is None:
        raise ValueError("completion references a missing report")
    if metadata_envelope is None:
        raise ValueError("completion references a missing review metadata record")
    report = _payload(report_envelope)
    metadata = _payload(metadata_envelope)
    _matching_schema(payload, intent, report, metadata)
    _app_provenance(payload)
    validate_intent(intent_envelope, trusted_authors=trusted)
    canonical_payload = _payload(canonical_intent)
    validate_intent(canonical_intent, trusted_authors=trusted)
    canonical_payload = dict(canonical_payload, _node_id=canonical_intent["node_id"])
    _canonical_binding(payload, canonical_payload, "canonical_intent_digest")
    target = _same_binding(payload, intent)
    if payload["head_sha"] != target.head_sha:
        raise ValueError("completion head SHA does not match target")
    if not _before(intent_envelope, envelope) or not _before(report_envelope, envelope) or not _before(metadata_envelope, envelope):
        raise ValueError("completion was published before its dependency")
    validate_review_metadata(
        metadata_envelope,
        intent_envelope,
        report_envelope,
        canonical_intent=canonical_intent,
        trusted_authors=trusted,
        configuration=configuration,
        capsule=capsule,
    )
    if payload["report_record_id"] != report.get("record_id") or payload["report_node_id"] != report_envelope.get("node_id"):
        raise ValueError("completion report binding does not match")
    if payload["report_digest"] != _payload_digest(report_envelope):
        raise ValueError("completion report digest does not match")
    if payload["metadata_record_id"] != metadata.get("record_id"):
        raise ValueError("completion metadata binding does not match")
    if payload["metadata_digest"] != metadata.get("metadata_digest"):
        raise ValueError("completion metadata digest does not match")
    _require_matching_coverage(report, payload)
    _require_matching_coverage(metadata, payload)
    _require_matching_app_provenance(intent, report, metadata, payload)


def validate_revocation(
    envelope: Mapping[str, Any],
    intent_envelope: Mapping[str, Any],
    *,
    trusted_authors: Iterable[str] | None = None,
) -> None:
    trusted = _trust_policy(trusted_authors)
    payload = _validate_envelope(envelope, trusted)
    intent = _payload(intent_envelope)
    validate_intent(intent_envelope, trusted_authors=trusted)
    required = {"schema", "record_type", "record_id", "target_key", "attempt_id", "canonical_intent_digest", "reason"}
    if set(payload) != required or payload["record_type"] != "revocation":
        raise ValueError("invalid revocation payload")
    if payload["target_key"] != intent.get("target_key") or payload["attempt_id"] != intent.get("attempt_id"):
        raise ValueError("revocation target does not match intent")
    if payload["canonical_intent_digest"] != intent.get("canonical_digest"):
        raise ValueError("revocation canonical intent digest does not match")
    _text(payload.get("reason"), "reason")
    if not _before(intent_envelope, envelope):
        raise ValueError("revocation was published before its intent")


def _record_id(envelope: Mapping[str, Any]) -> str:
    return _text(_payload(envelope).get("record_id"), "logical record ID")


def _unique_records(records: Sequence[Mapping[str, Any]], trusted: frozenset[str] | None = None) -> None:
    logical: set[str] = set()
    nodes: set[str] = set()
    for envelope in records:
        if not isinstance(envelope, GitHubEnvelope):
            raise ValueError("append-only history requires typed GitHubEnvelope values")
        payload = envelope.payload
        if _parse_time(envelope.updated_at, "updated_at") != _time(envelope):
            raise ValueError("edited protocol records are not allowed: updated_at differs from created_at")
        logical_id = _record_id(envelope)
        node_id = _text(envelope.get("node_id"), "node_id")
        if logical_id in logical:
            raise ValueError("duplicate logical record ID")
        if node_id in nodes:
            raise ValueError("duplicate authenticated node ID")
        logical.add(logical_id)
        nodes.add(node_id)


def validate_append_only(records: Sequence[Mapping[str, Any]], previous: Sequence[Mapping[str, Any]] = ()) -> None:
    _unique_records(records)
    _unique_records(previous)
    old = {_record_id(record): canonical_json(_plain(record)) for record in previous}
    current = {_record_id(record): canonical_json(_plain(record)) for record in records}
    if not set(old).issubset(current):
        raise ValueError("append-only log deleted a record")
    for logical_id, encoded in old.items():
        if current[logical_id] != encoded:
            raise ValueError("append-only log altered an existing record")


def elect_canonical_attempt(
    intents: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
    *,
    expected_target: ReviewTarget,
    revocations: Sequence[Mapping[str, Any]] = (),
    reports: Sequence[Mapping[str, Any]] = (),
    review_metadata: Sequence[Mapping[str, Any]] = (),
    trusted_authors: Iterable[str] | None = None,
    configuration: "ReviewConfiguration | None" = None,
    capsule: Any = None,
) -> Mapping[str, Any]:
    trusted = _trust_policy(trusted_authors)
    expected = _expected_target(expected_target)
    records = [*intents, *reports, *review_metadata, *completions, *revocations]
    _unique_records(records)
    intent_by_id: dict[str, Mapping[str, Any]] = {}
    attempt_ids: set[str] = set()
    events = []
    for envelope in intents:
        payload = _payload(envelope)
        validate_intent(envelope, trusted_authors=trusted)
        _require_target(payload, expected)
        logical_id = _record_id(envelope)
        if logical_id in intent_by_id:
            raise ValueError("duplicate intent logical record ID")
        attempt_id = _text(payload.get("attempt_id"), "attempt_id")
        if attempt_id in attempt_ids:
            raise ValueError("duplicate intent attempt ID")
        attempt_ids.add(attempt_id)
        intent_by_id[logical_id] = envelope
        events.append((_event_key(envelope), 0, "intent", envelope))
    for envelope in reports:
        events.append((_event_key(envelope), 1, "report", envelope))
    for envelope in review_metadata:
        events.append((_event_key(envelope), 2, "review-metadata", envelope))
    for envelope in completions:
        events.append((_event_key(envelope), 3, "completion", envelope))
    for envelope in revocations:
        events.append((_event_key(envelope), 4, "revocation", envelope))
    events.sort(key=lambda event: (event[0][0], event[0][1], event[1]))
    active: list[Mapping[str, Any]] = []
    published_reports: dict[str, Mapping[str, Any]] = {}
    published_metadata: dict[str, Mapping[str, Any]] = {}
    for _, _, event_type, envelope in events:
        if event_type == "intent":
            active.append(envelope)
            continue
        payload = _payload(envelope)
        logical_intent_id = payload.get("intent_record_id")
        intent = intent_by_id.get(logical_intent_id)
        if event_type == "revocation":
            if not active:
                raise ValueError("revocation has no current canonical intent")
            current = min(active, key=_event_key)
            current_payload = _payload(current)
            if payload.get("target_key") != current_payload.get("target_key") or payload.get("attempt_id") != current_payload.get("attempt_id"):
                raise ValueError("revocation does not target the current canonical intent")
            validate_revocation(envelope, current, trusted_authors=trusted)
            active = [item for item in active if item is not current]
            continue
        if intent is None or not active:
            raise ValueError(f"{event_type} is before its intent or references an unknown attempt")
        _require_target(payload, expected)
        current = min(active, key=_event_key)
        current_payload = _payload(current)
        if payload.get("target_key") != current_payload.get("target_key") or payload.get("attempt_id") != current_payload.get("attempt_id"):
            raise ValueError(f"{event_type} does not target the current canonical intent")
        if event_type == "report":
            validate_report(
                envelope, intent, canonical_intent=current, trusted_authors=trusted,
                configuration=configuration, capsule=capsule,
            )
            published_reports[_record_id(envelope)] = envelope
        elif event_type == "review-metadata":
            report = published_reports.get(payload.get("report_record_id"))
            if report is None:
                raise ValueError("review metadata is before its referenced report")
            validate_review_metadata(
                envelope, intent, report, canonical_intent=current, trusted_authors=trusted,
                configuration=configuration, capsule=capsule,
            )
            published_metadata[_record_id(envelope)] = envelope
        else:
            report = published_reports.get(payload.get("report_record_id"))
            metadata = published_metadata.get(payload.get("metadata_record_id"))
            validate_completion(
                envelope,
                intent,
                report,
                metadata,
                canonical_intent=current,
                trusted_authors=trusted,
                configuration=configuration,
                capsule=capsule,
            )
    if not active:
        raise ValueError("no valid non-revoked intent")
    return min(active, key=_event_key)


def validate_protocol(
    records: Sequence[Mapping[str, Any]], *, expected_target: ReviewTarget,
    trusted_authors: Iterable[str] | None = None,
    configuration: "ReviewConfiguration | None" = None,
    capsule: Any = None,
) -> Mapping[str, Any]:
    trusted = _trust_policy(trusted_authors)
    expected = _expected_target(expected_target)
    validate_append_only(records)
    grouped = {record_type: [] for record_type in _RECORD_TYPES}
    for envelope in records:
        payload = _payload(envelope)
        record_type = payload["record_type"]
        if record_type not in grouped:
            raise ValueError("unknown review record type")
        grouped[record_type].append(envelope)
    return elect_canonical_attempt(
        grouped["intent"],
        grouped["completion"],
        expected_target=expected,
        reports=grouped["report"],
        review_metadata=grouped["review-metadata"],
        revocations=grouped["revocation"],
        trusted_authors=trusted,
        configuration=configuration,
        capsule=capsule,
    )
