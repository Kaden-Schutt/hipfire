# Copyright (c) Kaden Schutt
"""Immutable policy and review contracts for the agentic review workflow."""

from collections.abc import Mapping
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

from .canonical import DEFAULT_MAX_BYTES, canonical_digest, canonical_json
from .validation import (
    MAX_VALIDATION_FIELD_BYTES,
    MAX_VALIDATION_RATIONALE_BYTES,
    MAX_VALIDATION_ROWS,
    validate_ledger_payload_shape,
    validate_ledger_row_mapping,
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_RAW_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VERDICTS = frozenset({"clean", "changes-requested", "incomplete"})
ACTIONABLE_SEVERITIES = frozenset({"error"})
NONBLOCKING_SEVERITIES = frozenset({"warning", "info"})
FINDING_SEVERITIES = ACTIONABLE_SEVERITIES | NONBLOCKING_SEVERITIES
_CAPABILITY_KEYS = frozenset(
    {
        "id",
        "parameters",
        "contract_digest",
        "allowed_suite_revisions",
        "required_checks",
        "eligible_hardware",
        "artifacts",
        "pass_criteria",
    }
)
_CAPABILITY_ROOT_KEYS = frozenset({"schema", "version", "capabilities", "profiles", "fixtures", "exemptions"})
_FIXTURE_KEYS = frozenset({
    "fixture_id", "model_architecture", "artifact_identity", "source_identity",
    "suite_revision", "digest_semantics", "fixture_digest",
})
_PROFILE_KEYS = frozenset({
    "id", "capability_id", "model_architecture", "fixture_id", "fixture_digest",
    "representative_hardware", "covered_hardware",
})
_EXEMPTION_KEYS = frozenset({"id", "path_globs"})
_PROVIDER_KEYS = frozenset(
    {
        "id",
        "adapter_id",
        "adapter_version",
        "endpoint",
        "model",
        "api_key_env",
        "max_requests",
        "request_deadline_seconds",
        "max_capsule_bytes",
        "max_response_bytes",
        "max_tokens",
        "max_cost_usd",
    }
)
_PROVIDER_ROOT_KEYS = frozenset({"schema", "version", "providers"})
_TRUSTED_ROOT_KEYS = frozenset({"schema", "version", "apps"})
_TRUSTED_APP_KEYS = frozenset(
    {"app_id", "login", "installation_id", "repository_id", "credential_attestation_digest"}
)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256 followed by 64 lowercase hex characters")


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{name} has unexpected or missing keys")


def _require_string_list(name: str, value: Any, *, nonempty: bool = True) -> None:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")


def _normalize_rationale(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized.encode("utf-8")) > MAX_VALIDATION_RATIONALE_BYTES:
        raise ValueError("rationale exceeds the maximum length")
    return normalized


@dataclass(frozen=True)
class ReviewTarget:
    repository: str
    number: int
    head_repository: str
    head_sha: str
    base_ref: str
    base_sha: str
    merge_base_sha: str

    def __post_init__(self) -> None:
        _require_text("repository", self.repository)
        _require_positive_integer("number", self.number)
        _require_text("head_repository", self.head_repository)
        _require_text("head_sha", self.head_sha)
        _require_text("base_ref", self.base_ref)
        _require_text("base_sha", self.base_sha)
        _require_text("merge_base_sha", self.merge_base_sha)

    def target_key(self) -> str:
        canonical = {
            "base_ref": self.base_ref,
            "base_sha": self.base_sha,
            "head_repository": self.head_repository,
            "head_sha": self.head_sha,
            "merge_base_sha": self.merge_base_sha,
            "number": self.number,
            "repository": self.repository,
        }
        encoded = canonical_json(canonical)
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GitHubEnvelope(Mapping[str, Any]):
    """Server-supplied GitHub facts paired with an immutable protocol payload.

    Construction is a typed data contract only.  This class does not prove
    provenance; the fixed-endpoint GitHub client in Task 3 must supply and
    authenticate these fields before protocol validators consume the value.
    """

    payload: Mapping[str, Any]
    node_id: str
    author: str
    created_at: str
    updated_at: str
    author_type: str = "User"

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_payload(self.payload))
        _require_text("node_id", self.node_id)
        _require_text("author", self.author)
        _require_text("created_at", self.created_at)
        _require_text("updated_at", self.updated_at)
        if self.author_type not in {"User", "Bot", "Organization"}:
            raise ValueError("author_type is not supported")

    def __getitem__(self, key: str) -> Any:
        if key not in {"payload", "node_id", "author", "created_at", "updated_at", "author_type"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("payload", "node_id", "author", "created_at", "updated_at", "author_type"))

    def __len__(self) -> int:
        return 6


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, ReviewTarget):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("payload mapping keys must be strings")
        return MappingProxyType({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise ValueError("payload must not contain sets")
    if value is not None and not isinstance(value, (bool, int, float, str)):
        raise ValueError("payload contains a mutable or unsupported value")
    return value


@dataclass(frozen=True)
class AttemptIntentConfig:
    target: ReviewTarget
    attempt_id: str
    capability_id: str
    suite_revision: str
    provider_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.target, ReviewTarget):
            raise ValueError("target must be a ReviewTarget")
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("capability_id", self.capability_id),
            ("suite_revision", self.suite_revision),
            ("provider_id", self.provider_id),
        ):
            _require_text(name, value)


@dataclass(frozen=True)
class IntentPayload:
    """Exact immutable model for the protocol's pre-publication intent payload."""

    schema: str
    record_type: str
    record_id: str
    target: ReviewTarget
    target_key: str
    attempt_id: str
    canonical_digest: str
    app_id: int | None = None
    installation_id: int | None = None
    repository_id: int | None = None
    credential_attestation_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema != "agentic-review/v1":
            raise ValueError("intent payload schema must be agentic-review/v1")
        if self.record_type != "intent":
            raise ValueError("intent payload record_type must be intent")
        _require_text("record_id", self.record_id)
        _require_text("attempt_id", self.attempt_id)
        if not isinstance(self.target, ReviewTarget):
            raise ValueError("target must be a ReviewTarget")
        if self.target_key != self.target.target_key():
            raise ValueError("intent payload target_key does not match target")
        app_values = (self.app_id, self.installation_id, self.repository_id, self.credential_attestation_digest)
        if any(value is not None for value in app_values):
            if (
                isinstance(self.app_id, bool) or not isinstance(self.app_id, int) or self.app_id <= 0
                or isinstance(self.installation_id, bool) or not isinstance(self.installation_id, int) or self.installation_id <= 0
                or isinstance(self.repository_id, bool) or not isinstance(self.repository_id, int) or self.repository_id <= 0
                or not isinstance(self.credential_attestation_digest, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.credential_attestation_digest)
            ):
                raise ValueError("intent App provenance is incomplete or malformed")
        if _RAW_SHA256_RE.fullmatch(self.canonical_digest) is None or self.canonical_digest != canonical_digest(
            {key: value for key, value in self.to_mapping().items() if key != "canonical_digest"}
        ):
            raise ValueError("canonical_digest must exactly match the intent payload")

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "target": self.target,
            "target_key": self.target_key,
            "attempt_id": self.attempt_id,
            "canonical_digest": self.canonical_digest,
        }
        if self.app_id is not None:
            result.update({
                "app_id": self.app_id,
                "installation_id": self.installation_id,
                "repository_id": self.repository_id,
                "credential_attestation_digest": self.credential_attestation_digest,
            })
        return result

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IntentPayload":
        expected = {"schema", "record_type", "record_id", "target", "target_key", "attempt_id", "canonical_digest"}
        app_fields = {"app_id", "installation_id", "repository_id", "credential_attestation_digest"}
        if not isinstance(payload, Mapping) or set(payload) not in (expected, expected | app_fields):
            raise ValueError("invalid intent payload shape")
        target = payload["target"]
        target_keys = {
            "repository", "number", "head_repository", "head_sha", "base_ref", "base_sha", "merge_base_sha"
        }
        if not isinstance(target, ReviewTarget):
            if not isinstance(target, Mapping) or set(target) != target_keys:
                raise ValueError("invalid intent payload target shape")
            target = ReviewTarget(**target)
        values = dict(payload)
        values["target"] = target
        return cls(**values)


@dataclass(frozen=True)
class Finding:
    path: str
    range: tuple[int, int]
    severity: str
    message: str

    def __post_init__(self) -> None:
        _require_text("path", self.path)
        if (
            not isinstance(self.range, tuple)
            or len(self.range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.range)
            or self.range[0] > self.range[1]
        ):
            raise ValueError("range must be a tuple of two positive integers")
        _require_text("severity", self.severity)
        if self.severity not in FINDING_SEVERITIES:
            raise ValueError("severity is not supported")
        _require_text("message", self.message)

@dataclass(frozen=True)
class HardwareValidationTriage:
    """Diff-informed triage of model families needing hardware validation."""

    impacted_model_families: tuple[str, ...]
    impacted_hardware: tuple[str, ...]
    coverage_decision: str
    rationale: str

    def __post_init__(self) -> None:
        _require_string_list("impacted_model_families", self.impacted_model_families, nonempty=False)
        _require_string_list("impacted_hardware", self.impacted_hardware, nonempty=False)
        if self.coverage_decision not in {"all-impacted", "representative-only", "none"}:
            raise ValueError("coverage_decision must be one of: all-impacted, representative-only, none")
        _require_text("rationale", self.rationale)
        if tuple(sorted(self.impacted_model_families)) != self.impacted_model_families:
            raise ValueError("impacted_model_families must be lexicographically ordered")
        if tuple(sorted(self.impacted_hardware)) != self.impacted_hardware:
            raise ValueError("impacted_hardware must be lexicographically ordered")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "impacted_model_families": list(self.impacted_model_families),
            "impacted_hardware": list(self.impacted_hardware),
            "coverage_decision": self.coverage_decision,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HardwareValidationTriage":
        if not isinstance(value, Mapping):
            raise ValueError("hardware_validation_triage must be an object")
        required = {"impacted_model_families", "impacted_hardware", "coverage_decision", "rationale"}
        if set(value) != required:
            raise ValueError("hardware_validation_triage has unexpected or missing keys")
        model_families = value["impacted_model_families"]
        hardware = value["impacted_hardware"]
        if isinstance(model_families, list):
            model_families = tuple(model_families)
        if isinstance(hardware, list):
            hardware = tuple(hardware)
        return cls(model_families, hardware, value["coverage_decision"], value["rationale"])



@dataclass(frozen=True)
class ReviewScope:
    """The exact model/hardware scope selected by the review model."""

    model_architectures: tuple[str, ...]
    hardware_architectures: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("model_architectures", self.model_architectures),
            ("hardware_architectures", self.hardware_architectures),
        ):
            _require_string_list(name, value, nonempty=False)
            if tuple(sorted(value)) != value:
                raise ValueError(f"{name} must be lexicographically ordered")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_architectures": list(self.model_architectures),
            "hardware_architectures": list(self.hardware_architectures),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewScope":
        if not isinstance(value, Mapping) or set(value) != {"model_architectures", "hardware_architectures"}:
            raise ValueError("review scope has unexpected or missing keys")
        model_architectures = value["model_architectures"]
        hardware_architectures = value["hardware_architectures"]
        if isinstance(model_architectures, list):
            model_architectures = tuple(model_architectures)
        if isinstance(hardware_architectures, list):
            hardware_architectures = tuple(hardware_architectures)
        return cls(model_architectures, hardware_architectures)


def fixture_descriptor_digest(fixture: Mapping[str, Any]) -> str:
    """Digest the complete protected fixture descriptor, excluding its digest."""
    if not isinstance(fixture, Mapping) or frozenset(fixture) != _FIXTURE_KEYS:
        raise ValueError("fixture has unexpected or missing keys")
    descriptor = {key: fixture[key] for key in fixture if key != "fixture_digest"}
    return "sha256:" + hashlib.sha256(canonical_json(descriptor)).hexdigest()


@dataclass(frozen=True)
class ValidationProfile:
    """Protected validation identity, not provenance for an artifact file.

    ``fixture_digest`` is protected descriptor/artifact provenance only when
    the authenticated policy has a protected digest source.  It must not be
    interpreted as proof that fixture bytes were retrieved or executed when
    no such source is available.
    """

    id: str
    capability_id: str
    model_architecture: str
    fixture_id: str
    fixture_digest: str
    representative_hardware: str
    covered_hardware: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("id", self.id),
            ("capability_id", self.capability_id),
            ("model_architecture", self.model_architecture),
            ("fixture_id", self.fixture_id),
            ("representative_hardware", self.representative_hardware),
        ):
            _require_text(name, value)
        _require_digest("fixture_digest", self.fixture_digest)
        _require_string_list("covered_hardware", self.covered_hardware)
        if tuple(sorted(self.covered_hardware)) != self.covered_hardware:
            raise ValueError("covered_hardware must be lexicographically ordered")
        if self.representative_hardware not in self.covered_hardware:
            raise ValueError("representative_hardware must be covered")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capability_id": self.capability_id,
            "model_architecture": self.model_architecture,
            "fixture_id": self.fixture_id,
            "fixture_digest": self.fixture_digest,
            "representative_hardware": self.representative_hardware,
            "covered_hardware": list(self.covered_hardware),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValidationProfile":
        if not isinstance(value, Mapping):
            raise ValueError("validation profile must be an object")
        _require_exact_keys(value, _PROFILE_KEYS, "validation profile")
        covered = value["covered_hardware"]
        if isinstance(covered, list):
            covered = tuple(covered)
        return cls(
            id=value["id"],
            capability_id=value["capability_id"],
            model_architecture=value["model_architecture"],
            fixture_id=value["fixture_id"],
            fixture_digest=value["fixture_digest"],
            representative_hardware=value["representative_hardware"],
            covered_hardware=covered,
        )


@dataclass(frozen=True)
class ProposedValidationObligation:
    profile_id: str
    rationale: str

    def __post_init__(self) -> None:
        _require_text("profile_id", self.profile_id)
        _require_text("rationale", self.rationale)
        object.__setattr__(self, "rationale", _normalize_rationale(self.rationale))


@dataclass(frozen=True, init=False)
class ValidationLedgerRow:
    """Typed, pending validation row serialized into a review proposal."""

    request_id: str
    profile_snapshot: Mapping[str, Any]
    profile_digest: str
    capability_id: str
    contract_digest: str
    model_architecture: str
    fixture_id: str
    fixture_digest: str
    representative_hardware: str
    covered_hardware: tuple[str, ...]
    coverage_kind: str
    status: str
    validator_snapshot: Mapping[str, Any]
    result_snapshot: Mapping[str, Any]
    rationales: tuple[str, ...]

    def __init__(
        self,
        profile: ValidationProfile,
        contract_digest: str,
        coverage_kind: str,
        obligations: tuple[ProposedValidationObligation, ...] | ProposedValidationObligation = (),
    ) -> None:
        if not isinstance(profile, ValidationProfile):
            raise ValueError("profile must be a ValidationProfile")
        _require_digest("contract_digest", contract_digest)
        _require_text("coverage_kind", coverage_kind)
        if isinstance(obligations, ProposedValidationObligation):
            obligations = (obligations,)
        if not isinstance(obligations, tuple) or any(
            not isinstance(obligation, ProposedValidationObligation) for obligation in obligations
        ):
            raise ValueError("obligations must be a tuple of ProposedValidationObligation values")
        if any(obligation.profile_id != profile.id for obligation in obligations):
            raise ValueError("obligation profile does not match ledger profile")
        snapshot = profile.to_mapping()
        object.__setattr__(self, "request_id", "vr-" + hashlib.sha256(profile.id.encode("utf-8")).hexdigest()[:16])
        object.__setattr__(self, "profile_snapshot", _freeze_payload(snapshot))
        object.__setattr__(self, "profile_digest", profile_digest(snapshot))
        object.__setattr__(self, "capability_id", profile.capability_id)
        object.__setattr__(self, "contract_digest", contract_digest)
        object.__setattr__(self, "model_architecture", profile.model_architecture)
        object.__setattr__(self, "fixture_id", profile.fixture_id)
        object.__setattr__(self, "fixture_digest", profile.fixture_digest)
        object.__setattr__(self, "representative_hardware", profile.representative_hardware)
        object.__setattr__(self, "covered_hardware", profile.covered_hardware)
        object.__setattr__(self, "coverage_kind", coverage_kind)
        object.__setattr__(self, "status", "pending")
        object.__setattr__(self, "validator_snapshot", MappingProxyType({}))
        object.__setattr__(self, "result_snapshot", MappingProxyType({}))
        object.__setattr__(self, "rationales", tuple(obligation.rationale for obligation in obligations))

    def to_mapping(self) -> dict[str, Any]:
        profile_snapshot = dict(self.profile_snapshot)
        profile_snapshot["covered_hardware"] = list(self.profile_snapshot["covered_hardware"])
        return {
            "request_id": self.request_id,
            "profile_snapshot": profile_snapshot,
            "profile_digest": self.profile_digest,
            "capability_id": self.capability_id,
            "contract_digest": self.contract_digest,
            "model_architecture": self.model_architecture,
            "fixture_id": self.fixture_id,
            "fixture_digest": self.fixture_digest,
            "representative_hardware": self.representative_hardware,
            "covered_hardware": list(self.covered_hardware),
            "coverage_kind": self.coverage_kind,
            "status": self.status,
            "validator_snapshot": {},
            "result_snapshot": {},
            "rationales": list(self.rationales),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValidationLedgerRow":
        validate_ledger_row_mapping(value)
        profile = ValidationProfile.from_mapping(value["profile_snapshot"])
        rationales = value["rationales"]
        row = cls(
            profile,
            value["contract_digest"],
            value["coverage_kind"],
            tuple(ProposedValidationObligation(profile.id, rationale) for rationale in rationales),
        )
        normalized_value = dict(value)
        normalized_value["covered_hardware"] = list(value["covered_hardware"])
        normalized_value["rationales"] = list(value["rationales"])
        profile_snapshot = dict(value["profile_snapshot"])
        profile_snapshot["covered_hardware"] = list(profile_snapshot["covered_hardware"])
        normalized_value["profile_snapshot"] = profile_snapshot
        if row.to_mapping() != normalized_value:
            raise ValueError("validation ledger row is not canonical")
        return row


@dataclass(frozen=True)
class ReviewProposal:
    target: ReviewTarget
    capsule_digest: str
    proposal_digest: str
    verdict: str
    findings: tuple[Finding, ...]
    adapter_id: str
    adapter_version: str
    model: str
    response_digest: str
    retrieved_file_count: int | None = None
    expected_file_count: int | None = None
    retrieved_blob_count: int | None = None
    expected_blob_count: int | None = None
    expected_content_count: int | None = None
    retrieved_content_count: int | None = None
    coverage_complete: bool | None = None
    validation_ledger: tuple[ValidationLedgerRow, ...] = ()
    configuration_source_digest: str | None = None
    exemption_ids: tuple[str, ...] = ()
    exemption_paths: tuple[str, ...] = ()
    scope: ReviewScope | None = None
    hardware_validation_triage: HardwareValidationTriage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, ReviewTarget):
            raise ValueError("target must be a ReviewTarget")
        _require_digest("capsule_digest", self.capsule_digest)
        _require_digest("proposal_digest", self.proposal_digest)
        _require_digest("response_digest", self.response_digest)
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("adapter_version", self.adapter_version),
            ("model", self.model),
        ):
            _require_text(name, value)
        if self.verdict not in _VERDICTS:
            raise ValueError("verdict is not supported")
        if not isinstance(self.findings, tuple) or any(not isinstance(finding, Finding) for finding in self.findings):
            raise ValueError("findings must be a tuple of Finding values")
        if self.scope is not None and not isinstance(self.scope, ReviewScope):
            raise ValueError("scope must be a ReviewScope")
        if self.hardware_validation_triage is not None and not isinstance(self.hardware_validation_triage, HardwareValidationTriage):
            raise ValueError("hardware_validation_triage must be a HardwareValidationTriage")
        if not isinstance(self.validation_ledger, tuple) or any(
            not isinstance(row, ValidationLedgerRow) for row in self.validation_ledger
        ):
            raise ValueError("validation_ledger must be a tuple of ValidationLedgerRow values")
        validate_ledger_payload_shape(tuple(row.to_mapping() for row in self.validation_ledger))
        if not isinstance(self.exemption_paths, tuple) or any(not isinstance(path, str) for path in self.exemption_paths):
            raise ValueError("exemption_paths must be a tuple of strings")
        if not isinstance(self.exemption_ids, tuple) or any(not isinstance(item, str) for item in self.exemption_ids):
            raise ValueError("exemption_ids must be a tuple of strings")
        if not self.exemption_ids:
            if self.exemption_paths:
                raise ValueError("exemption IDs are required for exemption paths")
        else:
            if tuple(sorted(set(self.exemption_ids))) != self.exemption_ids:
                raise ValueError("exemption IDs must be sorted and unique")
            for exemption_id in self.exemption_ids:
                _require_text("exemption_id", exemption_id)
            if not self.exemption_paths:
                raise ValueError("exemption paths are required for an exemption")
            normalized_paths = tuple(normalize_repository_path(path) for path in self.exemption_paths)
            if normalized_paths != self.exemption_paths or normalized_paths != tuple(sorted(set(normalized_paths))):
                raise ValueError("exemption paths must be normalized, sorted, and unique")
            if any(len(exemption_id.encode("utf-8")) > MAX_VALIDATION_FIELD_BYTES for exemption_id in self.exemption_ids) or any(
                len(path.encode("utf-8")) > MAX_VALIDATION_FIELD_BYTES for path in self.exemption_paths
            ):
                raise ValueError(f"exemption evidence fields exceed {MAX_VALIDATION_FIELD_BYTES} bytes")
            if self.validation_ledger:
                raise ValueError("exemption evidence cannot accompany validation rows")
        has_validation = bool(self.validation_ledger) or self.configuration_source_digest is not None
        if has_validation:
            if self.configuration_source_digest is None:
                raise ValueError("configuration_source_digest is required for validation binding")
            _require_digest("configuration_source_digest", self.configuration_source_digest)
        if self.exemption_ids and self.configuration_source_digest is None:
            raise ValueError("exemption evidence requires a configuration source digest")
        if self.configuration_source_digest is not None and not self.validation_ledger and not self.exemption_ids:
            raise ValueError("empty validation ledger requires protected exemption evidence")
        has_actionable_finding = any(finding.severity in ACTIONABLE_SEVERITIES for finding in self.findings)
        if self.verdict == "clean" and has_actionable_finding:
            raise ValueError("clean proposals cannot contain actionable findings")
        if self.verdict == "changes-requested" and not has_actionable_finding:
            raise ValueError("changes-requested proposals require an actionable finding")
        coverage_values = (
            self.retrieved_file_count, self.expected_file_count, self.retrieved_blob_count,
            self.expected_blob_count, self.retrieved_content_count, self.expected_content_count,
            self.coverage_complete,
        )
        if all(value is None for value in coverage_values):
            bind_coverage = False
            counts = ()
        elif any(value is None for value in coverage_values):
            raise ValueError("coverage evidence must be complete or entirely absent")
        else:
            counts = (
                ("retrieved_file_count", self.retrieved_file_count, self.expected_file_count),
                ("retrieved_blob_count", self.retrieved_blob_count, self.expected_blob_count),
                ("retrieved_content_count", self.retrieved_content_count, self.expected_content_count),
            )
            bind_coverage = True
        for name, retrieved, expected_count in counts:
            if (
                isinstance(retrieved, bool) or not isinstance(retrieved, int) or retrieved < 0
                or isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0
                or retrieved > expected_count
            ):
                raise ValueError(f"{name} and its expected count must be non-negative and ordered")
        if bind_coverage and not isinstance(self.coverage_complete, bool):
            raise ValueError("coverage_complete must be a boolean")
        if bind_coverage and self.coverage_complete and any(retrieved != expected_count for _, retrieved, expected_count in counts):
            raise ValueError("complete coverage must have matching retrieved and expected counts")
        coverage = {
            "retrieved_file_count": self.retrieved_file_count,
            "expected_file_count": self.expected_file_count,
            "retrieved_blob_count": self.retrieved_blob_count,
            "expected_blob_count": self.expected_blob_count,
            "retrieved_content_count": self.retrieved_content_count,
            "expected_content_count": self.expected_content_count,
            "coverage_complete": self.coverage_complete,
        }
        # Keep positional/legacy proposals constructible while binding real
        # capsule coverage evidence into every new proposal digest.
        digest_values = {
            "target": self.target,
            "target_key": self.target.target_key(),
            "capsule_digest": self.capsule_digest,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "model": self.model,
            "response_digest": self.response_digest,
            "verdict": self.verdict,
            "findings": self.findings,
        }
        if bind_coverage:
            digest_values["coverage"] = coverage
        if has_validation:
            digest_values["validation_ledger"] = tuple(row.to_mapping() for row in self.validation_ledger)
            digest_values["configuration_source_digest"] = self.configuration_source_digest
            if self.exemption_ids:
                digest_values["exemption_ids"] = self.exemption_ids
                digest_values["exemption_paths"] = self.exemption_paths
        if self.scope is not None:
            digest_values["scope"] = self.scope.to_mapping()
        if self.hardware_validation_triage is not None:
            digest_values["hardware_validation_triage"] = self.hardware_validation_triage.to_mapping()
        expected = "sha256:" + canonical_digest(digest_values)
        if self.proposal_digest != expected:
            raise ValueError("proposal digest is not bound to target, capsule, provider, and response")

    def coverage_mapping(self) -> dict[str, Any]:
        if any(value is None for value in (
            self.retrieved_file_count, self.expected_file_count, self.retrieved_blob_count,
            self.expected_blob_count, self.retrieved_content_count, self.expected_content_count,
            self.coverage_complete,
        )):
            raise ValueError("proposal has no complete coverage evidence")
        return {
            "retrieved_file_count": self.retrieved_file_count,
            "expected_file_count": self.expected_file_count,
            "retrieved_blob_count": self.retrieved_blob_count,
            "expected_blob_count": self.expected_blob_count,
            "retrieved_content_count": self.retrieved_content_count,
            "expected_content_count": self.expected_content_count,
            "coverage_complete": self.coverage_complete,
        }

@dataclass(frozen=True)
class ValidationRequest:
    target: ReviewTarget
    request_id: str
    capability_id: str
    contract_digest: str
    report_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, ReviewTarget):
            raise ValueError("target must be a ReviewTarget")
        _require_text("request_id", self.request_id)
        _require_text("capability_id", self.capability_id)
        _require_digest("contract_digest", self.contract_digest)
        _require_digest("report_digest", self.report_digest)


@dataclass(frozen=True)
class ProviderPolicy:
    provider_id: str
    adapter_id: str
    adapter_version: str
    endpoint: str
    model: str
    api_key_env: str
    max_requests: int
    request_deadline_seconds: float
    max_capsule_bytes: int
    max_response_bytes: int
    max_tokens: int
    max_cost_usd: float

    def __post_init__(self) -> None:
        for name, value in (
            ("provider_id", self.provider_id),
            ("adapter_id", self.adapter_id),
            ("adapter_version", self.adapter_version),
            ("model", self.model),
            ("api_key_env", self.api_key_env),
        ):
            _require_text(name, value)
        parsed_endpoint = urlparse(self.endpoint)
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc or any(char.isspace() for char in self.endpoint):
            raise ValueError("endpoint must be an HTTPS URL")
        if self.max_requests != 1:
            raise ValueError("max_requests must be exactly 1")
        for name, value in (
            ("max_capsule_bytes", self.max_capsule_bytes),
            ("max_response_bytes", self.max_response_bytes),
            ("max_tokens", self.max_tokens),
        ):
            _require_positive_integer(name, value)
        if self.max_capsule_bytes > DEFAULT_MAX_BYTES or self.max_response_bytes > DEFAULT_MAX_BYTES:
            raise ValueError("provider capsule and response byte limits exceed canonical digest ceiling")
        if (
            isinstance(self.request_deadline_seconds, bool)
            or not isinstance(self.request_deadline_seconds, (int, float))
            or not math.isfinite(self.request_deadline_seconds)
            or self.request_deadline_seconds <= 0
        ):
            raise ValueError("request_deadline_seconds must be finite and positive")
        if (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd <= 0
        ):
            raise ValueError("max_cost_usd must be finite and positive")


@dataclass(frozen=True)
class TrustedApp:
    app_id: int
    login: str
    installation_id: int
    repository_id: int
    credential_attestation_digest: str

    def __post_init__(self) -> None:
        _require_positive_integer("app_id", self.app_id)
        _require_text("login", self.login)
        _require_positive_integer("installation_id", self.installation_id)
        _require_positive_integer("repository_id", self.repository_id)
        _require_digest("credential_attestation_digest", self.credential_attestation_digest)


@dataclass(frozen=True)
class TrustedPublisher:
    apps: tuple[TrustedApp, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.apps, tuple):
            raise ValueError("apps must be a tuple")
        if any(not isinstance(app, TrustedApp) for app in self.apps):
            raise ValueError("apps must contain TrustedApp values")


def capability_contract_digest(capability: Mapping[str, Any]) -> str:
    """Return the digest of canonical JSON for the complete capability sans digest.

    The serialization is UTF-8 RFC 8785-compatible JSON with deterministic
    key ordering and compact separators.  ``contract_digest`` is excluded;
    every other capability field is included.
    """
    if not isinstance(capability, Mapping) or frozenset(capability) != _CAPABILITY_KEYS:
        raise ValueError("capability has unexpected or missing keys")
    without_digest = {key: capability[key] for key in capability if key != "contract_digest"}
    return "sha256:" + hashlib.sha256(canonical_json(without_digest)).hexdigest()


def profile_digest(profile: ValidationProfile | Mapping[str, Any]) -> str:
    """Return the digest of the complete profile snapshot."""
    snapshot = profile.to_mapping() if isinstance(profile, ValidationProfile) else profile
    if not isinstance(snapshot, Mapping) or frozenset(snapshot) != _PROFILE_KEYS:
        raise ValueError("profile has unexpected or missing keys")
    return "sha256:" + hashlib.sha256(canonical_json(snapshot)).hexdigest()


def normalize_repository_path(path: str) -> str:
    """Validate, but do not rewrite, a repository-relative path or glob."""
    if not isinstance(path, str) or not path:
        raise ValueError("repository path must be a non-empty string")
    if path.startswith("/") or any(part == ".." for part in path.split("/")):
        raise ValueError("repository path must be repository-relative")
    return path


def _repository_glob_matches(path: str, pattern: str) -> bool:
    """Match repository segments with bounded iterative glob semantics."""
    path_parts = path.split("/")
    pattern_parts = pattern.split("/")
    reachable = [True] + [False] * len(path_parts)
    for pattern_part in pattern_parts:
        next_reachable = [False] * (len(path_parts) + 1)
        if pattern_part == "**":
            # A globstar consumes zero or more complete path segments.  The
            # left-to-right prefix propagation is linear and cannot recurse.
            for path_index in range(len(path_parts) + 1):
                next_reachable[path_index] = reachable[path_index] or (
                    path_index > 0 and next_reachable[path_index - 1]
                )
        else:
            for path_index, is_reachable in enumerate(reachable[:-1]):
                if is_reachable and fnmatch.fnmatchcase(path_parts[path_index], pattern_part):
                    next_reachable[path_index + 1] = True
        reachable = next_reachable
    return reachable[-1]


def protected_exemption_matches(exemptions: Any, path: str) -> bool:
    """Return whether ``path`` matches a normalized protected exemption glob."""
    normalized = normalize_repository_path(path)
    if not isinstance(exemptions, (list, tuple)):
        raise ValueError("protected exemptions must be a list")
    for exemption in exemptions:
        if not isinstance(exemption, Mapping) or frozenset(exemption) != _EXEMPTION_KEYS:
            raise ValueError("exemption has unexpected or missing keys")
        globs = exemption["path_globs"]
        _require_string_list("path_globs", globs)
        if any(_repository_glob_matches(normalized, normalize_repository_path(pattern)) for pattern in globs):
            return True
    return False


def protected_exemption_evidence(
    exemptions: Any, capsule_paths: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return deterministic protected exemption evidence for exact paths."""
    if not isinstance(capsule_paths, (list, tuple)) or not capsule_paths:
        return None
    normalized = tuple(normalize_repository_path(path) for path in capsule_paths)
    if len(normalized) != len(set(normalized)):
        return None
    normalized = tuple(sorted(normalized))
    if not isinstance(exemptions, (list, tuple)):
        raise ValueError("protected exemptions must be a list")
    covered_paths: set[str] = set()
    matches: set[str] = set()
    for exemption in exemptions:
        if not isinstance(exemption, Mapping) or frozenset(exemption) != _EXEMPTION_KEYS:
            raise ValueError("exemption has unexpected or missing keys")
        globs = exemption["path_globs"]
        _require_string_list("path_globs", globs)
        normalized_globs = tuple(normalize_repository_path(pattern) for pattern in globs)
        matching_paths = {
            path for path in normalized
            if any(_repository_glob_matches(path, pattern) for pattern in normalized_globs)
        }
        if matching_paths:
            _require_text("exemption id", exemption["id"])
            matches.add(exemption["id"])
            covered_paths.update(matching_paths)
    if covered_paths != set(normalized):
        return None
    return (tuple(sorted(matches)), normalized) if matches else None


def capsule_paths_are_exempt(exemptions: Any, capsule_paths: Any) -> bool:
    """Require every capsule path to match a protected exemption glob."""
    if not isinstance(capsule_paths, (list, tuple)):
        raise ValueError("capsule paths must be a list")
    return protected_exemption_evidence(exemptions, capsule_paths) is not None


def derive_protected_review_scope(capsule: Any, policy: Mapping[str, Any]) -> ReviewScope:
    """Derive the conservative v1 scope from an immutable capsule and policy.

    A fully protected exemption has no validation scope.  Every other capsule
    receives the complete registered model inventory and the union of all
    registered covered hardware.  Unknown capsule shapes or incomplete policy
    data fail closed rather than guessing from paths or source contents.
    """
    manifest = getattr(capsule, "manifest", capsule)
    if not isinstance(manifest, (list, tuple)):
        raise ValueError("capsule manifest is required for scope derivation")
    paths: list[str] = []
    for entry in manifest:
        path = (
            entry if isinstance(entry, str)
            else entry.get("path") if isinstance(entry, Mapping)
            else getattr(entry, "path", None)
        )
        if not isinstance(path, str):
            raise ValueError("capsule manifest contains an invalid path")
        paths.append(path)
    validate_capability_policy(policy)
    if protected_exemption_evidence(policy["exemptions"], paths) is not None:
        return ReviewScope((), ())
    profiles = policy.get("profiles")
    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise ValueError("non-exempt scope has no protected profiles")
    typed_profiles = tuple(ValidationProfile.from_mapping(profile) for profile in profiles)
    return ReviewScope(
        tuple(sorted({profile.model_architecture for profile in typed_profiles})),
        tuple(sorted({hardware for profile in typed_profiles for hardware in profile.covered_hardware})),
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("policy must be a JSON object")
    return value


def validate_capability_policy(policy: Mapping[str, Any]) -> None:
    """Validate the checked-in v1 capability policy and each contract digest."""
    if not isinstance(policy, Mapping):
        raise ValueError("capability policy must be an object")
    _require_exact_keys(policy, _CAPABILITY_ROOT_KEYS, "capability policy")
    if policy["schema"] != "hipfire.agentic-review.capabilities" or policy["version"] != 1:
        raise ValueError("invalid capability policy schema or version")
    capabilities = policy["capabilities"]
    if not isinstance(capabilities, (list, tuple)) or not capabilities:
        raise ValueError("capability policy must contain capabilities")
    expected_ids = {
        "hipfire/rdna3-smoke@1",
        "hipfire/gfx1151-kernel-validation@1",
        "hipfire/dflash-coherence@1",
    }
    actual_ids = []
    for capability in capabilities:
        if not isinstance(capability, Mapping):
            raise ValueError("capability must be an object")
        _require_exact_keys(capability, _CAPABILITY_KEYS, "capability")
        _require_text("capability id", capability["id"])
        actual_ids.append(capability["id"])
        if capability["parameters"] != {}:
            raise ValueError("capability parameters must be an empty object")
        for field in ("allowed_suite_revisions", "required_checks", "eligible_hardware", "artifacts"):
            _require_string_list(field, capability[field])
        if capability["pass_criteria"] != {"all_required_checks_pass": True}:
            raise ValueError("pass_criteria must require all_required_checks_pass")
        _require_digest("contract_digest", capability["contract_digest"])
        if capability["contract_digest"] != capability_contract_digest(capability):
            raise ValueError("capability contract digest does not match capability")
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("capability policy has the wrong capability IDs")
    profiles = policy["profiles"]
    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise ValueError("capability policy must contain profiles")
    if len(profiles) > MAX_VALIDATION_ROWS:
        raise ValueError(f"capability policy cannot contain more than {MAX_VALIDATION_ROWS} profiles")
    profile_ids: list[str] = []
    profile_capability_ids: list[str] = []
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("profile must be an object")
        _require_exact_keys(profile, _PROFILE_KEYS, "profile")
        typed_profile = ValidationProfile.from_mapping(profile)
        profile_ids.append(typed_profile.id)
        profile_capability_ids.append(typed_profile.capability_id)
        if typed_profile.capability_id not in expected_ids:
            raise ValueError("profile references an unknown capability")
        capability = next(item for item in capabilities if item["id"] == typed_profile.capability_id)
        eligible = tuple(capability["eligible_hardware"])
        if typed_profile.representative_hardware not in eligible:
            raise ValueError("representative_hardware is not eligible for capability")
        if any(hardware not in eligible for hardware in typed_profile.covered_hardware):
            raise ValueError("covered_hardware contains ineligible hardware")
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("profile IDs must be unique")
    if set(profile_capability_ids) != set(actual_ids):
        raise ValueError("profiles must cover each capability at least once")
    fixtures = policy["fixtures"]
    if not isinstance(fixtures, (list, tuple)) or not fixtures:
        raise ValueError("capability policy must contain fixtures")
    fixture_ids: list[str] = []
    fixture_map: dict[str, Mapping[str, Any]] = {}
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise ValueError("fixture must be an object")
        _require_exact_keys(fixture, _FIXTURE_KEYS, "fixture")
        for field in ("fixture_id", "model_architecture", "artifact_identity", "source_identity", "suite_revision", "digest_semantics"):
            _require_text(f"fixture {field}", fixture[field])
        _require_digest("fixture_digest", fixture["fixture_digest"])
        if fixture["fixture_digest"] != fixture_descriptor_digest(fixture):
            raise ValueError("fixture descriptor digest does not match fixture fields")
        fixture_ids.append(fixture["fixture_id"])
        fixture_map[fixture["fixture_id"]] = fixture
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("fixture IDs must be unique")
    for profile in profiles:
        fixture = fixture_map.get(profile["fixture_id"])
        if fixture is None:
            raise ValueError("profile references an unknown fixture")
        if profile["model_architecture"] != fixture["model_architecture"]:
            raise ValueError("profile and fixture model architecture do not match")
        if profile["fixture_digest"] != fixture["fixture_digest"]:
            raise ValueError("profile fixture digest does not match fixture manifest")
        capability = next(item for item in capabilities if item["id"] == profile["capability_id"])
        if fixture["suite_revision"] not in capability["allowed_suite_revisions"]:
            raise ValueError("fixture suite revision is not allowed by capability")
        if fixture["artifact_identity"] not in capability["artifacts"]:
            raise ValueError("fixture artifact is not allowed by capability")
    exemptions = policy["exemptions"]
    if not isinstance(exemptions, (list, tuple)):
        raise ValueError("exemptions must be a list")
    exemption_ids: list[str] = []
    for exemption in exemptions:
        if not isinstance(exemption, Mapping):
            raise ValueError("exemption must be an object")
        _require_exact_keys(exemption, _EXEMPTION_KEYS, "exemption")
        _require_text("exemption id", exemption["id"])
        exemption_ids.append(exemption["id"])
        _require_string_list("path_globs", exemption["path_globs"])
        for path_glob in exemption["path_globs"]:
            normalize_repository_path(path_glob)
    if len(exemption_ids) != len(set(exemption_ids)):
        raise ValueError("exemption IDs must be unique")


def load_capability_policy(path: str | Path) -> dict[str, Any]:
    policy = _load_json(path)
    validate_capability_policy(policy)
    return policy


def validate_provider_policy(policy: Mapping[str, Any]) -> None:
    if not isinstance(policy, Mapping):
        raise ValueError("provider policy must be an object")
    _require_exact_keys(policy, _PROVIDER_ROOT_KEYS, "provider policy")
    if policy["schema"] != "hipfire.agentic-review.providers" or policy["version"] != 1:
        raise ValueError("invalid provider policy schema or version")
    providers = policy["providers"]
    if not isinstance(providers, (list, tuple)):
        raise ValueError("providers must be a list")
    ids: list[str] = []
    for provider in providers:
        if not isinstance(provider, Mapping):
            raise ValueError("provider must be an object")
        _require_exact_keys(provider, _PROVIDER_KEYS, "provider")
        ids.append(provider["id"])
        ProviderPolicy(
            provider_id=provider["id"],
            adapter_id=provider["adapter_id"],
            adapter_version=provider["adapter_version"],
            endpoint=provider["endpoint"],
            model=provider["model"],
            api_key_env=provider["api_key_env"],
            max_requests=provider["max_requests"],
            request_deadline_seconds=provider["request_deadline_seconds"],
            max_capsule_bytes=provider["max_capsule_bytes"],
            max_response_bytes=provider["max_response_bytes"],
            max_tokens=provider["max_tokens"],
            max_cost_usd=provider["max_cost_usd"],
        )
    if len(ids) != len(set(ids)):
        raise ValueError("provider IDs must be unique")


def load_provider_policy(path: str | Path, provider_id: str | None = None) -> dict[str, Any]:
    if not provider_id:
        raise ValueError("provider ID is required")
    policy = _load_json(path)
    validate_provider_policy(policy)
    for provider in policy["providers"]:
        if provider["id"] == provider_id:
            return provider
    raise ValueError("provider is not configured")


def validate_trusted_publishers_policy(policy: Mapping[str, Any]) -> None:
    if not isinstance(policy, Mapping):
        raise ValueError("trusted publisher policy must be an object")
    _require_exact_keys(policy, _TRUSTED_ROOT_KEYS, "trusted publisher policy")
    if policy["schema"] != "hipfire.agentic-review.trusted-publishers" or policy["version"] != 1:
        raise ValueError("invalid trusted publisher schema or version")
    apps = policy["apps"]
    if not isinstance(apps, (list, tuple)):
        raise ValueError("apps must be a list")
    for app in apps:
        if not isinstance(app, Mapping):
            raise ValueError("app entries must be structured objects")
        _require_exact_keys(app, _TRUSTED_APP_KEYS, "trusted app")
        TrustedApp(**app)


def load_trusted_publishers_policy(path: str | Path) -> dict[str, Any]:
    policy = _load_json(path)
    validate_trusted_publishers_policy(policy)
    return policy
