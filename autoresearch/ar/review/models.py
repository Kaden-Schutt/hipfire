# Copyright (c) Kaden Schutt
"""Immutable policy and review contracts for the agentic review workflow."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

from .canonical import canonical_digest, canonical_json

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
_CAPABILITY_ROOT_KEYS = frozenset({"schema", "version", "capabilities"})
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
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")


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

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_payload(self.payload))
        _require_text("node_id", self.node_id)
        _require_text("author", self.author)
        _require_text("created_at", self.created_at)

    def __getitem__(self, key: str) -> Any:
        if key not in {"payload", "node_id", "author", "created_at"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("payload", "node_id", "author", "created_at"))

    def __len__(self) -> int:
        return 4


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise ValueError("payload must not contain sets")
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
        if _RAW_SHA256_RE.fullmatch(self.canonical_digest) is None or self.canonical_digest != canonical_digest(
            {key: value for key, value in self.to_mapping().items() if key != "canonical_digest"}
        ):
            raise ValueError("canonical_digest must exactly match the intent payload")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "target": self.target,
            "target_key": self.target_key,
            "attempt_id": self.attempt_id,
            "canonical_digest": self.canonical_digest,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IntentPayload":
        expected = {"schema", "record_type", "record_id", "target", "target_key", "attempt_id", "canonical_digest"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("invalid intent payload shape")
        return cls(**payload)


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
class ReviewProposal:
    target: ReviewTarget
    capsule_digest: str
    proposal_digest: str
    verdict: str
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, ReviewTarget):
            raise ValueError("target must be a ReviewTarget")
        _require_digest("capsule_digest", self.capsule_digest)
        _require_digest("proposal_digest", self.proposal_digest)
        if self.verdict not in _VERDICTS:
            raise ValueError("verdict is not supported")
        if not isinstance(self.findings, tuple) or any(not isinstance(finding, Finding) for finding in self.findings):
            raise ValueError("findings must be a tuple of Finding values")
        has_actionable_finding = any(finding.severity in ACTIONABLE_SEVERITIES for finding in self.findings)
        if self.verdict == "clean" and has_actionable_finding:
            raise ValueError("clean proposals cannot contain actionable findings")
        if self.verdict == "changes-requested" and not has_actionable_finding:
            raise ValueError("changes-requested proposals require an actionable finding")
        if self.verdict == "incomplete":
            return


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
    if not isinstance(capabilities, list) or not capabilities:
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
    if not isinstance(providers, list):
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
    if not isinstance(apps, list):
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
