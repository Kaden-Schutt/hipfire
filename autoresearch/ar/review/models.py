# Copyright (c) Kaden Schutt
"""Small, immutable data contracts for agentic review state."""

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttemptIntent:
    target: ReviewTarget
    capability_id: str
    suite_revision: str
    provider_id: str = "default"

    def __post_init__(self) -> None:
        _require_text("capability_id", self.capability_id)
        _require_text("suite_revision", self.suite_revision)
        _require_text("provider_id", self.provider_id)


@dataclass(frozen=True)
class ReviewProposal:
    target_key: str
    capability_id: str
    attempt_id: str
    verdict: str
    summary: str

    def __post_init__(self) -> None:
        for name, value in (
            ("target_key", self.target_key),
            ("capability_id", self.capability_id),
            ("attempt_id", self.attempt_id),
            ("verdict", self.verdict),
            ("summary", self.summary),
        ):
            _require_text(name, value)


@dataclass(frozen=True)
class ValidationRequest:
    target_key: str
    capability_id: str
    suite_revision: str

    def __post_init__(self) -> None:
        _require_text("target_key", self.target_key)
        _require_text("capability_id", self.capability_id)
        _require_text("suite_revision", self.suite_revision)


@dataclass(frozen=True)
class ProviderPolicy:
    provider_id: str
    endpoint_env: str
    api_key_env: str
    model_env: str
    max_requests: int
    max_response_bytes: int
    max_tokens: int
    max_cost_usd: float

    def __post_init__(self) -> None:
        _require_text("provider_id", self.provider_id)
        for name, value in (
            ("endpoint_env", self.endpoint_env),
            ("api_key_env", self.api_key_env),
            ("model_env", self.model_env),
        ):
            _require_text(name, value)
        for name, value in (
            ("max_requests", self.max_requests),
            ("max_response_bytes", self.max_response_bytes),
            ("max_tokens", self.max_tokens),
        ):
            _require_positive_integer(name, value)
        if isinstance(self.max_cost_usd, bool) or not isinstance(self.max_cost_usd, (int, float)) or self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")


@dataclass(frozen=True)
class TrustedPublisher:
    users: tuple[str, ...]
    apps: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, values in (("users", self.users), ("apps", self.apps)):
            if not isinstance(values, tuple):
                raise ValueError(f"{name} must be a tuple")
            for value in values:
                _require_text(name, value)


def validate_capability_policy(policy: Mapping[str, Any]) -> None:
    """Validate the checked-in v1 capability policy shape."""
    if not isinstance(policy, Mapping):
        raise ValueError("capability policy must be an object")
    if policy.get("schema") != "hipfire.agentic-review.capabilities":
        raise ValueError("invalid capability policy schema")
    if policy.get("version") != 1:
        raise ValueError("unsupported capability policy version")

    capabilities = policy.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("capability policy must contain capabilities")
    expected_ids = {
        "hipfire/rdna3-smoke@1",
        "hipfire/gfx1151-kernel-validation@1",
        "hipfire/dflash-coherence@1",
    }
    actual_ids = {item.get("id") for item in capabilities if isinstance(item, Mapping)}
    if actual_ids != expected_ids or len(capabilities) != len(expected_ids):
        raise ValueError("capability policy has the wrong capability IDs")

    required_fields = (
        "contract_digest",
        "allowed_suite_revisions",
        "required_checks",
        "artifacts",
        "pass_criteria",
    )
    for capability in capabilities:
        if not isinstance(capability, Mapping) or capability.get("parameters") != {}:
            raise ValueError("capability parameters must be an empty object")
        if any(field not in capability for field in required_fields):
            raise ValueError("capability is missing a required contract field")
        if not isinstance(capability["contract_digest"], str) or not capability["contract_digest"].startswith("sha256:"):
            raise ValueError("capability contract digest must be sha256-prefixed")
        for field in ("allowed_suite_revisions", "required_checks", "artifacts"):
            if not isinstance(capability[field], list) or not capability[field]:
                raise ValueError(f"capability {field} must be non-empty")
        if not isinstance(capability["pass_criteria"], Mapping) or not capability["pass_criteria"]:
            raise ValueError("capability pass criteria must be a non-empty object")
