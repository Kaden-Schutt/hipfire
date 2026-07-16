# Copyright (c) Kaden Schutt
"""Protected repository configuration for the agentic review boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any

from .models import (
    load_capability_policy,
    load_trusted_publishers_policy,
    validate_provider_policy,
    validate_trusted_publishers_policy,
)


_CONFIG_DIR = ".github/agentic-review"
_PROVIDERS = f"{_CONFIG_DIR}/providers.json"
_CAPABILITIES = f"{_CONFIG_DIR}/capabilities-v1.json"
_TRUSTED = f"{_CONFIG_DIR}/trusted-publishers.json"
_OPERATOR = f"{_CONFIG_DIR}/operator-credentials.json"
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_WRITE_PERMISSION_NAMES = {"issues", "pull_requests"}
_WRITE_PERMISSION_LEVELS = {"write", "admin"}
_OPERATOR_SCHEMA = "hipfire.agentic-review.operator-credentials"


@dataclass(frozen=True)
class ReviewConfiguration:
    providers: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    trusted_publishers: Mapping[str, Any]
    _loaded_from_protected_paths: bool = field(default=False, init=False, repr=False)

    @property
    def is_protected(self) -> bool:
        return self._loaded_from_protected_paths

    def with_trusted_publishers(self, policy: Mapping[str, Any]) -> "ReviewConfiguration":
        validate_trusted_publishers_policy(policy)
        return replace(self, trusted_publishers=dict(policy))


def _safe_path(root: str | Path, override: str) -> Path:
    root_path = Path(root)
    if not isinstance(override, str) or not override or Path(override).is_absolute():
        raise ValueError("configuration path must be repository-root-relative")
    relative = Path(override)
    if ".." in relative.parts:
        raise ValueError("configuration path traversal is not allowed")
    root_resolved = root_path.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("configuration path escapes repository root") from exc
    return candidate


def load_review_configuration(
    repository_root: str | Path,
    *,
    providers_path: str = _PROVIDERS,
    capabilities_path: str = _CAPABILITIES,
    trusted_publishers_path: str = _TRUSTED,
) -> ReviewConfiguration:
    """Load only the three checked-in policy files below ``repository_root``."""
    # The provider validator intentionally requires a selected provider.  Task
    # 3 needs the complete policy, including the valid empty repository policy.
    provider_file = _safe_path(repository_root, providers_path)
    capability_file = _safe_path(repository_root, capabilities_path)
    trusted_file = _safe_path(repository_root, trusted_publishers_path)
    with provider_file.open(encoding="utf-8") as stream:
        provider_policy = json.load(stream)
    validate_provider_policy(provider_policy)
    configuration = ReviewConfiguration(
        providers=provider_policy,
        capabilities=load_capability_policy(capability_file),
        trusted_publishers=load_trusted_publishers_policy(trusted_file),
    )
    if providers_path == _PROVIDERS and capabilities_path == _CAPABILITIES and trusted_publishers_path == _TRUSTED:
        object.__setattr__(configuration, "_loaded_from_protected_paths", True)
    return configuration


def validate_operator_credential_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("operator credential manifest must be an object")
    expected = {
        "schema", "version", "repository", "principal", "allowed_operations",
        "write_permissions", "credential_attestation_digest",
    }
    if set(manifest) != expected:
        raise ValueError("operator credential manifest has unexpected or missing keys")
    if manifest["schema"] != _OPERATOR_SCHEMA or manifest["version"] != 1:
        raise ValueError("invalid operator credential manifest schema")
    if not isinstance(manifest["repository"], str) or re.fullmatch(_REPOSITORY_RE, manifest["repository"]) is None:
        raise ValueError("operator repository is invalid")
    principal = manifest["principal"]
    if not isinstance(principal, Mapping) or set(principal) != {"login", "type"}:
        raise ValueError("operator principal must contain login and type")
    if not isinstance(principal["login"], str) or not principal["login"].strip():
        raise ValueError("operator login must be non-empty")
    if not isinstance(principal["type"], str) or principal["type"] not in {"User", "Bot", "Organization"}:
        raise ValueError("operator principal type is unsupported")
    operations = manifest["allowed_operations"]
    if not isinstance(operations, list) or not operations or any(
        operation not in {"discover", "publish", "dismiss-workflow-review"} for operation in operations
    ):
        raise ValueError("operator allowed_operations is unsupported or empty")
    permissions = manifest["write_permissions"]
    if not isinstance(permissions, Mapping) or not permissions or any(
        permission not in _WRITE_PERMISSION_NAMES or level not in _WRITE_PERMISSION_LEVELS
        for permission, level in permissions.items()
    ):
        raise ValueError("operator write_permissions is unsupported or empty")
    digest = manifest["credential_attestation_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("operator credential attestation digest is invalid")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ValueError("operator credential attestation digest is invalid") from exc


def load_operator_credential_manifest(
    repository_root: str | Path,
    *,
    manifest_path: str = _OPERATOR,
) -> dict[str, Any]:
    """Load the checked-in operator manifest from a repository-relative path."""
    path = _safe_path(repository_root, manifest_path)
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_operator_credential_manifest(manifest)
    return manifest
