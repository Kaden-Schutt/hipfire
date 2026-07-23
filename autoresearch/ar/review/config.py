# Copyright (c) Kaden Schutt
"""Protected repository configuration for the agentic review boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
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
_PROVIDERS_LOCAL = f"{_CONFIG_DIR}/providers.local.json"
_OPERATOR = f"{_CONFIG_DIR}/operator-credentials.json"
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_WRITE_PERMISSION_NAMES = {"issues", "pull_requests"}
_WRITE_PERMISSION_LEVELS = {"write", "admin"}
_OPERATOR_SCHEMA = "hipfire.agentic-review.operator-credentials"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SOURCE_PROOF = object()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("configuration mapping keys must be strings")
        from types import MappingProxyType
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise ValueError("configuration must not contain sets")
    if value is not None and not isinstance(value, (bool, int, float, str)):
        raise ValueError("configuration contains a mutable or unsupported value")
    return value


def _root_identity(root: str | Path) -> str:
    return "sha256:" + hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()


def configuration_source_digest(*contents: bytes) -> str:
    """Digest the complete protected policy files in fixed repository order.

    The capabilities argument is the raw, complete ``capabilities-v1.json``
    byte stream; callers must not digest a parsed or field-filtered policy.
    """
    digest = hashlib.sha256()
    for content in contents:
        if not isinstance(content, bytes):
            raise ValueError("configuration source contents must be bytes")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class AuthenticatedConfigSource:
    repository: str
    default_branch: str
    commit_sha: str
    config_digest: str
    root_identity: str
    _proof: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.repository, self.default_branch, self.commit_sha,
        )):
            raise ValueError("authenticated config source identity is incomplete")
        if _DIGEST_RE.fullmatch(self.config_digest) is None or _DIGEST_RE.fullmatch(self.root_identity) is None:
            raise ValueError("authenticated config source digests are invalid")

    @classmethod
    def _from_authenticated_boundary(
        cls, proof: object, repository: str, default_branch: str, commit_sha: str, config_digest: str, root: str | Path
    ) -> "AuthenticatedConfigSource":
        if proof is not _SOURCE_PROOF:
            raise ValueError("authenticated config source may only be issued by the GitHub boundary")
        source = cls(repository, default_branch, commit_sha, config_digest, _root_identity(root))
        object.__setattr__(source, "_proof", _SOURCE_PROOF)
        return source

    @property
    def authenticated(self) -> bool:
        return self._proof is _SOURCE_PROOF


@dataclass(frozen=True)
class ReviewConfiguration:
    providers: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    trusted_publishers: Mapping[str, Any]
    source: AuthenticatedConfigSource | None = None
    _loaded_from_protected_paths: bool = field(default=False, init=False, repr=False)
    _loaded_source_digest: str | None = field(default=None, init=False, repr=False)
    _loaded_root_identity: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", _freeze(self.providers))
        object.__setattr__(self, "capabilities", _freeze(self.capabilities))
        object.__setattr__(self, "trusted_publishers", _freeze(self.trusted_publishers))
        if self.source is not None and not isinstance(self.source, AuthenticatedConfigSource):
            raise ValueError("configuration source must be typed provenance")

    @property
    def is_protected(self) -> bool:
        return bool(
            self._loaded_from_protected_paths
            and self.source is not None
            and self.source.authenticated
            and self._loaded_source_digest == self.source.config_digest
            and self._loaded_root_identity == self.source.root_identity
        )

    def with_trusted_publishers(self, policy: Mapping[str, Any]) -> "ReviewConfiguration":
        validate_trusted_publishers_policy(policy)
        return replace(self, trusted_publishers=policy)


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

def _merge_providers(base: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    """Merge local provider overrides into the base provider policy.

    Providers from ``local`` with an ``id`` already present in ``base`` replace
    the checked-in entry.  New ids are appended.  Schema/version come from base.
    """
    if not isinstance(base, Mapping) or not isinstance(local, Mapping):
        raise ValueError("provider policies must be objects")
    if base.get("schema") != "hipfire.agentic-review.providers" or base.get("version") != 1:
        raise ValueError("base provider policy has invalid schema or version")
    if local.get("schema") != "hipfire.agentic-review.providers" or local.get("version") != 1:
        raise ValueError("local provider policy has invalid schema or version")
    base_providers = list(base.get("providers", []))
    local_providers = list(local.get("providers", []))
    if not all(isinstance(p, Mapping) and isinstance(p.get("id"), str) for p in base_providers):
        raise ValueError("base provider entries must have an id")
    if not all(isinstance(p, Mapping) and isinstance(p.get("id"), str) for p in local_providers):
        raise ValueError("local provider entries must have an id")
    # Build id→entry map from base, then overlay local entries
    merged_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for p in base_providers:
        pid = p["id"]
        merged_by_id[pid] = dict(p)
        order.append(pid)
    for p in local_providers:
        pid = p["id"]
        merged_by_id[pid] = dict(p)
        if pid not in order:
            order.append(pid)
    result = dict(base)
    result["providers"] = [merged_by_id[pid] for pid in order]
    return result



def load_review_configuration(
    repository_root: str | Path,
    *,
    providers_path: str = _PROVIDERS,
    capabilities_path: str = _CAPABILITIES,
    trusted_publishers_path: str = _TRUSTED,
    source: AuthenticatedConfigSource | None = None,
) -> ReviewConfiguration:
    """Load only the three checked-in policy files below ``repository_root``."""
    # The provider validator intentionally requires a selected provider.  Task
    # 3 needs the complete policy, including the valid empty repository policy.
    provider_file = _safe_path(repository_root, providers_path)
    capability_file = _safe_path(repository_root, capabilities_path)
    trusted_file = _safe_path(repository_root, trusted_publishers_path)
    provider_bytes = provider_file.read_bytes()
    capabilities_bytes = capability_file.read_bytes()
    trusted_bytes = trusted_file.read_bytes()
    provider_policy = json.loads(provider_bytes)

    # Merge local provider overrides if present (gitignored, per-developer).
    # The local file has the same schema; its providers replace checked-in
    # entries with the same id and append new ids.  The config digest still
    # covers only the checked-in file so the authenticated boundary holds.
    local_file = _safe_path(repository_root, _PROVIDERS_LOCAL)
    if local_file.exists():
        provider_policy = _merge_providers(provider_policy, json.loads(local_file.read_bytes()))

    validate_provider_policy(provider_policy)
    configuration = ReviewConfiguration(
        providers=provider_policy,
        capabilities=load_capability_policy(capability_file),
        trusted_publishers=load_trusted_publishers_policy(trusted_file),
        source=source,
    )
    if (
        providers_path == _PROVIDERS
        and capabilities_path == _CAPABILITIES
        and trusted_publishers_path == _TRUSTED
        and source is not None
        and source.authenticated
        and source.root_identity == _root_identity(repository_root)
        and source.config_digest == configuration_source_digest(provider_bytes, capabilities_bytes, trusted_bytes)
    ):
        object.__setattr__(configuration, "_loaded_from_protected_paths", True)
        object.__setattr__(configuration, "_loaded_source_digest", source.config_digest)
        object.__setattr__(configuration, "_loaded_root_identity", source.root_identity)
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


def validate_publisher_operator_credential(manifest: Mapping[str, Any], repository: str) -> None:
    """Validate the stricter credential contract required by publication."""
    validate_operator_credential_manifest(manifest)
    if manifest["repository"] != repository:
        raise ValueError("operator credential repository does not match target repository")
    principal = manifest["principal"]
    if principal["type"] not in {"User", "Bot"} or not principal["login"].strip():
        raise ValueError("publisher operator principal is unsupported")
    if not {"publish", "dismiss-workflow-review"}.issubset(manifest["allowed_operations"]):
        raise ValueError("publisher operator is missing a required operation")
    for permission in ("issues", "pull_requests"):
        if manifest["write_permissions"].get(permission) not in _WRITE_PERMISSION_LEVELS:
            raise ValueError("publisher operator is missing a required write permission")


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
