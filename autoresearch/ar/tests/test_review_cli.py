# Copyright (c) Kaden Schutt
"""Focused tests for CLI configuration provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.ar.review import cli
from autoresearch.ar.review import (
    MAX_VALIDATION_LEDGER_BYTES,
    ProposedValidationObligation,
    ValidationLedgerRow,
    ValidationProfile,
    render_validation_section,
    validate_ledger_payload_shape,
)
from autoresearch.ar.review.config import (
    AuthenticatedConfigSource,
    _SOURCE_PROOF,
    configuration_source_digest,
)


ROOT = Path(__file__).parents[3]
REPO = "owner/repo"
CONFIG_PATHS = (
    ".github/agentic-review/providers.json",
    ".github/agentic-review/capabilities-v1.json",
    ".github/agentic-review/trusted-publishers.json",
)


class ConfigClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_repository(self, repository: str):
        self.calls.append(("repository", repository))
        return type("Response", (), {"data": {"default_branch": "main"}})()

    def get_branch_head(self, repository: str, branch: str) -> str:
        self.calls.append(("branch", (repository, branch)))
        return "c" * 40

    def authenticated_config_source(self, repository: str, *, commit_sha: str, repository_root: str):
        self.calls.append(("authenticated_source", (repository, commit_sha, repository_root)))
        contents = tuple((Path(repository_root) / path).read_bytes() for path in CONFIG_PATHS)
        return AuthenticatedConfigSource._from_authenticated_boundary(
            _SOURCE_PROOF,
            repository,
            "main",
            commit_sha,
            configuration_source_digest(*contents),
            repository_root,
        )


def test_cli_loads_repository_config_through_authenticated_source():
    client = ConfigClient()

    configuration = cli._config(client, REPO, ROOT)
    assert configuration.is_protected
    assert configuration.source is not None
    assert configuration.source.repository == REPO
    # _config reads from local disk and constructs the source directly;
    # it calls get_repository to resolve the default branch SHA.
    assert [name for name, _ in client.calls] == [
        "repository", "branch",
    ]


def test_cli_provenance_includes_the_complete_capabilities_policy():
    client = ConfigClient()
    configuration = cli._config(client, REPO, ROOT)
    contents = tuple((ROOT / path).read_bytes() for path in CONFIG_PATHS)
    capabilities = (ROOT / CONFIG_PATHS[1]).read_bytes()

    assert configuration.is_protected
    assert configuration.source is not None
    assert configuration.source.config_digest == configuration_source_digest(*contents)
    assert configuration_source_digest(*contents) != configuration_source_digest(
        contents[0], capabilities + b" ", contents[2]
    )


def test_validation_contracts_are_public_and_protocol_vectors_keep_legacy_shape():
    assert MAX_VALIDATION_LEDGER_BYTES == 64 * 1024
    assert ProposedValidationObligation.__name__ == "ProposedValidationObligation"
    assert ValidationLedgerRow.__name__ == "ValidationLedgerRow"
    assert ValidationProfile.__name__ == "ValidationProfile"
    assert render_validation_section

    vectors = json.loads(
        (Path(__file__).parent / "fixtures" / "review_protocol_vectors.json").read_text(encoding="utf-8")
    )
    assert {"canonical", "metadata", "regressions", "validation"} <= set(vectors)
    valid = vectors["validation"]["valid_ledger"]
    rows = validate_ledger_payload_shape(valid["validation_ledger"])
    assert rows[0]["request_id"] == "vr-03fbaa4bfe42cff0"


def test_ledger_vector_requires_authenticated_capsule_for_protocol_validation():
    client = ConfigClient()
    configuration = cli._config(client, REPO, ROOT)
    vectors = json.loads(
        (Path(__file__).parent / "fixtures" / "review_protocol_vectors.json").read_text(encoding="utf-8")
    )["validation"]

    from autoresearch.ar.review.protocol import validate_validation_ledger

    valid = vectors["valid_ledger"]
    with pytest.raises(ValueError, match="capsule"):
        validate_validation_ledger(valid, configuration=configuration)
    invalid = vectors["invalid_profile_config_binding"]
    with pytest.raises(ValueError, match="row|profile|policy"):
        validate_validation_ledger(invalid, configuration=configuration)
